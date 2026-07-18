"""Prepare temperature data and train R ``ensembleMOS`` models."""

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from rpy2 import rinterface, robjects
from rpy2.robjects import vectors
from rpy2.robjects.packages import importr

CELSIUS_TO_KELVIN = 273.15
DEFAULT_DATA_DIR = Path("data")


def load_forecast(
    city_name: str,
    model_name: str,
    rev: bool = False,
    *,
    data_dir: str | Path = DEFAULT_DATA_DIR,
) -> list[dict]:
    """Load one city's model forecast history from a JSONL data directory."""
    path = Path(data_dir) / "forecast" / city_name / model_name / "fc.jsonl"
    with path.open("r", encoding="utf-8") as forecast_file:
        forecasts = [json.loads(line) for line in forecast_file if line.strip()]
    forecasts.sort(
        key=lambda forecast: forecast["meta"]["last_run_initialisation_time"],
        reverse=rev,
    )
    return forecasts


def load_temperature(
    city_name: str,
    rev: bool = False,
    *,
    data_dir: str | Path = DEFAULT_DATA_DIR,
) -> list[dict]:
    """Load one city's observed temperature history from a JSONL data directory."""
    path = Path(data_dir) / "temperature" / city_name / "tem.jsonl"
    with path.open("r", encoding="utf-8") as temperature_file:
        temperatures = [
            json.loads(line) for line in temperature_file if line.strip()
        ]
    temperatures.sort(key=lambda temperature: temperature["time"], reverse=rev)
    return temperatures


def match_forecast(
    fcs: list[dict], timestamps: list[int], offset: int = 0
) -> list[dict | None]:
    """Return the forecast run available at each decision timestamp.

    ``offset=0`` selects the latest available run, ``offset=1`` the run before
    that, and so on.  ``None`` is returned when that many runs were not yet
    available.  This function selects runs; EMOS parameters must still be
    grouped by initialization hour and forecast lead.
    """
    if offset < 0:
        raise ValueError("offset must be non-negative")

    sorted_fcs = sorted(
        fcs,
        key=lambda fc: fc["meta"]["last_run_availability_time"],
    )
    if not sorted_fcs:
        return [None] * len(timestamps)

    available_ts = np.array(
        [fc["meta"]["last_run_availability_time"] for fc in sorted_fcs],
        dtype=np.int64,
    )
    indices = (
        np.searchsorted(
            available_ts,
            np.asarray(timestamps, dtype=np.int64),
            side="right",
        )
        - 1
        - offset
    )
    return [None if index < 0 else sorted_fcs[int(index)] for index in indices]


def _temperature_member_names(forecast: dict) -> tuple[str, ...]:
    """Return the control and perturbed 2 m temperature member names."""
    names = [
        name
        for name, values in forecast.items()
        if isinstance(values, list)
        and (name == "temperature_2m" or name.startswith("temperature_2m_member"))
    ]
    return tuple(sorted(names, key=lambda name: (name != "temperature_2m", name)))


def group_emos_training_data(
    fcs: list[dict],
    temperatures: list[dict],
    *,
    lead_step_hours: int | None = None,
    max_lead_hours: int | None = 96,
    require_available_before_valid: bool = True,
) -> dict[tuple[str, int], dict[str, Any]]:
    """Join forecasts to observations and group homogeneous EMOS cases.

    The result is keyed by ``(initialization_hour_utc, lead_hour)``.  Each
    value contains the parallel arrays needed to construct one R
    ``ensembleData`` object: ``forecasts``, ``valid_times`` and
    ``observations``.  Forecast rows before their run initialization are
    always discarded.  By default, rows that were not available before their
    valid time are also discarded because they could not have been issued as
    operational forecasts.

    Set ``lead_step_hours=6`` to retain only the native six-hourly AIFS lead
    times; leave it as ``None`` to retain Open-Meteo's interpolated hours too.
    Leads greater than ``max_lead_hours`` are discarded; pass ``None`` to
    disable the default 96-hour upper limit.
    """
    if lead_step_hours is not None and lead_step_hours <= 0:
        raise ValueError("lead_step_hours must be positive")
    if max_lead_hours is not None and max_lead_hours < 0:
        raise ValueError("max_lead_hours must be non-negative")

    observations_by_time = {
        int(item["time"]): float(item["temperature"]) for item in temperatures
    }

    groups: dict[tuple[str, int], dict[str, Any]] = {}
    seen_cases: set[tuple[int, int]] = set()

    # If a run was captured more than once, use its earliest captured version
    # and do not count it as multiple independent training cases.
    ordered_fcs = sorted(
        fcs,
        key=lambda fc: (
            fc["meta"]["last_run_initialisation_time"],
            fc["meta"]["last_run_availability_time"],
        ),
    )
    expected_member_names: tuple[str, ...] | None = None

    for forecast in ordered_fcs:
        meta = forecast["meta"]
        initialization_time = int(meta["last_run_initialisation_time"])
        availability_time = int(meta["last_run_availability_time"])
        initialization_hour = datetime.fromtimestamp(
            initialization_time, timezone.utc
        ).strftime("%H")

        member_names = _temperature_member_names(forecast)
        if not member_names:
            raise ValueError("forecast contains no 2 m temperature members")
        if expected_member_names is None:
            expected_member_names = member_names
        elif member_names != expected_member_names:
            raise ValueError("forecast runs have inconsistent ensemble members")

        valid_times = forecast.get("time")
        if not isinstance(valid_times, list):
            raise ValueError("forecast time must be a list")
        if any(len(forecast[name]) != len(valid_times) for name in member_names):
            raise ValueError("forecast member and time arrays have different lengths")

        for row_index, raw_valid_time in enumerate(valid_times):
            valid_time = int(raw_valid_time)
            case_key = (initialization_time, valid_time)
            if case_key in seen_cases:
                continue

            lead_seconds = valid_time - initialization_time
            if lead_seconds < 0 or lead_seconds % 3600 != 0:
                continue
            lead_hour = lead_seconds // 3600
            if max_lead_hours is not None and lead_hour > max_lead_hours:
                continue
            if lead_step_hours is not None and lead_hour % lead_step_hours != 0:
                continue
            if require_available_before_valid and availability_time > valid_time:
                continue

            observation = observations_by_time.get(valid_time)
            if observation is None:
                continue

            seen_cases.add(case_key)
            group_key = (initialization_hour, lead_hour)
            group = groups.setdefault(
                group_key,
                {
                    "initialization_time": initialization_hour,
                    "forecast_hour": lead_hour,
                    "member_names": member_names,
                    "initialization_times": [],
                    "valid_times": [],
                    "forecasts": [],
                    "observations": [],
                },
            )
            group["initialization_times"].append(initialization_time)
            group["valid_times"].append(valid_time)
            group["forecasts"].append(
                tuple(forecast[name][row_index] for name in member_names)
            )
            group["observations"].append(observation)

    for group in groups.values():
        order = sorted(
            range(len(group["valid_times"])),
            key=lambda index: group["valid_times"][index],
        )
        for field in (
            "initialization_times",
            "valid_times",
            "forecasts",
            "observations",
        ):
            group[field] = [group[field][index] for index in order]

    return groups


def _temperature_to_kelvin(value: Any, input_unit: str) -> float:
    """Convert one finite temperature value to Kelvin."""
    temperature = float(value)
    if not math.isfinite(temperature):
        raise ValueError("temperature values must be finite")

    normalized_unit = input_unit.lower()
    if normalized_unit in {"c", "celsius"}:
        temperature += CELSIUS_TO_KELVIN
    elif normalized_unit not in {"k", "kelvin"}:
        raise ValueError("input_unit must be 'celsius' or 'kelvin'")

    if temperature < 0:
        raise ValueError("temperature cannot be below absolute zero")
    return temperature


def _exchangeable_member_groups(
    member_names: tuple[str, ...],
    exchangeable: bool | list[str] | tuple[str, ...],
) -> tuple[str, ...] | None:
    """Resolve exchangeability labels for control and perturbed members."""
    if exchangeable is False:
        return None
    if exchangeable is True:
        return tuple(
            "control" if name == "temperature_2m" else "perturbed"
            for name in member_names
        )

    labels = tuple(exchangeable)
    if len(labels) != len(member_names):
        raise ValueError("exchangeable must contain one label per ensemble member")
    return labels


def build_temperature_ensemble_data(
    group: dict[str, Any],
    *,
    input_unit: str = "celsius",
    exchangeable: bool | list[str] | tuple[str, ...] = True,
) -> Any:
    """Build one R ``ensembleData`` object from a homogeneous EMOS group.

    Open-Meteo temperatures are Celsius by default.  Both forecasts and
    observations are converted to Kelvin before crossing the Python/R bridge.
    ``exchangeable=True`` keeps the control member separate and constrains all
    perturbed members to share one coefficient.
    """
    member_names = tuple(group["member_names"])
    forecast_rows = list(group["forecasts"])
    valid_times = list(group["valid_times"])
    observations = list(group["observations"])

    number_of_rows = len(forecast_rows)
    number_of_members = len(member_names)
    if number_of_rows == 0:
        raise ValueError("EMOS group contains no forecast cases")
    if number_of_members == 0:
        raise ValueError("EMOS group contains no ensemble members")
    if len(valid_times) != number_of_rows or len(observations) != number_of_rows:
        raise ValueError("EMOS group fields have different numbers of cases")
    if any(len(row) != number_of_members for row in forecast_rows):
        raise ValueError("forecast rows do not match member_names")

    forecast_values = [
        _temperature_to_kelvin(value, input_unit)
        for row in forecast_rows
        for value in row
    ]
    observation_values = [
        _temperature_to_kelvin(value, input_unit) for value in observations
    ]
    dates = [
        datetime.fromtimestamp(int(timestamp), timezone.utc).strftime("%Y%m%d%H")
        for timestamp in valid_times
    ]

    forecast_matrix = robjects.r["matrix"](
        vectors.FloatVector(forecast_values),
        nrow=number_of_rows,
        ncol=number_of_members,
        byrow=True,
    )  # type: ignore
    forecast_matrix.colnames = vectors.StrVector(member_names)

    exchangeable_groups = _exchangeable_member_groups(
        member_names,
        exchangeable,
    )
    arguments: dict[str, Any] = {
        "forecasts": forecast_matrix,
        "dates": vectors.StrVector(dates),
        "observations": vectors.FloatVector(observation_values),
        "forecastHour": int(group["forecast_hour"]),
        "initializationTime": str(group["initialization_time"]),
    }
    if exchangeable_groups is not None:
        arguments["exchangeable"] = vectors.StrVector(exchangeable_groups)

    ensemble_bma = importr("ensembleBMA")
    return ensemble_bma.ensembleData(**arguments)


def ensemble_mos(
    ensemble_data: Any,
    training_days: int,
    *,
    consecutive: bool = False,
    dates: Any = rinterface.NULL,
    control: Any | None = None,
    warm_start: bool = False,
    model: str = "normal",
    exchangeable: Any = rinterface.NULL,
) -> Any:
    """Call ``ensembleMOS::ensembleMOS`` and return its raw R result.

    ``ensemble_data`` must be an R object inheriting from ``ensembleData``.
    No Python/R data-frame conversion or weather-data preprocessing is done in
    this low-level wrapper.  ``model="normal"`` is the usual EMOS model for
    temperature forecasts.
    """
    ensemble_mos_package = importr("ensembleMOS")
    arguments = {
        "trainingDays": training_days,
        "consecutive": consecutive,
        "dates": dates,
        "warmStart": warm_start,
        "model": model,
        "exchangeable": exchangeable,
    }
    if control is not None and control is not rinterface.NULL:
        arguments["control"] = control

    return ensemble_mos_package.ensembleMOS(ensemble_data, **arguments)


def train_grouped_ensemble_mos(
    groups: dict[tuple[str, int], dict[str, Any]],
    training_days: int | None = None,
    *,
    input_unit: str = "celsius",
    exchangeable: bool | list[str] | tuple[str, ...] = True,
    consecutive: bool = False,
    control: Any | None = None,
    warm_start: bool = False,
    skip_insufficient: bool = False,
) -> dict[tuple[str, int], Any]:
    """Train a Gaussian EMOS model for every initialization/lead group.

    With ``dates=NULL`` the R package fits all modeling dates allowed by the
    rolling ``trainingDays`` rule.  Set ``training_days=None`` to use all
    distinct valid dates available in each group, so groups may use different
    training-window lengths.  With an explicit value, every group needs at
    least that many dates; set ``skip_insufficient=True`` to ignore groups that
    have not accumulated enough history yet.
    """

    if training_days is not None and training_days <= 0:
        raise ValueError("training_days must be positive")

    available_dates = {
        key: len(set(group["valid_times"])) for key, group in groups.items()
    }
    if training_days is None:
        insufficient = {
            key: number_of_dates
            for key, number_of_dates in available_dates.items()
            if number_of_dates == 0
        }
    else:
        insufficient = {
            key: number_of_dates
            for key, number_of_dates in available_dates.items()
            if number_of_dates < training_days
        }
    if insufficient and not skip_insufficient:
        details = ", ".join(
            f"{key}: {number_of_dates}"
            for key, number_of_dates in sorted(insufficient.items())
        )
        required_dates = "at least one" if training_days is None else str(training_days)
        raise ValueError(
            f"insufficient EMOS training data; need {required_dates} dates per "
            f"group ({details})"
        )

    fits: dict[tuple[str, int], Any] = {}
    for key, group in sorted(groups.items()):
        if key in insufficient:
            continue
        ensemble_data = build_temperature_ensemble_data(
            group,
            input_unit=input_unit,
            exchangeable=exchangeable,
        )
        fits[key] = ensemble_mos(
            ensemble_data,
            training_days=(
                available_dates[key] if training_days is None else training_days
            ),
            consecutive=consecutive,
            control=control,
            warm_start=warm_start,
            model="normal",
        )

    return fits


def train_ensemble_mos(
    city_name: str,
    model_name: str,
    *,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    training_days: int | None = None,
    lead_step_hours: int | None = None,
    max_lead_hours: int | None = 96,
    require_available_before_valid: bool = True,
    input_unit: str = "celsius",
    exchangeable: bool | list[str] | tuple[str, ...] = True,
    consecutive: bool = False,
    control: Any | None = None,
    warm_start: bool = False,
    skip_insufficient: bool = False,
) -> dict[tuple[str, int], Any]:
    """Load one city/model dataset, group its cases, and train EMOS models.

    ``data_dir`` must contain ``forecast/<city>/<model>/fc.jsonl`` and
    ``temperature/<city>/tem.jsonl``.  It defaults to the project's ``data``
    directory; tests and callers can point it at another directory such as
    ``test-data/data2``.
    """
    forecasts = load_forecast(
        city_name,
        model_name,
        data_dir=data_dir,
    )
    temperatures = load_temperature(
        city_name,
        data_dir=data_dir,
    )
    groups = group_emos_training_data(
        forecasts,
        temperatures,
        lead_step_hours=lead_step_hours,
        max_lead_hours=max_lead_hours,
        require_available_before_valid=require_available_before_valid,
    )
    if not groups:
        raise ValueError(
            f"no matching EMOS training cases for {city_name}/{model_name}"
        )

    return train_grouped_ensemble_mos(
        groups,
        training_days=training_days,
        input_unit=input_unit,
        exchangeable=exchangeable,
        consecutive=consecutive,
        control=control,
        warm_start=warm_start,
        skip_insufficient=skip_insufficient,
    )

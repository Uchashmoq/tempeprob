"""Prepare temperature data and train R ``ensembleMOS`` models."""

import hashlib
import json
import math
import os
import platform
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any
from uuid import uuid4
from zoneinfo import ZoneInfo

import numpy as np
from rpy2 import rinterface, robjects
from rpy2.rinterface_lib import callbacks
from rpy2.robjects import vectors
from rpy2.robjects.packages import importr

CELSIUS_TO_KELVIN = 273.15
DEFAULT_DATA_DIR = Path("data")
DEFAULT_MODEL_ARTIFACTS_DIR = Path("train/model_artifacts")
HOURLY_TEMPERATURE_EMOS_ARTIFACT_TYPE = "hourly_temperature_emos"
EMOS_ARTIFACT_SCHEMA_VERSION = 1
EmosGroupKey = tuple[str, int]


@dataclass(frozen=True)
class HourlyTemperatureEmosArtifact:
    """One restored version of hourly temperature EMOS fits."""

    version: str
    path: Path
    fits: dict[EmosGroupKey, Any]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class _HourlyEmosBatchCity:
    name: str
    timezone: ZoneInfo
    input_unit: str
    model_names: tuple[str, ...]


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
        temperatures = [json.loads(line) for line in temperature_file if line.strip()]
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

    # ensembleMOS prints every modeling date and its complete coefficient
    # vector. Keep warnings/errors visible, but hide that verbose progress
    # output so one concise Python log line can represent a completed model.
    with callbacks.replace_in_module(
        callbacks,
        "consolewrite_print",
        lambda _message: None,
    ):
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


def _validate_artifact_component(value: str, field_name: str) -> str:
    """Validate one human-readable directory name without allowing traversal."""
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
        or "\0" in value
    ):
        raise ValueError(f"{field_name} is not a safe path component")
    return value


def _normalize_temperature_unit(input_unit: str) -> str:
    if not isinstance(input_unit, str):
        raise ValueError("input_unit must be Celsius or Kelvin")
    normalized_unit = input_unit.lower()
    if normalized_unit in {"c", "celsius"}:
        return "celsius"
    if normalized_unit in {"k", "kelvin"}:
        return "kelvin"
    raise ValueError("input_unit must be Celsius or Kelvin")


def _utc_artifact_time(value: datetime | None, field_name: str) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError(f"{field_name} must be a timezone-aware datetime")
    if value.utcoffset() is None:
        raise ValueError(f"{field_name} must have a valid UTC offset")
    return value.astimezone(timezone.utc)


def _format_utc_artifact_time(value: datetime) -> str:
    return value.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _format_unix_time_utc(value: int) -> str:
    return _format_utc_artifact_time(datetime.fromtimestamp(value, timezone.utc))


def _json_metadata_copy(value: Any, field_name: str) -> Any:
    try:
        serialized = json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field_name} must be JSON serializable") from error
    return json.loads(serialized)


def _write_artifact_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as output_file:
            json.dump(
                value,
                output_file,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            output_file.write("\n")
            output_file.flush()
            os.fsync(output_file.fileno())
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def _read_artifact_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as input_file:
            value = json.load(input_file)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read EMOS artifact metadata: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"EMOS artifact metadata must be an object: {path}")
    return value


def _artifact_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _save_hourly_emos_r_fit(fit: Any, path: Path) -> None:
    robjects.r["saveRDS"](fit, str(path), compress=True)  # type: ignore


def _load_hourly_emos_r_fit(path: Path) -> Any:
    return robjects.r["readRDS"](str(path))  # type: ignore


def _describe_hourly_emos_fit(fit: Any) -> dict[str, Any]:
    fit_classes = [str(value) for value in robjects.r["class"](fit)]  # type: ignore
    if "ensembleMOSnormal" not in fit_classes:
        raise ValueError("hourly temperature fit must inherit ensembleMOSnormal")
    try:
        training = list(fit.rx2("training"))
        forecast_hour = int(robjects.r["attr"](fit, "forecastHour")[0])  # type: ignore
        initialization_time = str(
            robjects.r["attr"](fit, "initializationTime")[0]  # type: ignore
        )
        fitted_intercepts = fit.rx2("a")
    except (AttributeError, LookupError, TypeError, ValueError) as error:
        raise ValueError("invalid ensembleMOSnormal fit structure") from error
    if len(training) < 2:
        raise ValueError("ensembleMOSnormal fit has invalid training metadata")

    column_names = getattr(fitted_intercepts, "colnames", None)
    modeled_dates = (
        [] if column_names is None else [str(value) for value in column_names]
    )
    return {
        "fit_class": fit_classes,
        "resolved_training_days": int(training[0]),
        "lag_days": int(training[1]),
        "training_case_counts": [int(value) for value in training[2:]],
        "forecast_hour": forecast_hour,
        "initialization_time": initialization_time,
        "modeled_dates": modeled_dates,
        "parameter_set_count": len(modeled_dates),
        "latest_parameter_date": modeled_dates[-1] if modeled_dates else None,
    }


def _hourly_emos_runtime_metadata() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "r": str(robjects.r("R.version.string")[0]),  # type: ignore
        "rpy2": package_version("rpy2"),
        "ensembleMOS": str(
            robjects.r('as.character(packageVersion("ensembleMOS"))')[0]  # type: ignore
        ),
        "ensembleBMA": str(
            robjects.r('as.character(packageVersion("ensembleBMA"))')[0]  # type: ignore
        ),
    }


def _validate_hourly_artifact_group(
    key: Any,
    group: dict[str, Any],
) -> dict[str, Any]:
    if (
        not isinstance(key, tuple)
        or len(key) != 2
        or not isinstance(key[0], str)
        or isinstance(key[1], bool)
        or not isinstance(key[1], int)
    ):
        raise ValueError(f"invalid hourly EMOS group key: {key!r}")
    try:
        initialization_time = str(group["initialization_time"])
        forecast_hour = int(group["forecast_hour"])
        member_names = tuple(group["member_names"])
        valid_times = tuple(int(value) for value in group["valid_times"])
        forecasts = tuple(group["forecasts"])
        observations = tuple(group["observations"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid hourly EMOS group {key!r}") from error
    if key != (initialization_time, forecast_hour):
        raise ValueError(f"hourly EMOS group metadata does not match {key!r}")
    if (
        len(initialization_time) != 2
        or not initialization_time.isdigit()
        or not 0 <= int(initialization_time) <= 23
        or forecast_hour < 0
    ):
        raise ValueError(f"invalid initialization/lead for group {key!r}")
    if not member_names or not all(
        isinstance(member_name, str) and member_name for member_name in member_names
    ):
        raise ValueError(f"invalid member_names for group {key!r}")
    if len(set(member_names)) != len(member_names):
        raise ValueError(f"duplicate member_names for group {key!r}")
    if not valid_times or not (len(valid_times) == len(forecasts) == len(observations)):
        raise ValueError(f"unaligned or empty hourly EMOS group {key!r}")
    if any(len(tuple(row)) != len(member_names) for row in forecasts):
        raise ValueError(f"forecast member count mismatch for group {key!r}")

    raw_initialization_times = group.get("initialization_times", ())
    try:
        initialization_times = tuple(int(value) for value in raw_initialization_times)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid initialization_times for group {key!r}") from error
    if initialization_times and len(initialization_times) != len(valid_times):
        raise ValueError(f"initialization_times are unaligned for group {key!r}")

    return {
        "initialization_time": initialization_time,
        "forecast_hour": forecast_hour,
        "member_names": member_names,
        "valid_times": valid_times,
        "initialization_times": initialization_times,
    }


def save_hourly_temperature_emos_fits(
    fits: dict[EmosGroupKey, Any],
    groups: dict[EmosGroupKey, dict[str, Any]],
    city_name: str,
    model_name: str,
    *,
    training_days: int | None = None,
    input_unit: str = "celsius",
    exchangeable: bool | list[str] | tuple[str, ...] = True,
    consecutive: bool = False,
    warm_start: bool = False,
    skip_insufficient: bool = False,
    training_completed_at: datetime | None = None,
    extra_metadata: dict[str, Any] | None = None,
    artifact_root: str | Path = DEFAULT_MODEL_ARTIFACTS_DIR,
) -> Path:
    """Save hourly EMOS R fits and metadata as a new immutable version."""
    city_name = _validate_artifact_component(city_name, "city_name")
    model_name = _validate_artifact_component(model_name, "model_name")
    if not fits:
        raise ValueError("cannot save an empty hourly EMOS fit collection")
    if training_days is not None and (
        isinstance(training_days, bool)
        or not isinstance(training_days, int)
        or training_days <= 0
    ):
        raise ValueError("training_days must be a positive integer or None")
    normalized_unit = _normalize_temperature_unit(input_unit)
    if not all(
        isinstance(option, bool)
        for option in (consecutive, warm_start, skip_insufficient)
    ):
        raise ValueError(
            "consecutive, warm_start and skip_insufficient must be booleans"
        )
    if extra_metadata is not None and not isinstance(extra_metadata, dict):
        raise ValueError("extra_metadata must be a dictionary or None")

    completed_at = _utc_artifact_time(
        training_completed_at,
        "training_completed_at",
    )
    saved_at = datetime.now(timezone.utc)
    normalized_exchangeable = _json_metadata_copy(
        exchangeable,
        "exchangeable",
    )
    normalized_extra_metadata = _json_metadata_copy(
        {} if extra_metadata is None else extra_metadata,
        "extra_metadata",
    )

    prepared_groups = {
        key: _validate_hourly_artifact_group(key, group)
        for key, group in groups.items()
    }
    unknown_fit_keys = set(fits) - set(prepared_groups)
    if unknown_fit_keys:
        raise ValueError(
            f"fits have no matching hourly source groups: "
            f"{sorted(unknown_fit_keys)!r}"
        )

    version = completed_at.strftime("%Y%m%dT%H%M%S.%fZ")
    version += f"-{uuid4().hex[:12]}"
    artifact_type_directory = (
        Path(artifact_root) / HOURLY_TEMPERATURE_EMOS_ARTIFACT_TYPE
    )
    model_directory = artifact_type_directory / city_name / model_name
    versions_directory = model_directory / "versions"
    versions_directory.mkdir(parents=True, exist_ok=True)
    version_directory = versions_directory / version
    if version_directory.exists():
        raise FileExistsError(f"hourly EMOS artifact already exists: {version}")

    temporary_directory = Path(
        tempfile.mkdtemp(prefix=f".tmp-{version}-", dir=versions_directory)
    )
    published = False
    try:
        fits_directory = temporary_directory / "fits"
        fits_directory.mkdir()
        fitted_group_entries: list[dict[str, Any]] = []

        for key in sorted(fits):
            initialization_time, forecast_hour = key
            prepared_group = prepared_groups[key]
            fit_description = _describe_hourly_emos_fit(fits[key])
            if fit_description["initialization_time"] != initialization_time:
                raise ValueError(f"fit initialization time does not match {key!r}")
            if fit_description["forecast_hour"] != forecast_hour:
                raise ValueError(f"fit forecast hour does not match {key!r}")
            if (
                training_days is not None
                and fit_description["resolved_training_days"] != training_days
            ):
                raise ValueError(f"fit trainingDays does not match {key!r}")

            relative_fit_path = Path(
                "fits",
                f"init_{initialization_time}_lead_{forecast_hour:03d}.rds",
            )
            fit_path = temporary_directory / relative_fit_path
            _save_hourly_emos_r_fit(fits[key], fit_path)
            valid_times = prepared_group["valid_times"]
            initialization_times = prepared_group["initialization_times"]
            fitted_group_entries.append(
                {
                    "initialization_hour_utc": initialization_time,
                    "lead_hour": forecast_hour,
                    "sample_count": len(valid_times),
                    "valid_time_start_unix": min(valid_times),
                    "valid_time_end_unix": max(valid_times),
                    "valid_time_start_utc": _format_unix_time_utc(min(valid_times)),
                    "valid_time_end_utc": _format_unix_time_utc(max(valid_times)),
                    "initialization_time_start_unix": (
                        min(initialization_times) if initialization_times else None
                    ),
                    "initialization_time_end_unix": (
                        max(initialization_times) if initialization_times else None
                    ),
                    "member_count": len(prepared_group["member_names"]),
                    "member_names": list(prepared_group["member_names"]),
                    "fit_file": relative_fit_path.as_posix(),
                    "fit_size_bytes": fit_path.stat().st_size,
                    "fit_sha256": _artifact_file_sha256(fit_path),
                    **fit_description,
                }
            )

        skipped_group_entries = [
            {
                "initialization_hour_utc": key[0],
                "lead_hour": key[1],
                "sample_count": len(prepared_groups[key]["valid_times"]),
            }
            for key in sorted(set(prepared_groups) - set(fits))
        ]
        manifest = {
            "schema_version": EMOS_ARTIFACT_SCHEMA_VERSION,
            "artifact_type": HOURLY_TEMPERATURE_EMOS_ARTIFACT_TYPE,
            "model_family": "ensemble_mos",
            "target_variable": "temperature_2m",
            "temporal_aggregation": "hourly",
            "version": version,
            "city_name": city_name,
            "forecast_model_name": model_name,
            "training_completed_at_utc": _format_utc_artifact_time(completed_at),
            "saved_at_utc": _format_utc_artifact_time(saved_at),
            "fit_format": "rds",
            "stored_temperature_unit": "kelvin",
            "runtime": _hourly_emos_runtime_metadata(),
            "training_options": {
                "requested_training_days": training_days,
                "input_unit": normalized_unit,
                "exchangeable": normalized_exchangeable,
                "consecutive": consecutive,
                "warm_start": warm_start,
                "skip_insufficient": skip_insufficient,
                "model": "normal",
            },
            "summary": {
                "source_group_count": len(prepared_groups),
                "fitted_group_count": len(fits),
                "skipped_group_count": len(prepared_groups) - len(fits),
                "source_sample_count": sum(
                    len(group["valid_times"]) for group in prepared_groups.values()
                ),
                "fitted_sample_count": sum(
                    len(prepared_groups[key]["valid_times"]) for key in fits
                ),
            },
            "groups": fitted_group_entries,
            "skipped_groups": skipped_group_entries,
            "extra_metadata": normalized_extra_metadata,
        }
        _write_artifact_json_atomic(
            temporary_directory / "manifest.json",
            manifest,
        )
        os.replace(temporary_directory, version_directory)
        published = True
        _write_artifact_json_atomic(
            model_directory / "latest.json",
            {
                "schema_version": EMOS_ARTIFACT_SCHEMA_VERSION,
                "artifact_type": HOURLY_TEMPERATURE_EMOS_ARTIFACT_TYPE,
                "version": version,
                "updated_at_utc": _format_utc_artifact_time(saved_at),
            },
        )
        return version_directory
    finally:
        if not published:
            shutil.rmtree(temporary_directory, ignore_errors=True)


def _resolve_hourly_emos_artifact_version(
    model_directory: Path,
    version: str | None,
) -> tuple[str, Path]:
    versions_directory = model_directory / "versions"
    if version is None or version == "latest":
        latest_path = model_directory / "latest.json"
        if latest_path.exists():
            latest = _read_artifact_json(latest_path)
            selected_version = latest.get("version")
            if not isinstance(selected_version, str):
                raise ValueError(f"invalid hourly EMOS latest pointer: {latest_path}")
        else:
            candidates = sorted(
                path.name
                for path in versions_directory.iterdir()
                if path.is_dir()
                and not path.name.startswith(".")
                and (path / "manifest.json").is_file()
            )
            if not candidates:
                raise FileNotFoundError(
                    f"no saved hourly EMOS versions in {model_directory}"
                )
            selected_version = candidates[-1]
    else:
        selected_version = version
    selected_version = _validate_artifact_component(
        selected_version,
        "version",
    )
    version_directory = versions_directory / selected_version
    if not version_directory.is_dir():
        raise FileNotFoundError(
            f"hourly EMOS version does not exist: {version_directory}"
        )
    return selected_version, version_directory


def load_hourly_temperature_emos_fits(
    city_name: str,
    model_name: str,
    *,
    version: str | None = "latest",
    artifact_root: str | Path = DEFAULT_MODEL_ARTIFACTS_DIR,
    verify_checksums: bool = True,
) -> HourlyTemperatureEmosArtifact:
    """Restore the latest or an explicitly versioned hourly EMOS artifact."""
    city_name = _validate_artifact_component(city_name, "city_name")
    model_name = _validate_artifact_component(model_name, "model_name")
    model_directory = (
        Path(artifact_root)
        / HOURLY_TEMPERATURE_EMOS_ARTIFACT_TYPE
        / city_name
        / model_name
    )
    if not model_directory.is_dir():
        raise FileNotFoundError(
            f"no hourly temperature EMOS artifacts for {city_name}/{model_name}"
        )
    selected_version, version_directory = _resolve_hourly_emos_artifact_version(
        model_directory, version
    )
    manifest_path = version_directory / "manifest.json"
    manifest = _read_artifact_json(manifest_path)
    if manifest.get("schema_version") != EMOS_ARTIFACT_SCHEMA_VERSION:
        raise ValueError(f"unsupported hourly EMOS schema: {manifest_path}")
    if manifest.get("artifact_type") != HOURLY_TEMPERATURE_EMOS_ARTIFACT_TYPE:
        raise ValueError(f"unexpected artifact type: {manifest_path}")
    if (
        manifest.get("version") != selected_version
        or manifest.get("city_name") != city_name
        or manifest.get("forecast_model_name") != model_name
    ):
        raise ValueError(f"hourly EMOS manifest identity mismatch: {manifest_path}")

    group_entries = manifest.get("groups")
    if not isinstance(group_entries, list) or not group_entries:
        raise ValueError(f"hourly EMOS manifest has no fitted groups: {manifest_path}")
    fits: dict[EmosGroupKey, Any] = {}
    for entry in group_entries:
        if not isinstance(entry, dict):
            raise ValueError(f"invalid hourly EMOS group metadata: {manifest_path}")
        try:
            initialization_time = str(entry["initialization_hour_utc"])
            forecast_hour = int(entry["lead_hour"])
            relative_fit_path = Path(entry["fit_file"])
            expected_checksum = str(entry["fit_sha256"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"invalid hourly EMOS group metadata: {manifest_path}"
            ) from error
        if (
            relative_fit_path.is_absolute()
            or ".." in relative_fit_path.parts
            or relative_fit_path.parts[:1] != ("fits",)
        ):
            raise ValueError(f"unsafe RDS path in manifest: {manifest_path}")
        fit_path = version_directory / relative_fit_path
        if not fit_path.is_file():
            raise FileNotFoundError(f"saved hourly EMOS fit is missing: {fit_path}")
        if verify_checksums and (_artifact_file_sha256(fit_path) != expected_checksum):
            raise ValueError(f"saved hourly EMOS checksum mismatch: {fit_path}")

        key = (initialization_time, forecast_hour)
        if key in fits:
            raise ValueError(f"duplicate hourly EMOS group: {key!r}")
        fit = _load_hourly_emos_r_fit(fit_path)
        fit_description = _describe_hourly_emos_fit(fit)
        if (
            fit_description["initialization_time"] != initialization_time
            or fit_description["forecast_hour"] != forecast_hour
        ):
            raise ValueError(f"restored hourly EMOS metadata mismatch: {fit_path}")
        fits[key] = fit

    return HourlyTemperatureEmosArtifact(
        version=selected_version,
        path=version_directory,
        fits=fits,
        metadata=manifest,
    )


def _prepare_hourly_emos_batch_cities(
    cities: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None,
    input_unit: str | None,
) -> tuple[_HourlyEmosBatchCity, ...]:
    if cities is None:
        import config

        configured_cities = config.CITY
    else:
        configured_cities = cities
    if not isinstance(configured_cities, (list, tuple)) or not configured_cities:
        raise ValueError("cities must contain at least one city configuration")

    prepared_cities: list[_HourlyEmosBatchCity] = []
    seen_pairs: set[tuple[str, str]] = set()
    for city_index, city in enumerate(configured_cities):
        if not isinstance(city, dict):
            raise ValueError(f"city configuration {city_index} must be a dictionary")
        try:
            city_name = _validate_artifact_component(city["name"], "city name")
            timezone_name = city["timezone"]
            model_configs = city["models"]
        except KeyError as error:
            raise ValueError(
                f"city configuration {city_index} is missing {error.args[0]!r}"
            ) from error
        if not isinstance(timezone_name, str) or not timezone_name:
            raise ValueError(f"city {city_name!r} has an invalid timezone")
        try:
            city_timezone = ZoneInfo(timezone_name)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"city {city_name!r} has unknown timezone {timezone_name!r}"
            ) from error

        configured_unit = input_unit
        if configured_unit is None:
            configured_unit = city.get("temp_unit", "celsius")
        normalized_unit = _normalize_temperature_unit(configured_unit)
        if not isinstance(model_configs, (list, tuple)) or not model_configs:
            raise ValueError(f"city {city_name!r} must configure at least one model")

        model_names: list[str] = []
        for model_index, model in enumerate(model_configs):
            if not isinstance(model, dict) or "name" not in model:
                raise ValueError(
                    f"model configuration {model_index} for {city_name!r} "
                    "must contain a name"
                )
            model_name = _validate_artifact_component(model["name"], "model name")
            pair = (city_name, model_name)
            if pair in seen_pairs:
                raise ValueError(
                    f"duplicate city/model configuration: {city_name}/{model_name}"
                )
            seen_pairs.add(pair)
            model_names.append(model_name)
        prepared_cities.append(
            _HourlyEmosBatchCity(
                name=city_name,
                timezone=city_timezone,
                input_unit=normalized_unit,
                model_names=tuple(model_names),
            )
        )
    return tuple(prepared_cities)


def train_all_ensemble_mos(
    *,
    cities: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None = None,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    artifact_root: str | Path = DEFAULT_MODEL_ARTIFACTS_DIR,
    training_days: int | None = 30,
    lead_step_hours: int | None = None,
    max_lead_hours: int | None = 96,
    require_available_before_valid: bool = True,
    input_unit: str | None = None,
    exchangeable: bool | list[str] | tuple[str, ...] = True,
    consecutive: bool = False,
    control: Any | None = None,
    warm_start: bool = False,
    skip_insufficient: bool = False,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[tuple[str, str], Path]:
    """Train and save hourly temperature EMOS fits for all config pairs.

    ``cities=None`` reads ``config.CITY`` at call time.  The production-safe
    default requires 30 dates per initialization/lead group; pass ``None``
    explicitly only when adaptive short-window fitting is intentional.
    """
    if extra_metadata is not None and not isinstance(extra_metadata, dict):
        raise ValueError("extra_metadata must be a dictionary or None")
    prepared_cities = _prepare_hourly_emos_batch_cities(cities, input_unit)
    source_directory = Path(data_dir)
    artifact_paths: dict[tuple[str, str], Path] = {}

    for city in prepared_cities:
        temperatures = load_temperature(city.name, data_dir=source_directory)
        for model_name in city.model_names:
            forecasts = load_forecast(
                city.name,
                model_name,
                data_dir=source_directory,
            )
            try:
                groups = group_emos_training_data(
                    forecasts,
                    temperatures,
                    lead_step_hours=lead_step_hours,
                    max_lead_hours=max_lead_hours,
                    require_available_before_valid=(require_available_before_valid),
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    "failed to group hourly EMOS data for "
                    f"{city.name}/{model_name}: {error}"
                ) from error
            if not groups:
                raise ValueError(
                    f"no hourly EMOS training groups for {city.name}/{model_name}"
                )
            try:
                fits = train_grouped_ensemble_mos(
                    groups,
                    training_days=training_days,
                    input_unit=city.input_unit,
                    exchangeable=exchangeable,
                    consecutive=consecutive,
                    control=control,
                    warm_start=warm_start,
                    skip_insufficient=skip_insufficient,
                )
            except ValueError as error:
                raise ValueError(
                    "failed to train hourly EMOS for "
                    f"{city.name}/{model_name}: {error}"
                ) from error
            except Exception as error:
                raise RuntimeError(
                    "R hourly EMOS training failed for "
                    f"{city.name}/{model_name}: {error}"
                ) from error
            if not fits:
                raise ValueError(
                    "no hourly EMOS fits produced for "
                    f"{city.name}/{model_name}; all groups may be insufficient"
                )

            artifact_metadata = dict(extra_metadata or {})
            artifact_metadata["batch_training"] = {
                "data_directory": str(source_directory),
                "city_timezone": city.timezone.key,
                "forecast_record_count": len(forecasts),
                "temperature_record_count": len(temperatures),
                "grouping_options": {
                    "lead_step_hours": lead_step_hours,
                    "max_lead_hours": max_lead_hours,
                    "require_available_before_valid": (require_available_before_valid),
                },
                "custom_control_supplied": control is not None,
            }
            artifact_paths[(city.name, model_name)] = save_hourly_temperature_emos_fits(
                fits,
                groups,
                city.name,
                model_name,
                training_days=training_days,
                input_unit=city.input_unit,
                exchangeable=exchangeable,
                consecutive=consecutive,
                warm_start=warm_start,
                skip_insufficient=skip_insufficient,
                training_completed_at=datetime.now(timezone.utc),
                extra_metadata=artifact_metadata,
                artifact_root=artifact_root,
            )
    return artifact_paths

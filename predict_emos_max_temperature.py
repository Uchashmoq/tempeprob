"""Predict a local-day maximum-temperature probability with a saved EMOS fit.

The training target in :mod:`train_emos_max_temperature` is one local
calendar day's maximum temperature for each ensemble member.  Prediction must
therefore reproduce the same calendar-day and forecast-run grouping before it
can pass a one-row ``ensembleData`` object to ``ensembleMOS::cdf``.
"""

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
import json
from math import isfinite
from pathlib import Path
from typing import Any, Sequence
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from train_emos_max_temperature import (
    DATA_DIR,
    HIGHEST_TEMPERATURE_EMOS_DIR,
    DailyMaxGroupKey,
    DailyMaxTemperatureEmosArtifact,
    _daily_member_maxima,
    _has_complete_regular_forecast_grid,
    _load_forecasts,
    _local_day_bounds,
    _ordered_forecasts,
    _prepare_forecast_run,
    load_daily_max_temperature_emos_fits,
)

__all__ = [
    "DailyMaxTemperatureIntervalPrediction",
    "TemperatureIntervalProbability",
    "predict_daily_max_temperature_intervals",
    "probability_daily_max_temperature_below",
]

_CELSIUS_TO_KELVIN = 273.15


class _PredictionUnavailableError(ValueError):
    """The requested day has no operational forecast/fit combination."""


class _NoMatchingForecastError(_PredictionUnavailableError):
    """No forecast case can use the requested artifact."""


@dataclass(frozen=True)
class TemperatureIntervalProbability:
    """Probability for ``lower_bound <= T < upper_bound``.

    A ``None`` bound represents negative or positive infinity respectively.
    """

    lower_bound: float | None
    upper_bound: float | None
    probability: float
    unit: str

    @property
    def label(self) -> str:
        unit_label = "°C" if self.unit == "celsius" else "K"

        def bound_label(value: float) -> str:
            return f"{value:g}{unit_label}"

        if self.lower_bound is None:
            if self.upper_bound is None:
                raise ValueError("temperature interval cannot be unbounded both ways")
            return f"T < {bound_label(self.upper_bound)}"
        if self.upper_bound is None:
            return f"{bound_label(self.lower_bound)} <= T"
        return (
            f"{bound_label(self.lower_bound)} <= T < "
            f"{bound_label(self.upper_bound)}"
        )


@dataclass(frozen=True)
class DailyMaxTemperatureIntervalPrediction:
    """Interval probabilities, or an explicit reason a day is unavailable."""

    target_date: date
    intervals: tuple[TemperatureIntervalProbability, ...]
    unavailable_reason: str | None = None

    @property
    def available(self) -> bool:
        return self.unavailable_reason is None


@dataclass(frozen=True)
class _ArtifactGroup:
    key: DailyMaxGroupKey
    fit: Any
    modeled_dates: frozenset[str]
    member_names: tuple[str, ...]
    timezone_name: str | None


@dataclass(frozen=True)
class _DailyMaxPredictionCase:
    """A forecast run reduced to the predictors required by one EMOS fit."""

    group: _ArtifactGroup
    initialization_time: int
    availability_time: int
    member_maxima: tuple[float, ...]


def _parse_target_date(value: date | str) -> date:
    if isinstance(value, datetime):
        raise TypeError("target_date must be a date or a YYYY-MM-DD string")
    if isinstance(value, date):
        return value
    if not isinstance(value, str):
        raise TypeError("target_date must be a date or a YYYY-MM-DD string")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as error:
        raise ValueError("target_date must use YYYY-MM-DD") from error
    if value != parsed.isoformat():
        raise ValueError("target_date must use YYYY-MM-DD")
    return parsed


def _normalize_as_of(value: datetime | None) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    if not isinstance(value, datetime):
        raise TypeError("as_of must be a timezone-aware datetime or None")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("as_of must be timezone-aware")
    return value.astimezone(timezone.utc)


def _parse_manifest_utc_time(value: Any, field_name: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"EMOS manifest {field_name} must be an UTC timestamp")
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as error:
        raise ValueError(
            f"EMOS manifest {field_name} must be an ISO timestamp"
        ) from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"EMOS manifest {field_name} must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _validate_artifact_availability(
    artifact: DailyMaxTemperatureEmosArtifact,
    as_of: datetime,
) -> None:
    """Prevent historical calls from loading a fit created in the future."""
    available_at = _artifact_available_at(artifact.metadata)
    if available_at > as_of:
        raise ValueError(
            f"EMOS artifact {artifact.version!r} was not available at "
            f"as_of={as_of.isoformat()}; choose an older artifact version"
        )


def _artifact_available_at(metadata: dict[str, Any]) -> datetime:
    available_at = _parse_manifest_utc_time(
        metadata.get("training_completed_at_utc"),
        "training_completed_at_utc",
    )
    saved_at_value = metadata.get("saved_at_utc")
    if saved_at_value is not None:
        available_at = max(
            available_at,
            _parse_manifest_utc_time(saved_at_value, "saved_at_utc"),
        )
    return available_at


def _automatic_artifact_versions(
    city_name: str,
    model_name: str,
    target_date: date,
    as_of: datetime,
    artifact_dir: str | Path,
) -> tuple[str, ...]:
    """Return newest-first saved versions that may model ``target_date``."""
    model_directory = Path(artifact_dir) / city_name / model_name
    if not model_directory.is_dir():
        raise _PredictionUnavailableError(
            f"no highest-temperature EMOS artifacts for " f"{city_name}/{model_name}"
        )

    target_r_date = target_date.strftime("%Y%m%d")
    candidates: list[tuple[datetime, str]] = []
    for version_directory in model_directory.iterdir():
        manifest_path = version_directory / "manifest.json"
        if (
            not version_directory.is_dir()
            or version_directory.name.startswith(".")
            or not manifest_path.is_file()
        ):
            continue
        try:
            with manifest_path.open("r", encoding="utf-8") as manifest_file:
                metadata = json.load(manifest_file)
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(
                f"cannot read EMOS metadata file: {manifest_path}"
            ) from error
        if not isinstance(metadata, dict):
            raise ValueError(f"EMOS metadata must be an object: {manifest_path}")
        if (
            metadata.get("city_name") != city_name
            or metadata.get("model_name") != model_name
            or metadata.get("version") != version_directory.name
        ):
            raise ValueError(f"EMOS manifest identity mismatch: {manifest_path}")

        groups = metadata.get("groups")
        if not isinstance(groups, list) or not any(
            isinstance(group, dict)
            and isinstance(group.get("modeled_dates"), list)
            and target_r_date in group["modeled_dates"]
            for group in groups
        ):
            continue
        available_at = _artifact_available_at(metadata)
        if available_at <= as_of:
            candidates.append((available_at, version_directory.name))

    candidates.sort(reverse=True)
    if not candidates:
        raise _PredictionUnavailableError(
            f"no saved EMOS artifact available by {as_of.isoformat()} has "
            f"parameters for target date {target_date}; run rolling training "
            "or pass an explicit artifact_version"
        )
    return tuple(version for _, version in candidates)


def _normalize_temperature_unit(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be 'celsius' or 'kelvin'")
    normalized = value.lower()
    if normalized in {"c", "celsius"}:
        return "celsius"
    if normalized in {"k", "kelvin"}:
        return "kelvin"
    raise ValueError(f"{field_name} must be 'celsius' or 'kelvin'")


def _temperature_to_kelvin(value: Any, input_unit: str) -> float:
    if isinstance(value, bool):
        raise ValueError("temperature must be a finite number")
    try:
        temperature = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("temperature must be a finite number") from error
    if not isfinite(temperature):
        raise ValueError("temperature must be a finite number")
    if input_unit == "celsius":
        temperature += _CELSIUS_TO_KELVIN
    elif input_unit != "kelvin":
        raise ValueError("input_unit must be 'celsius' or 'kelvin'")
    if temperature < 0:
        raise ValueError("temperature cannot be below absolute zero")
    return temperature


def _configured_city(city_name: str) -> dict[str, Any] | None:
    import config

    matches = [
        city
        for city in config.CITY
        if isinstance(city, dict) and city.get("name") == city_name
    ]
    if len(matches) > 1:
        raise ValueError(f"config contains duplicate city entries for {city_name!r}")
    return None if not matches else matches[0]


def _artifact_groups(
    artifact: DailyMaxTemperatureEmosArtifact,
) -> dict[DailyMaxGroupKey, _ArtifactGroup]:
    raw_entries = artifact.metadata.get("groups")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise ValueError("EMOS artifact manifest contains no fitted groups")

    groups: dict[DailyMaxGroupKey, _ArtifactGroup] = {}
    for entry in raw_entries:
        if not isinstance(entry, dict):
            raise ValueError("EMOS artifact contains invalid group metadata")
        try:
            initialization_hour = str(entry["initialization_hour_utc"])
            raw_day_ahead = entry["day_ahead"]
            raw_modeled_dates = entry["modeled_dates"]
            raw_member_names = entry["member_names"]
        except KeyError as error:
            raise ValueError(
                "EMOS artifact group is missing prediction metadata"
            ) from error

        if (
            len(initialization_hour) != 2
            or not initialization_hour.isdigit()
            or not 0 <= int(initialization_hour) <= 23
        ):
            raise ValueError("EMOS artifact contains an invalid initialization hour")
        if (
            isinstance(raw_day_ahead, bool)
            or not isinstance(raw_day_ahead, int)
            or raw_day_ahead <= 0
        ):
            raise ValueError("EMOS artifact contains an invalid day_ahead")
        if not isinstance(raw_modeled_dates, list) or not raw_modeled_dates:
            raise ValueError("EMOS artifact group contains no modeled_dates")
        modeled_dates: set[str] = set()
        for raw_date in raw_modeled_dates:
            if not isinstance(raw_date, str):
                raise ValueError("EMOS artifact modeled_dates must be strings")
            try:
                parsed_date = datetime.strptime(raw_date, "%Y%m%d").date()
            except ValueError as error:
                raise ValueError(
                    f"invalid EMOS artifact modeled date: {raw_date!r}"
                ) from error
            if raw_date != parsed_date.strftime("%Y%m%d"):
                raise ValueError(f"invalid EMOS artifact modeled date: {raw_date!r}")
            modeled_dates.add(raw_date)

        if not isinstance(raw_member_names, list) or not raw_member_names:
            raise ValueError("EMOS artifact group contains no member_names")
        member_names = tuple(raw_member_names)
        if not all(isinstance(name, str) and name for name in member_names) or len(
            set(member_names)
        ) != len(member_names):
            raise ValueError("EMOS artifact group contains invalid member_names")

        timezone_name = entry.get("timezone")
        if timezone_name is not None and not isinstance(timezone_name, str):
            raise ValueError("EMOS artifact group contains an invalid timezone")

        key = (initialization_hour, raw_day_ahead)
        if key in groups:
            raise ValueError(f"duplicate EMOS artifact group: {key!r}")
        if key not in artifact.fits:
            raise ValueError(f"EMOS artifact group has no restored fit: {key!r}")
        if entry.get("forecast_hour") != raw_day_ahead * 24:
            raise ValueError(f"EMOS artifact forecast hour does not match {key!r}")
        if str(entry.get("initialization_time")) != initialization_hour:
            raise ValueError(
                f"EMOS artifact initialization time does not match {key!r}"
            )

        groups[key] = _ArtifactGroup(
            key=key,
            fit=artifact.fits[key],
            modeled_dates=frozenset(modeled_dates),
            member_names=member_names,
            timezone_name=timezone_name,
        )

    unknown_fit_keys = set(artifact.fits) - set(groups)
    if unknown_fit_keys:
        raise ValueError(
            f"restored fits have no manifest groups: {sorted(unknown_fit_keys)!r}"
        )
    return groups


def _resolve_city_timezone(
    city_name: str,
    groups: dict[DailyMaxGroupKey, _ArtifactGroup],
    supplied_timezone: ZoneInfo | str | None,
) -> ZoneInfo:
    artifact_timezone_names = {
        group.timezone_name
        for group in groups.values()
        if group.timezone_name is not None
    }
    if len(artifact_timezone_names) > 1:
        raise ValueError("EMOS artifact groups use inconsistent timezones")
    artifact_timezone_name = (
        None if not artifact_timezone_names else next(iter(artifact_timezone_names))
    )

    configured = _configured_city(city_name)
    configured_timezone_name = (
        None if configured is None else configured.get("timezone")
    )
    if configured_timezone_name is not None and not isinstance(
        configured_timezone_name,
        str,
    ):
        raise ValueError(f"config timezone for {city_name!r} must be a string")

    if isinstance(supplied_timezone, ZoneInfo):
        timezone_value = supplied_timezone
    else:
        timezone_name = supplied_timezone
        if timezone_name is None:
            timezone_name = artifact_timezone_name or configured_timezone_name
        if not isinstance(timezone_name, str) or not timezone_name:
            raise ValueError(
                f"cannot determine timezone for {city_name!r}; pass city_timezone"
            )
        try:
            timezone_value = ZoneInfo(timezone_name)
        except ZoneInfoNotFoundError as error:
            raise ValueError(f"unknown city timezone: {timezone_name!r}") from error

    if (
        artifact_timezone_name is not None
        and timezone_value.key != artifact_timezone_name
    ):
        raise ValueError(
            f"city timezone {timezone_value.key!r} does not match the artifact "
            f"timezone {artifact_timezone_name!r}"
        )
    if (
        configured_timezone_name is not None
        and timezone_value.key != configured_timezone_name
    ):
        raise ValueError(
            f"city timezone {timezone_value.key!r} does not match config "
            f"timezone {configured_timezone_name!r}"
        )
    return timezone_value


def _batch_grouping_options(metadata: dict[str, Any]) -> dict[str, Any]:
    extra_metadata = metadata.get("extra_metadata")
    if not isinstance(extra_metadata, dict):
        return {}
    batch_training = extra_metadata.get("batch_training")
    if not isinstance(batch_training, dict):
        return {}
    grouping_options = batch_training.get("grouping_options")
    return grouping_options if isinstance(grouping_options, dict) else {}


def _resolve_expected_interval_seconds(
    metadata: dict[str, Any],
    supplied_value: int | None,
) -> int:
    stored_value = _batch_grouping_options(metadata).get("expected_interval_seconds")
    value = (
        stored_value
        if supplied_value is None and stored_value is not None
        else supplied_value
    )
    if value is None:
        value = 3600
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("expected_interval_seconds must be a positive integer")
    if (
        supplied_value is not None
        and stored_value is not None
        and supplied_value != stored_value
    ):
        raise ValueError(
            "expected_interval_seconds must match the value used for training"
        )
    return value


def _resolve_minimum_notice_hours(
    metadata: dict[str, Any],
    supplied_value: float | None,
) -> float:
    stored_value = _batch_grouping_options(metadata).get("minimum_notice_hours")
    raw_value = (
        stored_value
        if supplied_value is None and stored_value is not None
        else supplied_value
    )
    if raw_value is None:
        raw_value = 0.0
    if isinstance(raw_value, bool):
        raise ValueError("minimum_notice_hours must be a non-negative number")
    try:
        value = float(raw_value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "minimum_notice_hours must be a non-negative number"
        ) from error
    if not isfinite(value) or value < 0:
        raise ValueError("minimum_notice_hours must be a non-negative number")
    if (
        supplied_value is not None
        and stored_value is not None
        and value != float(stored_value)
    ):
        raise ValueError("minimum_notice_hours must match the value used for training")
    return value


def _resolve_forecast_input_unit(
    city_name: str,
    metadata: dict[str, Any],
    supplied_unit: str | None,
) -> str:
    training_options = metadata.get("training_options")
    if not isinstance(training_options, dict):
        raise ValueError("EMOS artifact is missing training_options")
    trained_input_unit = _normalize_temperature_unit(
        training_options.get("input_unit"),
        "EMOS training input_unit",
    )
    if training_options.get("model") != "normal":
        raise ValueError("daily maximum-temperature artifact must use model='normal'")
    if (
        _normalize_temperature_unit(
            metadata.get("stored_temperature_unit"),
            "EMOS stored_temperature_unit",
        )
        != "kelvin"
    ):
        raise ValueError("daily maximum-temperature fits must be stored in kelvin")

    if supplied_unit is not None:
        return _normalize_temperature_unit(supplied_unit, "forecast_input_unit")

    configured = _configured_city(city_name)
    configured_unit = None if configured is None else configured.get("temp_unit")
    if configured_unit is not None:
        config_unit = _normalize_temperature_unit(
            configured_unit,
            f"config temperature unit for {city_name!r}",
        )
        if config_unit != trained_input_unit:
            raise ValueError(
                f"config temperature unit {config_unit!r} does not match the "
                f"artifact training unit {trained_input_unit!r}"
            )
    return trained_input_unit


def _select_daily_max_forecast(
    forecasts: list[dict[str, Any]],
    *,
    model_name: str,
    target_date: date,
    city_timezone: ZoneInfo,
    artifact_groups: dict[DailyMaxGroupKey, _ArtifactGroup],
    as_of: datetime,
    expected_interval_seconds: int,
    minimum_notice_hours: float,
) -> _DailyMaxPredictionCase:
    """Select the latest run, using the first complete snapshot of that run."""
    target_r_date = target_date.strftime("%Y%m%d")
    eligible_groups = {
        key: group
        for key, group in artifact_groups.items()
        if target_r_date in group.modeled_dates
    }
    if not eligible_groups:
        available_parameter_dates = sorted(
            {
                modeled_date
                for group in artifact_groups.values()
                for modeled_date in group.modeled_dates
            }
        )
        raise _PredictionUnavailableError(
            f"EMOS artifact has no parameters for target date {target_date}; "
            f"available modeled dates: {available_parameter_dates!r}. "
            "Run rolling training for the requested date."
        )

    day_start, day_end = _local_day_bounds(target_date, city_timezone)
    as_of_timestamp = as_of.timestamp()
    notice_seconds = minimum_notice_hours * 3600
    first_complete_by_initialization: dict[int, _DailyMaxPredictionCase] = {}

    for forecast in _ordered_forecasts(forecasts):
        run = _prepare_forecast_run(forecast, city_timezone)
        if run.model_name != model_name:
            raise ValueError(
                f"forecast record belongs to {run.model_name!r}, not {model_name!r}"
            )
        if run.initialization_time in first_complete_by_initialization:
            continue
        if run.availability_time > as_of_timestamp:
            continue
        if (
            run.availability_time >= day_start
            or day_start - run.availability_time < notice_seconds
        ):
            continue

        day_ahead = (target_date - run.initialization_local_date).days
        if day_ahead <= 0:
            continue
        key = (run.initialization_hour, day_ahead)
        group = eligible_groups.get(key)
        if group is None:
            continue
        if run.member_names != group.member_names:
            raise ValueError(
                f"forecast ensemble members do not match EMOS group {key!r}"
            )

        day_timestamps = run.timestamps_by_date.get(target_date)
        if not day_timestamps or not _has_complete_regular_forecast_grid(
            day_timestamps,
            day_start,
            day_end,
            expected_interval_seconds,
        ):
            continue
        member_maxima = _daily_member_maxima(
            run.data,
            run.member_names,
            day_timestamps,
            run.index_by_time,
        )
        if member_maxima is None:
            continue

        first_complete_by_initialization[run.initialization_time] = (
            _DailyMaxPredictionCase(
                group=group,
                initialization_time=run.initialization_time,
                availability_time=run.availability_time,
                member_maxima=member_maxima,
            )
        )

    if not first_complete_by_initialization:
        raise _NoMatchingForecastError(
            f"no complete {model_name!r} forecast for local date {target_date} "
            f"matches an EMOS parameter group and was available by "
            f"{as_of.isoformat()}"
        )
    return max(
        first_complete_by_initialization.values(),
        key=lambda case: case.initialization_time,
    )


def _build_daily_max_prediction_ensemble_data(
    case: _DailyMaxPredictionCase,
    target_date: date,
    *,
    forecast_input_unit: str,
    exchangeable: bool | list[str] | tuple[str, ...],
) -> Any:
    """Build the one-row, Kelvin ``ensembleData`` expected by the saved fit."""
    # Import the R bridge lazily so selecting forecast data does not initialize R.
    from rpy2 import robjects
    from rpy2.robjects import vectors
    from rpy2.robjects.packages import importr

    from train_emos import _exchangeable_member_groups

    maxima_kelvin = [
        _temperature_to_kelvin(value, forecast_input_unit)
        for value in case.member_maxima
    ]
    forecast_matrix = robjects.r["matrix"](
        vectors.FloatVector(maxima_kelvin),
        nrow=1,
        ncol=len(case.group.member_names),
        byrow=True,
    )  # type: ignore
    forecast_matrix.colnames = vectors.StrVector(case.group.member_names)

    initialization_hour, day_ahead = case.group.key
    arguments: dict[str, Any] = {
        "forecasts": forecast_matrix,
        "dates": vectors.StrVector([target_date.strftime("%Y%m%d")]),
        "forecastHour": day_ahead * 24,
        "initializationTime": initialization_hour,
    }
    exchangeable_groups = _exchangeable_member_groups(
        case.group.member_names,
        exchangeable,
    )
    if exchangeable_groups is not None:
        arguments["exchangeable"] = vectors.StrVector(exchangeable_groups)

    ensemble_bma = importr("ensembleBMA")
    return ensemble_bma.ensembleData(**arguments)


def _call_ensemble_mos_cdf_values(
    fit: Any,
    ensemble_data: Any,
    thresholds_kelvin: tuple[float, ...],
    target_date: date,
) -> tuple[float, ...]:
    """Evaluate the Gaussian EMOS CDF at one or more Kelvin thresholds."""
    if not thresholds_kelvin:
        raise ValueError("thresholds_kelvin must not be empty")

    from rpy2.robjects import vectors
    from rpy2.robjects.packages import importr

    ensemble_mos = importr("ensembleMOS")
    result = ensemble_mos.cdf(
        fit,
        ensemble_data,
        values=vectors.FloatVector(thresholds_kelvin),
        dates=vectors.StrVector([target_date.strftime("%Y%m%d")]),
    )
    if len(result) != len(thresholds_kelvin):
        raise ValueError(
            "ensembleMOS::cdf returned an unexpected number of probabilities"
        )

    probabilities = tuple(float(value) for value in result)
    for probability in probabilities:
        if not isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError(
                "ensembleMOS::cdf returned an invalid probability: " f"{probability!r}"
            )
    return probabilities


def _call_ensemble_mos_cdf(
    fit: Any,
    ensemble_data: Any,
    threshold_kelvin: float,
    target_date: date,
) -> float:
    """Evaluate ``P(Tmax <= threshold)`` from one Gaussian EMOS fit."""
    return _call_ensemble_mos_cdf_values(
        fit,
        ensemble_data,
        (threshold_kelvin,),
        target_date,
    )[0]


def _probabilities_daily_max_temperature_below_thresholds(
    city_name: str,
    model_name: str,
    target_date: date | str,
    thresholds: tuple[float, ...],
    *,
    threshold_unit: str = "celsius",
    city_timezone: ZoneInfo | str | None = None,
    forecast_input_unit: str | None = None,
    as_of: datetime | None = None,
    data_dir: str | Path = DATA_DIR,
    artifact_dir: str | Path = HIGHEST_TEMPERATURE_EMOS_DIR,
    artifact_version: str | None = "auto",
    expected_interval_seconds: int | None = None,
    minimum_notice_hours: float | None = None,
    verify_checksums: bool = True,
) -> tuple[float, ...]:
    """Return the CDF values for one local day and several thresholds.

    ``target_date`` is a calendar date in the city's configured timezone.
    Forecast snapshots come from
    ``data/forecast/<city>/<model>/fc.jsonl`` by default.  The selected fit
    comes from
    ``train/highest_temperature_emos/<city>/<model>/<version>``.

    The fit is usable only when its exact modeled date, UTC initialization
    hour, day-ahead group and ensemble-member schema all match the selected
    forecast.  Within one initialization the first complete snapshot is used,
    matching the training grouper; among distinct initializations the latest
    usable run is selected.

    By default, ``artifact_version="auto"`` searches immutable saved versions
    newest-first and uses the newest version that was available at ``as_of``
    and has a usable fit for ``target_date``.  Pass ``"latest"`` or a concrete
    version string to require that exact loader selection.

    Both the member daily maxima and thresholds are converted to Kelvin before
    calling the official ``ensembleMOS::cdf`` method.
    """
    parsed_target_date = _parse_target_date(target_date)
    parsed_as_of = _normalize_as_of(as_of)
    threshold_unit_normalized = _normalize_temperature_unit(
        threshold_unit,
        "threshold_unit",
    )
    if not thresholds:
        raise ValueError("thresholds must not be empty")
    thresholds_kelvin = tuple(
        _temperature_to_kelvin(threshold, threshold_unit_normalized)
        for threshold in thresholds
    )

    automatic_version = artifact_version is None or artifact_version == "auto"
    preloaded_artifact: DailyMaxTemperatureEmosArtifact | None = None
    if automatic_version:
        versions = _automatic_artifact_versions(
            city_name,
            model_name,
            parsed_target_date,
            parsed_as_of,
            artifact_dir,
        )
    else:
        preloaded_artifact = load_daily_max_temperature_emos_fits(
            city_name,
            model_name,
            version=artifact_version,
            output_dir=artifact_dir,
            verify_checksums=verify_checksums,
        )
        _validate_artifact_availability(preloaded_artifact, parsed_as_of)
        versions = (artifact_version,)

    forecasts = _load_forecasts(
        city_name,
        model_name,
        data_dir=data_dir,
    )
    last_no_forecast_error: _NoMatchingForecastError | None = None
    for version in versions:
        # Reuse the artifact loader's identity, checksum and R-fit validation.
        if preloaded_artifact is not None:
            artifact = preloaded_artifact
        else:
            artifact = load_daily_max_temperature_emos_fits(
                city_name,
                model_name,
                version=version,
                output_dir=artifact_dir,
                verify_checksums=verify_checksums,
            )
            _validate_artifact_availability(artifact, parsed_as_of)
        groups = _artifact_groups(artifact)
        resolved_timezone = _resolve_city_timezone(
            city_name,
            groups,
            city_timezone,
        )
        resolved_interval = _resolve_expected_interval_seconds(
            artifact.metadata,
            expected_interval_seconds,
        )
        resolved_notice = _resolve_minimum_notice_hours(
            artifact.metadata,
            minimum_notice_hours,
        )
        resolved_forecast_unit = _resolve_forecast_input_unit(
            city_name,
            artifact.metadata,
            forecast_input_unit,
        )
        try:
            case = _select_daily_max_forecast(
                forecasts,
                model_name=model_name,
                target_date=parsed_target_date,
                city_timezone=resolved_timezone,
                artifact_groups=groups,
                as_of=parsed_as_of,
                expected_interval_seconds=resolved_interval,
                minimum_notice_hours=resolved_notice,
            )
        except _NoMatchingForecastError as error:
            if not automatic_version:
                raise
            last_no_forecast_error = error
            continue

        training_options = artifact.metadata["training_options"]
        exchangeable = training_options.get("exchangeable", True)
        if not isinstance(exchangeable, (bool, list, tuple)):
            raise ValueError("EMOS artifact contains an invalid exchangeable option")
        ensemble_data = _build_daily_max_prediction_ensemble_data(
            case,
            parsed_target_date,
            forecast_input_unit=resolved_forecast_unit,
            exchangeable=exchangeable,
        )
        if len(thresholds_kelvin) == 1:
            return (
                _call_ensemble_mos_cdf(
                    case.group.fit,
                    ensemble_data,
                    thresholds_kelvin[0],
                    parsed_target_date,
                ),
            )
        return _call_ensemble_mos_cdf_values(
            case.group.fit,
            ensemble_data,
            thresholds_kelvin,
            parsed_target_date,
        )

    if last_no_forecast_error is not None:
        raise _PredictionUnavailableError(
            f"no saved EMOS version has both parameters and a usable forecast "
            f"for local date {parsed_target_date}"
        ) from last_no_forecast_error
    raise RuntimeError("no EMOS artifact version was evaluated")


def probability_daily_max_temperature_below(
    city_name: str,
    model_name: str,
    target_date: date | str,
    threshold: float,
    *,
    threshold_unit: str = "celsius",
    city_timezone: ZoneInfo | str | None = None,
    forecast_input_unit: str | None = None,
    as_of: datetime | None = None,
    data_dir: str | Path = DATA_DIR,
    artifact_dir: str | Path = HIGHEST_TEMPERATURE_EMOS_DIR,
    artifact_version: str | None = "auto",
    expected_interval_seconds: int | None = None,
    minimum_notice_hours: float | None = None,
    verify_checksums: bool = True,
) -> float:
    """Return ``P(local daily Tmax < threshold)`` for one city and model."""
    return _probabilities_daily_max_temperature_below_thresholds(
        city_name,
        model_name,
        target_date,
        (threshold,),
        threshold_unit=threshold_unit,
        city_timezone=city_timezone,
        forecast_input_unit=forecast_input_unit,
        as_of=as_of,
        data_dir=data_dir,
        artifact_dir=artifact_dir,
        artifact_version=artifact_version,
        expected_interval_seconds=expected_interval_seconds,
        minimum_notice_hours=minimum_notice_hours,
        verify_checksums=verify_checksums,
    )[0]


def _validate_interval_boundaries(
    boundaries: Sequence[float],
    unit: str,
) -> tuple[float, ...]:
    if isinstance(boundaries, (str, bytes)):
        raise ValueError("boundaries must be a non-empty sequence of temperatures")
    try:
        raw_boundaries = tuple(boundaries)
    except TypeError as error:
        raise ValueError(
            "boundaries must be a non-empty sequence of temperatures"
        ) from error
    if not raw_boundaries:
        raise ValueError("boundaries must not be empty")

    normalized: list[float] = []
    for boundary in raw_boundaries:
        if isinstance(boundary, bool):
            raise ValueError("temperature boundaries must be finite numbers")
        try:
            value = float(boundary)
        except (TypeError, ValueError) as error:
            raise ValueError("temperature boundaries must be finite numbers") from error
        if not isfinite(value):
            raise ValueError("temperature boundaries must be finite numbers")
        _temperature_to_kelvin(value, unit)
        if normalized and value <= normalized[-1]:
            raise ValueError("temperature boundaries must be strictly increasing")
        normalized.append(value)
    return tuple(normalized)


def _interval_probabilities_from_cdf(
    boundaries: tuple[float, ...],
    cdf_values: tuple[float, ...],
    unit: str,
) -> tuple[TemperatureIntervalProbability, ...]:
    if len(boundaries) != len(cdf_values):
        raise ValueError("CDF count does not match temperature boundary count")

    monotonic_cdf: list[float] = []
    previous = 0.0
    for cdf_value in cdf_values:
        if not isfinite(cdf_value) or not 0 <= cdf_value <= 1:
            raise ValueError(f"invalid CDF value: {cdf_value!r}")
        if cdf_value + 1e-12 < previous:
            raise ValueError("EMOS CDF values are not non-decreasing")
        adjusted = max(previous, cdf_value)
        monotonic_cdf.append(adjusted)
        previous = adjusted

    edges = (0.0, *monotonic_cdf, 1.0)
    probabilities = tuple(
        edges[index + 1] - edges[index] for index in range(len(edges) - 1)
    )
    intervals: list[TemperatureIntervalProbability] = []
    for index, probability in enumerate(probabilities):
        intervals.append(
            TemperatureIntervalProbability(
                lower_bound=None if index == 0 else boundaries[index - 1],
                upper_bound=(None if index == len(boundaries) else boundaries[index]),
                probability=probability,
                unit=unit,
            )
        )
    return tuple(intervals)


def predict_daily_max_temperature_intervals(
    city_name: str,
    model_name: str,
    boundaries: Sequence[float],
    *,
    start_date: date | str | None = None,
    days: int = 2,
    threshold_unit: str = "celsius",
    city_timezone: ZoneInfo | str | None = None,
    forecast_input_unit: str | None = None,
    as_of: datetime | None = None,
    data_dir: str | Path = DATA_DIR,
    artifact_dir: str | Path = HIGHEST_TEMPERATURE_EMOS_DIR,
    artifact_version: str | None = "auto",
    expected_interval_seconds: int | None = None,
    minimum_notice_hours: float | None = None,
    verify_checksums: bool = True,
    allow_partial: bool = False,
) -> tuple[DailyMaxTemperatureIntervalPrediction, ...]:
    """Predict local daily-maximum probabilities for consecutive dates.

    With the defaults, the dates are today and tomorrow in the city's local
    timezone.  For boundaries ``(38, 39, 40)``, every available day contains
    the four intervals ``T < 38``,
    ``38 <= T < 39``, ``39 <= T < 40`` and ``40 <= T``.

    The boundaries are passed to ``ensembleMOS::cdf`` in one vector for each
    date.  Adjacent CDF values are differenced, producing mutually exclusive
    probabilities that sum to one.

    By default, one unavailable date fails the entire call.  With
    ``allow_partial=True``, the returned tuple still contains that date, with
    empty ``intervals`` and a non-empty ``unavailable_reason``.  Corrupt
    artifacts, invalid data and R calculation errors are never converted into
    partial results.
    """
    if isinstance(days, bool) or not isinstance(days, int) or days <= 0:
        raise ValueError("days must be a positive integer")
    if not isinstance(allow_partial, bool):
        raise ValueError("allow_partial must be a boolean")

    normalized_unit = _normalize_temperature_unit(
        threshold_unit,
        "threshold_unit",
    )
    normalized_boundaries = _validate_interval_boundaries(
        boundaries,
        normalized_unit,
    )
    parsed_as_of = _normalize_as_of(as_of)
    if start_date is None:
        reference_timezone = _resolve_city_timezone(
            city_name,
            {},
            city_timezone,
        )
        first_date = parsed_as_of.astimezone(reference_timezone).date()
    else:
        first_date = _parse_target_date(start_date)

    predictions: list[DailyMaxTemperatureIntervalPrediction] = []
    for offset in range(days):
        target_date = first_date + timedelta(days=offset)
        try:
            cdf_values = _probabilities_daily_max_temperature_below_thresholds(
                city_name,
                model_name,
                target_date,
                normalized_boundaries,
                threshold_unit=normalized_unit,
                city_timezone=city_timezone,
                forecast_input_unit=forecast_input_unit,
                as_of=parsed_as_of,
                data_dir=data_dir,
                artifact_dir=artifact_dir,
                artifact_version=artifact_version,
                expected_interval_seconds=expected_interval_seconds,
                minimum_notice_hours=minimum_notice_hours,
                verify_checksums=verify_checksums,
            )
        except _PredictionUnavailableError as error:
            if not allow_partial:
                raise _PredictionUnavailableError(
                    f"cannot predict local date {target_date}: {error}"
                ) from error
            predictions.append(
                DailyMaxTemperatureIntervalPrediction(
                    target_date=target_date,
                    intervals=(),
                    unavailable_reason=str(error),
                )
            )
            continue

        predictions.append(
            DailyMaxTemperatureIntervalPrediction(
                target_date=target_date,
                intervals=_interval_probabilities_from_cdf(
                    normalized_boundaries,
                    cdf_values,
                    normalized_unit,
                ),
            )
        )
    return tuple(predictions)

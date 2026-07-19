"""Prepare daily maximum-temperature cases for EMOS training.

One case represents one forecast run and one complete local calendar day.
For each ensemble member, the predictor is that member's maximum forecast
temperature during the day.  The response is the maximum observed temperature
during the same local day.
"""

from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
import json
from math import isfinite
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

DailyMaxGroupKey = tuple[str, int]
DATA_DIR = Path("data")

__all__ = [
    "DailyMaxGroupKey",
    "group_daily_max_temperature_emos_training_data",
]


@dataclass(frozen=True)
class _ForecastRun:
    data: dict[str, Any]
    model_name: str | None
    initialization_time: int
    availability_time: int
    initialization_hour: str
    initialization_local_date: date
    member_names: tuple[str, ...]
    timestamps_by_date: dict[date, list[int]]
    index_by_time: dict[int, int]


@dataclass(frozen=True)
class _DailyMaxCase:
    target_date: date
    day_ahead: int
    day_start: int
    day_end: int
    member_maxima: tuple[float, ...]
    observed_maximum: float
    forecast_count: int
    observation_count: int
    observation_coverage: float


def _temperature_member_names(forecast: dict[str, Any]) -> tuple[str, ...]:
    """Return the control member followed by numbered temperature members."""
    names = [
        name
        for name, values in forecast.items()
        if isinstance(values, list)
        and (name == "temperature_2m" or name.startswith("temperature_2m_member"))
    ]

    def sort_key(name: str) -> tuple[int, int, str]:
        if name == "temperature_2m":
            return (0, 0, "")
        suffix = name.removeprefix("temperature_2m_member")
        if suffix.isdigit():
            return (1, int(suffix), "")
        return (2, 0, name)

    return tuple(sorted(names, key=sort_key))


def _local_day_bounds(target_date: date, city_timezone: ZoneInfo) -> tuple[int, int]:
    """Return UTC Unix timestamps delimiting one local calendar day."""
    local_start = datetime.combine(target_date, time.min, tzinfo=city_timezone)
    local_end = datetime.combine(
        target_date + timedelta(days=1),
        time.min,
        tzinfo=city_timezone,
    )
    return int(local_start.timestamp()), int(local_end.timestamp())


def _expected_slot_count(
    day_start: int,
    day_end: int,
    interval_seconds: int,
) -> int:
    duration = day_end - day_start
    if duration <= 0 or duration % interval_seconds != 0:
        raise ValueError(
            "local day duration must be divisible by expected_interval_seconds"
        )
    return duration // interval_seconds


def _has_complete_regular_forecast_grid(
    timestamps: list[int],
    day_start: int,
    day_end: int,
    interval_seconds: int,
) -> bool:
    """Check that a forecast contains exactly one value for every day slot."""
    expected_count = _expected_slot_count(day_start, day_end, interval_seconds)
    if len(timestamps) != expected_count:
        return False
    return sorted(timestamps) == [
        day_start + index * interval_seconds for index in range(expected_count)
    ]


def _observation_slot_coverage(
    timestamps: list[int],
    day_start: int,
    day_end: int,
    interval_seconds: int,
) -> float:
    """Return the fraction of local-day time slots containing an observation."""
    expected_count = _expected_slot_count(day_start, day_end, interval_seconds)
    covered_slots = {
        (timestamp - day_start) // interval_seconds
        for timestamp in timestamps
        if day_start <= timestamp < day_end
    }
    return len(covered_slots) / expected_count


def _group_observations_by_local_date(
    temperatures: list[dict[str, Any]],
    city_timezone: ZoneInfo,
) -> dict[date, list[tuple[int, float]]]:
    """Deduplicate observations by timestamp and group them by local date.

    If a timestamp was stored more than once, the record with the latest
    ``update_time`` is treated as the final observation.
    {date:[(timestamp,temperature)]}
    """
    latest_by_time: dict[int, tuple[float, int, float]] = {}
    for position, item in enumerate(temperatures):
        try:
            timestamp = int(item["time"])
            temperature = float(item["temperature"])
            update_time = float(item["update_time"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("invalid temperature observation") from error

        if not isfinite(temperature) or not isfinite(update_time):
            raise ValueError("temperature observations must contain finite values")

        previous = latest_by_time.get(timestamp)
        if previous is None or update_time >= previous[0]:
            latest_by_time[timestamp] = (update_time, position, temperature)

    observations_by_date: dict[date, list[tuple[int, float]]] = defaultdict(list)
    for timestamp, (_, _, temperature) in latest_by_time.items():
        local_date = datetime.fromtimestamp(timestamp, city_timezone).date()
        observations_by_date[local_date].append((timestamp, temperature))

    for observations in observations_by_date.values():
        observations.sort(key=lambda item: item[0])
    return dict(observations_by_date)


def _ordered_forecasts(
    forecasts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Order runs by initialization and captured-version availability.

    Multiple records with the same initialization are deliberately retained.
    The caller can then fall back to a later captured version when an earlier
    one does not yet contain a complete target day.
    """
    try:
        ordered = sorted(
            forecasts,
            key=lambda forecast: (
                int(forecast["meta"]["last_run_initialisation_time"]),
                int(forecast["meta"]["last_run_availability_time"]),
            ),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("forecast metadata is missing or invalid") from error

    return ordered


def _validate_forecast_arrays(
    forecast: dict[str, Any],
    member_names: tuple[str, ...],
) -> tuple[list[int], dict[int, int]]:
    """Validate parallel forecast arrays and return timestamp lookup data."""
    raw_times = forecast.get("time")
    if not isinstance(raw_times, list):
        raise ValueError("forecast time must be a list")
    if not member_names:
        raise ValueError("forecast contains no 2 m temperature members")
    if any(len(forecast[name]) != len(raw_times) for name in member_names):
        raise ValueError("forecast member and time arrays have different lengths")

    try:
        timestamps = [int(value) for value in raw_times]
    except (TypeError, ValueError) as error:
        raise ValueError("forecast timestamps must be integers") from error
    if len(set(timestamps)) != len(timestamps):
        raise ValueError("forecast contains duplicate timestamps")
    return timestamps, {timestamp: index for index, timestamp in enumerate(timestamps)}


def _daily_member_maxima(
    forecast: dict[str, Any],
    member_names: tuple[str, ...],
    day_timestamps: list[int],
    index_by_time: dict[int, int],
) -> tuple[float, ...] | None:
    """Return each member's daily maximum, or None for non-finite data."""
    maxima: list[float] = []
    for member_name in member_names:
        try:
            values = [
                float(forecast[member_name][index_by_time[timestamp]])
                for timestamp in day_timestamps
            ]
        except (TypeError, ValueError) as error:
            raise ValueError("forecast temperatures must be numeric") from error
        if not all(isfinite(value) for value in values):
            return None
        maxima.append(max(values))
    return tuple(maxima)


def _prepare_forecast_run(
    forecast: dict[str, Any],
    city_timezone: ZoneInfo,
) -> _ForecastRun:
    """Validate and normalize one captured forecast run."""
    meta = forecast["meta"]
    initialization_time = int(meta["last_run_initialisation_time"])
    availability_time = int(meta["last_run_availability_time"])
    if availability_time < initialization_time:
        raise ValueError(
            "forecast availability time cannot precede initialization time"
        )

    model_name = forecast.get("model")
    if model_name is not None and not isinstance(model_name, str):
        raise ValueError("forecast model must be a string")

    member_names = _temperature_member_names(
        forecast
    )  # [temperature_2m, temperature_2m_member01
    #           {timestamp: index}
    timestamps, index_by_time = _validate_forecast_arrays(
        forecast,
        member_names,
    )
    timestamps_by_date: dict[date, list[int]] = defaultdict(list)
    for timestamp in timestamps:
        # Some Open-Meteo captures contain values before the actual model run.
        # Those are not forecasts made by this initialization.
        if timestamp < initialization_time:
            continue
        local_date = datetime.fromtimestamp(timestamp, city_timezone).date()
        timestamps_by_date[local_date].append(timestamp)

    return _ForecastRun(
        data=forecast,
        model_name=model_name,
        initialization_time=initialization_time,
        availability_time=availability_time,
        initialization_hour=datetime.fromtimestamp(
            initialization_time,
            timezone.utc,
        ).strftime("%H"),
        initialization_local_date=datetime.fromtimestamp(
            initialization_time,
            city_timezone,
        ).date(),
        member_names=member_names,
        timestamps_by_date=dict(timestamps_by_date),
        index_by_time=index_by_time,
    )


def _make_daily_max_case(
    run: _ForecastRun,
    target_date: date,
    day_timestamps: list[int],
    observations_by_date: dict[date, list[tuple[int, float]]],
    city_timezone: ZoneInfo,
    *,
    interval_seconds: int,
    minimum_observation_coverage: float,
    notice_seconds: float,
    max_day_ahead: int | None,
) -> _DailyMaxCase | None:
    """Create one complete, operationally available run/day case."""
    day_start, day_end = _local_day_bounds(target_date, city_timezone)
    # Availability itself must be strictly before local midnight.  A
    # configured notice period is inclusive: exactly N hours is sufficient.
    if (
        run.availability_time >= day_start
        or day_start - run.availability_time < notice_seconds
    ):
        return None
    if not _has_complete_regular_forecast_grid(
        day_timestamps,
        day_start,
        day_end,
        interval_seconds,
    ):
        return None

    day_ahead = (target_date - run.initialization_local_date).days
    if day_ahead <= 0:
        return None
    if max_day_ahead is not None and day_ahead > max_day_ahead:
        return None

    observations = observations_by_date.get(target_date)
    if not observations:
        return None
    observation_coverage = _observation_slot_coverage(
        [timestamp for timestamp, _ in observations],
        day_start,
        day_end,
        interval_seconds,
    )
    if observation_coverage < minimum_observation_coverage:
        return None

    member_maxima = _daily_member_maxima(
        run.data,
        run.member_names,
        day_timestamps,
        run.index_by_time,
    )
    if member_maxima is None:
        return None

    return _DailyMaxCase(
        target_date=target_date,
        day_ahead=day_ahead,
        day_start=day_start,
        day_end=day_end,
        member_maxima=member_maxima,
        observed_maximum=max(value for _, value in observations),
        forecast_count=len(day_timestamps),
        observation_count=len(observations),
        observation_coverage=observation_coverage,
    )


def _append_daily_max_case(
    groups: dict[DailyMaxGroupKey, dict[str, Any]],
    run: _ForecastRun,
    case: _DailyMaxCase,
    city_timezone: ZoneInfo,
    interval_seconds: int,
) -> None:
    """Append one case to its homogeneous group."""
    group_key = (run.initialization_hour, case.day_ahead)
    group = groups.setdefault(
        group_key,
        {
            "initialization_hour": run.initialization_hour,
            # Keep the hourly grouper's spelling for shared R-data builders.
            "initialization_time": run.initialization_hour,
            "day_ahead": case.day_ahead,
            "model": run.model_name,
            "timezone": city_timezone.key,
            "member_names": run.member_names,
            "expected_interval_seconds": interval_seconds,
            "target_dates": [],
            "target_day_start_times": [],
            "target_day_end_times": [],
            "initialization_times": [],
            "availability_times": [],
            "forecasts": [],
            "observations": [],
            "forecast_counts": [],
            "observation_counts": [],
            "observation_coverages": [],
        },
    )
    group["target_dates"].append(case.target_date.isoformat())
    group["target_day_start_times"].append(case.day_start)
    group["target_day_end_times"].append(case.day_end)
    group["initialization_times"].append(run.initialization_time)
    group["availability_times"].append(run.availability_time)
    group["forecasts"].append(case.member_maxima)
    group["observations"].append(case.observed_maximum)
    group["forecast_counts"].append(case.forecast_count)
    group["observation_counts"].append(case.observation_count)
    group["observation_coverages"].append(case.observation_coverage)


def _sort_parallel_group_fields(group: dict[str, Any]) -> None:
    """Sort a result group by target local date while preserving row alignment."""
    order = sorted(
        range(len(group["target_dates"])),
        key=lambda index: (
            group["target_dates"][index],
            group["initialization_times"][index],
        ),
    )
    for field_name in (
        "target_dates",
        "target_day_start_times",
        "target_day_end_times",
        "initialization_times",
        "availability_times",
        "forecasts",
        "observations",
        "forecast_counts",
        "observation_counts",
        "observation_coverages",
    ):
        group[field_name] = [group[field_name][index] for index in order]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load non-empty JSONL lines and require one JSON object per line."""
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as jsonl_file:
        for line_number, line in enumerate(jsonl_file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSON in {path} at line {line_number}"
                ) from error
            if not isinstance(record, dict):
                raise ValueError(
                    f"expected a JSON object in {path} at line {line_number}"
                )
            records.append(record)
    return records


def _load_forecasts(
    city_name: str,
    model_name: str,
) -> list[dict[str, Any]]:
    """Load one city's model forecast records from the default data folder."""
    path = DATA_DIR / "forecast" / city_name / model_name / "fc.jsonl"
    forecasts = _load_jsonl(path)
    try:
        forecasts.sort(
            key=lambda forecast: (
                int(forecast["meta"]["last_run_initialisation_time"]),
                int(forecast["meta"]["last_run_availability_time"]),
            )
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid forecast metadata in {path}") from error
    return forecasts


def _load_temperatures(city_name: str) -> list[dict[str, Any]]:
    """Load one city's hourly temperature observations from the data folder."""
    path = DATA_DIR / "temperature" / city_name / "tem.jsonl"
    temperatures = _load_jsonl(path)
    try:
        temperatures.sort(
            key=lambda observation: (
                int(observation["time"]),
                float(observation["update_time"]),
            )
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid temperature observation in {path}") from error
    return temperatures


def group_daily_max_temperature_emos_training_data(
    forecasts: list[dict[str, Any]],
    temperatures: list[dict[str, Any]],
    city_timezone: ZoneInfo,
    *,
    expected_interval_seconds: int = 3600,
    minimum_observation_coverage: float = 1.0,
    minimum_notice_hours: float = 0.0,
    max_day_ahead: int | None = None,
) -> dict[DailyMaxGroupKey, dict[str, Any]]:
    """Build homogeneous daily-maximum EMOS training groups.

    Groups are keyed by ``(initialization_hour_utc, day_ahead)``.  ``day_ahead``
    is the difference between the target local date and the initialization's
    local date.  Only complete forecast days that were available before the
    target local day (plus an optional notice buffer) are retained.

    Forecast temperature arrays are expected on a regular hourly grid by
    default.  Observation records may be more frequent, but their time-slot
    coverage must reach ``minimum_observation_coverage``.  Daily maxima remain
    in the input unit; Celsius-to-Kelvin conversion should happen when the
    resulting groups are converted to R ``ensembleData``.
    """
    if not isinstance(city_timezone, ZoneInfo):
        raise TypeError("city_timezone must be a zoneinfo.ZoneInfo instance")
    if expected_interval_seconds <= 0:
        raise ValueError("expected_interval_seconds must be positive")
    if (
        not isfinite(minimum_observation_coverage)
        or not 0 < minimum_observation_coverage <= 1
    ):
        raise ValueError("minimum_observation_coverage must be in (0, 1]")
    if not isfinite(minimum_notice_hours) or minimum_notice_hours < 0:
        raise ValueError("minimum_notice_hours must be non-negative")
    if max_day_ahead is not None and max_day_ahead < 0:
        raise ValueError("max_day_ahead must be non-negative")
    # {date:[(timestamp,temperature)]}
    observations_by_date = _group_observations_by_local_date(
        temperatures,
        city_timezone,
    )
    notice_seconds = minimum_notice_hours * 3600

    groups: dict[DailyMaxGroupKey, dict[str, Any]] = {}
    expected_member_names: tuple[str, ...] | None = None
    expected_model_name: str | None = None
    model_name_was_set = False
    # A captured forecast version is not an independent training case.  Mark
    # the exact run/day only after a usable complete version has been found so
    # a later snapshot can fill in an incomplete early one.
    seen_cases: set[tuple[int, date]] = set()

    for forecast in _ordered_forecasts(forecasts):
        run = _prepare_forecast_run(forecast, city_timezone)
        if not model_name_was_set:
            expected_model_name = run.model_name
            model_name_was_set = True
        elif run.model_name != expected_model_name:
            raise ValueError("forecast runs contain inconsistent model names")

        member_names = run.member_names
        if expected_member_names is None:
            expected_member_names = member_names
        elif member_names != expected_member_names:
            raise ValueError("forecast runs have inconsistent ensemble members")

        for target_date, day_timestamps in sorted(run.timestamps_by_date.items()):
            case_key = (run.initialization_time, target_date)
            if case_key in seen_cases:
                continue

            case = _make_daily_max_case(
                run,
                target_date,
                day_timestamps,
                observations_by_date,
                city_timezone,
                interval_seconds=expected_interval_seconds,
                minimum_observation_coverage=minimum_observation_coverage,
                notice_seconds=notice_seconds,
                max_day_ahead=max_day_ahead,
            )
            if case is None:
                continue

            seen_cases.add(case_key)
            _append_daily_max_case(
                groups,
                run,
                case,
                city_timezone,
                expected_interval_seconds,
            )

    for group in groups.values():
        _sort_parallel_group_fields(group)
    return groups

#!/usr/bin/env python3
"""Report cities whose temperature JSONL has missing hourly observation slots."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from math import isfinite
from pathlib import Path
from runpy import run_path
from typing import Any, Sequence
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_TEMPERATURE_DIR = PROJECT_DIR / "data" / "temperature"
SECONDS_PER_HOUR = 3600


@dataclass(frozen=True)
class HourlyGap:
    previous_slot: int
    next_slot: int

    @property
    def missing_hours(self) -> int:
        return self.next_slot - self.previous_slot - 1

    @staticmethod
    def _format_slot(slot: int) -> str:
        return datetime.fromtimestamp(
            slot * SECONDS_PER_HOUR,
            timezone.utc,
        ).strftime("%Y-%m-%d %H:00 UTC")

    @property
    def description(self) -> str:
        first_missing = self.previous_slot + 1
        last_missing = self.next_slot - 1
        if first_missing == last_missing:
            missing_range = self._format_slot(first_missing)
        else:
            missing_range = (
                f"{self._format_slot(first_missing)} to "
                f"{self._format_slot(last_missing)}"
            )
        return f"{missing_range} ({self.missing_hours} missing hour(s))"


@dataclass(frozen=True)
class CityContinuityResult:
    city_name: str
    observation_count: int
    covered_hour_count: int
    gaps: tuple[HourlyGap, ...]
    invalid_lines: tuple[int, ...]
    file_error: str | None = None

    @property
    def has_error(self) -> bool:
        return bool(self.file_error or self.invalid_lines or self.gaps)

    @property
    def missing_hour_count(self) -> int:
        return sum(gap.missing_hours for gap in self.gaps)

    @property
    def missing_slots(self) -> tuple[int, ...]:
        return tuple(
            slot
            for gap in self.gaps
            for slot in range(gap.previous_slot + 1, gap.next_slot)
        )


def _valid_observation(value: Any) -> tuple[int, float, float]:
    if not isinstance(value, dict):
        raise ValueError("observation must be a JSON object")
    try:
        raw_time = value["time"]
        raw_temperature = value["temperature"]
        raw_update_time = value["update_time"]
    except KeyError as error:
        raise ValueError(f"missing field: {error.args[0]}") from error

    if isinstance(raw_time, bool):
        raise ValueError("time must be an integer Unix timestamp")
    try:
        timestamp = int(raw_time)
        temperature = float(raw_temperature)
        update_time = float(raw_update_time)
    except (TypeError, ValueError) as error:
        raise ValueError("observation fields have invalid types") from error

    try:
        if float(raw_time) != timestamp:
            raise ValueError("time must be an integer Unix timestamp")
    except (TypeError, ValueError) as error:
        raise ValueError("time must be an integer Unix timestamp") from error
    if not isfinite(temperature) or not isfinite(update_time):
        raise ValueError("temperature and update_time must be finite")
    return timestamp, temperature, update_time


def check_temperature_file(path: Path, city_name: str) -> CityContinuityResult:
    """Check one city's file, allowing multiple observations within an hour."""
    timestamps: list[int] = []
    invalid_lines: list[int] = []
    try:
        with path.open("r", encoding="utf-8") as input_file:
            for line_number, line in enumerate(input_file, 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                    timestamp, _, _ = _valid_observation(value)
                except (json.JSONDecodeError, ValueError):
                    invalid_lines.append(line_number)
                    continue
                timestamps.append(timestamp)
    except OSError as error:
        return CityContinuityResult(
            city_name=city_name,
            observation_count=0,
            covered_hour_count=0,
            gaps=(),
            invalid_lines=(),
            file_error=str(error),
        )

    if not timestamps:
        return CityContinuityResult(
            city_name=city_name,
            observation_count=0,
            covered_hour_count=0,
            gaps=(),
            invalid_lines=tuple(invalid_lines),
            file_error="no valid temperature observations",
        )

    # Training only needs at least one observation in each hourly slot.  METAR
    # stations that report every 30 minutes therefore remain valid.
    covered_slots = sorted({timestamp // SECONDS_PER_HOUR for timestamp in timestamps})
    gaps = tuple(
        HourlyGap(previous_slot, next_slot)
        for previous_slot, next_slot in zip(
            covered_slots,
            covered_slots[1:],
        )
        if next_slot - previous_slot > 1
    )
    return CityContinuityResult(
        city_name=city_name,
        observation_count=len(timestamps),
        covered_hour_count=len(covered_slots),
        gaps=gaps,
        invalid_lines=tuple(invalid_lines),
    )


def check_temperature_directory(
    temperature_dir: str | Path = DEFAULT_TEMPERATURE_DIR,
) -> tuple[CityContinuityResult, ...]:
    """Check every immediate city directory under ``temperature_dir``."""
    root = Path(temperature_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"temperature directory does not exist: {root}")

    city_directories = sorted(
        path for path in root.iterdir() if path.is_dir()
    )
    if not city_directories:
        raise FileNotFoundError(
            f"temperature directory contains no city directories: {root}"
        )

    return tuple(
        check_temperature_file(
            city_directory / "tem.jsonl",
            city_directory.name,
        )
        for city_directory in city_directories
    )


def _configured_city_timezones() -> dict[str, ZoneInfo]:
    """Load city timezones without depending on the caller's working directory."""
    try:
        configuration = run_path(str(PROJECT_DIR / "config.py"))
    except (OSError, RuntimeError, SyntaxError):
        return {}

    raw_cities = configuration.get("CITY")
    if not isinstance(raw_cities, (list, tuple)):
        return {}

    timezones: dict[str, ZoneInfo] = {}
    for city in raw_cities:
        if not isinstance(city, dict):
            continue
        city_name = city.get("name")
        timezone_name = city.get("timezone")
        if not isinstance(city_name, str) or not isinstance(timezone_name, str):
            continue
        try:
            timezones[city_name] = ZoneInfo(timezone_name)
        except ZoneInfoNotFoundError:
            continue
    return timezones


def _missing_hours_by_local_date(
    result: CityContinuityResult,
    city_timezone: ZoneInfo,
) -> dict[str, list[datetime]]:
    missing_by_date: dict[str, list[datetime]] = {}
    for slot in result.missing_slots:
        local_time = datetime.fromtimestamp(
            slot * SECONDS_PER_HOUR,
            city_timezone,
        )
        missing_by_date.setdefault(
            local_time.date().isoformat(),
            [],
        ).append(local_time)
    return missing_by_date


def _format_missing_local_hours(values: list[datetime]) -> str:
    labels = [value.strftime("%H:%M") for value in values]
    duplicate_labels = {
        label for label in labels if labels.count(label) > 1
    }
    return ", ".join(
        (
            value.strftime("%H:%M %z")
            if label in duplicate_labels
            else label
        )
        for value, label in zip(values, labels, strict=True)
    )


def _print_result(
    result: CityContinuityResult,
    *,
    details: bool,
    names_only: bool,
    city_timezone: ZoneInfo,
) -> None:
    if names_only:
        print(result.city_name)
        return

    problems: list[str] = []
    if result.file_error:
        problems.append(result.file_error)
    if result.invalid_lines:
        problems.append(
            f"invalid JSONL line(s): "
            f"{', '.join(str(value) for value in result.invalid_lines)}"
        )
    if result.gaps:
        problems.append(
            f"{result.missing_hour_count} missing hourly slot(s) "
            f"across {len(result.gaps)} gap(s)"
        )
    print(
        f"{result.city_name} [{city_timezone.key}]: "
        f"{'; '.join(problems)}"
    )

    for local_date, missing_times in _missing_hours_by_local_date(
        result,
        city_timezone,
    ).items():
        print(
            f"  {local_date} 缺少小时: "
            f"{_format_missing_local_hours(missing_times)}"
        )

    if details:
        for gap in result.gaps:
            print(f"  UTC gap: {gap.description}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Print cities whose temperature observations do not cover every "
            "hourly time slot."
        )
    )
    parser.add_argument(
        "--temperature-dir",
        type=Path,
        default=DEFAULT_TEMPERATURE_DIR,
        help=(
            "temperature root directory "
            f"(default: {DEFAULT_TEMPERATURE_DIR})"
        ),
    )
    parser.add_argument(
        "--details",
        action="store_true",
        help="print each missing UTC hourly range",
    )
    parser.add_argument(
        "--names-only",
        action="store_true",
        help="print only city directory names",
    )
    arguments = parser.parse_args(argv)

    try:
        results = check_temperature_directory(arguments.temperature_dir)
    except FileNotFoundError as error:
        parser.error(str(error))

    failed = tuple(result for result in results if result.has_error)
    configured_timezones = _configured_city_timezones()
    for result in failed:
        _print_result(
            result,
            details=arguments.details,
            names_only=arguments.names_only,
            city_timezone=configured_timezones.get(
                result.city_name,
                ZoneInfo("UTC"),
            ),
        )

    if not failed and not arguments.names_only:
        print("All temperature series are hourly-continuous.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

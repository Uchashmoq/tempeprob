"""Read-only Bottle dashboard for saved daily maximum-temperature forecasts."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import date, datetime, timezone
import json
import logging
import math
from pathlib import Path
import re
from threading import Lock
from time import monotonic
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import quote
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from bottle import (
    Bottle,
    HTTPError,
    abort,
    request,
    response,
    static_file,
    template,
)

from polymarket import (
    PolymarketAPIError,
    PolymarketDataError,
    TemperatureBucketYesPrice,
    fetch_polymarket_event_by_slug,
    temperature_yes_prices_from_event,
)

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_PREDICTION_DIR = PROJECT_DIR / "prediction" / "highest_temperature_emos"
TEMPLATE_DIR = PROJECT_DIR / "web" / "templates"
STATIC_DIR = PROJECT_DIR / "web" / "static"
DEFAULT_MARKET_PRICE_TIMEOUT_SECONDS = 5.0
DEFAULT_MARKET_PRICE_CACHE_SECONDS = 300.0
DEFAULT_MARKET_PRICE_WORKERS = 6

PREDICTION_RECORD_TYPE = "daily_max_temperature_emos_intervals"
SUPPORTED_SCHEMA_VERSION = 1
_HEX_SHA256 = re.compile(r"[0-9a-f]{64}")
_SAFE_EVENT_SLUG = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_WEEKDAYS_ZH = ("周一", "周二", "周三", "周四", "周五", "周六", "周日")
_MODEL_LABELS = {
    "ecmwf_aifs025_ensemble": "ECMWF AIFS 0.25°",
    "ecmwf_ifs025_ensemble": "ECMWF IFS 0.25°",
}


class PredictionDataError(ValueError):
    """One JSONL record cannot be displayed safely."""


@dataclass(frozen=True)
class DataIssue:
    source: str
    line_number: int | None
    message: str

    @property
    def location(self) -> str:
        if self.line_number is None:
            return self.source
        return f"{self.source}:{self.line_number}"


@dataclass(frozen=True)
class MarketPriceSnapshot:
    """One server-side Polymarket quote snapshot for a temperature event."""

    event_slug: str
    fetched_at_utc: datetime
    prices: tuple[TemperatureBucketYesPrice, ...] = ()
    unavailable_reason: str | None = None

    @property
    def available(self) -> bool:
        return self.unavailable_reason is None and bool(self.prices)

    @property
    def fetched_at_text(self) -> str:
        return self.fetched_at_utc.strftime("%Y-%m-%d %H:%M:%S UTC")


@dataclass(frozen=True)
class IntervalView:
    lower_bound: float | None
    upper_bound: float | None
    label: str
    probability: float
    unit: str

    @staticmethod
    def _format_value(value: float) -> str:
        return f"{value:g}"

    @property
    def display_label(self) -> str:
        unit_symbol = "°C" if self.unit == "celsius" else "K"
        if self.lower_bound is None and self.upper_bound is not None:
            highest_value = self.upper_bound - 1
            return f"{self._format_value(highest_value)} or below"
        if self.upper_bound is None and self.lower_bound is not None:
            return f"{self._format_value(self.lower_bound)} or higher"
        if self.lower_bound is not None and self.upper_bound is not None:
            return f"{self._format_value(self.lower_bound)}{unit_symbol}"
        return self.label

    @property
    def percent_text(self) -> str:
        if self.probability == 0:
            return "0%"
        if self.probability < 0.0001:
            return "<0.01%"
        if self.probability < 0.01:
            return f"{self.probability:.2%}"
        return f"{self.probability:.1%}"

    @property
    def precise_percent_text(self) -> str:
        return f"{self.probability:.8%}"

    @property
    def bar_width(self) -> str:
        if self.probability <= 0:
            return "0"
        return f"{max(self.probability * 100, 0.25):.6f}"


def _probability_text(value: float) -> str:
    if value == 0:
        return "0%"
    if value < 0.0001:
        return "<0.01%"
    if value < 0.01:
        return f"{value:.2%}"
    return f"{value:.1%}"


@dataclass(frozen=True)
class IntervalComparisonView:
    """One EMOS interval aligned with its Polymarket Yes price."""

    interval: IntervalView
    market_yes_price: float | None

    @property
    def market_percent_text(self) -> str:
        if self.market_yes_price is None:
            return "—"
        return _probability_text(self.market_yes_price)

    @property
    def market_precise_percent_text(self) -> str:
        if self.market_yes_price is None:
            return "市场价格暂不可用"
        return f"{self.market_yes_price:.8%}"

    @property
    def difference_percentage_points(self) -> float | None:
        if self.market_yes_price is None:
            return None
        return (self.interval.probability - self.market_yes_price) * 100

    @property
    def difference_text(self) -> str:
        difference = self.difference_percentage_points
        if difference is None:
            return "—"
        if math.isclose(difference, 0.0, abs_tol=0.05):
            return "0.0 pp"
        return f"{difference:+.1f} pp"

    @property
    def difference_class(self) -> str:
        difference = self.difference_percentage_points
        if difference is None or math.isclose(difference, 0.0, abs_tol=0.05):
            return "is-neutral"
        return "is-positive" if difference > 0 else "is-negative"

    @property
    def market_bar_width(self) -> str:
        if self.market_yes_price is None or self.market_yes_price <= 0:
            return "0"
        return f"{max(self.market_yes_price * 100, 0.25):.6f}"


@dataclass(frozen=True)
class MarketComparisonView:
    """Comparison state for one saved model prediction and market event."""

    rows: tuple[IntervalComparisonView, ...]
    fetched_at_text: str | None
    unavailable_reason: str | None = None

    @property
    def available(self) -> bool:
        return self.unavailable_reason is None and all(
            row.market_yes_price is not None for row in self.rows
        )


@dataclass(frozen=True)
class PredictionRecord:
    raw: dict[str, Any]
    source: str
    line_number: int
    revision: int
    prediction_id: str
    city_name: str
    model_name: str
    city_timezone: str
    target_date: date
    generated_at_utc: datetime
    market_boundaries: tuple[float, ...]
    intervals: tuple[IntervalView, ...]

    @property
    def series_key(self) -> tuple[str, str, str, str]:
        return (
            str(self.raw["record_type"]),
            self.city_name,
            self.model_name,
            self.target_date.isoformat(),
        )

    @property
    def short_id(self) -> str:
        return self.prediction_id[:12]

    @property
    def city_label(self) -> str:
        parts = self.city_name.rsplit("-", 1)
        if len(parts) == 2 and re.fullmatch(r"[A-Z0-9]{4}", parts[1]):
            return f"{parts[0].replace('-', ' ')} · {parts[1]}"
        return self.city_name.replace("-", " ")

    @property
    def model_label(self) -> str:
        return _MODEL_LABELS.get(
            self.model_name,
            self.model_name.replace("_", " "),
        )

    @property
    def target_date_label(self) -> str:
        return (
            f"{self.target_date:%Y-%m-%d} "
            f"{_WEEKDAYS_ZH[self.target_date.weekday()]}"
        )

    @property
    def peak_interval(self) -> IntervalView:
        return max(self.intervals, key=lambda interval: interval.probability)

    @property
    def generated_local_text(self) -> str:
        return _format_in_timezone(
            self.generated_at_utc,
            self.city_timezone,
        )

    @property
    def generated_utc_text(self) -> str:
        return self.generated_at_utc.strftime("%Y-%m-%d %H:%M:%S UTC")

    @property
    def market(self) -> Mapping[str, Any]:
        value = self.raw.get("market")
        return value if isinstance(value, Mapping) else {}

    @property
    def market_event_slug(self) -> str | None:
        slug = self.market.get("event_slug")
        if not isinstance(slug, str) or not _SAFE_EVENT_SLUG.fullmatch(slug):
            return None
        return slug

    @property
    def market_url(self) -> str | None:
        slug = self.market_event_slug
        if slug is None:
            return None
        return f"https://polymarket.com/event/{quote(slug, safe='')}"

    @property
    def city_url(self) -> str:
        return f"/city/{quote(self.city_name, safe='')}"

    @property
    def forecast(self) -> Mapping[str, Any]:
        value = self.raw.get("forecast")
        return value if isinstance(value, Mapping) else {}

    @property
    def forecast_initialization_text(self) -> str:
        return _format_record_time(
            self.forecast.get("initialization_time_utc"),
            self.city_timezone,
        )

    @property
    def forecast_availability_text(self) -> str:
        return _format_record_time(
            self.forecast.get("availability_time_utc"),
            self.city_timezone,
        )

    @property
    def forecast_day_ahead(self) -> str:
        value = self.forecast.get("day_ahead")
        return "—" if value is None else str(value)

    @property
    def forecast_members(self) -> tuple[tuple[str, float], ...]:
        names = self.forecast.get("member_names")
        maxima = self.forecast.get("daily_member_maxima")
        if not isinstance(names, list) or not isinstance(maxima, list):
            return ()
        if len(names) != len(maxima):
            return ()
        members = []
        for name, maximum in zip(names, maxima, strict=True):
            if not isinstance(name, str) or isinstance(maximum, bool):
                return ()
            try:
                numeric_maximum = float(maximum)
            except (TypeError, ValueError):
                return ()
            if not math.isfinite(numeric_maximum):
                return ()
            members.append((name, numeric_maximum))
        return tuple(members)

    @property
    def artifact(self) -> Mapping[str, Any]:
        value = self.raw.get("emos_artifact")
        return value if isinstance(value, Mapping) else {}

    @property
    def artifact_version(self) -> str:
        value = self.artifact.get("version")
        return value if isinstance(value, str) else "—"

    @property
    def artifact_short_version(self) -> str:
        version = self.artifact_version
        if version == "—" or len(version) <= 24:
            return version
        return f"{version[:20]}…"

    @property
    def artifact_group(self) -> Mapping[str, Any]:
        value = self.artifact.get("group")
        return value if isinstance(value, Mapping) else {}

    @property
    def correction_parameters(self) -> Mapping[str, Any]:
        value = self.artifact.get("correction_parameters")
        return value if isinstance(value, Mapping) else {}

    @property
    def b_parameters(self) -> tuple[tuple[str, float], ...]:
        values = self.correction_parameters.get("B_by_member")
        if not isinstance(values, Mapping):
            return ()
        result = []
        for name, coefficient in values.items():
            if not isinstance(name, str) or isinstance(coefficient, bool):
                continue
            try:
                numeric_coefficient = float(coefficient)
            except (TypeError, ValueError):
                continue
            if math.isfinite(numeric_coefficient):
                result.append((name, numeric_coefficient))
        return tuple(result)

    @property
    def detail_url(self) -> str:
        return (
            f"/prediction/{quote(self.city_name, safe='')}/"
            f"{quote(self.model_name, safe='')}/"
            f"{self.target_date.isoformat()}/{self.revision}"
        )


@dataclass(frozen=True)
class PredictionCatalog:
    root: Path
    records: tuple[PredictionRecord, ...]
    issues: tuple[DataIssue, ...]
    loaded_at_utc: datetime

    @property
    def latest_records(self) -> tuple[PredictionRecord, ...]:
        latest: dict[tuple[str, str, str, str], PredictionRecord] = {}
        for record in self.records:
            latest[record.series_key] = record
        return tuple(
            sorted(
                latest.values(),
                key=lambda record: (
                    record.target_date,
                    record.city_name,
                    record.model_name,
                ),
            )
        )

    @property
    def cities(self) -> tuple[str, ...]:
        return tuple(sorted({record.city_name for record in self.records}))

    @property
    def models(self) -> tuple[str, ...]:
        return tuple(sorted({record.model_name for record in self.records}))

    @property
    def dates(self) -> tuple[str, ...]:
        return tuple(
            sorted({record.target_date.isoformat() for record in self.records})
        )

    @property
    def loaded_at_text(self) -> str:
        return self.loaded_at_utc.strftime("%Y-%m-%d %H:%M:%S UTC")

    def history_for(
        self,
        record: PredictionRecord,
    ) -> tuple[PredictionRecord, ...]:
        return tuple(
            candidate
            for candidate in self.records
            if candidate.series_key == record.series_key
        )

    def find_revision(
        self,
        city_name: str,
        model_name: str,
        target_date: str,
        revision: int,
    ) -> PredictionRecord | None:
        for record in self.records:
            if (
                record.city_name == city_name
                and record.model_name == model_name
                and record.target_date.isoformat() == target_date
                and record.revision == revision
            ):
                return record
        return None


class PolymarketPriceCache:
    """Fetch and briefly cache server-side market snapshots by event slug."""

    def __init__(
        self,
        *,
        timeout: float = DEFAULT_MARKET_PRICE_TIMEOUT_SECONDS,
        cache_seconds: float = DEFAULT_MARKET_PRICE_CACHE_SECONDS,
        max_workers: int = DEFAULT_MARKET_PRICE_WORKERS,
        event_fetcher: Callable[..., dict[str, Any]] | None = None,
    ) -> None:
        if isinstance(timeout, bool):
            raise ValueError("timeout must be a positive finite number")
        try:
            parsed_timeout = float(timeout)
        except (TypeError, ValueError) as error:
            raise ValueError("timeout must be a positive finite number") from error
        if not math.isfinite(parsed_timeout) or parsed_timeout <= 0:
            raise ValueError("timeout must be a positive finite number")
        if isinstance(cache_seconds, bool):
            raise ValueError("cache_seconds must be a positive finite number")
        try:
            parsed_cache_seconds = float(cache_seconds)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "cache_seconds must be a positive finite number"
            ) from error
        if not math.isfinite(parsed_cache_seconds) or parsed_cache_seconds <= 0:
            raise ValueError("cache_seconds must be a positive finite number")
        if (
            isinstance(max_workers, bool)
            or not isinstance(max_workers, int)
            or max_workers <= 0
        ):
            raise ValueError("max_workers must be a positive integer")
        self.timeout = parsed_timeout
        self.cache_seconds = parsed_cache_seconds
        self.max_workers = max_workers
        self._event_fetcher = (
            fetch_polymarket_event_by_slug if event_fetcher is None else event_fetcher
        )
        self._cache: dict[str, tuple[float, MarketPriceSnapshot]] = {}
        self._lock = Lock()

    def _fetch_snapshot(self, event_slug: str) -> MarketPriceSnapshot:
        try:
            event = self._event_fetcher(event_slug, timeout=self.timeout)
            prices = temperature_yes_prices_from_event(event)
        except PolymarketAPIError as error:
            logging.warning(
                "Could not fetch Polymarket prices for %s: %s",
                event_slug,
                error,
            )
            return MarketPriceSnapshot(
                event_slug=event_slug,
                fetched_at_utc=datetime.now(timezone.utc),
                unavailable_reason="市场价格暂不可用",
            )
        except PolymarketDataError as error:
            logging.warning(
                "Invalid Polymarket prices for %s: %s",
                event_slug,
                error,
            )
            return MarketPriceSnapshot(
                event_slug=event_slug,
                fetched_at_utc=datetime.now(timezone.utc),
                unavailable_reason="市场价格数据无法解析",
            )
        return MarketPriceSnapshot(
            event_slug=event_slug,
            fetched_at_utc=datetime.now(timezone.utc),
            prices=prices,
        )

    def get_many(
        self,
        event_slugs: Sequence[str],
    ) -> dict[str, MarketPriceSnapshot]:
        """Return one snapshot per unique slug, fetching cache misses in parallel."""
        requested = tuple(sorted(set(event_slugs)))
        if not requested:
            return {}

        now = monotonic()
        snapshots: dict[str, MarketPriceSnapshot] = {}
        missing: list[str] = []
        with self._lock:
            for event_slug in requested:
                cached = self._cache.get(event_slug)
                if cached is not None and cached[0] > now:
                    snapshots[event_slug] = cached[1]
                else:
                    missing.append(event_slug)

        if missing:
            worker_count = min(self.max_workers, len(missing))
            if worker_count == 1:
                fetched = {missing[0]: self._fetch_snapshot(missing[0])}
            else:
                with ThreadPoolExecutor(max_workers=worker_count) as executor:
                    fetched = dict(
                        zip(
                            missing,
                            executor.map(self._fetch_snapshot, missing),
                            strict=True,
                        )
                    )
            expires_at = monotonic() + self.cache_seconds
            with self._lock:
                for event_slug, snapshot in fetched.items():
                    self._cache[event_slug] = (expires_at, snapshot)
            snapshots.update(fetched)

        return snapshots


def _require_string(
    value: Any,
    field_name: str,
    *,
    pattern: re.Pattern[str] | None = None,
) -> str:
    if not isinstance(value, str) or not value:
        raise PredictionDataError(f"{field_name} must be a non-empty string")
    if pattern is not None and not pattern.fullmatch(value):
        raise PredictionDataError(f"{field_name} has an invalid format")
    return value


def _parse_iso_datetime(value: Any, field_name: str) -> datetime:
    text = _require_string(value, field_name)
    normalized = f"{text[:-1]}+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as error:
        raise PredictionDataError(
            f"{field_name} must be an ISO-8601 datetime"
        ) from error
    if parsed.tzinfo is None:
        raise PredictionDataError(f"{field_name} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _parse_target_date(value: Any) -> date:
    text = _require_string(value, "target_date_local")
    try:
        parsed = date.fromisoformat(text)
    except ValueError as error:
        raise PredictionDataError("target_date_local must use YYYY-MM-DD") from error
    if parsed.isoformat() != text:
        raise PredictionDataError("target_date_local must use YYYY-MM-DD")
    return parsed


def _parse_intervals(value: Any) -> tuple[IntervalView, ...]:
    if not isinstance(value, list) or not value:
        raise PredictionDataError("intervals must be a non-empty list")
    intervals = []
    probability_sum = 0.0
    for index, raw_interval in enumerate(value):
        if not isinstance(raw_interval, Mapping):
            raise PredictionDataError(f"intervals[{index}] must be an object")
        raw_probability = raw_interval.get("probability")
        if isinstance(raw_probability, bool):
            raise PredictionDataError(f"intervals[{index}].probability must be numeric")
        try:
            probability = float(raw_probability)  # type: ignore
        except (TypeError, ValueError) as error:
            raise PredictionDataError(
                f"intervals[{index}].probability must be numeric"
            ) from error
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise PredictionDataError(
                f"intervals[{index}].probability must be between 0 and 1"
            )

        def optional_bound(field_name: str) -> float | None:
            raw_bound = raw_interval.get(field_name)
            if raw_bound is None:
                return None
            if isinstance(raw_bound, bool):
                raise PredictionDataError(
                    f"intervals[{index}].{field_name} must be numeric"
                )
            try:
                bound = float(raw_bound)
            except (TypeError, ValueError) as error:
                raise PredictionDataError(
                    f"intervals[{index}].{field_name} must be numeric"
                ) from error
            if not math.isfinite(bound):
                raise PredictionDataError(
                    f"intervals[{index}].{field_name} must be finite"
                )
            return bound

        intervals.append(
            IntervalView(
                lower_bound=optional_bound("lower_bound"),
                upper_bound=optional_bound("upper_bound"),
                label=_require_string(
                    raw_interval.get("label"),
                    f"intervals[{index}].label",
                ),
                probability=probability,
                unit=_require_string(
                    raw_interval.get("unit"),
                    f"intervals[{index}].unit",
                ),
            )
        )
        probability_sum += probability
    if not math.isclose(probability_sum, 1.0, abs_tol=1e-6):
        raise PredictionDataError("interval probabilities must sum to 1")
    return tuple(intervals)


def _parse_market_boundaries(value: Any) -> tuple[float, ...]:
    if not isinstance(value, Mapping):
        raise PredictionDataError("market must be an object")
    raw_boundaries = value.get("boundaries")
    if not isinstance(raw_boundaries, list) or not raw_boundaries:
        raise PredictionDataError("market.boundaries must be a non-empty list")
    boundaries = []
    for index, raw_boundary in enumerate(raw_boundaries):
        if isinstance(raw_boundary, bool):
            raise PredictionDataError(f"market.boundaries[{index}] must be numeric")
        try:
            boundary = float(raw_boundary)
        except (TypeError, ValueError) as error:
            raise PredictionDataError(
                f"market.boundaries[{index}] must be numeric"
            ) from error
        if not math.isfinite(boundary):
            raise PredictionDataError(f"market.boundaries[{index}] must be finite")
        boundaries.append(boundary)
    if any(
        current <= previous for previous, current in zip(boundaries, boundaries[1:])
    ):
        raise PredictionDataError("market.boundaries must be strictly increasing")
    return tuple(boundaries)


def _parse_prediction_record(
    value: Any,
    *,
    source: str,
    line_number: int,
    expected_city: str,
    expected_model: str,
) -> PredictionRecord:
    if not isinstance(value, dict):
        raise PredictionDataError("record must be a JSON object")
    if value.get("schema_version") != SUPPORTED_SCHEMA_VERSION:
        raise PredictionDataError(
            f"unsupported schema_version: {value.get('schema_version')!r}"
        )
    if value.get("record_type") != PREDICTION_RECORD_TYPE:
        raise PredictionDataError(
            f"unsupported record_type: {value.get('record_type')!r}"
        )
    city_name = _require_string(value.get("city_name"), "city_name")
    model_name = _require_string(value.get("model_name"), "model_name")
    if city_name != expected_city or model_name != expected_model:
        raise PredictionDataError("record city/model does not match its directory")
    city_timezone = _require_string(
        value.get("city_timezone"),
        "city_timezone",
    )
    try:
        ZoneInfo(city_timezone)
    except (ValueError, ZoneInfoNotFoundError) as error:
        raise PredictionDataError(
            f"unknown city_timezone: {city_timezone!r}"
        ) from error
    return PredictionRecord(
        raw=value,
        source=source,
        line_number=line_number,
        revision=0,
        prediction_id=_require_string(
            value.get("prediction_id"),
            "prediction_id",
            pattern=_HEX_SHA256,
        ),
        city_name=city_name,
        model_name=model_name,
        city_timezone=city_timezone,
        target_date=_parse_target_date(value.get("target_date_local")),
        generated_at_utc=_parse_iso_datetime(
            value.get("prediction_generated_at_utc"),
            "prediction_generated_at_utc",
        ),
        market_boundaries=_parse_market_boundaries(value.get("market")),
        intervals=_parse_intervals(value.get("intervals")),
    )


def load_prediction_catalog(
    prediction_dir: str | Path = DEFAULT_PREDICTION_DIR,
) -> PredictionCatalog:
    """Load valid JSONL records while isolating individual file/line errors."""
    root = Path(prediction_dir)
    loaded_at = datetime.now(timezone.utc)
    if not root.exists():
        return PredictionCatalog(root, (), (), loaded_at)
    try:
        resolved_root = root.resolve(strict=True)
    except OSError as error:
        return PredictionCatalog(
            root,
            (),
            (DataIssue(str(root), None, f"cannot resolve directory: {error}"),),
            loaded_at,
        )
    if not resolved_root.is_dir():
        return PredictionCatalog(
            root,
            (),
            (DataIssue(str(root), None, "prediction path is not a directory"),),
            loaded_at,
        )

    records: list[PredictionRecord] = []
    issues: list[DataIssue] = []
    revision_counts: dict[tuple[str, str, str, str], int] = {}
    candidates = sorted(root.glob("*/*/predictions.jsonl"))
    for path in candidates:
        source = str(path.relative_to(root))
        try:
            resolved_path = path.resolve(strict=True)
        except OSError as error:
            issues.append(DataIssue(source, None, f"cannot resolve file: {error}"))
            continue
        if not resolved_path.is_relative_to(resolved_root):
            issues.append(
                DataIssue(source, None, "file resolves outside prediction root")
            )
            continue
        expected_city = path.parent.parent.name
        expected_model = path.parent.name
        try:
            with resolved_path.open("r", encoding="utf-8") as input_file:
                for line_number, line in enumerate(input_file, start=1):
                    if not line.strip():
                        continue
                    try:
                        value = json.loads(line)
                        record = _parse_prediction_record(
                            value,
                            source=source,
                            line_number=line_number,
                            expected_city=expected_city,
                            expected_model=expected_model,
                        )
                    except (
                        json.JSONDecodeError,
                        PredictionDataError,
                    ) as error:
                        issues.append(DataIssue(source, line_number, str(error)))
                        continue
                    next_revision = revision_counts.get(record.series_key, 0) + 1
                    revision_counts[record.series_key] = next_revision
                    records.append(replace(record, revision=next_revision))
        except (OSError, UnicodeError) as error:
            issues.append(DataIssue(source, None, f"cannot read file: {error}"))
    return PredictionCatalog(
        root=resolved_root,
        records=tuple(records),
        issues=tuple(issues),
        loaded_at_utc=loaded_at,
    )


def _format_in_timezone(value: datetime, timezone_name: str) -> str:
    try:
        target_timezone = ZoneInfo(timezone_name)
    except (ValueError, ZoneInfoNotFoundError):
        target_timezone = timezone.utc
    return value.astimezone(target_timezone).strftime("%Y-%m-%d %H:%M:%S %Z")


def _format_record_time(value: Any, timezone_name: str) -> str:
    try:
        parsed = _parse_iso_datetime(value, "time")
    except PredictionDataError:
        return "—"
    return _format_in_timezone(parsed, timezone_name)


def _city_label(city_name: str) -> str:
    parts = city_name.rsplit("-", 1)
    if len(parts) == 2 and re.fullmatch(r"[A-Z0-9]{4}", parts[1]):
        return f"{parts[0].replace('-', ' ')} · {parts[1]}"
    return city_name.replace("-", " ")


def _model_label(model_name: str) -> str:
    return _MODEL_LABELS.get(
        model_name,
        model_name.replace("_", " "),
    )


def _validated_filter(
    value: str,
    allowed_values: Sequence[str],
) -> str | None:
    return value if value in allowed_values else None


def _market_comparison(
    record: PredictionRecord,
    snapshot: MarketPriceSnapshot | None,
) -> MarketComparisonView:
    unavailable_rows = tuple(
        IntervalComparisonView(interval=interval, market_yes_price=None)
        for interval in record.intervals
    )
    if record.market_event_slug is None:
        return MarketComparisonView(
            rows=unavailable_rows,
            fetched_at_text=None,
            unavailable_reason="预测记录缺少有效的市场标识",
        )
    if snapshot is None:
        return MarketComparisonView(
            rows=unavailable_rows,
            fetched_at_text=None,
            unavailable_reason="市场价格暂不可用",
        )
    if not snapshot.available:
        return MarketComparisonView(
            rows=unavailable_rows,
            fetched_at_text=snapshot.fetched_at_text,
            unavailable_reason=(snapshot.unavailable_reason or "市场价格暂不可用"),
        )

    expected_interval_keys = (
        ((None, record.market_boundaries[0]),)
        + tuple(zip(record.market_boundaries, record.market_boundaries[1:]))
        + ((record.market_boundaries[-1], None),)
    )
    record_interval_keys = tuple(
        (interval.lower_bound, interval.upper_bound)
        for interval in record.intervals
        if interval.unit == "celsius"
    )
    market_interval_keys = tuple(
        (price.lower_bound_celsius, price.upper_bound_celsius)
        for price in snapshot.prices
    )
    if (
        len(record_interval_keys) != len(record.intervals)
        or record_interval_keys != expected_interval_keys
        or market_interval_keys != expected_interval_keys
    ):
        return MarketComparisonView(
            rows=unavailable_rows,
            fetched_at_text=snapshot.fetched_at_text,
            unavailable_reason="市场价格档位与预测档位不一致",
        )

    price_by_interval = {
        interval_key: price.yes_price
        for interval_key, price in zip(
            market_interval_keys,
            snapshot.prices,
            strict=True,
        )
    }
    return MarketComparisonView(
        rows=tuple(
            IntervalComparisonView(
                interval=interval,
                market_yes_price=price_by_interval[
                    (interval.lower_bound, interval.upper_bound)
                ],
            )
            for interval in record.intervals
        ),
        fetched_at_text=snapshot.fetched_at_text,
    )


def _market_snapshots_for_records(
    records: Sequence[PredictionRecord],
    market_price_cache: PolymarketPriceCache,
) -> dict[str, MarketPriceSnapshot]:
    return market_price_cache.get_many(
        tuple(
            event_slug
            for record in records
            if (event_slug := record.market_event_slug) is not None
        )
    )


def _dashboard_groups(
    records: Sequence[PredictionRecord],
    catalog: PredictionCatalog,
    market_snapshots: Mapping[str, MarketPriceSnapshot],
) -> list[dict[str, Any]]:
    history_counts: dict[tuple[str, str, str, str], int] = {}
    for historical_record in catalog.records:
        history_counts[historical_record.series_key] = (
            history_counts.get(historical_record.series_key, 0) + 1
        )

    by_date: dict[date, dict[str, list[PredictionRecord]]] = {}
    for record in records:
        by_date.setdefault(record.target_date, {}).setdefault(
            record.city_name,
            [],
        ).append(record)

    groups = []
    for target_date in sorted(by_date):
        city_groups = []
        for city_name in sorted(by_date[target_date]):
            city_records = sorted(
                by_date[target_date][city_name],
                key=lambda record: record.model_name,
            )
            boundaries = {record.market_boundaries for record in city_records}
            city_groups.append(
                {
                    "city_name": city_name,
                    "city_label": _city_label(city_name),
                    "city_url": f"/city/{quote(city_name, safe='')}",
                    "records": [
                        {
                            "record": record,
                            "history_count": history_counts[record.series_key],
                            "market_comparison": _market_comparison(
                                record,
                                market_snapshots.get(record.market_event_slug or ""),
                            ),
                        }
                        for record in city_records
                    ],
                    "boundary_mismatch": len(boundaries) > 1,
                }
            )
        groups.append(
            {
                "target_date": target_date.isoformat(),
                "target_date_label": (
                    f"{target_date:%Y-%m-%d} " f"{_WEEKDAYS_ZH[target_date.weekday()]}"
                ),
                "cities": city_groups,
            }
        )
    return groups


def _base_context(
    catalog: PredictionCatalog,
    *,
    page_title: str,
    active_city: str | None,
) -> dict[str, Any]:
    return {
        "page_title": page_title,
        "active_city": active_city,
        "site_cities": [
            {
                "name": city,
                "label": _city_label(city),
                "url": f"/city/{quote(city, safe='')}",
            }
            for city in catalog.cities
        ],
        "loaded_at_text": catalog.loaded_at_text,
        "record_count": len(catalog.records),
    }


def _render_dashboard(
    catalog: PredictionCatalog,
    *,
    market_price_cache: PolymarketPriceCache,
    active_city: str | None = None,
) -> str:
    selected_city = active_city or _validated_filter(
        request.query.get("city", ""),  # type: ignore
        catalog.cities,
    )
    selected_model = _validated_filter(
        request.query.get("model", ""),  # type: ignore
        catalog.models,
    )
    selected_date = _validated_filter(
        request.query.get("date", ""),  # type: ignore
        catalog.dates,
    )
    records = list(catalog.latest_records)
    if selected_city is not None:
        records = [record for record in records if record.city_name == selected_city]
    if selected_model is not None:
        records = [record for record in records if record.model_name == selected_model]
    if selected_date is not None:
        records = [
            record
            for record in records
            if record.target_date.isoformat() == selected_date
        ]

    page_title = (
        f"{_city_label(selected_city)}预测"
        if selected_city is not None
        else "最高气温概率总览"
    )
    market_snapshots = _market_snapshots_for_records(
        records,
        market_price_cache,
    )
    return template(
        "dashboard",
        template_lookup=[str(TEMPLATE_DIR)],
        catalog=catalog,
        groups=_dashboard_groups(records, catalog, market_snapshots),
        latest_count=len(catalog.latest_records),
        visible_count=len(records),
        selected_city=selected_city,
        selected_model=selected_model,
        selected_date=selected_date,
        city_options=[(city, _city_label(city)) for city in catalog.cities],
        model_options=[(model, _model_label(model)) for model in catalog.models],
        date_options=catalog.dates,
        **_base_context(
            catalog,
            page_title=page_title,
            active_city=selected_city,
        ),
    )


def create_app(
    prediction_dir: str | Path = DEFAULT_PREDICTION_DIR,
    *,
    market_price_cache: PolymarketPriceCache | None = None,
) -> Bottle:
    """Create a read-only Bottle application for one prediction directory."""
    prediction_root = Path(prediction_dir)
    price_cache = (
        PolymarketPriceCache() if market_price_cache is None else market_price_cache
    )
    bottle_app = Bottle()

    @bottle_app.hook("after_request")
    def add_security_headers() -> None:
        response.set_header("Cache-Control", "no-store")
        response.set_header("X-Content-Type-Options", "nosniff")
        response.set_header("X-Frame-Options", "DENY")
        response.set_header("Referrer-Policy", "no-referrer")
        response.set_header(
            "Permissions-Policy",
            "camera=(), microphone=(), geolocation=()",
        )
        response.set_header(
            "Content-Security-Policy",
            "default-src 'self'; style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; object-src 'none'; base-uri 'none'; "
            "frame-ancestors 'none'; form-action 'self'",
        )

    @bottle_app.get("/")  # type: ignore
    def dashboard() -> str:
        return _render_dashboard(
            load_prediction_catalog(prediction_root),
            market_price_cache=price_cache,
        )

    @bottle_app.get("/city/<city_name>")  # type: ignore
    def city_dashboard(city_name: str) -> str:
        catalog = load_prediction_catalog(prediction_root)
        if city_name not in catalog.cities:
            abort(404, "没有找到该城市的预测数据")
        return _render_dashboard(
            catalog,
            market_price_cache=price_cache,
            active_city=city_name,
        )

    @bottle_app.get(
        "/prediction/<city_name>/<model_name>/<target_date>/<revision:int>"  # type: ignore
    )
    def prediction_detail(
        city_name: str,
        model_name: str,
        target_date: str,
        revision: int,
    ) -> str:
        catalog = load_prediction_catalog(prediction_root)
        record = catalog.find_revision(
            city_name,
            model_name,
            target_date,
            revision,
        )
        if record is None:
            abort(404, "没有找到该预测版本")
        history = tuple(reversed(catalog.history_for(record)))  # type: ignore
        market_snapshots = _market_snapshots_for_records(
            (record,),  # type: ignore
            price_cache,
        )
        market_comparison = _market_comparison(
            record,  # type: ignore
            market_snapshots.get(record.market_event_slug or ""),  # type: ignore
        )
        return template(
            "detail",
            template_lookup=[str(TEMPLATE_DIR)],
            catalog=catalog,
            record=record,
            history=history,
            market_comparison=market_comparison,
            **_base_context(
                catalog,
                page_title=f"{record.city_label} · {record.target_date_label}",  # type: ignore
                active_city=record.city_name,  # type: ignore
            ),
        )

    @bottle_app.get("/assets/app.css")  # type: ignore
    def stylesheet():
        return static_file(
            "app.css",
            root=str(STATIC_DIR),
            mimetype="text/css",  # type: ignore
        )

    @bottle_app.get("/healthz")  # type: ignore
    def health() -> str:
        catalog = load_prediction_catalog(prediction_root)
        response.content_type = "application/json"
        return json.dumps(
            {
                "status": "ok" if not catalog.issues else "degraded",
                "records": len(catalog.records),
                "latest_records": len(catalog.latest_records),
                "issues": len(catalog.issues),
                "loaded_at_utc": catalog.loaded_at_utc.isoformat(),
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )

    @bottle_app.error(404)  # type: ignore
    def not_found(error: HTTPError) -> str:
        catalog = load_prediction_catalog(prediction_root)
        return template(
            "error",
            template_lookup=[str(TEMPLATE_DIR)],
            status_code=404,
            message=str(error.body or "页面不存在"),
            **_base_context(
                catalog,
                page_title="页面不存在",
                active_city=None,
            ),
        )

    return bottle_app


app = create_app()


def _start_background_collection() -> None:
    """Start collectors lazily so normal dashboard imports stay lightweight."""
    from collect import start_collection_threads

    start_collection_threads(daemon=True)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Serve the saved EMOS prediction dashboard.",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="listen address (default: 0.0.0.0, all network interfaces)",
    )
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument(
        "--prediction-dir",
        type=Path,
        default=DEFAULT_PREDICTION_DIR,
    )
    parser.add_argument(
        "--collect",
        action="store_true",
        help=(
            "collect forecast and temperature data in background threads "
            "while serving the dashboard"
        ),
    )
    arguments = parser.parse_args(argv)
    bottle_app = create_app(arguments.prediction_dir)
    if arguments.collect:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        _start_background_collection()
    bottle_app.run(
        host=arguments.host,
        port=arguments.port,
        debug=False,
        reloader=False,
    )


if __name__ == "__main__":
    main()

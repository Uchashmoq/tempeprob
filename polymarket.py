"""Read daily maximum-temperature market boundaries from Polymarket.

Polymarket groups a temperature event into binary markets such as
``33°C or below``, ``34°C`` and ``43°C or higher``.  The EMOS prediction API
uses interval *boundaries* instead.  For that example, the compatible
boundaries are ``(34, 35, ..., 43)``:

* ``T < 34`` maps to ``33°C or below``;
* ``34 <= T < 35`` maps to ``34°C``;
* ``43 <= T`` maps to ``43°C or higher``.
"""

from datetime import date, datetime
import math
import re
from typing import Any, Mapping

import requests

from config import CITY

GAMMA_API_BASE_URL = "https://gamma-api.polymarket.com"
DEFAULT_REQUEST_TIMEOUT_SECONDS = 10.0

_MONTH_SLUGS = (
    "",
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
)
_SLUG_PREFIX_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*\Z")
_TEMPERATURE_BUCKET_PATTERN = re.compile(
    r"""
    \A\s*
    (?P<temperature>[+-]?\d+(?:\.\d+)?)
    \s*°\s*(?P<unit>[CF])
    (?:
        \s+or\s+
        (?P<direction>below|lower|less|under|higher|above|more)
    )?
    \s*\Z
    """,
    re.IGNORECASE | re.VERBOSE,
)
_LOWER_DIRECTIONS = frozenset({"below", "lower", "less", "under"})
_UPPER_DIRECTIONS = frozenset({"higher", "above", "more"})

__all__ = [
    "DEFAULT_REQUEST_TIMEOUT_SECONDS",
    "GAMMA_API_BASE_URL",
    "PolymarketAPIError",
    "PolymarketDataError",
    "build_daily_max_temperature_event_slug",
    "fetch_polymarket_event_by_slug",
    "get_city_daily_max_temperature_boundaries",
    "get_daily_max_temperature_boundaries",
    "temperature_boundaries_from_event",
]


class PolymarketAPIError(RuntimeError):
    """Polymarket could not be reached or returned a non-success response."""


class PolymarketDataError(ValueError):
    """A Polymarket response cannot be mapped safely to EMOS intervals."""


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


def _validate_slug_prefix(value: str) -> str:
    if not isinstance(value, str) or not _SLUG_PREFIX_PATTERN.fullmatch(value):
        raise ValueError(
            "slug_prefix must be a lowercase Polymarket slug without a date"
        )
    if not value.startswith("highest-temperature-in-"):
        raise ValueError(
            "slug_prefix must start with 'highest-temperature-in-'"
        )
    return value


def _validate_timeout(value: float) -> float:
    if isinstance(value, bool):
        raise ValueError("timeout must be a positive finite number")
    try:
        timeout = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("timeout must be a positive finite number") from error
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be a positive finite number")
    return timeout


def build_daily_max_temperature_event_slug(
    slug_prefix: str,
    target_date: date | str,
) -> str:
    """Build the event slug used in Polymarket's ``/event/{slug}`` URL."""
    prefix = _validate_slug_prefix(slug_prefix)
    parsed_date = _parse_target_date(target_date)
    return (
        f"{prefix}-on-{_MONTH_SLUGS[parsed_date.month]}-"
        f"{parsed_date.day}-{parsed_date.year}"
    )


def fetch_polymarket_event_by_slug(
    event_slug: str,
    *,
    timeout: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Fetch one event through Polymarket's public Gamma API."""
    if not isinstance(event_slug, str) or not _SLUG_PREFIX_PATTERN.fullmatch(
        event_slug
    ):
        raise ValueError("event_slug must be a lowercase Polymarket slug")
    request_timeout = _validate_timeout(timeout)
    url = f"{GAMMA_API_BASE_URL}/events/slug/{event_slug}"
    try:
        response = requests.get(
            url,
            headers={"Accept": "application/json"},
            timeout=request_timeout,
        )
        response.raise_for_status()
    except requests.RequestException as error:
        raise PolymarketAPIError(
            f"failed to fetch Polymarket event {event_slug!r}"
        ) from error

    try:
        event = response.json()
    except ValueError as error:
        raise PolymarketDataError(
            f"Polymarket event {event_slug!r} did not return valid JSON"
        ) from error
    if not isinstance(event, dict):
        raise PolymarketDataError(
            f"Polymarket event {event_slug!r} must be a JSON object"
        )
    if event.get("slug") != event_slug:
        raise PolymarketDataError(
            f"Polymarket response slug does not match {event_slug!r}"
        )
    return event


def _parse_temperature_bucket(title: Any) -> tuple[float, str]:
    if not isinstance(title, str):
        raise PolymarketDataError(
            "temperature market is missing a string groupItemTitle"
        )
    match = _TEMPERATURE_BUCKET_PATTERN.fullmatch(title)
    if match is None:
        raise PolymarketDataError(
            f"unsupported Polymarket temperature bucket: {title!r}"
        )

    unit = match.group("unit").upper()
    if unit != "C":
        raise PolymarketDataError(
            "only Celsius Polymarket temperature buckets are supported"
        )
    temperature = float(match.group("temperature"))
    direction = match.group("direction")
    if direction is None:
        return temperature, "exact"
    normalized_direction = direction.lower()
    if normalized_direction in _LOWER_DIRECTIONS:
        return temperature, "lower"
    if normalized_direction in _UPPER_DIRECTIONS:
        return temperature, "upper"
    raise PolymarketDataError(
        f"unsupported Polymarket temperature bucket: {title!r}"
    )


def temperature_boundaries_from_event(
    event: Mapping[str, Any],
) -> tuple[float, ...]:
    """Convert an event's market buckets to increasing Celsius boundaries."""
    if not isinstance(event, Mapping):
        raise TypeError("event must be a mapping")
    markets = event.get("markets")
    if not isinstance(markets, list) or not markets:
        raise PolymarketDataError(
            "Polymarket temperature event must contain a non-empty markets list"
        )

    buckets: list[tuple[float, str]] = []
    for market in markets:
        if not isinstance(market, Mapping):
            raise PolymarketDataError(
                "each Polymarket temperature market must be an object"
            )
        buckets.append(
            _parse_temperature_bucket(market.get("groupItemTitle"))
        )

    if len(set(buckets)) != len(buckets):
        raise PolymarketDataError(
            "Polymarket temperature event contains duplicate buckets"
        )
    lower_values = sorted(
        temperature for temperature, kind in buckets if kind == "lower"
    )
    exact_values = sorted(
        temperature for temperature, kind in buckets if kind == "exact"
    )
    upper_values = sorted(
        temperature for temperature, kind in buckets if kind == "upper"
    )
    if len(lower_values) != 1 or len(upper_values) != 1:
        raise PolymarketDataError(
            "Polymarket temperature event must have exactly one lower and "
            "one upper open-ended bucket"
        )

    ordered_temperatures = [
        lower_values[0],
        *exact_values,
        upper_values[0],
    ]
    for lower, upper in zip(
        ordered_temperatures,
        ordered_temperatures[1:],
    ):
        if not math.isclose(upper - lower, 1.0, abs_tol=1e-9):
            raise PolymarketDataError(
                "Polymarket temperature buckets must be consecutive "
                "one-degree Celsius values"
            )

    # The lower open-ended bucket consumes the first temperature.  Every
    # remaining bucket begins at one boundary accepted by the EMOS predictor.
    return tuple((*exact_values, upper_values[0]))


def get_daily_max_temperature_boundaries(
    slug_prefix: str,
    target_date: date | str,
    *,
    timeout: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
) -> tuple[float, ...]:
    """Fetch Celsius boundaries ready for ``predict_daily_max_temperature_intervals``."""
    event_slug = build_daily_max_temperature_event_slug(
        slug_prefix,
        target_date,
    )
    event = fetch_polymarket_event_by_slug(event_slug, timeout=timeout)
    return temperature_boundaries_from_event(event)


def get_city_daily_max_temperature_boundaries(
    city_name: str,
    target_date: date | str,
    *,
    timeout: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
) -> tuple[float, ...]:
    """Resolve a configured city's ``slug_prefix`` and fetch its boundaries."""
    if not isinstance(city_name, str) or not city_name:
        raise ValueError("city_name must be a non-empty string")
    matching_cities = [
        city
        for city in CITY
        if isinstance(city, Mapping) and city.get("name") == city_name
    ]
    if not matching_cities:
        raise ValueError(f"unknown configured city: {city_name!r}")
    if len(matching_cities) != 1:
        raise ValueError(f"duplicate configured city: {city_name!r}")

    slug_prefix = matching_cities[0].get("slug_prefix")
    if not isinstance(slug_prefix, str):
        raise ValueError(
            f"configured city {city_name!r} has no valid slug_prefix"
        )
    return get_daily_max_temperature_boundaries(
        slug_prefix,
        target_date,
        timeout=timeout,
    )

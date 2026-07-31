"""Read daily maximum-temperature market boundaries from Polymarket.

Polymarket groups a temperature event into binary markets such as
``33°C or below``, ``34°C`` and ``43°C or higher``.  The EMOS prediction API
uses interval *boundaries* instead.  For that example, the compatible
boundaries are ``(34, 35, ..., 43)``:

* ``T < 34`` maps to ``33°C or below``;
* ``34 <= T < 35`` maps to ``34°C``;
* ``43 <= T`` maps to ``43°C or higher``.
"""

from dataclasses import dataclass
from datetime import date, datetime
import json
import math
import re
from typing import Any, Literal, Mapping

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
    "TemperatureBucketYesPrice",
    "build_daily_max_temperature_event_slug",
    "fetch_polymarket_event_by_slug",
    "get_city_daily_max_temperature_boundaries",
    "get_city_daily_max_temperature_yes_prices",
    "get_daily_max_temperature_boundaries",
    "get_daily_max_temperature_yes_prices",
    "temperature_boundaries_from_event",
    "temperature_yes_prices_from_event",
]


class PolymarketAPIError(RuntimeError):
    """Polymarket could not be reached or returned a non-success response."""

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code


class PolymarketDataError(ValueError):
    """A Polymarket response cannot be mapped safely to EMOS intervals."""


TemperatureBucketKind = Literal["lower", "exact", "upper"]


@dataclass(frozen=True)
class TemperatureBucketYesPrice:
    """One temperature bucket and its current Polymarket Yes price."""

    title: str
    temperature_celsius: float
    bucket_kind: TemperatureBucketKind
    yes_price: float
    market_id: str | None
    market_slug: str | None

    @property
    def lower_bound_celsius(self) -> float | None:
        """Inclusive lower bound matching an EMOS interval."""
        if self.bucket_kind == "lower":
            return None
        return self.temperature_celsius

    @property
    def upper_bound_celsius(self) -> float | None:
        """Exclusive upper bound matching an EMOS interval."""
        if self.bucket_kind == "upper":
            return None
        return self.temperature_celsius + 1.0


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
        status_code = (
            None
            if error.response is None
            else error.response.status_code
        )
        raise PolymarketAPIError(
            f"failed to fetch Polymarket event {event_slug!r}",
            status_code=status_code,
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


def _parse_temperature_bucket(
    title: Any,
) -> tuple[float, TemperatureBucketKind]:
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


def _parse_market_array(value: Any, field_name: str) -> list[Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as error:
            raise PolymarketDataError(
                f"temperature market {field_name} is not valid JSON"
            ) from error
    if not isinstance(value, list) or not value:
        raise PolymarketDataError(
            f"temperature market {field_name} must be a non-empty array"
        )
    return value


def _parse_yes_price(market: Mapping[str, Any]) -> float:
    outcomes = _parse_market_array(market.get("outcomes"), "outcomes")
    prices = _parse_market_array(
        market.get("outcomePrices"),
        "outcomePrices",
    )
    if len(outcomes) != len(prices):
        raise PolymarketDataError(
            "temperature market outcomes and outcomePrices must have "
            "the same length"
        )

    normalized_outcomes: list[str] = []
    for outcome in outcomes:
        if not isinstance(outcome, str) or not outcome.strip():
            raise PolymarketDataError(
                "temperature market outcomes must contain non-empty strings"
            )
        normalized_outcomes.append(outcome.strip().casefold())
    if len(set(normalized_outcomes)) != len(normalized_outcomes):
        raise PolymarketDataError(
            "temperature market outcomes must not contain duplicates"
        )
    yes_indices = [
        index
        for index, outcome in enumerate(normalized_outcomes)
        if outcome == "yes"
    ]
    if len(yes_indices) != 1:
        raise PolymarketDataError(
            "temperature market outcomes must contain exactly one Yes"
        )

    parsed_prices: list[float] = []
    for price in prices:
        if isinstance(price, bool):
            raise PolymarketDataError(
                "temperature market outcomePrices must contain numbers "
                "between 0 and 1"
            )
        try:
            parsed_price = float(price)
        except (TypeError, ValueError) as error:
            raise PolymarketDataError(
                "temperature market outcomePrices must contain numbers "
                "between 0 and 1"
            ) from error
        if not math.isfinite(parsed_price) or not 0 <= parsed_price <= 1:
            raise PolymarketDataError(
                "temperature market outcomePrices must contain numbers "
                "between 0 and 1"
            )
        parsed_prices.append(parsed_price)
    return parsed_prices[yes_indices[0]]


def _optional_market_string(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise PolymarketDataError(
            f"temperature market {field_name} must be a non-empty string"
        )
    return value


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


def temperature_yes_prices_from_event(
    event: Mapping[str, Any],
) -> tuple[TemperatureBucketYesPrice, ...]:
    """Return temperature buckets ordered from coldest to warmest.

    Gamma represents ``outcomes`` and ``outcomePrices`` either as JSON-encoded
    strings or as arrays.  Their positions map one-to-one, so the Yes price is
    selected by the outcome name rather than by assuming it is first.
    """
    # Validate the complete bucket geometry before returning any prices.  This
    # guarantees the result aligns one-to-one with the EMOS intervals produced
    # from ``temperature_boundaries_from_event``.
    temperature_boundaries_from_event(event)
    markets = event.get("markets")
    assert isinstance(markets, list)  # validated above

    prices: list[TemperatureBucketYesPrice] = []
    for market in markets:
        assert isinstance(market, Mapping)  # validated above
        title = market.get("groupItemTitle")
        temperature, bucket_kind = _parse_temperature_bucket(title)
        assert isinstance(title, str)  # validated by the parser
        prices.append(
            TemperatureBucketYesPrice(
                title=title,
                temperature_celsius=temperature,
                bucket_kind=bucket_kind,
                yes_price=_parse_yes_price(market),
                market_id=_optional_market_string(market.get("id"), "id"),
                market_slug=_optional_market_string(
                    market.get("slug"),
                    "slug",
                ),
            )
        )
    return tuple(
        sorted(prices, key=lambda price: price.temperature_celsius)
    )


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


def get_daily_max_temperature_yes_prices(
    slug_prefix: str,
    target_date: date | str,
    *,
    timeout: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
) -> tuple[TemperatureBucketYesPrice, ...]:
    """Fetch current Yes prices for one daily temperature event."""
    event_slug = build_daily_max_temperature_event_slug(
        slug_prefix,
        target_date,
    )
    event = fetch_polymarket_event_by_slug(event_slug, timeout=timeout)
    return temperature_yes_prices_from_event(event)


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


def get_city_daily_max_temperature_yes_prices(
    city_name: str,
    target_date: date | str,
    *,
    timeout: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
) -> tuple[TemperatureBucketYesPrice, ...]:
    """Resolve a configured city and fetch its current temperature prices."""
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
    return get_daily_max_temperature_yes_prices(
        slug_prefix,
        target_date,
        timeout=timeout,
    )

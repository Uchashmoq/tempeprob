"""Tests for Polymarket daily maximum-temperature markets."""

import json
import unittest
from datetime import date, datetime, timezone
from unittest.mock import Mock, patch

import requests

from polymarket import (
    GAMMA_API_BASE_URL,
    PolymarketAPIError,
    PolymarketDataError,
    build_daily_max_temperature_event_slug,
    get_city_daily_max_temperature_boundaries,
    get_city_daily_max_temperature_yes_prices,
    get_daily_max_temperature_boundaries,
    get_daily_max_temperature_yes_prices,
    temperature_boundaries_from_event,
    temperature_yes_prices_from_event,
)


SLUG_PREFIX = "highest-temperature-in-chongqing"
EVENT_SLUG = f"{SLUG_PREFIX}-on-july-30-2026"


def make_temperature_event(
    *,
    event_slug: str = EVENT_SLUG,
    titles: tuple[str, ...] = (
        "37°C",
        "38°C",
        "40°C",
        "33°C or below",
        "34°C",
        "35°C",
        "42°C",
        "43°C or higher",
        "36°C",
        "39°C",
        "41°C",
    ),
) -> dict:
    return {
        "id": "759570",
        "slug": event_slug,
        "title": "Highest temperature in Chongqing on July 30?",
        "markets": [
            {
                "id": str(index),
                "groupItemTitle": title,
            }
            for index, title in enumerate(titles)
        ],
    }


def add_yes_prices(
    event: dict,
    *,
    json_encoded: bool = True,
) -> dict[str, float]:
    """Add deterministic Yes/No prices while varying outcome order."""
    expected: dict[str, float] = {}
    for index, market in enumerate(event["markets"]):
        yes_price = (index + 1) / 20
        title = market["groupItemTitle"]
        expected[title] = yes_price
        if index % 2:
            outcomes = ["Yes", "No"]
            prices = [yes_price, 1 - yes_price]
        else:
            outcomes = ["No", "Yes"]
            prices = [1 - yes_price, yes_price]
        market.update(
            {
                "slug": f"temperature-market-{index}",
                "outcomes": (
                    json.dumps(outcomes) if json_encoded else outcomes
                ),
                "outcomePrices": (
                    json.dumps([str(price) for price in prices])
                    if json_encoded
                    else prices
                ),
            }
        )
    return expected


class PolymarketTemperatureBoundariesTest(unittest.TestCase):
    def test_builds_event_slug_without_zero_padding_day(self):
        self.assertEqual(
            build_daily_max_temperature_event_slug(
                SLUG_PREFIX,
                date(2026, 7, 30),
            ),
            EVENT_SLUG,
        )
        self.assertEqual(
            build_daily_max_temperature_event_slug(
                "highest-temperature-in-madrid",
                "2026-05-03",
            ),
            "highest-temperature-in-madrid-on-may-3-2026",
        )

    def test_rejects_invalid_slug_prefix_and_date(self):
        invalid_prefixes = (
            "",
            "Highest-temperature-in-chongqing",
            "highest_temperature_in_chongqing",
            "temperature-in-chongqing",
        )
        for slug_prefix in invalid_prefixes:
            with self.subTest(slug_prefix=slug_prefix):
                with self.assertRaises(ValueError):
                    build_daily_max_temperature_event_slug(
                        slug_prefix,
                        "2026-07-30",
                    )

        with self.assertRaises(TypeError):
            build_daily_max_temperature_event_slug(
                SLUG_PREFIX,
                datetime(2026, 7, 30, tzinfo=timezone.utc),
            )
        with self.assertRaises(ValueError):
            build_daily_max_temperature_event_slug(
                SLUG_PREFIX,
                "2026-7-30",
            )

    def test_extracts_predictor_boundaries_from_unordered_markets(self):
        boundaries = temperature_boundaries_from_event(
            make_temperature_event()
        )

        self.assertEqual(
            boundaries,
            tuple(float(value) for value in range(34, 44)),
        )
        self.assertEqual(boundaries[0], 34.0)
        self.assertEqual(boundaries[-1], 43.0)

    def test_fetches_event_by_slug_and_returns_boundaries(self):
        response = Mock()
        response.json.return_value = make_temperature_event()
        with patch(
            "polymarket.requests.get",
            return_value=response,
        ) as request:
            boundaries = get_daily_max_temperature_boundaries(
                SLUG_PREFIX,
                "2026-07-30",
                timeout=3.5,
            )

        self.assertEqual(
            boundaries,
            tuple(float(value) for value in range(34, 44)),
        )
        request.assert_called_once_with(
            f"{GAMMA_API_BASE_URL}/events/slug/{EVENT_SLUG}",
            headers={"Accept": "application/json"},
            timeout=3.5,
        )
        response.raise_for_status.assert_called_once_with()

    def test_configured_city_resolves_its_slug_prefix(self):
        expected = tuple(float(value) for value in range(34, 44))
        with patch(
            "polymarket.get_daily_max_temperature_boundaries",
            return_value=expected,
        ) as get_boundaries:
            boundaries = get_city_daily_max_temperature_boundaries(
                "Chongqing-ZUCK",
                "2026-07-30",
                timeout=4,
            )

        self.assertEqual(boundaries, expected)
        get_boundaries.assert_called_once_with(
            SLUG_PREFIX,
            "2026-07-30",
            timeout=4.0,
        )

    def test_extracts_ordered_yes_prices_from_json_strings(self):
        event = make_temperature_event()
        expected = add_yes_prices(event)

        prices = temperature_yes_prices_from_event(event)

        self.assertEqual(
            [price.title for price in prices],
            [
                "33°C or below",
                "34°C",
                "35°C",
                "36°C",
                "37°C",
                "38°C",
                "39°C",
                "40°C",
                "41°C",
                "42°C",
                "43°C or higher",
            ],
        )
        self.assertEqual(
            [price.yes_price for price in prices],
            [expected[price.title] for price in prices],
        )
        self.assertEqual(
            (
                prices[0].lower_bound_celsius,
                prices[0].upper_bound_celsius,
            ),
            (None, 34.0),
        )
        self.assertEqual(
            (
                prices[1].lower_bound_celsius,
                prices[1].upper_bound_celsius,
            ),
            (34.0, 35.0),
        )
        self.assertEqual(
            (
                prices[-1].lower_bound_celsius,
                prices[-1].upper_bound_celsius,
            ),
            (43.0, None),
        )
        self.assertEqual(prices[0].market_id, "3")
        self.assertEqual(prices[0].market_slug, "temperature-market-3")

    def test_accepts_outcomes_and_prices_as_lists(self):
        event = make_temperature_event(
            titles=("9°C or below", "10°C or higher"),
        )
        expected = add_yes_prices(event, json_encoded=False)

        prices = temperature_yes_prices_from_event(event)

        self.assertEqual(len(prices), 2)
        self.assertEqual(
            [price.yes_price for price in prices],
            [expected[price.title] for price in prices],
        )

    def test_fetches_yes_prices_and_configured_city_wrapper(self):
        event = make_temperature_event()
        add_yes_prices(event)
        expected = temperature_yes_prices_from_event(event)
        with patch(
            "polymarket.fetch_polymarket_event_by_slug",
            return_value=event,
        ) as fetch_event:
            prices = get_daily_max_temperature_yes_prices(
                SLUG_PREFIX,
                "2026-07-30",
                timeout=2.5,
            )

        self.assertEqual(prices, expected)
        fetch_event.assert_called_once_with(EVENT_SLUG, timeout=2.5)

        with patch(
            "polymarket.get_daily_max_temperature_yes_prices",
            return_value=expected,
        ) as get_prices:
            city_prices = get_city_daily_max_temperature_yes_prices(
                "Chongqing-ZUCK",
                "2026-07-30",
                timeout=4,
            )

        self.assertEqual(city_prices, expected)
        get_prices.assert_called_once_with(
            SLUG_PREFIX,
            "2026-07-30",
            timeout=4.0,
        )

    def test_rejects_invalid_outcomes_and_prices(self):
        invalid_fields = (
            ("outcomes", "not-json"),
            ("outcomes", ["No", "Maybe"]),
            ("outcomes", ["Yes", "YES"]),
            ("outcomes", ["Yes", 1]),
            ("outcomePrices", [0.5]),
            ("outcomePrices", [-0.1, 1.1]),
            ("outcomePrices", [float("nan"), 0.5]),
            ("outcomePrices", [True, 0.5]),
            ("outcomePrices", ["not-a-price", 0.5]),
        )
        for field_name, invalid_value in invalid_fields:
            with self.subTest(field_name=field_name, value=invalid_value):
                event = make_temperature_event(
                    titles=("9°C or below", "10°C or higher"),
                )
                add_yes_prices(event, json_encoded=False)
                event["markets"][0][field_name] = invalid_value
                with self.assertRaises(PolymarketDataError):
                    temperature_yes_prices_from_event(event)

    def test_yes_prices_require_valid_temperature_bucket_geometry(self):
        event = make_temperature_event(
            titles=(
                "33°C or below",
                "35°C",
                "36°C or higher",
            ),
        )
        add_yes_prices(event)

        with self.assertRaisesRegex(
            PolymarketDataError,
            "must be consecutive",
        ):
            temperature_yes_prices_from_event(event)

    def test_rejects_missing_duplicate_or_nonconsecutive_buckets(self):
        invalid_title_sets = (
            (
                "33°C or below",
                "34°C",
                "35°C",
            ),
            (
                "33°C or below",
                "34°C",
                "34°C",
                "35°C or higher",
            ),
            (
                "33°C or below",
                "35°C",
                "36°C or higher",
            ),
            (
                "33°F or below",
                "34°F",
                "35°F or higher",
            ),
        )
        for titles in invalid_title_sets:
            with self.subTest(titles=titles):
                with self.assertRaises(PolymarketDataError):
                    temperature_boundaries_from_event(
                        make_temperature_event(titles=titles)
                    )

    def test_wraps_api_and_json_errors(self):
        with patch(
            "polymarket.requests.get",
            side_effect=requests.Timeout("timed out"),
        ):
            with self.assertRaisesRegex(
                PolymarketAPIError,
                "failed to fetch Polymarket event",
            ):
                get_daily_max_temperature_boundaries(
                    SLUG_PREFIX,
                    "2026-07-30",
                )

        response = Mock()
        response.json.side_effect = ValueError("invalid JSON")
        with patch("polymarket.requests.get", return_value=response):
            with self.assertRaisesRegex(
                PolymarketDataError,
                "did not return valid JSON",
            ):
                get_daily_max_temperature_boundaries(
                    SLUG_PREFIX,
                    "2026-07-30",
                )

    def test_api_error_preserves_http_status_code(self):
        response = Mock(status_code=404)
        response.raise_for_status.side_effect = requests.HTTPError(
            "not found",
            response=response,
        )
        with patch("polymarket.requests.get", return_value=response):
            with self.assertRaises(PolymarketAPIError) as raised:
                get_daily_max_temperature_boundaries(
                    SLUG_PREFIX,
                    "2026-07-30",
                )

        self.assertEqual(raised.exception.status_code, 404)

    def test_rejects_response_for_a_different_event_slug(self):
        response = Mock()
        response.json.return_value = make_temperature_event(
            event_slug="highest-temperature-in-chongqing-on-july-29-2026"
        )
        with patch("polymarket.requests.get", return_value=response):
            with self.assertRaisesRegex(
                PolymarketDataError,
                "response slug does not match",
            ):
                get_daily_max_temperature_boundaries(
                    SLUG_PREFIX,
                    "2026-07-30",
                )


if __name__ == "__main__":
    unittest.main()

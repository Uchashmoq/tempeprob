"""Tests for the read-only Bottle prediction dashboard."""

import json
import unittest
from html.parser import HTMLParser
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch
from urllib.parse import quote
from wsgiref.util import setup_testing_defaults

from web_server import (
    IntervalView,
    PREDICTION_RECORD_TYPE,
    PolymarketPriceCache,
    create_app,
    load_latest_temperature_observations,
    load_prediction_catalog,
    main,
)
from polymarket import PolymarketAPIError


CITY = "City-ZZZZ"
MODEL = "model-a"
TARGET_DATE = "2026-08-01"
EVENT_SLUG = "highest-temperature-in-city-on-august-1-2026"
EVENT_URL = f"https://polymarket.com/event/{EVENT_SLUG}"


def make_record(
    *,
    prediction_id: str = "a" * 64,
    city_name: str = CITY,
    model_name: str = MODEL,
    target_date: str = TARGET_DATE,
    generated_at: str = "2026-07-31T04:00:00.000000Z",
    event_slug: str = EVENT_SLUG,
    probabilities: tuple[float, float] = (0.25, 0.75),
    first_label: str = "T < 30°C",
) -> dict:
    return {
        "schema_version": 1,
        "algorithm_version": 1,
        "record_type": PREDICTION_RECORD_TYPE,
        "prediction_id": prediction_id,
        "city_name": city_name,
        "model_name": model_name,
        "city_timezone": "Asia/Shanghai",
        "target_date_local": target_date,
        "market": {
            "slug_prefix": "highest-temperature-in-city",
            "event_slug": event_slug,
            "boundaries": [30.0],
            "unit": "celsius",
            "fetched_at_utc": "2026-07-31T03:59:00.000000Z",
        },
        "forecast": {
            "initialization_time_utc": "2026-07-30T00:00:00.000000Z",
            "availability_time_utc": "2026-07-30T08:00:00.000000Z",
            "day_ahead": 2,
            "member_names": ["control", "member01"],
            "daily_member_maxima": [31.0, 32.0],
            "input_unit": "celsius",
        },
        "emos_artifact": {
            "version": "artifact-v1",
            "parameter_date": "20260801",
            "group": {
                "sample_count": 20,
                "resolved_training_days": 14,
                "fit_sha256": "b" * 64,
            },
            "correction_parameters": {
                "a": 1.0,
                "B_by_member": {
                    "control": 0.6,
                    "member01": 0.4,
                },
                "c": 2.0,
                "d": 0.5,
            },
        },
        "options": {
            "expected_interval_seconds": 3600,
            "minimum_notice_hours": 0.0,
        },
        "prediction_generated_at_utc": generated_at,
        "forecast_artifact_as_of_utc": (
            "2026-07-31T03:58:00.000000Z"
        ),
        "intervals": [
            {
                "lower_bound": None,
                "upper_bound": 30.0,
                "label": first_label,
                "probability": probabilities[0],
                "unit": "celsius",
            },
            {
                "lower_bound": 30.0,
                "upper_bound": None,
                "label": "30°C <= T",
                "probability": probabilities[1],
                "unit": "celsius",
            },
        ],
    }


def make_market_event(
    *,
    event_slug: str = EVENT_SLUG,
    yes_prices: tuple[float, float] = (0.4, 0.6),
    titles: tuple[str, str] = (
        "29°C or below",
        "30°C or higher",
    ),
) -> dict:
    return {
        "id": "event-1",
        "slug": event_slug,
        "markets": [
            {
                "id": f"market-{index}",
                "slug": f"{event_slug}-{index}",
                "groupItemTitle": title,
                "outcomes": json.dumps(["Yes", "No"]),
                "outcomePrices": json.dumps(
                    [str(yes_price), str(1 - yes_price)]
                ),
            }
            for index, (title, yes_price) in enumerate(
                zip(titles, yes_prices, strict=True)
            )
        ],
    }


def make_market_price_cache(
    *,
    event: dict | None = None,
    side_effect=None,
) -> tuple[PolymarketPriceCache, MagicMock]:
    fetcher = MagicMock(
        return_value=make_market_event() if event is None else event,
        side_effect=side_effect,
    )
    return (
        PolymarketPriceCache(
            event_fetcher=fetcher,
            cache_seconds=300,
        ),
        fetcher,
    )


def write_records(
    root: Path,
    records: list[dict],
    *,
    city_name: str = CITY,
    model_name: str = MODEL,
) -> Path:
    path = root / city_name / model_name / "predictions.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        for record in records:
            json.dump(record, output_file, ensure_ascii=False)
            output_file.write("\n")
    return path


def write_temperatures(
    root: Path,
    records: list[dict],
    *,
    city_name: str = CITY,
) -> Path:
    path = root / city_name / "tem.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        for record in records:
            json.dump(record, output_file, ensure_ascii=False)
            output_file.write("\n")
    return path


def wsgi_get(app, path: str, query: str = ""):
    environ = {}
    setup_testing_defaults(environ)
    environ["REQUEST_METHOD"] = "GET"
    environ["PATH_INFO"] = path
    environ["QUERY_STRING"] = query
    captured = {}

    def start_response(status, headers, exc_info=None):
        captured["status"] = status
        captured["headers"] = dict(headers)

    result = app(environ, start_response)
    try:
        body = b"".join(result).decode("utf-8")
    finally:
        close = getattr(result, "close", None)
        if close is not None:
            close()
    return captured["status"], captured["headers"], body


class _AnchorCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.anchors: list[dict[str, str]] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        if tag == "a":
            self.anchors.append(
                {name: value or "" for name, value in attrs}
            )


def anchors_with_class(html: str, class_name: str) -> list[dict[str, str]]:
    collector = _AnchorCollector()
    collector.feed(html)
    return [
        anchor
        for anchor in collector.anchors
        if class_name in anchor.get("class", "").split()
    ]


class PredictionCatalogTest(unittest.TestCase):
    def test_interval_display_labels_match_market_bucket_format(self):
        intervals = (
            IntervalView(None, 34.0, "T < 34°C", 0.1, "celsius"),
            IntervalView(
                34.0,
                35.0,
                "34°C <= T < 35°C",
                0.7,
                "celsius",
            ),
            IntervalView(43.0, None, "43°C <= T", 0.2, "celsius"),
        )

        self.assertEqual(
            [interval.display_label for interval in intervals],
            ["33 or below", "34°C", "43 or higher"],
        )

    def test_latest_revision_uses_jsonl_append_order(self):
        first = make_record(prediction_id="a" * 64)
        second = make_record(
            prediction_id="b" * 64,
            probabilities=(0.6, 0.4),
            generated_at="2026-07-31T05:00:00.000000Z",
        )
        reverted = make_record(
            prediction_id="a" * 64,
            generated_at="2026-07-31T02:00:00.000000Z",
        )

        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            write_records(root, [first, second, reverted])
            catalog = load_prediction_catalog(root)

        self.assertEqual(len(catalog.records), 3)
        self.assertEqual(
            [record.revision for record in catalog.records],
            [1, 2, 3],
        )
        self.assertEqual(catalog.latest_records[0].revision, 3)
        self.assertEqual(catalog.latest_records[0].prediction_id, "a" * 64)
        self.assertEqual(
            len(catalog.history_for(catalog.latest_records[0])),
            3,
        )

    def test_bad_line_is_reported_without_hiding_valid_record(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            path = write_records(root, [make_record()])
            original = path.read_text(encoding="utf-8")
            path.write_text(f"not-json\n{original}", encoding="utf-8")

            catalog = load_prediction_catalog(root)

        self.assertEqual(len(catalog.records), 1)
        self.assertEqual(len(catalog.issues), 1)
        self.assertEqual(catalog.issues[0].line_number, 1)

    def test_invalid_timezone_and_market_boundaries_are_isolated(self):
        bad_timezone = make_record(prediction_id="b" * 64)
        bad_timezone["city_timezone"] = "../invalid"
        bad_boundaries = make_record(prediction_id="c" * 64)
        bad_boundaries["market"]["boundaries"] = [["not-a-number"]]

        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            write_records(
                root,
                [
                    make_record(),
                    bad_timezone,
                    bad_boundaries,
                ],
            )
            catalog = load_prediction_catalog(root)
            market_price_cache, _ = make_market_price_cache()
            status, _, dashboard = wsgi_get(
                create_app(root, market_price_cache=market_price_cache),
                "/",
            )

        self.assertEqual(len(catalog.records), 1)
        self.assertEqual(len(catalog.issues), 2)
        self.assertTrue(status.startswith("200"))
        self.assertIn("2 条数据未能载入", dashboard)

    def test_symlink_outside_prediction_root_is_rejected(self):
        with TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)
            root = temporary_root / "predictions"
            outside = temporary_root / "outside.jsonl"
            outside.write_text(
                f"{json.dumps(make_record())}\n",
                encoding="utf-8",
            )
            link = root / CITY / MODEL / "predictions.jsonl"
            link.parent.mkdir(parents=True)
            link.symlink_to(outside)

            catalog = load_prediction_catalog(root)

        self.assertFalse(catalog.records)
        self.assertEqual(len(catalog.issues), 1)
        self.assertIn("outside prediction root", catalog.issues[0].message)


class TemperatureObservationTest(unittest.TestCase):
    def test_loader_selects_latest_observation_and_newest_publication(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            path = write_temperatures(
                root,
                [
                    {
                        "temperature": 31,
                        "time": 1785470400,
                        "update_time": 1785470700,
                    },
                    {
                        "temperature": 99,
                        "time": 1785466800,
                        "update_time": 1785474000,
                    },
                    {
                        "temperature": 32.5,
                        "time": 1785470400,
                        "update_time": 1785470760,
                    },
                ],
            )
            with path.open("a", encoding="utf-8") as output_file:
                output_file.write('{"temperature":')

            observations = load_latest_temperature_observations(
                root,
                (CITY,),
            )

        self.assertEqual(set(observations), {CITY})
        observation = observations[CITY]
        self.assertEqual(observation.temperature_celsius, 32.5)
        self.assertEqual(observation.temperature_text, "32.5°C")
        self.assertEqual(observation.line_number, 3)

    def test_dashboard_displays_temperature_and_local_publication_time(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            prediction_root = root / "predictions"
            temperature_root = root / "temperature"
            write_records(prediction_root, [make_record()])
            write_temperatures(
                temperature_root,
                [
                    {
                        "temperature": 32.5,
                        "time": 1785470400,
                        "update_time": 1785470760,
                    }
                ],
            )
            market_price_cache, _ = make_market_price_cache()
            app = create_app(
                prediction_root,
                temperature_dir=temperature_root,
                market_price_cache=market_price_cache,
            )

            status, _, dashboard = wsgi_get(app, "/")
            city_status, _, city_page = wsgi_get(
                app,
                f"/city/{quote(CITY, safe='')}",
            )

        self.assertTrue(status.startswith("200"))
        self.assertTrue(city_status.startswith("200"))
        for page in (dashboard, city_page):
            self.assertIn("最新温度", page)
            self.assertIn("32.5°C", page)
            self.assertIn("发布时间", page)
            self.assertIn("2026-07-31 12:06:00 CST", page)
            self.assertIn("观测时间：2026-07-31 12:00:00 CST", page)


class WebServerRouteTest(unittest.TestCase):
    def test_dashboard_detail_health_and_static_routes(self):
        dangerous_value = "<script>alert('x')</script>"
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            record = make_record(probabilities=(0.8, 0.2))
            record["emos_artifact"]["version"] = dangerous_value
            write_records(
                root,
                [record],
            )
            market_price_cache, fetch_market = make_market_price_cache()
            app = create_app(
                root,
                market_price_cache=market_price_cache,
            )

            status, headers, dashboard = wsgi_get(app, "/")
            city_status, _, city_page = wsgi_get(
                app,
                f"/city/{quote(CITY, safe='')}",
            )
            detail_status, _, detail = wsgi_get(
                app,
                (
                    f"/prediction/{quote(CITY, safe='')}/"
                    f"{quote(MODEL, safe='')}/{TARGET_DATE}/1"
                ),
            )
            health_status, _, health_body = wsgi_get(app, "/healthz")
            css_status, css_headers, css_body = wsgi_get(
                app,
                "/assets/app.css",
            )

        self.assertTrue(status.startswith("200"))
        self.assertIn("最高气温概率总览", dashboard)
        self.assertIn("City · ZZZZ", dashboard)
        self.assertIn("&lt;script&gt;", dashboard)
        self.assertNotIn(dangerous_value, dashboard)
        self.assertIn("29 or below", dashboard)
        self.assertIn("30 or higher", dashboard)
        self.assertIn("Polymarket Yes", dashboard)
        self.assertIn("40.0%", dashboard)
        self.assertIn("+40.0 pp", dashboard)
        city_only_links = anchors_with_class(dashboard, "city-only-link")
        self.assertEqual(len(city_only_links), 1)
        self.assertEqual(city_only_links[0]["href"], f"/city/{CITY}")
        self.assertEqual(headers["X-Frame-Options"], "DENY")
        self.assertIn(
            "default-src 'self'",
            headers["Content-Security-Policy"],
        )

        self.assertTrue(city_status.startswith("200"))
        self.assertIn("City · ZZZZ预测", city_page)
        self.assertTrue(detail_status.startswith("200"))
        self.assertIn("模型概率 vs 市场价格", detail)
        self.assertNotIn("Polymarket 快照", detail)
        self.assertNotIn("market-snapshot-note", detail)
        self.assertIn("预报与修正来源", detail)
        self.assertIn("&lt;script&gt;", detail)
        self.assertNotIn("train/highest_temperature_emos", detail)

        self.assertTrue(health_status.startswith("200"))
        health = json.loads(health_body)
        self.assertEqual(health["status"], "ok")
        self.assertEqual(health["records"], 1)

        self.assertTrue(css_status.startswith("200"))
        self.assertTrue(css_headers["Content-Type"].startswith("text/css"))
        self.assertIn(".prediction-card", css_body)
        self.assertIn(".city-heading-actions", css_body)
        fetch_market.assert_called_once_with(EVENT_SLUG, timeout=5.0)

    def test_dashboard_deduplicates_market_fetch_and_compares_two_models(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            write_records(root, [make_record()])
            second = make_record(
                prediction_id="b" * 64,
                model_name="model-b",
                probabilities=(0.55, 0.45),
            )
            write_records(root, [second], model_name="model-b")
            market_price_cache, fetch_market = make_market_price_cache()
            before = {
                path: path.read_bytes()
                for path in root.glob("*/*/predictions.jsonl")
            }

            status, _, dashboard = wsgi_get(
                create_app(root, market_price_cache=market_price_cache),
                "/",
            )
            after = {
                path: path.read_bytes()
                for path in root.glob("*/*/predictions.jsonl")
            }

        self.assertTrue(status.startswith("200"))
        self.assertIn("模型概率", dashboard)
        self.assertIn("Polymarket Yes", dashboard)
        self.assertIn("-15.0 pp", dashboard)
        self.assertIn("+15.0 pp", dashboard)
        market_links = anchors_with_class(dashboard, "city-market-link")
        self.assertEqual(len(market_links), 1)
        self.assertEqual(market_links[0]["href"], EVENT_URL)
        self.assertEqual(before, after)
        fetch_market.assert_called_once_with(EVENT_SLUG, timeout=5.0)

    def test_city_market_link_uses_selected_target_date(self):
        august_2_slug = "highest-temperature-in-city-on-august-2-2026"
        august_2_url = f"https://polymarket.com/event/{august_2_slug}"
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            write_records(
                root,
                [
                    make_record(),
                    make_record(
                        prediction_id="b" * 64,
                        target_date="2026-08-02",
                        event_slug=august_2_slug,
                    ),
                ],
            )
            market_price_cache, _ = make_market_price_cache(
                event=make_market_event(event_slug=august_2_slug),
            )

            status, _, dashboard = wsgi_get(
                create_app(root, market_price_cache=market_price_cache),
                "/",
                query="city=&date=2026-08-02&model=",
            )

        self.assertTrue(status.startswith("200"))
        market_links = anchors_with_class(dashboard, "city-market-link")
        self.assertEqual(len(market_links), 1)
        self.assertEqual(market_links[0]["href"], august_2_url)
        self.assertEqual(market_links[0]["target"], "_blank")
        self.assertEqual(
            set(market_links[0]["rel"].split()),
            {"noopener", "noreferrer"},
        )
        self.assertIn("2026-08-02", market_links[0]["aria-label"])
        self.assertNotIn(EVENT_URL, dashboard)

    def test_city_market_link_hides_model_slug_conflict_even_when_filtered(self):
        conflicting_slug = (
            "highest-temperature-in-city-conflict-on-august-1-2026"
        )
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            write_records(root, [make_record()])
            write_records(
                root,
                [
                    make_record(
                        prediction_id="b" * 64,
                        model_name="model-b",
                        event_slug=conflicting_slug,
                    )
                ],
                model_name="model-b",
            )
            market_price_cache = MagicMock()
            market_price_cache.get_many.return_value = {}

            status, _, dashboard = wsgi_get(
                create_app(root, market_price_cache=market_price_cache),
                "/",
                query="model=model-a",
            )

        self.assertTrue(status.startswith("200"))
        self.assertIn('<option value="model-a" selected>', dashboard)
        self.assertFalse(anchors_with_class(dashboard, "city-market-link"))

    def test_market_api_failure_keeps_saved_prediction_visible(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            write_records(root, [make_record()])
            market_price_cache, fetch_market = make_market_price_cache(
                side_effect=PolymarketAPIError(
                    "service unavailable",
                    status_code=503,
                )
            )
            app = create_app(
                root,
                market_price_cache=market_price_cache,
            )

            status, _, dashboard = wsgi_get(app, "/")
            detail_status, _, detail = wsgi_get(
                app,
                (
                    f"/prediction/{quote(CITY, safe='')}/"
                    f"{quote(MODEL, safe='')}/{TARGET_DATE}/1"
                ),
            )

        self.assertTrue(status.startswith("200"))
        self.assertTrue(detail_status.startswith("200"))
        self.assertIn("25.0%", dashboard)
        self.assertNotIn("market-snapshot-note", dashboard)
        self.assertNotIn("market-snapshot-note", detail)
        self.assertIn("—", dashboard)
        self.assertIn("—", detail)
        fetch_market.assert_called_once_with(EVENT_SLUG, timeout=5.0)

    def test_market_bucket_mismatch_is_not_compared_by_position(self):
        mismatched_event = make_market_event(
            titles=("28°C or below", "29°C or higher"),
        )
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            write_records(root, [make_record()])
            market_price_cache, _ = make_market_price_cache(
                event=mismatched_event,
            )

            status, _, dashboard = wsgi_get(
                create_app(root, market_price_cache=market_price_cache),
                "/",
            )

        self.assertTrue(status.startswith("200"))
        self.assertNotIn("market-snapshot-note", dashboard)
        self.assertIn("—", dashboard)
        self.assertNotIn("-15.0 pp", dashboard)

    def test_empty_state_and_missing_detail(self):
        with TemporaryDirectory() as temporary_directory:
            app = create_app(temporary_directory)

            status, _, dashboard = wsgi_get(app, "/")
            missing_status, _, missing_page = wsgi_get(
                app,
                "/prediction/none/none/2026-08-01/1",
            )

        self.assertTrue(status.startswith("200"))
        self.assertIn("暂无符合条件的预测", dashboard)
        self.assertTrue(missing_status.startswith("404"))
        self.assertIn("没有找到该预测版本", missing_page)


class WebServerCommandTest(unittest.TestCase):
    def test_collect_option_starts_background_collectors_before_server(self):
        bottle_app = MagicMock()

        with (
            patch(
                "web_server.create_app",
                return_value=bottle_app,
            ) as create,
            patch(
                "web_server._start_background_collection",
            ) as start_collection,
            patch("web_server.logging.basicConfig") as configure_logging,
        ):
            main(
                [
                    "--host",
                    "0.0.0.0",
                    "--port",
                    "9000",
                    "--prediction-dir",
                    "custom-predictions",
                    "--collect",
                ]
            )

        create.assert_called_once_with(Path("custom-predictions"))
        configure_logging.assert_called_once()
        start_collection.assert_called_once_with()
        bottle_app.run.assert_called_once_with(
            host="0.0.0.0",
            port=9000,
            debug=False,
            reloader=False,
        )

    def test_collectors_are_not_started_without_option(self):
        bottle_app = MagicMock()

        with (
            patch(
                "web_server.create_app",
                return_value=bottle_app,
            ),
            patch(
                "web_server._start_background_collection",
            ) as start_collection,
        ):
            main([])

        start_collection.assert_not_called()
        bottle_app.run.assert_called_once_with(
            host="0.0.0.0",
            port=8001,
            debug=False,
            reloader=False,
        )


if __name__ == "__main__":
    unittest.main()

"""Tests for the read-only Bottle prediction dashboard."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import quote
from wsgiref.util import setup_testing_defaults

from web_server import (
    IntervalView,
    PREDICTION_RECORD_TYPE,
    create_app,
    load_prediction_catalog,
)


CITY = "City-ZZZZ"
MODEL = "model-a"
TARGET_DATE = "2026-08-01"


def make_record(
    *,
    prediction_id: str = "a" * 64,
    city_name: str = CITY,
    model_name: str = MODEL,
    target_date: str = TARGET_DATE,
    generated_at: str = "2026-07-31T04:00:00.000000Z",
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
            "event_slug": (
                "highest-temperature-in-city-on-august-1-2026"
            ),
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
            status, _, dashboard = wsgi_get(create_app(root), "/")

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
            app = create_app(root)

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
        self.assertEqual(headers["X-Frame-Options"], "DENY")
        self.assertIn(
            "default-src 'self'",
            headers["Content-Security-Policy"],
        )

        self.assertTrue(city_status.startswith("200"))
        self.assertIn("City · ZZZZ预测", city_page)
        self.assertTrue(detail_status.startswith("200"))
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


if __name__ == "__main__":
    unittest.main()

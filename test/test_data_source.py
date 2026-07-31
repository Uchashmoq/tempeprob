"""Tests for readable upstream forecast errors."""

import unittest
from unittest.mock import MagicMock, patch

import requests

import data_source


class OpenMeteoForecastErrorTest(unittest.TestCase):
    def test_connection_error_records_request_stage(self):
        with (
            patch.object(
                data_source.requests,
                "get",
                side_effect=requests.ConnectionError(
                    "connection reset by peer"
                ),
            ),
            self.assertRaises(data_source.OpenMeteoForecastError) as raised,
        ):
            data_source.ensemble_forcast_meta(
                "ecmwf_aifs025_ensemble"
            )

        self.assertEqual(raised.exception.stage, "forecast metadata")
        self.assertIsNone(raised.exception.status_code)
        self.assertIsNone(raised.exception.response_body)
        self.assertIn("ConnectionError", str(raised.exception))
        self.assertIn("connection reset by peer", str(raised.exception))

    def test_invalid_json_preserves_http_response(self):
        response = MagicMock()
        response.status_code = 200
        response.text = "temporary upstream failure"
        response.raise_for_status.return_value = None
        response.json.side_effect = ValueError("not JSON")

        with (
            patch.object(
                data_source.requests,
                "get",
                return_value=response,
            ),
            self.assertRaises(data_source.OpenMeteoForecastError) as raised,
        ):
            data_source.ensemble_forcast(1.0, 2.0)

        self.assertEqual(raised.exception.stage, "ensemble forecast")
        self.assertEqual(raised.exception.status_code, 200)
        self.assertEqual(
            raised.exception.response_body,
            "temporary upstream failure",
        )
        self.assertEqual(
            str(raised.exception),
            "Open-Meteo returned invalid JSON",
        )

    def test_empty_metadata_is_rejected_before_it_can_be_saved(self):
        response = MagicMock()
        response.status_code = 200
        response.text = "{}"
        response.raise_for_status.return_value = None
        response.json.return_value = {}

        with (
            patch.object(
                data_source.requests,
                "get",
                return_value=response,
            ),
            self.assertRaises(data_source.OpenMeteoForecastError) as raised,
        ):
            data_source.ensemble_forcast_meta(
                "ecmwf_aifs025_ensemble"
            )

        self.assertIn("missing required field", str(raised.exception))
        self.assertEqual(raised.exception.status_code, 200)
        self.assertEqual(raised.exception.response_body, "{}")


if __name__ == "__main__":
    unittest.main()

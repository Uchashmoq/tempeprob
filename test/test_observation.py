"""Tests for temperature collection and error reporting."""

import unittest
from unittest.mock import MagicMock, patch

import data_source
import observation


class AviationWeatherTemperatureTest(unittest.TestCase):
    def test_invalid_json_error_preserves_response_details(self):
        response = MagicMock()
        response.status_code = 200
        response.text = "<html>temporary upstream error</html>"
        response.raise_for_status.return_value = None
        response.json.side_effect = ValueError("not JSON")

        with (
            patch.object(data_source.requests, "get", return_value=response),
            self.assertRaises(
                data_source.AviationWeatherResponseError
            ) as raised,
        ):
            data_source.aviationweather_temp("ZUCK")

        self.assertEqual(raised.exception.status_code, 200)
        self.assertEqual(
            raised.exception.response_body,
            "<html>temporary upstream error</html>",
        )
        self.assertEqual(
            str(raised.exception),
            "Aviation Weather returned invalid JSON",
        )


class TemperatureUpdateLoggingTest(unittest.TestCase):
    def test_first_observation_for_city_is_saved(self):
        city = {"name": "Chongqing-ZUCK", "ICAO": "ZUCK"}
        temperature = {
            "temperature": 31.0,
            "time": 1785477600,
            "update_time": 1785477900,
        }

        with (
            patch.dict(observation.latest_temperature, {}, clear=True),
            patch.object(
                observation.data_source,
                "aviationweather_temp",
                return_value=temperature,
            ),
            patch.object(observation, "save_temperature") as save_temperature,
        ):
            updated = observation.update_temperature(city)

        self.assertTrue(updated)
        save_temperature.assert_called_once_with("Chongqing-ZUCK", temperature)

    def test_response_error_logs_city_and_body_without_traceback(self):
        city = {"name": "Chongqing-ZUCK", "ICAO": "ZUCK"}
        response_error = data_source.AviationWeatherResponseError(
            "Aviation Weather returned invalid JSON",
            status_code=502,
            response_body="<html>Bad Gateway</html>",
        )

        with (
            patch.object(
                observation,
                "update_temperature",
                side_effect=response_error,
            ),
            self.assertLogs(level="ERROR") as error_logs,
        ):
            updated = observation._update_temperature_safely(city)

        output = "\n".join(error_logs.output)
        self.assertFalse(updated)
        self.assertIn("Chongqing-ZUCK (ZUCK)", output)
        self.assertIn("HTTP status=502", output)
        self.assertIn("response_body='<html>Bad Gateway</html>'", output)
        self.assertNotIn("Traceback", output)

    def test_other_error_is_logged_without_traceback(self):
        city = {"name": "Chongqing-ZUCK", "ICAO": "ZUCK"}

        with (
            patch.object(
                observation,
                "update_temperature",
                side_effect=OSError("network unavailable"),
            ),
            self.assertLogs(level="ERROR") as error_logs,
        ):
            updated = observation._update_temperature_safely(city)

        output = "\n".join(error_logs.output)
        self.assertFalse(updated)
        self.assertIn("Chongqing-ZUCK (ZUCK)", output)
        self.assertIn("OSError: network unavailable", output)
        self.assertNotIn("Traceback", output)


if __name__ == "__main__":
    unittest.main()

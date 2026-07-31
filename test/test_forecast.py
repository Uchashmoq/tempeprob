"""Tests for periodic forecast updates and triggered EMOS training."""

import unittest
from pathlib import Path
from unittest.mock import call, patch

import forecast


class ForecastPeriodicUpdateTest(unittest.TestCase):
    def setUp(self):
        self.cities = [
            {
                "name": "City-One",
                "lat": 1.0,
                "lon": 2.0,
                "models": [
                    {"name": "model-a"},
                    {"name": "model-b"},
                ],
            }
        ]

    def test_trains_only_forecast_that_was_updated(self):
        latest = {
            "City-One/model-a": {
                "meta": {"last_run_initialisation_time": 100}
            },
            "City-One/model-b": {
                "meta": {
                    "last_run_initialisation_time": 200,
                    "last_run_availability_time": 300,
                }
            },
        }
        version_path = Path(
            "train/highest_temperature_emos/City-One/model-b/versions/v1"
        )

        with (
            patch.object(forecast.config, "CITY", self.cities),
            patch.object(forecast.config, "AUTO_TRAIN", True),
            patch.dict(forecast.latest_forecast, latest, clear=True),
            patch.object(
                forecast,
                "update_forecast",
                side_effect=[False, True],
            ) as updater,
            patch.object(
                forecast.train_emos_max_temperature,
                "train_daily_max_temperature_emos_for_city_model",
                return_value=version_path,
            ) as trainer,
            self.assertLogs(level="INFO") as info_logs,
        ):
            forecast._update_forecasts_once()

        self.assertEqual(
            updater.call_args_list,
            [
                call(self.cities[0], "model-a"),
                call(self.cities[0], "model-b"),
            ],
        )
        trainer.assert_called_once_with(
            "City-One",
            "model-b",
            extra_metadata={
                "trigger": "forecast_update",
                "forecast_meta": {
                    "last_run_initialisation_time": 200,
                    "last_run_availability_time": 300,
                },
            },
        )
        self.assertIn(str(version_path), "\n".join(info_logs.output))

    def test_training_failure_does_not_stop_later_models(self):
        with (
            patch.object(forecast.config, "CITY", self.cities),
            patch.object(forecast.config, "AUTO_TRAIN", True),
            patch.dict(forecast.latest_forecast, {}, clear=True),
            patch.object(
                forecast,
                "update_forecast",
                return_value=True,
            ),
            patch.object(
                forecast.train_emos_max_temperature,
                "train_daily_max_temperature_emos_for_city_model",
                side_effect=[RuntimeError("R optimizer failed"), None],
            ) as trainer,
            self.assertLogs(level="ERROR") as error_logs,
        ):
            forecast._update_forecasts_once()

        self.assertEqual(
            [training_call.args[:2] for training_call in trainer.call_args_list],
            [("City-One", "model-a"), ("City-One", "model-b")],
        )
        self.assertIn(
            "Forecast updated but failed to train daily-max EMOS "
            "for City-One/model-a",
            "\n".join(error_logs.output),
        )

    def test_update_failure_does_not_train_or_stop_later_models(self):
        with (
            patch.object(forecast.config, "CITY", self.cities),
            patch.object(forecast.config, "AUTO_TRAIN", True),
            patch.dict(forecast.latest_forecast, {}, clear=True),
            patch.object(
                forecast,
                "update_forecast",
                side_effect=[RuntimeError("download failed"), True],
            ),
            patch.object(
                forecast.train_emos_max_temperature,
                "train_daily_max_temperature_emos_for_city_model",
                return_value=None,
            ) as trainer,
            self.assertLogs(level="ERROR") as error_logs,
        ):
            forecast._update_forecasts_once()

        trainer.assert_called_once()
        self.assertEqual(trainer.call_args.args[:2], ("City-One", "model-b"))
        self.assertIn(
            "Failed to update forecast for City-One/model-a",
            "\n".join(error_logs.output),
        )

    def test_does_not_train_when_auto_train_is_disabled(self):
        with (
            patch.object(forecast.config, "CITY", self.cities),
            patch.object(forecast.config, "AUTO_TRAIN", False),
            patch.dict(forecast.latest_forecast, {}, clear=True),
            patch.object(
                forecast,
                "update_forecast",
                return_value=True,
            ) as updater,
            patch.object(
                forecast.train_emos_max_temperature,
                "train_daily_max_temperature_emos_for_city_model",
            ) as trainer,
        ):
            forecast._update_forecasts_once()

        self.assertEqual(updater.call_count, 2)
        trainer.assert_not_called()

    def test_periodic_entry_initializes_cache_before_first_update(self):
        latest = {"meta": {"last_run_initialisation_time": 100}}

        class StopPeriodicLoop(Exception):
            pass

        with (
            patch.object(
                forecast.config,
                "CITY",
                [self.cities[0] | {"models": [{"name": "model-a"}]}],
            ),
            patch.object(
                forecast.config,
                "UPDATE_FORECAST_INTERVAL",
                123,
            ),
            patch.dict(forecast.latest_forecast, {}, clear=True),
            patch.object(
                forecast,
                "get_latest_forecast",
                return_value=latest,
            ) as loader,
            patch.object(forecast, "_update_forecasts_once") as update_once,
            patch.object(
                forecast.time,
                "sleep",
                side_effect=StopPeriodicLoop,
            ) as sleeper,
        ):
            with self.assertRaises(StopPeriodicLoop):
                forecast.update_forecast_periotically()

            self.assertEqual(
                forecast.latest_forecast,
                {"City-One/model-a": latest},
            )

        loader.assert_called_once_with("City-One", "model-a")
        update_once.assert_called_once_with()
        sleeper.assert_called_once_with(123)


if __name__ == "__main__":
    unittest.main()

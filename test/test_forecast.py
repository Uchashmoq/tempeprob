"""Tests for forecast updates and triggered EMOS training and prediction."""

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import data_source
import forecast


class _InlineFuture:
    def __init__(self, result=None, exception: Exception | None = None):
        self._result = result
        self._exception = exception

    def result(self):
        if self._exception is not None:
            raise self._exception
        return self._result

    def add_done_callback(self, callback):
        callback(self)


class _InlineExecutor:
    def submit(self, function, *args, **kwargs):
        try:
            result = function(*args, **kwargs)
        except Exception as error:
            return _InlineFuture(exception=error)
        return _InlineFuture(result=result)


class ForecastPeriodicUpdateTest(unittest.TestCase):
    def setUp(self):
        executor_patcher = patch.object(
            forecast,
            "_FORECAST_POSTPROCESS_EXECUTOR",
            _InlineExecutor(),
        )
        executor_patcher.start()
        self.addCleanup(executor_patcher.stop)
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

    def test_postprocessing_trains_before_predicting(self):
        events = []
        city = self.cities[0]
        model = city["models"][0]

        with (
            patch.object(
                forecast,
                "_train_updated_forecast",
                side_effect=lambda *_: events.append("train"),
            ),
            patch.object(
                forecast,
                "_predict_updated_forecast",
                side_effect=lambda *_: events.append("predict"),
            ),
        ):
            forecast._postprocess_updated_forecast(
                city,
                model,
                auto_train=True,
                auto_predict=True,
            )

        self.assertEqual(events, ["train", "predict"])

    def test_scheduler_queues_work_without_running_it_inline(self):
        executor = Mock()
        future = Mock()
        executor.submit.return_value = future
        city = self.cities[0]
        model = city["models"][0]

        with patch.object(
            forecast,
            "_FORECAST_POSTPROCESS_EXECUTOR",
            executor,
        ):
            result = forecast._schedule_updated_forecast_postprocessing(
                city,
                model,
                auto_train=True,
                auto_predict=True,
            )

        self.assertIs(result, future)
        executor.submit.assert_called_once()
        submitted = executor.submit.call_args
        self.assertIs(
            submitted.args[0],
            forecast._postprocess_updated_forecast,
        )
        self.assertIsNot(submitted.args[1], city)
        self.assertIsNot(submitted.args[2], model)
        self.assertEqual(submitted.kwargs["auto_train"], True)
        self.assertEqual(submitted.kwargs["auto_predict"], True)
        future.add_done_callback.assert_called_once()

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
            patch.object(forecast.config, "AUTO_PREDICT", False),
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
            patch.object(forecast.config, "AUTO_PREDICT", True),
            patch.object(forecast.config, "PREDICT_DAYS", 2),
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
            patch.object(
                forecast.predict_emos_max_temperature,
                "predict_all_configured_daily_max_temperature_intervals",
                return_value=(),
            ) as predictor,
            self.assertLogs(level="ERROR") as error_logs,
        ):
            forecast._update_forecasts_once()

        self.assertEqual(
            [training_call.args[:2] for training_call in trainer.call_args_list],
            [("City-One", "model-a"), ("City-One", "model-b")],
        )
        self.assertEqual(predictor.call_count, 2)
        self.assertIn(
            "Forecast updated but failed to train daily-max EMOS "
            "for City-One/model-a",
            "\n".join(error_logs.output),
        )

    def test_update_failure_does_not_train_or_stop_later_models(self):
        with (
            patch.object(forecast.config, "CITY", self.cities),
            patch.object(forecast.config, "AUTO_TRAIN", True),
            patch.object(forecast.config, "AUTO_PREDICT", False),
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
            "Failed to update forecast for City-One/model-a: "
            "RuntimeError: download failed",
            "\n".join(error_logs.output),
        )
        self.assertNotIn("Traceback", "\n".join(error_logs.output))

    def test_open_meteo_error_logs_readable_response_without_traceback(self):
        error = data_source.OpenMeteoForecastError(
            "Open-Meteo returned invalid JSON",
            stage="forecast metadata",
            status_code=502,
            response_body="<html>Bad Gateway</html>",
        )

        with (
            patch.object(forecast.config, "CITY", self.cities),
            patch.object(forecast.config, "AUTO_TRAIN", False),
            patch.object(forecast.config, "AUTO_PREDICT", False),
            patch.object(
                forecast,
                "update_forecast",
                side_effect=[error, False],
            ),
            self.assertLogs(level="ERROR") as error_logs,
        ):
            forecast._update_forecasts_once()

        output = "\n".join(error_logs.output)
        self.assertIn("City-One/model-a", output)
        self.assertIn("stage=forecast metadata", output)
        self.assertIn("reason=Open-Meteo returned invalid JSON", output)
        self.assertIn("HTTP status=502", output)
        self.assertIn("response_body='<html>Bad Gateway</html>'", output)
        self.assertNotIn("Traceback", output)

    def test_connection_error_log_marks_response_as_unavailable(self):
        error = data_source.OpenMeteoForecastError(
            "ConnectionError: connection reset by peer",
            stage="forecast metadata",
        )

        with self.assertLogs(level="ERROR") as error_logs:
            forecast._log_forecast_update_failure(
                "Wuhan-ZHHH",
                "ecmwf_aifs025_ensemble",
                error,
            )

        output = "\n".join(error_logs.output)
        self.assertIn(
            "Wuhan-ZHHH/ecmwf_aifs025_ensemble",
            output,
        )
        self.assertIn("ConnectionError: connection reset by peer", output)
        self.assertIn("HTTP status=unavailable", output)
        self.assertIn("response_body=<unavailable>", output)
        self.assertNotIn("Traceback", output)

    def test_does_not_train_when_auto_train_is_disabled(self):
        with (
            patch.object(forecast.config, "CITY", self.cities),
            patch.object(forecast.config, "AUTO_TRAIN", False),
            patch.object(forecast.config, "AUTO_PREDICT", False),
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
            patch.object(
                forecast.predict_emos_max_temperature,
                "predict_all_configured_daily_max_temperature_intervals",
            ) as predictor,
        ):
            forecast._update_forecasts_once()

        self.assertEqual(updater.call_count, 2)
        trainer.assert_not_called()
        predictor.assert_not_called()

    def test_predicts_only_forecast_that_was_updated(self):
        with (
            patch.object(forecast.config, "CITY", self.cities),
            patch.object(forecast.config, "AUTO_TRAIN", False),
            patch.object(forecast.config, "AUTO_PREDICT", True),
            patch.object(forecast.config, "PREDICT_DAYS", 3),
            patch.dict(forecast.latest_forecast, {}, clear=True),
            patch.object(
                forecast,
                "update_forecast",
                side_effect=[False, True],
            ),
            patch.object(
                forecast.predict_emos_max_temperature,
                "predict_all_configured_daily_max_temperature_intervals",
                return_value=(
                    SimpleNamespace(appended=True),
                    SimpleNamespace(appended=False),
                ),
            ) as predictor,
            self.assertLogs(level="INFO") as info_logs,
        ):
            forecast._update_forecasts_once()

        predictor.assert_called_once_with(
            cities=[
                {
                    **self.cities[0],
                    "models": [{"name": "model-b"}],
                }
            ],
            predict_days=3,
        )
        self.assertIn(
            "1 appended, 1 unchanged",
            "\n".join(info_logs.output),
        )

    def test_prediction_failure_does_not_stop_later_models(self):
        with (
            patch.object(forecast.config, "CITY", self.cities),
            patch.object(forecast.config, "AUTO_TRAIN", False),
            patch.object(forecast.config, "AUTO_PREDICT", True),
            patch.object(forecast.config, "PREDICT_DAYS", 2),
            patch.dict(forecast.latest_forecast, {}, clear=True),
            patch.object(
                forecast,
                "update_forecast",
                return_value=True,
            ),
            patch.object(
                forecast.predict_emos_max_temperature,
                "predict_all_configured_daily_max_temperature_intervals",
                side_effect=[RuntimeError("prediction failed"), ()],
            ) as predictor,
            self.assertLogs(level="ERROR") as error_logs,
        ):
            forecast._update_forecasts_once()

        self.assertEqual(predictor.call_count, 2)
        self.assertIn(
            "Forecast updated but failed to predict daily-max temperature "
            "for City-One/model-a",
            "\n".join(error_logs.output),
        )

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

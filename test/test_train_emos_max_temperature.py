"""Tests for daily maximum-temperature EMOS case preparation."""

import json
import math
import unittest
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import call, patch
from zoneinfo import ZoneInfo

from train_emos_max_temperature import (
    DEFAULT_MINIMUM_TRAINING_DAYS,
    _load_forecasts,
    _load_temperatures,
    build_daily_max_temperature_ensemble_data,
    group_daily_max_temperature_emos_training_data,
    load_daily_max_temperature_emos_fits,
    save_daily_max_temperature_emos_fits,
    train_all_daily_max_temperature_emos,
    train_daily_max_temperature_emos,
    train_daily_max_temperature_emos_for_city_model,
)

TEST_DATA_DIR = Path(__file__).resolve().parents[1] / "test-data" / "data2"


def local_day_timestamps(target_date: date, zone: ZoneInfo) -> list[int]:
    """Return the UTC hourly grid contained in one local calendar day."""
    start = datetime.combine(target_date, time.min, tzinfo=zone)
    end = datetime.combine(target_date + timedelta(days=1), time.min, tzinfo=zone)
    return list(range(int(start.timestamp()), int(end.timestamp()), 3600))


def make_forecast(
    initialization_time: int,
    availability_time: int,
    timestamps: list[int],
    *,
    base: float = 0.0,
) -> dict:
    return {
        "model": "test_ensemble",
        "time": timestamps,
        "temperature_2m": [base + index for index in range(len(timestamps))],
        "temperature_2m_member01": [
            base + 100.0 + index for index in range(len(timestamps))
        ],
        "meta": {
            "last_run_initialisation_time": initialization_time,
            "last_run_availability_time": availability_time,
        },
    }


def make_observations(timestamps: list[int], *, base: float = 10.0) -> list[dict]:
    return [
        {
            "time": timestamp,
            "temperature": base + index,
            "update_time": timestamp + 300,
        }
        for index, timestamp in enumerate(timestamps)
    ]


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as jsonl_file:
        return [json.loads(line) for line in jsonl_file if line.strip()]


def make_daily_max_group(
    number_of_days: int,
    *,
    initialization_time: str = "00",
    day_ahead: int = 1,
) -> dict:
    """Build deterministic grouped daily maxima for EMOS bridge tests."""
    target_start = date(2026, 1, 1)
    forecasts = []
    observations = []
    for index in range(number_of_days):
        temperature = 20.0 + 0.12 * index + 1.7 * math.sin(index / 3)
        forecasts.append(
            (
                temperature - 0.8 + 0.2 * math.cos(index),
                temperature + 0.6,
                temperature + 0.15 * math.sin(index / 2),
            )
        )
        observations.append(temperature + 0.3 * math.cos(index * 1.4))

    return {
        "initialization_time": initialization_time,
        "initialization_hour": initialization_time,
        "day_ahead": day_ahead,
        "member_names": (
            "temperature_2m",
            "temperature_2m_member01",
            "temperature_2m_member02",
        ),
        "target_dates": [
            (target_start + timedelta(days=index)).isoformat()
            for index in range(number_of_days)
        ],
        "forecasts": forecasts,
        "observations": observations,
    }


class DailyMaxTemperatureGroupingTest(unittest.TestCase):
    def test_build_daily_max_ensemble_data_converts_celsius_to_kelvin(self):
        group = {
            "initialization_time": "06",
            "day_ahead": 2,
            "member_names": (
                "temperature_2m",
                "temperature_2m_member01",
            ),
            "target_dates": ["2026-01-01"],
            "forecasts": [(0.0, 1.0)],
            "observations": [0.5],
        }

        ensemble_data = build_daily_max_temperature_ensemble_data(group)

        from rpy2.robjects.packages import importr

        ensemble_bma = importr("ensembleBMA")
        self.assertEqual(
            list(ensemble_bma.ensembleForecasts(ensemble_data)),
            [273.15, 274.15],
        )
        self.assertEqual(
            list(ensemble_bma.dataVerifObs(ensemble_data)),
            [273.65],
        )
        self.assertEqual(
            list(ensemble_bma.ensembleValidDates(ensemble_data)),
            ["20260101"],
        )
        self.assertEqual(list(ensemble_bma.ensembleFhour(ensemble_data)), [48])
        self.assertEqual(list(ensemble_bma.ensembleItime(ensemble_data)), ["06"])
        self.assertEqual(
            list(ensemble_bma.ensembleGroups(ensemble_data)),
            ["control", "perturbed"],
        )

    def test_train_daily_max_emos_adapts_window_per_group(self):
        groups = {
            ("00", 1): make_daily_max_group(
                DEFAULT_MINIMUM_TRAINING_DAYS
            ),
            ("06", 2): make_daily_max_group(
                DEFAULT_MINIMUM_TRAINING_DAYS + 1,
                initialization_time="06",
                day_ahead=2,
            ),
        }

        with (
            patch(
                "train_emos_max_temperature."
                "build_daily_max_temperature_ensemble_data",
                side_effect=("data00", "data06"),
            ),
            patch(
                "train_emos_max_temperature._call_daily_max_ensemble_mos",
                side_effect=("fit00", "fit06"),
            ) as fit,
        ):
            fits = train_daily_max_temperature_emos(groups)

        self.assertEqual(fits, {("00", 1): "fit00", ("06", 2): "fit06"})
        self.assertEqual(
            [call.args[1] for call in fit.call_args_list],
            [
                DEFAULT_MINIMUM_TRAINING_DAYS,
                DEFAULT_MINIMUM_TRAINING_DAYS + 1,
            ],
        )
        self.assertTrue(
            all(call.kwargs["consecutive"] is False for call in fit.call_args_list)
        )

    def test_train_daily_max_emos_skips_insufficient_groups_by_default(self):
        insufficient_count = max(1, DEFAULT_MINIMUM_TRAINING_DAYS - 2)
        sufficient_count = DEFAULT_MINIMUM_TRAINING_DAYS + 1
        groups = {
            ("00", 1): make_daily_max_group(insufficient_count),
            ("06", 2): make_daily_max_group(
                sufficient_count,
                initialization_time="06",
                day_ahead=2,
            ),
        }

        with (
            patch(
                "train_emos_max_temperature."
                "build_daily_max_temperature_ensemble_data",
                return_value="data06",
            ),
            patch(
                "train_emos_max_temperature._call_daily_max_ensemble_mos",
                return_value="fit06",
            ) as fit,
            self.assertLogs(
                "train_emos_max_temperature",
                level="WARNING",
            ) as warning_logs,
        ):
            fits = train_daily_max_temperature_emos(groups)

        self.assertEqual(fits, {("06", 2): "fit06"})
        fit.assert_called_once()
        self.assertEqual(fit.call_args.args[1], sufficient_count)
        warning_text = "\n".join(warning_logs.output)
        self.assertIn("('00', 1)", warning_text)
        self.assertIn(
            f"only {insufficient_count} training date(s)",
            warning_text,
        )
        self.assertIn(
            f"need at least {DEFAULT_MINIMUM_TRAINING_DAYS}",
            warning_text,
        )

        with self.assertRaisesRegex(
            ValueError,
            f"need at least {DEFAULT_MINIMUM_TRAINING_DAYS} dates",
        ):
            train_daily_max_temperature_emos(
                groups,
                skip_insufficient=False,
            )

    def test_train_daily_max_emos_runs_gaussian_r_fit(self):
        from rpy2 import robjects

        groups = {("00", 1): make_daily_max_group(20)}

        fits = train_daily_max_temperature_emos(groups, training_days=14)

        fit = fits[("00", 1)]
        self.assertIn("ensembleMOSnormal", set(robjects.r["class"](fit)))  # type: ignore
        self.assertTrue({"training", "a", "B", "c", "d"}.issubset(fit.names))
        for parameter_name in ("a", "B", "c", "d"):
            self.assertTrue(
                all(math.isfinite(float(value)) for value in fit.rx2(parameter_name))
            )

        storage_groups = {
            **groups,
            ("06", 2): make_daily_max_group(
                5,
                initialization_time="06",
                day_ahead=2,
            ),
        }
        completed_at = datetime(2026, 7, 19, 1, 2, 3, tzinfo=timezone.utc)
        with TemporaryDirectory() as output_dir:
            first_path = save_daily_max_temperature_emos_fits(
                fits,
                storage_groups,
                "Chongqing-ZUCK",
                "test_ensemble",
                training_days=14,
                training_completed_at=completed_at,
                extra_metadata={"source": "unit-test"},
                output_dir=output_dir,
            )
            second_path = save_daily_max_temperature_emos_fits(
                fits,
                storage_groups,
                "Chongqing-ZUCK",
                "test_ensemble",
                training_days=14,
                training_completed_at=completed_at,
                output_dir=output_dir,
            )

            self.assertNotEqual(first_path, second_path)
            self.assertTrue(first_path.is_dir())
            self.assertTrue(second_path.is_dir())

            latest = load_daily_max_temperature_emos_fits(
                "Chongqing-ZUCK",
                "test_ensemble",
                output_dir=output_dir,
            )
            historical = load_daily_max_temperature_emos_fits(
                "Chongqing-ZUCK",
                "test_ensemble",
                version=first_path.name,
                output_dir=output_dir,
            )

            self.assertEqual(latest.path, second_path)
            self.assertEqual(historical.path, first_path)
            self.assertTrue(
                bool(robjects.r["identical"](fit, historical.fits[("00", 1)])[0])  # type: ignore
            )
            manifest = historical.metadata
            self.assertEqual(manifest["summary"]["fitted_sample_count"], 20)
            self.assertEqual(manifest["groups"][0]["sample_count"], 20)
            self.assertEqual(
                manifest["groups"][0]["resolved_training_days"],
                14,
            )
            self.assertEqual(
                manifest["training_options"]["minimum_training_days"],
                DEFAULT_MINIMUM_TRAINING_DAYS,
            )
            self.assertEqual(
                manifest["skipped_groups"],
                [
                    {
                        "initialization_hour_utc": "06",
                        "day_ahead": 2,
                        "sample_count": 5,
                        "reason": "insufficient_training_dates",
                        "required_training_dates": 14,
                    }
                ],
            )
            self.assertEqual(manifest["groups"][0]["parameter_set_count"], 7)
            self.assertEqual(
                manifest["training_completed_at_utc"],
                "2026-07-19T01:02:03.000000Z",
            )
            self.assertEqual(
                manifest["extra_metadata"],
                {"source": "unit-test"},
            )

            latest_fit_file = latest.path / latest.metadata["groups"][0]["fit_file"]
            with latest_fit_file.open("ab") as output_file:
                output_file.write(b"tampered")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                load_daily_max_temperature_emos_fits(
                    "Chongqing-ZUCK",
                    "test_ensemble",
                    output_dir=output_dir,
                )

    def test_daily_max_emos_storage_rejects_unsafe_categories(self):
        with self.assertRaisesRegex(ValueError, "safe path component"):
            save_daily_max_temperature_emos_fits(
                {},
                {},
                "../Chongqing-ZUCK",
                "test_ensemble",
            )
        with self.assertRaisesRegex(ValueError, "safe path component"):
            load_daily_max_temperature_emos_fits(
                "Chongqing-ZUCK",
                "../test_ensemble",
            )

    def test_daily_max_emos_storage_does_not_publish_partial_version(self):
        groups = {
            ("00", 1): make_daily_max_group(2),
            ("06", 1): make_daily_max_group(
                2,
                initialization_time="06",
            ),
        }
        fits = {("00", 1): "fit00", ("06", 1): "fit06"}

        def describe_fit(fit):
            initialization_time = "00" if fit == "fit00" else "06"
            return {
                "fit_class": ["ensembleMOSnormal"],
                "resolved_training_days": 2,
                "lag_days": 1,
                "training_case_counts": [2],
                "forecast_hour": 24,
                "initialization_time": initialization_time,
                "modeled_dates": ["20260103"],
                "parameter_set_count": 1,
                "latest_parameter_date": "20260103",
            }

        def save_fit(fit, path):
            if fit == "fit06":
                raise RuntimeError("simulated RDS failure")
            path.write_bytes(b"mock-rds")

        with TemporaryDirectory() as output_dir:
            with (
                patch(
                    "train_emos_max_temperature._describe_r_fit",
                    side_effect=describe_fit,
                ),
                patch(
                    "train_emos_max_temperature._save_r_fit",
                    side_effect=save_fit,
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "simulated RDS failure"):
                    save_daily_max_temperature_emos_fits(
                        fits,
                        groups,
                        "Chongqing-ZUCK",
                        "test_ensemble",
                        training_days=2,
                        minimum_training_days=2,
                        output_dir=output_dir,
                    )

            model_directory = Path(output_dir) / "Chongqing-ZUCK" / "test_ensemble"
            self.assertTrue(model_directory.is_dir())
            self.assertEqual(list(model_directory.iterdir()), [])

    def test_train_all_daily_max_emos_traverses_config_and_saves(self):
        cities = [
            {
                "name": "City-One",
                "timezone": "Asia/Shanghai",
                "temp_unit": "C",
                "models": [{"name": "model-a"}, {"name": "model-b"}],
            },
            {
                "name": "City-Two",
                "timezone": "Europe/Paris",
                "temp_unit": "K",
                "models": [{"name": "model-c"}],
            },
        ]
        temperature_records = {
            "City-One": [{"city": "City-One"}],
            "City-Two": [{"city": "City-Two"}, {"city": "City-Two"}],
        }

        def load_temperatures(city_name, *, data_dir):
            return temperature_records[city_name]

        def load_forecasts(city_name, model_name, *, data_dir):
            return [{"pair": (city_name, model_name)}]

        def group_forecasts(forecasts, temperatures, city_timezone, **options):
            return {
                ("00", 1): {
                    "pair": forecasts[0]["pair"],
                    "temperature_count": len(temperatures),
                    "timezone": city_timezone.key,
                    "options": options,
                }
            }

        def train_groups(groups, **options):
            return {("00", 1): (groups[("00", 1)]["pair"], options)}

        def save_fits(fits, groups, city_name, model_name, **options):
            return Path(options["output_dir"]) / city_name / model_name / "version"

        with (
            patch("config.CITY", cities),
            patch(
                "train_emos_max_temperature._load_temperatures",
                side_effect=load_temperatures,
            ) as temperature_loader,
            patch(
                "train_emos_max_temperature._load_forecasts",
                side_effect=load_forecasts,
            ) as forecast_loader,
            patch(
                "train_emos_max_temperature."
                "group_daily_max_temperature_emos_training_data",
                side_effect=group_forecasts,
            ) as grouper,
            patch(
                "train_emos_max_temperature.train_daily_max_temperature_emos",
                side_effect=train_groups,
            ) as trainer,
            patch(
                "train_emos_max_temperature." "save_daily_max_temperature_emos_fits",
                side_effect=save_fits,
            ) as saver,
        ):
            artifacts = train_all_daily_max_temperature_emos(
                data_dir="input-data",
                output_dir="output-models",
                training_days=30,
                expected_interval_seconds=3600,
                minimum_observation_coverage=0.8,
                minimum_notice_hours=9.0,
                max_day_ahead=4,
                exchangeable=False,
                consecutive=True,
                control="custom-control",
                warm_start=True,
                skip_insufficient=True,
                extra_metadata={"run": "nightly"},
            )

        self.assertEqual(
            artifacts,
            {
                ("City-One", "model-a"): Path("output-models/City-One/model-a/version"),
                ("City-One", "model-b"): Path("output-models/City-One/model-b/version"),
                ("City-Two", "model-c"): Path("output-models/City-Two/model-c/version"),
            },
        )
        self.assertEqual(
            temperature_loader.call_args_list,
            [
                call("City-One", data_dir=Path("input-data")),
                call("City-Two", data_dir=Path("input-data")),
            ],
        )
        self.assertEqual(
            forecast_loader.call_args_list,
            [
                call("City-One", "model-a", data_dir=Path("input-data")),
                call("City-One", "model-b", data_dir=Path("input-data")),
                call("City-Two", "model-c", data_dir=Path("input-data")),
            ],
        )
        self.assertEqual(grouper.call_count, 3)
        self.assertEqual(trainer.call_count, 3)
        self.assertEqual(saver.call_count, 3)
        self.assertTrue(
            all(
                training_call.kwargs["minimum_training_days"]
                == DEFAULT_MINIMUM_TRAINING_DAYS
                for training_call in trainer.call_args_list
            )
        )
        self.assertEqual(
            [
                training_call.kwargs["input_unit"]
                for training_call in trainer.call_args_list
            ],
            ["celsius", "celsius", "kelvin"],
        )
        for save_call in saver.call_args_list:
            self.assertEqual(save_call.kwargs["training_days"], 30)
            self.assertEqual(
                save_call.kwargs["minimum_training_days"],
                DEFAULT_MINIMUM_TRAINING_DAYS,
            )
            self.assertEqual(save_call.kwargs["output_dir"], "output-models")
            self.assertEqual(save_call.kwargs["extra_metadata"]["run"], "nightly")
            self.assertIs(
                save_call.kwargs["training_completed_at"].tzinfo,
                timezone.utc,
            )
            batch_metadata = save_call.kwargs["extra_metadata"]["batch_training"]
            self.assertEqual(batch_metadata["data_directory"], "input-data")
            self.assertEqual(
                batch_metadata["grouping_options"]["minimum_notice_hours"],
                9.0,
            )
            self.assertTrue(batch_metadata["custom_control_supplied"])

    def test_train_daily_max_emos_for_one_city_model(self):
        cities = [
            {
                "name": "City-One",
                "timezone": "Asia/Shanghai",
                "temp_unit": "C",
                "models": [{"name": "model-a"}, {"name": "model-b"}],
            },
            {
                "name": "City-Two",
                "timezone": "Europe/Paris",
                "temp_unit": "K",
                "models": [{"name": "model-c"}],
            },
        ]
        forecasts = [{"forecast": 1}]
        temperatures = [{"observation": 1}, {"observation": 2}]
        groups = {("00", 1): {"group": 1}}
        fits = {("00", 1): "fit"}
        version_path = Path(
            "output-models/City-One/model-b/versions/test-version"
        )

        with (
            patch(
                "train_emos_max_temperature._load_temperatures",
                return_value=temperatures,
            ) as temperature_loader,
            patch(
                "train_emos_max_temperature._load_forecasts",
                return_value=forecasts,
            ) as forecast_loader,
            patch(
                "train_emos_max_temperature."
                "group_daily_max_temperature_emos_training_data",
                return_value=groups,
            ) as grouper,
            patch(
                "train_emos_max_temperature.train_daily_max_temperature_emos",
                return_value=fits,
            ) as trainer,
            patch(
                "train_emos_max_temperature."
                "save_daily_max_temperature_emos_fits",
                return_value=version_path,
            ) as saver,
        ):
            result = train_daily_max_temperature_emos_for_city_model(
                "City-One",
                "model-b",
                cities=cities,
                data_dir="input-data",
                output_dir="output-models",
                training_days=30,
                minimum_training_days=12,
                expected_interval_seconds=1800,
                minimum_observation_coverage=0.75,
                minimum_notice_hours=9.0,
                max_day_ahead=7,
                exchangeable=False,
                consecutive=True,
                control="custom-control",
                warm_start=True,
                extra_metadata={"trigger": "forecast_update"},
            )

        self.assertEqual(result, version_path)
        temperature_loader.assert_called_once_with(
            "City-One",
            data_dir=Path("input-data"),
        )
        forecast_loader.assert_called_once_with(
            "City-One",
            "model-b",
            data_dir=Path("input-data"),
        )
        grouper.assert_called_once_with(
            forecasts,
            temperatures,
            ZoneInfo("Asia/Shanghai"),
            expected_interval_seconds=1800,
            minimum_observation_coverage=0.75,
            minimum_notice_hours=9.0,
            max_day_ahead=7,
        )
        trainer.assert_called_once_with(
            groups,
            training_days=30,
            minimum_training_days=12,
            input_unit="celsius",
            exchangeable=False,
            consecutive=True,
            control="custom-control",
            warm_start=True,
            skip_insufficient=True,
        )
        saver.assert_called_once()
        save_call = saver.call_args
        self.assertEqual(save_call.args[:4], (fits, groups, "City-One", "model-b"))
        self.assertEqual(save_call.kwargs["output_dir"], "output-models")
        self.assertEqual(
            save_call.kwargs["extra_metadata"]["trigger"],
            "forecast_update",
        )
        batch_metadata = save_call.kwargs["extra_metadata"]["batch_training"]
        self.assertEqual(batch_metadata["forecast_record_count"], 1)
        self.assertEqual(batch_metadata["temperature_record_count"], 2)
        self.assertEqual(batch_metadata["city_timezone"], "Asia/Shanghai")

    def test_train_daily_max_emos_for_one_city_model_skips_empty_groups(self):
        city = {
            "name": "City-One",
            "timezone": "Asia/Shanghai",
            "models": [{"name": "model-a"}],
        }
        with (
            patch(
                "train_emos_max_temperature._load_temperatures",
                return_value=[{"observation": 1}],
            ),
            patch(
                "train_emos_max_temperature._load_forecasts",
                return_value=[{"forecast": 1}],
            ),
            patch(
                "train_emos_max_temperature."
                "group_daily_max_temperature_emos_training_data",
                return_value={},
            ),
            patch(
                "train_emos_max_temperature.train_daily_max_temperature_emos"
            ) as trainer,
            patch(
                "train_emos_max_temperature."
                "save_daily_max_temperature_emos_fits"
            ) as saver,
        ):
            with self.assertLogs(
                "train_emos_max_temperature",
                level="WARNING",
            ) as warning_logs:
                result = train_daily_max_temperature_emos_for_city_model(
                    "City-One",
                    "model-a",
                    cities=[city],
                )

        self.assertIsNone(result)
        self.assertIn(
            "No daily-max EMOS training groups for City-One/model-a",
            "\n".join(warning_logs.output),
        )
        trainer.assert_not_called()
        saver.assert_not_called()

    def test_train_daily_max_emos_for_one_city_model_validates_before_io(self):
        city = {
            "name": "City-One",
            "timezone": "Asia/Shanghai",
            "models": [{"name": "model-a"}],
        }
        with patch(
            "train_emos_max_temperature._load_temperatures"
        ) as temperature_loader:
            with self.assertRaisesRegex(
                ValueError,
                "model 'missing' is not configured",
            ):
                train_daily_max_temperature_emos_for_city_model(
                    "City-One",
                    "missing",
                    cities=[city],
                )
        temperature_loader.assert_not_called()

    def test_train_all_daily_max_emos_rejects_empty_pipeline_results(self):
        city = {
            "name": "City-One",
            "timezone": "Asia/Shanghai",
            "temp_unit": "C",
            "models": [{"name": "model-a"}],
        }
        with (
            patch(
                "train_emos_max_temperature._load_temperatures",
                return_value=[{"observation": 1}],
            ),
            patch(
                "train_emos_max_temperature._load_forecasts",
                return_value=[{"forecast": 1}],
            ),
            patch(
                "train_emos_max_temperature."
                "group_daily_max_temperature_emos_training_data",
                return_value={},
            ),
            patch(
                "train_emos_max_temperature.train_daily_max_temperature_emos"
            ) as trainer,
            patch(
                "train_emos_max_temperature." "save_daily_max_temperature_emos_fits"
            ) as saver,
        ):
            with self.assertRaisesRegex(
                ValueError,
                "No daily-max EMOS training groups for City-One/model-a",
            ):
                train_all_daily_max_temperature_emos(
                    cities=[city],
                    skip_insufficient=False,
                )
        trainer.assert_not_called()
        saver.assert_not_called()

        with (
            patch(
                "train_emos_max_temperature._load_temperatures",
                return_value=[{"observation": 1}],
            ),
            patch(
                "train_emos_max_temperature._load_forecasts",
                return_value=[{"forecast": 1}],
            ),
            patch(
                "train_emos_max_temperature."
                "group_daily_max_temperature_emos_training_data",
                return_value={("00", 1): {"group": 1}},
            ),
            patch(
                "train_emos_max_temperature.train_daily_max_temperature_emos",
                return_value={},
            ),
            patch(
                "train_emos_max_temperature." "save_daily_max_temperature_emos_fits"
            ) as saver,
        ):
            with self.assertLogs(
                "train_emos_max_temperature",
                level="WARNING",
            ) as warning_logs:
                artifacts = train_all_daily_max_temperature_emos(
                    cities=[city],
                )
        self.assertEqual(artifacts, {})
        self.assertIn(
            "No daily-max EMOS fits produced for City-One/model-a",
            "\n".join(warning_logs.output),
        )
        saver.assert_not_called()

    def test_train_all_daily_max_emos_validates_config_before_loading(self):
        duplicate_models = [
            {
                "name": "City-One",
                "timezone": "Asia/Shanghai",
                "models": [{"name": "model-a"}, {"name": "model-a"}],
            }
        ]
        with patch("train_emos_max_temperature._load_temperatures") as loader:
            with self.assertRaisesRegex(ValueError, "duplicate city/model"):
                train_all_daily_max_temperature_emos(cities=duplicate_models)
        loader.assert_not_called()

        invalid_timezone = [
            {
                "name": "City-One",
                "timezone": "Not/A-Timezone",
                "models": [{"name": "model-a"}],
            }
        ]
        with self.assertRaisesRegex(ValueError, "unknown timezone"):
            train_all_daily_max_temperature_emos(cities=invalid_timezone)

    def test_daily_max_ensemble_data_rejects_duplicate_dates(self):
        group = make_daily_max_group(2)
        group["target_dates"] = ["2026-01-01", "2026-01-01"]

        with self.assertRaisesRegex(ValueError, "must be unique"):
            build_daily_max_temperature_ensemble_data(group)

    def test_loads_forecasts_and_temperatures_from_data_directory(self):
        with patch("train_emos_max_temperature.DATA_DIR", TEST_DATA_DIR):
            forecasts = _load_forecasts(
                "Chongqing-ZUCK",
                "ecmwf_aifs025_ensemble",
            )
            temperatures = _load_temperatures("Chongqing-ZUCK")

        self.assertEqual(len(forecasts), 12)
        self.assertEqual(len(temperatures), 67)
        self.assertTrue(all(isinstance(item, dict) for item in forecasts))
        self.assertTrue(all(isinstance(item, dict) for item in temperatures))
        self.assertEqual(
            [item["meta"]["last_run_initialisation_time"] for item in forecasts],
            sorted(item["meta"]["last_run_initialisation_time"] for item in forecasts),
        )
        self.assertEqual(
            [item["time"] for item in temperatures],
            sorted(item["time"] for item in temperatures),
        )

    def test_groups_one_complete_local_day_and_calculates_member_maxima(self):
        zone = ZoneInfo("Asia/Shanghai")
        target_date = date(2026, 7, 18)
        timestamps = local_day_timestamps(target_date, zone)
        initialization_time = int(
            datetime(2026, 7, 17, tzinfo=timezone.utc).timestamp()
        )
        availability_time = int(
            datetime(2026, 7, 17, 9, tzinfo=timezone.utc).timestamp()
        )

        groups = group_daily_max_temperature_emos_training_data(
            [make_forecast(initialization_time, availability_time, timestamps)],
            make_observations(timestamps),
            zone,
        )

        self.assertEqual(set(groups), {("00", 1)})
        group = groups[("00", 1)]
        self.assertEqual(group["target_dates"], ["2026-07-18"])
        self.assertEqual(
            group["member_names"],
            (
                "temperature_2m",
                "temperature_2m_member01",
            ),
        )
        self.assertEqual(group["forecasts"], [(23.0, 123.0)])
        self.assertEqual(group["observations"], [33.0])
        self.assertEqual(group["forecast_counts"], [24])
        self.assertEqual(group["observation_counts"], [24])
        self.assertEqual(group["observation_coverages"], [1.0])

    def test_rejects_forecast_not_available_before_local_day(self):
        zone = ZoneInfo("Asia/Shanghai")
        target_date = date(2026, 7, 18)
        timestamps = local_day_timestamps(target_date, zone)
        initialization_time = int(
            datetime(2026, 7, 17, tzinfo=timezone.utc).timestamp()
        )
        local_day_start = timestamps[0]

        groups = group_daily_max_temperature_emos_training_data(
            [make_forecast(initialization_time, local_day_start, timestamps)],
            make_observations(timestamps),
            zone,
        )

        self.assertEqual(groups, {})

    def test_requires_complete_days_unless_observation_coverage_is_relaxed(self):
        zone = ZoneInfo("Asia/Shanghai")
        target_date = date(2026, 7, 18)
        timestamps = local_day_timestamps(target_date, zone)
        initialization_time = int(
            datetime(2026, 7, 16, tzinfo=timezone.utc).timestamp()
        )
        availability_time = initialization_time + 3600
        forecast = make_forecast(
            initialization_time,
            availability_time,
            timestamps,
        )

        incomplete_forecast = make_forecast(
            initialization_time,
            availability_time,
            timestamps[:-1],
        )
        self.assertEqual(
            group_daily_max_temperature_emos_training_data(
                [incomplete_forecast],
                make_observations(timestamps),
                zone,
            ),
            {},
        )

        incomplete_observations = make_observations(timestamps[:-1])
        self.assertEqual(
            group_daily_max_temperature_emos_training_data(
                [forecast],
                incomplete_observations,
                zone,
            ),
            {},
        )

        relaxed = group_daily_max_temperature_emos_training_data(
            [forecast],
            incomplete_observations,
            zone,
            minimum_observation_coverage=23 / 24,
        )
        self.assertEqual(relaxed[("00", 2)]["observation_coverages"], [23 / 24])

    def test_uses_first_complete_snapshot_without_duplicating_a_run(self):
        zone = ZoneInfo("Asia/Shanghai")
        target_date = date(2026, 7, 18)
        timestamps = local_day_timestamps(target_date, zone)
        initialization_time = int(
            datetime(2026, 7, 16, tzinfo=timezone.utc).timestamp()
        )
        forecasts = [
            make_forecast(
                initialization_time,
                initialization_time + 3600,
                timestamps[:-1],
                base=1000.0,
            ),
            make_forecast(
                initialization_time,
                initialization_time + 7200,
                timestamps,
                base=10.0,
            ),
            make_forecast(
                initialization_time,
                initialization_time + 10800,
                timestamps,
                base=2000.0,
            ),
        ]

        groups = group_daily_max_temperature_emos_training_data(
            forecasts,
            make_observations(timestamps),
            zone,
        )

        group = groups[("00", 2)]
        self.assertEqual(group["forecasts"], [(33.0, 133.0)])
        self.assertEqual(
            group["availability_times"],
            [initialization_time + 7200],
        )

    def test_dst_day_uses_23_hour_local_calendar_day(self):
        zone = ZoneInfo("Europe/Paris")
        target_date = date(2026, 3, 29)
        timestamps = local_day_timestamps(target_date, zone)
        self.assertEqual(len(timestamps), 23)
        initialization_time = int(
            datetime(2026, 3, 28, tzinfo=timezone.utc).timestamp()
        )

        groups = group_daily_max_temperature_emos_training_data(
            [
                make_forecast(
                    initialization_time,
                    initialization_time + 3600,
                    timestamps,
                )
            ],
            make_observations(timestamps),
            zone,
        )

        group = groups[("00", 1)]
        self.assertEqual(group["forecast_counts"], [23])
        self.assertEqual(group["observation_counts"], [23])
        self.assertEqual(group["forecasts"], [(22.0, 122.0)])

    def test_dst_day_uses_25_hour_local_calendar_day(self):
        zone = ZoneInfo("Europe/Paris")
        target_date = date(2026, 10, 25)
        timestamps = local_day_timestamps(target_date, zone)
        self.assertEqual(len(timestamps), 25)
        initialization_time = int(
            datetime(2026, 10, 24, tzinfo=timezone.utc).timestamp()
        )

        groups = group_daily_max_temperature_emos_training_data(
            [
                make_forecast(
                    initialization_time,
                    initialization_time + 3600,
                    timestamps,
                )
            ],
            make_observations(timestamps),
            zone,
        )

        group = groups[("00", 1)]
        self.assertEqual(group["forecast_counts"], [25])
        self.assertEqual(group["observation_counts"], [25])
        self.assertEqual(group["forecasts"], [(24.0, 124.0)])

    def test_groups_real_chongqing_data(self):
        forecasts = load_jsonl(
            TEST_DATA_DIR
            / "forecast"
            / "Chongqing-ZUCK"
            / "ecmwf_aifs025_ensemble"
            / "fc.jsonl"
        )
        temperatures = load_jsonl(
            TEST_DATA_DIR / "temperature" / "Chongqing-ZUCK" / "tem.jsonl"
        )

        groups = group_daily_max_temperature_emos_training_data(
            forecasts,
            temperatures,
            ZoneInfo("Asia/Shanghai"),
        )

        self.assertEqual(
            {key: len(group["target_dates"]) for key, group in groups.items()},
            {
                ("00", 1): 2,
                ("00", 2): 1,
                ("06", 1): 2,
                ("06", 2): 1,
                ("12", 2): 1,
                ("18", 1): 1,
            },
        )
        self.assertNotIn(("12", 1), groups)
        self.assertEqual(
            groups[("00", 1)]["target_dates"],
            [
                "2026-07-16",
                "2026-07-17",
            ],
        )
        self.assertEqual(groups[("00", 1)]["observations"], [40.0, 34.0])


if __name__ == "__main__":
    unittest.main()

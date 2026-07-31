"""Integration tests for the Python-to-R EMOS bridge."""

import io
import math
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from rpy2 import robjects
from rpy2.robjects import vectors
from rpy2.robjects.packages import importr

from train_emos import (
    HOURLY_TEMPERATURE_EMOS_ARTIFACT_TYPE,
    build_temperature_ensemble_data,
    ensemble_mos,
    group_emos_training_data,
    load_hourly_temperature_emos_fits,
    match_forecast,
    save_hourly_temperature_emos_fits,
    train_all_ensemble_mos,
    train_ensemble_mos,
    train_grouped_ensemble_mos,
)

TEST_DATA_DIR = Path(__file__).resolve().parents[1] / "test-data" / "data2"


def make_mock_ensemble_data(number_of_days: int = 20):
    """Build deterministic synthetic temperature data as an R ensembleData."""
    forecast_values = []
    observations = []

    for day in range(number_of_days):
        temperature = 18.0 + 0.15 * day + 1.8 * math.sin(day / 3)
        forecast_values.extend(
            (
                temperature - 1.0 + 0.15 * math.cos(day),
                temperature + 0.7,
                temperature + 0.2 * math.sin(day / 2),
            )
        )
        observations.append(temperature + 0.35 * math.cos(day * 1.7))

    forecasts = robjects.r["matrix"](
        vectors.FloatVector(forecast_values),
        nrow=number_of_days,
        byrow=True,
    )  # type: ignore
    forecasts.colnames = vectors.StrVector(("member01", "member02", "member03"))
    dates = [
        (datetime(2026, 1, 1) + timedelta(days=day)).strftime("%Y%m%d%H")
        for day in range(number_of_days)
    ]

    ensemble_bma = importr("ensembleBMA")
    ensemble_data = ensemble_bma.ensembleData(
        forecasts=forecasts,
        dates=vectors.StrVector(dates),
        observations=vectors.FloatVector(observations),
        forecastHour=24,
        initializationTime="00",
    )
    return ensemble_data, dates


def make_mock_group(number_of_days: int = 20):
    """Build the Python-side representation of one homogeneous EMOS group."""
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    valid_times = []
    forecasts = []
    observations = []
    for day in range(number_of_days):
        temperature = 18.0 + 0.15 * day + 1.8 * math.sin(day / 3)
        valid_times.append(int((start + timedelta(days=day)).timestamp()))
        forecasts.append(
            (
                temperature - 1.0 + 0.15 * math.cos(day),
                temperature + 0.7,
                temperature + 0.2 * math.sin(day / 2),
            )
        )
        observations.append(temperature + 0.35 * math.cos(day * 1.7))

    return {
        "initialization_time": "00",
        "forecast_hour": 24,
        "member_names": ("member01", "member02", "member03"),
        "initialization_times": [valid_time - 24 * 3600 for valid_time in valid_times],
        "valid_times": valid_times,
        "forecasts": forecasts,
        "observations": observations,
    }


class EnsembleMosTest(unittest.TestCase):
    def test_build_temperature_ensemble_data_converts_celsius_to_kelvin(self):
        timestamp = int(datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp())
        group = {
            "initialization_time": "00",
            "forecast_hour": 24,
            "member_names": (
                "temperature_2m",
                "temperature_2m_member01",
            ),
            "valid_times": [timestamp],
            "forecasts": [(0.0, 1.0)],
            "observations": [0.5],
        }

        ensemble_data = build_temperature_ensemble_data(group)
        ensemble_bma = importr("ensembleBMA")
        forecasts = ensemble_bma.ensembleForecasts(ensemble_data)

        self.assertAlmostEqual(forecasts[0], 273.15)
        self.assertAlmostEqual(forecasts[1], 274.15)
        self.assertAlmostEqual(
            ensemble_bma.dataVerifObs(ensemble_data)[0],
            273.65,
        )
        self.assertEqual(
            list(ensemble_bma.ensembleValidDates(ensemble_data)),
            ["2026010100"],
        )
        self.assertEqual(list(ensemble_bma.ensembleFhour(ensemble_data)), [24])
        self.assertEqual(list(ensemble_bma.ensembleItime(ensemble_data)), ["00"])
        self.assertEqual(
            list(ensemble_bma.ensembleGroups(ensemble_data)),
            ["control", "perturbed"],
        )

    def test_match_forecast_uses_latest_available_run(self):
        forecasts = [
            {
                "name": "late",
                "meta": {"last_run_availability_time": 30},
            },
            {
                "name": "early",
                "meta": {"last_run_availability_time": 10},
            },
            {
                "name": "middle",
                "meta": {"last_run_availability_time": 20},
            },
        ]

        matched = match_forecast(forecasts, [5, 10, 19, 20, 35])

        self.assertEqual(
            [None if forecast is None else forecast["name"] for forecast in matched],
            [None, "early", "early", "middle", "late"],
        )
        previous = match_forecast(forecasts, [20, 35], offset=1)
        self.assertEqual(
            [None if forecast is None else forecast["name"] for forecast in previous],
            ["early", "middle"],
        )

    def test_group_emos_training_data_groups_by_cycle_and_lead(self):
        day = int(datetime(2026, 7, 15, tzinfo=timezone.utc).timestamp())
        forecasts = [
            {
                "time": [day - 3600, day + 12 * 3600, day + 18 * 3600],
                "temperature_2m": [99.0, 30.0, 31.0],
                "temperature_2m_member01": [99.0, 29.0, 30.0],
                "meta": {
                    "last_run_initialisation_time": day,
                    "last_run_availability_time": day + 3600,
                },
            },
            {
                "time": [day + 12 * 3600, day + 18 * 3600],
                "temperature_2m": [32.0, 33.0],
                "temperature_2m_member01": [31.0, 32.0],
                "meta": {
                    "last_run_initialisation_time": day + 6 * 3600,
                    "last_run_availability_time": day + 7 * 3600,
                },
            },
        ]
        temperatures = [
            {"time": day + 12 * 3600, "temperature": 34.0},
            {"time": day + 18 * 3600, "temperature": 35.0},
        ]

        groups = group_emos_training_data(
            forecasts,
            temperatures,
            lead_step_hours=6,
        )

        self.assertEqual(
            set(groups),
            {("00", 12), ("00", 18), ("06", 6), ("06", 12)},
        )
        self.assertEqual(groups[("00", 12)]["forecasts"], [(30.0, 29.0)])
        self.assertEqual(groups[("06", 6)]["forecasts"], [(32.0, 31.0)])
        self.assertEqual(groups[("00", 12)]["observations"], [34.0])
        self.assertEqual(groups[("06", 6)]["observations"], [34.0])
        self.assertEqual(
            groups[("00", 12)]["member_names"],
            ("temperature_2m", "temperature_2m_member01"),
        )

    def test_group_emos_training_data_defaults_to_96_hour_maximum_lead(self):
        initialization_time = int(
            datetime(2026, 7, 15, tzinfo=timezone.utc).timestamp()
        )
        valid_time_96h = initialization_time + 96 * 3600
        valid_time_97h = initialization_time + 97 * 3600
        forecasts = [
            {
                "time": [valid_time_96h, valid_time_97h],
                "temperature_2m": [30.0, 31.0],
                "temperature_2m_member01": [29.0, 30.0],
                "meta": {
                    "last_run_initialisation_time": initialization_time,
                    "last_run_availability_time": initialization_time + 3600,
                },
            }
        ]
        temperatures = [
            {"time": valid_time_96h, "temperature": 32.0},
            {"time": valid_time_97h, "temperature": 33.0},
        ]

        groups = group_emos_training_data(forecasts, temperatures)

        self.assertEqual(set(groups), {("00", 96)})

    def test_ensemble_mos_runs_with_mock_temperature_data(self):
        ensemble_data, dates = make_mock_ensemble_data()

        console_output = io.StringIO()
        with redirect_stdout(console_output):
            result = ensemble_mos(
                ensemble_data,
                training_days=14,
                dates=vectors.StrVector((dates[-1],)),
            )

        result_classes = set(robjects.r["class"](result))  # type: ignore
        self.assertIn("ensembleMOSnormal", result_classes)
        self.assertTrue({"training", "a", "B", "c", "d"}.issubset(result.names))
        self.assertNotIn("modeling for date", console_output.getvalue())

        for parameter_name in ("a", "B", "c", "d"):
            parameter_values = result.rx2(parameter_name)
            self.assertTrue(
                all(math.isfinite(value) for value in parameter_values),
                f"{parameter_name} contains a non-finite fitted value",
            )

    def test_train_grouped_ensemble_mos_runs_for_each_sufficient_group(self):
        groups = {("00", 24): make_mock_group()}

        fits = train_grouped_ensemble_mos(groups, training_days=14)

        self.assertEqual(set(fits), {("00", 24)})
        result_classes = set(robjects.r["class"](fits[("00", 24)]))  # type: ignore
        self.assertIn("ensembleMOSnormal", result_classes)

    def test_hourly_emos_storage_versions_and_restores_r_fit(self):
        groups = {("00", 24): make_mock_group()}
        fits = train_grouped_ensemble_mos(groups, training_days=14)
        completed_at = datetime(2026, 7, 19, 2, 3, 4, tzinfo=timezone.utc)

        with TemporaryDirectory() as artifact_root:
            first_path = save_hourly_temperature_emos_fits(
                fits,
                groups,
                "Chongqing-ZUCK",
                "test_ensemble",
                training_days=14,
                training_completed_at=completed_at,
                extra_metadata={"source": "unit-test"},
                artifact_root=artifact_root,
            )
            second_path = save_hourly_temperature_emos_fits(
                fits,
                groups,
                "Chongqing-ZUCK",
                "test_ensemble",
                training_days=14,
                training_completed_at=completed_at,
                artifact_root=artifact_root,
            )

            expected_parent = (
                Path(artifact_root)
                / HOURLY_TEMPERATURE_EMOS_ARTIFACT_TYPE
                / "Chongqing-ZUCK"
                / "test_ensemble"
                / "versions"
            )
            self.assertEqual(first_path.parent, expected_parent)
            self.assertEqual(second_path.parent, expected_parent)
            self.assertNotEqual(first_path, second_path)

            latest = load_hourly_temperature_emos_fits(
                "Chongqing-ZUCK",
                "test_ensemble",
                artifact_root=artifact_root,
            )
            historical = load_hourly_temperature_emos_fits(
                "Chongqing-ZUCK",
                "test_ensemble",
                version=first_path.name,
                artifact_root=artifact_root,
            )
            self.assertEqual(latest.path, second_path)
            self.assertEqual(historical.path, first_path)
            self.assertTrue(
                bool(
                    robjects.r["identical"](
                        fits[("00", 24)],
                        historical.fits[("00", 24)],
                    )[  # type: ignore
                        0
                    ]  # type: ignore
                )
            )

            manifest = historical.metadata
            self.assertEqual(manifest["artifact_type"], "hourly_temperature_emos")
            self.assertEqual(manifest["summary"]["fitted_sample_count"], 20)
            self.assertEqual(manifest["groups"][0]["lead_hour"], 24)
            self.assertEqual(manifest["groups"][0]["sample_count"], 20)
            self.assertEqual(
                manifest["groups"][0]["resolved_training_days"],
                14,
            )
            self.assertEqual(manifest["extra_metadata"], {"source": "unit-test"})

            latest_fit_path = latest.path / latest.metadata["groups"][0]["fit_file"]
            with latest_fit_path.open("ab") as output_file:
                output_file.write(b"tampered")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                load_hourly_temperature_emos_fits(
                    "Chongqing-ZUCK",
                    "test_ensemble",
                    artifact_root=artifact_root,
                )

    def test_hourly_emos_storage_does_not_publish_partial_version(self):
        first_group = make_mock_group(number_of_days=2)
        second_group = make_mock_group(number_of_days=2)
        second_group["initialization_time"] = "06"
        second_group["forecast_hour"] = 12
        groups = {("00", 24): first_group, ("06", 12): second_group}
        fits = {("00", 24): "fit00", ("06", 12): "fit06"}

        def describe_fit(fit):
            initialization_time = "00" if fit == "fit00" else "06"
            forecast_hour = 24 if fit == "fit00" else 12
            return {
                "fit_class": ["ensembleMOSnormal"],
                "resolved_training_days": 2,
                "lag_days": 1,
                "training_case_counts": [2],
                "forecast_hour": forecast_hour,
                "initialization_time": initialization_time,
                "modeled_dates": ["2026010300"],
                "parameter_set_count": 1,
                "latest_parameter_date": "2026010300",
            }

        def save_fit(fit, path):
            if fit == "fit06":
                raise RuntimeError("simulated hourly RDS failure")
            path.write_bytes(b"mock-rds")

        with TemporaryDirectory() as artifact_root:
            with (
                patch(
                    "train_emos._describe_hourly_emos_fit",
                    side_effect=describe_fit,
                ),
                patch(
                    "train_emos._save_hourly_emos_r_fit",
                    side_effect=save_fit,
                ),
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "simulated hourly RDS failure",
                ):
                    save_hourly_temperature_emos_fits(
                        fits,
                        groups,
                        "Chongqing-ZUCK",
                        "test_ensemble",
                        training_days=2,
                        artifact_root=artifact_root,
                    )

            model_directory = (
                Path(artifact_root)
                / HOURLY_TEMPERATURE_EMOS_ARTIFACT_TYPE
                / "Chongqing-ZUCK"
                / "test_ensemble"
            )
            self.assertFalse((model_directory / "latest.json").exists())
            self.assertEqual(list((model_directory / "versions").iterdir()), [])

    def test_train_all_ensemble_mos_traverses_config_and_saves(self):
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

        def group_forecasts(forecasts, temperatures, **options):
            return {
                ("00", 12): {
                    "pair": forecasts[0]["pair"],
                    "temperature_count": len(temperatures),
                    "options": options,
                }
            }

        def train_groups(groups, **options):
            return {("00", 12): (groups[("00", 12)]["pair"], options)}

        def save_fits(fits, groups, city_name, model_name, **options):
            return Path(options["artifact_root"]) / city_name / model_name / "version"

        with (
            patch("config.CITY", cities),
            patch(
                "train_emos.load_temperature",
                side_effect=load_temperatures,
            ) as temperature_loader,
            patch(
                "train_emos.load_forecast",
                side_effect=load_forecasts,
            ) as forecast_loader,
            patch(
                "train_emos.group_emos_training_data",
                side_effect=group_forecasts,
            ) as grouper,
            patch(
                "train_emos.train_grouped_ensemble_mos",
                side_effect=train_groups,
            ) as trainer,
            patch(
                "train_emos.save_hourly_temperature_emos_fits",
                side_effect=save_fits,
            ) as saver,
        ):
            artifacts = train_all_ensemble_mos(
                data_dir="input-data",
                artifact_root="trained-models",
                training_days=30,
                lead_step_hours=6,
                max_lead_hours=48,
                require_available_before_valid=True,
                exchangeable=False,
                consecutive=True,
                control="custom-control",
                warm_start=True,
                skip_insufficient=True,
                extra_metadata={"run": "nightly"},
            )

        self.assertEqual(
            set(artifacts),
            {
                ("City-One", "model-a"),
                ("City-One", "model-b"),
                ("City-Two", "model-c"),
            },
        )
        self.assertEqual(temperature_loader.call_count, 2)
        self.assertEqual(forecast_loader.call_count, 3)
        self.assertEqual(grouper.call_count, 3)
        self.assertEqual(trainer.call_count, 3)
        self.assertEqual(saver.call_count, 3)
        self.assertEqual(
            [fit_call.kwargs["input_unit"] for fit_call in trainer.call_args_list],
            ["celsius", "celsius", "kelvin"],
        )
        self.assertEqual(
            [fit_call.kwargs["training_days"] for fit_call in trainer.call_args_list],
            [30, 30, 30],
        )
        for save_call in saver.call_args_list:
            self.assertEqual(save_call.kwargs["artifact_root"], "trained-models")
            self.assertEqual(save_call.kwargs["extra_metadata"]["run"], "nightly")
            batch_metadata = save_call.kwargs["extra_metadata"]["batch_training"]
            self.assertEqual(batch_metadata["data_directory"], "input-data")
            self.assertEqual(
                batch_metadata["grouping_options"]["lead_step_hours"],
                6,
            )
            self.assertTrue(batch_metadata["custom_control_supplied"])

    def test_train_all_ensemble_mos_rejects_empty_pipeline_results(self):
        city = {
            "name": "City-One",
            "timezone": "Asia/Shanghai",
            "temp_unit": "C",
            "models": [{"name": "model-a"}],
        }
        with (
            patch("train_emos.load_temperature", return_value=[{"obs": 1}]),
            patch("train_emos.load_forecast", return_value=[{"forecast": 1}]),
            patch("train_emos.group_emos_training_data", return_value={}),
            patch("train_emos.train_grouped_ensemble_mos") as trainer,
            patch("train_emos.save_hourly_temperature_emos_fits") as saver,
        ):
            with self.assertRaisesRegex(
                ValueError,
                "no hourly EMOS training groups for City-One/model-a",
            ):
                train_all_ensemble_mos(cities=[city])
        trainer.assert_not_called()
        saver.assert_not_called()

        with (
            patch("train_emos.load_temperature", return_value=[{"obs": 1}]),
            patch("train_emos.load_forecast", return_value=[{"forecast": 1}]),
            patch(
                "train_emos.group_emos_training_data",
                return_value={("00", 12): {"group": 1}},
            ),
            patch("train_emos.train_grouped_ensemble_mos", return_value={}),
            patch("train_emos.save_hourly_temperature_emos_fits") as saver,
        ):
            with self.assertRaisesRegex(
                ValueError,
                "no hourly EMOS fits produced for City-One/model-a",
            ):
                train_all_ensemble_mos(
                    cities=[city],
                    skip_insufficient=True,
                )
        saver.assert_not_called()

    def test_hourly_emos_storage_and_batch_reject_unsafe_config(self):
        with self.assertRaisesRegex(ValueError, "safe path component"):
            load_hourly_temperature_emos_fits(
                "../City-One",
                "model-a",
            )

        duplicate_models = [
            {
                "name": "City-One",
                "timezone": "Asia/Shanghai",
                "models": [{"name": "model-a"}, {"name": "model-a"}],
            }
        ]
        with patch("train_emos.load_temperature") as temperature_loader:
            with self.assertRaisesRegex(ValueError, "duplicate city/model"):
                train_all_ensemble_mos(cities=duplicate_models)
        temperature_loader.assert_not_called()

        invalid_timezone = [
            {
                "name": "City-One",
                "timezone": "Not/A-Timezone",
                "models": [{"name": "model-a"}],
            }
        ]
        with self.assertRaisesRegex(ValueError, "unknown timezone"):
            train_all_ensemble_mos(cities=invalid_timezone)

    def test_train_grouped_ensemble_mos_adapts_days_for_each_group(self):
        group_24h = make_mock_group(number_of_days=5)
        group_48h = make_mock_group(number_of_days=8)
        group_48h["forecast_hour"] = 48
        groups = {("00", 24): group_24h, ("00", 48): group_48h}

        with patch(
            "train_emos.ensemble_mos",
            side_effect=("fit24", "fit48"),
        ) as fit:
            fits = train_grouped_ensemble_mos(groups, training_days=None)

        self.assertEqual(fits, {("00", 24): "fit24", ("00", 48): "fit48"})
        self.assertEqual(
            [call.kwargs["training_days"] for call in fit.call_args_list],
            [5, 8],
        )

    def test_train_grouped_ensemble_mos_rejects_insufficient_groups(self):
        groups = {("00", 24): make_mock_group(number_of_days=5)}

        with self.assertRaisesRegex(ValueError, "need 14 dates"):
            train_grouped_ensemble_mos(groups, training_days=14)

        self.assertEqual(
            train_grouped_ensemble_mos(
                groups,
                training_days=14,
                skip_insufficient=True,
            ),
            {},
        )

    def test_train_ensemble_mos_loads_city_model_from_data_directory(self):
        expected_fits = {("00", 24): "fit"}

        with patch(
            "train_emos.train_grouped_ensemble_mos",
            return_value=expected_fits,
        ) as grouped_train:
            fits = train_ensemble_mos(
                "Wuhan-ZHHH",
                "ecmwf_aifs025_ensemble",
                data_dir=TEST_DATA_DIR,
                lead_step_hours=6,
                max_lead_hours=48,
                warm_start=True,
            )

        self.assertEqual(fits, expected_fits)
        groups = grouped_train.call_args.args[0]
        self.assertTrue(groups)
        self.assertTrue(all(lead <= 48 for _, lead in groups))
        self.assertTrue(
            all("temperature_2m" in group["member_names"] for group in groups.values())
        )
        self.assertIsNone(grouped_train.call_args.kwargs["training_days"])
        self.assertTrue(grouped_train.call_args.kwargs["warm_start"])

    def test_train_ensemble_mos_fits_chongqing_data_and_prints_result(self):
        """Smoke-test a real R fit; three dates are not enough for production."""
        fits = train_ensemble_mos(
            "Chongqing-ZUCK",
            "ecmwf_aifs025_ensemble",
            data_dir=TEST_DATA_DIR,
            training_days=3,
            lead_step_hours=6,
            max_lead_hours=12,
            skip_insufficient=True,
        )

        self.assertEqual(set(fits), {("00", 12), ("06", 12), ("12", 12)})
        for key, fit in sorted(fits.items()):
            result_classes = set(robjects.r["class"](fit))  # type: ignore
            self.assertIn("ensembleMOSnormal", result_classes)

            training = fit.rx2("training")
            a = fit.rx2("a")
            coefficients = fit.rx2("B")
            c = fit.rx2("c")
            d = fit.rx2("d")
            for parameter in (a, coefficients, c, d):
                self.assertTrue(all(math.isfinite(float(value)) for value in parameter))

            member_names = list(coefficients.rownames)
            coefficient_values = list(coefficients)
            coefficient_by_member = dict(zip(member_names, coefficient_values))
            print(
                "\nChongqing EMOS fit:",
                {
                    "group": key,
                    "model_date": list(a.colnames)[-1],
                    "training_days": int(training[0]),
                    "training_cases": int(training[-1]),
                    "a": float(a[-1]),
                    "B_control": float(coefficient_by_member["temperature_2m"]),
                    "B_perturbed_member": float(
                        coefficient_by_member["temperature_2m_member01"]
                    ),
                    "c": float(c[-1]),
                    "d": float(d[-1]),
                },
            )


if __name__ == "__main__":
    unittest.main()

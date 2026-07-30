"""Tests for daily maximum-temperature EMOS prediction."""

import math
import unittest
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from unittest.mock import patch, sentinel
from zoneinfo import ZoneInfo

from predict_emos_max_temperature import (
    _ArtifactGroup,
    _DailyMaxPredictionCase,
    _PredictionUnavailableError,
    _artifact_groups,
    _build_daily_max_prediction_ensemble_data,
    _call_ensemble_mos_cdf,
    _call_ensemble_mos_cdf_values,
    _interval_probabilities_from_cdf,
    _select_daily_max_forecast,
    _validate_interval_boundaries,
    predict_daily_max_temperature_intervals,
    probability_daily_max_temperature_below,
)
from train_emos_max_temperature import DailyMaxTemperatureEmosArtifact


CITY = "Chongqing-ZUCK"
MODEL = "ecmwf_aifs025_ensemble"
ZONE = ZoneInfo("Asia/Shanghai")
MEMBERS = ("temperature_2m", "temperature_2m_member01")


def local_day_timestamps(target_date: date) -> list[int]:
    start = datetime.combine(target_date, time.min, tzinfo=ZONE)
    end = datetime.combine(target_date + timedelta(days=1), time.min, tzinfo=ZONE)
    return list(range(int(start.timestamp()), int(end.timestamp()), 3600))


def timestamp(value: str) -> int:
    return int(datetime.fromisoformat(value).timestamp())


def make_forecast(
    initialization_time: int,
    availability_time: int,
    timestamps: list[int],
    *,
    base: float = 10.0,
) -> dict:
    return {
        "model": MODEL,
        "time": timestamps,
        "temperature_2m": [
            base + index / 10 for index in range(len(timestamps))
        ],
        "temperature_2m_member01": [
            base + 1 + index / 10 for index in range(len(timestamps))
        ],
        "meta": {
            "last_run_initialisation_time": initialization_time,
            "last_run_availability_time": availability_time,
        },
    }


def make_group(
    key: tuple[str, int],
    modeled_dates: set[str],
    *,
    fit=sentinel.fit,
) -> _ArtifactGroup:
    return _ArtifactGroup(
        key=key,
        fit=fit,
        modeled_dates=frozenset(modeled_dates),
        member_names=MEMBERS,
        timezone_name=ZONE.key,
    )


def make_artifact(
    key: tuple[str, int] = ("12", 2),
    *,
    modeled_date: str = "20260103",
    training_completed_at: str = "2026-01-02T00:00:00Z",
) -> DailyMaxTemperatureEmosArtifact:
    entry = {
        "initialization_hour_utc": key[0],
        "initialization_time": key[0],
        "day_ahead": key[1],
        "forecast_hour": key[1] * 24,
        "modeled_dates": [modeled_date],
        "member_names": list(MEMBERS),
        "timezone": ZONE.key,
    }
    return DailyMaxTemperatureEmosArtifact(
        version="test-version",
        path=Path("/mock/test-version"),
        fits={key: sentinel.fit},
        metadata={
            "training_completed_at_utc": training_completed_at,
            "saved_at_utc": training_completed_at,
            "stored_temperature_unit": "kelvin",
            "training_options": {
                "input_unit": "celsius",
                "exchangeable": True,
                "model": "normal",
            },
            "groups": [entry],
            "extra_metadata": {
                "batch_training": {
                    "grouping_options": {
                        "expected_interval_seconds": 3600,
                        "minimum_notice_hours": 0.0,
                    }
                }
            },
        },
    )


class DailyMaxTemperatureForecastSelectionTest(unittest.TestCase):
    def test_latest_run_without_target_parameters_falls_back_to_older_run(self):
        target = date(2026, 1, 3)
        hours = local_day_timestamps(target)
        older = make_forecast(
            timestamp("2026-01-01T12:00:00+00:00"),
            timestamp("2026-01-01T20:00:00+00:00"),
            hours,
            base=20,
        )
        newer = make_forecast(
            timestamp("2026-01-02T00:00:00+00:00"),
            timestamp("2026-01-02T10:00:00+00:00"),
            hours,
            base=30,
        )
        groups = {
            ("12", 2): make_group(("12", 2), {"20260103"}),
            # This newer run has a fit, but not one modeled for the target date.
            ("00", 1): make_group(("00", 1), {"20260102"}),
        }

        selected = _select_daily_max_forecast(
            [newer, older],
            model_name=MODEL,
            target_date=target,
            city_timezone=ZONE,
            artifact_groups=groups,
            as_of=datetime(2026, 1, 2, 12, tzinfo=timezone.utc),
            expected_interval_seconds=3600,
            minimum_notice_hours=0,
        )

        self.assertEqual(selected.group.key, ("12", 2))
        self.assertEqual(
            selected.initialization_time,
            timestamp("2026-01-01T12:00:00+00:00"),
        )

    def test_uses_first_complete_snapshot_of_selected_initialization(self):
        target = date(2026, 1, 3)
        hours = local_day_timestamps(target)
        initialization = timestamp("2026-01-01T12:00:00+00:00")
        incomplete = make_forecast(
            initialization,
            timestamp("2026-01-01T19:00:00+00:00"),
            hours[:-1],
            base=1,
        )
        first_complete = make_forecast(
            initialization,
            timestamp("2026-01-01T20:00:00+00:00"),
            hours,
            base=10,
        )
        later_complete = make_forecast(
            initialization,
            timestamp("2026-01-01T21:00:00+00:00"),
            hours,
            base=100,
        )

        selected = _select_daily_max_forecast(
            [later_complete, incomplete, first_complete],
            model_name=MODEL,
            target_date=target,
            city_timezone=ZONE,
            artifact_groups={
                ("12", 2): make_group(("12", 2), {"20260103"})
            },
            as_of=datetime(2026, 1, 2, 12, tzinfo=timezone.utc),
            expected_interval_seconds=3600,
            minimum_notice_hours=0,
        )

        self.assertEqual(
            selected.availability_time,
            timestamp("2026-01-01T20:00:00+00:00"),
        )
        self.assertAlmostEqual(selected.member_maxima[0], 12.3)
        self.assertAlmostEqual(selected.member_maxima[1], 13.3)

    def test_day_ahead_uses_initialization_local_date(self):
        target = date(2026, 1, 3)
        hours = local_day_timestamps(target)
        # 18Z on January 1 is already January 2 in Asia/Shanghai.
        forecast = make_forecast(
            timestamp("2026-01-01T18:00:00+00:00"),
            timestamp("2026-01-02T02:00:00+00:00"),
            hours,
        )

        selected = _select_daily_max_forecast(
            [forecast],
            model_name=MODEL,
            target_date=target,
            city_timezone=ZONE,
            artifact_groups={
                ("18", 1): make_group(("18", 1), {"20260103"})
            },
            as_of=datetime(2026, 1, 2, 12, tzinfo=timezone.utc),
            expected_interval_seconds=3600,
            minimum_notice_hours=0,
        )

        self.assertEqual(selected.group.key, ("18", 1))

    def test_as_of_excludes_forecast_that_was_not_yet_available(self):
        target = date(2026, 1, 3)
        hours = local_day_timestamps(target)
        older = make_forecast(
            timestamp("2026-01-01T12:00:00+00:00"),
            timestamp("2026-01-01T20:00:00+00:00"),
            hours,
        )
        newer = make_forecast(
            timestamp("2026-01-02T00:00:00+00:00"),
            timestamp("2026-01-02T11:00:00+00:00"),
            hours,
        )

        selected = _select_daily_max_forecast(
            [older, newer],
            model_name=MODEL,
            target_date=target,
            city_timezone=ZONE,
            artifact_groups={
                ("12", 2): make_group(("12", 2), {"20260103"}),
                ("00", 1): make_group(("00", 1), {"20260103"}),
            },
            as_of=datetime(2026, 1, 2, 10, tzinfo=timezone.utc),
            expected_interval_seconds=3600,
            minimum_notice_hours=0,
        )

        self.assertEqual(selected.group.key, ("12", 2))

    def test_member_schema_must_match_fit_group(self):
        target = date(2026, 1, 3)
        forecast = make_forecast(
            timestamp("2026-01-01T12:00:00+00:00"),
            timestamp("2026-01-01T20:00:00+00:00"),
            local_day_timestamps(target),
        )
        mismatched = _ArtifactGroup(
            key=("12", 2),
            fit=sentinel.fit,
            modeled_dates=frozenset({"20260103"}),
            member_names=("temperature_2m",),
            timezone_name=ZONE.key,
        )

        with self.assertRaisesRegex(ValueError, "ensemble members"):
            _select_daily_max_forecast(
                [forecast],
                model_name=MODEL,
                target_date=target,
                city_timezone=ZONE,
                artifact_groups={("12", 2): mismatched},
                as_of=datetime(2026, 1, 2, 12, tzinfo=timezone.utc),
                expected_interval_seconds=3600,
                minimum_notice_hours=0,
            )


class DailyMaxTemperatureProbabilityTest(unittest.TestCase):
    def test_prediction_ensemble_data_converts_maxima_to_kelvin(self):
        case = _DailyMaxPredictionCase(
            group=make_group(("12", 2), {"20260103"}),
            initialization_time=timestamp("2026-01-01T12:00:00+00:00"),
            availability_time=timestamp("2026-01-01T20:00:00+00:00"),
            member_maxima=(30.0, 31.0),
        )

        ensemble_data = _build_daily_max_prediction_ensemble_data(
            case,
            date(2026, 1, 3),
            forecast_input_unit="celsius",
            exchangeable=True,
        )

        from rpy2.robjects.packages import importr

        ensemble_bma = importr("ensembleBMA")
        self.assertEqual(
            list(ensemble_bma.ensembleForecasts(ensemble_data)),
            [303.15, 304.15],
        )
        self.assertEqual(
            list(ensemble_bma.ensembleValidDates(ensemble_data)),
            ["20260103"],
        )
        self.assertEqual(list(ensemble_bma.ensembleFhour(ensemble_data)), [48])
        self.assertEqual(list(ensemble_bma.ensembleItime(ensemble_data)), ["12"])
        self.assertEqual(
            list(ensemble_bma.ensembleGroups(ensemble_data)),
            ["control", "perturbed"],
        )

    def test_cdf_result_is_probability_below_not_its_complement(self):
        with patch("rpy2.robjects.packages.importr") as importr:
            importr.return_value.cdf.return_value = [0.23]

            probability = _call_ensemble_mos_cdf(
                sentinel.fit,
                sentinel.ensemble_data,
                303.15,
                date(2026, 1, 3),
            )

        self.assertEqual(probability, 0.23)
        cdf_call = importr.return_value.cdf.call_args
        self.assertEqual(cdf_call.args, (sentinel.fit, sentinel.ensemble_data))
        self.assertAlmostEqual(float(cdf_call.kwargs["values"][0]), 303.15)
        self.assertEqual(list(cdf_call.kwargs["dates"]), ["20260103"])

    def test_cdf_rejects_non_finite_probability(self):
        with patch("rpy2.robjects.packages.importr") as importr:
            importr.return_value.cdf.return_value = [math.nan]
            with self.assertRaisesRegex(ValueError, "invalid probability"):
                _call_ensemble_mos_cdf(
                    sentinel.fit,
                    sentinel.ensemble_data,
                    303.15,
                    date(2026, 1, 3),
                )

    def test_public_function_matches_paths_date_group_and_threshold_unit(self):
        target = date(2026, 1, 3)
        forecast = make_forecast(
            timestamp("2026-01-01T12:00:00+00:00"),
            timestamp("2026-01-01T20:00:00+00:00"),
            local_day_timestamps(target),
        )
        artifact = make_artifact()

        with (
            patch(
                "predict_emos_max_temperature."
                "load_daily_max_temperature_emos_fits",
                return_value=artifact,
            ) as load_artifact,
            patch(
                "predict_emos_max_temperature._load_forecasts",
                return_value=[forecast],
            ) as load_forecasts,
            patch(
                "predict_emos_max_temperature."
                "_build_daily_max_prediction_ensemble_data",
                return_value=sentinel.ensemble_data,
            ) as build_data,
            patch(
                "predict_emos_max_temperature._call_ensemble_mos_cdf",
                return_value=0.23,
            ) as cdf,
        ):
            probability = probability_daily_max_temperature_below(
                CITY,
                MODEL,
                "2026-01-03",
                30.0,
                as_of=datetime(2026, 1, 2, 12, tzinfo=timezone.utc),
                data_dir=Path("/forecast-data"),
                artifact_dir=Path("/emos-train"),
                artifact_version="version-1",
            )

        self.assertEqual(probability, 0.23)
        load_artifact.assert_called_once_with(
            CITY,
            MODEL,
            version="version-1",
            output_dir=Path("/emos-train"),
            verify_checksums=True,
        )
        load_forecasts.assert_called_once_with(
            CITY,
            MODEL,
            data_dir=Path("/forecast-data"),
        )
        selected_case = build_data.call_args.args[0]
        self.assertEqual(selected_case.group.key, ("12", 2))
        self.assertEqual(
            build_data.call_args.kwargs["forecast_input_unit"],
            "celsius",
        )
        self.assertAlmostEqual(cdf.call_args.args[2], 303.15)
        self.assertEqual(cdf.call_args.args[3], target)

    def test_rejects_artifact_created_after_as_of(self):
        artifact = make_artifact(
            training_completed_at="2026-01-02T13:00:00Z"
        )
        with (
            patch(
                "predict_emos_max_temperature."
                "load_daily_max_temperature_emos_fits",
                return_value=artifact,
            ),
            patch("predict_emos_max_temperature._load_forecasts") as loader,
        ):
            with self.assertRaisesRegex(ValueError, "not available at"):
                probability_daily_max_temperature_below(
                    CITY,
                    MODEL,
                    "2026-01-03",
                    30.0,
                    as_of=datetime(2026, 1, 2, 12, tzinfo=timezone.utc),
                    artifact_version="latest",
                )

        loader.assert_not_called()

    def test_auto_version_falls_back_to_version_with_usable_forecast_group(self):
        target = date(2026, 1, 3)
        forecast = make_forecast(
            timestamp("2026-01-01T12:00:00+00:00"),
            timestamp("2026-01-01T20:00:00+00:00"),
            local_day_timestamps(target),
        )
        newer_artifact = make_artifact(key=("00", 1))
        older_artifact = make_artifact(key=("12", 2))

        with (
            patch(
                "predict_emos_max_temperature._automatic_artifact_versions",
                return_value=("newer", "older"),
            ),
            patch(
                "predict_emos_max_temperature."
                "load_daily_max_temperature_emos_fits",
                side_effect=(newer_artifact, older_artifact),
            ) as load_artifact,
            patch(
                "predict_emos_max_temperature._load_forecasts",
                return_value=[forecast],
            ),
            patch(
                "predict_emos_max_temperature."
                "_build_daily_max_prediction_ensemble_data",
                return_value=sentinel.ensemble_data,
            ),
            patch(
                "predict_emos_max_temperature._call_ensemble_mos_cdf",
                return_value=0.4,
            ),
        ):
            probability = probability_daily_max_temperature_below(
                CITY,
                MODEL,
                target,
                30,
                as_of=datetime(2026, 1, 2, 12, tzinfo=timezone.utc),
            )

        self.assertEqual(probability, 0.4)
        self.assertEqual(
            [
                call.kwargs["version"]
                for call in load_artifact.call_args_list
            ],
            ["newer", "older"],
        )

    def test_artifact_group_index_requires_exact_modeled_date_metadata(self):
        artifact = make_artifact(modeled_date="20260103")

        groups = _artifact_groups(artifact)

        self.assertEqual(set(groups), {("12", 2)})
        self.assertEqual(groups[("12", 2)].modeled_dates, {"20260103"})


class DailyMaxTemperatureIntervalProbabilityTest(unittest.TestCase):
    def test_interval_probabilities_are_adjacent_cdf_differences(self):
        intervals = _interval_probabilities_from_cdf(
            (38.0, 39.0, 40.0),
            (0.1, 0.4, 0.9),
            "celsius",
        )

        self.assertEqual(
            [
                (interval.lower_bound, interval.upper_bound)
                for interval in intervals
            ],
            [
                (None, 38.0),
                (38.0, 39.0),
                (39.0, 40.0),
                (40.0, None),
            ],
        )
        self.assertEqual(
            [interval.label for interval in intervals],
            [
                "T < 38°C",
                "38°C <= T < 39°C",
                "39°C <= T < 40°C",
                "40°C <= T",
            ],
        )
        self.assertAlmostEqual(intervals[0].probability, 0.1)
        self.assertAlmostEqual(intervals[1].probability, 0.3)
        self.assertAlmostEqual(intervals[2].probability, 0.5)
        self.assertAlmostEqual(intervals[3].probability, 0.1)
        self.assertAlmostEqual(
            sum(interval.probability for interval in intervals),
            1.0,
        )

    def test_interval_boundaries_must_be_finite_and_strictly_increasing(self):
        invalid_boundaries = (
            (),
            (38, 38),
            (39, 38),
            (38, math.nan),
            (38, math.inf),
            (38, True),
        )
        for boundaries in invalid_boundaries:
            with self.subTest(boundaries=boundaries):
                with self.assertRaises(ValueError):
                    _validate_interval_boundaries(boundaries, "celsius")

    def test_vector_cdf_passes_all_thresholds_in_one_r_call(self):
        with patch("rpy2.robjects.packages.importr") as importr:
            importr.return_value.cdf.return_value = [0.1, 0.4, 0.9]

            probabilities = _call_ensemble_mos_cdf_values(
                sentinel.fit,
                sentinel.ensemble_data,
                (311.15, 312.15, 313.15),
                date(2026, 1, 3),
            )

        self.assertEqual(probabilities, (0.1, 0.4, 0.9))
        cdf = importr.return_value.cdf
        cdf.assert_called_once()
        self.assertEqual(
            list(cdf.call_args.kwargs["values"]),
            [311.15, 312.15, 313.15],
        )
        self.assertEqual(list(cdf.call_args.kwargs["dates"]), ["20260103"])

    def test_three_days_are_based_on_one_as_of_in_city_timezone(self):
        as_of = datetime(2026, 7, 30, 15, 59, tzinfo=timezone.utc)
        with patch(
            "predict_emos_max_temperature."
            "_probabilities_daily_max_temperature_below_thresholds",
            return_value=(0.1, 0.4, 0.9),
        ) as daily_probabilities:
            predictions = predict_daily_max_temperature_intervals(
                CITY,
                MODEL,
                (38, 39, 40),
                as_of=as_of,
                days=3,
            )

        self.assertEqual(
            [prediction.target_date for prediction in predictions],
            [
                date(2026, 7, 30),
                date(2026, 7, 31),
                date(2026, 8, 1),
            ],
        )
        self.assertTrue(all(prediction.available for prediction in predictions))
        self.assertEqual(daily_probabilities.call_count, 3)
        self.assertEqual(
            [call.args[2] for call in daily_probabilities.call_args_list],
            [prediction.target_date for prediction in predictions],
        )
        self.assertTrue(
            all(
                call.kwargs["as_of"] == as_of
                for call in daily_probabilities.call_args_list
            )
        )

    def test_three_day_strict_mode_identifies_unavailable_date(self):
        with patch(
            "predict_emos_max_temperature."
            "_probabilities_daily_max_temperature_below_thresholds",
            side_effect=(
                (0.1, 0.4, 0.9),
                (0.2, 0.5, 0.8),
                _PredictionUnavailableError("missing modeled date"),
            ),
        ):
            with self.assertRaisesRegex(
                ValueError,
                "cannot predict local date 2026-08-01",
            ):
                predict_daily_max_temperature_intervals(
                    CITY,
                    MODEL,
                    (38, 39, 40),
                    start_date="2026-07-30",
                    days=3,
                )

    def test_three_day_partial_mode_marks_unavailable_date(self):
        with patch(
            "predict_emos_max_temperature."
            "_probabilities_daily_max_temperature_below_thresholds",
            side_effect=(
                (0.1, 0.4, 0.9),
                (0.2, 0.5, 0.8),
                _PredictionUnavailableError("missing modeled date"),
            ),
        ):
            predictions = predict_daily_max_temperature_intervals(
                CITY,
                MODEL,
                (38, 39, 40),
                start_date="2026-07-30",
                days=3,
                allow_partial=True,
            )

        self.assertEqual(len(predictions), 3)
        self.assertTrue(predictions[0].available)
        self.assertTrue(predictions[1].available)
        self.assertFalse(predictions[2].available)
        self.assertEqual(predictions[2].intervals, ())
        self.assertEqual(
            predictions[2].unavailable_reason,
            "missing modeled date",
        )


class DailyMaxTemperatureRealDataTest(unittest.TestCase):
    def test_predicts_two_chongqing_aifs_days_and_prints_probabilities(self):
        """Run the saved Chongqing AIFS fit for boundaries from 32 to 43°C."""
        project_directory = Path(__file__).resolve().parents[1]
        data_directory = project_directory / "data"
        artifact_directory = (
            project_directory / "train" / "highest_temperature_emos"
        )
        artifact_version = "20260730T121809.496761Z-e10ab9007155"
        forecast_path = (
            data_directory
            / "forecast"
            / CITY
            / MODEL
            / "fc.jsonl"
        )
        version_directory = (
            artifact_directory / CITY / MODEL / artifact_version
        )
        if (
            not forecast_path.is_file()
            or not (version_directory / "manifest.json").is_file()
            or not (version_directory / "fits").is_dir()
        ):
            self.skipTest(
                "real Chongqing forecast data and trained artifact are required"
            )

        boundaries = tuple(range(32, 44))
        predictions = predict_daily_max_temperature_intervals(
            CITY,
            MODEL,
            boundaries,
            start_date="2026-07-30",
            days=2,
            threshold_unit="celsius",
            city_timezone=ZONE,
            as_of=datetime(2026, 7, 30, 12, 30, tzinfo=timezone.utc),
            data_dir=data_directory,
            artifact_dir=artifact_directory,
            artifact_version=artifact_version,
            verify_checksums=True,
        )

        self.assertEqual(
            [prediction.target_date for prediction in predictions],
            [date(2026, 7, 30), date(2026, 7, 31)],
        )
        for prediction in predictions:
            self.assertTrue(prediction.available)
            self.assertEqual(len(prediction.intervals), len(boundaries) + 1)
            probabilities = [
                interval.probability for interval in prediction.intervals
            ]
            self.assertTrue(
                all(0.0 <= probability <= 1.0 for probability in probabilities)
            )
            self.assertAlmostEqual(sum(probabilities), 1.0)

            print(
                f"\nChongqing AIFS daily Tmax probabilities "
                f"({prediction.target_date}):"
            )
            for interval in prediction.intervals:
                print(f"  {interval.label}: {interval.probability:.6%}")


if __name__ == "__main__":
    unittest.main()

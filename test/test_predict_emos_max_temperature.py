"""Tests for daily maximum-temperature EMOS prediction."""

import json
import math
import unittest
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch, sentinel
from zoneinfo import ZoneInfo

from predict_emos_max_temperature import (
    _ArtifactGroup,
    _DailyMaxPredictionCase,
    _PredictionUnavailableError,
    _append_prediction_record,
    _artifact_groups,
    _build_daily_max_prediction_ensemble_data,
    _build_prediction_record,
    _call_ensemble_mos_cdf,
    _call_ensemble_mos_cdf_values,
    _emos_parameters_for_date,
    _interval_probabilities_from_cdf,
    _select_daily_max_forecast,
    _validate_interval_boundaries,
    DailyMaxTemperatureIntervalPrediction,
    predict_all_configured_daily_max_temperature_intervals,
    predict_daily_max_temperature_intervals,
    probability_daily_max_temperature_below,
)
from polymarket import PolymarketAPIError
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
        "fit_file": f"fits/init_{key[0]}_day_{key[1]:03d}.rds",
        "fit_sha256": "a" * 64,
        "resolved_training_days": 14,
        "sample_count": 20,
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


def make_persistable_prediction(
    target_date: date,
    boundaries: tuple[float, ...],
    *,
    model_name: str = MODEL,
    artifact_version: str = "artifact-v1",
) -> DailyMaxTemperatureIntervalPrediction:
    cdf_values = tuple(
        (index + 1) / (len(boundaries) + 1)
        for index in range(len(boundaries))
    )
    intervals = _interval_probabilities_from_cdf(
        boundaries,
        cdf_values,
        "celsius",
    )
    initialization_time = timestamp("2026-07-29T12:00:00+00:00")
    availability_time = timestamp("2026-07-29T20:00:00+00:00")
    return DailyMaxTemperatureIntervalPrediction(
        target_date=target_date,
        intervals=intervals,
        provenance={
            "city_timezone": ZONE.key,
            "forecast": {
                "initialization_time_unix": initialization_time,
                "initialization_time_utc": "2026-07-29T12:00:00.000000Z",
                "availability_time_unix": availability_time,
                "availability_time_utc": "2026-07-29T20:00:00.000000Z",
                "initialization_hour_utc": "12",
                "day_ahead": 2,
                "meta": {
                    "last_run_initialisation_time": initialization_time,
                    "last_run_availability_time": availability_time,
                },
                "member_names": list(MEMBERS),
                "daily_member_maxima": [39.0, 40.0],
                "input_unit": "celsius",
                "predictor_sha256": "b" * 64,
            },
            "emos_artifact": {
                "version": artifact_version,
                "path": (
                    f"train/highest_temperature_emos/{CITY}/"
                    f"{model_name}/{artifact_version}"
                ),
                "training_completed_at_utc": "2026-07-30T00:00:00Z",
                "saved_at_utc": "2026-07-30T00:00:01Z",
                "parameter_date": target_date.strftime("%Y%m%d"),
                "correction_parameters": {
                    "model": "normal",
                    "parameter_date": target_date.strftime("%Y%m%d"),
                    "a": 1.0,
                    "B_by_member": {
                        MEMBERS[0]: 0.6,
                        MEMBERS[1]: 0.4,
                    },
                    "c": 2.0,
                    "d": 0.5,
                },
                "group": {
                    "initialization_hour_utc": "12",
                    "day_ahead": 2,
                    "forecast_hour": 48,
                    "fit_file": "fits/init_12_day_002.rds",
                    "fit_sha256": "a" * 64,
                    "resolved_training_days": 14,
                    "sample_count": 20,
                },
            },
            "options": {
                "expected_interval_seconds": 3600,
                "minimum_notice_hours": 0.0,
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

    def test_interval_prediction_includes_selected_forecast_and_fit_provenance(self):
        target = date(2026, 1, 3)
        initialization_time = timestamp("2026-01-01T12:00:00+00:00")
        availability_time = timestamp("2026-01-01T20:00:00+00:00")
        forecast = make_forecast(
            initialization_time,
            availability_time,
            local_day_timestamps(target),
        )
        artifact = make_artifact()

        with (
            patch(
                "predict_emos_max_temperature."
                "load_daily_max_temperature_emos_fits",
                return_value=artifact,
            ),
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
                "predict_emos_max_temperature."
                "_call_ensemble_mos_cdf_values",
                return_value=(0.25, 0.75),
            ),
            patch(
                "predict_emos_max_temperature."
                "_emos_parameters_for_date",
                return_value={
                    "model": "normal",
                    "parameter_date": "20260103",
                    "a": 1.0,
                    "B_by_member": {
                        MEMBERS[0]: 0.6,
                        MEMBERS[1]: 0.4,
                    },
                    "c": 2.0,
                    "d": 0.5,
                },
            ),
        ):
            predictions = predict_daily_max_temperature_intervals(
                CITY,
                MODEL,
                (30.0, 31.0),
                start_date=target,
                days=1,
                as_of=datetime(2026, 1, 2, 12, tzinfo=timezone.utc),
                artifact_version="version-1",
                include_provenance=True,
            )

        provenance = predictions[0].provenance
        self.assertIsInstance(provenance, dict)
        self.assertEqual(
            provenance["forecast"]["initialization_time_unix"],
            initialization_time,
        )
        self.assertEqual(
            provenance["forecast"]["availability_time_unix"],
            availability_time,
        )
        self.assertEqual(
            provenance["forecast"]["meta"],
            forecast["meta"],
        )
        self.assertEqual(
            provenance["emos_artifact"]["version"],
            "test-version",
        )
        self.assertEqual(
            provenance["emos_artifact"]["group"]["fit_sha256"],
            "a" * 64,
        )
        self.assertEqual(
            provenance["emos_artifact"]["parameter_date"],
            "20260103",
        )
        self.assertEqual(
            provenance["emos_artifact"]["correction_parameters"]["a"],
            1.0,
        )

    def test_extracts_exact_emos_parameter_column_for_target_date(self):
        class Matrix(list):
            def __init__(
                self,
                values,
                dimensions,
                column_names,
                row_names,
            ):
                super().__init__(values)
                self.dim = dimensions
                self.colnames = column_names
                self.rownames = row_names

        class Fit:
            parameters = {
                "a": Matrix(
                    [1.0, 2.0],
                    (1, 2),
                    ("20260102", "20260103"),
                    ("a",),
                ),
                "B": Matrix(
                    [0.1, 0.2, 0.3, 0.4],
                    (2, 2),
                    ("20260102", "20260103"),
                    MEMBERS,
                ),
                "c": Matrix(
                    [3.0, 4.0],
                    (1, 2),
                    ("20260102", "20260103"),
                    ("c",),
                ),
                "d": Matrix(
                    [0.5, 0.6],
                    (1, 2),
                    ("20260102", "20260103"),
                    ("d",),
                ),
            }

            def rx2(self, name):
                return self.parameters[name]

        parameters = _emos_parameters_for_date(
            Fit(),
            date(2026, 1, 3),
            MEMBERS,
        )

        self.assertEqual(
            parameters,
            {
                "model": "normal",
                "parameter_date": "20260103",
                "a": 2.0,
                "B_by_member": {
                    MEMBERS[0]: 0.3,
                    MEMBERS[1]: 0.4,
                },
                "c": 4.0,
                "d": 0.6,
            },
        )

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


class DailyMaxTemperaturePredictionPersistenceTest(unittest.TestCase):
    def test_prediction_record_append_is_idempotent_and_versioned(self):
        target = date(2026, 7, 31)
        boundaries = (38.0, 39.0)
        first_prediction = make_persistable_prediction(
            target,
            boundaries,
        )
        first_record = _build_prediction_record(
            CITY,
            MODEL,
            "highest-temperature-in-chongqing",
            "highest-temperature-in-chongqing-on-july-31-2026",
            boundaries,
            first_prediction,
            generated_at=datetime(
                2026,
                7,
                30,
                12,
                tzinfo=timezone.utc,
            ),
            as_of=datetime(
                2026,
                7,
                30,
                11,
                tzinfo=timezone.utc,
            ),
            market_fetched_at=datetime(
                2026,
                7,
                30,
                11,
                30,
                tzinfo=timezone.utc,
            ),
        )
        second_record = _build_prediction_record(
            CITY,
            MODEL,
            "highest-temperature-in-chongqing",
            "highest-temperature-in-chongqing-on-july-31-2026",
            boundaries,
            make_persistable_prediction(
                target,
                boundaries,
                artifact_version="artifact-v2",
            ),
            generated_at=datetime(
                2026,
                7,
                30,
                13,
                tzinfo=timezone.utc,
            ),
            as_of=datetime(
                2026,
                7,
                30,
                12,
                tzinfo=timezone.utc,
            ),
            market_fetched_at=datetime(
                2026,
                7,
                30,
                12,
                30,
                tzinfo=timezone.utc,
            ),
        )
        changed_prediction = DailyMaxTemperatureIntervalPrediction(
            target_date=target,
            intervals=_interval_probabilities_from_cdf(
                boundaries,
                (0.2, 0.8),
                "celsius",
            ),
            provenance=first_prediction.provenance,
        )
        changed_record = _build_prediction_record(
            CITY,
            MODEL,
            "highest-temperature-in-chongqing",
            "highest-temperature-in-chongqing-on-july-31-2026",
            boundaries,
            changed_prediction,
            generated_at=datetime(
                2026,
                7,
                30,
                14,
                tzinfo=timezone.utc,
            ),
            as_of=datetime(
                2026,
                7,
                30,
                13,
                tzinfo=timezone.utc,
            ),
            market_fetched_at=datetime(
                2026,
                7,
                30,
                13,
                30,
                tzinfo=timezone.utc,
            ),
        )
        relocated_provenance = json.loads(
            json.dumps(first_prediction.provenance)
        )
        relocated_provenance["emos_artifact"]["path"] = (
            "/absolute/elsewhere/artifact-v1"
        )
        relocated_record = _build_prediction_record(
            CITY,
            MODEL,
            "highest-temperature-in-chongqing",
            "highest-temperature-in-chongqing-on-july-31-2026",
            boundaries,
            DailyMaxTemperatureIntervalPrediction(
                target_date=target,
                intervals=first_prediction.intervals,
                provenance=relocated_provenance,
            ),
            generated_at=datetime(
                2026,
                7,
                30,
                15,
                tzinfo=timezone.utc,
            ),
            as_of=datetime(
                2026,
                7,
                30,
                14,
                tzinfo=timezone.utc,
            ),
            market_fetched_at=datetime(
                2026,
                7,
                30,
                14,
                30,
                tzinfo=timezone.utc,
            ),
        )

        with TemporaryDirectory() as output_dir:
            first_path, first_appended = _append_prediction_record(
                first_record,
                output_dir=output_dir,
            )
            duplicate_path, duplicate_appended = _append_prediction_record(
                first_record,
                output_dir=output_dir,
            )
            _, relocated_appended = _append_prediction_record(
                relocated_record,
                output_dir=output_dir,
            )
            _, changed_appended = _append_prediction_record(
                changed_record,
                output_dir=output_dir,
            )
            second_path, second_appended = _append_prediction_record(
                second_record,
                output_dir=output_dir,
            )

            with first_path.open("r", encoding="utf-8") as input_file:
                saved_records = [
                    json.loads(line)
                    for line in input_file
                    if line.strip()
                ]

        self.assertTrue(first_appended)
        self.assertFalse(duplicate_appended)
        self.assertFalse(relocated_appended)
        self.assertTrue(changed_appended)
        self.assertTrue(second_appended)
        self.assertEqual(first_path, duplicate_path)
        self.assertEqual(first_path, second_path)
        self.assertEqual(len(saved_records), 3)
        self.assertEqual(
            first_record["prediction_id"],
            relocated_record["prediction_id"],
        )
        self.assertNotEqual(
            first_record["prediction_id"],
            changed_record["prediction_id"],
        )
        self.assertNotEqual(
            first_record["prediction_id"],
            second_record["prediction_id"],
        )
        self.assertEqual(
            saved_records[0]["forecast"]["initialization_time_unix"],
            timestamp("2026-07-29T12:00:00+00:00"),
        )
        self.assertEqual(
            saved_records[0]["emos_artifact"]["group"]["fit_sha256"],
            "a" * 64,
        )
        self.assertEqual(
            saved_records[0]["emos_artifact"]["correction_parameters"]["a"],
            1.0,
        )
        self.assertEqual(saved_records[0]["algorithm_version"], 1)
        self.assertEqual(
            saved_records[0]["forecast_artifact_as_of_utc"],
            "2026-07-30T11:00:00.000000Z",
        )
        self.assertEqual(
            saved_records[0]["market"]["fetched_at_utc"],
            "2026-07-30T11:30:00.000000Z",
        )
        self.assertEqual(
            [interval["label"] for interval in saved_records[0]["intervals"]],
            [
                "T < 38°C",
                "38°C <= T < 39°C",
                "39°C <= T",
            ],
        )

    def test_configured_batch_reuses_boundaries_skips_404_and_deduplicates(self):
        cities = [
            {
                "name": CITY,
                "timezone": ZONE.key,
                "slug_prefix": "highest-temperature-in-chongqing",
                "models": [
                    {"name": "model-a"},
                    {"name": "model-b"},
                ],
            }
        ]
        first_boundaries = (38.0, 39.0)
        third_boundaries = (36.0, 37.0, 38.0)
        missing_market = PolymarketAPIError(
            "event not found",
            status_code=404,
        )

        def predict(city_name, model_name, boundaries, **options):
            return (
                make_persistable_prediction(
                    options["start_date"],
                    tuple(boundaries),
                    model_name=model_name,
                ),
            )

        with TemporaryDirectory() as output_dir:
            with (
                patch(
                    "predict_emos_max_temperature."
                    "get_daily_max_temperature_boundaries",
                    side_effect=[
                        first_boundaries,
                        missing_market,
                        third_boundaries,
                        first_boundaries,
                        missing_market,
                        third_boundaries,
                    ],
                ) as get_boundaries,
                patch(
                    "predict_emos_max_temperature."
                    "predict_daily_max_temperature_intervals",
                    side_effect=predict,
                ) as predict_intervals,
                self.assertLogs(
                    "predict_emos_max_temperature",
                    level="WARNING",
                ) as warning_logs,
            ):
                first_writes = (
                    predict_all_configured_daily_max_temperature_intervals(
                        cities=cities,
                        predict_days=3,
                        as_of=datetime(
                            2026,
                            7,
                            30,
                            16,
                            tzinfo=timezone.utc,
                        ),
                        output_dir=output_dir,
                    )
                )
                duplicate_writes = (
                    predict_all_configured_daily_max_temperature_intervals(
                        cities=cities,
                        predict_days=3,
                        as_of=datetime(
                            2026,
                            7,
                            30,
                            16,
                            tzinfo=timezone.utc,
                        ),
                        output_dir=output_dir,
                    )
                )

            saved_by_model = {}
            for model_name in ("model-a", "model-b"):
                path = (
                    Path(output_dir)
                    / CITY
                    / model_name
                    / "predictions.jsonl"
                )
                with path.open("r", encoding="utf-8") as input_file:
                    saved_by_model[model_name] = [
                        json.loads(line)
                        for line in input_file
                        if line.strip()
                    ]

        self.assertEqual(get_boundaries.call_count, 6)
        self.assertEqual(
            [
                boundary_call.args[1]
                for boundary_call in get_boundaries.call_args_list[:3]
            ],
            [
                date(2026, 7, 31),
                date(2026, 8, 1),
                date(2026, 8, 2),
            ],
        )
        self.assertEqual(predict_intervals.call_count, 8)
        self.assertTrue(
            all(
                prediction_call.kwargs["days"] == 1
                and prediction_call.kwargs["include_provenance"] is True
                for prediction_call in predict_intervals.call_args_list
            )
        )
        self.assertEqual(len(first_writes), 4)
        self.assertTrue(all(write.appended for write in first_writes))
        self.assertEqual(len(duplicate_writes), 4)
        self.assertTrue(
            all(not write.appended for write in duplicate_writes)
        )
        self.assertIn(
            "No Polymarket market for Chongqing-ZUCK on 2026-08-01",
            "\n".join(warning_logs.output),
        )
        for records in saved_by_model.values():
            self.assertEqual(len(records), 2)
            self.assertEqual(
                [
                    record["target_date_local"]
                    for record in records
                ],
                ["2026-07-31", "2026-08-02"],
            )
            self.assertEqual(
                records[0]["market"]["boundaries"],
                [38.0, 39.0],
            )
            self.assertEqual(
                records[1]["market"]["boundaries"],
                [36.0, 37.0, 38.0],
            )

    def test_configured_batch_does_not_hide_prediction_data_errors(self):
        cities = [
            {
                "name": CITY,
                "timezone": ZONE.key,
                "slug_prefix": "highest-temperature-in-chongqing",
                "models": [{"name": MODEL}],
            }
        ]
        with (
            patch(
                "predict_emos_max_temperature."
                "get_daily_max_temperature_boundaries",
                return_value=(38.0, 39.0),
            ),
            patch(
                "predict_emos_max_temperature."
                "predict_daily_max_temperature_intervals",
                side_effect=ValueError("corrupt EMOS artifact"),
            ),
        ):
            with self.assertRaisesRegex(
                ValueError,
                "corrupt EMOS artifact",
            ):
                predict_all_configured_daily_max_temperature_intervals(
                    cities=cities,
                    predict_days=1,
                    as_of=datetime(
                        2026,
                        7,
                        30,
                        16,
                        tzinfo=timezone.utc,
                    ),
                    output_dir="/unused",
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

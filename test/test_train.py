"""Integration tests for the Python-to-R EMOS bridge."""

import math
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from rpy2 import robjects
from rpy2.robjects import vectors
from rpy2.robjects.packages import importr

from train import (
    build_temperature_ensemble_data,
    ensemble_mos,
    group_emos_training_data,
    match_forecast,
    train_grouped_ensemble_mos,
)


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
        "valid_times": valid_times,
        "forecasts": forecasts,
        "observations": observations,
    }


class EnsembleMosTest(unittest.TestCase):
    def test_build_temperature_ensemble_data_converts_celsius_to_kelvin(self):
        timestamp = int(
            datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp()
        )
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

        result = ensemble_mos(
            ensemble_data,
            training_days=14,
            dates=vectors.StrVector((dates[-1],)),
        )

        result_classes = set(robjects.r["class"](result))  # type: ignore
        self.assertIn("ensembleMOSnormal", result_classes)
        self.assertTrue({"training", "a", "B", "c", "d"}.issubset(result.names))

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

    def test_train_grouped_ensemble_mos_adapts_days_for_each_group(self):
        group_24h = make_mock_group(number_of_days=5)
        group_48h = make_mock_group(number_of_days=8)
        group_48h["forecast_hour"] = 48
        groups = {("00", 24): group_24h, ("00", 48): group_48h}

        with patch("train.ensemble_mos", side_effect=("fit24", "fit48")) as fit:
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


if __name__ == "__main__":
    unittest.main()

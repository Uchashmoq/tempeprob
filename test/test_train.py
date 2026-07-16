"""Integration tests for the Python-to-R EMOS bridge."""

import math
import unittest
from datetime import datetime, timedelta

from rpy2 import robjects
from rpy2.robjects import vectors
from rpy2.robjects.packages import importr

from train import ensemble_mos


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


class EnsembleMosTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()

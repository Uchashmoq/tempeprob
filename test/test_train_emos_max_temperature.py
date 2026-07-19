"""Tests for daily maximum-temperature EMOS case preparation."""

import json
import math
import unittest
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

from train_emos_max_temperature import (
    _load_forecasts,
    _load_temperatures,
    build_daily_max_temperature_ensemble_data,
    group_daily_max_temperature_emos_training_data,
    train_daily_max_temperature_emos,
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
            ("00", 1): make_daily_max_group(5),
            ("06", 2): make_daily_max_group(
                8,
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
            [5, 8],
        )
        self.assertTrue(
            all(
                call.kwargs["consecutive"] is False
                for call in fit.call_args_list
            )
        )

    def test_train_daily_max_emos_rejects_insufficient_groups(self):
        groups = {("00", 1): make_daily_max_group(5)}

        with self.assertRaisesRegex(ValueError, "need 6 dates"):
            train_daily_max_temperature_emos(groups, training_days=6)

        self.assertEqual(
            train_daily_max_temperature_emos(
                groups,
                training_days=6,
                skip_insufficient=True,
            ),
            {},
        )

    def test_train_daily_max_emos_runs_gaussian_r_fit(self):
        from rpy2 import robjects

        groups = {("00", 1): make_daily_max_group(20)}

        fits = train_daily_max_temperature_emos(groups, training_days=14)

        fit = fits[("00", 1)]
        self.assertIn("ensembleMOSnormal", set(robjects.r["class"](fit)))
        self.assertTrue({"training", "a", "B", "c", "d"}.issubset(fit.names))
        for parameter_name in ("a", "B", "c", "d"):
            self.assertTrue(
                all(math.isfinite(float(value)) for value in fit.rx2(parameter_name))
            )

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
            sorted(
                item["meta"]["last_run_initialisation_time"]
                for item in forecasts
            ),
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

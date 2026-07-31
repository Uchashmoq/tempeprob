"""Tests for the temperature continuity checker script."""

import json
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest


PROJECT_DIR = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_DIR / "script" / "check_temperature_continuity.py"


def write_observations(
    root: Path,
    city_name: str,
    timestamps: list[int],
) -> None:
    path = root / city_name / "tem.jsonl"
    path.parent.mkdir(parents=True)
    with path.open("w", encoding="utf-8") as output_file:
        for timestamp in timestamps:
            json.dump(
                {
                    "time": timestamp,
                    "temperature": 20.0,
                    "update_time": timestamp + 60,
                },
                output_file,
            )
            output_file.write("\n")


class TemperatureContinuityScriptTest(unittest.TestCase):
    def run_script(self, temperature_dir: Path, *arguments: str):
        return subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--temperature-dir",
                str(temperature_dir),
                *arguments,
            ],
            check=False,
            capture_output=True,
            text=True,
        )

    def test_allows_multiple_observations_in_each_hour(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            write_observations(
                root,
                "HalfHourly-AAAA",
                [1800, 3600, 5400, 7200, 9000, 10800],
            )

            result = self.run_script(root)

        self.assertEqual(result.returncode, 0)
        self.assertIn("All temperature series", result.stdout)

    def test_prints_only_cities_with_missing_hourly_slots(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            write_observations(
                root,
                "Continuous-AAAA",
                [0, 3600, 7200, 10800],
            )
            write_observations(
                root,
                "Missing-BBBB",
                [0, 3600, 10800, 14400],
            )

            result = self.run_script(root, "--details")

        self.assertEqual(result.returncode, 1)
        self.assertNotIn("Continuous-AAAA", result.stdout)
        self.assertIn(
            "Missing-BBBB [UTC]: 1 missing hourly slot",
            result.stdout,
        )
        self.assertIn("1970-01-01 缺少小时: 02:00", result.stdout)
        self.assertIn(
            "UTC gap: 1970-01-01 02:00 UTC",
            result.stdout,
        )

    def test_configured_city_uses_its_local_date_and_hour(self):
        previous = int(
            datetime(2026, 7, 29, 17, tzinfo=timezone.utc).timestamp()
        )
        following = int(
            datetime(2026, 7, 29, 19, tzinfo=timezone.utc).timestamp()
        )

        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            write_observations(
                root,
                "Chongqing-ZUCK",
                [previous, following],
            )

            result = self.run_script(root)

        self.assertEqual(result.returncode, 1)
        self.assertIn(
            "Chongqing-ZUCK [Asia/Shanghai]",
            result.stdout,
        )
        self.assertIn("2026-07-30 缺少小时: 02:00", result.stdout)

    def test_names_only_prints_city_name_for_invalid_jsonl(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            write_observations(root, "Broken-CCCC", [0, 3600])
            path = root / "Broken-CCCC" / "tem.jsonl"
            with path.open("a", encoding="utf-8") as output_file:
                output_file.write("not-json\n")

            result = self.run_script(root, "--names-only")

        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout.strip(), "Broken-CCCC")


if __name__ == "__main__":
    unittest.main()

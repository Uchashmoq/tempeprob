"""Tests for the shared data-collection thread launcher."""

import unittest
from unittest.mock import MagicMock, call, patch

import collect


class CollectionThreadTest(unittest.TestCase):
    def test_start_collection_threads_uses_requested_daemon_mode(self):
        forecast_thread = MagicMock()
        temperature_thread = MagicMock()

        with (
            patch.object(
                collect.threading,
                "Thread",
                side_effect=(forecast_thread, temperature_thread),
            ) as thread_factory,
            patch.object(collect.config, "CITY", [{}, {}]),
        ):
            threads = collect.start_collection_threads(daemon=True)

        self.assertEqual(threads, (forecast_thread, temperature_thread))
        self.assertEqual(
            thread_factory.call_args_list,
            [
                call(
                    target=collect.update_forecast_periotically,
                    name="forecast-updater",
                    daemon=True,
                ),
                call(
                    target=collect.update_temperature_periotically,
                    name="temperature-updater",
                    daemon=True,
                ),
            ],
        )
        forecast_thread.start.assert_called_once_with()
        temperature_thread.start.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()

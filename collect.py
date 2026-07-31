import logging
import threading
import config
from forecast import update_forecast_periotically
from observation import update_temperature_periotically


def start_collection_threads(
    *,
    daemon: bool = False,
) -> tuple[threading.Thread, threading.Thread]:
    """Start the forecast and temperature collection loops.

    ``daemon=True`` is suitable when another process component, such as the
    dashboard server, owns the application lifetime.  The standalone
    collector keeps the default non-daemon behavior and joins both threads.
    """
    threads = (
        threading.Thread(
            target=update_forecast_periotically,
            name="forecast-updater",
            daemon=daemon,
        ),
        threading.Thread(
            target=update_temperature_periotically,
            name="temperature-updater",
            daemon=daemon,
        ),
    )

    for thread in threads:
        thread.start()
    logging.info(
        "Collecting forecast and temperature data in %d cities",
        len(config.CITY),
    )
    return threads


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=(logging.StreamHandler(), logging.FileHandler("log.txt")),
    )
    threads = start_collection_threads()
    for thread in threads:
        thread.join()


if __name__ == "__main__":
    main()

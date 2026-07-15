import logging
import threading
import config
from forecast import update_forecast_periotically
from observation import update_temperature_periotically


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    threads = (
        threading.Thread(
            target=update_forecast_periotically,
            name="forecast-updater",
        ),
        threading.Thread(
            target=update_temperature_periotically,
            name="temperature-updater",
        ),
    )

    for thread in threads:
        thread.start()
    logging.info(
        f"Collecting forecast and temperature data in {len(config.CITY)} cities"
    )
    for thread in threads:
        thread.join()


if __name__ == "__main__":
    main()

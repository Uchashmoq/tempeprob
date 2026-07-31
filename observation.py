import data_source
from pathlib import Path
import json
import logging
import time
import config

latest_temperature: dict[str, dict] = {}
MAX_LOGGED_RESPONSE_BODY_CHARS = 4096


def get_latest_temperature(city_name):
    path = Path("data/temperature") / city_name / "tem.jsonl"
    try:
        with path.open("rb") as file:
            file.seek(0, 2)
            position = file.tell()
            if position == 0:
                return None

            chunks = []
            while position > 0:
                size = min(8192, position)
                position -= size
                file.seek(position)
                chunk = file.read(size)
                if not chunks:
                    chunk = chunk.rstrip()
                newline = chunk.rfind(b"\n")
                if newline != -1:
                    chunks.append(chunk[newline + 1 :])
                    break
                chunks.append(chunk)

            line = b"".join(reversed(chunks))
            return json.loads(line) if line else None
    except (FileNotFoundError, NotADirectoryError, IsADirectoryError):
        return None


def save_temperature(city_name, tem):
    path = Path("data/temperature") / city_name / "tem.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        json.dump(tem, file, ensure_ascii=False, separators=(",", ":"))
        file.write("\n")


def _tem_eq(tem1, tem2):
    return tem1["temperature"] == tem2["temperature"] and tem1["time"] == tem2["time"]


def update_temperature(city):
    tem = data_source.aviationweather_temp(city["ICAO"])
    city_name = city["name"]

    if city_name not in latest_temperature:
        latest_temperature[city_name] = {}
    ltem = latest_temperature[city_name]
    if ltem and _tem_eq(tem, ltem):
        return False
    save_temperature(city_name, tem)
    latest_temperature[city_name] = tem
    logging.info("Updated temperature for %s: %s", city_name, tem)
    return True


def _response_body_for_log(response_body: str) -> str:
    if len(response_body) <= MAX_LOGGED_RESPONSE_BODY_CHARS:
        return response_body
    omitted = len(response_body) - MAX_LOGGED_RESPONSE_BODY_CHARS
    return (
        response_body[:MAX_LOGGED_RESPONSE_BODY_CHARS]
        + f"... <{omitted} character(s) omitted>"
    )


def _update_temperature_safely(city) -> bool:
    """Update one city and log failures without emitting a traceback."""
    city_name = city.get("name", "<unknown>")
    icao = city.get("ICAO", "<unknown>")
    try:
        return update_temperature(city)
    except data_source.AviationWeatherResponseError as error:
        logging.error(
            "Failed to update temperature for %s (%s): %s; "
            "HTTP status=%s; response_body=%r",
            city_name,
            icao,
            error,
            error.status_code,
            _response_body_for_log(error.response_body),
        )
    except Exception as error:
        logging.error(
            "Failed to update temperature for %s (%s): %s: %s",
            city_name,
            icao,
            type(error).__name__,
            error,
        )
    return False


def update_temperature_periotically():
    for city in config.CITY:
        city_name = city["name"]
        tem = get_latest_temperature(city_name)
        if tem != None:
            latest_temperature[city_name] = tem
        else:
            latest_temperature[city_name] = {}

    while True:
        for city in config.CITY:
            _update_temperature_safely(city)

        time.sleep(config.UPDATE_TEMPERATURE_INTERVAL)

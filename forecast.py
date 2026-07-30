import json
import logging
import time
from datetime import datetime
from pathlib import Path

import config
import data_source
import train_emos_max_temperature

latest_forecast: dict[str, dict] = {}


def get_latest_forecast(city_name, model_name):
    path = Path("data/forecast") / city_name / model_name / "fc.jsonl"
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


def save_forcast(fc, city_name, model_name):
    path = Path("data/forecast") / city_name / model_name / "fc.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        json.dump(fc, file, ensure_ascii=False, separators=(",", ":"))
        file.write("\n")


def _fc_eq(fc1, fc2):
    fc1c = {k: v for k, v in fc1.items() if k != "meta"}
    fc2c = {k: v for k, v in fc2.items() if k != "meta"}
    return fc1c == fc2c


def update_forecast(city, model):
    has_new = False
    meta = data_source.ensemble_forcast_meta(model)
    key = f"{city["name"]}/{model}"
    if key not in latest_forecast:
        latest_forecast[key] = {}
    lfc = latest_forecast[key]
    if lfc:
        lmeta = lfc["meta"]
        has_new = lmeta != meta
    else:
        has_new = True

    if not has_new:
        return False

    fc = data_source.ensemble_forcast(city["lat"], city["lon"], model)
    if _fc_eq(fc, lfc):
        logging.warning(
            "Forecast metadata changed but forecast data did not for %s", key
        )
        return False

    fc["meta"] = meta
    save_forcast(fc, city["name"], model)
    latest_forecast[key] = fc
    initialisation_time = datetime.fromtimestamp(
        meta["last_run_initialisation_time"]
    ).strftime("%Y-%m-%d %H:%M:%S")
    availability_time = datetime.fromtimestamp(
        meta["last_run_availability_time"]
    ).strftime("%Y-%m-%d %H:%M:%S")
    logging.info(
        "Updated forecast for %s (initialised: %s, available: %s)",
        key,
        initialisation_time,
        availability_time,
    )
    return True


def _initialize_latest_forecasts():
    for city in config.CITY:
        city_name = city["name"]
        for model in city["models"]:
            model_name = model["name"]
            key = f"{city_name}/{model_name}"
            lfc = get_latest_forecast(city_name, model_name)
            if lfc is not None:
                latest_forecast[key] = lfc
            else:
                latest_forecast[key] = {}


def _forecast_update_training_metadata(
    city_name: str,
    model_name: str,
) -> dict:
    metadata = {"trigger": "forecast_update"}
    forecast = latest_forecast.get(f"{city_name}/{model_name}")
    if isinstance(forecast, dict):
        forecast_metadata = forecast.get("meta")
        if isinstance(forecast_metadata, dict):
            metadata["forecast_meta"] = dict(forecast_metadata)  # type: ignore
    return metadata


def _update_forecasts_once():
    """Update every configured forecast and train each changed city/model."""
    for city in config.CITY:
        city_name = city["name"]
        for model in city["models"]:
            model_name = model["name"]
            try:
                updated = update_forecast(city, model_name)
            except Exception:
                logging.exception(
                    "Failed to update forecast for %s/%s",
                    city_name,
                    model_name,
                )
                continue

            if updated is not True:
                continue

            try:
                train_city_model = (
                    train_emos_max_temperature.train_daily_max_temperature_emos_for_city_model
                )
                artifact_path = train_city_model(
                    city_name,
                    model_name,
                    extra_metadata=_forecast_update_training_metadata(
                        city_name,
                        model_name,
                    ),
                )
            except Exception:
                logging.exception(
                    "Forecast updated but failed to train daily-max EMOS " "for %s/%s",
                    city_name,
                    model_name,
                )
                continue

            if artifact_path is not None:
                logging.info(
                    "Trained daily-max EMOS for %s/%s: %s",
                    city_name,
                    model_name,
                    artifact_path,
                )


def update_forecast_periotically():
    _initialize_latest_forecasts()

    while True:
        _update_forecasts_once()
        time.sleep(config.UPDATE_FORECAST_INTERVAL)

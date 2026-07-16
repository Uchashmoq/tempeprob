import re
import requests
import pandas as pd
from datetime import datetime


def ensemble_forcast(lat, lon, model="ecmwf_aifs025_ensemble") -> dict:
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m",
        "models": [model],
        "forecast_days": 3,
        "timeformat": "unixtime",
        "wind_speed_unit": "ms",
    }
    resp = requests.get(
        url="https://ensemble-api.open-meteo.com/v1/ensemble", params=params
    ).json()

    fc = {"model": model}
    hourly = resp["hourly"]
    for k, v in hourly.items():
        fc[k] = v
    return fc


def ensemble_forcast_meta(model="ecmwf_aifs025_ensemble") -> dict:
    """
    return {
      "data_end_time": 1785304800,
      "last_run_availability_time": 1784022814,
      "last_run_initialisation_time": 1783987200,
      "last_run_modification_time": 1784022814,
      "temporal_resolution_seconds": 21600,
      "update_interval_seconds": 21600
    }
    """
    resp = requests.get(
        f"https://ensemble-api.open-meteo.com/data/{model}/static/meta.json"
    ).json()
    resp.pop("chunk_time_length", None)
    resp.pop("crs_wkt", None)
    return resp


def time_format1(time_str):
    dt = datetime.fromisoformat(time_str.replace("Z", "+00:00"))
    ts = dt.timestamp()
    return int(ts)


def aviationweather_temp(icao):
    resp = requests.get(
        f"https://aviationweather.gov/api/data/metar?ids={icao}&format=json"
    ).json()
    if isinstance(resp, list):
        resp = resp[0]
    else:
        raise RuntimeError("unexpected response: ", resp)
    return {
        "temperature": resp["temp"],
        "time": resp["obsTime"],
        "update_time": time_format1(resp["receiptTime"]),
    }


if __name__ == "__main__":
    # url = "https://www.wunderground.com/weather/cn/chongqing/ZUCK"
    # # url = "https://www.wunderground.com/weather/kr/incheon/RKSI"
    # t1 = wunderground_temperature(url)
    # t2 = wunderground_tomorrow_high(url)
    # print(f"{url}\nNow: {t1} F\nTomorrow: {t2}F")

    pass

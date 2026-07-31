import re
import requests
import pandas as pd
from datetime import datetime


class AviationWeatherResponseError(RuntimeError):
    """An Aviation Weather response that cannot be used as a temperature."""

    def __init__(
        self,
        reason: str,
        *,
        status_code: int | None,
        response_body: str,
    ) -> None:
        super().__init__(reason)
        self.status_code = status_code
        self.response_body = response_body


class OpenMeteoForecastError(RuntimeError):
    """A readable Open-Meteo request or response failure."""

    def __init__(
        self,
        reason: str,
        *,
        stage: str,
        status_code: int | None = None,
        response_body: str | None = None,
    ) -> None:
        super().__init__(reason)
        self.stage = stage
        self.status_code = status_code
        self.response_body = response_body


def _open_meteo_json(
    url: str,
    *,
    stage: str,
    params: dict | None = None,
):
    try:
        response = requests.get(url=url, params=params)
    except requests.RequestException as error:
        error_text = " ".join(str(error).splitlines())
        raise OpenMeteoForecastError(
            f"{type(error).__name__}: {error_text}",
            stage=stage,
        ) from error

    status_code = response.status_code
    response_body = response.text
    try:
        response.raise_for_status()
    except requests.RequestException as error:
        raise OpenMeteoForecastError(
            "Open-Meteo returned an HTTP error",
            stage=stage,
            status_code=status_code,
            response_body=response_body,
        ) from error

    try:
        payload = response.json()
    except ValueError as error:
        raise OpenMeteoForecastError(
            "Open-Meteo returned invalid JSON",
            stage=stage,
            status_code=status_code,
            response_body=response_body,
        ) from error
    return payload, status_code, response_body


def ensemble_forcast(lat, lon, model="ecmwf_aifs025_ensemble") -> dict:
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m",
        "models": [model],
        "forecast_days": 7,
        "timeformat": "unixtime",
        "wind_speed_unit": "ms",
    }
    resp, status_code, response_body = _open_meteo_json(
        "https://ensemble-api.open-meteo.com/v1/ensemble",
        stage="ensemble forecast",
        params=params,
    )
    if not isinstance(resp, dict):
        raise OpenMeteoForecastError(
            "Open-Meteo forecast response must be a JSON object",
            stage="ensemble forecast",
            status_code=status_code,
            response_body=response_body,
        )

    fc = {"model": model}
    hourly = resp.get("hourly")
    if not isinstance(hourly, dict) or not hourly:
        raise OpenMeteoForecastError(
            "Open-Meteo forecast response contains no hourly data",
            stage="ensemble forecast",
            status_code=status_code,
            response_body=response_body,
        )
    times = hourly.get("time")
    member_names = [
        name
        for name, values in hourly.items()
        if isinstance(values, list)
        and (
            name == "temperature_2m"
            or name.startswith("temperature_2m_member")
        )
    ]
    if not isinstance(times, list) or not times or not member_names:
        raise OpenMeteoForecastError(
            "Open-Meteo forecast response has no usable temperature time series",
            stage="ensemble forecast",
            status_code=status_code,
            response_body=response_body,
        )
    if any(len(hourly[name]) != len(times) for name in member_names):
        raise OpenMeteoForecastError(
            "Open-Meteo forecast response has unaligned temperature arrays",
            stage="ensemble forecast",
            status_code=status_code,
            response_body=response_body,
        )
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
    resp, status_code, response_body = _open_meteo_json(
        f"https://ensemble-api.open-meteo.com/data/{model}/static/meta.json",
        stage="forecast metadata",
    )
    if not isinstance(resp, dict):
        raise OpenMeteoForecastError(
            "Open-Meteo metadata response must be a JSON object",
            stage="forecast metadata",
            status_code=status_code,
            response_body=response_body,
        )
    required_fields = (
        "last_run_availability_time",
        "last_run_initialisation_time",
    )
    missing_fields = [field for field in required_fields if field not in resp]
    if missing_fields:
        raise OpenMeteoForecastError(
            "Open-Meteo metadata response is missing required field(s): "
            + ", ".join(missing_fields),
            stage="forecast metadata",
            status_code=status_code,
            response_body=response_body,
        )
    invalid_fields = [
        field
        for field in required_fields
        if isinstance(resp[field], bool)
        or not isinstance(resp[field], (int, float))
        or resp[field] < 0
    ]
    if invalid_fields:
        raise OpenMeteoForecastError(
            "Open-Meteo metadata response has invalid field(s): "
            + ", ".join(invalid_fields),
            stage="forecast metadata",
            status_code=status_code,
            response_body=response_body,
        )
    resp.pop("chunk_time_length", None)
    resp.pop("crs_wkt", None)
    return resp


def time_format1(time_str):
    dt = datetime.fromisoformat(time_str.replace("Z", "+00:00"))
    ts = dt.timestamp()
    return int(ts)


def aviationweather_temp(icao):
    response = requests.get(
        f"https://aviationweather.gov/api/data/metar?ids={icao}&format=json"
    )
    status_code = response.status_code
    response_body = response.text

    try:
        response.raise_for_status()
    except requests.RequestException as error:
        raise AviationWeatherResponseError(
            "Aviation Weather returned an HTTP error",
            status_code=status_code,
            response_body=response_body,
        ) from error

    try:
        payload = response.json()
    except ValueError as error:
        raise AviationWeatherResponseError(
            "Aviation Weather returned invalid JSON",
            status_code=status_code,
            response_body=response_body,
        ) from error

    if not isinstance(payload, list) or not payload:
        raise AviationWeatherResponseError(
            "Aviation Weather returned an unexpected JSON payload",
            status_code=status_code,
            response_body=response_body,
        )

    try:
        metar = payload[0]
        return {
            "temperature": metar["temp"],
            "time": metar["obsTime"],
            "update_time": time_format1(metar["receiptTime"]),
        }
    except (KeyError, TypeError, ValueError) as error:
        raise AviationWeatherResponseError(
            "Aviation Weather returned an invalid METAR record",
            status_code=status_code,
            response_body=response_body,
        ) from error


if __name__ == "__main__":
    # url = "https://www.wunderground.com/weather/cn/chongqing/ZUCK"
    # # url = "https://www.wunderground.com/weather/kr/incheon/RKSI"
    # t1 = wunderground_temperature(url)
    # t2 = wunderground_tomorrow_high(url)
    # print(f"{url}\nNow: {t1} F\nTomorrow: {t2}F")

    pass

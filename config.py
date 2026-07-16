CITY = [
    {
        "name": "Chongqing-ZUCK",
        "lat": 29.718,
        "lon": 106.639,
        "timezone": "UTC+8",
        "models": [
            {"name": "ecmwf_aifs025_ensemble", "update_interval": 6 * 3600},
        ],
        "ICAO": "ZUCK",
        "temp_unit": "C",
        "questions": range(28, 43),  # 30: 30 or below, 40: 40 or above, 37: 37<=T<38
    },
    {
        "name": "Chongqing-center",
        "lat": 29.57,
        "lon": 106.55,
        "timezone": "UTC+8",
        "models": [
            {"name": "ecmwf_aifs025_ensemble", "update_interval": 6 * 3600},
        ],
        "ICAO": "ZUCK",
        "temp_unit": "C",
        "questions": range(28, 43),  # 30: 30 or below, 40: 40 or above, 37: 37<=T<38
    },
]

UPDATE_FORECAST_INTERVAL = 60
UPDATE_TEMPERATURE_INTERVAL = 600

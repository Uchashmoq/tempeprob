CITY = [
    {
        "name": "Chongqing-ZUCK",
        "lat": 29.718,
        "lon": 106.639,
        "timezone": "Asia/Shanghai",
        "models": [
            {"name": "ecmwf_aifs025_ensemble", "update_interval": 6 * 3600},
            {"name": "ecmwf_ifs025_ensemble", "update_interval": 6 * 3600},
        ],
        "ICAO": "ZUCK",
        "temp_unit": "C",
    },
    {
        "name": "Chengdu-ZUUU",
        "lat": 30.578,
        "lon": 103.947,
        "timezone": "Asia/Shanghai",
        "models": [
            {"name": "ecmwf_aifs025_ensemble", "update_interval": 6 * 3600},
            {"name": "ecmwf_ifs025_ensemble", "update_interval": 6 * 3600},
        ],
        "ICAO": "ZUUU",
        "temp_unit": "C",
    },
    {
        "name": "Shenzhen-ZGSZ",
        "lat": 22.639,
        "lon": 113.803,
        "timezone": "Asia/Shanghai",
        "models": [
            {"name": "ecmwf_aifs025_ensemble", "update_interval": 6 * 3600},
            {"name": "ecmwf_ifs025_ensemble", "update_interval": 6 * 3600},
        ],
        "ICAO": "ZGSZ",
        "temp_unit": "C",
    },
    {
        "name": "Wuhan-ZHHH",
        "lat": 30.783,
        "lon": 114.205,
        "timezone": "Asia/Shanghai",
        "models": [
            {"name": "ecmwf_aifs025_ensemble", "update_interval": 6 * 3600},
            {"name": "ecmwf_ifs025_ensemble", "update_interval": 6 * 3600},
        ],
        "ICAO": "ZHHH",
        "temp_unit": "C",
    },
    {
        "name": "Seoul-RKSI",
        "lat": 37.469,
        "lon": 126.451,
        "timezone": "Asia/Seoul",
        "models": [
            {"name": "ecmwf_aifs025_ensemble", "update_interval": 6 * 3600},
            {"name": "ecmwf_ifs025_ensemble", "update_interval": 6 * 3600},
        ],
        "ICAO": "RKSI",
        "temp_unit": "C",
    },
    {
        "name": "Paris-LFPB",
        "lat": 48.967,
        "lon": 2.428,
        "timezone": "Europe/Paris",
        "models": [
            {"name": "ecmwf_aifs025_ensemble", "update_interval": 6 * 3600},
            {"name": "ecmwf_ifs025_ensemble", "update_interval": 6 * 3600},
        ],
        "ICAO": "LFPB",
        "temp_unit": "C",
    },
    {
        "name": "Madrid-LEMD",
        "lat": 40.466,
        "lon": -3.555,
        "timezone": "Europe/Madrid",
        "models": [
            {"name": "ecmwf_aifs025_ensemble", "update_interval": 6 * 3600},
            {"name": "ecmwf_ifs025_ensemble", "update_interval": 6 * 3600},
        ],
        "ICAO": "LEMD",
        "temp_unit": "C",
    },
]

UPDATE_FORECAST_INTERVAL = 600
UPDATE_TEMPERATURE_INTERVAL = 600
TRAIN_MODEL_INTERVAL = 3600 * 3

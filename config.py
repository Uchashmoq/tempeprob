UPDATE_FORECAST_INTERVAL = 400
UPDATE_TEMPERATURE_INTERVAL = 400
TRAIN_MODEL_INTERVAL = 3600 * 3
AUTO_TRAIN = True
AUTO_PREDICT = True
PREDICT_DAYS = 3

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
        "slug_prefix": "highest-temperature-in-chongqing",
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
        "slug_prefix": "highest-temperature-in-chengdu",
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
        "slug_prefix": "highest-temperature-in-shenzhen",
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
        "slug_prefix": "highest-temperature-in-wuhan",
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
        "slug_prefix": "highest-temperature-in-seoul",
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
        "slug_prefix": "highest-temperature-in-paris",
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
        "slug_prefix": "highest-temperature-in-madrid",
    },
    {
        "name": "Warsaw-EPWA",
        "lat": 52.163,
        "lon": 20.961,
        "timezone": "Europe/Warsaw",
        "models": [
            {"name": "ecmwf_aifs025_ensemble", "update_interval": 6 * 3600},
            {"name": "ecmwf_ifs025_ensemble", "update_interval": 6 * 3600},
        ],
        "ICAO": "EPWA",
        "temp_unit": "C",
        "slug_prefix": "highest-temperature-in-warsaw",
    },
    {
        "name": "Beijing-ZBAA",
        "lat": 40.082,
        "lon": 116.603,
        "timezone": "Asia/Shanghai",
        "models": [
            {"name": "ecmwf_aifs025_ensemble", "update_interval": 6 * 3600},
            {"name": "ecmwf_ifs025_ensemble", "update_interval": 6 * 3600},
        ],
        "ICAO": "ZBAA",
        "temp_unit": "C",
        "slug_prefix": "highest-temperature-in-beijing",
    },
    {
        "name": "Buenos-Aires-SAEZ",
        "lat": -34.822,
        "lon": -58.536,
        "timezone": "America/Argentina/Buenos_Aires",
        "models": [
            {"name": "ecmwf_aifs025_ensemble", "update_interval": 6 * 3600},
            {"name": "ecmwf_ifs025_ensemble", "update_interval": 6 * 3600},
        ],
        "ICAO": "SAEZ",
        "temp_unit": "C",
        "slug_prefix": "highest-temperature-in-buenos-aires",
    },
    {
        "name": "Munich-EDDM",
        "lat": 48.348,
        "lon": 11.813,
        "timezone": "Europe/Berlin",
        "models": [
            {"name": "ecmwf_aifs025_ensemble", "update_interval": 6 * 3600},
            {"name": "ecmwf_ifs025_ensemble", "update_interval": 6 * 3600},
        ],
        "ICAO": "EDDM",
        "temp_unit": "C",
        "slug_prefix": "highest-temperature-in-munich",
    },
    {
        "name": "Ankara-LTAC",
        "lat": 40.128,
        "lon": 32.995,
        "timezone": "Europe/Istanbul",
        "models": [
            {"name": "ecmwf_aifs025_ensemble", "update_interval": 6 * 3600},
            {"name": "ecmwf_ifs025_ensemble", "update_interval": 6 * 3600},
        ],
        "ICAO": "LTAC",
        "temp_unit": "C",
        "slug_prefix": "highest-temperature-in-ankara",
    },
]

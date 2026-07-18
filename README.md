# Temperature probability
## Init
```bash
python3 -m venv venv
source venv/bin/activate
pip install pandas requests numpy rpy2
```


## Example data
### open-meteo essemble api
`https://ensemble-api.open-meteo.com/v1/ensemble?latitude=29.57&longitude=106.55&hourly=temperature_2m&models=ecmwf_ifs025_ensemble&forecast_days=3&timeformat=unixtime&wind_speed_unit=ms`
```json
//fetch at 2026-7-14 20:22 GMT+8
{
  "latitude": 29.5,
  "longitude": 106.5,
  "generationtime_ms": 0.79345703125,
  "utc_offset_seconds": 0,
  "timezone": "GMT",
  "timezone_abbreviation": "GMT",
  "elevation": 168,
  "hourly_units": {
    "time": "unixtime",
    "temperature_2m": "°C",
    "temperature_2m_member01": "°C",
    "temperature_2m_member02": "°C",
    // ...
  },
  "hourly": {
    "time": [1783987200, 1783990800, 1783994400, ...],
    "temperature_2m": [33.9, 35.4, 37.1, ...],
    "temperature_2m_member01": [34.2, 35.6, 37.5, ...],
    "temperature_2m_member02": [33.2, 34.8, 36.8, ...],
  }
}
```

### open-meteo forcast meta
`https://ensemble-api.open-meteo.com/data/ecmwf_aifs025_ensemble/static/meta.json`
```json
//fetch at 2026-7-14 20:22 GMT+8
{
  "chunk_time_length": 72,
  "crs_wkt": "GEOGCRS[\"WGS 84\",\n    DATUM[\"World Geodetic System 1984\",\n        ELLIPSOID[\"WGS 84\",6378137,298.257223563]],\n    CS[ellipsoidal,2],\n        AXIS[\"latitude\",north],\n        AXIS[\"longitude\",east],\n        ANGLEUNIT[\"degree\",0.0174532925199433]\n    USAGE[\n        SCOPE[\"grid\"],\n        BBOX[-90.0,-180.0,90.0,179.75]]]",
  "data_end_time": 1785391200,
  "last_run_availability_time": 1784100444,
  "last_run_initialisation_time": 1784073600,
  "last_run_modification_time": 1784100444,
  "temporal_resolution_seconds": 21600,
  "update_interval_seconds": 21600
}
```
### aviationweather.gov
`https://aviationweather.gov/api/data/metar?ids=ZUCK&format=json`
```json
[
  {
    "icaoId": "ZUCK",
    "receiptTime": "2026-07-15T09:06:06.481Z",
    "obsTime": 1784106000,
    "reportTime": "2026-07-15T09:00:00.000Z",
    "temp": 40,
    "dewp": 22,
    "wdir": 80,
    "wspd": 8,
    "visib": "6+",
    "altim": 997,
    "qcField": 16,
    "metarType": "METAR",
    "rawOb": "METAR ZUCK 150900Z 08004MPS 010V140 9999 FEW040 40/22 Q0997 NOSIG",
    "lat": 29.718,
    "lon": 106.639,
    "elev": 416,
    "name": "Chongqing/Jiangbei Intl, CQ, CN",
    "cover": "FEW",
    "clouds": [
      {
        "cover": "FEW",
        "base": 4000
      }
    ],
    "fltCat": "VFR"
  }
]
```
### fc.jsonl
```json
{
   "model":"ecmwf_aifs025_ensemble",
   "time":[
      1784073600,
      1784077200,
      1784080800,
...
   ],
   "temperature_2m":[
      33.2,
      34.2,
      35.8,
...
   ],
   "temperature_2m_member01":[
      33.7,
      34.6,
      35.9,
...
   ],
   "meta":{
      "data_end_time":1785391200,
      "last_run_availability_time":1784100444,
      "last_run_initialisation_time":1784073600,
      "last_run_modification_time":1784100444,
      "temporal_resolution_seconds":21600,
      "update_interval_seconds":21600
   }
}
```

### tem.jsonl
```json
{
   "temperature":40,
   "time":1784109600,
   "update_time":1784109700
}
```
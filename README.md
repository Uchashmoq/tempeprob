# Temperature probability
## Init
```bash
python3 -m venv venv
source venv/bin/activate
pip install pandas requests numpy rpy2 bottle
```

## Prediction dashboard

The read-only Bottle dashboard scans
`prediction/highest_temperature_emos/*/*/predictions.jsonl` on every request,
so newly appended prediction revisions appear after a page refresh.
It also fetches each visible Polymarket event's current `Yes` prices through
the public Gamma API and compares them with the EMOS interval probabilities.
Quotes are server-side snapshots cached for up to five minutes; the page does
not poll automatically. If Polymarket is unavailable, saved model predictions
remain visible and the market columns show an unavailable status.

```bash
venv/bin/python web_server.py
```

The built-in server listens on all network interfaces by default. Open
`http://SERVER_IP:8001`, or <http://127.0.0.1:8001> from the server itself.
Optional arguments:

```bash
venv/bin/python web_server.py \
  --host 0.0.0.0 \
  --port 8001 \
  --prediction-dir prediction/highest_temperature_emos
```

Use `--host 127.0.0.1` when the dashboard should only be reachable locally or
through a reverse proxy.

Add `--collect` to run the forecast and temperature collectors in background
threads owned by the dashboard process:

```bash
venv/bin/python web_server.py --collect
```

Do not run `collect.py` separately at the same time, or the two processes may
attempt to collect and append the same updates.

Check every city's temperature JSONL for missing hourly slots:

```bash
venv/bin/python script/check_temperature_continuity.py
```

The default output groups missing hours by each configured city's local date.
Use `--details` to additionally print the underlying UTC ranges, or
`--names-only` to print only the city directory names. The command exits with
status 1 when any city has a gap or invalid record.

For a public deployment, expose the Bottle WSGI application through a
production server instead of the built-in development server:

```bash
venv/bin/gunicorn --workers 2 --bind 127.0.0.1:8001 web_server:app
```

## Init R Package
`https://cran.r-project.org/web/packages/ensembleMOS/index.html`

Install the required R packages into `.r-library` under the project root:

```bash
./script/install_r_packages.sh
export R_LIBS_USER="$PWD/.r-library"
```

The script detects the project root from its own location, so the repository
does not need to be installed under `/opt`. To select another R library:

```bash
./script/install_r_packages.sh /srv/tempeprob-r-library
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

### group
```json
{
    "initialization_hour": "00",
    "initialization_time": "00",
    "day_ahead": 1,
    "model": "ecmwf_aifs025_ensemble",
    "timezone": "Asia/Shanghai",
    "member_names": (
        "temperature_2m",
        "temperature_2m_member01",
        "temperature_2m_member02",
    ),
    "expected_interval_seconds": 3600,

    "target_dates": [
        "2026-07-16",
        "2026-07-17",
    ],
    "target_day_start_times": [
        1784131200,
        1784217600,
    ],
    "target_day_end_times": [
        1784217600,
        1784304000,
    ],
    "initialization_times": [
        1784073600,
        1784160000,
    ],
    "availability_times": [
        1784100444,
        1784187584,
    ],
    "forecasts": [
        (36.2, 37.1, 35.8), highest temperature of every member
        (34.5, 35.3, 34.9),
    ],
    "observations": [
        40.0,
        34.0,
    ],
    "forecast_counts": [
        24,
        24,
    ],
    "observation_counts": [
        24,
        24,
    ],
    "observation_coverages": [
        1.0,
        1.0,
    ],
}
```

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

Application, collection, and HTTP access logs are printed to the console and
appended to `log.txt` in the project root. Select another file with
`--log-file`.

The built-in server listens on all network interfaces by default. Open
`http://SERVER_IP:8001`, or <http://127.0.0.1:8001> from the server itself.
Optional arguments:

```bash
venv/bin/python web_server.py \
  --host 0.0.0.0 \
  --port 8001 \
  --prediction-dir prediction/highest_temperature_emos \
  --log-file log.txt
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

## Future Work

结论：当前最值得做的是“扩展 METAR + 扩展同一起报的 ECMWF 辅助变量”，然后使用带正则化的多协变量 NGR/EMOS。暂时不建议直接上随机森林或神经网络，现有样本量太少。

### 1. 优先加入的观测数据

当前 [aviationweather_temp()](/Users/mhr/Documents/tempeprob/data_source.py:200) 请求的 METAR 实际包含很多字段，但代码只保存了温度、观测时间和接收时间。

| 优先级 | METAR 字段 | 建议衍生特征 | 主要作用 |
|---|---|---|---|
| 最高 | `dewp` 露点 | `temperature - dewpoint`、相对湿度 | 湿度、蒸发冷却、闷热天气 |
| 最高 | `wdir/wspd/wgst` | 风的 `u/v` 分量、最大风速 | 混合、平流、海陆风 |
| 最高 | `clouds/cover` | 低云量、最低 BKN/OVC 云底 | 判断日照能否把最高温推高 |
| 高 | `wxString` | 雨、雪、雾、雷暴标记 | 云雨降温、边界层状态 |
| 高 | `precip/pcp3hr/pcp6hr/pcp24hr` | 过去1～24小时降水 | 湿土、蒸发冷却 |
| 高 | `altim/slp/presTend` | 气压及3小时变化 | 锋面、气团转换 |
| 中 | `visib` | 低能见度/雾标记 | 低云、雾、气溶胶代理 |
| 冬季重要 | `snow`、降雪现象 | 雪深、是否有积雪 | 反照率、融雪、夜间逆温 |
| 审计 | `rawOb/qcField/metarType` | 原始报文、质量标记 | 修订追踪和重新解析 |

这些字段来自同一次请求，不增加 API 调用次数。METAR 官方说明其主体包含风、能见度、天气现象、云况、温度、露点和气压，[AviationWeather METAR 文档](https://aviationweather.gov/help/data/)也说明部分站点还会报告降水、积雪和更高精度温度。

需要注意：

- 字段缺失不能当成零，特别是降水和积雪。
- `wdir` 可能是 `VRB`，`visib` 可能是 `6+`，需要允许混合类型。
- 当前 [_tem_eq()](/Users/mhr/Documents/tempeprob/observation.py:49) 只比较温度和时刻。加入新字段后，应改为按 `ICAO + obsTime + receiptTime/rawOb` 去重，否则温度未变但云、风被修订的记录会被吞掉。
- AviationWeather 在线接口只能回看约15天，因此要从现在开始自行归档。[官方 API 说明](https://aviationweather.gov/data/api/)

### 2. 同时加入 ECMWF 辅助预报

未来的云、风、土壤和雪无法提前观测，所以训练和预测还需要保存“同一次起报”中的：

- `dew_point_2m`、`relative_humidity_2m`
- `cloud_cover`、`cloud_cover_low`
- `shortwave_radiation`、`sunshine_duration`
- `wind_speed_10m`、`wind_direction_10m`
- `precipitation`、`snowfall`、`snow_depth`
- `surface_temperature`
- 浅层土壤温度、土壤湿度
- 925/850 hPa 温度、湿度、风、位势高度

Open-Meteo Ensemble API 支持其中多数变量，但要确认每个 ECMWF 模型实际返回哪些字段。[变量清单](https://open-meteo.com/en/docs/ensemble-api)和[ECMWF Open Data 参数表](https://www.ecmwf.int/en/forecasts/datasets/open-data)均包含这些地面和高空变量。

日最高温特征建议这样聚合：

- 每个成员先求自己的日最高温，再计算成员均值和离散度。
- 白昼云量取当地约 09–18 时均值。
- 短波辐射取白昼积分。
- 风向转为 `u/v`，避免 359° 和 1° 被当成差异很大。
- 降水取目标日前24小时及目标日累计。
- 雪深、土壤湿度取目标日早晨或起报时状态。

不要先把51个成员平均再求最高温，这会压低极值和集合离散度。

### 3. 推荐的修正算法

最适合当前项目的是正则化的多协变量高斯 NGR：

\[
T_{\max}\sim N(\mu,\sigma^2)
\]

均值模型：

\[
\mu =
\beta_0+
\beta_1\overline{T_{\max}}+
\beta_2(T-T_d)+
\beta_3 Cloud+
\beta_4 Radiation+
\beta_5 Wind_{u,v}+
\beta_6 Precip+
\beta_7 RecentBias
\]

方差模型：

\[
\log \sigma =
\gamma_0+
\gamma_1\log(Spread_{T_{\max}}+\epsilon)+
\gamma_2 CloudSpread+
\gamma_3 WindSpread
\]

建议：

- 用负对数似然或 CRPS 优化。
- 对系数使用 ridge 或 elastic-net。
- 初期只新增露点差、白昼云量/辐射、风、近期偏差这3～4组特征。
- 多城市共同训练，使用城市、起报时次和 `day_ahead` 截距或分层效应，避免继续把少量数据切成大量小组。

当前 `ensembleMOS` 的正态模型固定为：

\[
N(a+\sum b_iX_i,\ c+dS^2)
\]

它没有独立外生协变量接口。[ensembleMOS 手册](https://cran.r-project.org/web/packages/ensembleMOS/ensembleMOS.pdf) 因此不能把露点、云量、风速直接伪装成“集合成员”，否则单位和集合方差都会错误。

有两种实现方式：

1. 推荐：自己实现上述带 ridge 的高斯分布回归，同时拟合 `mu` 和 `sigma`。
2. 改动较小：先保留现有 EMOS，再用 ridge/GAM 学习其滚动验证残差：
   `最终均值 = EMOS均值 + 辅助特征预测的残差`，同时再校准方差。残差模型必须使用 out-of-fold EMOS 结果训练，不能使用同一批数据的拟合内残差。

ECMWF 的研究中，线性回归、随机森林和神经网络对2米温度误差的改善接近，因此当前阶段简单、正则化、可解释的模型更合适。[ECMWF Technical Memorandum 896](https://www.ecmwf.int/en/elibrary/81297-statistical-modelling-2m-temperature-and-10m-wind-speed-forecast-errors)

### 4. 必须防止的数据泄漏

预测生成时刻为 `cutoff` 时：

允许使用：

- 同一次起报中目标日的云、风、降水等预报；
- `update_time <= cutoff` 的真实观测；
- 历史滚动偏差、站点地形和季节特征。

禁止使用：

- `cutoff` 之后的目标日真实云量、风、露点或降水；
- 事后生成的 ERA5/ERA5-Land，却假装起报时已经可用；
- 目标日结束后才知道的完整日观测特征。

预测“今天”时，可以加入 `max_temperature_observed_so_far`，最终分布必须保证低于当前已观测最高温的概率为零。预测明天、后天时则不能使用这些未来观测。

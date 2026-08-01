% rebase(
%   "base",
%   page_title=page_title,
%   active_city=active_city,
%   site_cities=site_cities,
%   loaded_at_text=loaded_at_text,
%   record_count=record_count,
% )

<section class="hero">
  <div>
    <p class="eyebrow">DAILY MAXIMUM · EMOS</p>
    <h1>{{page_title}}</h1>
  </div>
  <dl class="summary-strip">
    <div>
      <dt>城市</dt>
      <dd>{{len(catalog.cities)}}</dd>
    </div>
    <div>
      <dt>模型</dt>
      <dd>{{len(catalog.models)}}</dd>
    </div>
    <div>
      <dt>最新预测</dt>
      <dd>{{latest_count}}</dd>
    </div>
    <div>
      <dt>当前显示</dt>
      <dd>{{visible_count}}</dd>
    </div>
  </dl>
</section>

% if latest_temperature_cards:
<section class="latest-observations" aria-labelledby="latest-observations-title">
  <header class="latest-observations-heading">
    <div>
      <p class="section-kicker">LATEST OBSERVATIONS</p>
      <h2 id="latest-observations-title">最新温度</h2>
    </div>
  </header>
  <div class="latest-observations-grid">
    % for observation in latest_temperature_cards:
    <article class="latest-observation-card{{'' if observation['available'] else ' is-unavailable'}}">
      <header>
        <a href="{{observation['city_url']}}">{{observation["city_label"]}}</a>
        <span>METAR</span>
      </header>
      <strong class="latest-observation-value">
        {{observation["temperature_text"]}}
      </strong>
      <div class="latest-observation-published">
        <span>发布时间</span>
        % if observation["available"]:
        <time
          datetime="{{observation['published_at_iso']}}"
          title="观测时间：{{observation['observed_at_text']}}"
        >{{observation["published_at_text"]}}</time>
        % else:
        <span>暂无观测</span>
        % end
      </div>
    </article>
    % end
  </div>
</section>
% end

<section class="filter-panel" aria-label="筛选预测">
  <form class="filters" action="/" method="get">
    <label>
      <span>城市</span>
      <select name="city">
        <option value="">全部城市</option>
        % for value, label in city_options:
          % if selected_city == value:
        <option value="{{value}}" selected>{{label}}</option>
          % else:
        <option value="{{value}}">{{label}}</option>
          % end
        % end
      </select>
    </label>
    <label>
      <span>目标日期</span>
      <select name="date">
        <option value="">全部日期</option>
        % for value in date_options:
          % if selected_date == value:
        <option value="{{value}}" selected>{{value}}</option>
          % else:
        <option value="{{value}}">{{value}}</option>
          % end
        % end
      </select>
    </label>
    <label>
      <span>模型</span>
      <select name="model">
        <option value="">全部模型</option>
        % for value, label in model_options:
          % if selected_model == value:
        <option value="{{value}}" selected>{{label}}</option>
          % else:
        <option value="{{value}}">{{label}}</option>
          % end
        % end
      </select>
    </label>
    <button type="submit">应用筛选</button>
    <a class="reset-link" href="/">清除</a>
  </form>
</section>

% if catalog.issues:
<details class="data-warning">
  <summary>
    <strong>{{len(catalog.issues)}} 条数据未能载入</strong>
    <span>其余有效预测仍正常显示</span>
  </summary>
  <ul>
    % for issue in catalog.issues:
    <li><code>{{issue.location}}</code> — {{issue.message}}</li>
    % end
  </ul>
</details>
% end

% if not groups:
<section class="empty-state">
  <span class="empty-symbol" aria-hidden="true">—</span>
  <h2>暂无符合条件的预测</h2>
  % if not catalog.records:
  <p>还没有可展示的 JSONL 记录。预测生成后刷新页面即可看到数据。</p>
  % else:
  <p>请调整城市、日期或模型筛选条件。</p>
  % end
</section>
% end

% for day in groups:
<section class="day-section">
  <header class="section-heading">
    <div>
      <p class="section-kicker">目标日期 · 当地日最高温</p>
      <h2>{{day["target_date_label"]}}</h2>
    </div>
    <span class="section-count">
      {{sum(len(city["records"]) for city in day["cities"])}} 个模型预测
    </span>
  </header>

  % for city in day["cities"]:
  <section class="city-section">
    <header class="city-heading">
      <div class="city-heading-primary">
        <h3>{{city["city_label"]}}</h3>
        % if city["market_url"]:
        <a
          class="city-market-link"
          href="{{city['market_url']}}"
          target="_blank"
          rel="noopener noreferrer"
          aria-label="打开 {{city['city_label']}} {{day['target_date']}} 的 Polymarket 市场（新窗口）"
        >Polymarket 市场 <span aria-hidden="true">↗</span></a>
        % end
      </div>
      <div class="city-heading-actions">
        % if city["boundary_mismatch"]:
        <span class="boundary-warning">模型市场边界不一致</span>
        % end
        <a class="city-only-link" href="{{city['city_url']}}">仅查看该城市</a>
      </div>
    </header>

    <div class="model-grid">
      % for item in city["records"]:
        % record = item["record"]
        % comparison = item["market_comparison"]
      <article class="prediction-card">
        <header class="card-heading">
          <div>
            <p class="model-family">{{record.model_label}}</p>
            <p class="model-code">{{record.model_name}}</p>
          </div>
          <span class="revision-badge">修订 {{record.revision}}</span>
        </header>

        <div class="peak-callout">
          <span>最高概率区间</span>
          <strong>{{record.peak_interval.display_label}}</strong>
          <b>{{record.peak_interval.percent_text}}</b>
        </div>

        <div class="probability-comparison-header" aria-hidden="true">
          <span>温度</span>
          <span>模型</span>
          <span>市场 Yes</span>
          <span title="模型概率减去市场 Yes 价格（百分点）">差值</span>
        </div>
        <ol class="probability-list comparison-list" aria-label="模型概率与 Polymarket Yes 价格对比">
          % for row in comparison.rows:
            % interval = row.interval
          <li class="probability-row comparison-row{{' is-peak' if interval == record.peak_interval else ''}}">
            <div class="probability-values">
              <span class="comparison-temperature">{{interval.display_label}}</span>
              <strong class="comparison-number model-number" title="模型概率 {{interval.precise_percent_text}}">
                {{interval.percent_text}}
              </strong>
              <strong class="comparison-number market-number" title="Polymarket Yes {{row.market_precise_percent_text}}">
                {{row.market_percent_text}}
              </strong>
              <strong class="comparison-number difference-number {{row.difference_class}}" title="模型概率减去市场 Yes 价格">
                {{row.difference_text}}
              </strong>
            </div>
            <div class="probability-bars" aria-hidden="true">
              <div class="probability-track is-model">
                <span style="width: {{interval.bar_width}}%"></span>
              </div>
              <div class="probability-track is-market">
                <span style="width: {{row.market_bar_width}}%"></span>
              </div>
            </div>
          </li>
          % end
        </ol>

        <div class="comparison-legend" aria-label="图例">
          <span><i class="legend-swatch is-model"></i>模型概率</span>
          <span><i class="legend-swatch is-market"></i>市场 Yes</span>
          <span>差值 = 模型 − 市场</span>
        </div>

        <dl class="card-metadata">
          <div>
            <dt>预报起报</dt>
            <dd>{{record.forecast_initialization_text}}</dd>
          </div>
          <div>
            <dt>EMOS 版本</dt>
            <dd title="{{record.artifact_version}}">
              {{record.artifact_short_version}}
            </dd>
          </div>
          <div>
            <dt>生成时间</dt>
            <dd>{{record.generated_local_text}}</dd>
          </div>
        </dl>

        <footer class="card-footer">
          <span>
            % if item["history_count"] > 1:
            共 {{item["history_count"]}} 个历史修订
            % else:
            首个修订
            % end
          </span>
          <a class="detail-link" href="{{record.detail_url}}">
            查看详情 <span aria-hidden="true">→</span>
          </a>
        </footer>
      </article>
      % end
    </div>
  </section>
  % end
</section>
% end

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
    <p class="hero-copy">
      对比不同集合模型经过 EMOS 修正后的日最高气温区间概率。
      页面默认展示每个城市、模型和目标日期的最新修订。
    </p>
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
      <div>
        <h3>{{city["city_label"]}}</h3>
        <a href="{{city['city_url']}}">仅查看该城市</a>
      </div>
      % if city["boundary_mismatch"]:
      <span class="boundary-warning">模型市场边界不一致</span>
      % end
    </header>

    <div class="model-grid">
      % for item in city["records"]:
        % record = item["record"]
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
          <strong>{{record.peak_interval.label}}</strong>
          <b>{{record.peak_interval.percent_text}}</b>
        </div>

        <ol class="probability-list" aria-label="温度区间概率">
          % for interval in record.intervals:
          <li class="probability-row{{' is-peak' if interval == record.peak_interval else ''}}">
            <div class="probability-label">
              <span>{{interval.label}}</span>
              <strong title="{{interval.precise_percent_text}}">
                {{interval.percent_text}}
              </strong>
            </div>
            <div class="probability-track" aria-hidden="true">
              <span style="width: {{interval.bar_width}}%"></span>
            </div>
          </li>
          % end
        </ol>

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

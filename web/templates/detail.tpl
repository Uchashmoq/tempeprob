% rebase(
%   "base",
%   page_title=page_title,
%   active_city=active_city,
%   site_cities=site_cities,
%   loaded_at_text=loaded_at_text,
%   record_count=record_count,
% )

<nav class="breadcrumbs" aria-label="面包屑">
  <a href="/">预测总览</a>
  <span>/</span>
  <a href="{{record.city_url}}">{{record.city_label}}</a>
  <span>/</span>
  <span>{{record.target_date}}</span>
</nav>

<section class="detail-hero">
  <div>
    <p class="eyebrow">{{record.model_label}}</p>
    <h1>{{record.city_label}}</h1>
    <p class="detail-date">{{record.target_date_label}} · 日最高气温</p>
  </div>
  <div class="detail-peak">
    <span>最高概率区间</span>
    <strong>{{record.peak_interval.display_label}}</strong>
    <b>{{record.peak_interval.percent_text}}</b>
  </div>
</section>

<div class="detail-layout">
  <div class="detail-main">
    <section class="detail-panel">
      <header class="panel-heading">
        <div>
          <p class="section-kicker">PROBABILITY DISTRIBUTION</p>
          <h2>模型概率 vs 市场价格</h2>
        </div>
      </header>

      <div class="probability-comparison-header probability-comparison-header-large" aria-hidden="true">
        <span>温度</span>
        <span>模型概率</span>
        <span>市场 Yes</span>
        <span title="模型概率减去市场 Yes 价格（百分点）">差值</span>
      </div>
      <ol class="probability-list comparison-list probability-list-large" aria-label="模型概率与 Polymarket Yes 价格对比">
        % for row in market_comparison.rows:
          % interval = row.interval
        <li class="probability-row comparison-row{{' is-peak' if interval == record.peak_interval else ''}}">
          <div class="probability-values probability-values-large">
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
        <span>差值 = 模型 − 市场（百分点）</span>
      </div>
      % if record.market_url:
      <a
        class="external-link"
        href="{{record.market_url}}"
        target="_blank"
        rel="noopener noreferrer"
      >打开对应 Polymarket 市场 ↗</a>
      % end
    </section>

    <section class="detail-panel">
      <header class="panel-heading">
        <div>
          <p class="section-kicker">PROVENANCE</p>
          <h2>预报与修正来源</h2>
        </div>
      </header>
      <div class="metadata-grid">
        <dl class="metadata-card">
          <dt>天气预报</dt>
          <dd>
            <span>模型</span>
            <strong>{{record.model_label}}</strong>
          </dd>
          <dd>
            <span>起报时间</span>
            <strong>{{record.forecast_initialization_text}}</strong>
          </dd>
          <dd>
            <span>公开时间</span>
            <strong>{{record.forecast_availability_text}}</strong>
          </dd>
          <dd>
            <span>Day ahead</span>
            <strong>{{record.forecast_day_ahead}}</strong>
          </dd>
          <dd>
            <span>成员数</span>
            <strong>{{len(record.forecast_members) or "—"}}</strong>
          </dd>
        </dl>
        <dl class="metadata-card">
          <dt>EMOS 修正</dt>
          <dd>
            <span>Artifact 版本</span>
            <strong title="{{record.artifact_version}}">
              {{record.artifact_short_version}}
            </strong>
          </dd>
          <dd>
            <span>参数日期</span>
            <strong>{{record.artifact.get("parameter_date", "—")}}</strong>
          </dd>
          <dd>
            <span>训练样本</span>
            <strong>{{record.artifact_group.get("sample_count", "—")}}</strong>
          </dd>
          <dd>
            <span>训练天数</span>
            <strong>{{record.artifact_group.get("resolved_training_days", "—")}}</strong>
          </dd>
          <dd>
            <span>Fit hash</span>
            <strong class="mono">
              {{str(record.artifact_group.get("fit_sha256", "—"))[:12]}}
            </strong>
          </dd>
        </dl>
      </div>

      <details class="technical-details">
        <summary>查看 EMOS 系数</summary>
        % parameters = record.correction_parameters
        <dl class="coefficient-summary">
          <div><dt>a</dt><dd>{{parameters.get("a", "—")}}</dd></div>
          <div><dt>c</dt><dd>{{parameters.get("c", "—")}}</dd></div>
          <div><dt>d</dt><dd>{{parameters.get("d", "—")}}</dd></div>
          <div><dt>B 成员</dt><dd>{{len(record.b_parameters)}}</dd></div>
        </dl>
        % if record.b_parameters:
        <div class="compact-table-wrapper">
          <table class="compact-table">
            <thead><tr><th>成员</th><th>B 系数</th></tr></thead>
            <tbody>
              % for name, coefficient in record.b_parameters:
              <tr><td>{{name}}</td><td>{{coefficient}}</td></tr>
              % end
            </tbody>
          </table>
        </div>
        % end
      </details>

      % if record.forecast_members:
      <details class="technical-details">
        <summary>查看成员日最高温输入</summary>
        <div class="compact-table-wrapper">
          <table class="compact-table">
            <thead><tr><th>成员</th><th>日最高温</th></tr></thead>
            <tbody>
              % for name, maximum in record.forecast_members:
              <tr>
                <td>{{name}}</td>
                <td>{{f"{maximum:.2f}"}} {{record.forecast.get("input_unit", "")}}</td>
              </tr>
              % end
            </tbody>
          </table>
        </div>
      </details>
      % end
    </section>
  </div>

  <aside class="detail-sidebar">
    <section class="detail-panel sticky-panel">
      <header class="panel-heading">
        <div>
          <p class="section-kicker">REVISION</p>
          <h2>版本信息</h2>
        </div>
      </header>
      <dl class="revision-facts">
        <div><dt>当前修订</dt><dd>{{record.revision}}</dd></div>
        <div><dt>生成时间</dt><dd>{{record.generated_local_text}}</dd></div>
        <div><dt>UTC</dt><dd>{{record.generated_utc_text}}</dd></div>
        <div><dt>Prediction ID</dt><dd class="mono">{{record.short_id}}</dd></div>
        <div><dt>来源</dt><dd>{{record.source}}:{{record.line_number}}</dd></div>
      </dl>

      <h3 class="history-title">该日期的历史修订</h3>
      <ol class="history-list">
        % for historical in history:
        <li class="{{'is-current' if historical.revision == record.revision else ''}}">
          <a href="{{historical.detail_url}}">
            <span>修订 {{historical.revision}}</span>
            <strong>{{historical.peak_interval.display_label}}</strong>
            <small>{{historical.generated_local_text}}</small>
          </a>
        </li>
        % end
      </ol>
    </section>
  </aside>
</div>

<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="light">
  <title>{{page_title}} · EMOS 温度预测</title>
  <link rel="stylesheet" href="/assets/app.css">
</head>
<body>
  <header class="site-header">
    <div class="shell header-inner">
      <a class="brand" href="/" aria-label="返回预测总览">
        <span class="brand-mark" aria-hidden="true">T°</span>
        <span>
          <strong>EMOS 温度预测</strong>
          <small>Daily maximum probability desk</small>
        </span>
      </a>
      <div class="header-status" title="页面在每次请求时重新读取预测文件">
        <span class="status-dot" aria-hidden="true"></span>
        <span>实时读取 JSONL</span>
      </div>
    </div>
    % if site_cities:
    <nav class="city-navigation" aria-label="城市">
      <div class="shell city-navigation-inner">
        <a class="city-link{{' is-active' if active_city is None else ''}}" href="/">
          全部城市
        </a>
        % for city in site_cities:
        <a
          class="city-link{{' is-active' if active_city == city['name'] else ''}}"
          href="{{city['url']}}"
        >{{city["label"]}}</a>
        % end
      </div>
    </nav>
    % end
  </header>

  <main class="shell page-content">
    {{!base}}
  </main>

  <footer class="site-footer">
    <div class="shell footer-inner">
      <span>只读预测面板</span>
      <span>{{record_count}} 条有效记录 · 读取于 {{loaded_at_text}}</span>
    </div>
  </footer>
</body>
</html>

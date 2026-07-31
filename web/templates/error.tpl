% rebase(
%   "base",
%   page_title=page_title,
%   active_city=active_city,
%   site_cities=site_cities,
%   loaded_at_text=loaded_at_text,
%   record_count=record_count,
% )

<section class="empty-state error-state">
  <span class="error-code">{{status_code}}</span>
  <h1>{{message}}</h1>
  <p>请求的预测页面不存在，或者对应版本已经不可用。</p>
  <a class="primary-link" href="/">返回预测总览</a>
</section>

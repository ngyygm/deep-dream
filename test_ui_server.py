#!/usr/bin/env python3
"""
Temporal_Memory_Graph 测试网页服务。
独立端口运行，页面内配置 TMG API 地址后即可：输入文本调用 Remember、自动刷新记忆图谱、测试各类 Find 接口。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

from flask import Flask, send_from_directory

app = Flask(__name__, static_folder=None)

# 内联 HTML，单文件部署
INDEX_HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>TMG 测试 - Remember & Find</title>
  <style>
    :root {
      --bg: #0f0f12;
      --surface: #1a1a20;
      --border: #2a2a32;
      --text: #e4e4e7;
      --muted: #71717a;
      --accent: #a78bfa;
      --ok: #4ade80;
      --err: #f87171;
    }
    * { box-sizing: border-box; }
    body {
      font-family: 'JetBrains Mono', 'SF Mono', Consolas, monospace;
      background: var(--bg);
      color: var(--text);
      margin: 0;
      padding: 1rem 1.5rem;
      line-height: 1.5;
    }
    h1 { font-size: 1.25rem; margin: 0 0 0.5rem; color: var(--accent); }
    h2 { font-size: 1rem; margin: 1rem 0 0.5rem; border-bottom: 1px solid var(--border); padding-bottom: 0.25rem; }
    .row { display: flex; gap: 0.75rem; align-items: center; flex-wrap: wrap; margin-bottom: 0.5rem; }
    label { min-width: 5rem; color: var(--muted); }
    input[type="text"], input[type="number"], textarea {
      background: var(--surface);
      border: 1px solid var(--border);
      color: var(--text);
      padding: 0.4rem 0.6rem;
      border-radius: 6px;
      font-family: inherit;
    }
    textarea { width: 100%; min-height: 80px; resize: vertical; }
    button {
      background: var(--accent);
      color: var(--bg);
      border: none;
      padding: 0.5rem 1rem;
      border-radius: 6px;
      cursor: pointer;
      font-family: inherit;
      font-weight: 600;
    }
    button:hover { filter: brightness(1.1); }
    button.secondary { background: var(--surface); color: var(--text); border: 1px solid var(--border); }
    .card {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 1rem;
      margin-bottom: 1rem;
    }
    .hint { font-size: 0.85rem; color: var(--muted); margin-top: 0.25rem; }
    pre { overflow: auto; max-height: 300px; font-size: 0.8rem; margin: 0.5rem 0; padding: 0.75rem; background: var(--bg); border-radius: 6px; }
    .success { color: var(--ok); }
    .error { color: var(--err); }
    #graphStats { margin-bottom: 0.5rem; }
    .find-section { margin-top: 0.75rem; }
    .find-section summary { cursor: pointer; padding: 0.25rem 0; }
    .find-item { margin: 0.5rem 0; padding: 0.5rem; background: var(--bg); border-radius: 6px; }
    table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
    th, td { text-align: left; padding: 0.35rem 0.5rem; border-bottom: 1px solid var(--border); }
    th { color: var(--muted); }
    .entity-id, .rel-id { font-size: 0.75rem; color: var(--muted); }
  </style>
</head>
<body>
  <h1>TMG 测试页 · Remember & Find</h1>

  <div class="card">
    <h2>1. API 配置</h2>
    <div class="row">
      <label>API 地址</label>
      <input type="text" id="apiBase" value="http://127.0.0.1:16200" placeholder="http://127.0.0.1:16200" style="width: 20rem;" />
      <button type="button" onclick="refreshGraph()">刷新图谱</button>
    </div>
    <p class="hint">确保 service_api 已在该端口运行（如 python service_api.py --config service_config.json）</p>
  </div>

  <div class="card">
    <h2>2. Remember（记忆入库）</h2>
    <div class="row">
      <label>文档名</label>
      <input type="text" id="docName" value="test_ui" placeholder="doc_name" style="width: 12rem;" />
    </div>
    <textarea id="rememberText" placeholder="输入任意文本，点击「入库」后会调用 POST /api/remember，处理完成后自动刷新下方图谱。"></textarea>
    <div class="row" style="margin-top: 0.5rem;">
      <button type="button" id="btnRemember" onclick="submitRemember()">入库 (Remember)</button>
      <span id="rememberStatus"></span>
    </div>
    <p class="hint">请求体: { "text": "…", "doc_name": "…", "load_cache_memory": 可选 }。入库会调 LLM，可能需 1～3 分钟，请耐心等待。</p>
  </div>

  <div class="card">
    <h2>3. 当前记忆图谱</h2>
    <div id="graphStats" class="hint"></div>
    <div class="row">
      <button type="button" class="secondary" onclick="refreshGraph()">刷新</button>
    </div>
    <h3 style="font-size: 0.95rem;">实体 (entities)</h3>
    <div id="entitiesTable"></div>
    <h3 style="font-size: 0.95rem;">关系 (relations)</h3>
    <div id="relationsTable"></div>
  </div>

  <div class="card">
    <h2>4. Find 接口测试</h2>
    <p class="hint">下方每项都标明了需要输入的参数，填写后点击「调用」即可测试。</p>

    <div class="find-section">
      <details>
        <summary><strong>GET /api/find/stats</strong> — 统计</summary>
        <div class="find-item">
          <p class="hint">无需参数。返回 total_entities, total_relations, total_memory_caches。</p>
          <button type="button" onclick="callFind('stats')">调用</button>
          <pre id="res-stats"></pre>
        </div>
      </details>
    </div>

    <div class="find-section">
      <details>
        <summary><strong>GET /api/find/entities/all</strong></summary>
        <div class="find-item">
          <div class="row"><label>limit</label><input type="number" id="find-entities-all-limit" placeholder="可选，如 50" /></div>
          <p class="hint">可选: limit (数量上限)</p>
          <button type="button" onclick="callFind('entities-all')">调用</button>
          <pre id="res-entities-all"></pre>
        </div>
      </details>
    </div>

    <div class="find-section">
      <details>
        <summary><strong>GET /api/find/entities/search</strong> — 语义搜索实体</summary>
        <div class="find-item">
          <div class="row"><label>query_name</label><input type="text" id="find-entities-search-query_name" placeholder="必填" /></div>
          <div class="row"><label>query_content</label><input type="text" id="find-entities-search-query_content" placeholder="可选" /></div>
          <div class="row"><label>threshold</label><input type="text" id="find-entities-search-threshold" value="0.7" /></div>
          <div class="row"><label>max_results</label><input type="number" id="find-entities-search-max_results" value="10" /></div>
          <p class="hint">必填: query_name。可选: query_content, threshold, max_results, text_mode, similarity_method</p>
          <button type="button" onclick="callFind('entities-search')">调用</button>
          <pre id="res-entities-search"></pre>
        </div>
      </details>
    </div>

    <div class="find-section">
      <details>
        <summary><strong>GET /api/find/entities/&lt;entity_id&gt;</strong> — 按 entity_id 查最新版本</summary>
        <div class="find-item">
          <div class="row"><label>entity_id</label><input type="text" id="find-entity-by-id" placeholder="如 person_张三" /></div>
          <p class="hint">路径参数: entity_id（逻辑 id，非 absolute_id）</p>
          <button type="button" onclick="callFind('entity-by-id')">调用</button>
          <pre id="res-entity-by-id"></pre>
        </div>
      </details>
    </div>

    <div class="find-section">
      <details>
        <summary><strong>GET /api/find/entities/&lt;entity_id&gt;/versions</strong></summary>
        <div class="row"><label>entity_id</label><input type="text" id="find-entity-versions-id" placeholder="entity_id" /></div>
        <p class="hint">路径参数: entity_id</p>
        <button type="button" onclick="callFind('entity-versions')">调用</button>
        <pre id="res-entity-versions"></pre>
      </details>
    </div>

    <div class="find-section">
      <details>
        <summary><strong>GET /api/find/entities/all-before-time</strong></summary>
        <div class="find-item">
          <div class="row"><label>time_point</label><input type="text" id="find-all-before-time_point" placeholder="ISO 如 2025-03-06T12:00:00" /></div>
          <div class="row"><label>limit</label><input type="number" id="find-all-before-limit" placeholder="可选" /></div>
          <p class="hint">必填: time_point (ISO 格式)。可选: limit</p>
          <button type="button" onclick="callFind('entities-all-before-time')">调用</button>
          <pre id="res-entities-all-before-time"></pre>
        </div>
      </details>
    </div>

    <div class="find-section">
      <details>
        <summary><strong>GET /api/find/relations/all</strong></summary>
        <div class="find-item">
          <p class="hint">无需参数。</p>
          <button type="button" onclick="callFind('relations-all')">调用</button>
          <pre id="res-relations-all"></pre>
        </div>
      </details>
    </div>

    <div class="find-section">
      <details>
        <summary><strong>GET /api/find/relations/search</strong> — 按内容相似度搜关系</summary>
        <div class="find-item">
          <div class="row"><label>query_text</label><input type="text" id="find-relations-search-query_text" placeholder="必填" /></div>
          <div class="row"><label>threshold</label><input type="text" id="find-relations-search-threshold" value="0.3" /></div>
          <div class="row"><label>max_results</label><input type="number" id="find-relations-search-max_results" value="10" /></div>
          <p class="hint">必填: query_text。可选: threshold, max_results</p>
          <button type="button" onclick="callFind('relations-search')">调用</button>
          <pre id="res-relations-search"></pre>
        </div>
      </details>
    </div>

    <div class="find-section">
      <details>
        <summary><strong>GET /api/find/relations/between</strong></summary>
        <div class="find-item">
          <div class="row"><label>from_entity_id</label><input type="text" id="find-relations-between-from" placeholder="entity_id" /></div>
          <div class="row"><label>to_entity_id</label><input type="text" id="find-relations-between-to" placeholder="entity_id" /></div>
          <p class="hint">必填: from_entity_id, to_entity_id（均为 entity_id）</p>
          <button type="button" onclick="callFind('relations-between')">调用</button>
          <pre id="res-relations-between"></pre>
        </div>
      </details>
    </div>

    <div class="find-section">
      <details>
        <summary><strong>POST /api/find/query-one</strong> — 按条件查子图（一次性）</summary>
        <div class="find-item">
          <div class="row"><label>entity_name / query_text</label><input type="text" id="find-query-one-entity_name" placeholder="可选，语义筛选实体" /></div>
          <div class="row"><label>time_before</label><input type="text" id="find-query-one-time_before" placeholder="ISO 可选" /></div>
          <div class="row"><label>time_after</label><input type="text" id="find-query-one-time_after" placeholder="ISO 可选" /></div>
          <div class="row"><label>max_entities</label><input type="number" id="find-query-one-max_entities" value="100" /></div>
          <div class="row"><label>max_relations</label><input type="number" id="find-query-one-max_relations" value="500" /></div>
          <p class="hint">Body: entity_name/query_text, time_before, time_after, max_entities, max_relations, include_entities, include_relations</p>
          <button type="button" onclick="callFind('query-one')">调用</button>
          <pre id="res-query-one"></pre>
        </div>
      </details>
    </div>

    <div class="find-section">
      <details>
        <summary><strong>GET /api/find/memory-cache/latest</strong></summary>
        <div class="find-item">
          <div class="row"><label>activity_type</label><input type="text" id="find-memory-latest-activity_type" placeholder="可选" /></div>
          <p class="hint">可选: activity_type</p>
          <button type="button" onclick="callFind('memory-cache-latest')">调用</button>
          <pre id="res-memory-cache-latest"></pre>
        </div>
      </details>
    </div>

    <div class="find-section">
      <details>
        <summary><strong>GET /api/find/memory-cache/&lt;cache_id&gt;/text</strong></summary>
        <div class="find-item">
          <div class="row"><label>cache_id</label><input type="text" id="find-memory-cache-id" placeholder="如 cache_20260306_152611_xxx" /></div>
          <p class="hint">路径参数: cache_id</p>
          <button type="button" onclick="callFind('memory-cache-text')">调用</button>
          <pre id="res-memory-cache-text"></pre>
        </div>
      </details>
    </div>
  </div>

  <script>
    function base() { return (document.getElementById('apiBase').value || '').replace(/\/$/, ''); }

    function setRememberStatus(msg, isErr) {
      var el = document.getElementById('rememberStatus');
      el.textContent = msg;
      el.className = isErr ? 'error' : 'success';
    }

    var REMEMBER_TIMEOUT_MS = 300000;

    async function submitRemember() {
      var text = (document.getElementById('rememberText').value || '').trim();
      if (!text) { setRememberStatus('请输入文本', true); return; }
      var docName = (document.getElementById('docName').value || 'test_ui').trim() || 'test_ui';
      var btn = document.getElementById('btnRemember');
      if (btn) btn.disabled = true;
      setRememberStatus('处理中…（会调 LLM，可能 1～3 分钟，请勿关闭页面）', false);
      var controller = new AbortController();
      var toId = setTimeout(function() { controller.abort(); }, REMEMBER_TIMEOUT_MS);
      try {
        var r = await fetch(base() + '/api/remember', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text: text, doc_name: docName }),
          signal: controller.signal
        });
        clearTimeout(toId);
        var j;
        try {
          j = await r.json();
        } catch (_) {
          var t = await r.text();
          setRememberStatus('服务返回非 JSON: ' + r.status + ' ' + (t.slice(0, 100) || ''), true);
          if (btn) btn.disabled = false;
          return;
        }
        if (j.success) {
          setRememberStatus('入库成功，正在刷新图谱…', false);
          await refreshGraph();
          setRememberStatus('完成', false);
        } else {
          setRememberStatus('失败: ' + (j.error || r.status), true);
        }
      } catch (e) {
        clearTimeout(toId);
        if (e.name === 'AbortError') {
          setRememberStatus('请求超时（5 分钟）。请检查 TMG 服务与 LLM 是否正常、或缩短文本再试。', true);
        } else {
          setRememberStatus('请求错误: ' + (e.message || String(e)), true);
        }
      }
      if (btn) btn.disabled = false;
    }

    async function refreshGraph() {
      var statsEl = document.getElementById('graphStats');
      var entitiesEl = document.getElementById('entitiesTable');
      var relationsEl = document.getElementById('relationsTable');
      statsEl.textContent = '加载中…';
      entitiesEl.innerHTML = '';
      relationsEl.innerHTML = '';
      try {
        var [statsRes, entitiesRes, relationsRes] = await Promise.all([
          fetch(base() + '/api/find/stats'),
          fetch(base() + '/api/find/entities/all?limit=200'),
          fetch(base() + '/api/find/relations/all')
        ]);
        var stats = statsRes.ok ? (await statsRes.json()) : null;
        var entities = entitiesRes.ok ? (await entitiesRes.json()) : null;
        var relations = relationsRes.ok ? (await relationsRes.json()) : null;

        if (stats && stats.success) {
          var d = stats.data;
          statsEl.textContent = '实体: ' + (d.total_entities || 0) + '，关系: ' + (d.total_relations || 0) + '，记忆缓存: ' + (d.total_memory_caches || 0);
        } else {
          statsEl.textContent = '获取 stats 失败';
          statsEl.className = 'error';
        }

        if (entities && entities.success && Array.isArray(entities.data)) {
          var rows = entities.data.slice(0, 100).map(function(e) {
            return '<tr><td>' + (e.name || '') + '</td><td class="entity-id">' + (e.entity_id || '') + '</td><td>' + (e.content || '').slice(0, 80) + '</td></tr>';
          }).join('');
          entitiesEl.innerHTML = '<table><thead><tr><th>name</th><th>entity_id</th><th>content</th></tr></thead><tbody>' + rows + '</tbody></table>';
          if (entities.data.length > 100) entitiesEl.innerHTML += '<p class="hint">仅显示前 100 条，共 ' + entities.data.length + ' 条</p>';
        } else {
          entitiesEl.innerHTML = '<p class="hint">无实体或请求失败</p>';
        }

        if (relations && relations.success && Array.isArray(relations.data)) {
          var rows = relations.data.slice(0, 100).map(function(r) {
            return '<tr><td class="rel-id">' + (r.relation_id || '') + '</td><td>' + (r.entity1_absolute_id || '') + '</td><td>' + (r.entity2_absolute_id || '') + '</td><td>' + (r.content || '').slice(0, 60) + '</td></tr>';
          }).join('');
          relationsEl.innerHTML = '<table><thead><tr><th>relation_id</th><th>entity1</th><th>entity2</th><th>content</th></tr></thead><tbody>' + rows + '</tbody></table>';
          if (relations.data.length > 100) relationsEl.innerHTML += '<p class="hint">仅显示前 100 条，共 ' + relations.data.length + ' 条</p>';
        } else {
          relationsEl.innerHTML = '<p class="hint">无关系或请求失败</p>';
        }
      } catch (e) {
        statsEl.textContent = '刷新失败: ' + e.message;
        statsEl.className = 'error';
      }
    }

    function getVal(id) { var el = document.getElementById(id); return el ? (el.value || '').trim() : ''; }
    function setRes(id, text) { var el = document.getElementById(id); if (el) el.textContent = text; }

    async function callFind(kind) {
      var baseUrl = base();
      var resId = 'res-' + kind.replace(/\//g, '-');
      setRes(resId, '请求中…');
      try {
        var url, opts = {};
        switch (kind) {
          case 'stats':
            url = baseUrl + '/api/find/stats';
            break;
          case 'entities-all':
            url = baseUrl + '/api/find/entities/all';
            var lim = getVal('find-entities-all-limit');
            if (lim) url += '?limit=' + encodeURIComponent(lim);
            break;
          case 'entities-search':
            var qn = getVal('find-entities-search-query_name');
            if (!qn) { setRes(resId, '请填写 query_name'); return; }
            url = baseUrl + '/api/find/entities/search?query_name=' + encodeURIComponent(qn);
            var qc = getVal('find-entities-search-query_content');
            if (qc) url += '&query_content=' + encodeURIComponent(qc);
            url += '&threshold=' + encodeURIComponent(getVal('find-entities-search-threshold') || '0.7');
            url += '&max_results=' + encodeURIComponent(getVal('find-entities-search-max_results') || '10');
            break;
          case 'entity-by-id':
            var eid = getVal('find-entity-by-id');
            if (!eid) { setRes(resId, '请填写 entity_id'); return; }
            url = baseUrl + '/api/find/entities/' + encodeURIComponent(eid);
            break;
          case 'entity-versions':
            var evId = getVal('find-entity-versions-id');
            if (!evId) { setRes(resId, '请填写 entity_id'); return; }
            url = baseUrl + '/api/find/entities/' + encodeURIComponent(evId) + '/versions';
            break;
          case 'entities-all-before-time':
            var tp = getVal('find-all-before-time_point');
            if (!tp) { setRes(resId, '请填写 time_point (ISO)'); return; }
            url = baseUrl + '/api/find/entities/all-before-time?time_point=' + encodeURIComponent(tp);
            var lim2 = getVal('find-all-before-limit');
            if (lim2) url += '&limit=' + encodeURIComponent(lim2);
            break;
          case 'relations-all':
            url = baseUrl + '/api/find/relations/all';
            break;
          case 'relations-search':
            var qt = getVal('find-relations-search-query_text');
            if (!qt) { setRes(resId, '请填写 query_text'); return; }
            url = baseUrl + '/api/find/relations/search?query_text=' + encodeURIComponent(qt) + '&threshold=' + encodeURIComponent(getVal('find-relations-search-threshold') || '0.3') + '&max_results=' + encodeURIComponent(getVal('find-relations-search-max_results') || '10');
            break;
          case 'relations-between':
            var fromId = getVal('find-relations-between-from');
            var toId = getVal('find-relations-between-to');
            if (!fromId || !toId) { setRes(resId, '请填写 from_entity_id 和 to_entity_id'); return; }
            url = baseUrl + '/api/find/relations/between?from_entity_id=' + encodeURIComponent(fromId) + '&to_entity_id=' + encodeURIComponent(toId);
            break;
          case 'query-one':
            url = baseUrl + '/api/find/query-one';
            opts = { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' };
            var body = {
              max_entities: parseInt(getVal('find-query-one-max_entities') || '100', 10) || 100,
              max_relations: parseInt(getVal('find-query-one-max_relations') || '500', 10) || 500,
              include_entities: true,
              include_relations: true
            };
            var en = getVal('find-query-one-entity_name');
            if (en) body.entity_name = en;
            var tb = getVal('find-query-one-time_before');
            if (tb) body.time_before = tb;
            var ta = getVal('find-query-one-time_after');
            if (ta) body.time_after = ta;
            opts.body = JSON.stringify(body);
            break;
          case 'memory-cache-latest':
            url = baseUrl + '/api/find/memory-cache/latest';
            var at = getVal('find-memory-latest-activity_type');
            if (at) url += '?activity_type=' + encodeURIComponent(at);
            break;
          case 'memory-cache-text':
            var cid = getVal('find-memory-cache-id');
            if (!cid) { setRes(resId, '请填写 cache_id'); return; }
            url = baseUrl + '/api/find/memory-cache/' + encodeURIComponent(cid) + '/text';
            break;
          default:
            setRes(resId, '未知: ' + kind);
            return;
        }
        if (!opts.method) opts.method = 'GET';
        var r = await fetch(url, opts);
        var j = await r.json().catch(function() { return { error: '非 JSON 或空' }; });
        setRes(resId, JSON.stringify(j, null, 2));
      } catch (e) {
        setRes(resId, '错误: ' + e.message);
      }
    }
  </script>
</body>
</html>
"""


@app.route("/")
def index():
    return INDEX_HTML


def main() -> int:
    parser = argparse.ArgumentParser(description="TMG 测试网页服务（独立端口）")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="监听 host")
    parser.add_argument("--port", type=int, default=5050, help="监听 port")
    parser.add_argument("--debug", action="store_true", help="Flask debug")
    args = parser.parse_args()
    print("TMG 测试页: http://{}:{}".format(args.host, args.port))
    print("请先启动 service_api，并在页面中配置 API 地址（默认 16200）")
    app.run(host=args.host, port=args.port, debug=args.debug)
    return 0


if __name__ == "__main__":
    sys.exit(main())

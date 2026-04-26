/* ==========================================
   API Test Page
   ========================================== */

(function() {
  // ---- API endpoint catalog ----
  const API_CATALOG = [
    {
      category: 'System',
      endpoints: [
        {
          name: 'Health Check',
          method: 'GET',
          path: '/api/v1/health',
          params: [
            { name: 'graph_id', type: 'text', default: 'default' },
          ],
        },
        {
          name: 'System Overview',
          method: 'GET',
          path: '/api/v1/system/overview',
          params: [],
        },
        {
          name: 'System Graphs',
          method: 'GET',
          path: '/api/v1/system/graphs',
          params: [],
        },
        {
          name: 'Graph Detail',
          method: 'GET',
          path: '/api/v1/system/graphs/{graph_id}',
          params: [
            { name: 'graph_id', type: 'text', default: 'default', pathParam: true },
          ],
        },
        {
          name: 'System Tasks',
          method: 'GET',
          path: '/api/v1/system/tasks',
          params: [
            { name: 'limit', type: 'number', default: 50 },
          ],
        },
        {
          name: 'System Logs',
          method: 'GET',
          path: '/api/v1/system/logs',
          params: [
            { name: 'limit', type: 'number', default: 50 },
            { name: 'level', type: 'select', options: ['', 'INFO', 'WARNING', 'ERROR'], default: '' },
            { name: 'source', type: 'text', default: '' },
          ],
        },
        {
          name: 'Access Stats',
          method: 'GET',
          path: '/api/v1/system/access-stats',
          params: [
            { name: 'since_seconds', type: 'number', default: 300 },
          ],
        },
      ],
    },
    {
      category: 'Graphs',
      endpoints: [
        {
          name: 'List Graphs',
          method: 'GET',
          path: '/api/v1/graphs',
          params: [],
        },
        {
          name: 'Find Stats',
          method: 'GET',
          path: '/api/v1/find/stats',
          params: [
            { name: 'graph_id', type: 'text', default: 'default' },
          ],
        },
      ],
    },
    {
      category: 'Remember',
      endpoints: [
        {
          name: 'Remember Text',
          method: 'POST',
          path: '/api/v1/remember',
          contentType: 'json',
          params: [
            { name: 'graph_id', type: 'text', default: 'default' },
            { name: 'text', type: 'textarea', default: '' },
            { name: 'source_name', type: 'text', default: '' },
            { name: 'event_time', type: 'datetime-local', default: '' },
            { name: 'load_cache_memory', type: 'checkbox', default: false },
          ],
        },
        {
          name: 'Remember File',
          method: 'POST',
          path: '/api/v1/remember',
          contentType: 'multipart',
          params: [
            { name: 'graph_id', type: 'text', default: 'default' },
            { name: 'file', type: 'file' },
            { name: 'source_name', type: 'text', default: '' },
            { name: 'event_time', type: 'datetime-local', default: '' },
          ],
          note: 'File upload is read-only in API test. Use Memory page for uploads.',
        },
        {
          name: 'Remember Tasks',
          method: 'GET',
          path: '/api/v1/remember/tasks',
          params: [
            { name: 'graph_id', type: 'text', default: 'default' },
            { name: 'limit', type: 'number', default: 50 },
          ],
        },
        {
          name: 'Task Status',
          method: 'GET',
          path: '/api/v1/remember/tasks/{task_id}',
          params: [
            { name: 'task_id', type: 'text', default: '', pathParam: true },
            { name: 'graph_id', type: 'text', default: 'default' },
          ],
        },
      ],
    },
    {
      category: 'Find',
      endpoints: [
        {
          name: 'Semantic Find',
          method: 'POST',
          path: '/api/v1/find',
          contentType: 'json',
          params: [
            { name: 'query', type: 'text', default: '' },
            { name: 'graph_id', type: 'text', default: 'default' },
            { name: 'similarity_threshold', type: 'number', default: 0.5 },
            { name: 'max_entities', type: 'number', default: 20 },
            { name: 'max_relations', type: 'number', default: 50 },
            { name: 'expand', type: 'checkbox', default: true },
            { name: 'time_before', type: 'datetime-local', default: '' },
            { name: 'time_after', type: 'datetime-local', default: '' },
          ],
        },
      ],
    },
    {
      category: 'Entities',
      endpoints: [
        {
          name: 'List Entities',
          method: 'GET',
          path: '/api/v1/find/entities',
          params: [
            { name: 'graph_id', type: 'text', default: 'default' },
            { name: 'limit', type: 'number', default: '' },
          ],
        },
        {
          name: 'Search Entities',
          method: 'POST',
          path: '/api/v1/find/entities/search',
          contentType: 'json',
          params: [
            { name: 'query_name', type: 'text', default: '' },
            { name: 'graph_id', type: 'text', default: 'default' },
            { name: 'query_content', type: 'text', default: '' },
            { name: 'similarity_threshold', type: 'number', default: 0.7 },
            { name: 'max_results', type: 'number', default: 20 },
            { name: 'text_mode', type: 'select', options: ['name_and_content', 'name_only', 'content_only'], default: 'name_and_content' },
            { name: 'similarity_method', type: 'select', options: ['embedding', 'keyword'], default: 'embedding' },
          ],
        },
        {
          name: 'Entity Versions',
          method: 'GET',
          path: '/api/v1/find/entities/{family_id}/versions',
          params: [
            { name: 'family_id', type: 'text', default: '', pathParam: true },
            { name: 'graph_id', type: 'text', default: 'default' },
          ],
        },
        {
          name: 'Entity Relations',
          method: 'GET',
          path: '/api/v1/find/entities/{family_id}/relations',
          params: [
            { name: 'family_id', type: 'text', default: '', pathParam: true },
            { name: 'graph_id', type: 'text', default: 'default' },
            { name: 'limit', type: 'number', default: '' },
          ],
        },
      ],
    },
    {
      category: 'Relations',
      endpoints: [
        {
          name: 'List Relations',
          method: 'GET',
          path: '/api/v1/find/relations',
          params: [
            { name: 'graph_id', type: 'text', default: 'default' },
            { name: 'limit', type: 'number', default: '' },
            { name: 'offset', type: 'number', default: '' },
          ],
        },
        {
          name: 'Search Relations',
          method: 'POST',
          path: '/api/v1/find/relations/search',
          contentType: 'json',
          params: [
            { name: 'query_text', type: 'text', default: '' },
            { name: 'graph_id', type: 'text', default: 'default' },
            { name: 'similarity_threshold', type: 'number', default: 0.3 },
            { name: 'max_results', type: 'number', default: 20 },
          ],
        },
        {
          name: 'Relations Between',
          method: 'POST',
          path: '/api/v1/find/relations/between',
          contentType: 'json',
          params: [
            { name: 'family_id_a', type: 'text', default: '' },
            { name: 'family_id_b', type: 'text', default: '' },
            { name: 'graph_id', type: 'text', default: 'default' },
          ],
        },
      ],
    },
    {
      category: 'Docs',
      endpoints: [
        {
          name: 'List Documents',
          method: 'GET',
          path: '/api/v1/docs',
          params: [
            { name: 'graph_id', type: 'text', default: 'default' },
          ],
        },
      ],
    },
  ];

  // ---- i18n keys ----
  const i18n = {
    title: () => t('apiTest.title'),
    sendRequest: () => t('apiTest.sendRequest'),
    sending: () => t('apiTest.sending'),
    response: () => t('apiTest.response'),
    request: () => t('apiTest.request'),
    time: () => t('apiTest.time'),
    curlCommand: () => t('apiTest.curlCommand'),
    copy: () => t('common.copy'),
    copied: () => t('common.copied'),
    noEndpoint: () => t('apiTest.noEndpoint'),
    params: () => t('apiTest.params'),
    status: () => t('apiTest.status'),
    body: () => t('apiTest.body'),
    noResponse: () => t('apiTest.noResponse'),
    pathParams: () => t('apiTest.pathParams'),
  };

  // ---- Local state ----
  let selectedEndpoint = null;

  // ---- Method badge color ----
  function methodBadge(method) {
    const colors = {
      GET: 'badge-success',
      POST: 'badge-primary',
      PUT: 'badge-warning',
      DELETE: 'badge-error',
      PATCH: 'badge-info',
    };
    return `<span class="badge ${colors[method] || 'badge-primary'}" style="font-size:0.7rem;min-width:40px;text-align:center;">${method}</span>`;
  }

  // ---- Render endpoint list ----
  function renderEndpointList() {
    return API_CATALOG.map(cat => `
      <div class="endpoint-category" data-category="${escapeHtml(cat.category)}">
        <button class="endpoint-category-header" data-category-toggle="${escapeHtml(cat.category)}">
          <i data-lucide="chevron-down" style="width:14px;height:14px;"></i>
          <span style="font-size:0.8rem;font-weight:600;">${escapeHtml(cat.category)}</span>
          <span class="badge badge-secondary" style="margin-left:4px;">${cat.endpoints.length}</span>
        </button>
        <div class="endpoint-category-list" data-category-list="${escapeHtml(cat.category)}">
          ${cat.endpoints.map(ep => `
            <button class="endpoint-item ${selectedEndpoint === ep ? 'active' : ''}" data-endpoint-index="${API_CATALOG.indexOf(cat)}-${cat.endpoints.indexOf(ep)}">
              ${methodBadge(ep.method)}
              <span style="font-size:0.78rem;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${escapeHtml(ep.name)}</span>
            </button>
          `).join('')}
        </div>
      </div>
    `).join('');
  }

  // ---- Render form for selected endpoint ----
  function renderEndpointForm() {
    if (!selectedEndpoint) {
      return `
        <div class="card h-full">
          <div style="padding:40px;">
            ${emptyState(i18n.noEndpoint(), 'mouse-pointer-click')}
          </div>
        </div>
      `;
    }

    const ep = selectedEndpoint;
    const pathParams = ep.params.filter(p => p.pathParam);
    const queryParams = ep.params.filter(p => !p.pathParam);

    return `
      <div class="card" style="margin-bottom:12px;">
        <div class="card-header" style="padding:12px 16px;">
          <div style="display:flex;align-items:center;gap:8px;">
            ${methodBadge(ep.method)}
            <code class="mono" style="font-size:0.82rem;color:var(--text-primary);">${escapeHtml(ep.path)}</code>
          </div>
        </div>
        ${ep.note ? `<div style="padding:0 16px 8px;font-size:0.75rem;color:var(--text-muted);">${escapeHtml(ep.note)}</div>` : ''}

        ${pathParams.length > 0 ? `
          <div style="padding:8px 16px 0;">
            <div style="font-size:0.75rem;font-weight:600;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.5px;margin-bottom:6px;">${i18n.pathParams()}</div>
            <div style="display:flex;flex-direction:column;gap:8px;">
              ${pathParams.map(p => renderFormField(p, 'path')).join('')}
            </div>
          </div>
        ` : ''}

        ${queryParams.length > 0 ? `
          <div style="padding:8px 16px 0;">
            <div style="font-size:0.75rem;font-weight:600;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.5px;margin-bottom:6px;">${i18n.params()}</div>
            <div style="display:flex;flex-direction:column;gap:8px;">
              ${queryParams.map(p => renderFormField(p, ep.method === 'GET' ? 'query' : 'body')).join('')}
            </div>
          </div>
        ` : ''}

        <div style="padding:12px 16px;display:flex;gap:8px;">
          <button class="btn btn-primary" id="api-send-btn" style="flex:1;">
            <i data-lucide="send" style="width:14px;height:14px;margin-right:6px;"></i>
            ${i18n.sendRequest()}
          </button>
        </div>
      </div>

      <!-- Response area -->
      <div class="card">
        <div class="card-header" style="padding:10px 16px;">
          <span style="font-size:0.8rem;font-weight:600;">${i18n.response()}</span>
          <div id="response-meta" style="display:none;display:flex;align-items:center;gap:8px;">
            <span id="response-status" class="badge"></span>
            <span id="response-time" style="font-size:0.75rem;color:var(--text-muted);"></span>
          </div>
        </div>
        <div id="response-content" style="padding:0 16px 16px;max-height:300px;overflow-y:auto;">
          <pre style="font-size:0.8rem;color:var(--text-muted);white-space:pre-wrap;">${i18n.noResponse()}</pre>
        </div>
      </div>
    `;
  }

  // ---- Render a single form field ----
  function renderFormField(param, location) {
    const id = `api-param-${param.name}`;
    const label = `<label class="form-label" style="margin-bottom:2px;font-size:0.78rem;">${escapeHtml(param.name)}</label>`;

    switch (param.type) {
      case 'textarea':
        return `
          <div>
            ${label}
            <textarea id="${id}" class="input" rows="3" placeholder="${escapeHtml(param.name)}" data-location="${location}" data-name="${escapeHtml(param.name)}" data-type="text">${escapeHtml(param.default || '')}</textarea>
          </div>
        `;
      case 'checkbox':
        return `
          <div style="display:flex;align-items:center;gap:8px;">
            <label class="toggle">
              <input type="checkbox" id="${id}" ${param.default ? 'checked' : ''} data-location="${location}" data-name="${escapeHtml(param.name)}" data-type="checkbox">
              <span class="toggle-slider"></span>
            </label>
            ${label}
          </div>
        `;
      case 'select':
        return `
          <div>
            ${label}
            <select id="${id}" class="input" data-location="${location}" data-name="${escapeHtml(param.name)}" data-type="text">
              ${param.options.map(o => `<option value="${escapeHtml(o)}" ${o === param.default ? 'selected' : ''}>${o || '—'}</option>`).join('')}
            </select>
          </div>
        `;
      case 'number':
        return `
          <div>
            ${label}
            <input type="number" id="${id}" class="input" value="${param.default || ''}" data-location="${location}" data-name="${escapeHtml(param.name)}" data-type="number">
          </div>
        `;
      case 'file':
        return `
          <div>
            ${label}
            <input type="file" id="${id}" class="input" data-location="${location}" data-name="${escapeHtml(param.name)}" data-type="file">
          </div>
        `;
      default:
        return `
          <div>
            ${label}
            <input type="text" id="${id}" class="input" value="${escapeHtml(param.default || '')}" data-location="${location}" data-name="${escapeHtml(param.name)}" data-type="text">
          </div>
        `;
    }
  }

  // ---- Collect form values ----
  function collectFormValues() {
    const pathValues = {};
    const queryValues = {};
    const bodyValues = {};
    const fileValues = {};

    document.querySelectorAll('[data-location][data-name]').forEach(el => {
      const name = el.dataset.name;
      const location = el.dataset.location;
      const type = el.dataset.type;

      let value;
      if (type === 'checkbox') {
        value = el.checked;
      } else if (type === 'number') {
        value = el.value ? parseFloat(el.value) : '';
      } else if (type === 'file') {
        value = el.files[0] || null;
        fileValues[name] = value;
        return;
      } else {
        value = el.value;
      }

      if (value === '' || value === undefined || value === null) return;

      if (location === 'path') pathValues[name] = value;
      else if (location === 'query') queryValues[name] = value;
      else bodyValues[name] = value;
    });

    return { pathValues, queryValues, bodyValues, fileValues };
  }

  // ---- Build URL from endpoint + values ----
  function buildUrl(ep, values) {
    let url = ep.path;
    // Replace path params
    Object.keys(values.pathValues).forEach(key => {
      url = url.replace(`{${key}}`, encodeURIComponent(values.pathValues[key]));
    });
    // Append query params
    const queryParams = Object.keys(values.queryValues);
    if (queryParams.length > 0) {
      const qs = queryParams
        .filter(k => values.queryValues[k] !== '' && values.queryValues[k] !== undefined)
        .map(k => `${encodeURIComponent(k)}=${encodeURIComponent(values.queryValues[k])}`)
        .join('&');
      if (qs) url += '?' + qs;
    }
    return url;
  }

  // ---- Generate curl command ----
  function generateCurl(ep, values) {
    const url = buildUrl(ep, values);
    const fullUrl = window.location.origin + url;
    let cmd = `curl -X ${ep.method} '${fullUrl}'`;

    if (ep.method === 'POST' || ep.method === 'PUT' || ep.method === 'PATCH') {
      if (ep.contentType === 'multipart') {
        cmd += ` \\\n  -F "graph_id=${values.bodyValues.graph_id || 'default'}"`;
        if (values.bodyValues.source_name) cmd += ` \\\n  -F "source_name=${values.bodyValues.source_name}"`;
        if (values.bodyValues.event_time) cmd += ` \\\n  -F "event_time=${values.bodyValues.event_time}"`;
        cmd += ` \\\n  -F "file=@/path/to/file"`;
      } else {
        cmd += ` \\\n  -H 'Content-Type: application/json'`;
        const body = JSON.stringify(values.bodyValues, null, 2);
        cmd += ` \\\n  -d '${body}'`;
      }
    }

    return cmd;
  }

  // ---- Syntax highlight JSON ----
  function syntaxHighlight(json) {
    if (typeof json !== 'string') json = JSON.stringify(json, null, 2);
    return json
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/"(\\u[a-zA-Z0-9]{4}|\\[^u]|[^\\"])*"(\s*:)?/g, (match) => {
        let cls = 'json-string'; // string value
        if (/:\s*$/.test(match)) {
          cls = 'json-key'; // key
        } else if (/^"https?:/.test(match)) {
          cls = 'json-url';
        }
        return `<span class="${cls}">${match}</span>`;
      })
      .replace(/\b(true|false)\b/g, '<span class="json-bool">$&</span>')
      .replace(/\b(null)\b/g, '<span class="json-null">$&</span>')
      .replace(/\b(-?\d+\.?\d*([eE][+-]?\d+)?)\b/g, '<span class="json-number">$&</span>');
  }

  // ---- Execute API request ----
  async function executeRequest() {
    if (!selectedEndpoint) return;

    const ep = selectedEndpoint;
    const values = collectFormValues();
    const url = buildUrl(ep, values);
    const btn = document.getElementById('api-send-btn');
    const responseMeta = document.getElementById('response-meta');
    const responseStatus = document.getElementById('response-status');
    const responseTime = document.getElementById('response-time');
    const responseContent = document.getElementById('response-content');

    // Update button state
    if (btn) {
      btn.disabled = true;
      btn.innerHTML = `${spinnerHtml('spinner-sm')} ${i18n.sending()}`;
    }

    if (responseContent) {
      responseContent.innerHTML = `<pre style="font-size:0.8rem;color:var(--text-muted);">${spinnerHtml('spinner-sm')} ${t('common.loading')}</pre>`;
    }

    const startTime = performance.now();

    try {
      const options = { method: ep.method };

      if ((ep.method === 'POST' || ep.method === 'PUT' || ep.method === 'PATCH') && ep.contentType !== 'multipart') {
        options.headers = { 'Content-Type': 'application/json' };
        options.body = JSON.stringify(values.bodyValues);
      } else if (ep.contentType === 'multipart') {
        const fd = new FormData();
        Object.keys(values.bodyValues).forEach(k => {
          if (values.bodyValues[k] !== '' && values.bodyValues[k] !== undefined) {
            fd.append(k, values.bodyValues[k]);
          }
        });
        Object.keys(values.fileValues).forEach(k => {
          if (values.fileValues[k]) fd.append(k, values.fileValues[k]);
        });
        options.body = fd;
      }

      const res = await fetch(url, options);
      const elapsed = ((performance.now() - startTime) / 1000).toFixed(3);
      let data;
      try { data = await res.json(); } catch { data = await res.text(); }

      // Update response meta
      if (responseMeta) responseMeta.style.display = 'flex';
      if (responseStatus) {
        const statusColors = { 2: 'badge-success', 3: 'badge-success', 4: 'badge-warning', 5: 'badge-error' };
        const colorClass = statusColors[Math.floor(res.status / 100)] || 'badge-primary';
        responseStatus.className = `badge ${colorClass}`;
        responseStatus.textContent = `${res.status} ${res.statusText}`;
      }
      if (responseTime) responseTime.textContent = `${elapsed}s`;

      // Render response
      if (responseContent) {
        const jsonStr = typeof data === 'string' ? data : JSON.stringify(data, null, 2);
        responseContent.innerHTML = `<pre style="font-size:0.78rem;line-height:1.5;white-space:pre-wrap;word-break:break-all;">${syntaxHighlight(jsonStr)}</pre>`;
      }
    } catch (err) {
      const elapsed = ((performance.now() - startTime) / 1000).toFixed(3);
      if (responseMeta) responseMeta.style.display = 'flex';
      if (responseStatus) {
        responseStatus.className = 'badge badge-error';
        responseStatus.textContent = 'Error';
      }
      if (responseTime) responseTime.textContent = `${elapsed}s`;
      if (responseContent) {
        responseContent.innerHTML = `<pre style="font-size:0.8rem;color:var(--error);">${escapeHtml(err.message)}</pre>`;
      }
    } finally {
      if (btn) {
        btn.disabled = false;
        btn.innerHTML = `<i data-lucide="send" style="width:14px;height:14px;margin-right:6px;"></i>${i18n.sendRequest()}`;
        if (window.lucide) lucide.createIcons({ nodes: [btn] });
      }
    }
  }

  // ---- Render curl panel ----
  function renderCurlPanel() {
    if (!selectedEndpoint) return '';
    const values = collectFormValues();
    const curl = generateCurl(selectedEndpoint, values);
    return `
      <div class="card" style="margin-bottom:12px;">
        <div class="card-header" style="padding:10px 16px;display:flex;align-items:center;justify-content:space-between;">
          <span style="font-size:0.8rem;font-weight:600;">${i18n.curlCommand()}</span>
          <button class="btn btn-ghost btn-sm" id="copy-curl-btn" style="color:var(--text-muted);">
            <i data-lucide="copy" style="width:14px;height:14px;margin-right:4px;"></i>
            ${i18n.copy()}
          </button>
        </div>
        <div style="padding:0 16px 12px;">
          <pre id="curl-output" class="mono" style="font-size:0.75rem;background:var(--bg-input);padding:12px;border-radius:6px;white-space:pre-wrap;word-break:break-all;color:var(--text-secondary);border:1px solid var(--border-color);overflow-x:auto;max-height:200px;">${escapeHtml(curl)}</pre>
        </div>
      </div>
    `;
  }

  // ---- Bind events ----
  function bindEvents(container) {
    // Endpoint list clicks
    container.querySelectorAll('.endpoint-category-header').forEach(header => {
      header.addEventListener('click', () => {
        const cat = header.dataset.categoryToggle;
        const list = container.querySelector(`[data-category-list="${cat}"]`);
        if (list) {
          const open = list.style.display !== 'none';
          list.style.display = open ? 'none' : 'block';
          const chevron = header.querySelector('[data-lucide]');
          if (chevron) chevron.style.transform = open ? '' : 'rotate(180deg)';
        }
      });
    });

    // Endpoint item clicks
    container.querySelectorAll('.endpoint-item').forEach(item => {
      item.addEventListener('click', () => {
        const [catIdx, epIdx] = item.dataset.endpointIndex.split('-').map(Number);
        selectedEndpoint = API_CATALOG[catIdx].endpoints[epIdx];

        // Update active state
        container.querySelectorAll('.endpoint-item').forEach(i => i.classList.remove('active'));
        item.classList.add('active');

        // Re-render right panel
        const rightPanel = container.querySelector('#api-right-panel');
        if (rightPanel) {
          rightPanel.innerHTML = renderCurlPanel() + renderEndpointForm();
          if (window.lucide) lucide.createIcons({ nodes: [rightPanel] });
          bindFormEvents(container);
        }
      });
    });

    // Default: expand first category
    const firstHeader = container.querySelector('.endpoint-category-header');
    if (firstHeader) firstHeader.click();

    // Bind form events for any pre-rendered form
    bindFormEvents(container);
  }

  function bindFormEvents(container) {
    // Send button
    const sendBtn = container.querySelector('#api-send-btn');
    if (sendBtn) sendBtn.addEventListener('click', executeRequest);

    // Copy curl button
    const copyBtn = container.querySelector('#copy-curl-btn');
    if (copyBtn) {
      copyBtn.addEventListener('click', () => {
        const curlOutput = container.querySelector('#curl-output');
        if (curlOutput) {
          navigator.clipboard.writeText(curlOutput.textContent).then(() => {
            showToast(i18n.copied(), 'success');
            copyBtn.innerHTML = `<i data-lucide="check" style="width:14px;height:14px;margin-right:4px;"></i>${i18n.copied()}`;
            if (window.lucide) lucide.createIcons({ nodes: [copyBtn] });
            setTimeout(() => {
              copyBtn.innerHTML = `<i data-lucide="copy" style="width:14px;height:14px;margin-right:4px;"></i>${i18n.copy()}`;
              if (window.lucide) lucide.createIcons({ nodes: [copyBtn] });
            }, 2000);
          });
        }
      });
    }

    // Auto-update curl on form changes
    container.querySelectorAll('[data-location][data-name]').forEach(el => {
      const eventType = el.dataset.type === 'checkbox' ? 'change' : 'input';
      el.addEventListener(eventType, () => {
        const curlOutput = container.querySelector('#curl-output');
        if (curlOutput && selectedEndpoint) {
          const values = collectFormValues();
          curlOutput.textContent = generateCurl(selectedEndpoint, values);
        }
      });
    });
  }

  // ---- Main render ----
  async function render(container, params) {
    selectedEndpoint = null;

    container.innerHTML = `
      <div class="page-enter">
        <div style="display:flex;gap:16px;height:calc(100vh - 120px);min-height:500px;">
          <!-- Left: Endpoint list -->
          <div style="width:260px;flex-shrink:0;overflow-y:auto;" class="card" id="api-left-panel">
            <div style="padding:12px 16px;border-bottom:1px solid var(--border-color);">
              <h2 style="font-size:0.85rem;font-weight:600;margin:0;display:flex;align-items:center;gap:6px;">
                <i data-lucide="terminal" style="width:16px;height:16px;color:var(--primary);"></i>
                ${i18n.title()}
              </h2>
            </div>
            <div style="padding:8px;">
              ${renderEndpointList()}
            </div>
          </div>

          <!-- Right: Form + Response -->
          <div style="flex:1;min-width:0;overflow-y:auto;" id="api-right-panel">
            ${renderEndpointForm()}
          </div>
        </div>
      </div>
    `;

    if (window.lucide) lucide.createIcons({ nodes: [container] });
    bindEvents(container);
  }

  // ---- Cleanup ----
  function destroy() {
    selectedEndpoint = null;
  }

  registerPage('api-test', { render, destroy });
})();

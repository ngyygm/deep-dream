(function () {
  let currentConfig = null;

  const schema = [
    {
      section: 'llm', path: 'llm.max_concurrency', label: 'settings.llmMaxConcurrency',
      help: 'settings.llmMaxConcurrencyHelp', type: 'int', min: 1, max: 128,
    },
    {
      section: 'llm', path: 'llm.model', label: 'settings.llmModel',
      help: 'settings.llmModelHelp', type: 'text',
    },
    {
      section: 'llm', path: 'llm.base_url', label: 'settings.llmBaseUrl',
      help: 'settings.llmBaseUrlHelp', type: 'url',
    },
    {
      section: 'llm', path: 'llm.api_key', label: 'settings.llmApiKey',
      help: 'settings.llmApiKeyHelp', type: 'secret', sensitive: true,
    },
    {
      section: 'runtime', path: 'runtime.concurrency.queue_workers', label: 'settings.rememberWorkers',
      help: 'settings.rememberWorkersHelp', type: 'int', min: 1, max: 64,
    },
    {
      section: 'embedding', path: 'embedding.model', label: 'settings.embeddingModelPath',
      help: 'settings.embeddingModelPathHelp', type: 'text',
    },
    {
      section: 'embedding', path: 'embedding.max_concurrency', label: 'settings.embeddingMaxConcurrency',
      help: 'settings.embeddingMaxConcurrencyHelp', type: 'int', min: 1, max: 64,
    },
    {
      section: 'chunking', path: 'chunking.window_size', label: 'settings.chunkingWindowSize',
      help: 'settings.chunkingWindowSizeHelp', type: 'int', min: 1, max: 1000000,
    },
    {
      section: 'chunking', path: 'chunking.overlap', label: 'settings.chunkingOverlap',
      help: 'settings.chunkingOverlapHelp', type: 'int', min: 0, max: 999999,
    },
    {
      section: 'runtime', path: 'runtime.task.load_cache_memory', label: 'settings.defaultLoadCache',
      help: 'settings.defaultLoadCacheHelp', type: 'bool',
    },
  ];

  const sectionLabels = {
    llm: ['settings.sectionLlm', 'LLM 服务'],
    embedding: ['settings.sectionEmbedding', 'Embedding'],
    runtime: ['settings.sectionRuntime', '运行时'],
    chunking: ['settings.sectionChunking', '文本切片'],
  };

  function valueAt(obj, path) {
    return path.split('.').reduce((cur, key) => (
      cur && cur[key] !== undefined ? cur[key] : undefined
    ), obj);
  }

  function setAt(obj, path, value) {
    const parts = path.split('.');
    let cur = obj;
    parts.slice(0, -1).forEach(key => {
      if (!cur[key] || typeof cur[key] !== 'object') cur[key] = {};
      cur = cur[key];
    });
    cur[parts[parts.length - 1]] = value;
  }

  function labelFor(key, fallback) {
    const text = t(key);
    return text === key ? fallback : text;
  }

  function inputFor(spec) {
    const raw = valueAt(currentConfig || {}, spec.path);
    const isBool = spec.type === 'bool';
    const isSecret = spec.sensitive;
    const inputType = isBool ? 'checkbox' : (isSecret ? 'password' : (spec.type === 'url' ? 'url' : (spec.type === 'int' ? 'number' : 'text')));
    const value = isBool ? '' : (isSecret ? '' : (raw === undefined || raw === null ? '' : String(raw)));
    const initial = isBool ? (raw ? 'true' : 'false') : (isSecret ? '' : value);
    const constraints = [
      spec.min !== undefined ? ` min="${spec.min}"` : '',
      spec.max !== undefined ? ` max="${spec.max}"` : '',
      spec.type === 'int' ? ' step="1"' : '',
      isSecret ? ` autocomplete="new-password" placeholder="${escapeAttr(labelFor('settings.secretPlaceholder', '已配置；留空保持不变'))}"` : '',
    ].join('');
    const control = isBool
      ? `<input type="checkbox" class="config-field config-toggle" data-config-path="${escapeAttr(spec.path)}" data-config-type="bool" data-config-initial="${initial}" ${raw ? 'checked' : ''} aria-label="${escapeAttr(labelFor(spec.label, spec.path))}">`
      : `<input class="input config-field" type="${inputType}" data-config-path="${escapeAttr(spec.path)}" data-config-type="${escapeAttr(spec.type)}" data-config-initial="${escapeAttr(initial)}" value="${escapeAttr(value)}"${constraints} aria-describedby="help-${escapeAttr(spec.path.replaceAll('.', '-'))}">`;
    return `<label class="settings-field ${isBool ? 'settings-field-toggle' : ''}">
      <span class="settings-field-label">${escapeHtml(labelFor(spec.label, spec.path))}</span>
      ${control}
      <span id="help-${escapeAttr(spec.path.replaceAll('.', '-'))}" class="settings-field-help">${escapeHtml(labelFor(spec.help, ''))}</span>
    </label>`;
  }

  function renderForm() {
    const hint = (t('settings.hint') || '').replace('{path}', escapeHtml(currentConfig?._config_path || 'service_config.json'));
    const groups = ['llm', 'embedding', 'runtime', 'chunking'].map(section => {
      const [key, fallback] = sectionLabels[section];
      return `<section class="settings-section"><div class="settings-section-heading"><div><h2>${escapeHtml(labelFor(key, fallback))}</h2><p>${escapeHtml(labelFor(`${key}Help`, ''))}</p></div></div><div class="settings-grid">${schema.filter(s => s.section === section).map(inputFor).join('')}</div></section>`;
    }).join('');
    return `<div class="page-enter settings-page">
      <form id="settings-form" novalidate>
        <div class="card settings-hero">
          <div><span class="eyebrow">DeepDream</span><h1>${escapeHtml(t('settings.title'))}</h1><p>${hint}</p></div>
          <button class="btn btn-primary" id="settings-save" type="submit"><i data-lucide="save"></i><span>${escapeHtml(t('settings.save'))}</span></button>
        </div>
        <div class="settings-sections">${groups}</div>
      </form>
      <details class="card settings-raw"><summary>${escapeHtml(t('settings.rawPreview'))}</summary><pre class="mono">${escapeHtml(JSON.stringify(currentConfig, null, 2))}</pre></details>
    </div>`;
  }

  function parseValue(el) {
    const type = el.getAttribute('data-config-type');
    if (type === 'bool') return !!el.checked;
    const raw = el.value.trim();
    if (type === 'secret') return raw;
    if (type === 'int') {
      if (!raw) return null;
      if (!/^\d+$/.test(raw)) throw new Error(`${el.dataset.configPath} 必须是整数`);
      const value = Number(raw);
      const min = Number(el.min); const max = Number(el.max);
      if (!Number.isSafeInteger(value) || value < min || value > max) throw new Error(`${el.dataset.configPath} 超出允许范围`);
      return value;
    }
    return raw;
  }

  function readPatch(container) {
    const patch = {};
    container.querySelectorAll('.config-field').forEach(el => {
      const path = el.getAttribute('data-config-path');
      const type = el.getAttribute('data-config-type');
      const value = parseValue(el);
      // Secrets are deliberately blank in the GET response.  Empty means
      // "preserve" and is never sent back to the server.
      if (type === 'secret' && !value) return;
      const initial = type === 'bool' ? el.getAttribute('data-config-initial') === 'true' : el.getAttribute('data-config-initial');
      if (value === null && initial === '') return;
      if (type !== 'secret' && String(value) === String(initial)) return;
      setAt(patch, path, value);
    });
    return patch;
  }

  function bind(container) {
    const form = container.querySelector('#settings-form');
    if (!form) return;
    const save = form.querySelector('#settings-save');
    form.addEventListener('submit', async event => {
      event.preventDefault();
      try {
        const patch = readPatch(form);
        if (!Object.keys(patch).length) {
          showToast(labelFor('settings.noChanges', '没有需要保存的改动'), 'info');
          return;
        }
        if (save) { save.disabled = true; save.querySelector('span').textContent = labelFor('settings.saving', '保存中…'); }
        const saved = await state.api.updateSystemConfig(patch);
        currentConfig = saved.data?.config || currentConfig;
        showToast(saved.data?.message || t('settings.saved'), 'success');
        // Re-mount so the new values become the next dirty-state baseline and
        // the submit handler is always attached after DOM replacement.
        await render(container);
      } catch (err) {
        showToast(t('settings.saveFailed') + ': ' + err.message, 'error');
        if (save) { save.disabled = false; save.querySelector('span').textContent = t('settings.save'); }
      }
    });
  }

  async function render(container) {
    container.innerHTML = `<div class="page-enter">${spinnerHtml()}</div>`;
    try {
      const res = await state.api.systemConfig();
      currentConfig = res.data?.config || {};
      container.innerHTML = renderForm();
      bind(container);
      if (window.lucide) lucide.createIcons({ nodes: [container] });
    } catch (err) {
      container.innerHTML = `<div class="card"><div class="empty-state"><p style="color:var(--error);">${escapeHtml(t('settings.loadFailed'))}: ${escapeHtml(err.message)}</p></div></div>`;
    }
  }

  registerPage('settings', { render });
})();

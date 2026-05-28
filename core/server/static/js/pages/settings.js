(function () {
  let currentConfig = null;

  function valueAt(obj, path) {
    return path.split('.').reduce((cur, key) => cur && cur[key] !== undefined ? cur[key] : undefined, obj);
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

  function field(path, label, help, type = 'text') {
    const raw = valueAt(currentConfig || {}, path);
    const val = raw === undefined || raw === null ? '' : String(raw);
    return `
      <label style="display:flex;flex-direction:column;gap:0.35rem;">
        <span style="font-weight:600;color:var(--text-secondary);">${escapeHtml(label)}</span>
        <input class="input config-field" data-config-path="${escapeAttr(path)}" data-config-type="${type}" value="${escapeAttr(val)}">
        <span style="font-size:0.75rem;color:var(--text-muted);line-height:1.4;">${escapeHtml(help)}</span>
      </label>
    `;
  }

  function boolField(path, label, help) {
    const checked = !!valueAt(currentConfig || {}, path);
    return `
      <label style="display:flex;align-items:flex-start;gap:0.6rem;">
        <input type="checkbox" class="config-field" data-config-path="${escapeAttr(path)}" data-config-type="bool" ${checked ? 'checked' : ''} style="margin-top:0.2rem;accent-color:var(--primary);">
        <span>
          <span style="display:block;font-weight:600;color:var(--text-secondary);">${escapeHtml(label)}</span>
          <span style="display:block;font-size:0.75rem;color:var(--text-muted);line-height:1.4;">${escapeHtml(help)}</span>
        </span>
      </label>
    `;
  }

  function renderForm() {
    const hintHtml = (t('settings.hint') || '')
      .replace('{path}', escapeHtml(currentConfig?._config_path || 'service_config.json'));
    return `
      <div class="page-enter" style="max-width:1100px;">
        <div class="card" style="margin-bottom:1rem;">
          <div class="card-header">
            <span class="card-title">${t('settings.title')}</span>
            <button class="btn btn-primary btn-sm" id="settings-save">
              <i data-lucide="save" style="width:14px;height:14px;"></i>
              ${t('settings.save')}
            </button>
          </div>
          <div style="color:var(--text-muted);font-size:0.85rem;line-height:1.6;margin-bottom:1rem;">
            ${hintHtml}
          </div>
          <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:1rem;">
            ${field('remember_workers', t('settings.rememberWorkers'), t('settings.rememberWorkersHelp'), 'int')}
            ${field('llm.max_concurrency', t('settings.llmMaxConcurrency'), t('settings.llmMaxConcurrencyHelp'), 'int')}
            ${field('llm.model', t('settings.llmModel'), t('settings.llmModelHelp'))}
            ${field('llm.base_url', t('settings.llmBaseUrl'), t('settings.llmBaseUrlHelp'))}
            ${field('llm.api_key', t('settings.llmApiKey'), t('settings.llmApiKeyHelp'))}
            ${field('embedding.model_path', t('settings.embeddingModelPath'), t('settings.embeddingModelPathHelp'))}
            ${field('embedding.max_concurrency', t('settings.embeddingMaxConcurrency'), t('settings.embeddingMaxConcurrencyHelp'), 'int')}
            ${field('chunking.window_size', t('settings.chunkingWindowSize'), t('settings.chunkingWindowSizeHelp'), 'int')}
            ${field('chunking.overlap', t('settings.chunkingOverlap'), t('settings.chunkingOverlapHelp'), 'int')}
            ${boolField('runtime.task.load_cache_memory', t('settings.defaultLoadCache'), t('settings.defaultLoadCacheHelp'))}
          </div>
        </div>
        <div class="card">
          <div class="card-header"><span class="card-title">${t('settings.rawPreview')}</span></div>
          <pre class="mono" style="white-space:pre-wrap;word-break:break-word;max-height:360px;overflow:auto;background:var(--bg-input);border:1px solid var(--border-color);border-radius:0.5rem;padding:0.75rem;">${escapeHtml(JSON.stringify(currentConfig, null, 2))}</pre>
        </div>
      </div>
    `;
  }

  function readPatch(container) {
    const patch = {};
    container.querySelectorAll('.config-field').forEach(el => {
      const path = el.getAttribute('data-config-path');
      const type = el.getAttribute('data-config-type');
      let value = type === 'bool' ? el.checked : el.value;
      if (type === 'int') value = value === '' ? null : parseInt(value, 10);
      if (type === 'float') value = value === '' ? null : parseFloat(value);
      setAt(patch, path, value);
    });
    return patch;
  }

  async function render(container) {
    container.innerHTML = `<div class="page-enter">${spinnerHtml()}</div>`;
    try {
      const res = await state.api.systemConfig();
      currentConfig = res.data?.config || {};
      container.innerHTML = renderForm();
      container.querySelector('#settings-save')?.addEventListener('click', async () => {
        try {
          const patch = readPatch(container);
          const saved = await state.api.updateSystemConfig(patch);
          currentConfig = saved.data?.config || currentConfig;
          showToast(saved.data?.message || t('settings.saved'), 'success');
          container.innerHTML = renderForm();
          if (window.lucide) lucide.createIcons({ nodes: [container] });
        } catch (err) {
          showToast(t('settings.saveFailed') + ': ' + err.message, 'error');
        }
      });
      if (window.lucide) lucide.createIcons({ nodes: [container] });
    } catch (err) {
      container.innerHTML = `<div class="card"><div class="empty-state"><p style="color:var(--error);">${t('settings.loadFailed')}: ${escapeHtml(err.message)}</p></div></div>`;
    }
  }

  registerPage('settings', { render });
})();

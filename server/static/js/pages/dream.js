/* ==========================================
   DeepDream Page - Memory Consolidation
   ========================================== */
registerPage('dream', (function () {
  'use strict';

  let _container = null;
  let _statusTimer = null;

  // ---- Render ----

  async function render(container) {
    _container = container;

    container.innerHTML = `
      <div class="space-y-4">
        <!-- Dream Control Panel -->
        <div class="card">
          <div class="card-header">
            <h2 class="card-title" style="margin:0;">
              <i data-lucide="sparkles" style="width:18px;height:18px;margin-right:6px;color:var(--primary);"></i>
              ${t('deepDream.dreamControl')}
            </h2>
          </div>
          <div style="padding:1rem 1.25rem;">
            <!-- Status -->
            <div id="dream-status-bar" style="margin-bottom:1rem;">
              <div class="flex items-center gap-2">
                <div class="spinner spinner-sm" id="dream-status-spinner"></div>
                <span id="dream-status-text" style="font-size:0.85rem;color:var(--text-muted);">${t('deepDream.title')}</span>
              </div>
            </div>

            <!-- Config -->
            <div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:1rem;margin-bottom:1rem;">
              <div>
                <label class="form-label">${t('deepDream.reviewWindowDays')}</label>
                <input type="number" id="dream-review-window" class="input" value="30" min="1" max="365">
              </div>
              <div>
                <label class="form-label">${t('deepDream.maxEntitiesPerCycle')}</label>
                <input type="number" id="dream-max-entities" class="input" value="100" min="10" max="1000">
              </div>
              <div>
                <label class="form-label">${t('deepDream.similarityThreshold')}</label>
                <input type="number" id="dream-similarity" class="input" value="0.8" min="0.1" max="1.0" step="0.05">
              </div>
            </div>

            <!-- Start Button -->
            <button class="btn btn-primary" id="dream-start-btn">
              <i data-lucide="play" style="width:16px;height:16px;margin-right:6px;"></i>
              ${t('deepDream.startDream')}
            </button>
          </div>
        </div>

        <!-- Dream Narrative (shown after completion) -->
        <div class="card" id="dream-narrative-card" style="display:none;">
          <div class="card-header">
            <h3 class="card-title" style="margin:0;">
              <i data-lucide="book-open" style="width:16px;height:16px;margin-right:6px;color:var(--primary);"></i>
              ${t('deepDream.dreamNarrative')}
            </h3>
          </div>
          <div style="padding:1rem 1.25rem;">
            <div id="dream-narrative-content" style="font-size:0.9rem;line-height:1.7;white-space:pre-wrap;word-break:break-word;color:var(--text-primary);"></div>
          </div>
        </div>

        <!-- Dream Insights -->
        <div class="card" id="dream-insights-card" style="display:none;">
          <div class="card-header">
            <h3 class="card-title" style="margin:0;">
              <i data-lucide="lightbulb" style="width:16px;height:16px;margin-right:6px;color:var(--warning);"></i>
              ${t('deepDream.dreamInsights')}
            </h3>
          </div>
          <div style="padding:1rem 1.25rem;">
            <div id="dream-insights-content"></div>
          </div>
        </div>

        <!-- Dream History -->
        <div class="card">
          <div class="card-header">
            <h3 class="card-title" style="margin:0;">
              <i data-lucide="history" style="width:16px;height:16px;margin-right:6px;"></i>
              ${t('deepDream.dreamHistory')}
            </h3>
          </div>
          <div style="padding:1rem 1.25rem;">
            <div id="dream-logs-list">
              <div class="flex items-center justify-center p-4">${spinnerHtml()}</div>
            </div>
          </div>
        </div>
      </div>
    `;

    if (window.lucide) lucide.createIcons({ nodes: [container] });

    // Bind events
    const startBtn = container.querySelector('#dream-start-btn');
    if (startBtn) {
      startBtn.addEventListener('click', startDream);
    }

    // Load initial state
    await Promise.all([
      refreshStatus(),
      loadDreamLogs(),
    ]);

    // Poll status every 3s if running
    _statusTimer = setInterval(refreshStatus, 3000);
  }

  // ---- Start Dream ----

  async function startDream() {
    const btn = _container.querySelector('#dream-start-btn');
    if (!btn) return;
    btn.disabled = true;
    btn.innerHTML = `${spinnerHtml('spinner-sm')} ${t('deepDream.dreamRunning')}`;

    try {
      const reviewWindow = parseInt(document.getElementById('dream-review-window')?.value || '30', 10);
      const maxEntities = parseInt(document.getElementById('dream-max-entities')?.value || '100', 10);
      const similarity = parseFloat(document.getElementById('dream-similarity')?.value || '0.8');

      await state.api.startDream({
        reviewWindowDays: reviewWindow,
        maxEntitiesPerCycle: maxEntities,
        similarityThreshold: similarity,
      });
      showToast(t('dream.startSuccess'), 'success');
      await refreshStatus();
    } catch (err) {
      showToast(t('dream.startFailed') + ': ' + err.message, 'error');
    } finally {
      btn.disabled = false;
      btn.innerHTML = `<i data-lucide="play" style="width:16px;height:16px;margin-right:6px;"></i>${t('deepDream.startDream')}`;
      if (window.lucide) lucide.createIcons({ nodes: [btn] });
    }
  }

  // ---- Refresh Status ----

  async function refreshStatus() {
    const statusText = _container?.querySelector('#dream-status-text');
    const statusSpinner = _container?.querySelector('#dream-status-spinner');

    try {
      const res = await state.api.dreamStatus();
      const data = res.data || {};
      const status = data.status || 'idle';

      if (statusText) {
        const statusMap = {
          idle: t('dream.statusIdle'),
          running: t('dream.statusRunning'),
          completed: t('dream.statusCompleted'),
          failed: t('dream.statusFailed'),
        };
        statusText.textContent = statusMap[status] || status;

        const colorMap = {
          idle: 'var(--text-muted)',
          running: 'var(--primary)',
          completed: 'var(--success)',
          failed: 'var(--error)',
        };
        statusText.style.color = colorMap[status] || 'var(--text-muted)';
      }

      if (statusSpinner) {
        statusSpinner.style.display = status === 'running' ? '' : 'none';
      }

      // If completed, show narrative and insights
      if (status === 'completed' && data.cycle_id) {
        await showDreamDetail(data.cycle_id);
      }
    } catch (err) {
      if (statusText) {
        statusText.textContent = t('dream.loadStatusFailed');
        statusText.style.color = 'var(--error)';
      }
      if (statusSpinner) statusSpinner.style.display = 'none';
    }
  }

  // ---- Show Dream Detail ----

  async function showDreamDetail(cycleId) {
    try {
      const res = await state.api.dreamLogDetail(cycleId);
      const data = res.data || {};

      // Narrative
      const narrativeCard = _container?.querySelector('#dream-narrative-card');
      const narrativeContent = _container?.querySelector('#dream-narrative-content');
      if (narrativeCard && narrativeContent && data.narrative) {
        narrativeContent.textContent = data.narrative;
        narrativeCard.style.display = '';
      }

      // Insights
      const insightsCard = _container?.querySelector('#dream-insights-card');
      const insightsContent = _container?.querySelector('#dream-insights-content');
      if (insightsCard && insightsContent) {
        const insights = data.insights || [];
        const connections = data.connections || [];
        const consolidations = data.consolidations || [];

        if (insights.length === 0 && connections.length === 0 && consolidations.length === 0) {
          insightsContent.innerHTML = `<p style="color:var(--text-muted);font-size:0.85rem;">${t('deepDream.noDreamLogs')}</p>`;
        } else {
          let html = '';
          if (insights.length > 0) {
            html += `<div style="margin-bottom:0.75rem;"><h4 style="font-size:0.85rem;margin-bottom:0.5rem;">${t('deepDream.dreamInsights')}</h4>`;
            for (const ins of insights) {
              html += `<div class="flex items-start gap-2 p-2 rounded mb-1" style="background:var(--bg-secondary);font-size:0.85rem;">
                <i data-lucide="zap" style="width:14px;height:14px;color:var(--warning);flex-shrink:0;margin-top:2px;"></i>
                <span>${escapeHtml(typeof ins === 'string' ? ins : JSON.stringify(ins))}</span>
              </div>`;
            }
            html += '</div>';
          }
          if (connections.length > 0) {
            html += `<div style="margin-bottom:0.75rem;"><h4 style="font-size:0.85rem;margin-bottom:0.5rem;">${t('deepDream.dreamConnections')}</h4>`;
            for (const conn of connections) {
              html += `<div class="flex items-start gap-2 p-2 rounded mb-1" style="background:var(--bg-secondary);font-size:0.85rem;">
                <i data-lucide="link" style="width:14px;height:14px;color:var(--primary);flex-shrink:0;margin-top:2px;"></i>
                <span>${escapeHtml(typeof conn === 'string' ? conn : JSON.stringify(conn))}</span>
              </div>`;
            }
            html += '</div>';
          }
          if (consolidations.length > 0) {
            html += `<div><h4 style="font-size:0.85rem;margin-bottom:0.5rem;">${t('deepDream.dreamConsolidations')}</h4>`;
            for (const cons of consolidations) {
              html += `<div class="flex items-start gap-2 p-2 rounded mb-1" style="background:var(--bg-secondary);font-size:0.85rem;">
                <i data-lucide="merge" style="width:14px;height:14px;color:var(--success);flex-shrink:0;margin-top:2px;"></i>
                <span>${escapeHtml(typeof cons === 'string' ? cons : JSON.stringify(cons))}</span>
              </div>`;
            }
            html += '</div>';
          }
          insightsContent.innerHTML = html;
        }
        insightsCard.style.display = '';
      }

      if (window.lucide) lucide.createIcons({ nodes: [narrativeCard, insightsCard] });
    } catch (err) {
      // Silently ignore
    }
  }

  // ---- Load Dream Logs ----

  async function loadDreamLogs() {
    const listEl = _container?.querySelector('#dream-logs-list');
    if (!listEl) return;

    try {
      const res = await state.api.dreamLogs(20);
      const logs = res.data || [];

      if (!Array.isArray(logs) || logs.length === 0) {
        listEl.innerHTML = `<div style="color:var(--text-muted);font-size:0.85rem;text-align:center;padding:1rem;">${t('deepDream.noDreamLogs')}</div>`;
        return;
      }

      let html = '<div class="space-y-2">';
      for (const log of logs) {
        const status = log.status || 'completed';
        const statusBadge = status === 'completed'
          ? '<span class="badge badge-success">completed</span>'
          : status === 'failed'
            ? '<span class="badge badge-error">failed</span>'
            : '<span class="badge badge-info">' + escapeHtml(status) + '</span>';

        html += `
          <div class="p-3 rounded cursor-pointer" style="background:var(--bg-secondary);border:1px solid var(--border-color);"
               onclick="window._dreamShowLog('${escapeAttr(log.cycle_id || log.id || '')}')">
            <div class="flex items-center justify-between gap-2 mb-1">
              <span class="mono text-xs" style="color:var(--text-muted);">${escapeHtml(log.cycle_id || log.id || '-')}</span>
              ${statusBadge}
            </div>
            <div class="flex items-center gap-3 text-xs" style="color:var(--text-secondary);">
              <span><i data-lucide="clock" style="width:12px;height:12px;display:inline;vertical-align:middle;margin-right:2px;"></i>${formatDate(log.start_time)}</span>
              ${log.entities_processed != null ? `<span>${log.entities_processed} ${t('dream.entitiesProcessed')}</span>` : ''}
            </div>
          </div>`;
      }
      html += '</div>';
      listEl.innerHTML = html;
    } catch (err) {
      listEl.innerHTML = `<div style="color:var(--error);font-size:0.85rem;text-align:center;padding:1rem;">${t('dream.loadLogsFailed')}</div>`;
    }
  }

  // ---- Show Log Detail ----

  window._dreamShowLog = async function (cycleId) {
    if (!cycleId) return;
    try {
      const res = await state.api.dreamLogDetail(cycleId);
      const data = res.data || {};

      let body = `<div style="display:flex;flex-direction:column;gap:0.75rem;">`;
      body += `<div style="display:grid;grid-template-columns:auto 1fr;gap:0.25rem 0.75rem;font-size:0.85rem;">
        <span style="color:var(--text-secondary);">Cycle ID:</span><span class="mono text-xs">${escapeHtml(data.cycle_id || cycleId)}</span>
        <span style="color:var(--text-secondary);">${t('common.status')}:</span><span>${escapeHtml(data.status || '-')}</span>
        <span style="color:var(--text-secondary);">${t('dream.startedAt')}:</span><span>${formatDate(data.start_time)}</span>
        ${data.end_time ? `<span style="color:var(--text-secondary);">${t('dream.finishedAt')}:</span><span>${formatDate(data.end_time)}</span>` : ''}
        ${data.entities_processed != null ? `<span style="color:var(--text-secondary);">${t('dream.entitiesProcessed')}:</span><span>${data.entities_processed}</span>` : ''}
        ${data.merges_performed != null ? `<span style="color:var(--text-secondary);">${t('dream.mergesPerformed')}:</span><span>${data.merges_performed}</span>` : ''}
      </div>`;

      if (data.narrative) {
        body += `<div>
          <h4 style="margin-bottom:0.5rem;">${t('dream.narrative')}</h4>
          <div style="max-height:300px;overflow-y:auto;background:var(--bg-secondary);padding:0.75rem;border-radius:0.5rem;font-size:0.85rem;line-height:1.7;white-space:pre-wrap;word-break:break-word;">${escapeHtml(data.narrative)}</div>
        </div>`;
      }

      body += '</div>';

      showModal({
        title: t('dream.logDetail'),
        content: body,
        size: 'lg',
      });
    } catch (err) {
      showToast(t('dream.loadLogsFailed') + ': ' + err.message, 'error');
    }
  };

  // ---- Destroy ----

  function destroy() {
    _container = null;
    if (_statusTimer) {
      clearInterval(_statusTimer);
      _statusTimer = null;
    }
    delete window._dreamShowLog;
  }

  return { render, destroy };
})());

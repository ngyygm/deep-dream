// DeepDream / Butler Management Page
// Module pattern - registers with app.js router
registerPage('dream', (function() {
  'use strict';

  let _refreshTimer = null;

  // ── Helpers ──────────────────────────────────────────────
  function priorityClass(p) {
    return { high: 'badge-error', medium: 'badge-warning', low: 'badge-info' }[p] || 'badge-primary';
  }

  function statusIcon(s) {
    const m = {
      done: '<span style="color:var(--success);">&#10003;</span>',
      preview: '<span style="color:var(--warning);">&#128065;</span>',
      skipped: '<span style="color:var(--text-muted);">&#8722;</span>',
      unknown: '<span style="color:var(--error);">?</span>',
    };
    return m[s] || m.unknown;
  }

  // ── Render ──────────────────────────────────────────────
  async function render(container) {
    container.innerHTML = `
      <div class="page-enter" style="padding:1.5rem;max-width:960px;margin:0 auto;">
        <!-- Header -->
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:1.5rem;">
          <h2 style="font-size:1.25rem;font-weight:700;color:var(--text-primary);">${t('dream.title')}</h2>
          <div style="display:flex;gap:0.5rem;">
            <button id="dream-refresh-btn" class="btn btn-secondary btn-sm" onclick="window._dreamRefresh()">
              <i data-lucide="refresh-cw" style="width:14px;height:14px;margin-right:4px;"></i>
              <span>${t('common.refresh')}</span>
            </button>
          </div>
        </div>

        <!-- Health Cards -->
        <div id="dream-health-cards" style="display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:0.75rem;margin-bottom:1.5rem;">
          ${spinnerHtml()}
        </div>

        <!-- Recommendations -->
        <div id="dream-recommendations" style="margin-bottom:1.5rem;"></div>

        <!-- Dream Status & Logs -->
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin-bottom:1.5rem;">
          <!-- Dream Status -->
          <div id="dream-status-panel" class="card" style="padding:1rem;">
            <h3 style="font-size:0.9rem;font-weight:600;color:var(--text-secondary);margin-bottom:0.75rem;">${t('dream.status')}</h3>
            <div id="dream-status-content">${spinnerHtml('sm')}</div>
          </div>
          <!-- Quality Summary -->
          <div id="dream-quality-panel" class="card" style="padding:1rem;">
            <h3 style="font-size:0.9rem;font-weight:600;color:var(--text-secondary);margin-bottom:0.75rem;">${t('dream.qualityReport')}</h3>
            <div id="dream-quality-content">${spinnerHtml('sm')}</div>
          </div>
        </div>

        <!-- Action Buttons -->
        <div id="dream-actions" style="margin-bottom:1.5rem;"></div>

        <!-- Dream Logs -->
        <div class="card" style="padding:1rem;">
          <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:0.75rem;">
            <h3 style="font-size:0.9rem;font-weight:600;color:var(--text-secondary);">${t('dream.logs')}</h3>
            <button class="btn btn-ghost btn-sm" onclick="window._dreamLoadLogs()">${t('common.refresh')}</button>
          </div>
          <div id="dream-logs-list">${spinnerHtml('sm')}</div>
        </div>
      </div>
    `;
    if (window.lucide) lucide.createIcons();
    await loadAll();
    _refreshTimer = setInterval(loadAll, 30000);
  }

  function destroy() {
    if (_refreshTimer) { clearInterval(_refreshTimer); _refreshTimer = null; }
  }

  // ── Data Loading ────────────────────────────────────────
  async function loadAll() {
    const gid = state.currentGraphId;
    await Promise.allSettled([
      loadHealthCards(gid),
      loadRecommendations(gid),
      loadDreamStatus(gid),
      loadQuality(gid),
      loadActionButtons(gid),
      loadDreamLogs(gid),
    ]);
  }

  async function loadHealthCards(gid) {
    const el = document.getElementById('dream-health-cards');
    if (!el) return;
    try {
      const res = await state.api.findStats(gid);
      const d = res.data || {};
      const items = [
        { label: t('dashboard.totalEntities'), value: formatNumber(d.total_entities), icon: 'circle-dot', color: 'var(--primary)' },
        { label: t('dashboard.totalRelations'), value: formatNumber(d.total_relations), icon: 'git-branch', color: 'var(--accent)' },
        { label: t('dream.episodes'), value: formatNumber(d.total_episodes), icon: 'film', color: 'var(--success)' },
        { label: t('communities.communitiesCount'), value: formatNumber(d.total_communities || 0), icon: 'layout-grid', color: 'var(--warning)' },
      ];
      el.innerHTML = items.map(i => `
        <div class="card" style="padding:0.75rem 1rem;display:flex;align-items:center;gap:0.75rem;">
          <div style="width:36px;height:36px;border-radius:8px;display:flex;align-items:center;justify-content:center;background:${i.color}20;">
            <i data-lucide="${i.icon}" style="width:18px;height:18px;color:${i.color};"></i>
          </div>
          <div>
            <div style="font-size:1.25rem;font-weight:700;color:var(--text-primary);">${i.value}</div>
            <div style="font-size:0.75rem;color:var(--text-muted);">${i.label}</div>
          </div>
        </div>
      `).join('');
      if (window.lucide) lucide.createIcons();
    } catch { el.innerHTML = '<div style="color:var(--error);">Failed to load stats</div>'; }
  }

  async function loadRecommendations(gid) {
    const el = document.getElementById('dream-recommendations');
    if (!el) return;
    try {
      const res = await state.api.butlerReport(gid);
      const recs = res.data?.recommendations || [];
      if (!recs.length) {
        el.innerHTML = `
          <div class="card" style="padding:1rem;display:flex;align-items:center;gap:0.75rem;">
            <i data-lucide="check-circle" style="width:20px;height:20px;color:var(--success);"></i>
            <span style="color:var(--text-secondary);">${t('dream.noRecommendations')}</span>
          </div>`;
        if (window.lucide) lucide.createIcons();
        return;
      }
      el.innerHTML = `
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:0.75rem;">
          <h3 style="font-size:0.9rem;font-weight:600;color:var(--text-secondary);">${t('dream.recommendations')} (${recs.length})</h3>
          <button class="btn btn-primary btn-sm" onclick="window._dreamExecuteAll()">
            <i data-lucide="play" style="width:14px;height:14px;margin-right:4px;"></i>
            ${t('dream.executeAll')}
          </button>
        </div>
        <div style="display:flex;flex-direction:column;gap:0.5rem;">
          ${recs.map((r, i) => `
            <div class="card" style="padding:0.75rem 1rem;display:flex;align-items:center;gap:0.75rem;">
              <span class="badge ${priorityClass(r.priority)}" style="flex-shrink:0;">${escapeHtml(r.priority)}</span>
              <div style="flex:1;min-width:0;">
                <div style="font-size:0.85rem;font-weight:500;color:var(--text-primary);">${escapeHtml(r.description)}</div>
                <div style="font-size:0.75rem;color:var(--text-muted);margin-top:2px;">${escapeHtml(r.estimated_impact || '')}</div>
              </div>
              <button class="btn btn-ghost btn-sm" onclick="window._dreamExecuteOne('${escapeHtml(r.action)}')" title="${t('dream.executeAction')}">
                <i data-lucide="play" style="width:14px;height:14px;"></i>
              </button>
            </div>
          `).join('')}
        </div>
      `;
      if (window.lucide) lucide.createIcons();
    } catch { el.innerHTML = ''; }
  }

  async function loadDreamStatus(gid) {
    const el = document.getElementById('dream-status-content');
    if (!el) return;
    try {
      const res = await state.api.dreamStatus(gid);
      const d = res.data || {};
      if (d.status === 'no_cycles' || d.status === 'not_available') {
        el.innerHTML = `<div style="color:var(--text-muted);font-size:0.85rem;">${t('dream.noLogs')}</div>`;
        return;
      }
      el.innerHTML = `
        <div style="display:grid;grid-template-columns:auto 1fr;gap:0.25rem 0.75rem;font-size:0.85rem;">
          <span style="color:var(--text-muted);">${t('dream.status')}:</span>
          <span>${statusBadge(d.status || 'idle')}</span>
          ${d.last_cycle_id ? `<span style="color:var(--text-muted);">${t('dream.cycleId')}:</span><span class="mono" style="font-size:0.8rem;">${escapeHtml(d.last_cycle_id)}</span>` : ''}
          ${d.last_cycle_time ? `<span style="color:var(--text-muted);">${t('dream.startedAt')}:</span><span>${formatDate(d.last_cycle_time)}</span>` : ''}
          ${d.entities_explored != null ? `<span style="color:var(--text-muted);">${t('dream.entitiesProcessed')}:</span><span>${d.entities_explored}</span>` : ''}
          ${d.relations_created != null ? `<span style="color:var(--text-muted);">${t('dream.newInsights')}:</span><span>${d.relations_created}</span>` : ''}
        </div>
      `;
    } catch { el.innerHTML = '<div style="color:var(--error);font-size:0.85rem;">Failed to load</div>'; }
  }

  async function loadQuality(gid) {
    const el = document.getElementById('dream-quality-content');
    if (!el) return;
    try {
      const res = await state.api.maintenanceHealth(gid);
      const d = res.data || {};
      const quality = d.quality || {};
      const iso = d.isolated_entity_count || 0;
      const stats = d.statistics || {};
      el.innerHTML = `
        <div style="display:grid;grid-template-columns:auto 1fr;gap:0.25rem 0.75rem;font-size:0.85rem;">
          <span style="color:var(--text-muted);">${t('dream.validEntities')}:</span><span>${formatNumber(quality.valid_entities || stats.total_entities || 0)}</span>
          <span style="color:var(--text-muted);">${t('dream.invalidatedEntities')}:</span><span style="color:${(quality.invalidated_entities || 0) > 0 ? 'var(--warning)' : ''};">${formatNumber(quality.invalidated_entities || 0)}</span>
          <span style="color:var(--text-muted);">${t('dream.isolatedEntities')}:</span><span style="color:${iso > 0 ? 'var(--warning)' : ''};">${formatNumber(iso)}</span>
          <span style="color:var(--text-muted);">${t('dream.validRelations')}:</span><span>${formatNumber(quality.valid_relations || stats.total_relations || 0)}</span>
          <span style="color:var(--text-muted);">${t('dream.invalidatedRelations')}:</span><span style="color:${(quality.invalidated_relations || 0) > 0 ? 'var(--warning)' : ''};">${formatNumber(quality.invalidated_relations || 0)}</span>
        </div>
      `;
    } catch { el.innerHTML = '<div style="color:var(--error);font-size:0.85rem;">Failed to load</div>'; }
  }

  async function loadActionButtons(gid) {
    const el = document.getElementById('dream-actions');
    if (!el) return;
    el.innerHTML = `
      <div style="display:flex;flex-wrap:wrap;gap:0.5rem;">
        <button class="btn btn-secondary btn-sm" onclick="window._dreamCleanup()">
          <i data-lucide="trash-2" style="width:14px;height:14px;margin-right:4px;"></i>
          ${t('dream.cleanup')}
        </button>
        <button class="btn btn-secondary btn-sm" onclick="window._dreamDetectCommunities()">
          <i data-lucide="layout-grid" style="width:14px;height:14px;margin-right:4px;"></i>
          ${t('dream.detectCommunities')}
        </button>
        <button class="btn btn-secondary btn-sm" onclick="window._dreamEvolveSummaries()">
          <i data-lucide="sparkles" style="width:14px;height:14px;margin-right:4px;"></i>
          ${t('dream.evolveSummaries')}
        </button>
        <button class="btn btn-secondary btn-sm" onclick="window._dreamGetSeeds()">
          <i data-lucide="dice-5" style="width:14px;height:14px;margin-right:4px;"></i>
          ${t('dream.getSeeds')}
        </button>
      </div>
    `;
    if (window.lucide) lucide.createIcons();
  }

  async function loadDreamLogs(gid) {
    const el = document.getElementById('dream-logs-list');
    if (!el) return;
    try {
      const res = await state.api.dreamLogs(gid, 10);
      const logs = Array.isArray(res.data) ? res.data : (res.data?.logs || []);
      if (!logs.length) {
        el.innerHTML = `<div style="color:var(--text-muted);font-size:0.85rem;padding:0.5rem;">${t('dream.noLogs')}</div>`;
        return;
      }
      el.innerHTML = `
        <div style="display:flex;flex-direction:column;gap:0.375rem;">
          ${logs.map(l => `
            <div style="display:flex;align-items:center;gap:0.75rem;padding:0.5rem 0.75rem;border-radius:0.5rem;background:var(--bg-secondary);font-size:0.85rem;cursor:pointer;" onclick="window._dreamLogDetail('${escapeHtml(l.cycle_id || l.id || '')}')">
              <span class="badge ${statusBadge(l.status || 'completed').includes('success') ? 'badge-success' : 'badge-info'}">${escapeHtml(l.status || 'completed')}</span>
              <span class="mono" style="flex-shrink:0;color:var(--text-muted);font-size:0.8rem;">${escapeHtml((l.cycle_id || l.id || '').slice(0, 12))}</span>
              <span style="flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:var(--text-secondary);">${escapeHtml(l.summary || l.narrative || '-')}</span>
              <span style="flex-shrink:0;color:var(--text-muted);font-size:0.75rem;">${formatDate(l.started_at || l.created_at)}</span>
            </div>
          `).join('')}
        </div>
      `;
    } catch { el.innerHTML = `<div style="color:var(--error);font-size:0.85rem;">${t('dream.loadLogsFailed')}</div>`; }
  }

  // ── Actions ──────────────────────────────────────────────
  window._dreamRefresh = async function() {
    await loadAll();
  };

  window._dreamCleanup = async function() {
    const ok = await showConfirm({ message: t('dream.confirmCleanup'), destructive: true });
    if (!ok) return;
    try {
      const gid = state.currentGraphId;
      const res = await state.api.butlerExecute(['cleanup_isolated', 'cleanup_invalidated'], false, gid);
      const d = res.data || {};
      const results = d.actions || {};
      showAlert({ title: t('dream.cleanupResult') || 'Cleanup', message: JSON.stringify(results, null, 2) });
      await loadAll();
    } catch (e) { showAlert({ title: t('dream.actionFailed') || 'Error', message: e.message }); }
  };

  window._dreamDetectCommunities = async function() {
    try {
      const gid = state.currentGraphId;
      const res = await state.api.detectCommunities(gid);
      const d = res.data || {};
      showAlert({ title: t('dream.communitiesDetected') || 'Communities', message: String(d.communities_created || d.count || 0) });
      await loadAll();
    } catch (e) { showAlert({ title: t('dream.actionFailed') || 'Error', message: e.message }); }
  };

  window._dreamEvolveSummaries = async function() {
    try {
      const gid = state.currentGraphId;
      const res = await state.api.butlerExecute(['evolve_summaries'], false, gid);
      const d = res.data?.actions?.evolve_summaries || {};
      showAlert({ title: t('dream.evolveResult') || 'Evolve', message: `evolved=${d.evolved || 0}, failed=${d.failed || 0}` });
      await loadAll();
    } catch (e) { showAlert({ title: t('dream.actionFailed') || 'Error', message: e.message }); }
  };

  window._dreamGetSeeds = async function() {
    try {
      const gid = state.currentGraphId;
      const res = await state.api.dreamSeeds(gid, 'random', 5);
      const seeds = res.data?.seeds || [];
      if (!seeds.length) { showAlert({ message: t('common.noResults') || 'No results' }); return; }
      const lines = seeds.map((s, i) => `${i + 1}. ${s.name || s.family_id} (degree: ${s.degree || '?'})`).join('\n');
      showAlert({ title: t('dream.seedsFound') || 'Seeds', message: lines });
    } catch (e) { showAlert({ title: t('dream.actionFailed') || 'Error', message: e.message }); }
  };

  window._dreamExecuteAll = async function() {
    const ok = await showConfirm({ message: t('dream.confirmExecuteAll') });
    if (!ok) return;
    try {
      const gid = state.currentGraphId;
      const reportRes = await state.api.butlerReport(gid);
      const recs = reportRes.data?.recommendations || [];
      const actions = recs.map(r => r.action);
      if (!actions.length) { showAlert({ message: t('dream.noRecommendations') || 'No recommendations' }); return; }
      const res = await state.api.butlerExecute(actions, false, gid);
      const d = res.data || {};
      showAlert({ title: t('dream.executeResult') || 'Result', message: JSON.stringify(d.actions, null, 2) });
      await loadAll();
    } catch (e) { showAlert({ title: t('dream.actionFailed') || 'Error', message: e.message }); }
  };

  window._dreamExecuteOne = async function(action) {
    try {
      const gid = state.currentGraphId;
      const res = await state.api.butlerExecute([action], false, gid);
      const d = res.data?.actions?.[action] || {};
      showAlert({ title: t('dream.executeResult') || 'Result', message: JSON.stringify(d, null, 2) });
      await loadAll();
    } catch (e) { showAlert({ title: t('dream.actionFailed') || 'Error', message: e.message }); }
  };

  window._dreamLogDetail = async function(cycleId) {
    if (!cycleId) return;
    try {
      const gid = state.currentGraphId;
      const res = await state.api.dreamLogDetail(cycleId, gid);
      const d = res.data || {};
      const content = d.narrative || d.summary || JSON.stringify(d, null, 2);
      showModal({
        title: t('dream.logDetail') || 'Dream Log',
        content: `<div class="md-content" style="max-height:60vh;overflow-y:auto;white-space:pre-wrap;word-break:break-word;font-size:0.875rem;line-height:1.7;">${renderMarkdown(content)}</div>`,
        size: 'lg',
      });
    } catch (e) { showAlert({ title: t('dream.loadLogsFailed') || 'Error', message: e.message }); }
  };

  window._dreamLoadLogs = function() { loadDreamLogs(state.currentGraphId); };

  return { render, destroy };
})());

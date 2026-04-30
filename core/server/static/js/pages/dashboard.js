// Module pattern - registers with app.js router
(function() {
  'use strict';

  // ---------------------------------------------------------------------------
  // Private state
  // ---------------------------------------------------------------------------
  let _logLevel = '';       // '' = all, 'INFO', 'WARN', 'ERROR'
  let _autoScroll = true;

  // ---------------------------------------------------------------------------
  // Helpers
  // ---------------------------------------------------------------------------

  /** Sum a numeric field across all graph objects. */
  function sumAcrossGraphs(graphs, field) {
    return graphs.reduce((sum, g) => sum + (g.storage[field] || 0), 0);
  }

  /** Compute elapsed / estimated time from task timestamps and progress. */
  function elapsedText(task) {
    const toMs = v => { const n = Number(v); return (isNaN(n) || !n) ? 0 : (n < 4102444800000 ? n * 1000 : n); };
    const now = Date.now();

    // 已结束：总耗时
    if (task.status === 'completed' || task.status === 'failed') {
      const start = toMs(task.started_at || task.created_at);
      const end = toMs(task.finished_at) || now;
      return formatRelativeTime(Math.max(0, Math.round((end - start) / 1000)));
    }

    // 排队中：等待时长
    if (task.status === 'queued') {
      const created = toMs(task.created_at);
      if (!created) return '-';
      return t('dashboard.waiting') + ' ' + formatRelativeTime(Math.max(0, Math.round((now - created) / 1000)));
    }

    // 运行中：已耗时 + 预估剩余
    const started = toMs(task.started_at);
    if (!started) return '-';
    const elapsed = Math.max(0, Math.round((now - started) / 1000));
    const done = task.processed_chunks || 0;
    const runStart = task.run_start_chunks || 0;
    const total = task.total_chunks || 0;
    const runDone = Math.max(0, done - runStart); // 本轮实际完成的 chunk 数
    const remaining = total - done;                // 还剩多少 chunk

    if (runDone > 0 && remaining > 0) {
      // 有实际速率数据：用本轮完成数 / 本轮耗时
      const avgPerChunk = elapsed / runDone;
      const estRemaining = avgPerChunk * remaining;
      return formatRelativeTime(elapsed) + ' / ~' + formatRelativeTime(Math.round(estRemaining));
    }
    // 还没跑完本轮第一个窗口，只显示已耗时
    return formatRelativeTime(elapsed);
  }

  /** Pick a color class for a log level badge. */
  function logLevelBadge(level) {
    const m = { INFO: 'badge-info', WARN: 'badge-warning', ERROR: 'badge-error' };
    return `<span class="badge ${m[level] || 'badge-primary'}">${escapeHtml(level)}</span>`;
  }

  /** Log row HTML. */
  function logRow(entry) {
    return `<div class="dashboard-log-row" style="padding:0.375rem 0.75rem;border-bottom:1px solid var(--border-color);font-size:0.8125rem;font-family:var(--font-mono);line-height:1.5;">
      <span style="color:var(--text-muted);margin-right:0.5rem;flex-shrink:0;">${escapeHtml(entry.time)}</span>
      ${logLevelBadge(entry.level)}
      <span style="color:var(--text-secondary);margin:0 0.5rem;flex-shrink:0;">[${escapeHtml(entry.source)}]</span>
      <span class="log-${entry.level.toLowerCase()}">${escapeHtml(entry.message)}</span>
    </div>`;
  }

  /** Horizontal bar for top-endpoints display. */
  function endpointBar(path, count, maxCount) {
    const pct = maxCount > 0 ? (count / maxCount * 100) : 0;
    return `<div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.375rem;">
      <span class="mono truncate" style="flex:1;min-width:0;font-size:0.8125rem;color:var(--text-secondary);" title="${escapeHtml(path)}">${escapeHtml(path)}</span>
      <div style="width:120px;flex-shrink:0;">
        <div class="progress-bar"><div class="progress-bar-fill" style="width:${pct.toFixed(1)}%;background:var(--primary);"></div></div>
      </div>
      <span class="mono" style="width:40px;text-align:right;font-size:0.8125rem;color:var(--text-muted);">${formatNumber(count)}</span>
    </div>`;
  }

  /** Update history buffer and return trend indicator. */
  function updateHistoryAndGetTrend(history, currentValue) {
    history.push(currentValue);
    if (history.length > MAX_HISTORY_SIZE) {
      history.shift();
    }

    if (history.length < 2) {
      return { indicator: '', direction: 'neutral' };
    }

    const previousValue = history[history.length - 2];
    const diff = currentValue - previousValue;
    const pctChange = previousValue > 0 ? (diff / previousValue) * 100 : 0;

    if (diff > 0) {
      return {
        indicator: `<span style="font-size:0.7rem;color:var(--success);margin-left:0.25rem;">↑${pctChange.toFixed(1)}%</span>`,
        direction: 'up'
      };
    } else if (diff < 0) {
      return {
        indicator: `<span style="font-size:0.7rem;color:var(--danger);margin-left:0.25rem;">↓${Math.abs(pctChange).toFixed(1)}%</span>`,
        direction: 'down'
      };
    }
    return { indicator: '', direction: 'neutral' };
  }

  /** Build inline SVG sparkline chart. */
  function buildSparkline(data, id) {
    if (data.length < 2) return '';

    const min = Math.min(...data);
    const max = Math.max(...data);
    const range = max - min || 1;
    const width = 80;
    const height = 24;

    // Generate polyline points
    const points = data.map((val, idx) => {
      const x = (idx / (data.length - 1)) * width;
      const y = height - ((val - min) / range) * height;
      return `${x},${y}`;
    }).join(' ');

    // Determine color based on trend
    const isUpward = data[data.length - 1] >= data[0];
    const color = isUpward ? 'var(--success)' : 'var(--danger)';

    return `<svg width="${width}" height="${height}" style="display:block;margin-top:0.25rem;" viewBox="0 0 ${width} ${height}">
      <polyline
        fill="none"
        stroke="${color}"
        stroke-width="1.5"
        stroke-linecap="round"
        stroke-linejoin="round"
        points="${points}"
      />
    </svg>`;
  }

  // ---------------------------------------------------------------------------
  // Render sections
  // ---------------------------------------------------------------------------

  /** Build the top stat cards. */
  function buildStatCards(overview, graphs, accessStats) {
    const totalEntities = sumAcrossGraphs(graphs, 'entities');
    const totalRelations = sumAcrossGraphs(graphs, 'relations');
    const totalEpisodes = sumAcrossGraphs(graphs, 'episodes');
    const successRate = accessStats.success_rate ?? 0;
    const avgLatency = accessStats.avg_duration_ms ?? 0;

    // Color-code success rate
    let srClass = 'text-success';
    if (successRate < 90) srClass = 'text-error';
    else if (successRate < 99) srClass = 'text-warning';

    // Update history buffers and compute trends
    const entityTrend = updateHistoryAndGetTrend(_entityCountHistory, totalEntities);
    const relationTrend = updateHistoryAndGetTrend(_relationCountHistory, totalRelations);

    const isNeo4jBackend = isNeo4j();
    const gridCols = isNeo4jBackend ? 'lg:grid-cols-8' : 'lg:grid-cols-6';

    let html = `<div class="grid grid-cols-2 md:grid-cols-3 ${gridCols} gap-4 mb-6">
      <!-- Uptime -->
      <div class="stat-card">
        <div class="stat-label">${t('dashboard.uptime')}</div>
        <div class="stat-value text-info">${escapeHtml(overview.uptime_display)}</div>
        <div style="font-size:0.75rem;color:var(--text-muted);margin-top:0.25rem;">${t('dashboard.since', { time: overview.start_time })}</div>
      </div>
      <!-- Graph Count -->
      <div class="stat-card">
        <div class="stat-label">${t('dashboard.graphs')}</div>
        <div class="stat-value text-primary">${formatNumber(overview.graph_count)}</div>
        <div style="font-size:0.75rem;color:var(--text-muted);margin-top:0.25rem;">${t('dashboard.threads', { count: overview.python_threads_total ?? 0 })}</div>
      </div>
      <!-- Total Entities -->
      <div class="stat-card">
        <div class="stat-label">${t('dashboard.totalEntities')}</div>
        <div style="display:flex;align-items:baseline;gap:0.25rem;">
          <div class="stat-value">${formatNumber(totalEntities)}</div>
          ${entityTrend.indicator}
        </div>
        ${_entityCountHistory.length > 1 ? buildSparkline(_entityCountHistory, 'entity-sparkline') : ''}
      </div>
      <!-- Total Relations -->
      <div class="stat-card">
        <div class="stat-label">${t('dashboard.totalRelations')}</div>
        <div style="display:flex;align-items:baseline;gap:0.25rem;">
          <div class="stat-value">${formatNumber(totalRelations)}</div>
          ${relationTrend.indicator}
        </div>
        ${_relationCountHistory.length > 1 ? buildSparkline(_relationCountHistory, 'relation-sparkline') : ''}
      </div>
      <!-- API Success Rate -->
      <div class="stat-card">
        <div class="stat-label">${t('dashboard.successRate')}</div>
        <div class="stat-value ${srClass}">${successRate.toFixed(1)}%</div>
        <div style="font-size:0.75rem;color:var(--text-muted);margin-top:0.25rem;">${t('dashboard.errors', { count: formatNumber(accessStats.error_count) })}</div>
      </div>
      <!-- Avg Latency -->
      <div class="stat-card">
        <div class="stat-label">${t('dashboard.avgLatency')}</div>
        <div class="stat-value">${avgLatency.toFixed(1)}<span style="font-size:0.875rem;color:var(--text-muted);"> ${t('dashboard.ms')}</span></div>
        <div style="font-size:0.75rem;color:var(--text-muted);margin-top:0.25rem;">${t('dashboard.peakMs', { count: formatNumber(accessStats.max_duration_ms) })}</div>
      </div>`;

    if (isNeo4jBackend) {
      html += `
      <!-- Episodes (Neo4j only) -->
      <div class="stat-card" style="cursor:pointer;" onclick="navigate('#episodes')">
        <div class="stat-label">${t('nav.episodes')}</div>
        <div class="stat-value" style="color:#14b8a6;">${formatNumber(totalEpisodes)}</div>
      </div>
      <!-- Communities (Neo4j only) -->
      <div class="stat-card" style="cursor:pointer;" onclick="navigate('#communities')">
        <div class="stat-label">${t('nav.communities')}</div>
        <div class="stat-value" style="color:#8b5cf6;">${formatNumber(_communityCount)}</div>
      </div>`;
    }

    html += '</div>';
    return html;
  }

  /** Build graph list cards. */
  function buildGraphList(graphs) {
    if (!graphs.length) return emptyState(t('dashboard.noGraphs'));

    const cards = graphs.map(g => {
      const s = g.storage || {};
      const q = g.queue || {};
      const th = g.threads || {};
      const isActive = (q.running_count || 0) > 0 || (q.queued_count || 0) > 0;

      return `<div class="card mb-3">
        <div class="card-header">
          <div style="display:flex;align-items:center;gap:0.5rem;">
            <i data-lucide="git-branch" style="width:16px;height:16px;color:var(--primary);"></i>
            <span class="card-title" style="cursor:pointer;" onclick="setGraphId('${escapeHtml(g.graph_id)}');navigate('#graph');">${escapeHtml(g.graph_id)}</span>
            ${isActive ? `<span class="badge badge-info" style="margin-left:0.25rem;">${t('dashboard.active')}</span>` : ''}
          </div>
          <span style="font-size:0.75rem;color:var(--text-muted);">${t('dashboard.queueRunning')}: ${q.running_count || 0} / ${t('dashboard.queueQueued')}: ${q.queued_count || 0}</span>
        </div>
        <div class="grid grid-cols-3 gap-4">
          <div>
            <div style="font-size:0.75rem;color:var(--text-muted);">${t('dashboard.totalEntities')}</div>
            <div class="mono" style="font-size:1rem;font-weight:600;">${formatNumber(s.entities)}</div>
          </div>
          <div>
            <div style="font-size:0.75rem;color:var(--text-muted);">${t('dashboard.totalRelations')}</div>
            <div class="mono" style="font-size:1rem;font-weight:600;">${formatNumber(s.relations)}</div>
          </div>
          <div>
            <div style="font-size:0.75rem;color:var(--text-muted);">${t('dashboard.episodes')}</div>
            <div class="mono" style="font-size:1rem;font-weight:600;">${formatNumber(s.episodes)}</div>
          </div>
        </div>
        ${(th.python_threads_total > 0) ? `<div style="margin-top:0.5rem;display:flex;flex-wrap:wrap;gap:0.5rem;font-size:0.7rem;">
          <span style="color:${(th.remember_worker_threads_busy || 0) > 0 ? 'var(--success)' : 'var(--text-muted)'};">${t('dashboard.taskThreads', { busy: th.remember_worker_threads_busy || 0, alive: th.remember_worker_threads_alive || 0 })}</span>
          <span style="color:${(th.window_threads_busy || 0) > 0 ? 'var(--info)' : 'var(--text-muted)'};">${t('dashboard.windowThreads', { busy: th.window_threads_busy || 0, max: th.window_threads_configured || th.window_threads_alive || 0 })}</span>
          <span style="color:${(th.llm_threads_busy || 0) > 0 ? 'var(--warning)' : 'var(--text-muted)'};">${t('dashboard.llmThreads', { busy: th.llm_threads_busy || 0, max: th.llm_threads_max || th.llm_threads_alive || 0 })}</span>
        </div>` : ''}
      </div>`;
    }).join('');

    return cards;
  }

  /** Build the system logs section. */
  function buildLogsSection(logs) {
    const filterBtns = [
      { label: t('dashboard.logAll'), val: '' },
      { label: t('dashboard.logInfo'), val: 'INFO' },
      { label: t('dashboard.logWarn'), val: 'WARN' },
      { label: t('dashboard.logError'), val: 'ERROR' },
    ].map(({ label, val }) => {
      const isActive = _logLevel === val;
      return `<button class="btn btn-sm ${isActive ? 'btn-secondary' : 'btn-ghost'}" data-log-filter="${val}">${label}</button>`;
    }).join('');

    const logBody = logs.length
      ? logs.map(entry => logRow(entry)).join('')
      : `<div class="empty-state" style="padding:2rem;"><p>${t('dashboard.noLogs')}</p></div>`;

    return `<div class="card">
      <div class="card-header">
        <div class="card-title" style="display:flex;align-items:center;gap:0.5rem;">
          <i data-lucide="scroll-text" style="width:16px;height:16px;"></i>
          ${t('dashboard.systemLogs')}
        </div>
        <div style="display:flex;align-items:center;gap:0.25rem;" id="dashboard-log-filters">
          ${filterBtns}
        </div>
      </div>
      <div id="dashboard-log-body" style="max-height:320px;overflow-y:auto;background:var(--bg-input);border-radius:0.5rem;border:1px solid var(--border-color);">
        ${logBody}
      </div>
      <div style="display:flex;align-items:center;justify-content:space-between;margin-top:0.5rem;">
        <label style="display:flex;align-items:center;gap:0.375rem;font-size:0.8125rem;color:var(--text-muted);cursor:pointer;">
          <input type="checkbox" id="dashboard-log-autoscroll" ${_autoScroll ? 'checked' : ''} style="accent-color:var(--primary);">
          ${t('dashboard.autoScroll')}
        </label>
        <span style="font-size:0.75rem;color:var(--text-muted);" id="dashboard-log-count">${t('dashboard.logRecords', { count: logs.length })}</span>
      </div>
    </div>`;
  }

  /** Build the active tasks section (card layout). */
  function buildTasksSection(tasks) {
    // 分组：running → queued → completed/failed，组内按 created_at 正序（入队时间从早到晚）
    const statusOrder = { running: 0, queued: 1, completed: 2, failed: 2 };
    const sorted = [...tasks].sort((a, b) => {
      const pa = statusOrder[a.status] ?? 3;
      const pb = statusOrder[b.status] ?? 3;
      if (pa !== pb) return pa - pb;
      return (a.created_at || 0) - (b.created_at || 0);
    });
    // active 全部显示，completed/failed 最多 5 个
    let doneCount = 0;
    const display = sorted.filter(tk => {
      if (tk.status === 'queued' || tk.status === 'running') return true;
      if (doneCount < 5) { doneCount++; return true; }
      return false;
    });

    if (!display.length) return emptyState(t('dashboard.noTasks'));

    const cards = display.map(tk => {
      const pct = tk.progress ?? 0;
      const pctCls = tk.status === 'failed' ? 'error' : tk.status === 'completed' ? 'success' : '';
      const isRunning = tk.status === 'running';
      const s6p = Math.min(1, Math.max(0, tk.step6_progress ?? 0));
      const s7p = Math.min(1, Math.max(0, tk.step7_progress ?? 0));
      const smp = Math.min(1, Math.max(0, tk.main_progress ?? 0));
      const overallPct = Math.min(1, Math.max(0, (smp + s6p + s7p) / 3));

      // 进度区域：running 时显示 总进度 + 滑窗(1–5) + 实体链 + 关系链
      let progressHtml;
      if (isRunning) {
        progressHtml = tripleProgressBar({
          smp, s6p, s7p,
          mainLabel: tk.main_label || '-',
          step6Label: tk.step6_label || '-',
          step7Label: tk.step7_label || '-',
        });
      } else {
        progressHtml = `
          ${progressBar(pct, pctCls)}
          <div style="font-size:0.7rem;color:var(--text-muted);margin-top:4px;">${escapeHtml(tk.phase_label || '-')}</div>`;
      }

      return `
        <div class="dashboard-task-card" data-task-id="${escapeHtml(tk.task_id)}" style="padding:10px 12px;border-bottom:1px solid var(--border-color);cursor:pointer;transition:background 0.1s;"
             onmouseenter="this.style.background='var(--bg-surface-hover)'" onmouseleave="this.style.background='transparent'">
          <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:6px;">
            <div style="display:flex;align-items:center;gap:8px;min-width:0;flex:1;">
              ${statusBadge(tk.status, tk.phase)}
              <span class="mono" style="font-size:0.7rem;color:var(--text-muted);flex-shrink:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;" title="${escapeHtml(tk.task_id)}">${escapeHtml(tk.task_id.slice(0, 10))}</span>
              <span style="font-size:0.8rem;color:var(--text-secondary);min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;" title="${escapeHtml(tk.source_name || '-')}">${escapeHtml(tk.source_name || '-')}</span>
              <span style="font-size:0.75rem;color:var(--text-muted);flex-shrink:0;white-space:nowrap;">${t('memory.overallProgress')} ${(overallPct * 100).toFixed(2)}%</span>
            </div>
            <span class="mono" style="font-size:0.75rem;color:var(--text-muted);flex-shrink:0;">${elapsedText(tk)}</span>
          </div>
          ${progressHtml}
        </div>
      `;
    }).join('');

    return `<div style="max-height:400px;overflow-y:auto;">${cards}</div>`;
  }

  /** Build the API stats section. */
  function buildApiStatsSection(stats) {
    const topEndpoints = (stats.top_endpoints || []).slice(0, 5);
    const maxCount = topEndpoints.length > 0 ? topEndpoints[0].count : 0;

    const bars = topEndpoints.length > 0
      ? topEndpoints.map(e => endpointBar(e.path, e.count, maxCount)).join('')
      : `<div style="font-size:0.8125rem;color:var(--text-muted);">${t('dashboard.noEndpoints')}</div>`;

    const recentErrors = (stats.recent_errors || []);
    const errorList = recentErrors.length > 0
      ? recentErrors.map(e => `<div style="display:flex;align-items:center;gap:0.5rem;padding:0.25rem 0;font-size:0.8125rem;">
          <span class="mono" style="color:var(--text-muted);flex-shrink:0;">${escapeHtml(e.time)}</span>
          <span class="badge badge-error">${e.status_code}</span>
          <span class="mono truncate" style="color:var(--text-secondary);min-width:0;" title="${escapeHtml(e.method + ' ' + e.path)}">${escapeHtml(e.method + ' ' + e.path)}</span>
        </div>`).join('')
      : `<div style="font-size:0.8125rem;color:var(--text-muted);">${t('dashboard.noRecentErrors')}</div>`;

    return `<div class="card">
      <div class="card-header">
        <div class="card-title" style="display:flex;align-items:center;gap:0.5rem;">
          <i data-lucide="activity" style="width:16px;height:16px;"></i>
          ${t('dashboard.apiStats')}
        </div>
      </div>

      <!-- Summary numbers -->
      <div class="grid grid-cols-2 gap-3 mb-4">
        <div>
          <div style="font-size:0.75rem;color:var(--text-muted);">${t('dashboard.requestsPerMin')}</div>
          <div class="mono" style="font-size:1.125rem;font-weight:600;color:var(--primary);">${formatNumber(stats.requests_per_minute)}</div>
        </div>
        <div>
          <div style="font-size:0.75rem;color:var(--text-muted);">${t('dashboard.totalRequests')}</div>
          <div class="mono" style="font-size:1.125rem;font-weight:600;">${formatNumber(stats.total_requests)}</div>
        </div>
      </div>

      <!-- Top endpoints -->
      <div style="margin-bottom:1rem;">
        <div style="font-size:0.8125rem;font-weight:600;color:var(--text-secondary);margin-bottom:0.5rem;">${t('dashboard.topEndpoints')}</div>
        ${bars}
      </div>

      <div class="divider"></div>

      <!-- Recent errors -->
      <div>
        <div style="font-size:0.8125rem;font-weight:600;color:var(--text-secondary);margin-bottom:0.5rem;">${t('dashboard.recentErrors')}</div>
        ${errorList}
      </div>
    </div>`;
  }

  // ---------------------------------------------------------------------------
  // Data fetching — per-section, independent
  // ---------------------------------------------------------------------------

  // Cached data per section
  let _overview = {};
  let _graphs = [];
  let _tasks = [];
  let _logs = [];
  let _accessStats = {};
  let _episodeCount = 0;
  let _communityCount = 0;

  // History buffers for trend indicators and sparklines (last 20 data points)
  let _entityCountHistory = [];
  let _relationCountHistory = [];
  const MAX_HISTORY_SIZE = 20;

  async function fetchOverview() {
    try {
      const res = await state.api.systemOverview();
      _overview = res.data || {};
      updateStatCards();
    } catch (err) { console.warn('fetchOverview failed:', err); }
  }

  async function fetchGraphs() {
    try {
      const res = await state.api.systemGraphs();
      _graphs = res.data || [];
      updateGraphList();
      updateStatCards(); // stat cards also depend on graph storage counts
    } catch (err) { console.warn('fetchGraphs failed:', err); }
  }

  async function fetchTasks() {
    try {
      const res = await state.api.systemTasks(50);
      _tasks = res.data || [];
      updateTasks();
    } catch (err) { console.warn('fetchTasks failed:', err); }
  }

  async function fetchLogs() {
    try {
      const res = await state.api.systemLogs(100, _logLevel || undefined);
      _logs = res.data || [];
      updateLogs();
    } catch (err) { console.warn('fetchLogs failed:', err); }
  }

  async function fetchAccessStats() {
    try {
      const res = await state.api.systemAccessStats(300);
      _accessStats = res.data || {};
      updateApiStats();
      updateStatCards(); // stat cards also depend on access stats
    } catch (err) { console.warn('fetchAccessStats failed:', err); }
  }

  async function fetchEpisodeCount() {
    if (!isNeo4j()) return;
    try {
      const res = await state.api.findStats(state.currentGraphId);
      _episodeCount = res.data?.total_episodes || 0;
      _communityCount = res.data?.total_communities || 0;
      updateStatCards();
    } catch (err) { console.warn('fetchEpisodeCount failed:', err); }
  }

  async function fetchGraphStats() {
    try {
      const res = await state.api.getGraphStats(state.currentGraphId);
      const stats = res.data || res;
      const container = document.getElementById('graphStatsContainer');
      if (!container) return;
      const graphLabel = state.currentGraphId || 'default';
      container.innerHTML = `
        <div class="card">
          <div class="card-header">
            <div class="card-title" style="display:flex;align-items:center;gap:0.5rem;">
              <i data-lucide="bar-chart-3" style="width:16px;height:16px;"></i>
              <span data-i18n="stats.graphStats">${t('stats.graphStats')}</span>
              <span style="font-size:0.75rem;color:var(--text-muted);margin-left:0.5rem;">[${escapeHtml(graphLabel)}]</span>
            </div>
          </div>
          <div class="stats-grid" style="display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;">
            <div class="stat-card">
              <div class="stat-value">${formatNumber(stats.entity_count || 0)}</div>
              <div class="stat-label" data-i18n="dashboard.totalEntities">${t('dashboard.totalEntities')}</div>
            </div>
            <div class="stat-card">
              <div class="stat-value">${formatNumber(stats.relation_count || 0)}</div>
              <div class="stat-label" data-i18n="dashboard.totalRelations">${t('dashboard.totalRelations')}</div>
            </div>
            <div class="stat-card">
              <div class="stat-value">${(stats.avg_relations_per_entity || 0).toFixed(2)}</div>
              <div class="stat-label" data-i18n="stats.avgRelations">${t('stats.avgRelations')}</div>
            </div>
            <div class="stat-card">
              <div class="stat-value">${formatNumber(stats.max_relations_per_entity || 0)}</div>
              <div class="stat-label" data-i18n="stats.maxRelations">${t('stats.maxRelations')}</div>
            </div>
            <div class="stat-card">
              <div class="stat-value">${formatNumber(stats.isolated_entities || 0)}</div>
              <div class="stat-label" data-i18n="stats.isolatedEntities">${t('stats.isolatedEntities')}</div>
            </div>
            <div class="stat-card">
              <div class="stat-value">${(stats.graph_density || 0).toFixed(4)}</div>
              <div class="stat-label" data-i18n="stats.density">${t('stats.density')}</div>
            </div>
          </div>
        </div>`;
      if (window.I18N) window.I18N.applyLang(window.I18N.currentLang);
      if (window.lucide) lucide.createIcons({ nodes: [container] });
    } catch (e) {
      console.error('Failed to load graph stats:', e);
      const container = document.getElementById('graphStatsContainer');
      if (container && container.querySelector('.spinner')) container.innerHTML = '';
    }
  }

  // ---------------------------------------------------------------------------
  // Partial DOM updates — each section updates only its own DOM
  // ---------------------------------------------------------------------------

  function updateStatCards() {
    const el = document.getElementById('dashboard-stats');
    if (el) el.innerHTML = buildStatCards(_overview, _graphs, _accessStats);
    if (window.lucide) lucide.createIcons({ nodes: [el] });
  }

  function updateGraphList() {
    const listEl = document.getElementById('dashboard-graph-list');
    const countEl = document.getElementById('dashboard-graph-count');
    if (listEl) listEl.innerHTML = buildGraphList(_graphs);
    if (countEl) countEl.textContent = t('dashboard.graphCount', { count: _graphs.length });
    if (window.lucide) lucide.createIcons({ nodes: [listEl] });
  }

  let _lastTasksHtml = '';
  function updateTasks() {
    const tasksEl = document.getElementById('dashboard-tasks');
    const countEl = document.getElementById('dashboard-task-count');
    if (tasksEl) {
      const html = buildTasksSection(_tasks);
      if (html !== _lastTasksHtml) {
        _lastTasksHtml = html;
        tasksEl.innerHTML = html;
        if (window.lucide) lucide.createIcons({ nodes: [tasksEl] });
      }
    }
    if (countEl) countEl.textContent = `${_tasks.length}`;
  }

  function updateLogs() {
    const logsEl = document.getElementById('dashboard-logs');
    if (!logsEl) return;
    logsEl.innerHTML = buildLogsSection(_logs);
    bindLogFilterListeners(logsEl);
    const cb = document.getElementById('dashboard-log-autoscroll');
    if (cb) cb.checked = _autoScroll;
    const logBody = document.getElementById('dashboard-log-body');
    if (logBody && _autoScroll) logBody.scrollTop = 0;
    if (window.lucide) lucide.createIcons({ nodes: [logsEl] });
  }

  function updateApiStats() {
    const el = document.getElementById('dashboard-api-stats');
    if (el) el.innerHTML = buildApiStatsSection(_accessStats);
    if (window.lucide) lucide.createIcons({ nodes: [el] });
  }

  /** Re-bind log filter buttons after DOM update. */
  function bindLogFilterListeners(parentEl) {
    const logFilters = parentEl.querySelector('#dashboard-log-filters');
    if (!logFilters) return;
    logFilters.addEventListener('click', (e) => {
      const btn = e.target.closest('[data-log-filter]');
      if (!btn) return;
      _logLevel = btn.dataset.logFilter;
      fetchLogs(); // only fetch logs, not everything
    });
  }

  // ---------------------------------------------------------------------------
  // Main render
  // ---------------------------------------------------------------------------

  async function render(container, params) {
    // Initial layout with loading placeholders
    container.innerHTML = `<div class="page-enter">

      <!-- Top stat cards -->
      <div id="dashboard-stats">
        <div class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4 mb-6">${Array(6).fill('<div class="stat-card"><div class="stat-label" style="height:0.75rem;background:var(--bg-input);border-radius:4px;width:60%;"></div><div class="stat-value" style="height:1.5rem;background:var(--bg-input);border-radius:4px;width:40%;margin-top:0.5rem;"></div></div>').join('')}</div>
      </div>

      <!-- Quick Actions -->
      <div class="mb-4" style="display:flex;flex-wrap:wrap;gap:0.5rem;">
        <a href="#search" class="btn btn-primary btn-sm" style="text-decoration:none;">
          <i data-lucide="search" style="width:14px;height:14px;margin-right:4px;"></i>
          ${t('nav.search') || 'Search'}
        </a>
        <a href="#chat" class="btn btn-secondary btn-sm" style="text-decoration:none;">
          <i data-lucide="message-circle" style="width:14px;height:14px;margin-right:4px;"></i>
          ${t('nav.chat') || 'Chat'}
        </a>
        <a href="#memory" class="btn btn-secondary btn-sm" style="text-decoration:none;">
          <i data-lucide="database" style="width:14px;height:14px;margin-right:4px;"></i>
          ${t('nav.memory') || 'Remember'}
        </a>
        <a href="#dream" class="btn btn-secondary btn-sm" style="text-decoration:none;">
          <i data-lucide="moon" style="width:14px;height:14px;margin-right:4px;"></i>
          ${t('nav.dream') || 'Dream'}
        </a>
        <a href="#graph" class="btn btn-secondary btn-sm" style="text-decoration:none;">
          <i data-lucide="network" style="width:14px;height:14px;margin-right:4px;"></i>
          ${t('nav.graph') || 'Graph'}
        </a>
      </div>

      <!-- Live Refresh Indicator -->
      <div class="mb-4" style="display:flex;align-items:center;gap:0.5rem;font-size:0.75rem;color:var(--text-muted);">
        <span class="live-indicator"></span>
        <span>${t('dashboard.liveRefresh') || 'Live auto-refresh active'}</span>
      </div>

      <!-- Graph Statistics -->
      <div id="graphStatsContainer" class="mb-6">${spinnerHtml()}</div>

      <!-- Two-column layout -->
      <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">

        <!-- Left column: Tasks + Logs -->
        <div class="lg:col-span-2 flex flex-col gap-6">
          <div class="card">
            <div class="card-header">
              <div class="card-title" style="display:flex;align-items:center;gap:0.5rem;">
                <i data-lucide="list-checks" style="width:16px;height:16px;"></i>
                ${t('dashboard.task')}
              </div>
              <span style="font-size:0.75rem;color:var(--text-muted);" id="dashboard-task-count">-</span>
            </div>
            <div id="dashboard-tasks">${spinnerHtml()}</div>
          </div>

          <div id="dashboard-logs">${spinnerHtml()}</div>
        </div>

        <!-- Right column: Graph List + API Stats -->
        <div class="flex flex-col gap-6">
          <div class="card">
            <div class="card-header">
              <div class="card-title" style="display:flex;align-items:center;gap:0.5rem;">
                <i data-lucide="git-branch" style="width:16px;height:16px;"></i>
                ${t('dashboard.graphList')}
              </div>
              <div style="display:flex;align-items:center;gap:0.5rem;">
                <span style="font-size:0.75rem;color:var(--text-muted);" id="dashboard-graph-count">-</span>
                <button class="btn btn-primary" style="padding:0.25rem 0.75rem;font-size:0.75rem;" onclick="showCreateGraphModal()">
                  <i data-lucide="plus" style="width:14px;height:14px;margin-right:0.25rem;"></i>${t('dashboard.createGraph')}
                </button>
              </div>
            </div>
            <div id="dashboard-graph-list">${spinnerHtml()}</div>
          </div>

          <div id="dashboard-api-stats">${spinnerHtml()}</div>
        </div>
      </div>
    </div>`;

    if (window.lucide) lucide.createIcons();

    // Bind event: auto-scroll checkbox (after logs section renders)
    const autoScrollCb = document.getElementById('dashboard-log-autoscroll');
    if (autoScrollCb) {
      autoScrollCb.addEventListener('change', () => { _autoScroll = autoScrollCb.checked; });
    }

    // Initial load: fetch all in parallel (one-time)
    await Promise.all([
      fetchOverview(),
      fetchGraphs(),
      fetchTasks(),
      fetchLogs(),
      fetchAccessStats(),
      fetchEpisodeCount(),
      fetchGraphStats(),
    ]);

    // --- Independent refresh timers per section ---

    // Tasks: every 3s (progress tracking needs responsiveness)
    state.refreshTimers.dash_tasks = setInterval(fetchTasks, 3000);

    // Logs: every 10s
    state.refreshTimers.dash_logs = setInterval(fetchLogs, 10000);

    // Overview + Graphs: every 15s (graphs is expensive: scans filesystem)
    state.refreshTimers.dash_graphs = setInterval(async () => {
      await fetchOverview();
      await fetchGraphs();
    }, 15000);

    // Access stats: every 30s (analytics, changes slowly)
    state.refreshTimers.dash_access = setInterval(fetchAccessStats, 30000);

    // Graph stats: every 30s
    state.refreshTimers.dash_graph_stats = setInterval(fetchGraphStats, 30000);
  }

  // ---------------------------------------------------------------------------
  // Create Graph
  // ---------------------------------------------------------------------------

  window.showCreateGraphModal = function() {
    if (document.getElementById('create-graph-overlay')) return;
    const overlay = document.createElement('div');
    overlay.id = 'create-graph-overlay';
    overlay.style.cssText = 'position:fixed;inset:0;background:rgba(0,0,0,0.5);display:flex;align-items:center;justify-content:center;z-index:1000;';
    overlay.innerHTML = `
      <div class="card" style="width:380px;max-width:90vw;">
        <div class="card-header">
          <span class="card-title">${t('dashboard.createGraph')}</span>
          <button onclick="document.getElementById('create-graph-overlay').remove()" style="background:none;border:none;color:var(--text-muted);cursor:pointer;font-size:1.2rem;">&times;</button>
        </div>
        <div style="padding:1rem;">
          <label style="font-size:0.8rem;color:var(--text-muted);display:block;margin-bottom:0.35rem;">${t('dashboard.graphId')}</label>
          <input id="create-graph-input" type="text" placeholder="${t('dashboard.graphIdPlaceholder')}"
            style="width:100%;padding:0.5rem;border:1px solid var(--border);border-radius:6px;background:var(--bg-secondary);color:var(--text);font-size:0.85rem;outline:none;"
            onkeydown="if(event.key==='Enter')doCreateGraph()" />
          <div id="create-graph-error" style="margin-top:0.35rem;font-size:0.75rem;color:var(--danger);min-height:1rem;"></div>
          <div style="display:flex;gap:0.5rem;margin-top:1rem;justify-content:flex-end;">
            <button class="btn btn-secondary" onclick="document.getElementById('create-graph-overlay').remove()" style="padding:0.4rem 1rem;font-size:0.8rem;">${t('common.cancel')}</button>
            <button class="btn btn-primary" onclick="doCreateGraph()" style="padding:0.4rem 1rem;font-size:0.8rem;">${t('common.confirm')}</button>
          </div>
        </div>
      </div>`;
    document.body.appendChild(overlay);
    overlay.addEventListener('click', e => { if (e.target === overlay) overlay.remove(); });
    setTimeout(() => document.getElementById('create-graph-input').focus(), 50);
  };

  window.doCreateGraph = async function() {
    const input = document.getElementById('create-graph-input');
    const errEl = document.getElementById('create-graph-error');
    const graphId = (input.value || '').trim();
    if (!graphId) { errEl.textContent = t('dashboard.graphIdRequired'); return; }
    errEl.textContent = '';
    try {
      const data = await state.api.createGraph(graphId);
      if (data.error) { errEl.textContent = data.error; return; }
      document.getElementById('create-graph-overlay').remove();
      setGraphId(graphId);
      // 刷新 dashboard 图谱列表
      await Promise.all([fetchOverview(), fetchGraphs()]);
      showToast(t('dashboard.graphCreated', { id: graphId }), 'success');
    } catch (e) {
      errEl.textContent = e.message;
    }
  };

  // ---------------------------------------------------------------------------
  // Cleanup
  // ---------------------------------------------------------------------------

  function destroy() {
    _logLevel = '';
    _autoScroll = true;
    _overview = {};
    _graphs = [];
    _tasks = [];
    _logs = [];
    _accessStats = {};
    _communityCount = 0;
  }

  // ---------------------------------------------------------------------------
  // Register page
  // ---------------------------------------------------------------------------
  registerPage('dashboard', { render, destroy });
})();

(function () {
  let docs = [];
  let filteredDocs = [];
  let selectedDocVersions = new Set();
  let graphData = null;
  let graphModel = null;
  let network = null;
  let nodesDataSet = null;
  let edgesDataSet = null;
  let edgeMetaById = new Map();
  let nodeMetaById = new Map();
  let pinnedPositions = {};
  let hoverPanel = null;
  let loadingGraph = false;
  let physicsFreezeTimer = null;
  let naturalFitTimer = null;
  let lastNaturalFitAt = 0;
  let relationStreamTimer = null;
  let growthController = null;
  let growthControllers = new Map();
  let loadedDocVersions = new Set();
  let growthOutlinesByDoc = new Map();
  let growthRunId = 0;
  let growthPauseRequested = false;
  let growthLoaded = { episodes: 0, concepts: 0, relations: 0, edges: 0 };
  let growthTotals = null;
  let growthOutline = null;
  let growthRatePerSecond = 60;
  const GROWTH_RATE_OPTIONS = [60, 100, 40, 20];

  let showRelations = true;
  let showSourceEdges = false;
  let showLabels = true;
  let focusFamilyId = null;
  let docsCollapsed = false;

  let playbackTimer = null;
  let playbackStep = -1; // -1 means full graph.
  let playbackSpeed = 1.0;

  const ROLE_COLORS = {
    document: { bg: '#d946ef', border: '#f0abfc', glow: 'rgba(217,70,239,0.28)' },
    episode: { bg: '#0ea5e9', border: '#7dd3fc', glow: 'rgba(14,165,233,0.22)' },
    entity: { bg: '#14b8a6', border: '#5eead4', glow: 'rgba(20,184,166,0.20)' },
    relation: { bg: '#f59e0b', border: '#fbbf24', glow: 'rgba(245,158,11,0.28)' },
  };

  function docTitle(doc) {
    return doc.version_title || doc.title || doc.relative_path || doc.absolute_path || doc.source_id || 'Untitled';
  }

  function docPath(doc) {
    return doc.relative_path || doc.absolute_path || doc.uri || doc.source_id || '';
  }

  function conceptTitle(concept) {
    return concept.name || concept.summary || truncate(concept.content || concept.family_id || '', 72);
  }

  function isLightTheme() {
    return document.documentElement.getAttribute('data-theme') === 'light';
  }

  function labelColor() {
    return isLightTheme() ? '#1e293b' : '#e2e8f0';
  }

  function mutedColor() {
    return isLightTheme() ? '#64748b' : '#94a3b8';
  }

  function field(label, value) {
    return `<div class="graph-field">
      <span>${escapeHtml(label)}</span>
      <strong>${value || '-'}</strong>
    </div>`;
  }

  function pill(text, color) {
    return `<span class="graph-pill" style="${color ? `border-color:${color};color:${color};` : ''}">${escapeHtml(text || '-')}</span>`;
  }

  function normalizeText(value, limit) {
    return truncate(String(value || '').replace(/\s+/g, ' ').trim(), limit || 120);
  }

  function render(container) {
    container.innerHTML = `
      <div class="page-enter graph-viz-shell">
        <aside class="card graph-doc-panel">
          <div class="card-header" style="gap:0.75rem;align-items:flex-start;">
            <button id="graph-doc-collapse" class="btn btn-secondary btn-sm" title="收起文档栏" style="flex-shrink:0;">
              <i data-lucide="panel-left-close" style="width:14px;height:14px;"></i>
            </button>
            <div class="graph-doc-head-text" style="min-width:0;">
              <div class="card-title">Markdown 文档</div>
              <div class="graph-subtitle">按入库时间排序，选择后渲染子图</div>
            </div>
            <button id="graph-doc-refresh" class="btn btn-secondary btn-sm graph-doc-refresh" title="刷新文档">
              <i data-lucide="refresh-cw" style="width:14px;height:14px;"></i>
            </button>
          </div>
          <div class="graph-doc-collapsed-label">文档</div>
          <div class="graph-doc-tools">
            <input id="graph-doc-filter" class="input" placeholder="搜索文档标题或路径..." />
            <div style="display:flex;gap:0.45rem;flex-wrap:wrap;">
              <button id="graph-select-visible" class="btn btn-secondary btn-sm">全选列表</button>
              <button id="graph-clear-selection" class="btn btn-secondary btn-sm">清空</button>
            </div>
            <div id="graph-doc-count" class="graph-subtitle"></div>
          </div>
          <div id="graph-doc-list" class="graph-doc-list"></div>
        </aside>

        <section class="card graph-canvas-card">
          <div class="card-header graph-toolbar">
            <div style="min-width:0;margin-right:auto;">
              <div id="graph-main-title" class="card-title">图谱可视化</div>
              <div id="graph-summary" class="graph-subtitle">选择一个或多个文档</div>
            </div>
            <label class="graph-toggle"><input id="graph-toggle-relations" type="checkbox" checked> 关系边</label>
            <label class="graph-toggle"><input id="graph-toggle-source" type="checkbox"> 溯源边</label>
            <label class="graph-toggle"><input id="graph-toggle-labels" type="checkbox" checked> 标签</label>
            <button id="graph-exit-focus" class="btn btn-secondary btn-sm" style="display:none;">退出聚焦</button>
            <button id="graph-fit" class="btn btn-secondary btn-sm">适配视图</button>
          </div>
          <div id="document-graph-canvas" class="graph-canvas-wrap">
            <div id="graph-empty" class="empty-state graph-empty">
              <i data-lucide="git-fork"></i>
              <p>选择左侧 Markdown 文档后显示文档、Episode、实体和关系边。</p>
            </div>
          </div>
          <div id="graph-playback" class="graph-playback" style="display:none;">
            <div class="timeline-live-dot"></div>
            <button id="graph-step-back" class="timeline-btn" title="后退一步"><i data-lucide="skip-back" style="width:11px;height:11px;"></i></button>
            <button id="graph-play" class="timeline-btn" title="播放"><i data-lucide="play" style="width:11px;height:11px;"></i></button>
            <button id="graph-step-forward" class="timeline-btn" title="前进一步"><i data-lucide="skip-forward" style="width:11px;height:11px;"></i></button>
            <button id="graph-reset-full" class="timeline-btn" title="回到完整图"><i data-lucide="maximize-2" style="width:11px;height:11px;"></i></button>
            <button id="graph-speed" class="timeline-btn" title="速度">60/s</button>
            <div class="graph-play-track" id="graph-play-track">
              <div class="graph-play-fill" id="graph-play-fill"></div>
            </div>
            <span id="graph-play-label" class="mono graph-play-label">完整图</span>
          </div>
        </section>

        <aside class="card graph-detail-panel">
          <div class="card-header">
            <div>
              <div class="card-title">详情</div>
              <div id="graph-detail-subtitle" class="graph-subtitle">点击节点或关系边查看详情</div>
            </div>
          </div>
          <div id="graph-detail" class="graph-detail-body">
            ${emptyState('暂无选中节点')}
          </div>
        </aside>
      </div>
      <style>
        .graph-viz-shell{height:calc(100vh - 6.5rem);min-height:640px;display:grid;grid-template-columns:310px minmax(0,1fr) 360px;gap:1rem;transition:grid-template-columns 0.18s ease;}
        .graph-viz-shell.docs-collapsed{grid-template-columns:52px minmax(0,1fr) 360px;}
        .graph-doc-panel,.graph-detail-panel,.graph-canvas-card{min-height:0;display:flex;flex-direction:column;overflow:hidden;}
        .graph-doc-collapsed-label{display:none;writing-mode:vertical-rl;letter-spacing:0.18em;color:var(--text-secondary);font-weight:600;align-self:center;margin-top:0.75rem;}
        .graph-viz-shell.docs-collapsed .graph-doc-head-text,
        .graph-viz-shell.docs-collapsed .graph-doc-refresh,
        .graph-viz-shell.docs-collapsed .graph-doc-tools,
        .graph-viz-shell.docs-collapsed .graph-doc-list{display:none;}
        .graph-viz-shell.docs-collapsed .graph-doc-panel .card-header{padding:0.65rem;justify-content:center;}
        .graph-viz-shell.docs-collapsed .graph-doc-collapsed-label{display:block;}
        .graph-doc-tools{padding:0 1rem 0.75rem;display:flex;flex-direction:column;gap:0.55rem;}
        .graph-doc-list{min-height:0;overflow:auto;padding:0 0.5rem 0.75rem;}
        .graph-subtitle{font-size:0.75rem;color:var(--text-muted);margin-top:0.18rem;}
        .graph-toolbar{gap:0.75rem;flex-wrap:wrap;}
        .graph-toggle{display:flex;align-items:center;gap:0.35rem;font-size:0.78rem;color:var(--text-secondary);white-space:nowrap;}
        .doc-graph-item{display:grid;grid-template-columns:auto 1fr;gap:0.55rem;padding:0.65rem;border-radius:0.5rem;cursor:pointer;border:1px solid transparent;}
        .doc-graph-item:hover{background:var(--bg-surface-hover);}
        .doc-graph-item.selected{background:var(--primary-dim);border-color:var(--primary);}
        .doc-graph-title{font-size:0.85rem;font-weight:600;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:var(--text-primary);}
        .doc-graph-meta{font-size:0.72rem;color:var(--text-muted);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;margin-top:0.15rem;}
        .graph-canvas-wrap{position:relative;min-height:0;flex:1;background:
          radial-gradient(circle at 25% 20%, color-mix(in srgb, var(--primary) 8%, transparent), transparent 28%),
          radial-gradient(circle at 80% 70%, rgba(14,165,233,0.08), transparent 30%),
          var(--bg-secondary);border-top:1px solid var(--border-color);overflow:hidden;}
        .graph-canvas-wrap::after{content:"DOCUMENT FIRST GRAPH";position:absolute;right:12px;bottom:10px;font-size:0.68rem;letter-spacing:0.08em;color:color-mix(in srgb,var(--text-muted) 35%,transparent);pointer-events:none;}
        .graph-empty{height:100%;}
        .graph-detail-body{min-height:0;overflow:auto;padding:1rem;font-size:0.85rem;}
        .graph-field{display:grid;grid-template-columns:92px minmax(0,1fr);gap:0.35rem 0.65rem;font-size:0.8rem;align-items:start;margin-bottom:0.42rem;}
        .graph-field span{color:var(--text-muted);}
        .graph-field strong{font-weight:500;color:var(--text-secondary);min-width:0;overflow-wrap:anywhere;}
        .graph-pill{display:inline-flex;align-items:center;border:1px solid var(--border-color);border-radius:999px;padding:0.12rem 0.45rem;font-size:0.68rem;background:var(--bg-surface-hover);color:var(--text-secondary);margin-right:0.25rem;margin-top:0.35rem;}
        .graph-detail-section{border-top:1px solid var(--border-color);padding-top:0.8rem;margin-top:0.8rem;}
        .graph-detail-actions{display:flex;gap:0.45rem;flex-wrap:wrap;margin:0.8rem 0;}
        .graph-playback{flex-shrink:0;display:flex;align-items:center;gap:0.45rem;padding:0.45rem 0.65rem;border-top:1px solid var(--border-color);background:color-mix(in srgb,var(--bg-surface) 92%,transparent);}
        .graph-play-track{position:relative;flex:1;height:6px;border-radius:999px;background:var(--bg-input);overflow:hidden;border:1px solid var(--border-color);}
        .graph-play-fill{height:100%;width:100%;background:linear-gradient(90deg,#0ea5e9,#14b8a6,#f59e0b);transition:width 0.18s ease;}
        .graph-play-label{font-size:0.72rem;color:var(--text-muted);min-width:92px;text-align:right;}
        .graph-version-row{border:1px solid var(--border-color);border-radius:0.5rem;padding:0.55rem;background:var(--bg-secondary);margin-bottom:0.45rem;}
        .graph-modal-grid{display:grid;grid-template-columns:110px minmax(0,1fr);gap:0.45rem 0.75rem;font-size:0.82rem;}
        .graph-modal-grid span{color:var(--text-muted);}
        .graph-modal-grid strong{font-weight:500;overflow-wrap:anywhere;}
        .graph-kind-badge{display:inline-flex;align-items:center;border-radius:999px;padding:0.14rem 0.5rem;font-size:0.7rem;font-weight:600;margin-bottom:0.6rem;}
        .graph-document-source{max-height:58vh;overflow:auto;background:var(--bg-secondary);border:1px solid var(--border-color);border-radius:0.5rem;padding:0.8rem;font-family:var(--font-mono);font-size:0.8rem;line-height:1.65;white-space:pre-wrap;word-break:break-word;color:var(--text-secondary);}
      </style>
    `;
    bindEvents();
    if (window.lucide) lucide.createIcons();
  }

  function bindEvents() {
    document.getElementById('graph-doc-collapse')?.addEventListener('click', () => setDocsCollapsed(!docsCollapsed));
    document.getElementById('graph-doc-refresh')?.addEventListener('click', loadDocs);
    document.getElementById('graph-doc-filter')?.addEventListener('input', debounce(applyFilter, 150));
    document.getElementById('graph-select-visible')?.addEventListener('click', () => {
      filteredDocs.forEach(d => selectedDocVersions.add(d.document_version_id));
      updateDocsList();
      loadSelectedGraph();
    });
    document.getElementById('graph-clear-selection')?.addEventListener('click', () => {
      selectedDocVersions.clear();
      updateDocsList();
      clearGraphCanvas();
    });
    document.getElementById('graph-toggle-relations')?.addEventListener('change', (e) => {
      showRelations = e.target.checked;
      drawGraph(playbackStep);
    });
    document.getElementById('graph-toggle-source')?.addEventListener('change', (e) => {
      showSourceEdges = e.target.checked;
      drawGraph(playbackStep);
    });
    document.getElementById('graph-toggle-labels')?.addEventListener('change', (e) => {
      showLabels = e.target.checked;
      drawGraph(playbackStep);
    });
    document.getElementById('graph-exit-focus')?.addEventListener('click', () => focusConcept(null));
    document.getElementById('graph-fit')?.addEventListener('click', () => network?.fit({ animation: { duration: 420, easingFunction: 'easeInOutQuad' } }));
    bindPlaybackEvents();
  }

  function bindPlaybackEvents() {
    document.getElementById('graph-play')?.addEventListener('click', () => {
      if (!toggleGraphGrowthPause()) togglePlayback();
    });
    document.getElementById('graph-step-back')?.addEventListener('click', () => stepPlayback(-1));
    document.getElementById('graph-step-forward')?.addEventListener('click', () => stepPlayback(1));
    document.getElementById('graph-reset-full')?.addEventListener('click', () => {
      stopPlayback();
      playbackStep = -1;
      drawGraph(-1);
    });
    document.getElementById('graph-speed')?.addEventListener('click', () => {
      if (hasActiveGrowth()) {
        const idx = GROWTH_RATE_OPTIONS.indexOf(growthRatePerSecond);
        growthRatePerSecond = GROWTH_RATE_OPTIONS[(idx + 1) % GROWTH_RATE_OPTIONS.length];
        document.getElementById('graph-speed').textContent = `${growthRatePerSecond}/s`;
        updateSummary(graphModel ? visibleModelForStep(-1) : null);
        return;
      }
      const speeds = [1.0, 0.6, 0.3, 1.5];
      const idx = speeds.indexOf(playbackSpeed);
      playbackSpeed = speeds[(idx + 1) % speeds.length];
      document.getElementById('graph-speed').textContent = `${playbackSpeed}s`;
      if (playbackTimer) {
        stopPlayback();
        startPlayback();
      }
    });
  }

  async function loadDocs() {
    const list = document.getElementById('graph-doc-list');
    if (list) list.innerHTML = `<div style="padding:1rem;">${spinnerHtml()} 加载文档...</div>`;
    try {
      const res = await state.api.listDocs(state.currentGraphId);
      docs = (res.data?.docs || []).slice().sort((a, b) => String(b.processed_time || '').localeCompare(String(a.processed_time || '')));
      selectedDocVersions = new Set([...selectedDocVersions].filter(id => docs.some(d => d.document_version_id === id)));
      applyFilter();
      updateGraphTitle();
    } catch (err) {
      if (list) list.innerHTML = emptyState(`加载文档失败: ${escapeHtml(err.message)}`);
    }
  }

  function applyFilter() {
    const q = (document.getElementById('graph-doc-filter')?.value || '').trim().toLowerCase();
    filteredDocs = q
      ? docs.filter(d => `${docTitle(d)} ${docPath(d)} ${d.content_hash || ''}`.toLowerCase().includes(q))
      : docs.slice();
    updateDocsList();
  }

  function updateDocsList() {
    const countEl = document.getElementById('graph-doc-count');
    if (countEl) countEl.textContent = `${filteredDocs.length} 个文档，已选 ${selectedDocVersions.size} 个`;
    const list = document.getElementById('graph-doc-list');
    if (!list) return;
    if (!docs.length) {
      list.innerHTML = emptyState('暂无 Markdown 文档。先在记忆页上传文件，或调用 vault index。');
      return;
    }
    if (!filteredDocs.length) {
      list.innerHTML = emptyState('没有匹配的文档');
      return;
    }
    list.innerHTML = filteredDocs.map(d => {
      const id = d.document_version_id;
      const selected = selectedDocVersions.has(id);
      return `
        <div class="doc-graph-item ${selected ? 'selected' : ''}" data-doc-id="${escapeAttr(id)}">
          <input class="doc-graph-check" type="checkbox" ${selected ? 'checked' : ''} data-doc-id="${escapeAttr(id)}" style="margin-top:0.2rem;">
          <div style="min-width:0;">
            <div class="doc-graph-title" title="${escapeAttr(docTitle(d))}">${escapeHtml(docTitle(d))}</div>
            <div class="doc-graph-meta" title="${escapeAttr(docPath(d))}">${escapeHtml(docPath(d) || d.source_id || '-')}</div>
            <div class="doc-graph-meta">${formatDateMs(d.processed_time || d.updated_at || d.created_at)}</div>
          </div>
        </div>
      `;
    }).join('');
    list.querySelectorAll('.doc-graph-check').forEach(cb => {
      cb.addEventListener('click', (e) => {
        e.stopPropagation();
        const id = cb.getAttribute('data-doc-id');
        if (cb.checked) selectedDocVersions.add(id);
        else selectedDocVersions.delete(id);
        updateDocsList();
        loadSelectedGraph();
      });
    });
    list.querySelectorAll('.doc-graph-item').forEach(item => {
      item.addEventListener('click', () => {
        const id = item.getAttribute('data-doc-id');
        if (!selectedDocVersions.has(id)) selectedDocVersions.add(id);
        updateDocsList();
        loadSelectedGraph();
      });
    });
  }

  async function loadSelectedGraph() {
    if (!selectedDocVersions.size) {
      clearGraphCanvas();
      return;
    }
    const selected = [...selectedDocVersions];

    if (!graphData || !graphModel) {
      cancelGraphGrowth();
      stopPlayback();
      playbackStep = -1;
      focusFamilyId = null;
      loadedDocVersions = new Set();
      growthOutlinesByDoc = new Map();
      growthOutline = null;
      growthTotals = null;
      growthLoaded = { episodes: 0, concepts: 0, relations: 0, edges: 0 };
      const exitFocus = document.getElementById('graph-exit-focus');
      if (exitFocus) exitFocus.style.display = 'none';
      const selectedDocRows = docs.filter(doc => selectedDocVersions.has(doc.document_version_id));
      graphData = normalizeGraphPayload({ documents: selectedDocRows, episodes: [], concepts: [], edges: [], versions: {}, counts: {} });
      graphModel = buildGraphModel(graphData);
      updateGraphTitle();
      drawGraph(-1, { progressive: true, skeleton: true });
    }

    const removed = [...loadedDocVersions].filter(id => !selectedDocVersions.has(id));
    if (removed.length) {
      rebuildSelectedGraph();
      return;
    }

    const activeOrLoaded = new Set([...loadedDocVersions, ...growthControllers.keys()]);
    const additions = selected.filter(id => !activeOrLoaded.has(id));
    if (!additions.length) {
      updateGraphTitle();
      updateSummary(graphModel ? visibleModelForStep(-1) : null);
      return;
    }
    updateGraphTitle();
    appendDocuments(additions);
  }

  async function rebuildSelectedGraph() {
    cancelGraphGrowth();
    const keepSelected = [...selectedDocVersions];
    graphData = null;
    graphModel = null;
    loadedDocVersions = new Set();
    growthOutlinesByDoc = new Map();
    growthOutline = null;
    growthTotals = null;
    growthLoaded = { episodes: 0, concepts: 0, relations: 0, edges: 0 };
    if (!keepSelected.length) {
      clearGraphCanvas();
      return;
    }
    const selectedDocRows = docs.filter(doc => selectedDocVersions.has(doc.document_version_id));
    graphData = normalizeGraphPayload({ documents: selectedDocRows, episodes: [], concepts: [], edges: [], versions: {}, counts: {} });
    graphModel = buildGraphModel(graphData);
    updateGraphTitle();
    drawGraph(-1, { progressive: true, skeleton: true });
    appendDocuments(keepSelected);
  }

  function appendDocuments(documentVersionIds) {
    documentVersionIds.forEach(id => appendDocumentGraph(id));
  }

  async function appendDocumentGraph(documentVersionId) {
    if (!documentVersionId || loadedDocVersions.has(documentVersionId) || growthControllers.has(documentVersionId)) return;
    const loadingController = { key: documentVersionId, cancelled: false, loading: true };
    growthControllers.set(documentVersionId, loadingController);
    refreshPrimaryGrowthController();
    updatePlaybackControls();
    const summary = document.getElementById('graph-summary');
    if (summary) summary.textContent = '加载文档骨架中...';
    try {
      const res = await state.api.documentGraphOutline(state.currentGraphId, {
        documentVersionIds: [documentVersionId],
        maxEpisodes: 10000,
      });
      if (loadingController.cancelled || growthControllers.get(documentVersionId) !== loadingController) return;
      const outline = normalizeGraphPayload(res.data || {});
      loadedDocVersions.add(documentVersionId);
      growthOutlinesByDoc.set(documentVersionId, outline);
      growthOutline = mergeGraphPayload(growthOutline || {}, outline);
      growthTotals = summarizeGrowthTotals();
      const docOnly = initialDocumentOnlyGraph(outline);
      graphData = mergeGraphPayload(graphData, docOnly);
      graphData.counts = growthTotals || {};
      graphModel = buildGraphModel(graphData);
      updateGraphTitle();
      if (network && nodesDataSet && edgesDataSet) {
        updateGraphStep(-1, { fitDelay: 80, fitDuration: 300, progressive: true });
      } else {
        drawGraph(-1, { progressive: true, skeleton: true });
      }
      updatePlaybackControls();
      startGraphGrowth({
        key: documentVersionId,
        documentVersionIds: [documentVersionId],
        cursor: outline.next_cursor ?? 0,
        limit: 1,
      });
    } catch (err) {
      if (growthControllers.get(documentVersionId) === loadingController) {
        growthControllers.delete(documentVersionId);
        refreshPrimaryGrowthController();
      }
      const summaryEl = document.getElementById('graph-summary');
      if (summaryEl) summaryEl.textContent = `加载文档失败: ${err.message}`;
    }
  }

  async function loadSelectedGraphFullFallback() {
    const res = await state.api.documentGraph(state.currentGraphId, {
        documentVersionIds: [...selectedDocVersions],
        includeRelations: true,
        includeVersions: true,
      });
    graphData = normalizeGraphPayload(res.data || {});
    graphModel = buildGraphModel(graphData);
    updateGraphTitle();
    drawGraph(-1);
    updatePlaybackControls();
  }

  function clearGraphCanvas(message) {
    cancelGraphGrowth();
    stopPlayback();
    destroyNetwork();
    graphData = null;
    graphModel = null;
    growthLoaded = { episodes: 0, concepts: 0, relations: 0, edges: 0 };
    growthTotals = null;
    growthOutline = null;
    loadedDocVersions = new Set();
    growthOutlinesByDoc = new Map();
    edgeMetaById = new Map();
    nodeMetaById = new Map();
    const canvas = document.getElementById('document-graph-canvas');
    if (canvas) {
      canvas.innerHTML = `<div id="graph-empty" class="empty-state graph-empty"><i data-lucide="git-fork"></i><p>${message || '选择左侧 Markdown 文档后显示文档、Episode、实体和关系边。'}</p></div>`;
    }
    const summary = document.getElementById('graph-summary');
    if (summary) summary.textContent = selectedDocVersions.size ? '未加载图谱' : '选择一个或多个文档';
    updateGraphTitle();
    const playback = document.getElementById('graph-playback');
    if (playback) playback.style.display = 'none';
    const exitFocus = document.getElementById('graph-exit-focus');
    if (exitFocus) exitFocus.style.display = 'none';
    const detail = document.getElementById('graph-detail');
    if (detail) detail.innerHTML = emptyState('暂无选中节点');
    if (window.lucide) lucide.createIcons();
  }

  function destroyNetwork() {
    clearTimeout(relationStreamTimer);
    relationStreamTimer = null;
    if (network) {
      network.destroy();
      network = null;
    }
    nodesDataSet = null;
    edgesDataSet = null;
    hoverPanel = null;
    window.__documentGraphNetwork = null;
    window.__documentGraphData = null;
    clearTimeout(physicsFreezeTimer);
    physicsFreezeTimer = null;
    clearTimeout(naturalFitTimer);
    naturalFitTimer = null;
    lastNaturalFitAt = 0;
  }

  function cancelGraphGrowth() {
    growthRunId += 1;
    growthControllers.forEach(controller => { controller.cancelled = true; });
    growthControllers = new Map();
    if (growthController) growthController.cancelled = true;
    growthController = null;
    growthPauseRequested = false;
  }

  function hasActiveGrowth() {
    return growthControllers.size > 0;
  }

  function refreshPrimaryGrowthController() {
    growthController = growthControllers.values().next().value || null;
  }

  function summarizeGrowthTotals() {
    const totals = { episodes: 0, concepts: 0, relations: 0 };
    growthOutlinesByDoc.forEach(outline => {
      const counts = outline.counts || outline.totals || {};
      totals.episodes += Number(counts.episodes || outline.episodes?.length || 0);
      totals.concepts += Number(counts.concepts || 0);
      totals.relations += Number(counts.relations || 0);
    });
    return totals;
  }

  function normalizeGraphPayload(data) {
    return {
      documents: data.documents || [],
      episodes: data.episodes || [],
      concepts: data.concepts || [],
      edges: data.edges || [],
      versions: data.versions || {},
      counts: data.counts || data.totals || {},
      episode_counts: data.episode_counts || {},
      cursor: data.cursor ?? 0,
      next_cursor: data.next_cursor ?? null,
    };
  }

  function initialDocumentOnlyGraph(outline) {
    const normalized = normalizeGraphPayload(outline || {});
    return {
      ...normalized,
      episodes: [],
      concepts: [],
      edges: (normalized.edges || []).filter(e => e.edge_type === 'DOCUMENT_LINK'),
      versions: {},
      cursor: 0,
      next_cursor: normalized.next_cursor,
    };
  }

  function chooseChunkLimit(episodeCount) {
    if (episodeCount > 160) return 8;
    if (episodeCount > 80) return 10;
    return 14;
  }

  function mergeGraphChunk(chunk) {
    graphData = mergeGraphPayload(graphData, chunk);
    growthLoaded = {
      episodes: graphData.episodes.length,
      concepts: graphData.concepts.filter(c => c.role !== 'relation').length,
      relations: graphData.concepts.filter(c => c.role === 'relation').length,
      edges: graphData.edges.length,
    };
  }

  function mergeGraphPayload(base, chunk) {
    const current = normalizeGraphPayload(base || {});
    const next = normalizeGraphPayload(chunk || {});
    const byDoc = new Map((current.documents || []).map(item => [item.document_version_id, item]));
    next.documents.forEach(item => byDoc.set(item.document_version_id, item));
    const byEpisode = new Map((current.episodes || []).map(item => [item.version_id, item]));
    next.episodes.forEach(item => byEpisode.set(item.version_id, item));
    const byConcept = new Map((current.concepts || []).map(item => [item.family_id, item]));
    next.concepts.forEach(item => byConcept.set(item.family_id, item));
    const byEdge = new Map((current.edges || []).map(item => [item.edge_id || item.id, item]));
    next.edges.forEach(item => byEdge.set(item.edge_id || item.id, item));

    return {
      ...current,
      documents: [...byDoc.values()],
      episodes: [...byEpisode.values()],
      concepts: [...byConcept.values()],
      edges: [...byEdge.values()],
      versions: { ...(current.versions || {}), ...(next.versions || {}) },
      counts: current.counts || next.counts || next.totals || {},
      episode_counts: { ...(current.episode_counts || {}), ...(next.episode_counts || {}) },
      cursor: next.cursor,
      next_cursor: next.next_cursor,
    };
  }

  async function startGraphGrowth({ key, documentVersionIds, cursor, limit }) {
    const runId = ++growthRunId;
    const controllerKey = key || documentVersionIds.join('|');
    const controller = { key: controllerKey, runId, cancelled: false };
    growthControllers.set(controllerKey, controller);
    growthController = controller;
    updatePlaybackControls();
    let nextCursor = cursor;
    if (nextCursor === null || nextCursor === undefined) {
      growthControllers.delete(controllerKey);
      refreshPrimaryGrowthController();
      updateSummary(graphModel ? visibleModelForStep(-1) : null);
      return;
    }

    try {
      while (!controller.cancelled && growthControllers.get(controllerKey) === controller && nextCursor !== null) {
        while (growthPauseRequested && !controller.cancelled) {
          await sleep(120);
        }
        if (controller.cancelled || growthControllers.get(controllerKey) !== controller) return;
        const chunk = await state.api.documentGraphChunk(state.currentGraphId, {
          documentVersionIds,
          cursor: nextCursor,
          limit,
          includeRelations: true,
          includeVersions: true,
        });
        if (controller.cancelled || growthControllers.get(controllerKey) !== controller) return;
        const data = chunk.data || {};
        await animateEpisodeChunk(data, controller);
        nextCursor = data.next_cursor;
        await sleep(120);
      }

      if (growthControllers.get(controllerKey) === controller) {
        growthControllers.delete(controllerKey);
        refreshPrimaryGrowthController();
        updatePlaybackControls();
        updateSummary(graphModel ? visibleModelForStep(-1) : null);
        if (!hasActiveGrowth()) schedulePhysicsFreeze(graphData.concepts?.length > 2500 ? 6500 : 3800);
      }
    } catch (err) {
      if (controller.cancelled || growthControllers.get(controllerKey) !== controller) return;
      growthControllers.delete(controllerKey);
      refreshPrimaryGrowthController();
      const summary = document.getElementById('graph-summary');
      if (summary) summary.textContent = `增量加载失败，尝试全量加载: ${err.message}`;
      try {
        await loadSelectedGraphFullFallback();
      } catch (fallbackErr) {
        clearGraphCanvas(`加载子图失败: ${escapeHtml(fallbackErr.message)}`);
      }
    }
  }

  function toggleGraphGrowthPause() {
    if (!hasActiveGrowth()) return false;
    growthPauseRequested = !growthPauseRequested;
    updatePlaybackControls();
    updateSummary(graphModel ? visibleModelForStep(-1) : null);
    return true;
  }

  async function animateEpisodeChunk(chunk, controller) {
    const next = normalizeGraphPayload(chunk || {});
    const episode = next.episodes?.[0];
    if (!episode || !network || !nodesDataSet || !edgesDataSet) {
      mergeGraphChunk(next);
      graphModel = buildGraphModel(graphData);
      updateGraphStep(-1, { fitDelay: 90, fitDuration: 260, freezeDelay: 5200, progressive: true });
      return;
    }

    const hasEpisodeEdge = (growthOutline?.edges || [])
      .find(edge => edge.edge_type === 'HAS_EPISODE' && edge.target_version_id === episode.version_id);
    if (hasEpisodeEdge && !next.edges.some(edge => (edge.edge_id || edge.id) === (hasEpisodeEdge.edge_id || hasEpisodeEdge.id))) {
      next.edges.push(hasEpisodeEdge);
    }
    const targetData = mergeGraphPayload(graphData, next);
    const previousData = graphData;
    const previousModel = graphModel;
    graphData = targetData;
    graphModel = buildGraphModel(targetData);
    const targetVisible = visibleModelForStep(-1);
    const targetVis = buildVisData(targetVisible);
    graphData = previousData;
    graphModel = previousModel;

    const nodeById = new Map(targetVis.nodes.map(node => [node.id, node]));
    const edgeById = new Map(targetVis.edges.map(edge => [edge.id, edge]));
    const episodeNodeId = `episode:${episode.version_id}`;
    const hasEpisodeEdgeId = hasEpisodeEdge?.edge_id || hasEpisodeEdge?.id;
    const conceptFamilyIds = next.concepts
      .filter(concept => concept.role !== 'relation')
      .map(concept => concept.family_id);
    const mentionEdges = next.edges
      .filter(edge => edge.edge_type === 'MENTIONS')
      .map(edge => edge.edge_id || edge.id);
    const relationEdges = targetVisible.relationEdges
      .filter(edge => edge.relation?.episode_version_id === episode.version_id)
      .map(edge => edge.id);

    graphData = targetData;
    graphModel = buildGraphModel(graphData);
    window.__documentGraphData = graphData;
    window.__documentGraphVisual = { graphModel, visible: targetVisible, playbackStep: -1 };

    await waitGrowth(controller);
    addVisualNode(nodeById.get(episodeNodeId));
    growthLoaded.episodes = Math.max(growthLoaded.episodes, graphData.episodes.length);
    updateSummary(targetVisible);
    network.setOptions({ physics: { enabled: true } });
    if (hasEpisodeEdgeId) addVisualEdge(edgeById.get(hasEpisodeEdgeId));
    await sleep(220);

    for (const familyId of conceptFamilyIds) {
      await waitGrowth(controller);
      const nodeId = `concept:${familyId}`;
      const added = addVisualNode(nodeById.get(nodeId));
      mentionEdges
        .map(id => edgeById.get(id))
        .filter(edge => edge && edge.to === nodeId)
        .forEach(edge => addVisualEdge(edge));
      if (added) growthLoaded.concepts += 1;
      updateSummary(targetVisible);
      await sleep(1000 / growthRatePerSecond);
    }

    for (const edgeId of relationEdges) {
      await waitGrowth(controller);
      const edge = edgeById.get(edgeId);
      if (!edge) continue;
      if (!nodesDataSet.get(edge.from) || !nodesDataSet.get(edge.to)) continue;
      if (addVisualEdge(edge)) growthLoaded.relations += 1;
      updateSummary(targetVisible);
      await sleep(1000 / growthRatePerSecond);
    }

    growthLoaded.edges = edgesDataSet.getIds().length;
    updateSummary(targetVisible);
    scheduleNaturalFit(180);
  }

  async function waitGrowth(controller) {
    while (growthPauseRequested && !controller.cancelled) {
      await sleep(120);
    }
    if (controller.cancelled || growthControllers.get(controller.key) !== controller) {
      throw new Error("graph growth cancelled");
    }
  }

  function addVisualNode(nodeWithMeta) {
    if (!nodeWithMeta || !nodesDataSet) return false;
    nodeMetaById.set(nodeWithMeta.id, nodeWithMeta._meta);
    if (nodesDataSet.get(nodeWithMeta.id)) return false;
    const { _meta, ...node } = nodeWithMeta;
    nodesDataSet.add(node);
    network?.setOptions({ physics: { enabled: true } });
    scheduleNaturalFit(120);
    return true;
  }

  function addVisualEdge(edgeWithMeta) {
    if (!edgeWithMeta || !edgesDataSet) return false;
    edgeMetaById.set(edgeWithMeta.id, edgeWithMeta._meta);
    if (edgesDataSet.get(edgeWithMeta.id)) return false;
    if (!nodesDataSet?.get(edgeWithMeta.from) || !nodesDataSet?.get(edgeWithMeta.to)) return false;
    const { _meta, ...edge } = edgeWithMeta;
    edgesDataSet.add(edge);
    network?.setOptions({ physics: { enabled: true } });
    scheduleNaturalFit(160);
    return true;
  }

  function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  function buildGraphModel(data) {
    const model = {
      documents: data.documents || [],
      episodes: (data.episodes || []).slice().sort((a, b) => {
        const da = String(a.document_version_id || '').localeCompare(String(b.document_version_id || ''));
        if (da !== 0) return da;
        return (a.chunk_index ?? 0) - (b.chunk_index ?? 0) || String(a.processed_time || '').localeCompare(String(b.processed_time || ''));
      }),
      entities: [],
      relations: [],
      sourceEdges: [],
      relationEdges: [],
      nodesById: new Map(),
      relationByEdgeId: new Map(),
      conceptByFamily: new Map(),
      versions: data.versions || {},
    };

    (data.concepts || []).forEach(c => {
      model.conceptByFamily.set(c.family_id, c);
      if (c.role === 'relation') model.relations.push(c);
      else model.entities.push(c);
    });

    const existingNodeIds = new Set();
    model.documents.forEach(doc => existingNodeIds.add(`doc:${doc.document_version_id}`));
    model.episodes.forEach(ep => existingNodeIds.add(`episode:${ep.version_id}`));
    model.entities.forEach(c => existingNodeIds.add(`concept:${c.family_id}`));

    (data.edges || []).forEach(e => {
      if (!e.from || !e.to) return;
      if (e.edge_type === 'CONNECTS') return;
      if (isRelationRef(e.from, model) || isRelationRef(e.to, model)) return;
      if (!existingNodeIds.has(e.from) || !existingNodeIds.has(e.to)) return;
      model.sourceEdges.push({ ...e, id: e.edge_id || `${e.from}-${e.to}-${e.edge_type}` });
    });

    const groupedByPair = new Map();
    model.relations.forEach(rel => {
      const endpoints = relationEndpoints(rel, data);
      if (endpoints.length < 2) return;
      const from = `concept:${endpoints[0]}`;
      const to = `concept:${endpoints[1]}`;
      if (!existingNodeIds.has(from) || !existingNodeIds.has(to)) return;
      const key = [from, to].sort().join('|');
      if (!groupedByPair.has(key)) groupedByPair.set(key, []);
      groupedByPair.get(key).push({ rel, from, to, key });
    });
    groupedByPair.forEach(items => {
      items.forEach((item, idx) => {
        const total = items.length;
        const curveIndex = idx - (total - 1) / 2;
        const direction = idx % 2 === 0 ? 'curvedCW' : 'curvedCCW';
        const roundness = Math.min(0.55, 0.12 + Math.abs(curveIndex) * 0.14);
        const edge = {
          id: `relation:${item.rel.family_id}`,
          from: item.from,
          to: item.to,
          edge_kind: 'relation_concept',
          relation_family_id: item.rel.family_id,
          relation_version_id: item.rel.version_id,
          relation: item.rel,
          parallel_count: total,
          parallel_index: idx,
          smooth: { enabled: true, type: total === 1 ? 'continuous' : direction, roundness },
        };
        model.relationEdges.push(edge);
        model.relationByEdgeId.set(edge.id, item.rel);
      });
    });

    return model;
  }

  function isRelationRef(ref, model) {
    if (!ref || !String(ref).startsWith('concept:')) return false;
    const familyId = String(ref).slice('concept:'.length);
    return model.conceptByFamily.get(familyId)?.role === 'relation';
  }

  function relationEndpoints(rel, data) {
    const meta = rel.metadata || {};
    const endpoints = [];
    if (meta.entity1_family_id) endpoints.push(meta.entity1_family_id);
    if (meta.entity2_family_id && meta.entity2_family_id !== meta.entity1_family_id) endpoints.push(meta.entity2_family_id);
    if (endpoints.length >= 2) return endpoints.slice(0, 2);
    (data.edges || [])
      .filter(e => e.edge_type === 'CONNECTS' && (e.relation_family_id === rel.family_id || e.source_family_id === rel.family_id))
      .forEach(e => {
        const fid = e.target_family_id && e.target_family_id !== rel.family_id ? e.target_family_id : e.source_family_id;
        if (fid && fid !== rel.family_id && !endpoints.includes(fid)) endpoints.push(fid);
      });
    return endpoints.slice(0, 2);
  }

  function drawGraph(step, options = {}) {
    if (!graphModel || !graphModel.documents.length) {
      clearGraphCanvas('所选文档没有可展示的子图');
      return;
    }
    const canvas = document.getElementById('document-graph-canvas');
    if (!canvas) return;
    destroyNetwork();
    canvas.innerHTML = '';

    const visible = visibleModelForStep(step);
    const { nodes, edges } = buildVisData(visible);
    nodeMetaById = new Map(nodes.map(n => [n.id, n._meta]));
    edgeMetaById = new Map(edges.map(e => [e.id, e._meta]));
    const streamRelations = step < 0 && showRelations && !focusFamilyId && visible.relationEdges.length > 800;
    const initialEdges = streamRelations ? edges.filter(e => e._meta?.type !== 'relation') : edges;
    const delayedRelationEdges = streamRelations ? edges.filter(e => e._meta?.type === 'relation') : [];
    nodesDataSet = new vis.DataSet(nodes.map(({ _meta, ...node }) => node));
    edgesDataSet = new vis.DataSet(initialEdges.map(({ _meta, ...edge }) => edge));

    network = new vis.Network(canvas, { nodes: nodesDataSet, edges: edgesDataSet }, networkOptions(nodes.length, edges.length, options));
    window.__documentGraphNetwork = network;
    window.__documentGraphData = graphData;
    window.__documentGraphVisual = { graphModel, visible, playbackStep: step };

    bindNetworkEvents(canvas);
    updateSummary(visible);
    updatePlaybackControls();

    network.once('stabilizationIterationsDone', () => {
      network?.fit({ animation: { duration: 700, easingFunction: 'easeInOutQuad' } });
      schedulePhysicsFreeze(1200);
    });
    setTimeout(() => network?.fit({ animation: { duration: 520, easingFunction: 'easeInOutQuad' } }), 600);
    setTimeout(() => network?.fit({ animation: { duration: 700, easingFunction: 'easeInOutQuad' } }), 2200);
    if (streamRelations) {
      startRelationEdgeStream(delayedRelationEdges);
    } else {
      schedulePhysicsFreeze(nodes.length > 2500 ? 9000 : 5000);
    }
  }

  function updateGraphStep(step, options = {}) {
    if (!graphModel || !graphModel.documents.length) {
      clearGraphCanvas('所选文档没有可展示的子图');
      return;
    }
    if (!network || !nodesDataSet || !edgesDataSet) {
      drawGraph(step);
      return;
    }

    const visible = visibleModelForStep(step);
    const { nodes, edges } = buildVisData(visible);
    nodeMetaById = new Map(nodes.map(n => [n.id, n._meta]));
    edgeMetaById = new Map(edges.map(e => [e.id, e._meta]));

    const currentNodeIds = nodesDataSet.getIds();
    const currentPositions = network.getPositions(currentNodeIds);
    syncDataSet(nodesDataSet, nodes, { preservePosition: true, positions: currentPositions });
    syncDataSet(edgesDataSet, edges);

    window.__documentGraphData = graphData;
    window.__documentGraphVisual = { graphModel, visible, playbackStep: step };
    updateSummary(visible);
    updatePlaybackControls();

    network.setOptions({ physics: { enabled: true } });
    clearTimeout(updateGraphStep._freezeTimer);
    updateGraphStep._freezeTimer = setTimeout(() => {
      if (!network) return;
      updateGraphStep._progressiveFitCounter = (updateGraphStep._progressiveFitCounter || 0) + 1;
      const shouldFit = !options.progressive || updateGraphStep._progressiveFitCounter % 3 === 0 || !hasActiveGrowth();
      if (shouldFit) {
        network.fit({
          animation: { duration: options.fitDuration || 420, easingFunction: 'easeInOutQuad' },
        });
      }
      schedulePhysicsFreeze(options.freezeDelay || 1800);
    }, options.fitDelay || 260);
  }

  function syncDataSet(dataSet, items, options = {}) {
    const nextIds = new Set(items.map(item => item.id));
    const existingIds = new Set(dataSet.getIds());
    const add = [];
    const update = [];
    items.forEach(({ _meta, ...item }) => {
      if (!existingIds.has(item.id)) {
        add.push(item);
        return;
      }
      if (options.preservePosition) {
        const current = dataSet.get(item.id) || {};
        const pos = options.positions?.[item.id] || {};
        update.push({
          ...item,
          x: pos.x ?? current.x,
          y: pos.y ?? current.y,
          fixed: current.fixed,
        });
      } else {
        update.push(item);
      }
    });
    const remove = [...existingIds].filter(id => !nextIds.has(id));
    if (remove.length) dataSet.remove(remove);
    if (add.length) dataSet.add(add);
    if (update.length) dataSet.update(update);
  }

  function visibleModelForStep(step) {
    const all = {
      documents: graphModel.documents,
      episodes: graphModel.episodes,
      entities: graphModel.entities,
      sourceEdges: graphModel.sourceEdges,
      relationEdges: showRelations ? graphModel.relationEdges : [],
    };
    if (step < 0) return focusFamilyId ? focusedVisibleModel(all, focusFamilyId) : all;

    const episodeIds = new Set(graphModel.episodes.slice(0, step).map(e => e.version_id));
    const visibleDocIds = new Set(graphModel.documents.map(d => d.document_version_id));
    const visibleEpisodeNodeIds = new Set([...episodeIds].map(id => `episode:${id}`));
    const visibleConceptFamilies = new Set();

    graphModel.sourceEdges.forEach(e => {
      if (e.edge_type === 'MENTIONS' && episodeIds.has(e.episode_version_id) && e.target_family_id) {
        const concept = graphModel.conceptByFamily.get(e.target_family_id);
        if (concept && concept.role !== 'relation') visibleConceptFamilies.add(e.target_family_id);
      }
    });
    graphModel.relationEdges.forEach(e => {
      const rel = e.relation;
      if (rel && episodeIds.has(rel.episode_version_id)) {
        const fromFamily = String(e.from).slice('concept:'.length);
        const toFamily = String(e.to).slice('concept:'.length);
        if (visibleConceptFamilies.has(fromFamily) && visibleConceptFamilies.has(toFamily)) {
          visibleConceptFamilies.add(fromFamily);
          visibleConceptFamilies.add(toFamily);
        }
      }
    });

    const visibleNodeIds = new Set([
      ...[...visibleDocIds].map(id => `doc:${id}`),
      ...visibleEpisodeNodeIds,
      ...[...visibleConceptFamilies].map(fid => `concept:${fid}`),
    ]);

    const visible = {
      documents: graphModel.documents,
      episodes: graphModel.episodes.filter(e => episodeIds.has(e.version_id)),
      entities: graphModel.entities.filter(c => visibleConceptFamilies.has(c.family_id)),
      sourceEdges: graphModel.sourceEdges.filter(e => visibleNodeIds.has(e.from) && visibleNodeIds.has(e.to)),
      relationEdges: showRelations ? graphModel.relationEdges.filter(e => {
        const rel = e.relation;
        return rel && episodeIds.has(rel.episode_version_id) && visibleNodeIds.has(e.from) && visibleNodeIds.has(e.to);
      }) : [],
    };
    return focusFamilyId ? focusedVisibleModel(visible, focusFamilyId) : visible;
  }

  function focusedVisibleModel(base, familyId) {
    if (!familyId) return base;
    const conceptId = `concept:${familyId}`;
    const nodeIds = new Set([conceptId]);
    const focusedEdges = [];
    base.sourceEdges.forEach(edge => {
      if (edge.from === conceptId || edge.to === conceptId || edge.target_family_id === familyId || edge.source_family_id === familyId) {
        focusedEdges.push(edge);
        if (edge.from) nodeIds.add(edge.from);
        if (edge.to) nodeIds.add(edge.to);
      }
    });
    const focusedRelations = [];
    base.relationEdges.forEach(edge => {
      if (edge.from === conceptId || edge.to === conceptId || edge.relation_family_id === familyId) {
        focusedRelations.push(edge);
        nodeIds.add(edge.from);
        nodeIds.add(edge.to);
      }
    });
    focusedEdges.forEach(edge => {
      if (String(edge.from || '').startsWith('episode:')) {
        const epId = String(edge.from).slice('episode:'.length);
        const ep = graphModel.episodes.find(item => item.version_id === epId);
        if (ep?.document_version_id) nodeIds.add(`doc:${ep.document_version_id}`);
      }
    });
    const docs = base.documents.filter(doc => nodeIds.has(`doc:${doc.document_version_id}`));
    const episodes = base.episodes.filter(ep => nodeIds.has(`episode:${ep.version_id}`));
    const entities = base.entities.filter(c => nodeIds.has(`concept:${c.family_id}`));
    return { documents: docs, episodes, entities, sourceEdges: focusedEdges, relationEdges: focusedRelations };
  }

  function ringPosition(idx, total, cx, cy, baseRadius, ringGap, minArc, phase) {
    let remaining = idx;
    let ring = 0;
    while (true) {
      const radius = baseRadius + ring * ringGap;
      const cap = Math.max(8, Math.floor((Math.PI * 2 * radius) / minArc));
      if (remaining < cap) {
        const angle = (remaining / cap) * Math.PI * 2 - Math.PI / 2 + ring * 0.31 + (phase || 0);
        const squash = total > 80 ? 0.78 : 0.86;
        return { x: cx + Math.cos(angle) * radius, y: cy + Math.sin(angle) * radius * squash };
      }
      remaining -= cap;
      ring += 1;
    }
  }

  function spiralOffset(idx, baseRadius, gap, scale) {
    const angle = idx * 2.399963229728653;
    const radius = baseRadius + Math.sqrt(idx) * gap * (scale || 1);
    return { x: Math.cos(angle) * radius, y: Math.sin(angle) * radius * 0.82 };
  }

  function averagePositions(points) {
    if (!points.length) return null;
    const sum = points.reduce((acc, p) => ({ x: acc.x + p.x, y: acc.y + p.y }), { x: 0, y: 0 });
    return { x: sum.x / points.length, y: sum.y / points.length };
  }

  function fallbackEntityAnchor(idx, total) {
    const radius = Math.max(480, Math.min(1800, 260 + Math.sqrt(Math.max(1, total)) * 28));
    const angle = idx * 2.399963229728653;
    return { x: Math.cos(angle) * radius, y: 120 + Math.sin(angle) * radius * 0.82 };
  }

  function entityCirclePosition(idx, total) {
    return ringPosition(idx, total, 0, 130, 760, 92, 54, 0);
  }

  function computeRelationComponents(relationEdges) {
    const parent = new Map();
    const families = new Set();
    const familyFromNode = id => String(id || '').startsWith('concept:') ? String(id).slice('concept:'.length) : '';
    const find = fid => {
      if (!parent.has(fid)) parent.set(fid, fid);
      const p = parent.get(fid);
      if (p === fid) return fid;
      const root = find(p);
      parent.set(fid, root);
      return root;
    };
    const union = (a, b) => {
      if (!a || !b || a === b) return;
      const ra = find(a);
      const rb = find(b);
      if (ra !== rb) parent.set(rb, ra);
    };
    (relationEdges || []).forEach(edge => {
      const a = familyFromNode(edge.from);
      const b = familyFromNode(edge.to);
      if (!a || !b) return;
      families.add(a);
      families.add(b);
      union(a, b);
    });

    const groups = new Map();
    families.forEach(fid => {
      const root = find(fid);
      if (!groups.has(root)) groups.set(root, []);
      groups.get(root).push(fid);
    });
    const sortedGroups = [...groups.values()].sort((a, b) => b.length - a.length);
    const offsetByFamily = new Map();
    const anchorByFamily = new Map();
    sortedGroups.forEach((group, groupIdx) => {
      const groupAnchor = entityCirclePosition(groupIdx, Math.max(sortedGroups.length, 1));
      group.forEach((fid, idx) => {
        const local = spiralOffset(idx, 0, 18, Math.max(0.6, Math.min(1.2, group.length / 18)));
        anchorByFamily.set(fid, groupAnchor);
        offsetByFamily.set(fid, { x: local.x * 0.45, y: local.y * 0.45 });
      });
    });
    return { anchorByFamily, offsetByFamily };
  }

  function buildVisData(visible) {
    const relationCount = {};
    visible.relationEdges.forEach(e => {
      relationCount[e.from] = (relationCount[e.from] || 0) + 1;
      relationCount[e.to] = (relationCount[e.to] || 0) + 1;
    });
    visible.sourceEdges.forEach(e => {
      if (e.edge_type === 'MENTIONS') {
        relationCount[e.to] = (relationCount[e.to] || 0) + 0.35;
      }
    });

    const nodes = [];
    const docs = visible.documents;
    const docSpacing = docs.length > 1 ? Math.max(720, Math.min(1100, 4800 / docs.length)) : 0;
    const startX = -((docs.length - 1) * docSpacing) / 2;
    const docX = new Map();
    const docY = -260;
    docs.forEach((doc, idx) => {
      const id = `doc:${doc.document_version_id}`;
      const x = startX + idx * docSpacing;
      docX.set(doc.document_version_id, x);
      nodes.push(makeBubbleNode({
        id,
        role: 'document',
        label: docTitle(doc),
        size: 36,
        x,
        y: docY,
        fixed: true,
        meta: { type: 'document', item: doc },
        title: docPath(doc),
      }));
    });

    const epsByDoc = {};
    visible.episodes.forEach(ep => {
      if (!epsByDoc[ep.document_version_id]) epsByDoc[ep.document_version_id] = [];
      epsByDoc[ep.document_version_id].push(ep);
    });
    const episodePos = new Map();
    Object.entries(epsByDoc).forEach(([docVersionId, eps]) => {
      const cx = docX.get(docVersionId) || 0;
      eps.forEach((ep, idx) => {
        const pos = ringPosition(idx, eps.length, cx, docY + 260, 210, 74, 64, idx * 0.003);
        episodePos.set(ep.version_id, pos);
        nodes.push(makeBubbleNode({
          id: `episode:${ep.version_id}`,
          role: 'episode',
          label: ep.heading_path || ep.name || `Episode ${idx + 1}`,
          size: 18,
          x: pos.x,
          y: pos.y,
          fixed: true,
          meta: { type: 'episode', item: ep },
          title: normalizeText(ep.content, 180),
        }));
      });
    });

    const entityEpisodes = new Map();
    visible.sourceEdges.forEach(e => {
      if (e.edge_type !== 'MENTIONS' || !e.target_family_id || !e.episode_version_id) return;
      if (!entityEpisodes.has(e.target_family_id)) entityEpisodes.set(e.target_family_id, []);
      const list = entityEpisodes.get(e.target_family_id);
      if (!list.includes(e.episode_version_id)) list.push(e.episode_version_id);
    });
    const entitiesByEpisode = new Map();
    visible.entities.forEach(c => {
      const epIds = entityEpisodes.get(c.family_id) || [];
      const primary = epIds.find(id => episodePos.has(id)) || '__unplaced__';
      if (!entitiesByEpisode.has(primary)) entitiesByEpisode.set(primary, []);
      entitiesByEpisode.get(primary).push(c);
    });
    const entityLocalIndex = new Map();
    entitiesByEpisode.forEach(group => {
      group
        .slice()
        .sort((a, b) => (relationCount[`concept:${b.family_id}`] || 0) - (relationCount[`concept:${a.family_id}`] || 0))
        .forEach((c, idx) => entityLocalIndex.set(c.family_id, idx));
    });
    const relationComponents = computeRelationComponents(visible.relationEdges);

    visible.entities.forEach((c, idx) => {
      const id = `concept:${c.family_id}`;
      const size = Math.max(14, Math.min(32, Math.round(14 + Math.sqrt(relationCount[id] || 1) * 4)));
      const versionTotal = graphModel.versions?.[c.family_id]?.total || 1;
      const epIds = entityEpisodes.get(c.family_id) || [];
      const attached = epIds.map(epId => episodePos.get(epId)).filter(Boolean);
      const anchored = !!attached.length;
      const componentAnchor = relationComponents.anchorByFamily.get(c.family_id);
      const anchor = anchored ? averagePositions(attached) : (componentAnchor || entityCirclePosition(idx, visible.entities.length));
      const localIdx = entityLocalIndex.get(c.family_id) ?? idx;
      const spread = anchored ? spiralOffset(localIdx, 82, 23, relationCount[id] > 5 ? 0.8 : 1) : { x: 0, y: 0 };
      const componentPull = relationComponents.offsetByFamily.get(c.family_id) || { x: 0, y: 0 };
      nodes.push(makeBubbleNode({
        id,
        role: 'entity',
        label: conceptTitle(c),
        size,
        x: anchor.x + spread.x + componentPull.x,
        y: anchor.y + spread.y + componentPull.y,
        versionTotal,
        meta: { type: 'entity', item: c },
        title: normalizeText(c.content || c.summary, 180),
      }));
    });

    const edges = [];
    visible.sourceEdges.forEach(e => {
      const style = sourceEdgeStyle(e.edge_type);
      edges.push({
        id: e.id,
        from: e.from,
        to: e.to,
        label: showLabels && e.edge_type === 'HAS_EPISODE' ? '' : '',
        color: style.color,
        dashes: style.dashes,
        width: style.width,
        arrows: { to: { enabled: false } },
        smooth: { enabled: true, type: 'continuous', roundness: 0.12 },
        _meta: { type: 'source_edge', item: e },
      });
    });
    visible.relationEdges.forEach(e => {
      const rel = e.relation;
      const title = conceptTitle(rel);
      edges.push({
        id: e.id,
        from: e.from,
        to: e.to,
        label: showLabels ? truncate(title, 34) : '',
        color: {
          color: ROLE_COLORS.relation.bg,
          highlight: ROLE_COLORS.relation.border,
          hover: ROLE_COLORS.relation.border,
        },
        width: e.parallel_count > 1 ? 1.8 : 2.2,
        dashes: false,
        arrows: { to: { enabled: false } },
        smooth: e.smooth,
        font: {
          size: 10,
          color: mutedColor(),
          strokeWidth: 3,
          strokeColor: isLightTheme() ? '#ffffff' : '#0f172a',
          align: 'middle',
        },
        shadow: {
          enabled: true,
          color: ROLE_COLORS.relation.glow,
          size: 6,
          x: 0,
          y: 0,
        },
        _meta: { type: 'relation', item: rel, edge: e },
      });
    });
    return { nodes, edges };
  }

  function makeBubbleNode({ id, role, label, size, x, y, fixed, versionTotal, meta, title }) {
    const colors = ROLE_COLORS[role] || ROLE_COLORS.entity;
    const multi = versionTotal > 1;
    const saved = pinnedPositions[id];
    return {
      id,
      label: showLabels ? truncate(label || id, role === 'document' ? 28 : 24) + (multi ? ` [v${versionTotal}]` : '') : '',
      shape: 'dot',
      size,
      x: saved ? saved.x : x,
      y: saved ? saved.y : y,
      fixed: saved ? { x: true, y: true } : (fixed ? { x: true, y: true } : false),
      color: {
        background: colors.bg,
        border: multi ? '#fbbf24' : colors.border,
        highlight: { background: colors.bg, border: multi ? '#fde68a' : colors.border },
        hover: { background: colors.bg, border: multi ? '#fde68a' : colors.border },
      },
      borderWidth: multi ? 3 : (role === 'document' ? 2.5 : 1.5),
      borderWidthSelected: 4,
      shadow: {
        enabled: true,
        color: multi ? 'rgba(251,191,36,0.45)' : colors.glow,
        size: multi ? Math.min(18, 7 + versionTotal) : 8,
        x: 0,
        y: 0,
      },
      font: {
        color: role === 'document' ? '#111827' : labelColor(),
        size: role === 'document' ? 14 : 11,
        face: 'Inter, sans-serif',
        strokeWidth: role === 'document' ? 4 : 3,
        strokeColor: role === 'document' ? '#fdf4ff' : (isLightTheme() ? '#ffffff' : '#0f172a'),
        background: role === 'document' ? 'rgba(253,244,255,0.72)' : undefined,
      },
      _meta: { ...meta, role, title, versionTotal: versionTotal || 1 },
    };
  }

  function sourceEdgeStyle(type) {
    if (type === 'HAS_EPISODE') return { color: { color: '#64748b', highlight: '#cbd5e1', hover: '#94a3b8' }, width: 1.4, dashes: false };
    if (type === 'MENTIONS' && !showSourceEdges) {
      return {
        color: { color: 'rgba(56,189,248,0.035)', highlight: '#bae6fd', hover: '#7dd3fc' },
        width: 0.25,
        dashes: false,
      };
    }
    if (type === 'MENTIONS') return { color: { color: 'rgba(56,189,248,0.42)', highlight: '#bae6fd', hover: '#7dd3fc' }, width: 0.85, dashes: [3, 5] };
    return { color: { color: '#475569', highlight: '#94a3b8', hover: '#94a3b8' }, width: 1, dashes: true };
  }

  function schedulePhysicsFreeze(delayMs) {
    clearTimeout(physicsFreezeTimer);
    physicsFreezeTimer = setTimeout(() => {
      if (!network) return;
      network.setOptions({ physics: { enabled: true } });
      network.fit({ animation: { duration: 520, easingFunction: 'easeInOutQuad' } });
    }, delayMs);
  }

  function scheduleNaturalFit(delayMs = 120) {
    if (!network) return;
    if (naturalFitTimer) return;
    const now = Date.now();
    const minGap = hasActiveGrowth() ? 850 : 1300;
    const waitForGap = Math.max(0, minGap - (now - lastNaturalFitAt));
    const wait = Math.max(delayMs, waitForGap);
    naturalFitTimer = setTimeout(() => {
      naturalFitTimer = null;
      if (!network) return;
      lastNaturalFitAt = Date.now();
      network.setOptions({ physics: { enabled: true } });
      network.fit({ animation: { duration: hasActiveGrowth() ? 380 : 520, easingFunction: 'easeInOutQuad' } });
    }, wait);
  }

  function startRelationEdgeStream(relationEdges) {
    clearTimeout(relationStreamTimer);
    if (!relationEdges.length || !edgesDataSet) {
      schedulePhysicsFreeze(5000);
      return;
    }
    const queue = relationEdges.slice();
    const total = queue.length;
    let added = 0;
    network?.setOptions({ physics: { enabled: true } });

    const tick = () => {
      if (!network || !edgesDataSet) return;
      const batchSize = total > 6000 ? 260 : total > 2500 ? 180 : 100;
      const batch = queue.splice(0, batchSize).map(({ _meta, ...edge }) => edge);
      if (batch.length) {
        edgesDataSet.add(batch);
        added += batch.length;
        if (added === batch.length || added % 1200 < batchSize) {
          network.fit({ animation: { duration: 260, easingFunction: 'easeInOutQuad' } });
        }
      }
      if (queue.length) {
        relationStreamTimer = setTimeout(tick, 70);
      } else {
        relationStreamTimer = null;
        network.fit({ animation: { duration: 560, easingFunction: 'easeInOutQuad' } });
        schedulePhysicsFreeze(total > 5000 ? 9000 : 6000);
      }
    };
    relationStreamTimer = setTimeout(tick, 260);
  }

  function networkOptions(nodeCount, edgeCount, options = {}) {
    const basePhysics = window.GraphUtils?.getPhysicsOptions ? window.GraphUtils.getPhysicsOptions() : {
      enabled: true,
      solver: 'forceAtlas2Based',
      forceAtlas2Based: { gravitationalConstant: -130, centralGravity: 0.006, springLength: 150, springConstant: 0.04, damping: 0.58, avoidOverlap: 0.75 },
      stabilization: { enabled: true, iterations: 300, updateInterval: 25 },
    };
    const large = nodeCount > 2500 || options.progressive;
    const huge = nodeCount > 7000;
    const physics = {
      ...basePhysics,
      enabled: true,
      solver: 'forceAtlas2Based',
      forceAtlas2Based: {
        ...(basePhysics.forceAtlas2Based || {}),
        gravitationalConstant: huge ? -260 : large ? -210 : -160,
        centralGravity: huge ? 0.004 : large ? 0.005 : 0.008,
        springLength: huge ? 190 : large ? 175 : 155,
        springConstant: huge ? 0.022 : large ? 0.028 : 0.036,
        damping: 0.68,
        avoidOverlap: 1,
      },
      timestep: huge ? 0.38 : 0.48,
      minVelocity: huge ? 0.45 : 0.3,
      stabilization: {
        enabled: !large,
        iterations: huge ? 0 : large ? 0 : 260,
        updateInterval: 40,
        fit: false,
      },
    };
    const interaction = window.GraphUtils?.getInteractionOptions
      ? window.GraphUtils.getInteractionOptions()
      : { hover: true, zoomView: true, dragView: true, keyboard: false };
    // Some older DeepDream visual defaults include vis-network options that were
    // removed from the bundled version. Strip them before constructing Network.
    delete interaction.hideTooltipOnDragMove;
    return {
      autoResize: true,
      physics,
      layout: { improvedLayout: false },
      interaction,
      nodes: { chosen: true },
      edges: {
        arrows: { to: { enabled: false } },
        selectionWidth: 2,
        hoverWidth: 1.5,
      },
    };
  }

  function bindNetworkEvents(canvas) {
    network.on('click', params => {
      hideHover();
      if (params.nodes?.length) showNodeDetail(params.nodes[0]);
      else if (params.edges?.length) showEdgeDetail(params.edges[0]);
    });
    network.on('hoverNode', params => showHoverForNode(params.node, canvas));
    network.on('hoverEdge', params => showHoverForEdge(params.edge, canvas));
    network.on('blurNode', hideHover);
    network.on('blurEdge', hideHover);
    canvas.addEventListener('mouseleave', hideHover);
    network.on('dragStart', params => {
      hideHover();
      if (!params.nodes?.length) return;
      params.nodes.forEach(id => nodesDataSet.update({ id, fixed: false }));
      network.setOptions({ physics: { enabled: true } });
    });
    network.on('dragEnd', params => {
      if (!params.nodes?.length) return;
      const pos = network.getPositions(params.nodes);
      params.nodes.forEach(id => {
        if (!pos[id]) return;
        pinnedPositions[id] = { x: pos[id].x, y: pos[id].y };
        nodesDataSet.update({ id, x: pos[id].x, y: pos[id].y, fixed: { x: true, y: true } });
      });
      network.setOptions({ physics: { enabled: true } });
      scheduleNaturalFit(280);
    });
    network.on('zoom', () => updateHoverPosition(canvas));
    network.on('dragging', () => updateHoverPosition(canvas));
  }

  function ensureHover(canvas) {
    if (hoverPanel && hoverPanel.parentElement) return hoverPanel;
    hoverPanel = document.createElement('div');
    hoverPanel.className = 'node-hover-info';
    hoverPanel.style.opacity = '0';
    canvas.appendChild(hoverPanel);
    return hoverPanel;
  }

  function showHoverForNode(id, canvas) {
    const meta = nodeMetaById.get(id);
    if (!meta) return;
    const item = meta.item;
    const typeLabel = meta.type === 'document' ? 'Document' : meta.type === 'episode' ? 'Episode' : 'Entity';
    const name = meta.type === 'document' ? docTitle(item) : meta.type === 'episode' ? (item.heading_path || item.name || 'Episode') : conceptTitle(item);
    const content = meta.type === 'episode' ? item.content : item.content || item.summary || docPath(item);
    const panel = ensureHover(canvas);
    panel.dataset.targetType = 'node';
    panel.dataset.targetId = id;
    panel.innerHTML = `
      <div class="nhv-name">${escapeHtml(name)}</div>
      <span class="nhv-version">${escapeHtml(typeLabel)}${meta.versionTotal > 1 ? ` · v${meta.versionTotal}` : ''}</span>
      <div class="nhv-content">${escapeHtml(normalizeText(content, 180))}</div>
      <div style="font-size:0.6875rem;color:var(--text-muted);margin-top:0.25rem;">${formatDateMs(item.processed_time || item.updated_at || item.created_at)}</div>
    `;
    positionHoverForNode(id, canvas);
    requestAnimationFrame(() => { if (hoverPanel) hoverPanel.style.opacity = '1'; });
  }

  function showHoverForEdge(id, canvas) {
    const meta = edgeMetaById.get(id);
    if (!meta) return;
    const panel = ensureHover(canvas);
    panel.dataset.targetType = 'edge';
    panel.dataset.targetId = id;
    const rel = meta.item;
    panel.innerHTML = meta.type === 'relation'
      ? `<div class="nhv-name">${escapeHtml(conceptTitle(rel))}</div>
         <span class="nhv-version">Relation${graphModel.versions?.[rel.family_id]?.total > 1 ? ` · v${graphModel.versions[rel.family_id].total}` : ''}</span>
         <div class="nhv-content">${escapeHtml(normalizeText(rel.content || rel.summary, 180))}</div>
         <div style="font-size:0.6875rem;color:var(--text-muted);margin-top:0.25rem;">${formatDateMs(rel.processed_time)}</div>`
      : `<div class="nhv-name">${escapeHtml(meta.item.edge_type || 'Edge')}</div>`;
    positionHoverForEdge(id, canvas);
    requestAnimationFrame(() => { if (hoverPanel) hoverPanel.style.opacity = '1'; });
  }

  function updateHoverPosition(canvas) {
    if (!hoverPanel || hoverPanel.style.opacity === '0') return;
    const type = hoverPanel.dataset.targetType;
    const id = hoverPanel.dataset.targetId;
    if (type === 'node') positionHoverForNode(id, canvas);
    else if (type === 'edge') positionHoverForEdge(id, canvas);
  }

  function positionHoverForNode(id, canvas) {
    if (!network || !hoverPanel) return;
    const pos = network.getPositions([id])[id];
    if (!pos) return;
    const dom = network.canvasToDOM(pos);
    placeHover(dom.x, dom.y - 18, canvas);
  }

  function positionHoverForEdge(id, canvas) {
    if (!network || !hoverPanel) return;
    const edge = edgesDataSet.get(id);
    if (!edge) return;
    const positions = network.getPositions([edge.from, edge.to]);
    const a = positions[edge.from];
    const b = positions[edge.to];
    if (!a || !b) return;
    const dom = network.canvasToDOM({ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 });
    placeHover(dom.x, dom.y - 20, canvas);
  }

  function placeHover(x, y, canvas) {
    const rect = canvas.getBoundingClientRect();
    const panelRect = hoverPanel.getBoundingClientRect();
    const left = Math.min(Math.max(8, x - panelRect.width / 2), rect.width - panelRect.width - 8);
    const top = Math.min(Math.max(8, y - panelRect.height), rect.height - panelRect.height - 8);
    hoverPanel.style.left = `${left}px`;
    hoverPanel.style.top = `${top}px`;
  }

  function hideHover() {
    if (hoverPanel) hoverPanel.style.opacity = '0';
  }

  function showNodeDetail(id) {
    const meta = nodeMetaById.get(id);
    if (!meta) return;
    const detail = document.getElementById('graph-detail');
    const subtitle = document.getElementById('graph-detail-subtitle');
    if (!detail) return;
    if (subtitle) subtitle.textContent = id;
    if (meta.type === 'document') detail.innerHTML = renderDocumentDetail(meta.item);
    else if (meta.type === 'episode') detail.innerHTML = renderEpisodeDetail(meta.item);
    else detail.innerHTML = renderConceptDetail(meta.item);
    bindDetailActions(detail, meta);
  }

  function showEdgeDetail(id) {
    const meta = edgeMetaById.get(id);
    if (!meta) return;
    const detail = document.getElementById('graph-detail');
    const subtitle = document.getElementById('graph-detail-subtitle');
    if (!detail) return;
    if (meta.type === 'relation') {
      if (subtitle) subtitle.textContent = `relation:${meta.item.family_id}`;
      detail.innerHTML = renderRelationDetail(meta.item);
      bindDetailActions(detail, meta);
      loadConceptVersions(meta.item.family_id);
      return;
    }
    if (subtitle) subtitle.textContent = id;
    const edgeEvidence = renderEvidenceList(meta.item?.provenance?.evidence || []);
    detail.innerHTML = `
      <h3 style="font-size:1.05rem;font-weight:700;margin-bottom:0.75rem;">Edge</h3>
      ${field('类型', escapeHtml(meta.item.edge_type || 'edge'))}
      ${field('From', `<span class="mono">${escapeHtml(meta.item.from || '')}</span>`)}
      ${field('To', `<span class="mono">${escapeHtml(meta.item.to || '')}</span>`)}
      ${edgeEvidence ? `<div class="graph-detail-section"><div style="font-weight:600;margin-bottom:0.35rem;">原文证据</div>${edgeEvidence}</div>` : ''}
    `;
  }

  function renderDocumentDetail(doc) {
    return `
      ${kindBadge('Document', ROLE_COLORS.document)}
      <h3 style="font-size:1.05rem;font-weight:700;margin-bottom:0.75rem;overflow-wrap:anywhere;">${escapeHtml(doc.title || docTitle(doc))}</h3>
      ${field('版本', `<span class="mono">${escapeHtml(doc.document_version_id || doc.version_id || '')}</span>`)}
      ${field('Family', `<span class="mono">${escapeHtml(doc.family_id || '')}</span>`)}
      ${field('路径', escapeHtml(docPath(doc) || '-'))}
      ${field('Hash', `<span class="mono">${escapeHtml(doc.content_hash || '')}</span>`)}
      ${field('大小', `${Number(doc.size || 0).toLocaleString()} B`)}
      ${field('入库时间', formatDateMs(doc.processed_time))}
      <div class="graph-detail-actions">
        <button class="btn btn-secondary btn-sm" data-graph-action="more-document">更多详情</button>
      </div>
    `;
  }

  function renderEpisodeDetail(ep) {
    const concepts = (graphData.edges || []).filter(e => e.episode_version_id === ep.version_id && ['MENTIONS', 'ASSERTS'].includes(e.edge_type)).length;
    const original = episodeOriginalText(ep);
    return `
      ${kindBadge('Episode', ROLE_COLORS.episode)}
      <h3 style="font-size:1.05rem;font-weight:700;margin-bottom:0.75rem;">${escapeHtml(ep.heading_path || ep.name || 'Episode')}</h3>
      ${field('版本', `<span class="mono">${escapeHtml(ep.version_id || '')}</span>`)}
      ${field('文档版本', `<span class="mono">${escapeHtml(ep.document_version_id || '')}</span>`)}
      ${field('Offset', `${ep.start_offset ?? '-'} - ${ep.end_offset ?? '-'}`)}
      ${field('概念数', String(concepts))}
      <div class="graph-detail-section">
        <div class="graph-tabbar" data-episode-tabs style="display:flex;gap:0.4rem;margin-bottom:0.5rem;">
          <button class="btn btn-secondary btn-sm active" data-episode-tab="thinking">思考内容</button>
          <button class="btn btn-secondary btn-sm" data-episode-tab="source">原文切片</button>
        </div>
        <div class="md-content graph-episode-pane" data-episode-pane="thinking" style="max-height:280px;overflow:auto;background:var(--bg-secondary);padding:0.75rem;border-radius:0.5rem;font-size:0.8rem;line-height:1.55;">${renderMarkdown(ep.content || '')}</div>
        <div class="md-content graph-episode-pane" data-episode-pane="source" style="display:none;max-height:280px;overflow:auto;background:var(--bg-secondary);padding:0.75rem;border-radius:0.5rem;font-size:0.8rem;line-height:1.55;">${renderMarkdown(original || '')}</div>
      </div>
      <div class="graph-detail-actions">
        <button class="btn btn-secondary btn-sm" data-graph-action="more-episode">更多详情</button>
      </div>
    `;
  }

  function renderConceptDetail(concept) {
    const versionInfo = graphModel.versions?.[concept.family_id];
    const evidenceHtml = renderEvidenceList(mentionEvidenceForConcept(concept.family_id));
    return `
      ${kindBadge('Entity', ROLE_COLORS.entity)}
      <h3 style="font-size:1.05rem;font-weight:700;margin-bottom:0.75rem;overflow-wrap:anywhere;">${escapeHtml(conceptTitle(concept) || 'Concept')}</h3>
      ${versionInfo?.total > 1 ? `<div style="margin-bottom:0.55rem;">${pill(`${versionInfo.total} versions`, '#f59e0b')}</div>` : ''}
      ${field('Family', `<span class="mono">${escapeHtml(concept.family_id || '')}</span>`)}
      ${field('版本', `<span class="mono">${escapeHtml(concept.version_id || '')}</span>`)}
      ${field('置信度', concept.confidence == null ? '-' : String(concept.confidence))}
      ${field('来源', escapeHtml(concept.source_document || '-'))}
      ${field('时间', formatDateMs(concept.processed_time))}
      ${versionInfo?.total > 1 ? `<div class="graph-detail-section" data-version-switcher="${escapeAttr(concept.family_id)}">${spinnerHtml('spinner-sm')} 加载版本切换器...</div>` : ''}
      <div class="graph-detail-section">
        <div style="font-weight:600;margin-bottom:0.35rem;">内容</div>
        <div class="md-content" data-concept-version-content style="max-height:220px;overflow:auto;background:var(--bg-secondary);padding:0.75rem;border-radius:0.5rem;font-size:0.8rem;line-height:1.55;">${renderMarkdown(concept.content || concept.summary || '')}</div>
      </div>
      <div class="graph-detail-section" data-concept-version-evidence style="${evidenceHtml ? '' : 'display:none;'}"><div style="font-weight:600;margin-bottom:0.35rem;">原文句子</div>${evidenceHtml}</div>
      <div class="graph-detail-actions">
        <button class="btn btn-secondary btn-sm" data-graph-action="more-concept">更多详情</button>
        <button class="btn btn-primary btn-sm" data-graph-action="versions">版本历史</button>
        <button class="btn btn-secondary btn-sm" data-graph-action="focus-concept">聚焦模式</button>
        ${focusFamilyId ? '<button class="btn btn-secondary btn-sm" data-graph-action="exit-focus">退出聚焦</button>' : ''}
      </div>
    `;
  }

  function renderRelationDetail(rel) {
    const endpoints = relationEndpoints(rel, graphData);
    const endpointNames = endpoints.map(fid => conceptTitle(graphModel.conceptByFamily.get(fid) || { family_id: fid }));
    const versionInfo = graphModel.versions?.[rel.family_id];
    return `
      ${kindBadge('Relation', ROLE_COLORS.relation)}
      <h3 style="font-size:1.05rem;font-weight:700;margin-bottom:0.75rem;overflow-wrap:anywhere;">${escapeHtml(conceptTitle(rel) || 'Relation')}</h3>
      ${versionInfo?.total > 1 ? `<div style="margin-bottom:0.55rem;">${pill(`${versionInfo.total} versions`, '#f59e0b')}</div>` : ''}
      ${field('Family', `<span class="mono">${escapeHtml(rel.family_id || '')}</span>`)}
      ${field('版本', `<span class="mono">${escapeHtml(rel.version_id || '')}</span>`)}
      ${field('连接', endpointNames.map(x => escapeHtml(x)).join(' ↔ '))}
      ${field('来源', escapeHtml(rel.source_document || '-'))}
      ${field('时间', formatDateMs(rel.processed_time))}
      ${versionInfo?.total > 1 ? `<div class="graph-detail-section" data-version-switcher="${escapeAttr(rel.family_id)}">${spinnerHtml('spinner-sm')} 加载版本切换器...</div>` : ''}
      <div class="graph-detail-section">
        <div style="font-weight:600;margin-bottom:0.35rem;">内容</div>
        <div class="md-content" data-concept-version-content style="max-height:220px;overflow:auto;background:var(--bg-secondary);padding:0.75rem;border-radius:0.5rem;font-size:0.8rem;line-height:1.55;">${renderMarkdown(rel.content || rel.summary || '')}</div>
      </div>
      <div class="graph-detail-actions">
        <button class="btn btn-secondary btn-sm" data-graph-action="more-relation">更多详情</button>
        <button class="btn btn-primary btn-sm" data-graph-action="versions">版本历史</button>
        <button class="btn btn-secondary btn-sm" data-graph-action="focus-concept">聚焦模式</button>
        ${focusFamilyId ? '<button class="btn btn-secondary btn-sm" data-graph-action="exit-focus">退出聚焦</button>' : ''}
      </div>
      <div class="graph-detail-section">
        <div style="font-weight:600;margin-bottom:0.5rem;">版本历史</div>
        <div id="graph-version-list">${spinnerHtml()} 加载版本...</div>
      </div>
    `;
  }

  function episodeOriginalText(ep) {
    return ep?.metadata?.source_text || ep?.source_text || ep?.source_span?.source_text || ep?.content || '';
  }

  function mentionEvidenceForConcept(familyId, limit = 6, versionId = '') {
    const items = [];
    const seen = new Set();
    (graphData.edges || []).forEach(edge => {
      if (edge.edge_type !== 'MENTIONS' || edge.target_family_id !== familyId) return;
      if (versionId && edge.target_version_id !== versionId) return;
      (edge.provenance?.evidence || []).forEach(ev => {
        const key = `${ev.start_offset}:${ev.end_offset}:${ev.sentence}`;
        if (seen.has(key)) return;
        seen.add(key);
        items.push({ ...ev, episode_version_id: edge.episode_version_id });
      });
    });
    return items.slice(0, limit);
  }

  function bindEpisodeTabs(root) {
    root.querySelectorAll('[data-episode-tab]').forEach(btn => {
      btn.addEventListener('click', () => {
        const target = btn.getAttribute('data-episode-tab');
        root.querySelectorAll('[data-episode-tab]').forEach(x => x.classList.toggle('active', x === btn));
        root.querySelectorAll('[data-episode-pane]').forEach(pane => {
          pane.style.display = pane.getAttribute('data-episode-pane') === target ? '' : 'none';
        });
      });
    });
  }

  function renderVersionSwitcher(versions, currentVersionId) {
    if (!versions?.length) return '';
    const sorted = versions.slice().sort(compareVersionTime);
    const currentIdx = Math.max(0, sorted.findIndex(v => v.version_id === currentVersionId));
    return `
      <div style="display:flex;gap:0.4rem;align-items:center;flex-wrap:wrap;">
        <button class="btn btn-secondary btn-sm" data-version-nav="-1" ${currentIdx <= 0 ? 'disabled' : ''}>上一版</button>
        <select class="input" data-version-select style="flex:1;min-width:160px;height:34px;">
          ${sorted.map((v, idx) => `<option value="${escapeAttr(v.version_id)}" ${idx === currentIdx ? 'selected' : ''}>v${v.version_seq || idx + 1} · ${escapeHtml(formatDateMs(v.processed_time))}</option>`).join('')}
        </select>
        <button class="btn btn-secondary btn-sm" data-version-nav="1" ${currentIdx >= sorted.length - 1 ? 'disabled' : ''}>下一版</button>
      </div>
    `;
  }

  async function loadInlineVersionSwitcher(detail, concept) {
    const box = detail.querySelector('[data-version-switcher]');
    if (!box || !concept?.family_id) return;
    try {
      const res = await state.api.entityVersions(concept.family_id, state.currentGraphId);
      const versions = res.data?.versions || [];
      box.innerHTML = renderVersionSwitcher(versions, concept.version_id);
      const select = box.querySelector('[data-version-select]');
      const applyVersion = (versionId) => {
        const selected = versions.find(v => v.version_id === versionId) || concept;
        const content = detail.querySelector('[data-concept-version-content]');
        if (content) content.innerHTML = renderMarkdown(selected.content || selected.summary || selected.name || '');
        const evidenceBox = detail.querySelector('[data-concept-version-evidence]');
        if (evidenceBox) {
          const evidenceHtml = renderEvidenceList(mentionEvidenceForConcept(concept.family_id, 6, selected.version_id));
          evidenceBox.style.display = evidenceHtml ? '' : 'none';
          evidenceBox.innerHTML = `<div style="font-weight:600;margin-bottom:0.35rem;">原文句子</div>${evidenceHtml}`;
        }
        box.innerHTML = renderVersionSwitcher(versions, selected.version_id);
        bindInlineVersionSwitcher(detail, concept, versions, applyVersion);
      };
      bindInlineVersionSwitcher(detail, concept, versions, applyVersion);
    } catch (err) {
      box.innerHTML = `<span style="color:var(--danger);">版本加载失败: ${escapeHtml(err.message)}</span>`;
    }
  }

  function bindInlineVersionSwitcher(detail, concept, versions, applyVersionFn) {
    const box = detail.querySelector('[data-version-switcher]');
    if (!box) return;
    const applyVersion = applyVersionFn || ((versionId) => {
      const selected = versions.find(v => v.version_id === versionId);
      if (!selected) return;
      const content = detail.querySelector('[data-concept-version-content]');
      if (content) content.innerHTML = renderMarkdown(selected.content || selected.summary || selected.name || '');
    });
    const select = box.querySelector('[data-version-select]');
    if (select) select.addEventListener('change', () => applyVersion(select.value));
    box.querySelectorAll('[data-version-nav]').forEach(btn => {
      btn.addEventListener('click', () => {
        const current = box.querySelector('[data-version-select]');
        const idx = versions.findIndex(v => v.version_id === current?.value);
        const next = versions[Math.max(0, Math.min(versions.length - 1, idx + Number(btn.getAttribute('data-version-nav') || 0)))];
        if (next) applyVersion(next.version_id);
      });
    });
  }

  function focusConcept(familyId) {
    focusFamilyId = familyId || null;
    updateGraphStep(playbackStep, { fitDelay: 40, fitDuration: 320 });
    const exit = document.getElementById('graph-exit-focus');
    if (exit) exit.style.display = focusFamilyId ? '' : 'none';
  }

  function renderEvidenceList(evidence) {
    if (!Array.isArray(evidence) || !evidence.length) return '';
    return `
      <div style="display:flex;flex-direction:column;gap:0.45rem;">
        ${evidence.map(ev => `
          <div style="border:1px solid var(--border-color);background:var(--bg-secondary);border-radius:0.55rem;padding:0.55rem 0.65rem;">
            <div style="font-size:0.78rem;line-height:1.55;color:var(--text-primary);">${highlightEvidenceSentence(ev)}</div>
            <div style="margin-top:0.35rem;display:flex;gap:0.35rem;flex-wrap:wrap;">
              ${pill(ev.match_type || 'match', '#38bdf8')}
              ${ev.confidence != null ? pill(`conf ${Number(ev.confidence).toFixed(2)}`, '#22c55e') : ''}
              ${ev.start_offset != null ? pill(`${ev.start_offset}-${ev.end_offset}`, '#94a3b8') : ''}
            </div>
          </div>
        `).join('')}
      </div>
    `;
  }

  function highlightEvidenceSentence(ev) {
    const sentence = String(ev?.sentence || '');
    const quote = String(ev?.quote || '');
    if (!sentence || !quote) return escapeHtml(sentence || quote);
    const idx = sentence.indexOf(quote);
    if (idx < 0) return escapeHtml(sentence);
    return `${escapeHtml(sentence.slice(0, idx))}<mark style="background:rgba(250,204,21,0.35);color:inherit;border-radius:0.2rem;padding:0 0.08rem;">${escapeHtml(quote)}</mark>${escapeHtml(sentence.slice(idx + quote.length))}`;
  }

  function kindBadge(text, color) {
    return `<span class="graph-kind-badge" style="background:${color.glow};color:${color.border};border:1px solid ${color.bg};">${escapeHtml(text)}</span>`;
  }

  function bindDetailActions(detail, meta) {
    bindEpisodeTabs(detail);
    if (meta.type === 'entity' || meta.type === 'relation') {
      loadInlineVersionSwitcher(detail, meta.item);
    }
    detail.querySelector('[data-graph-action="more-document"]')?.addEventListener('click', () => openDocumentModal(meta.item));
    detail.querySelector('[data-graph-action="more-episode"]')?.addEventListener('click', () => openEpisodeModal(meta.item));
    detail.querySelector('[data-graph-action="more-concept"]')?.addEventListener('click', () => openConceptModal(meta.item));
    detail.querySelector('[data-graph-action="more-relation"]')?.addEventListener('click', () => openConceptModal(meta.item));
    detail.querySelector('[data-graph-action="versions"]')?.addEventListener('click', () => openVersionsModal(meta.item));
    detail.querySelector('[data-graph-action="focus-concept"]')?.addEventListener('click', () => focusConcept(meta.item.family_id));
    detail.querySelector('[data-graph-action="exit-focus"]')?.addEventListener('click', () => focusConcept(null));
    if (window.lucide) lucide.createIcons({ nodes: [detail] });
  }

  function openDocumentModal(doc) {
    const modal = showModal({
      title: docTitle(doc),
      size: 'lg',
      content: `
        <div class="graph-modal-grid">
          <span>Document</span><strong class="mono">${escapeHtml(doc.document_version_id || '')}</strong>
          <span>Family</span><strong class="mono">${escapeHtml(doc.family_id || '')}</strong>
          <span>Path</span><strong>${escapeHtml(docPath(doc) || '-')}</strong>
          <span>Hash</span><strong class="mono">${escapeHtml(doc.content_hash || '')}</strong>
          <span>Processed</span><strong>${formatDateMs(doc.processed_time)}</strong>
        </div>
        <div class="graph-detail-section">
          <div style="display:flex;align-items:center;justify-content:space-between;gap:0.75rem;margin-bottom:0.5rem;">
            <div style="font-weight:600;">Markdown 原文</div>
            <span id="graph-doc-content-status" class="mono" style="font-size:0.72rem;color:var(--text-muted);"></span>
          </div>
          <div id="graph-doc-content" class="graph-document-source">${spinnerHtml()} 加载文档正文...</div>
          <div style="display:flex;justify-content:center;margin-top:0.65rem;">
            <button id="graph-doc-load-more" class="btn btn-secondary btn-sm" style="display:none;">加载更多</button>
          </div>
        </div>
      `,
    });
    loadDocumentContent(modal, doc, 0, false);
  }

  async function loadDocumentContent(modal, doc, offset, append) {
    const body = modal.overlay.querySelector('#graph-doc-content');
    const status = modal.overlay.querySelector('#graph-doc-content-status');
    const more = modal.overlay.querySelector('#graph-doc-load-more');
    if (!body || !doc.document_version_id) return;
    if (!append) body.innerHTML = `${spinnerHtml()} 加载文档正文...`;
    if (more) more.style.display = 'none';
    try {
      const res = await state.api.documentContent(doc.document_version_id, state.currentGraphId, {
        offset,
        limit: 20000,
      });
      const data = res.data || {};
      const chunk = escapeHtml(data.content || '');
      if (append) body.innerHTML += chunk;
      else body.innerHTML = chunk || '<span style="color:var(--text-muted);">空文档</span>';
      if (status) {
        const shown = Math.min(Number(data.next_offset || data.total_chars || 0), Number(data.total_chars || 0));
        status.textContent = `${shown.toLocaleString()} / ${Number(data.total_chars || 0).toLocaleString()} chars`;
      }
      if (more && data.next_offset != null) {
        more.style.display = 'inline-flex';
        more.onclick = () => loadDocumentContent(modal, doc, data.next_offset, true);
      }
    } catch (err) {
      body.innerHTML = `<div style="color:var(--danger);">加载文档正文失败: ${escapeHtml(err.message)}</div>`;
    }
  }

  function openEpisodeModal(ep) {
    const original = episodeOriginalText(ep);
    const modal = showModal({
      title: ep.heading_path || ep.name || 'Episode',
      size: 'lg',
      content: `
        <div class="graph-modal-grid">
          <span>Episode</span><strong class="mono">${escapeHtml(ep.version_id || '')}</strong>
          <span>Document</span><strong class="mono">${escapeHtml(ep.document_version_id || '')}</strong>
          <span>Offset</span><strong>${ep.start_offset ?? '-'} - ${ep.end_offset ?? '-'}</strong>
          <span>Chunk</span><strong class="mono">${escapeHtml(ep.chunk_hash || '')}</strong>
        </div>
        <div class="graph-detail-section">
          <div class="graph-tabbar" data-episode-tabs style="display:flex;gap:0.4rem;margin-bottom:0.5rem;">
            <button class="btn btn-secondary btn-sm active" data-episode-tab="thinking">思考内容</button>
            <button class="btn btn-secondary btn-sm" data-episode-tab="source">原文切片</button>
          </div>
          <div class="graph-episode-pane md-content" data-episode-pane="thinking">${renderMarkdown(ep.content || '')}</div>
          <div class="graph-episode-pane md-content" data-episode-pane="source" style="display:none;">${renderMarkdown(original || '')}</div>
        </div>
      `,
    });
    bindEpisodeTabs(modal.overlay);
  }

  function openConceptModal(concept) {
    const isRelation = concept.role === 'relation';
    const provenance = (graphData.edges || []).filter(e =>
      e.target_family_id === concept.family_id || e.relation_family_id === concept.family_id || e.source_family_id === concept.family_id
    );
    const evidenceHtml = !isRelation ? renderEvidenceList(mentionEvidenceForConcept(concept.family_id, 12)) : '';
    showModal({
      title: conceptTitle(concept),
      size: 'lg',
      content: `
        <div style="display:flex;gap:0.4rem;flex-wrap:wrap;margin-bottom:0.75rem;">
          ${kindBadge(isRelation ? 'Relation' : 'Entity', isRelation ? ROLE_COLORS.relation : ROLE_COLORS.entity)}
          ${pill(`${graphModel.versions?.[concept.family_id]?.total || 1} versions`, '#f59e0b')}
        </div>
        <div class="graph-modal-grid">
          <span>Family</span><strong class="mono">${escapeHtml(concept.family_id || '')}</strong>
          <span>Version</span><strong class="mono">${escapeHtml(concept.version_id || '')}</strong>
          <span>Source</span><strong>${escapeHtml(concept.source_document || '-')}</strong>
          <span>Episode</span><strong class="mono">${escapeHtml(concept.episode_version_id || '')}</strong>
          <span>Processed</span><strong>${formatDateMs(concept.processed_time)}</strong>
        </div>
        <div class="graph-detail-section md-content">${renderMarkdown(concept.content || concept.summary || '')}</div>
        ${evidenceHtml ? `<div class="graph-detail-section"><div style="font-weight:600;margin-bottom:0.4rem;">原文句子</div>${evidenceHtml}</div>` : ''}
        <div class="graph-detail-section">
          <div style="font-weight:600;margin-bottom:0.4rem;">Provenance edges</div>
          ${provenance.slice(0, 30).map(e => pill(e.edge_type || 'edge')).join('') || emptyState('暂无溯源')}
        </div>
      `,
    });
  }

  async function openVersionsModal(concept) {
    const modal = showModal({
      title: `版本历史 - ${truncate(conceptTitle(concept), 48)}`,
      size: 'lg',
      content: `<div style="padding:1rem;">${spinnerHtml()} 加载版本...</div>`,
    });
    try {
      const res = await state.api.entityVersions(concept.family_id, state.currentGraphId);
      const versions = res.data?.versions || [];
      modal.overlay.querySelector('.modal-body').innerHTML = renderVersionsTimeline(versions);
    } catch (err) {
      modal.overlay.querySelector('.modal-body').innerHTML = `<div style="color:var(--danger);">加载版本失败: ${escapeHtml(err.message)}</div>`;
    }
  }

  function renderVersionsTimeline(versions) {
    if (!versions.length) return emptyState('暂无版本');
    const sorted = versions.slice().sort(compareVersionTime);
    const newestFirst = sorted.slice().reverse();
    return `
      <div style="max-height:62vh;overflow:auto;padding:0.25rem;">
        ${newestFirst.map((v, idx) => {
          const chronologicalIndex = sorted.findIndex(item => item.version_id === v.version_id);
          const prev = chronologicalIndex > 0 ? sorted[chronologicalIndex - 1] : null;
          return `
          <div style="position:relative;padding-left:1.4rem;padding-bottom:1rem;">
            ${idx < newestFirst.length - 1 ? '<div style="position:absolute;left:5px;top:13px;bottom:0;width:1px;background:var(--border-color);"></div>' : ''}
            <div style="position:absolute;left:0;top:4px;width:12px;height:12px;border-radius:50%;background:${v.content_changed ? 'var(--primary)' : 'var(--border-color)'};border:2px solid ${v.content_changed ? 'var(--primary-hover)' : 'var(--border-hover)'};"></div>
            <div class="graph-version-row">
              <div style="display:flex;justify-content:space-between;gap:0.5rem;margin-bottom:0.25rem;">
                <span class="mono" style="font-size:0.72rem;">${escapeHtml(v.version_id || '')}</span>
                <span class="badge ${v.content_changed ? 'badge-primary' : ''}">${v.content_changed ? 'changed' : 'same'}</span>
              </div>
              <div style="font-size:0.75rem;color:var(--text-muted);">${formatDateMs(v.processed_time)} · ${escapeHtml(v.source_document || '')}</div>
              ${renderVersionDiff(v, prev)}
              <details class="graph-version-full">
                <summary>完整内容</summary>
                <div class="md-content" style="font-size:0.8rem;margin-top:0.45rem;line-height:1.5;">${renderMarkdown(versionText(v))}</div>
              </details>
            </div>
          </div>
        `;}).join('')}
      </div>
    `;
  }

  function compareVersionTime(a, b) {
    const ta = a.processed_time ? new Date(a.processed_time).getTime() : 0;
    const tb = b.processed_time ? new Date(b.processed_time).getTime() : 0;
    if (ta !== tb) return ta - tb;
    return String(a.version_id || '').localeCompare(String(b.version_id || ''));
  }

  function versionText(v) {
    return String(v.content || v.summary || v.name || '').trim();
  }

  function tokenizeDiffText(text) {
    const tokens = String(text || '').match(/[\u3400-\u9fff]|[A-Za-z0-9_]+|[\r\n]+|\s+|[^\sA-Za-z0-9_\u3400-\u9fff]/g);
    return tokens || [];
  }

  function renderVersionDiff(current, previous) {
    const currentText = versionText(current);
    const previousText = previous ? versionText(previous) : '';
    if (!previous) {
      return `
        <div class="version-diff-container graph-version-diff">
          <div class="version-diff-header">
            <span>初始版本</span>
            <span>+${tokenizeDiffText(currentText).filter(t => t.trim()).length}</span>
          </div>
          <div class="version-diff-body version-diff-inline">
            ${currentText ? `<span class="version-diff-token added">${escapeHtml(currentText)}</span>` : '<span class="version-diff-line unchanged">空内容</span>'}
          </div>
        </div>
      `;
    }

    if (currentText === previousText) {
      return `
        <div class="version-diff-container graph-version-diff">
          <div class="version-diff-header">
            <span>相比上一版本</span>
            <span>无内容变化</span>
          </div>
          <div class="version-diff-body">
            <div class="version-diff-line unchanged">${escapeHtml(truncate(currentText || '空内容', 220))}</div>
          </div>
        </div>
      `;
    }

    const diff = buildInlineDiff(previousText, currentText);
    return `
      <div class="version-diff-container graph-version-diff">
        <div class="version-diff-header">
          <span>相比上一版本</span>
          <span><span class="diff-count added">+${diff.added}</span> <span class="diff-count removed">-${diff.removed}</span></span>
        </div>
        <div class="version-diff-body version-diff-inline">${diff.html}</div>
      </div>
    `;
  }

  function buildInlineDiff(oldText, newText) {
    const oldTokens = tokenizeDiffText(oldText);
    const newTokens = tokenizeDiffText(newText);
    if (oldTokens.length * newTokens.length > 120000) {
      return buildCoarseDiff(oldText, newText);
    }

    const rows = oldTokens.length + 1;
    const cols = newTokens.length + 1;
    const dp = Array.from({ length: rows }, () => new Uint16Array(cols));
    for (let i = oldTokens.length - 1; i >= 0; i--) {
      for (let j = newTokens.length - 1; j >= 0; j--) {
        dp[i][j] = oldTokens[i] === newTokens[j]
          ? dp[i + 1][j + 1] + 1
          : Math.max(dp[i + 1][j], dp[i][j + 1]);
      }
    }

    const parts = [];
    let added = 0;
    let removed = 0;
    let i = 0;
    let j = 0;
    while (i < oldTokens.length && j < newTokens.length) {
      if (oldTokens[i] === newTokens[j]) {
        parts.push({ type: 'same', text: oldTokens[i] });
        i++;
        j++;
      } else if (dp[i + 1][j] >= dp[i][j + 1]) {
        parts.push({ type: 'removed', text: oldTokens[i] });
        if (oldTokens[i].trim()) removed++;
        i++;
      } else {
        parts.push({ type: 'added', text: newTokens[j] });
        if (newTokens[j].trim()) added++;
        j++;
      }
    }
    while (i < oldTokens.length) {
      parts.push({ type: 'removed', text: oldTokens[i] });
      if (oldTokens[i].trim()) removed++;
      i++;
    }
    while (j < newTokens.length) {
      parts.push({ type: 'added', text: newTokens[j] });
      if (newTokens[j].trim()) added++;
      j++;
    }

    const merged = [];
    parts.forEach(part => {
      const last = merged[merged.length - 1];
      if (last && last.type === part.type) last.text += part.text;
      else merged.push({ ...part });
    });
    const html = merged.map(part => {
      if (!part.text) return '';
      if (part.type === 'same') return `<span class="version-diff-token unchanged">${escapeHtml(part.text)}</span>`;
      return `<span class="version-diff-token ${part.type}">${escapeHtml(part.text)}</span>`;
    }).join('');
    return { html, added, removed };
  }

  function buildCoarseDiff(oldText, newText) {
    const oldLines = String(oldText || '').split(/\n+/).filter(Boolean);
    const newLines = String(newText || '').split(/\n+/).filter(Boolean);
    return {
      added: newLines.length,
      removed: oldLines.length,
      html: `
        ${oldLines.map(line => `<div class="version-diff-line removed">-${escapeHtml(line)}</div>`).join('')}
        ${newLines.map(line => `<div class="version-diff-line added">+${escapeHtml(line)}</div>`).join('')}
      `,
    };
  }

  async function loadConceptVersions(familyId) {
    const el = document.getElementById('graph-version-list');
    if (!el) return;
    try {
      const res = await state.api.entityVersions(familyId, state.currentGraphId);
      const versions = res.data?.versions || [];
      el.innerHTML = versions.length ? versions.slice().reverse().slice(0, 6).map(v => `
        <div class="graph-version-row">
          <div style="display:flex;justify-content:space-between;gap:0.5rem;margin-bottom:0.25rem;">
            <span class="mono" style="font-size:0.72rem;">${escapeHtml(v.version_id || '')}</span>
            <span class="badge ${v.content_changed ? 'badge-primary' : ''}">${v.content_changed ? 'changed' : 'same'}</span>
          </div>
          <div style="font-size:0.75rem;color:var(--text-muted);">${formatDateMs(v.processed_time)}</div>
          <div style="font-size:0.8rem;margin-top:0.4rem;line-height:1.45;">${escapeHtml(truncate(v.content || v.summary || v.name || '', 160))}</div>
        </div>
      `).join('') : emptyState('暂无版本');
    } catch (err) {
      el.innerHTML = `<div style="color:var(--danger);">加载版本失败: ${escapeHtml(err.message)}</div>`;
    }
  }

  function updateSummary(visible) {
    const summary = document.getElementById('graph-summary');
    if (!summary) return;
    if (!visible) {
      summary.textContent = '选择一个或多个文档';
      return;
    }
    const relCount = visible.relationEdges.length;
    const total = growthTotals || graphData?.counts || {};
    const growing = hasActiveGrowth();
    const loading = growing ? (growthPauseRequested ? '已暂停 · ' : `并行生长中(${growthControllers.size}) · `) : '';
    const shownEpisodes = growing ? growthLoaded.episodes : visible.episodes.length;
    const shownEntities = growing ? growthLoaded.concepts : visible.entities.length;
    const shownRelations = growing ? growthLoaded.relations : relCount;
    const totalHint = total.concepts || total.relations
      ? ` · 进度 ${formatNumber(shownEntities)}/${formatNumber(total.concepts || visible.entities.length)} 实体，${formatNumber(shownRelations)}/${formatNumber(total.relations || relCount)} 关系`
      : '';
    summary.textContent = `${loading}${focusFamilyId ? '聚焦模式 · ' : ''}${visible.documents.length} 文档 / ${shownEpisodes} Episode / ${shownEntities} 实体节点 / ${shownRelations} 关系边${totalHint}`;
  }

  function selectedDocNames() {
    const names = docs
      .filter(d => selectedDocVersions.has(d.document_version_id))
      .map(d => docTitle(d));
    return names;
  }

  function updateGraphTitle() {
    const title = document.getElementById('graph-main-title');
    if (!title) return;
    const names = selectedDocNames();
    if (!names.length) {
      title.textContent = '图谱可视化';
    } else if (names.length === 1) {
      title.textContent = names[0];
    } else {
      title.textContent = `${names.length} 个文档：${truncate(names.slice(0, 3).join('、'), 42)}`;
    }
  }

  function setDocsCollapsed(collapsed) {
    docsCollapsed = !!collapsed;
    const shell = document.querySelector('.graph-viz-shell');
    shell?.classList.toggle('docs-collapsed', docsCollapsed);
    const btn = document.getElementById('graph-doc-collapse');
    if (btn) {
      btn.title = docsCollapsed ? '展开文档栏' : '收起文档栏';
      btn.innerHTML = `<i data-lucide="${docsCollapsed ? 'panel-left-open' : 'panel-left-close'}" style="width:14px;height:14px;"></i>`;
    }
    if (window.lucide) lucide.createIcons({ nodes: [btn || document.body] });
    setTimeout(() => network?.fit({ animation: { duration: 240, easingFunction: 'easeInOutQuad' } }), 220);
  }

  function updatePlaybackControls() {
    const bar = document.getElementById('graph-playback');
    if (!bar || !graphModel) return;
    bar.style.display = graphModel.episodes.length ? 'flex' : 'none';
    const label = document.getElementById('graph-play-label');
    const fill = document.getElementById('graph-play-fill');
    const playBtn = document.getElementById('graph-play');
    const speedBtn = document.getElementById('graph-speed');
    const total = graphModel.episodes.length;
    const pos = playbackStep < 0 ? total : playbackStep;
    if (fill) fill.style.width = `${Math.round((pos / Math.max(1, total)) * 100)}%`;
    if (label) {
      if (hasActiveGrowth()) label.textContent = growthPauseRequested ? '增量加载已暂停' : `并行增量加载中(${growthControllers.size})`;
      else if (playbackStep < 0) label.textContent = '完整图';
      else if (playbackStep === 0) label.textContent = '仅文档';
      else {
        const ep = graphModel.episodes[playbackStep - 1];
        label.textContent = `${playbackStep}/${total}: ${truncate(ep.heading_path || ep.name || 'Episode', 22)}`;
      }
    }
    if (playBtn) {
      const active = hasActiveGrowth() ? !growthPauseRequested : !!playbackTimer;
      const icon = hasActiveGrowth() ? (growthPauseRequested ? 'play' : 'pause') : (playbackTimer ? 'pause' : 'play');
      playBtn.innerHTML = `<i data-lucide="${icon}" style="width:11px;height:11px;"></i>`;
      playBtn.classList.toggle('active', active);
    }
    if (speedBtn) speedBtn.textContent = hasActiveGrowth() ? `${growthRatePerSecond}/s` : `${playbackSpeed}s`;
    if (window.lucide) lucide.createIcons({ nodes: [bar] });
  }

  function togglePlayback() {
    if (playbackTimer) stopPlayback();
    else startPlayback();
  }

  function startPlayback() {
    if (!graphModel?.episodes?.length) return;
    if (playbackStep < 0 || playbackStep >= graphModel.episodes.length) playbackStep = 0;
    updateGraphStep(playbackStep, { fitDelay: 60, fitDuration: 300 });
    playbackTimer = setInterval(() => {
      if (!graphModel || playbackStep >= graphModel.episodes.length) {
        stopPlayback();
        playbackStep = -1;
        updateGraphStep(-1, { fitDelay: 80, fitDuration: 360 });
        return;
      }
      playbackStep += 1;
      updateGraphStep(playbackStep, { fitDelay: 80, fitDuration: 360 });
    }, Math.max(120, playbackSpeed * 1000));
    updatePlaybackControls();
  }

  function stopPlayback() {
    if (playbackTimer) clearInterval(playbackTimer);
    playbackTimer = null;
    updatePlaybackControls();
  }

  function stepPlayback(direction) {
    if (!graphModel?.episodes?.length) return;
    stopPlayback();
    const total = graphModel.episodes.length;
    const current = playbackStep < 0 ? total : playbackStep;
    playbackStep = Math.max(0, Math.min(total, current + direction));
    updateGraphStep(playbackStep, { fitDelay: 80, fitDuration: 360 });
  }

  registerPage('graph', {
    async render(container) {
      selectedDocVersions = new Set();
      graphData = null;
      graphModel = null;
      focusFamilyId = null;
      playbackStep = -1;
      render(container);
      setDocsCollapsed(false);
      await loadDocs();
    },
    destroy() {
      stopPlayback();
      destroyNetwork();
      window.__documentGraphVisual = null;
    },
    getCommands() {
      return [
        { label: '刷新文档图谱', icon: 'refresh-cw', action: () => loadDocs() },
        { label: '播放 Episode 展示', icon: 'play', action: () => startPlayback() },
        { label: '适配图谱视图', icon: 'maximize', action: () => network?.fit({ animation: true }) },
      ];
    },
  });
})();

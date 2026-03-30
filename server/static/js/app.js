/* ==========================================
   TMG Web Dashboard - Core Application
   ========================================== */

// ---- API Client ----
/** 浏览器对 fetch 失败的消息不一致：Chrome 多为 Failed to fetch，Firefox 多为 NetworkError when attempting... */
function _isFetchNetworkFailure(err) {
  if (!err) return false;
  if (err.name === 'NetworkError') return true;
  if (err.name !== 'TypeError') return false;
  const m = String(err.message || '').toLowerCase();
  return (
    m === 'failed to fetch'
    || m.includes('failed to fetch')
    || m.includes('networkerror')
    || m.includes('load failed')
    || m.includes('network request failed')
  );
}

class TMGApi {
  constructor() {
    this.baseUrl = '';
  }

  async request(method, path, options = {}) {
    const url = this.baseUrl + path;
    const headers = { ...options.headers };

    if (options.json !== undefined) {
      headers['Content-Type'] = 'application/json';
      options.body = JSON.stringify(options.json);
    }

    try {
      const res = await fetch(url, {
        method,
        headers,
        body: options.body || null,
      });
      let data;
      const ct = (res.headers.get('content-type') || '').toLowerCase();
      if (ct.includes('application/json')) {
        data = await res.json();
      } else {
        const text = await res.text();
        try {
          data = text ? JSON.parse(text) : {};
        } catch {
          data = { error: text ? text.slice(0, 200) : `HTTP ${res.status}` };
        }
      }
      if (!res.ok) {
        throw new Error(data.error || `HTTP ${res.status}`);
      }
      return data;
    } catch (err) {
      if (_isFetchNetworkFailure(err)) {
        throw new Error(t('error.networkError'));
      }
      throw err;
    }
  }

  get(path) { return this.request('GET', path); }
  post(path, json) { return this.request('POST', path, { json }); }
  delete(path) { return this.request('DELETE', path); }
  postForm(path, formData) { return this.request('POST', path, { body: formData }); }

  // System
  health(graphId = 'default') {
    return this.get(`/api/v1/health?graph_id=${encodeURIComponent(graphId)}`);
  }
  systemOverview() { return this.get('/api/v1/system/overview'); }
  systemGraphs() { return this.get('/api/v1/system/graphs'); }
  systemGraphDetail(graphId) { return this.get(`/api/v1/system/graphs/${encodeURIComponent(graphId)}`); }
  systemTasks(limit = 50) { return this.get(`/api/v1/system/tasks?limit=${limit}`); }
  systemLogs(limit = 50, level, source) {
    let q = `limit=${limit}`;
    if (level) q += `&level=${encodeURIComponent(level)}`;
    if (source) q += `&source=${encodeURIComponent(source)}`;
    return this.get(`/api/v1/system/logs?${q}`);
  }
  systemAccessStats(sinceSeconds = 300) { return this.get(`/api/v1/system/access-stats?since_seconds=${sinceSeconds}`); }

  // Graphs
  listGraphs() { return this.get('/api/v1/graphs'); }
  findStats(graphId = 'default') {
    return this.get(`/api/v1/find/stats?graph_id=${encodeURIComponent(graphId)}`);
  }

  // Remember
  rememberText(graphId, text, options = {}) {
    const payload = {
      graph_id: graphId,
      text,
      source_name: options.source_name || '',
      event_time: options.event_time || '',
    };
    if (typeof options.load_cache === 'boolean') {
      payload.load_cache_memory = options.load_cache;
    }
    return this.post('/api/v1/remember', payload);
  }
  rememberFile(graphId, file, options = {}) {
    const fd = new FormData();
    fd.append('graph_id', graphId);
    fd.append('file', file);
    if (options.source_name) fd.append('source_name', options.source_name);
    if (options.event_time) fd.append('event_time', options.event_time);
    if (typeof options.load_cache === 'boolean') {
      fd.append('load_cache_memory', options.load_cache ? 'true' : 'false');
    }
    return this.postForm('/api/v1/remember', fd);
  }
  rememberTasks(graphId = 'default', limit = 50) {
    return this.get(`/api/v1/remember/tasks?graph_id=${encodeURIComponent(graphId)}&limit=${limit}`);
  }
  rememberStatus(taskId, graphId = 'default') {
    return this.get(`/api/v1/remember/tasks/${taskId}?graph_id=${encodeURIComponent(graphId)}`);
  }
  rememberDelete(taskId, graphId = 'default') {
    return this.delete(`/api/v1/remember/tasks/${taskId}?graph_id=${encodeURIComponent(graphId)}`);
  }
  rememberPause(taskId, graphId = 'default') {
    return this.post(`/api/v1/remember/tasks/${taskId}/pause?graph_id=${encodeURIComponent(graphId)}`, {});
  }
  rememberResume(taskId, graphId = 'default') {
    return this.post(`/api/v1/remember/tasks/${taskId}/resume?graph_id=${encodeURIComponent(graphId)}`, {});
  }

  // Find
  find(query, options = {}) {
    const body = {
      query,
      graph_id: options.graphId || 'default',
      similarity_threshold: options.threshold ?? 0.5,
      max_entities: options.maxEntities ?? 20,
      max_relations: options.maxRelations ?? 50,
      expand: options.expand ?? true,
    };
    if (options.searchMode) body.search_mode = options.searchMode;
    if (options.timeBefore) body.time_before = options.timeBefore;
    if (options.timeAfter) body.time_after = options.timeAfter;
    if (options.reranker) body.reranker = options.reranker;
    return this.post('/api/v1/find', body);
  }

  // Entities
  listEntities(graphId = 'default', limit) {
    let q = `graph_id=${encodeURIComponent(graphId)}`;
    if (limit) q += `&limit=${limit}`;
    return this.get(`/api/v1/find/entities?${q}`);
  }
  searchEntities(query, graphId = 'default', options = {}) {
    const body = {
      query_name: query,
      graph_id: graphId,
      query_content: options.queryContent || query,
      similarity_threshold: options.threshold ?? 0.7,
      max_results: options.maxResults ?? 20,
      text_mode: options.textMode || 'name_and_content',
      similarity_method: options.method || 'embedding',
    };
    if (options.searchMode) body.search_mode = options.searchMode;
    return this.post('/api/v1/find/entities/search', body);
  }
  entityVersions(entityId, graphId = 'default') {
    return this.get(`/api/v1/find/entities/${encodeURIComponent(entityId)}/versions?graph_id=${encodeURIComponent(graphId)}`);
  }
  entityRelations(entityId, graphId = 'default', options = {}) {
    let q = `graph_id=${encodeURIComponent(graphId)}`;
    if (options.limit) q += `&limit=${options.limit}`;
    if (options.maxVersionAbsoluteId) q += `&max_version_absolute_id=${encodeURIComponent(options.maxVersionAbsoluteId)}`;
    if (options.relationScope) q += `&relation_scope=${encodeURIComponent(options.relationScope)}`;
    return this.get(`/api/v1/find/entities/${encodeURIComponent(entityId)}/relations?${q}`);
  }
  entityVersionCounts(entityIds, graphId = 'default') {
    return this.post('/api/v1/find/entities/version-counts', {
      entity_ids: entityIds,
      graph_id: graphId,
    });
  }
  entityOneHop(absoluteId, graphId = 'default') {
    return this.get(`/api/v1/find/entities/absolute/${encodeURIComponent(absoluteId)}/relations?graph_id=${encodeURIComponent(graphId)}`);
  }
  entityByAbsoluteId(absoluteId, graphId = 'default') {
    return this.get(`/api/v1/find/entities/absolute/${encodeURIComponent(absoluteId)}?graph_id=${encodeURIComponent(graphId)}`);
  }

  // --- CRUD ---
  updateEntity(entityId, data) {
    return this.request(`/api/v1/find/entities/${encodeURIComponent(entityId)}`, {
      method: 'PUT',
      body: JSON.stringify(data),
    });
  }

  // --- Entity v3: Summary Evolution ---
  evolveEntitySummary(entityId) {
    return this.post(`/api/v1/find/entities/${encodeURIComponent(entityId)}/evolve-summary`, {});
  }

  // --- Entity v3: Contradictions ---
  entityContradictions(entityId) {
    return this.get(`/api/v1/find/entities/${encodeURIComponent(entityId)}/contradictions`);
  }

  resolveContradiction(entityId, data) {
    return this.post(`/api/v1/find/entities/${encodeURIComponent(entityId)}/resolve-contradiction`, data);
  }

  // --- Entity v3: Provenance ---
  entityProvenance(entityId) {
    return this.get(`/api/v1/find/entities/${encodeURIComponent(entityId)}/provenance`);
  }

  // --- Graph Traversal (Phase B) ---
  traverseGraph(seedEntityIds, maxDepth = 3, maxNodes = 100) {
    return this.post('/api/v1/find/traverse', {
      seed_entity_ids: seedEntityIds,
      max_depth: maxDepth,
      max_nodes: maxNodes,
    });
  }

  // --- Episodes v3: Batch Ingest ---
  batchIngestEpisodes(episodes) {
    return this.post('/api/v1/find/episodes/batch-ingest', { episodes });
  }

  deleteEntity(entityId, cascade = false) {
    return this.request(`/api/v1/find/entities/${encodeURIComponent(entityId)}?cascade=${cascade}`, {
      method: 'DELETE',
    });
  }

  batchDeleteEntities(entityIds, cascade = false) {
    return this.request('/api/v1/find/entities/batch-delete', {
      method: 'POST',
      body: JSON.stringify({ entity_ids: entityIds, cascade }),
    });
  }

  updateRelation(relationId, data) {
    return this.request(`/api/v1/find/relations/${encodeURIComponent(relationId)}`, {
      method: 'PUT',
      body: JSON.stringify(data),
    });
  }

  deleteRelation(relationId) {
    return this.request(`/api/v1/find/relations/${encodeURIComponent(relationId)}`, {
      method: 'DELETE',
    });
  }

  batchDeleteRelations(relationIds) {
    return this.request('/api/v1/find/relations/batch-delete', {
      method: 'POST',
      body: JSON.stringify({ relation_ids: relationIds }),
    });
  }

  mergeEntities(targetEntityId, sourceEntityIds) {
    return this.request('/api/v1/find/entities/merge', {
      method: 'POST',
      body: JSON.stringify({ target_entity_id: targetEntityId, source_entity_ids: sourceEntityIds }),
    });
  }

  // Relations
  relationVersions(relationId, graphId = 'default') {
    return this.get(`/api/v1/find/relations/${encodeURIComponent(relationId)}/versions?graph_id=${encodeURIComponent(graphId)}`);
  }
  relationByAbsoluteId(absoluteId, graphId = 'default') {
    return this.get(`/api/v1/find/relations/absolute/${encodeURIComponent(absoluteId)}?graph_id=${encodeURIComponent(graphId)}`);
  }

  // Relations
  listRelations(graphId = 'default', limit, offset) {
    let q = `graph_id=${encodeURIComponent(graphId)}`;
    if (limit) q += `&limit=${limit}`;
    if (offset) q += `&offset=${offset}`;
    return this.get(`/api/v1/find/relations?${q}`);
  }
  searchRelations(query, graphId = 'default', options = {}) {
    const body = {
      query_text: query,
      graph_id: graphId,
      similarity_threshold: options.threshold ?? 0.3,
      max_results: options.maxResults ?? 20,
    };
    if (options.searchMode) body.search_mode = options.searchMode;
    return this.post('/api/v1/find/relations/search', body);
  }
  relationsBetween(entityA, entityB, graphId = 'default') {
    return this.post('/api/v1/find/relations/between', {
      entity_id_a: entityA,
      entity_id_b: entityB,
      graph_id: graphId,
    });
  }
  shortestPaths(entityA, entityB, graphId = 'default', options = {}) {
    return this.post('/api/v1/find/paths/shortest', {
      entity_id_a: entityA,
      entity_id_b: entityB,
      graph_id: graphId,
      max_depth: options.maxDepth || 6,
      max_paths: options.maxPaths || 10,
    });
  }

  // Neo4j: Entity Neighbors
  entityNeighbors(entityUuid, graphId = 'default', depth = 1) {
    return this.get(`/api/v1/find/entities/${encodeURIComponent(entityUuid)}/neighbors?graph_id=${encodeURIComponent(graphId)}&depth=${depth}`);
  }

  // Neo4j: Cypher Shortest Path
  shortestPathCypher(entityA, entityB, graphId = 'default', maxDepth = 6) {
    return this.post('/api/v1/find/paths/shortest-cypher', {
      entity_id_a: entityA,
      entity_id_b: entityB,
      graph_id: graphId,
      max_depth: maxDepth,
    });
  }

  // Episodes
  listEpisodes(graphId = 'default', limit = 20, offset = 0) {
    return this.get(`/api/v1/episodes?graph_id=${encodeURIComponent(graphId)}&limit=${limit}&offset=${offset}`);
  }
  getEpisode(uuid, graphId = 'default') {
    return this.get(`/api/v1/episodes/${encodeURIComponent(uuid)}?graph_id=${encodeURIComponent(graphId)}`);
  }
  getEpisodeEntities(uuid, graphId = 'default') {
    return this.get(`/api/v1/episodes/${encodeURIComponent(uuid)}/entities?graph_id=${encodeURIComponent(graphId)}`);
  }
  searchEpisodes(query, graphId = 'default', limit = 20) {
    return this.post('/api/v1/episodes/search', { query, graph_id: graphId, limit });
  }
  deleteEpisode(uuid, graphId = 'default') {
    return this.delete(`/api/v1/episodes/${encodeURIComponent(uuid)}?graph_id=${encodeURIComponent(graphId)}`);
  }

  // Communities
  detectCommunities(graphId = 'default', algorithm = 'louvain', resolution = 1.0) {
    return this.post('/api/v1/communities/detect', { algorithm, resolution, graph_id: graphId });
  }
  listCommunities(graphId = 'default', minSize = 3, limit = 50) {
    return this.get(`/api/v1/communities?graph_id=${encodeURIComponent(graphId)}&min_size=${minSize}&limit=${limit}`);
  }
  getCommunity(cid, graphId = 'default') {
    return this.get(`/api/v1/communities/${encodeURIComponent(cid)}?graph_id=${encodeURIComponent(graphId)}`);
  }
  getCommunityGraph(cid, graphId = 'default') {
    return this.get(`/api/v1/communities/${encodeURIComponent(cid)}/graph?graph_id=${encodeURIComponent(graphId)}`);
  }
  clearCommunities(graphId = 'default') {
    return this.delete(`/api/v1/communities?graph_id=${encodeURIComponent(graphId)}`);
  }

  // --- Time Travel ---
  getSnapshot(time) {
    return this.request(`/api/v1/find/snapshot?time=${encodeURIComponent(time)}`);
  }

  getChanges(since, until) {
    let url = `/api/v1/find/changes?since=${encodeURIComponent(since)}`;
    if (until) url += `&until=${encodeURIComponent(until)}`;
    return this.request(url);
  }

  invalidateRelation(relationId, reason = '') {
    return this.request(`/api/v1/find/relations/${encodeURIComponent(relationId)}/invalidate`, {
      method: 'POST',
      body: JSON.stringify({ reason }),
    });
  }

  getInvalidatedRelations(limit = 100) {
    return this.request(`/api/v1/find/relations/invalidated?limit=${limit}`);
  }

  // --- Stats & Timeline ---
  getGraphStats() {
    return this.request('/api/v1/find/graph-stats');
  }

  getEntityTimeline(entityId) {
    return this.request(`/api/v1/find/entities/${encodeURIComponent(entityId)}/timeline`);
  }

  // Docs
  listDocs(graphId = 'default') {
    return this.get(`/api/v1/docs?graph_id=${encodeURIComponent(graphId)}`);
  }
  getDocContent(filename, graphId = 'default') {
    return this.get(`/api/v1/docs/${encodeURIComponent(filename)}?graph_id=${encodeURIComponent(graphId)}`);
  }

  memoryCacheDoc(cacheId) {
    return this.get(`/api/v1/find/memory-caches/${encodeURIComponent(cacheId)}/doc`);
  }

  // --- DeepDream (Phase E) ---
  startDream(options = {}) {
    return this.post('/api/v1/find/dream/start', {
      review_window_days: options.reviewWindowDays ?? 30,
      max_entities_per_cycle: options.maxEntitiesPerCycle ?? 100,
      similarity_threshold: options.similarityThreshold ?? 0.8,
    });
  }

  dreamStatus() {
    return this.get('/api/v1/find/dream/status');
  }

  dreamLogs(limit = 20) {
    return this.get(`/api/v1/find/dream/logs?limit=${limit}`);
  }

  dreamLogDetail(cycleId) {
    return this.get(`/api/v1/find/dream/logs/${encodeURIComponent(cycleId)}`);
  }

  // --- Agent Meta Query (Phase F) ---
  agentAsk(question) {
    return this.post('/api/v1/find/ask', { question });
  }

  explainEntity(entityId, aspect) {
    return this.post('/api/v1/find/explain', { entity_id: entityId, aspect });
  }

  smartSuggestions() {
    return this.get('/api/v1/find/suggestions');
  }

  // System
  systemOverview() {
    return this.get('/api/v1/system/overview');
  }
  systemGraphs() {
    return this.get('/api/v1/system/graphs');
  }
  systemTasks(limit = 50) {
    return this.get(`/api/v1/system/tasks?limit=${limit}`);
  }
  systemLogs(limit = 100, level) {
    let q = `limit=${limit}`;
    if (level) q += `&level=${encodeURIComponent(level)}`;
    return this.get(`/api/v1/system/logs?${q}`);
  }
  systemAccessStats(since = 300) {
    return this.get(`/api/v1/system/access-stats?since_seconds=${since}`);
  }
  systemDashboard(opts = {}) {
    let q = `task_limit=${opts.taskLimit || 50}&log_limit=${opts.logLimit || 100}`;
    if (opts.logLevel) q += `&log_level=${encodeURIComponent(opts.logLevel)}`;
    if (opts.logSource) q += `&log_source=${encodeURIComponent(opts.logSource)}`;
    if (opts.accessSince) q += `&access_since=${opts.accessSince}`;
    return this.get(`/api/v1/system/dashboard?${q}`);
  }
}

// ---- Global State ----
const state = {
  api: new TMGApi(),
  currentGraphId: localStorage.getItem('tmg_graph_id') || 'default',
  refreshTimers: {},
  currentPage: null,
  backendType: 'sqlite',
};

function isNeo4j() {
  return state.backendType === 'neo4j';
}

function setGraphId(id) {
  state.currentGraphId = id;
  localStorage.setItem('tmg_graph_id', id);
  const sel = document.getElementById('graph-selector');
  if (sel) sel.value = id;
  // Re-render current page if it has a graph-aware render
  if (typeof window.onGraphChange === 'function') {
    window.onGraphChange(id);
  }
}

// ---- Utilities ----
function formatDate(isoStr) {
  if (!isoStr) return '-';
  try {
    const d = new Date(isoStr);
    return d.toLocaleString('zh-CN', {
      year: 'numeric', month: '2-digit', day: '2-digit',
      hour: '2-digit', minute: '2-digit', second: '2-digit',
    });
  } catch { return isoStr; }
}

function formatRelativeTime(seconds) {
  if (seconds == null) return '-';
  seconds = Math.max(0, Math.round(seconds));
  if (seconds < 60) return `${seconds}s`;
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  if (m < 60) return `${m}m${String(s).padStart(2, '0')}s`;
  const h = Math.floor(m / 60);
  const rm = m % 60;
  return `${h}h${String(rm).padStart(2, '0')}m`;
}

function formatNumber(n) {
  if (n == null) return '0';
  return n.toLocaleString();
}

function debounce(fn, ms = 300) {
  let timer;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), ms);
  };
}

function escapeHtml(str) {
  if (!str) return '';
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

function truncate(str, maxLen = 80) {
  if (!str) return '';
  return str.length > maxLen ? str.slice(0, maxLen) + '...' : str;
}

function statusBadge(status) {
  const map = {
    queued: 'badge-warning',
    running: 'badge-info',
    paused: 'badge-warning',
    completed: 'badge-success',
    failed: 'badge-error',
  };
  return `<span class="badge ${map[status] || 'badge-primary'}">${escapeHtml(status)}</span>`;
}

function progressBar(pct, cls = '') {
  const w = Math.min(100, Math.max(0, (pct || 0) * 100));
  return `<div class="progress-bar"><div class="progress-bar-fill ${cls}" style="width:${w.toFixed(1)}%"></div></div>`;
}

function spinnerHtml(cls = '') {
  return `<div class="spinner ${cls}"></div>`;
}

function emptyState(text, icon = 'inbox') {
  return `<div class="empty-state"><i data-lucide="${icon}"></i><p>${escapeHtml(text)}</p></div>`;
}

// ---- Router ----
const pages = {};
const pageTitles = {
  dashboard: t('nav.dashboard'),
  graph: t('nav.graph'),
  memory: t('nav.memory'),
  search: t('nav.search'),
  entities: t('nav.entities'),
  relations: t('nav.relations'),
  episodes: t('nav.episodes'),
  communities: t('nav.communities'),
  dream: t('nav.dream'),
  'api-test': t('nav.apiTest'),
};

function registerPage(name, module) {
  pages[name] = module;
}

function navigate(hash) {
  window.location.hash = hash;
}

async function handleRoute() {
  const hash = (window.location.hash || '#dashboard').slice(1);
  const [page, ...params] = hash.split('/').filter(Boolean);
  const pageName = page || 'dashboard';
  const pageModule = pages[pageName];

  // Clear refresh timers
  Object.values(state.refreshTimers).forEach(t => clearInterval(t));
  state.refreshTimers = {};

  // Update sidebar active state
  document.querySelectorAll('.sidebar-link').forEach(link => {
    link.classList.toggle('active', link.getAttribute('data-page') === pageName);
  });

  const container = document.getElementById('page-content');

  if (!pageModule) {
    container.innerHTML = `<div class="page-enter">${emptyState(t('common.pageNotFound'))}</div>`;
    return;
  }

  // Set title
  document.getElementById('page-title').textContent = pageTitles[pageName] || pageName;

  // Call destroy if exists
  if (state.currentPage && pages[state.currentPage] && pages[state.currentPage].destroy) {
    pages[state.currentPage].destroy();
  }

  state.currentPage = pageName;
  container.innerHTML = `<div class="page-enter"><div class="flex items-center justify-center p-12">${spinnerHtml()}</div></div>`;

  try {
    await pageModule.render(container, params);
  } catch (err) {
    console.error(`Error rendering page ${pageName}:`, err);
    container.innerHTML = `<div class="page-enter"><div class="empty-state"><i data-lucide="alert-triangle"></i><p>${t('common.pageLoadError')}: ${escapeHtml(err.message)}</p><button class="btn btn-secondary mt-3" onclick="handleRoute()">${t('common.retry')}</button></div></div>`;
  }

  // Re-render lucide icons
  if (window.lucide) lucide.createIcons();
}

// ---- Graph selector (top bar) ----
async function loadGraphSelector() {
  try {
    const res = await state.api.listGraphs();
    const graphs = res.data?.graphs || [];
    const sel = document.getElementById('graph-selector');
    if (!sel) return;

    const currentVal = state.currentGraphId;
    sel.innerHTML = graphs.map(g =>
      `<option value="${escapeHtml(g)}" ${g === currentVal ? 'selected' : ''}>${escapeHtml(g)}</option>`
    ).join('');

    if (!graphs.includes(currentVal)) {
      sel.innerHTML = `<option value="${escapeHtml(currentVal)}" selected>${escapeHtml(currentVal)}</option>` + sel.innerHTML;
    }
  } catch { /* ignore */ }
}

// ---- Theme ----
function toggleTheme() {
  const html = document.documentElement;
  const isDark = html.getAttribute('data-theme') !== 'light';
  const newTheme = isDark ? 'light' : 'dark';
  html.setAttribute('data-theme', newTheme);
  localStorage.setItem('tmg_theme', newTheme);
  updateThemeIcon(newTheme);
}

function updateThemeIcon(theme) {
  const darkIcon = document.getElementById('theme-icon-dark');
  const lightIcon = document.getElementById('theme-icon-light');
  if (darkIcon) darkIcon.style.display = theme === 'dark' ? '' : 'none';
  if (lightIcon) lightIcon.style.display = theme === 'light' ? '' : 'none';
}

function initTheme() {
  const saved = localStorage.getItem('tmg_theme') || 'dark';
  document.documentElement.setAttribute('data-theme', saved);
  updateThemeIcon(saved);
}

// ---- Init ----
document.addEventListener('DOMContentLoaded', async () => {
  initTheme();
  window.I18N.init();

  // Theme toggle
  const themeBtn = document.getElementById('theme-toggle');
  if (themeBtn) {
    themeBtn.addEventListener('click', toggleTheme);
  }
  // Setup graph selector
  const sel = document.getElementById('graph-selector');
  if (sel) {
    sel.value = state.currentGraphId;
    sel.addEventListener('change', () => setGraphId(sel.value));
  }

  // Setup hash routing
  window.addEventListener('hashchange', handleRoute);
  handleRoute();

  // Load graphs
  loadGraphSelector();

  // Detect backend type
  try {
    const h = await state.api.health(state.currentGraphId);
    if (h.data?.storage_backend) {
      state.backendType = h.data.storage_backend;
    }
  } catch { /* ignore */ }

  // Show/hide Neo4j-only nav items
  const neo4jNavItems = ['nav-episodes', 'nav-communities', 'nav-dream'];
  neo4jNavItems.forEach(id => {
    const el = document.getElementById(id);
    if (el) el.style.display = isNeo4j() ? '' : 'none';
  });

  // Mobile sidebar toggle
  const toggle = document.getElementById('sidebar-toggle');
  const sidebar = document.getElementById('sidebar');
  if (toggle && sidebar) {
    toggle.addEventListener('click', () => sidebar.classList.toggle('open'));
    document.addEventListener('click', (e) => {
      if (!sidebar.contains(e.target) && !toggle.contains(e.target)) {
        sidebar.classList.remove('open');
      }
    });
  }
});

// ---- Global: Show document content modal ----
window.showDocContent = async function(cacheId) {
  if (!cacheId) return;
  try {
    const res = await state.api.memoryCacheDoc(cacheId);
    const data = res.data || {};
    const meta = data.meta || {};

    const sourceName = meta.source_document || meta.doc_name || cacheId;
    const eventTime = meta.event_time || '-';
    const original = data.original || '';
    const cache = data.cache || '';

    let body = `
      <div style="display:flex;flex-direction:column;gap:1rem;">
        <div style="display:grid;grid-template-columns:auto 1fr;gap:0.25rem 0.75rem;font-size:0.85rem;">
          <span style="color:var(--text-secondary);">${t('memory.taskSource')}:</span><span>${escapeHtml(sourceName)}</span>
          <span style="color:var(--text-secondary);">${t('memory.docTime')}:</span><span>${formatDate(eventTime)}</span>
        </div>
    `;

    if (cache) {
      body += `
        <div>
          <h4 style="margin-bottom:0.5rem;">${t('memory.cacheSummary')}</h4>
          <div style="max-height:400px;overflow-y:auto;background:var(--bg-secondary);padding:0.75rem;border-radius:0.5rem;font-size:0.85rem;line-height:1.6;white-space:pre-wrap;word-break:break-word;">${escapeHtml(cache)}</div>
        </div>
      `;
    }

    if (original) {
      body += `
        <div>
          <h4 style="margin-bottom:0.5rem;">${t('memory.originalText')}</h4>
          <div style="max-height:400px;overflow-y:auto;background:var(--bg-secondary);padding:0.75rem;border-radius:0.5rem;font-size:0.85rem;line-height:1.6;white-space:pre-wrap;word-break:break-word;">${escapeHtml(original)}</div>
        </div>
      `;
    }

    body += '</div>';

    showModal({
      title: t('memory.docContent') + ' - ' + escapeHtml(truncate(sourceName, 30)),
      content: body,
      size: 'lg',
    });
  } catch (err) {
    showToast(t('memory.loadDocContentFailed') + ': ' + err.message, 'error');
  }
};

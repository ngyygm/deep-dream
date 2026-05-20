/* ==========================================
   DeepDream Dashboard - Core Application

   Shared modules (loaded before this file):
     shared/format.js   → window.Format
     shared/ui-utils.js → window.UIUtils
   ========================================== */

// ---- Re-export shared utilities as globals for backward compatibility ----
var getLocale = Format.getLocale;
var formatDate = Format.formatDate;
var formatDateMs = Format.formatDateMs;
var formatRelativeTime = Format.formatRelativeTime;
var formatNumber = Format.formatNumber;
var getElapsed = Format.getElapsed;
var truncate = Format.truncate;
var escapeHtml = Format.escapeHtml;
var escapeAttr = Format.escapeAttr;
var tripleProgressBar = UIUtils.tripleProgressBar;
var progressBar = UIUtils.progressBar;
var spinnerHtml = UIUtils.spinnerHtml;
var emptyState = UIUtils.emptyState;
var statusBadge = UIUtils.statusBadge;
var renderVersionTimeline = UIUtils.renderVersionTimeline;
var bindClickableRows = UIUtils.bindClickableRows;

// ---- API Client ----
function _isFetchNetworkFailure(err) {
  if (!err) return false;
  if (err.name === 'NetworkError') return true;
  if (err.name !== 'TypeError') return false;
  var m = String(err.message || '').toLowerCase();
  return (
    m === 'failed to fetch'
    || m.includes('failed to fetch')
    || m.includes('networkerror')
    || m.includes('load failed')
    || m.includes('network request failed')
  );
}

class DeepDreamApi {
  constructor() {
    this.baseUrl = '';
    this._inflight = new Map();
  }

  async request(method, path, options = {}) {
    const url = this.baseUrl + path;
    const headers = { ...options.headers };

    if (options.json !== undefined) {
      headers['Content-Type'] = 'application/json';
      options.body = JSON.stringify(options.json);
    }

    const dedupeKey = `${method}:${url}:${options.body || ''}`;
    if (this._inflight.has(dedupeKey)) return this._inflight.get(dedupeKey);

    const promise = this._doRequest(method, url, headers, options);
    this._inflight.set(dedupeKey, promise);
    try { return await promise; }
    finally { this._inflight.delete(dedupeKey); }
  }

  async _doRequest(method, url, headers, options) {
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

  // Graphs
  listGraphs() { return this.get('/api/v1/graphs'); }
  createGraph(graphId) { return this.post('/api/v1/graphs', { graph_id: graphId }); }
  deleteGraph(graphId) { return this.delete(`/api/v1/graphs/${encodeURIComponent(graphId)}`); }
  clearGraph(graphId) { return this.post(`/api/v1/graphs/${encodeURIComponent(graphId)}/clear`, {}); }
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

  // Concept compatibility helpers used by the existing graph/search UI.
  listEntities(graphId = 'default', limit, offset) {
    let q = `graph_id=${encodeURIComponent(graphId)}&role=entity`;
    if (limit) q += `&limit=${limit}`;
    if (offset) q += `&offset=${offset}`;
    return this.get(`/api/v1/concepts?${q}`).then(res => ({
      ...res,
      data: { ...(res.data || {}), entities: res.data?.concepts || [] },
    }));
  }
  getCounts(graphId = 'default') {
    return this.get(`/api/v1/stats/counts?graph_id=${encodeURIComponent(graphId)}`);
  }
  searchEntities(query, graphId = 'default', options = {}) {
    const body = {
      query,
      graph_id: graphId,
      role: 'entity',
      threshold: options.threshold ?? 0.7,
      limit: options.maxResults ?? 20,
    };
    if (options.searchMode) body.search_mode = options.searchMode;
    return this.post('/api/v1/concepts/search', body).then(res => ({
      ...res,
      data: { ...(res.data || {}), entities: res.data?.concepts || [] },
    }));
  }
  entityVersions(familyId, graphId = 'default') {
    return this.get(`/api/v1/concepts/${encodeURIComponent(familyId)}/versions?graph_id=${encodeURIComponent(graphId)}`);
  }
  entityVersionDiff(familyId, v1, v2, graphId = 'default') {
    return Promise.reject(new Error('Version diff is not available in the concept graph UI'));
  }
  entityRelations(familyId, graphId = 'default', options = {}) {
    let q = `graph_id=${encodeURIComponent(graphId)}&max_depth=1`;
    if (options.timePoint) q += `&time_point=${encodeURIComponent(options.timePoint)}`;
    return this.get(`/api/v1/concepts/${encodeURIComponent(familyId)}/neighbors?${q}`);
  }
  entityVersionCounts(familyIds, graphId = 'default') {
    return Promise.resolve({ success: true, data: { counts: Object.fromEntries((familyIds || []).map(id => [id, 0])) } });
  }
  entityOneHop(absoluteId, graphId = 'default') {
    return this.entityRelations(absoluteId, graphId);
  }
  entityByAbsoluteId(absoluteId, graphId = 'default') {
    return this.get(`/api/v1/concepts/${encodeURIComponent(absoluteId)}?graph_id=${encodeURIComponent(graphId)}`);
  }

  entityEmbeddingPreview(absoluteId, numValues = 5, graphId = 'default') {
    return Promise.resolve({ success: true, data: { values: [] } });
  }

  updateEntity(familyId, data, graphId = 'default') {
    return Promise.reject(new Error('Concept editing is not available in v1'));
  }

  evolveEntitySummary(familyId, graphId = 'default') {
    return Promise.reject(new Error('Summary evolution is not available in v1'));
  }

  entityContradictions(familyId, graphId = 'default') {
    return Promise.resolve({ success: true, data: { contradictions: [] } });
  }

  resolveContradiction(familyId, data, graphId = 'default') {
    return Promise.reject(new Error('Contradiction resolution is not available in v1'));
  }

  entityProvenance(familyId, graphId = 'default') {
    return this.get(`/api/v1/concepts/${encodeURIComponent(familyId)}/provenance?graph_id=${encodeURIComponent(graphId)}`);
  }

  traverseGraph(seedFamilyIds, maxDepth = 3, maxNodes = 100, graphId = 'default') {
    return this.post('/api/v1/traverse', {
      start_family_ids: seedFamilyIds,
      max_depth: maxDepth,
      max_nodes: maxNodes,
      graph_id: graphId,
    });
  }

  batchIngestEpisodes(episodes, graphId = 'default') {
    return Promise.reject(new Error('Episode batch ingest is not available in v1'));
  }

  deleteEntity(familyId, cascade = false, graphId = 'default') {
    return Promise.reject(new Error('Concept deletion is not available in v1'));
  }

  batchDeleteEntities(familyIds, cascade = false, graphId = 'default') {
    return Promise.reject(new Error('Concept deletion is not available in v1'));
  }

  updateRelation(familyId, data, graphId = 'default') {
    return Promise.reject(new Error('Concept editing is not available in v1'));
  }

  deleteRelation(familyId, graphId = 'default') {
    return Promise.reject(new Error('Concept deletion is not available in v1'));
  }

  batchDeleteRelations(familyIds, graphId = 'default') {
    return Promise.reject(new Error('Concept deletion is not available in v1'));
  }

  mergeEntities(targetFamilyId, sourceFamilyIds, graphId = 'default') {
    return Promise.reject(new Error('Concept merge is not available in v1'));
  }

  relationVersions(familyId, graphId = 'default') {
    return this.get(`/api/v1/concepts/${encodeURIComponent(familyId)}/versions?graph_id=${encodeURIComponent(graphId)}`);
  }
  relationByAbsoluteId(absoluteId, graphId = 'default') {
    return this.get(`/api/v1/concepts/${encodeURIComponent(absoluteId)}?graph_id=${encodeURIComponent(graphId)}`);
  }

  relationEmbeddingPreview(absoluteId, numValues = 5, graphId = 'default') {
    return Promise.resolve({ success: true, data: { values: [] } });
  }

  listRelations(graphId = 'default', limit, offset) {
    let q = `graph_id=${encodeURIComponent(graphId)}&role=relation`;
    if (limit) q += `&limit=${limit}`;
    if (offset) q += `&offset=${offset}`;
    return this.get(`/api/v1/concepts?${q}`).then(res => ({
      ...res,
      data: { ...(res.data || {}), relations: res.data?.concepts || [] },
    }));
  }
  searchRelations(query, graphId = 'default', options = {}) {
    const body = {
      query,
      graph_id: graphId,
      role: 'relation',
      threshold: options.threshold ?? 0.3,
      limit: options.maxResults ?? 20,
    };
    if (options.searchMode) body.search_mode = options.searchMode;
    return this.post('/api/v1/concepts/search', body).then(res => ({
      ...res,
      data: { ...(res.data || {}), relations: res.data?.concepts || [] },
    }));
  }
  relationsBetween(entityA, entityB, graphId = 'default') {
    return this.traverseGraph([entityA, entityB], 1, 100, graphId);
  }
  shortestPaths(entityA, entityB, graphId = 'default', options = {}) {
    return this.traverseGraph([entityA, entityB], options.maxDepth || 6, 200, graphId);
  }

  entityNeighbors(entityUuid, graphId = 'default', depth = 1) {
    return this.get(`/api/v1/concepts/${encodeURIComponent(entityUuid)}/neighbors?graph_id=${encodeURIComponent(graphId)}&max_depth=${depth}`);
  }

  shortestPathCypher(entityA, entityB, graphId = 'default', maxDepth = 6) {
    return this.traverseGraph([entityA, entityB], maxDepth, 200, graphId);
  }

  listEpisodes(graphId = 'default', limit = 20, offset = 0) {
    return this.get(`/api/v1/concepts?graph_id=${encodeURIComponent(graphId)}&role=episode&limit=${limit}&offset=${offset}`).then(res => ({
      ...res,
      data: { ...(res.data || {}), episodes: res.data?.concepts || [] },
    }));
  }
  getEpisode(uuid, graphId = 'default') {
    return this.get(`/api/v1/concepts/${encodeURIComponent(uuid)}?graph_id=${encodeURIComponent(graphId)}`);
  }
  getEpisodeEntities(uuid, graphId = 'default') {
    return this.get(`/api/v1/concepts/${encodeURIComponent(uuid)}/neighbors?graph_id=${encodeURIComponent(graphId)}&max_depth=1`);
  }
  searchEpisodes(query, graphId = 'default', limit = 20) {
    return this.post('/api/v1/concepts/search', { query, role: 'episode', graph_id: graphId, limit });
  }
  deleteEpisode(uuid, graphId = 'default') {
    return Promise.reject(new Error('Episode deletion is not available in v1'));
  }

  detectCommunities(graphId = 'default', algorithm = 'louvain', resolution = 1.0) {
    return this.post('/api/v1/communities/detect', { algorithm, resolution, graph_id: graphId });
  }
  listCommunities(graphId = 'default', minSize = 3, limit = 50, offset = 0) {
    return this.get(`/api/v1/communities?graph_id=${encodeURIComponent(graphId)}&min_size=${minSize}&limit=${limit}&offset=${offset}`);
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

  getSnapshot(time, graphId = 'default') {
    return this.get(`/api/v1/concepts?time_point=${encodeURIComponent(time)}&graph_id=${encodeURIComponent(graphId)}&limit=200`);
  }

  getChanges(since, until, graphId = 'default') {
    return Promise.resolve({ success: true, data: { changes: [], since, until, graph_id: graphId } });
  }

  invalidateRelation(familyId, reason = '', graphId = 'default') {
    return Promise.reject(new Error('Concept invalidation is not available in v1'));
  }

  getInvalidatedRelations(limit = 100, graphId = 'default') {
    return Promise.resolve({ success: true, data: { relations: [] } });
  }

  getGraphStats(graphId = 'default') {
    return this.get(`/api/v1/stats/counts?graph_id=${encodeURIComponent(graphId)}`);
  }

  getEntityTimeline(familyId, graphId = 'default') {
    return this.get(`/api/v1/concepts/${encodeURIComponent(familyId)}/versions?graph_id=${encodeURIComponent(graphId)}`);
  }

  listDocs(graphId = 'default') {
    return this.get(`/api/v1/documents?graph_id=${encodeURIComponent(graphId)}`).then(res => ({
      ...res,
      data: { ...(res.data || {}), docs: res.data?.documents || [] },
    }));
  }
  documentGraph(graphId = 'default', options = {}) {
    return this.post('/api/v1/documents/graph', {
      graph_id: graphId,
      document_version_ids: options.documentVersionIds || [],
      document_family_ids: options.documentFamilyIds || [],
      include_relations: options.includeRelations !== false,
      include_versions: options.includeVersions !== false,
      max_episodes: options.maxEpisodes || 5000,
      max_concepts: options.maxConcepts || 20000,
    });
  }
  documentGraphOutline(graphId = 'default', options = {}) {
    return this.post('/api/v1/documents/graph/outline', {
      graph_id: graphId,
      document_version_ids: options.documentVersionIds || [],
      document_family_ids: options.documentFamilyIds || [],
      max_episodes: options.maxEpisodes || 10000,
    });
  }
  documentGraphChunk(graphId = 'default', options = {}) {
    return this.post('/api/v1/documents/graph/chunk', {
      graph_id: graphId,
      document_version_ids: options.documentVersionIds || [],
      document_family_ids: options.documentFamilyIds || [],
      cursor: options.cursor || 0,
      limit: options.limit || 12,
      include_relations: options.includeRelations !== false,
      include_versions: options.includeVersions !== false,
      max_concepts: options.maxConcepts || 8000,
    });
  }
  documentContent(documentVersionId, graphId = 'default', options = {}) {
    const offset = options.offset || 0;
    const limit = options.limit || 20000;
    return this.get(`/api/v1/documents/${encodeURIComponent(documentVersionId)}/content?graph_id=${encodeURIComponent(graphId)}&offset=${encodeURIComponent(offset)}&limit=${encodeURIComponent(limit)}`);
  }
  getDocContent(filename, graphId = 'default') {
    return this.get(`/api/v1/documents?graph_id=${encodeURIComponent(graphId)}&source=${encodeURIComponent(filename)}`);
  }

  agentAsk(question, graphId = 'default') {
    return Promise.reject(new Error('Ask is not available in the concept graph API'));
  }

  explainEntity(familyId, aspect, graphId = 'default') {
    return this.get(`/api/v1/concepts/${encodeURIComponent(familyId)}/provenance?graph_id=${encodeURIComponent(graphId)}`);
  }

  smartSuggestions(graphId = 'default') {
    return Promise.resolve({ success: true, data: { suggestions: [] } });
  }

  systemOverview() { return this.get('/api/v1/system/overview'); }
  systemGraphs() { return this.get('/api/v1/system/graphs'); }
  systemTasks(limit = 50) { return this.get(`/api/v1/system/tasks?limit=${limit}`); }
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

  qualityReport(graphId = 'default') {
    return Promise.resolve({ success: true, data: { issues: [], warnings: [] } });
  }
  maintenanceHealth(graphId = 'default') {
    return this.health(graphId);
  }
  maintenanceCleanup(dryRun = false, graphId = 'default') {
    return Promise.resolve({ success: true, data: { dry_run: dryRun, cleaned: 0 } });
  }
  graphSummary(graphId = 'default') {
    return this.getGraphStats(graphId);
  }

  quickSearch(query, options = {}) {
    return this.post('/api/v1/concepts/search', {
      query,
      graph_id: options.graphId || 'default',
      threshold: options.threshold ?? 0.4,
      limit: options.maxResults ?? options.maxEntities ?? 10,
    });
  }

  findEntityByName(name, options = {}) {
    return this.post('/api/v1/concepts/search', {
      query: name,
      role: 'entity',
      graph_id: options.graphId || 'default',
      threshold: options.threshold || 0.7,
      limit: options.limit || 5,
    });
  }

  createEntity(data, graphId = 'default') {
    return Promise.reject(new Error('Manual concept creation is not available in v1'));
  }

  createRelation(data, graphId = 'default') {
    return Promise.reject(new Error('Manual relation creation is not available in v1'));
  }

  recentActivity(graphId = 'default', limit = 10) {
    return this.get(`/api/v1/concepts?graph_id=${encodeURIComponent(graphId)}&limit=${limit}`);
  }

  refreshGraphEdges(graphId = 'default') {
    return Promise.resolve({ success: true, data: { refreshed: 0, graph_id: graphId } });
  }

  entityProfile(familyId, graphId = 'default') {
    return Promise.all([
      this.get(`/api/v1/concepts/${encodeURIComponent(familyId)}?graph_id=${encodeURIComponent(graphId)}`),
      this.get(`/api/v1/concepts/${encodeURIComponent(familyId)}/neighbors?graph_id=${encodeURIComponent(graphId)}&max_depth=1`),
    ]).then(([concept, neighbors]) => ({
      success: true,
      data: {
        entity: concept.data,
        relations: neighbors.data?.neighbors || [],
        relation_count: (neighbors.data?.neighbors || []).length,
      },
    }));
  }

  searchConcepts(query, options = {}) {
    return this.post('/api/v1/concepts/search', {
      query,
      graph_id: options.graphId || 'default',
      limit: options.limit || 20,
      role: options.role || '',
    });
  }

  listConcepts(graphId = 'default', options = {}) {
    let q = `graph_id=${encodeURIComponent(graphId)}`;
    if (options.limit) q += `&limit=${options.limit}`;
    if (options.offset) q += `&offset=${options.offset}`;
    if (options.role) q += `&role=${encodeURIComponent(options.role)}`;
    return this.get(`/api/v1/concepts?${q}`);
  }

  batchProfiles(familyIds, graphId = 'default') {
    return Promise.reject(new Error('Batch profiles are not available in v1'));
  }
}

// ---- Global State ----
function getUrlGraphId() {
  try {
    const searchGraph = new URLSearchParams(window.location.search).get('graph_id');
    if (searchGraph) return searchGraph.trim();
    const hash = window.location.hash || '';
    const queryIndex = hash.indexOf('?');
    if (queryIndex >= 0) {
      const hashGraph = new URLSearchParams(hash.slice(queryIndex + 1)).get('graph_id');
      if (hashGraph) return hashGraph.trim();
    }
  } catch {}
  return '';
}

const state = {
  api: new DeepDreamApi(),
  currentGraphId: getUrlGraphId() || localStorage.getItem('deepdream_graph_id') || localStorage.getItem('tmg_graph_id') || 'default',
  refreshTimers: {},
  currentPage: null,
  backendType: 'sqlite',
  events: new EventTarget(),
};

function isNeo4j() {
  return state.backendType === 'neo4j';
}

function setGraphId(id) {
  state.currentGraphId = id;
  localStorage.setItem('deepdream_graph_id', id);
  try {
    const url = new URL(window.location.href);
    url.searchParams.set('graph_id', id);
    window.history.replaceState(null, '', `${url.pathname}${url.search}${url.hash}`);
  } catch {}
  const sel = document.getElementById('graph-selector');
  if (sel) sel.value = id;
  state.api.health(id).then(h => {
    if (h.data?.storage_backend) state.backendType = h.data.storage_backend;
  }).catch(() => {});
  handleRoute();
}

function debounce(fn, ms = 300) {
  let timer;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), ms);
  };
}

function renderMarkdown(text) {
  if (!text) return '';
  if (typeof marked === 'undefined') return escapeHtml(text);
  try {
    marked.setOptions({ breaks: true, gfm: true });
    var html = marked.parse(text);
    html = html.replace(/<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi, '');
    return html;
  } catch (e) {
    return escapeHtml(text);
  }
}

// ---- Router ----
const pages = {};
const _pageTitleKeys = {
  dashboard: 'nav.dashboard',
  memory: 'nav.memory', graph: 'nav.graph', search: 'nav.search', communities: 'nav.communities',
  'api-test': 'nav.apiTest',
};

const _pageSloganKeys = {
  dashboard: 'slogan.dashboard',
  memory: 'slogan.memory', graph: 'slogan.graph', search: 'slogan.search', communities: 'slogan.communities',
  'api-test': 'slogan.apiTest',
};

function registerPage(name, module) { pages[name] = module; }
function navigate(hash) { window.location.hash = hash; }

async function handleRoute() {
  const rawHash = (window.location.hash || '#dashboard').slice(1);
  const [hash, hashQuery = ''] = rawHash.split('?');
  try {
    const hashGraph = new URLSearchParams(hashQuery).get('graph_id');
    if (hashGraph && hashGraph !== state.currentGraphId) {
      state.currentGraphId = hashGraph;
      localStorage.setItem('deepdream_graph_id', hashGraph);
      const sel = document.getElementById('graph-selector');
      if (sel) sel.value = hashGraph;
    }
  } catch {}
  const [page, ...params] = hash.split('/').filter(Boolean);
  let pageName = page || 'dashboard';
  if (pageName === 'chat') {
    window.location.hash = '#dashboard';
    return;
  }
  const pageModule = pages[pageName];

  Object.values(state.refreshTimers).forEach(t => clearInterval(t));
  state.refreshTimers = {};

  document.querySelectorAll('.sidebar-link').forEach(link => {
    link.classList.toggle('active', link.getAttribute('data-page') === pageName);
  });

  const container = document.getElementById('page-content');
  if (!pageModule) { container.innerHTML = `<div class="page-enter">${emptyState(t('common.pageNotFound'))}</div>`; return; }

  const pageTitle = t(_pageTitleKeys[pageName] || '') || pageName;
  const breadcrumb = document.getElementById('breadcrumb');
  if (breadcrumb) {
    let bc = `<span id="page-title" class="font-semibold" style="color:var(--text-secondary);">${escapeHtml(pageTitle)}</span>`;
    if (params.length > 0) {
      const paramLabel = decodeURIComponent(params[0]);
      bc = `<a href="#${pageName}" class="breadcrumb-link" style="color:var(--text-muted);text-decoration:none;font-weight:500;">${escapeHtml(pageTitle)}</a><span style="color:var(--text-muted);font-size:0.75rem;">/</span><span id="page-title" class="font-semibold" style="color:var(--text-secondary);">${escapeHtml(truncate(paramLabel, 30))}</span>`;
    }
    breadcrumb.innerHTML = bc;
  } else {
    const titleEl = document.getElementById('page-title');
    if (titleEl) titleEl.textContent = pageTitle;
  }

  const sloganEl = document.getElementById('sidebar-slogan');
  if (sloganEl) sloganEl.textContent = t(_pageSloganKeys[pageName] || _pageSloganKeys['dashboard']);

  if (state.currentPage && pages[state.currentPage] && pages[state.currentPage].destroy) { pages[state.currentPage].destroy(); }

  state.currentPage = pageName;
  container.innerHTML = '<div class="page-enter"><div class="skeleton-card" style="margin-bottom:1rem;"><div class="skeleton skeleton-line w-1/4 h-8"></div><div class="skeleton skeleton-line w-3/4"></div><div class="skeleton skeleton-line w-1/2"></div></div><div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:1rem;"><div class="skeleton-card"><div class="skeleton skeleton-line w-1/2 h-6"></div><div class="skeleton skeleton-line w-full"></div></div><div class="skeleton-card"><div class="skeleton skeleton-line w-1/2 h-6"></div><div class="skeleton skeleton-line w-full"></div></div><div class="skeleton-card"><div class="skeleton skeleton-line w-1/2 h-6"></div><div class="skeleton skeleton-line w-full"></div></div></div></div>';

  try { await pageModule.render(container, params); }
  catch (err) {
    console.error(`Error rendering page ${pageName}:`, err);
    container.innerHTML = `<div class="page-enter"><div class="empty-state"><i data-lucide="alert-triangle"></i><p>${t('common.pageLoadError')}: ${escapeHtml(err.message)}</p><button class="btn btn-secondary mt-3" onclick="handleRoute()">${t('common.retry')}</button></div></div>`;
  }
  if (window.lucide) lucide.createIcons();
}

async function loadGraphSelector() {
  try {
    const res = await state.api.listGraphs();
    const graphs = res.data?.graphs || [];
    const sel = document.getElementById('graph-selector');
    if (!sel) return;
    sel.innerHTML = graphs.map(g => `<option value="${escapeHtml(g)}" ${g === state.currentGraphId ? 'selected' : ''}>${escapeHtml(g)}</option>`).join('');
    if (!graphs.includes(state.currentGraphId)) sel.innerHTML = `<option value="${escapeHtml(state.currentGraphId)}" selected>${escapeHtml(state.currentGraphId)}</option>` + sel.innerHTML;
    const delBtn = document.getElementById('graph-delete-btn');
    if (delBtn) { delBtn.style.display = graphs.length > 1 ? '' : 'none'; if (window.lucide) lucide.createIcons(); }
  } catch {}
}

async function deleteCurrentGraph() {
  const graphId = state.currentGraphId;
  const graphs = Array.from(document.getElementById('graph-selector')?.options || []).map(o => o.value);
  if (graphs.length <= 1) { showToast('至少保留一个图谱', 'warning'); return; }
  const confirmed = await showConfirm({ title: '删除图谱', message: `确定要删除图谱 "${graphId}" 吗？此操作将永久删除该图谱的所有实体、关系和文档，不可恢复。`, confirmLabel: '删除', cancelLabel: '取消', destructive: true });
  if (!confirmed) return;
  try { await state.api.deleteGraph(graphId); showToast(`图谱 "${graphId}" 已删除`, 'success'); setGraphId(graphs.filter(g => g !== graphId)[0] || 'default'); loadGraphSelector(); }
  catch (e) { showToast(`删除失败: ${e.message || e}`, 'error'); }
}

async function clearCurrentGraph() {
  const graphId = state.currentGraphId;
  const confirmed = await showConfirm({ title: t('graph.clearTitle'), message: t('graph.clearMessage', { name: graphId }), confirmLabel: t('graph.clearConfirm'), cancelLabel: t('common.cancel') });
  if (!confirmed) return;
  try { await state.api.clearGraph(graphId); showToast(t('graph.clearSuccess', { name: graphId }), 'success'); handleRoute(); }
  catch (e) { showToast(t('graph.clearFailed') + `: ${e.message || e}`, 'error'); }
}

function toggleTheme() { const html = document.documentElement; const isDark = html.getAttribute('data-theme') !== 'light'; const newTheme = isDark ? 'light' : 'dark'; html.setAttribute('data-theme', newTheme); localStorage.setItem('deepdream_theme', newTheme); updateThemeIcon(newTheme); }
function updateThemeIcon(theme) { const darkIcon = document.getElementById('theme-icon-dark'); const lightIcon = document.getElementById('theme-icon-light'); if (darkIcon) darkIcon.style.display = theme === 'dark' ? '' : 'none'; if (lightIcon) lightIcon.style.display = theme === 'light' ? '' : 'none'; }
function initTheme() { const saved = localStorage.getItem('deepdream_theme') || localStorage.getItem('tmg_theme') || 'dark'; document.documentElement.setAttribute('data-theme', saved); updateThemeIcon(saved); }

document.addEventListener('DOMContentLoaded', async () => {
  initTheme();
  window.I18N.init();
  const themeBtn = document.getElementById('theme-toggle');
  if (themeBtn) themeBtn.addEventListener('click', toggleTheme);
  const sel = document.getElementById('graph-selector');
  if (sel) { sel.value = state.currentGraphId; sel.addEventListener('change', () => setGraphId(sel.value)); }
  const graphDelBtn = document.getElementById('graph-delete-btn');
  if (graphDelBtn) graphDelBtn.addEventListener('click', deleteCurrentGraph);
  const graphClearBtn = document.getElementById('graph-clear-btn');
  if (graphClearBtn) graphClearBtn.addEventListener('click', clearCurrentGraph);
  window.addEventListener('hashchange', handleRoute);
  try { const h = await state.api.health(state.currentGraphId); if (h.data?.storage_backend) state.backendType = h.data.storage_backend; } catch {}
  handleRoute();
  loadGraphSelector();
  ['nav-communities'].forEach(id => { const el = document.getElementById(id); if (el) el.style.display = isNeo4j() ? '' : 'none'; });
  const toggle = document.getElementById('sidebar-toggle');
  const sidebar = document.getElementById('sidebar');
  if (toggle && sidebar) {
    const backdrop = document.createElement('div'); backdrop.className = 'sidebar-backdrop'; backdrop.id = 'sidebar-backdrop'; document.body.appendChild(backdrop);
    toggle.addEventListener('click', () => { sidebar.classList.toggle('open'); backdrop.classList.toggle('active', sidebar.classList.contains('open')); });
    backdrop.addEventListener('click', () => { sidebar.classList.remove('open'); backdrop.classList.remove('active'); });
    sidebar.querySelectorAll('.sidebar-link').forEach(link => { link.addEventListener('click', () => { if (window.innerWidth < 768) { sidebar.classList.remove('open'); backdrop.classList.remove('active'); } }); });
    const collapsed = localStorage.getItem('deepdream_sidebar_collapsed') === 'true';
    if (collapsed && window.innerWidth >= 768) sidebar.classList.add('collapsed');
  }
});

function _toggleSidebarCollapse() { const sidebar = document.getElementById('sidebar'); if (!sidebar) return; sidebar.classList.toggle('collapsed'); localStorage.setItem('deepdream_sidebar_collapsed', sidebar.classList.contains('collapsed')); }

// ---- Episode/doc modal ---- (kept in app.js since used by multiple pages)

function _renderDocModal(sourceName, eventTime, cache, original) {
  const hasCache = !!cache, hasOriginal = !!original, showTabs = hasCache && hasOriginal;
  let body = `<div style="display:flex;flex-direction:column;gap:1rem;"><div style="display:grid;grid-template-columns:auto 1fr;gap:0.25rem 0.75rem;font-size:0.85rem;"><span style="color:var(--text-secondary);">${t('memory.taskSource')}:</span><span>${escapeHtml(sourceName)}</span><span style="color:var(--text-secondary);">${t('memory.docTime')}:</span><span>${formatDate(eventTime)}</span></div>`;
  if (showTabs) { body += `<div style="display:flex;gap:0;border-bottom:1px solid var(--border-color);margin-bottom:0;"><button class="doc-tab-btn active" data-doc-tab="cache" style="padding:0.5rem 1rem;font-size:0.85rem;border:none;background:none;cursor:pointer;border-bottom:2px solid var(--primary);color:var(--text-primary);font-weight:600;">${t('memory.cacheSummary')}</button><button class="doc-tab-btn" data-doc-tab="original" style="padding:0.5rem 1rem;font-size:0.85rem;border:none;background:none;cursor:pointer;border-bottom:2px solid transparent;color:var(--text-muted);">${t('memory.originalText')}</button></div><div id="doc-tab-cache" class="doc-tab-panel" style="max-height:400px;overflow-y:auto;background:var(--bg-secondary);padding:0.75rem;border-radius:0.5rem;font-size:0.85rem;line-height:1.6;white-space:pre-wrap;word-break:break-word;">${escapeHtml(cache)}</div><div id="doc-tab-original" class="doc-tab-panel" style="display:none;max-height:400px;overflow-y:auto;background:var(--bg-secondary);padding:0.75rem;border-radius:0.5rem;font-size:0.85rem;line-height:1.6;white-space:pre-wrap;word-break:break-word;">${escapeHtml(original)}</div>`; }
  else if (hasCache) { body += `<div><h4 style="margin-bottom:0.5rem;">${t('memory.cacheSummary')}</h4><div style="max-height:400px;overflow-y:auto;background:var(--bg-secondary);padding:0.75rem;border-radius:0.5rem;font-size:0.85rem;line-height:1.6;white-space:pre-wrap;word-break:break-word;">${escapeHtml(cache)}</div></div>`; }
  else if (hasOriginal) { body += `<div><h4 style="margin-bottom:0.5rem;">${t('memory.originalText')}</h4><div style="max-height:400px;overflow-y:auto;background:var(--bg-secondary);padding:0.75rem;border-radius:0.5rem;font-size:0.85rem;line-height:1.6;white-space:pre-wrap;word-break:break-word;">${escapeHtml(original)}</div></div>`; }
  body += '</div>';
  const { overlay } = showModal({ title: t('memory.docContent') + ' - ' + escapeHtml(truncate(sourceName, 30)), content: body, size: 'lg' });
  if (showTabs) { overlay.querySelectorAll('.doc-tab-btn').forEach(btn => { btn.addEventListener('click', () => { const tab = btn.getAttribute('data-doc-tab'); overlay.querySelectorAll('.doc-tab-btn').forEach(b => { b.classList.remove('active'); b.style.borderBottom = '2px solid transparent'; b.style.color = 'var(--text-muted)'; b.style.fontWeight = '400'; }); btn.classList.add('active'); btn.style.borderBottom = '2px solid var(--primary)'; btn.style.color = 'var(--text-primary)'; btn.style.fontWeight = '600'; overlay.querySelectorAll('.doc-tab-panel').forEach(p => p.style.display = 'none'); const panel = overlay.querySelector('#doc-tab-' + tab); if (panel) panel.style.display = ''; }); }); }
}

window.showDocContent = async function(filename) { if (!filename) return; try { const res = await state.api.getDocContent(filename, state.currentGraphId); const data = res.data || {}; const meta = data.meta || {}; _renderDocModal(meta.source_document || meta.doc_name || filename, meta.event_time || '-', data.cache || '', data.original || ''); } catch (err) { showToast(t('memory.loadDocContentFailed') + ': ' + err.message, 'error'); } };

// NOTE: showEpisodeDetailModal is kept in app.js since it references multiple page-level concerns.
// The full implementation is identical to the original — only the utility function calls
// now route through window.Format / window.UIUtils instead of local definitions.

// ---- Keyboard Shortcut System ----
const _shortcuts = [];
const _SHORTCUT_IGNORE_TAGS = new Set(['INPUT', 'TEXTAREA', 'SELECT']);
function registerShortcut(key, desc, handler, opts = {}) { _shortcuts.push({ key, desc, handler, global: opts.global || false, ctrlKey: opts.ctrlKey, shiftKey: opts.shiftKey, altKey: opts.altKey }); }
function _matchShortcut(e, s) { if (s.key !== e.key) return false; const needCtrl = s.ctrlKey !== undefined; if (needCtrl) { const ctrlOk = e.ctrlKey || e.metaKey; if (s.ctrlKey && !ctrlOk) return false; if (!s.ctrlKey && ctrlOk) return false; } if (s.shiftKey && !e.shiftKey) return false; if (s.altKey && !e.altKey) return false; return true; }

document.addEventListener('keydown', (e) => {
  for (const s of _shortcuts) { if (s.global && _matchShortcut(e, s)) { e.preventDefault(); s.handler(e); return; } }
  const tag = document.activeElement?.tagName; if (_SHORTCUT_IGNORE_TAGS.has(tag)) return;
  for (const s of _shortcuts) { if (!s.global && _matchShortcut(e, s)) { e.preventDefault(); s.handler(e); return; } }
});

registerShortcut('k', 'Open command palette', () => _openCommandPalette(), { ctrlKey: true, global: true });
registerShortcut('/', 'Focus search', () => { navigate('#search'); setTimeout(() => { const el = document.getElementById('search-input'); if (el) el.focus(); }, 100); }, { global: true });
registerShortcut('b', 'Toggle sidebar', () => { const sidebar = document.getElementById('sidebar'); if (sidebar) sidebar.classList.toggle('collapsed'); }, { ctrlKey: true, global: true });
registerShortcut('?', 'Show shortcuts', () => _showShortcutsHelp(), { ctrlKey: true, global: true });
registerShortcut('1', 'Dashboard', () => navigate('#dashboard'), { altKey: true });
registerShortcut('2', 'Memory', () => navigate('#memory'), { altKey: true });
registerShortcut('3', 'Graph', () => navigate('#graph'), { altKey: true });
registerShortcut('4', 'Search', () => navigate('#search'), { altKey: true });
registerShortcut('5', 'API Test', () => navigate('#api-test'), { altKey: true });

// ---- Command Palette ---- (unchanged from original, shortened for readability)
function _openCommandPalette() {
  const commands = [
    { label: t('nav.dashboard') || 'Dashboard', icon: 'layout-dashboard', action: () => navigate('#dashboard') },
    { label: t('nav.memory') || 'Memory', icon: 'database', action: () => navigate('#memory') },
    { label: t('nav.graph') || 'Graph', icon: 'git-fork', action: () => navigate('#graph') },
    { label: t('nav.search') || 'Search', icon: 'search', action: () => navigate('#search') },
    { label: t('nav.apiTest') || 'API Test', icon: 'terminal', action: () => navigate('#api-test') },
    { label: t('common.toggleTheme') || 'Toggle Theme', icon: 'sun', action: () => toggleTheme() },
    { label: t('common.writeMemory') || 'Write Memory', icon: 'plus', action: () => navigate('#memory') },
    { label: t('common.healthCheck') || 'Health Check', icon: 'heart', action: async () => { try { await state.api.health(state.currentGraphId); showToast(t('common.statusOk') || 'API Connected', 'success'); } catch (e) { showToast(t('common.statusError') || 'API Error: ' + e.message, 'error'); } } },
  ];
  const currentPageModule = pages[state.currentPage]; if (currentPageModule && currentPageModule.getCommands) commands.push(...currentPageModule.getCommands());
  const overlay = document.createElement('div'); overlay.className = 'modal-overlay'; overlay.setAttribute('role', 'dialog'); overlay.setAttribute('aria-label', 'Command Palette'); overlay.style.background = 'rgba(0,0,0,0.5)'; overlay.style.alignItems = 'flex-start'; overlay.style.paddingTop = '18vh';
  overlay.innerHTML = `<div class="command-palette" style="background:var(--bg-surface);border:1px solid var(--border-color);border-radius:0.75rem;width:90%;max-width:500px;max-height:400px;display:flex;flex-direction:column;box-shadow:var(--shadow-lg);animation:modal-scale-in 0.12s ease;"><div style="padding:0.75rem;border-bottom:1px solid var(--border-color);display:flex;align-items:center;gap:0.5rem;"><i data-lucide="search" style="width:18px;height:18px;color:var(--text-muted);flex-shrink:0;"></i><input type="text" id="command-palette-input" class="input" placeholder="${t('common.searchCommands') || 'Type a command...'}" style="border:none;background:transparent;box-shadow:none;padding:0;font-size:0.9rem;"><kbd style="font-size:0.7rem;padding:2px 6px;border-radius:4px;background:var(--bg-surface-hover);border:1px solid var(--border-color);color:var(--text-muted);font-family:var(--font-mono);">ESC</kbd></div><div id="command-palette-list" style="overflow-y:auto;padding:0.25rem;"></div><div style="padding:0.5rem 0.75rem;border-top:1px solid var(--border-color);display:flex;gap:1rem;font-size:0.7rem;color:var(--text-muted);"><span><kbd style="padding:1px 4px;border-radius:3px;background:var(--bg-surface-hover);border:1px solid var(--border-color);">↑↓</kbd> navigate</span><span><kbd style="padding:1px 4px;border-radius:3px;background:var(--bg-surface-hover);border:1px solid var(--border-color);">↵</kbd> select</span></div></div>`;
  let selectedIdx = 0; let filtered = [...commands];
  function render() { const list = overlay.querySelector('#command-palette-list'); if (!list) return; list.innerHTML = filtered.length === 0 ? '<div style="padding:1rem;text-align:center;color:var(--text-muted);font-size:0.85rem;">No results</div>' : filtered.map((cmd, i) => `<div class="command-item" data-cmd-idx="${i}" style="display:flex;align-items:center;gap:0.75rem;padding:0.5rem 0.75rem;border-radius:0.5rem;cursor:pointer;transition:background 0.1s ease;${i === selectedIdx ? 'background:var(--primary-dim);' : ''}"><i data-lucide="${cmd.icon || 'circle'}" style="width:16px;height:16px;color:var(--text-muted);flex-shrink:0;"></i><span style="font-size:0.85rem;color:${i === selectedIdx ? 'var(--primary-hover)' : 'var(--text-primary)'};">${escapeHtml(cmd.label)}</span>${cmd.desc ? `<span style="font-size:0.7rem;color:var(--text-muted);margin-left:auto;">${escapeHtml(cmd.desc)}</span>` : ''}</div>`).join(''); if (window.lucide) lucide.createIcons({ nodes: [overlay] }); }
  const input = overlay.querySelector('#command-palette-input');
  input.addEventListener('input', () => { const q = input.value.toLowerCase().trim(); filtered = q ? commands.filter(c => c.label.toLowerCase().includes(q) || (c.desc || '').toLowerCase().includes(q)) : [...commands]; selectedIdx = 0; render(); });
  input.addEventListener('keydown', (e) => { if (e.key === 'ArrowDown') { e.preventDefault(); selectedIdx = Math.min(selectedIdx + 1, filtered.length - 1); render(); } else if (e.key === 'ArrowUp') { e.preventDefault(); selectedIdx = Math.max(selectedIdx - 1, 0); render(); } else if (e.key === 'Enter') { e.preventDefault(); if (filtered[selectedIdx]) { close(); filtered[selectedIdx].action(); } } else if (e.key === 'Escape') { e.preventDefault(); close(); } });
  const listEl = overlay.querySelector('#command-palette-list');
  listEl.addEventListener('click', (e) => { const item = e.target.closest('.command-item'); if (!item) return; const idx = parseInt(item.dataset.cmdIdx); if (filtered[idx]) { close(); filtered[idx].action(); } });
  listEl.addEventListener('mousemove', (e) => { const item = e.target.closest('.command-item'); if (!item) return; const idx = parseInt(item.dataset.cmdIdx); if (idx === selectedIdx) return; selectedIdx = idx; listEl.querySelectorAll('.command-item').forEach((el, i) => { el.style.background = i === selectedIdx ? 'var(--primary-dim)' : ''; }); });
  const close = () => overlay.remove(); overlay.addEventListener('click', (e) => { if (e.target === overlay) close(); });
  document.body.appendChild(overlay); render(); requestAnimationFrame(() => input.focus());
}

function _showShortcutsHelp() {
  const shortcutList = _shortcuts.map(s => { let keys = []; if (s.ctrlKey) keys.push('Ctrl'); if (s.altKey) keys.push('Alt'); if (s.shiftKey) keys.push('Shift'); keys.push(s.key.length === 1 ? s.key.toUpperCase() : s.key); return { keys: keys.join(' + '), desc: s.desc }; });
  const content = `<div style="display:flex;flex-direction:column;gap:0.25rem;">${shortcutList.map(s => `<div style="display:flex;align-items:center;justify-content:space-between;padding:0.5rem 0;border-bottom:1px solid var(--border-color);"><span style="font-size:0.85rem;color:var(--text-secondary);">${escapeHtml(s.desc)}</span><kbd style="font-size:0.75rem;padding:3px 8px;border-radius:4px;background:var(--bg-surface-hover);border:1px solid var(--border-color);font-family:var(--font-mono);color:var(--primary);">${escapeHtml(s.keys)}</kbd></div>`).join('')}</div>`;
  showModal({ title: t('common.keyboardShortcuts') || 'Keyboard Shortcuts', content, size: 'sm' });
}

// NOTE: window.showEpisodeDetailModal is intentionally NOT re-included here in the shortened
// version to keep app.js focused. The original 300-line function remains unchanged —
// it is loaded from the original app.js backup if needed. For the split, we reference it
// via the showEpisodeDoc/showEpisodeDetailModal globals which are defined below.

// Pull in the showEpisodeDetailModal from the original backup or keep inline.
// Since this is a self-contained function, we keep it as a separate file:
// shared/episode-modal.js — loaded before pages that need it.

// For now, keep the showEpisodeDetailModal inline (unchanged logic, just references Format/UIUtils):
window.showEpisodeDetailModal = async function(uuid) {
  if (!uuid) return;
  try {
    const [epRes, entRes] = await Promise.all([
      state.api.getEpisode(uuid, state.currentGraphId),
      state.api.getEpisodeEntities(uuid, state.currentGraphId),
    ]);
    const ep = epRes.data || {};
    const allItems = entRes.data?.entities || [];
    const cacheText = ep.content || '';
    const originalText = ep.source_text || '';
    const epEntities = allItems.filter(x => x.target_type === 'entity');
    const epRelations = allItems.filter(x => x.target_type === 'relation');
    const hasCache = !!cacheText;
    const hasOriginal = !!originalText;
    const hasConcepts = epEntities.length > 0 || epRelations.length > 0;
    const _entityNameMap = {};
    epEntities.forEach(e => { _entityNameMap[e.absolute_id] = e.name || e.family_id; });
    const _fullRelations = {};
    if (epRelations.length > 0) { await Promise.all(epRelations.map(r => state.api.relationByAbsoluteId(r.absolute_id, state.currentGraphId).then(res => { if (res.data) _fullRelations[r.absolute_id] = res.data; }).catch(() => {}))); }
    function _relationLabel(rel) { const full = _fullRelations[rel.absolute_id]; if (!full) return escapeHtml(rel.family_id || '-'); const e1 = _entityNameMap[full.entity1_absolute_id] || full.entity1_absolute_id || '?'; const e2 = _entityNameMap[full.entity2_absolute_id] || full.entity2_absolute_id || '?'; const content = truncate(full.content || '', 40); return `${escapeHtml(e1)} <span style="color:var(--text-muted);">→</span> ${escapeHtml(content)} <span style="color:var(--text-muted);">→</span> ${escapeHtml(e2)}`; }
    function _buildMainHtml() { let body = `<div style="display:flex;flex-direction:column;gap:1rem;"><div style="display:grid;grid-template-columns:auto 1fr;gap:0.25rem 0.75rem;font-size:0.85rem;"><span style="color:var(--text-secondary);">UUID:</span><span class="font-mono text-xs">${escapeHtml(ep.uuid || '')}</span>${ep.episode_type ? `<span style="color:var(--text-secondary);">${t('episodes.episodeType')}:</span><span>${escapeHtml(ep.episode_type)}</span>` : ''}<span style="color:var(--text-secondary);">${t('common.source')}:</span><span>${escapeHtml(ep.source_document || '-')}</span><span style="color:var(--text-secondary);">${t('relations.eventTime')}:</span><span class="mono" style="font-size:0.8125rem;">${formatDate(ep.event_time)}</span>${ep.processed_time ? `<span style="color:var(--text-secondary);">${t('relations.processedTime')}:</span><span class="mono" style="font-size:0.8125rem;">${formatDateMs(ep.processed_time)}</span>` : ''}</div>`; const mainTabs = []; if (hasCache) mainTabs.push('cache'); if (hasOriginal) mainTabs.push('original'); if (hasConcepts) mainTabs.push('concepts'); if (mainTabs.length === 0) mainTabs.push('cache'); const useMainTabs = mainTabs.length > 1; if (useMainTabs) { const tabBtns = mainTabs.map((tab, i) => { const labels = { cache: t('memory.cacheSummary'), original: t('memory.originalText'), concepts: `${t('episodes.tabConcepts')} (${epEntities.length + epRelations.length})` }; const active = i === 0; return `<button class="ep-main-tab ${active ? 'active' : ''}" data-ep-main="${tab}" style="padding:0.5rem 1rem;font-size:0.85rem;border:none;background:none;cursor:pointer;border-bottom:2px solid ${active ? 'var(--primary)' : 'transparent'};color:${active ? 'var(--text-primary)' : 'var(--text-muted)'};font-weight:${active ? '600' : '400'};">${labels[tab]}</button>`; }).join(''); body += `<div style="display:flex;gap:0;border-bottom:1px solid var(--border-color);">${tabBtns}</div>`; } if (hasCache) { body += `<div id="ep-panel-cache" class="ep-main-panel" style="${useMainTabs && mainTabs[0] !== 'cache' ? 'display:none;' : ''}max-height:500px;overflow-y:auto;background:var(--bg-secondary);padding:0.75rem;border-radius:0.5rem;font-size:0.85rem;line-height:1.6;white-space:pre-wrap;word-break:break-word;">${renderMarkdown(cacheText)}</div>`; } if (hasOriginal) { body += `<div id="ep-panel-original" class="ep-main-panel" style="${useMainTabs && mainTabs[0] !== 'original' ? 'display:none;' : ''}max-height:500px;overflow-y:auto;background:var(--bg-secondary);padding:0.75rem;border-radius:0.5rem;font-size:0.85rem;line-height:1.6;white-space:pre-wrap;word-break:break-word;">${escapeHtml(originalText)}</div>`; } if (!hasCache && !hasOriginal) { body += `<div style="color:var(--text-muted);font-size:0.85rem;padding:1rem;">${t('episodes.noContent')}</div>`; } if (hasConcepts) { const entityList = epEntities.length > 0 ? epEntities.map((ent, i) => `<div class="flex items-center gap-2 p-2 rounded cursor-pointer" style="background:var(--bg-secondary);font-size:0.85rem;margin-bottom:4px;" data-ep-entity-idx="${i}"><i data-lucide="circle-dot" style="width:14px;height:14px;color:var(--primary);flex-shrink:0;"></i><span class="font-medium">${escapeHtml(ent.name || ent.family_id || '-')}</span></div>`).join('') : `<p style="color:var(--text-muted);font-size:0.85rem;padding:0.5rem;">${t('episodes.noEntities')}</p>`; const relationList = epRelations.length > 0 ? epRelations.map((rel, i) => `<div class="flex items-center gap-2 p-2 rounded cursor-pointer" style="background:var(--bg-secondary);font-size:0.85rem;margin-bottom:4px;" data-ep-relation-idx="${i}"><i data-lucide="link" style="width:14px;height:14px;color:var(--warning);flex-shrink:0;"></i><span>${_relationLabel(rel)}</span></div>`).join('') : `<p style="color:var(--text-muted);font-size:0.85rem;padding:0.5rem;">${t('episodes.noRelations')}</p>`; body += `<div id="ep-panel-concepts" class="ep-main-panel" style="display:none;"><div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;"><div><h4 style="margin-bottom:0.5rem;font-size:0.85rem;color:var(--text-secondary);"><i data-lucide="circle-dot" style="width:14px;height:14px;color:var(--primary);vertical-align:middle;"></i>${t('episodes.entities')} (${epEntities.length})</h4>${entityList}</div><div><h4 style="margin-bottom:0.5rem;font-size:0.85rem;color:var(--text-secondary);"><i data-lucide="link" style="width:14px;height:14px;color:var(--warning);vertical-align:middle;"></i>${t('episodes.relations')} (${epRelations.length})</h4>${relationList}</div></div></div>`; } body += '</div>'; return body; }
    function _buildEntityDetailHtml(entity) { let h = `<div style="display:flex;flex-direction:column;gap:0.75rem;"><div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.25rem;"><button class="btn btn-secondary btn-sm" id="ep-back-btn" title="${t('common.back')}"><i data-lucide="arrow-left" style="width:14px;height:14px;"></i></button><span class="badge badge-primary">${t('graph.entityDetail')}</span></div><h3 style="font-size:1.1rem;font-weight:600;color:var(--text-primary);word-break:break-word;">${escapeHtml(entity.name || t('graph.unnamedEntity'))}</h3><div style="display:grid;grid-template-columns:auto 1fr;gap:0.25rem 0.75rem;font-size:0.85rem;"><span style="color:var(--text-secondary);">ID:</span><span class="mono text-xs">${escapeHtml(entity.absolute_id || '')}</span><span style="color:var(--text-secondary);">Family:</span><span class="mono text-xs">${escapeHtml(entity.family_id || '')}</span>${entity.episode_type ? `<span style="color:var(--text-secondary);">${t('episodes.episodeType')}:</span><span>${escapeHtml(entity.episode_type)}</span>` : ''}${entity.event_time ? `<span style="color:var(--text-secondary);">${t('relations.eventTime')}:</span><span class="mono" style="font-size:0.8125rem;">${formatDate(entity.event_time)}</span>` : ''}</div>${entity.content ? `<div><span class="form-label" style="margin-bottom:0.25rem;">${t('graph.content')}</span><div class="md-content" style="max-height:350px;overflow-y:auto;background:var(--bg-secondary);padding:0.75rem;border-radius:0.5rem;font-size:0.85rem;line-height:1.6;">${renderMarkdown(entity.content)}</div></div>` : ''}</div>`; return h; }
    function _buildRelationDetailHtml(relation) { const e1 = _entityNameMap[relation.entity1_absolute_id] || relation.entity1_absolute_id || '?'; const e2 = _entityNameMap[relation.entity2_absolute_id] || relation.entity2_absolute_id || '?'; let h = `<div style="display:flex;flex-direction:column;gap:0.75rem;"><div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.25rem;"><button class="btn btn-secondary btn-sm" id="ep-back-btn" title="${t('common.back')}"><i data-lucide="arrow-left" style="width:14px;height:14px;"></i></button><span class="badge" style="background:var(--info-dim);color:var(--info);">${t('graph.relationDetail')}</span></div><h3 style="font-size:1.1rem;font-weight:600;color:var(--text-primary);word-break:break-word;">${escapeHtml(truncate(relation.content || t('graph.unnamedRelation'), 80))}</h3><div style="display:grid;grid-template-columns:auto 1fr;gap:0.25rem 0.75rem;font-size:0.85rem;"><span style="color:var(--text-secondary);">${t('graph.fromEntity')}:</span><span style="color:var(--info);cursor:pointer;" class="ep-nav-entity" data-nav-abs="${escapeHtml(relation.entity1_absolute_id)}">${escapeHtml(e1)}</span><span style="color:var(--text-secondary);">${t('graph.toEntity')}:</span><span style="color:var(--info);cursor:pointer;" class="ep-nav-entity" data-nav-abs="${escapeHtml(relation.entity2_absolute_id)}">${escapeHtml(e2)}</span><span style="color:var(--text-secondary);">${t('graph.relationId')}:</span><span class="mono text-xs">${escapeHtml(relation.family_id || '-')}</span>${relation.event_time ? `<span style="color:var(--text-secondary);">${t('graph.eventTime')}:</span><span class="mono" style="font-size:0.8125rem;">${formatDate(relation.event_time)}</span>` : ''}${relation.processed_time ? `<span style="color:var(--text-secondary);">${t('graph.processedTime')}:</span><span class="mono" style="font-size:0.8125rem;">${formatDateMs(relation.processed_time)}</span>` : ''}</div>${relation.content ? `<div><span class="form-label" style="margin-bottom:0.25rem;">${t('graph.content')}</span><div class="md-content" style="max-height:350px;overflow-y:auto;background:var(--bg-secondary);padding:0.75rem;border-radius:0.5rem;font-size:0.85rem;line-height:1.6;">${renderMarkdown(relation.content)}</div></div>` : ''}</div>`; return h; }
    const _navStack = [];
    const { overlay, close } = showModal({ title: t('episodes.detail'), content: _buildMainHtml(), size: 'lg' });
    const _bodyEl = overlay.querySelector('.modal-body');
    if (window.lucide) lucide.createIcons({ nodes: [overlay] });
    function _saveMainAndSwitch(detailHtml) { if (_bodyEl) _navStack.push(_bodyEl.innerHTML); if (_bodyEl) _bodyEl.innerHTML = detailHtml; if (window.lucide) lucide.createIcons({ nodes: [overlay] }); _bindDetailEvents(); }
    function _restoreView(html) { if (_bodyEl) _bodyEl.innerHTML = html; if (window.lucide) lucide.createIcons({ nodes: [overlay] }); if (_bodyEl?.querySelector('.ep-main-tab')) _bindMainEvents(); else _bindDetailEvents(); }
    function _bindDetailEvents() { const backBtn = _bodyEl?.querySelector('#ep-back-btn'); if (backBtn) backBtn.addEventListener('click', () => { const prev = _navStack.pop(); if (prev) _restoreView(prev); }); _bodyEl?.querySelectorAll('.ep-nav-entity').forEach(el => { el.addEventListener('click', () => { const absId = el.getAttribute('data-nav-abs'); if (absId) _navigateToEntity(absId); }); }); }
    async function _navigateToEntity(absoluteId) { try { const res = await state.api.entityByAbsoluteId(absoluteId, state.currentGraphId); const entity = res.data; if (!entity) { showToast('Entity not found', 'error'); return; } _saveMainAndSwitch(_buildEntityDetailHtml(entity)); } catch (err) { showToast(err.message, 'error'); } }
    async function _navigateToRelation(absoluteId) { const full = _fullRelations[absoluteId]; if (full) { if (!_entityNameMap[full.entity1_absolute_id]) { try { const r = await state.api.entityByAbsoluteId(full.entity1_absolute_id, state.currentGraphId); if (r.data) _entityNameMap[full.entity1_absolute_id] = r.data.name || r.data.family_id; } catch (_) {} } if (!_entityNameMap[full.entity2_absolute_id]) { try { const r = await state.api.entityByAbsoluteId(full.entity2_absolute_id, state.currentGraphId); if (r.data) _entityNameMap[full.entity2_absolute_id] = r.data.name || r.data.family_id; } catch (_) {} } _saveMainAndSwitch(_buildRelationDetailHtml(full)); } else { try { const res = await state.api.relationByAbsoluteId(absoluteId, state.currentGraphId); const rel = res.data; if (!rel) { showToast('Relation not found', 'error'); return; } _fullRelations[absoluteId] = rel; for (const aid of [rel.entity1_absolute_id, rel.entity2_absolute_id]) { if (!_entityNameMap[aid]) { try { const r = await state.api.entityByAbsoluteId(aid, state.currentGraphId); if (r.data) _entityNameMap[aid] = r.data.name || r.data.family_id; } catch (_) {} } } _saveMainAndSwitch(_buildRelationDetailHtml(rel)); } catch (err) { showToast(err.message, 'error'); } } }
    function _bindMainEvents() { overlay.querySelectorAll('.ep-main-tab').forEach(btn => { btn.addEventListener('click', () => { const tab = btn.getAttribute('data-ep-main'); overlay.querySelectorAll('.ep-main-tab').forEach(b => { b.classList.remove('active'); b.style.borderBottom = '2px solid transparent'; b.style.color = 'var(--text-muted)'; b.style.fontWeight = '400'; }); btn.classList.add('active'); btn.style.borderBottom = '2px solid var(--primary)'; btn.style.color = 'var(--text-primary)'; btn.style.fontWeight = '600'; overlay.querySelectorAll('.ep-main-panel').forEach(p => p.style.display = 'none'); const panel = overlay.querySelector(`#ep-panel-${tab}`); if (panel) panel.style.display = ''; }); }); overlay.querySelectorAll('[data-ep-entity-idx]').forEach(el => { el.addEventListener('click', () => { const idx = parseInt(el.getAttribute('data-ep-entity-idx'), 10); const ref = epEntities[idx]; if (ref) _navigateToEntity(ref.absolute_id); }); }); overlay.querySelectorAll('[data-ep-relation-idx]').forEach(el => { el.addEventListener('click', () => { const idx = parseInt(el.getAttribute('data-ep-relation-idx'), 10); const ref = epRelations[idx]; if (ref) _navigateToRelation(ref.absolute_id); }); }); }
    _bindMainEvents();
  } catch (err) { showToast(err.message, 'error'); }
};
window.showEpisodeDoc = window.showEpisodeDetailModal;

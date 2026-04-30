/* ==========================================
   DeepDream Dashboard - Core Application
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

class DeepDreamApi {
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

  // Entities
  listEntities(graphId = 'default', limit, offset) {
    let q = `graph_id=${encodeURIComponent(graphId)}`;
    if (limit) q += `&limit=${limit}`;
    if (offset) q += `&offset=${offset}`;
    return this.get(`/api/v1/find/entities?${q}`);
  }
  getCounts(graphId = 'default') {
    return this.get(`/api/v1/stats/counts?graph_id=${encodeURIComponent(graphId)}`);
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
  entityVersions(familyId, graphId = 'default') {
    return this.get(`/api/v1/find/entities/${encodeURIComponent(familyId)}/versions?graph_id=${encodeURIComponent(graphId)}`);
  }
  entityVersionDiff(familyId, v1, v2, graphId = 'default') {
    return this.get(`/api/v1/find/entities/${encodeURIComponent(familyId)}/version-diff?v1=${encodeURIComponent(v1)}&v2=${encodeURIComponent(v2)}&graph_id=${encodeURIComponent(graphId)}`);
  }
  entityRelations(familyId, graphId = 'default', options = {}) {
    let q = `graph_id=${encodeURIComponent(graphId)}`;
    if (options.limit) q += `&limit=${options.limit}`;
    if (options.maxVersionAbsoluteId) q += `&max_version_absolute_id=${encodeURIComponent(options.maxVersionAbsoluteId)}`;
    if (options.relationScope) q += `&relation_scope=${encodeURIComponent(options.relationScope)}`;
    return this.get(`/api/v1/find/entities/${encodeURIComponent(familyId)}/relations?${q}`);
  }
  entityVersionCounts(familyIds, graphId = 'default') {
    return this.post('/api/v1/find/entities/version-counts', {
      family_ids: familyIds,
      graph_id: graphId,
    });
  }
  entityOneHop(absoluteId, graphId = 'default') {
    return this.get(`/api/v1/find/entities/absolute/${encodeURIComponent(absoluteId)}/relations?graph_id=${encodeURIComponent(graphId)}`);
  }
  entityByAbsoluteId(absoluteId, graphId = 'default') {
    return this.get(`/api/v1/find/entities/absolute/${encodeURIComponent(absoluteId)}?graph_id=${encodeURIComponent(graphId)}`);
  }

  entityEmbeddingPreview(absoluteId, numValues = 5, graphId = 'default') {
    return this.get(`/api/v1/find/entities/absolute/${encodeURIComponent(absoluteId)}/embedding-preview?num_values=${numValues}&graph_id=${encodeURIComponent(graphId)}`);
  }

  // --- CRUD ---
  updateEntity(familyId, data, graphId = 'default') {
    return this.request('PUT', `/api/v1/find/entities/${encodeURIComponent(familyId)}?graph_id=${encodeURIComponent(graphId)}`, {
      json: data,
    });
  }

  // --- Entity v3: Summary Evolution ---
  evolveEntitySummary(familyId, graphId = 'default') {
    return this.post(`/api/v1/find/entities/${encodeURIComponent(familyId)}/evolve-summary?graph_id=${encodeURIComponent(graphId)}`, {});
  }

  // --- Entity v3: Contradictions ---
  entityContradictions(familyId, graphId = 'default') {
    return this.get(`/api/v1/find/entities/${encodeURIComponent(familyId)}/contradictions?graph_id=${encodeURIComponent(graphId)}`);
  }

  resolveContradiction(familyId, data, graphId = 'default') {
    return this.post(`/api/v1/find/entities/${encodeURIComponent(familyId)}/resolve-contradiction?graph_id=${encodeURIComponent(graphId)}`, data);
  }

  // --- Entity v3: Provenance ---
  entityProvenance(familyId, graphId = 'default') {
    return this.get(`/api/v1/find/entities/${encodeURIComponent(familyId)}/provenance?graph_id=${encodeURIComponent(graphId)}`);
  }

  // --- Graph Traversal (Phase B) ---
  traverseGraph(seedFamilyIds, maxDepth = 3, maxNodes = 100, graphId = 'default') {
    return this.post('/api/v1/find/traverse', {
      seed_family_ids: seedFamilyIds,
      max_depth: maxDepth,
      max_nodes: maxNodes,
      graph_id: graphId,
    });
  }

  // --- Episodes v3: Batch Ingest ---
  batchIngestEpisodes(episodes, graphId = 'default') {
    return this.post('/api/v1/find/episodes/batch-ingest', { episodes, graph_id: graphId });
  }

  deleteEntity(familyId, cascade = false, graphId = 'default') {
    return this.request('DELETE', `/api/v1/find/entities/${encodeURIComponent(familyId)}?cascade=${cascade}&graph_id=${encodeURIComponent(graphId)}`);
  }

  batchDeleteEntities(familyIds, cascade = false, graphId = 'default') {
    return this.request('POST', '/api/v1/find/entities/batch-delete', {
      json: { family_ids: familyIds, cascade, graph_id: graphId },
    });
  }

  updateRelation(familyId, data, graphId = 'default') {
    return this.request('PUT', `/api/v1/find/relations/${encodeURIComponent(familyId)}?graph_id=${encodeURIComponent(graphId)}`, {
      json: data,
    });
  }

  deleteRelation(familyId, graphId = 'default') {
    return this.request('DELETE', `/api/v1/find/relations/${encodeURIComponent(familyId)}?graph_id=${encodeURIComponent(graphId)}`);
  }

  batchDeleteRelations(familyIds, graphId = 'default') {
    return this.request('POST', '/api/v1/find/relations/batch-delete', {
      json: { family_ids: familyIds, graph_id: graphId },
    });
  }

  mergeEntities(targetFamilyId, sourceFamilyIds, graphId = 'default') {
    return this.post('/api/v1/find/entities/merge', { target_family_id: targetFamilyId, source_family_ids: sourceFamilyIds, graph_id: graphId });
  }

  // Relations
  relationVersions(familyId, graphId = 'default') {
    return this.get(`/api/v1/find/relations/${encodeURIComponent(familyId)}/versions?graph_id=${encodeURIComponent(graphId)}`);
  }
  relationByAbsoluteId(absoluteId, graphId = 'default') {
    return this.get(`/api/v1/find/relations/absolute/${encodeURIComponent(absoluteId)}?graph_id=${encodeURIComponent(graphId)}`);
  }

  relationEmbeddingPreview(absoluteId, numValues = 5, graphId = 'default') {
    return this.get(`/api/v1/find/relations/absolute/${encodeURIComponent(absoluteId)}/embedding-preview?num_values=${numValues}&graph_id=${encodeURIComponent(graphId)}`);
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
      family_id_a: entityA,
      family_id_b: entityB,
      graph_id: graphId,
    });
  }
  shortestPaths(entityA, entityB, graphId = 'default', options = {}) {
    return this.post('/api/v1/find/paths/shortest', {
      family_id_a: entityA,
      family_id_b: entityB,
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
      family_id_a: entityA,
      family_id_b: entityB,
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

  // --- Time Travel ---
  getSnapshot(time, graphId = 'default') {
    return this.get(`/api/v1/find/snapshot?time=${encodeURIComponent(time)}&graph_id=${encodeURIComponent(graphId)}`);
  }

  getChanges(since, until, graphId = 'default') {
    let url = `/api/v1/find/changes?since=${encodeURIComponent(since)}&graph_id=${encodeURIComponent(graphId)}`;
    if (until) url += `&until=${encodeURIComponent(until)}`;
    return this.get(url);
  }

  invalidateRelation(familyId, reason = '', graphId = 'default') {
    return this.post(`/api/v1/find/relations/${encodeURIComponent(familyId)}/invalidate?graph_id=${encodeURIComponent(graphId)}`, { reason });
  }

  getInvalidatedRelations(limit = 100, graphId = 'default') {
    return this.get(`/api/v1/find/relations/invalidated?limit=${limit}&graph_id=${encodeURIComponent(graphId)}`);
  }

  // --- Stats & Timeline ---
  getGraphStats(graphId = 'default') {
    return this.get(`/api/v1/find/graph-stats?graph_id=${encodeURIComponent(graphId)}`);
  }

  getEntityTimeline(familyId, graphId = 'default') {
    return this.get(`/api/v1/find/entities/${encodeURIComponent(familyId)}/timeline?graph_id=${encodeURIComponent(graphId)}`);
  }

  // Docs
  listDocs(graphId = 'default') {
    return this.get(`/api/v1/docs?graph_id=${encodeURIComponent(graphId)}`);
  }
  getDocContent(filename, graphId = 'default') {
    return this.get(`/api/v1/docs/${encodeURIComponent(filename)}?graph_id=${encodeURIComponent(graphId)}`);
  }

  // --- Agent Meta Query (Phase F) ---
  agentAsk(question, graphId = 'default') {
    return this.post('/api/v1/find/ask', { question, graph_id: graphId });
  }

  explainEntity(familyId, aspect, graphId = 'default') {
    return this.post('/api/v1/find/explain', { family_id: familyId, aspect, graph_id: graphId });
  }

  smartSuggestions(graphId = 'default') {
    return this.get(`/api/v1/find/suggestions?graph_id=${encodeURIComponent(graphId)}`);
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

  // Butler
  butlerReport(graphId = 'default') {
    return this.get(`/api/v1/butler/report?graph_id=${encodeURIComponent(graphId)}`);
  }
  butlerExecute(actions, dryRun = false, graphId = 'default') {
    return this.post(`/api/v1/butler/execute?graph_id=${encodeURIComponent(graphId)}`, {
      actions,
      dry_run: dryRun,
    });
  }

  // Dream
  dreamStatus(graphId = 'default') {
    return this.get(`/api/v1/find/dream/status?graph_id=${encodeURIComponent(graphId)}`);
  }
  dreamLogs(graphId = 'default', limit = 20) {
    return this.get(`/api/v1/find/dream/logs?graph_id=${encodeURIComponent(graphId)}&limit=${limit}`);
  }
  dreamLogDetail(cycleId, graphId = 'default') {
    return this.get(`/api/v1/find/dream/logs/${encodeURIComponent(cycleId)}?graph_id=${encodeURIComponent(graphId)}`);
  }
  dreamSeeds(graphId = 'default', strategy = 'random', count = 5) {
    return this.post(`/api/v1/find/dream/seeds?graph_id=${encodeURIComponent(graphId)}`, {
      strategy, count,
    });
  }

  // Quality / Maintenance
  qualityReport(graphId = 'default') {
    return this.get(`/api/v1/find/quality-report?graph_id=${encodeURIComponent(graphId)}`);
  }
  maintenanceHealth(graphId = 'default') {
    return this.get(`/api/v1/find/maintenance/health?graph_id=${encodeURIComponent(graphId)}`);
  }
  maintenanceCleanup(dryRun = false, graphId = 'default') {
    return this.post(`/api/v1/find/maintenance/cleanup?graph_id=${encodeURIComponent(graphId)}`, { dry_run: dryRun });
  }
  graphSummary(graphId = 'default') {
    return this.get(`/api/v1/find/graph-summary?graph_id=${encodeURIComponent(graphId)}`);
  }

  // Quick Search (unified search in one call)
  quickSearch(query, options = {}) {
    return this.post('/api/v1/find/quick-search', {
      query,
      graph_id: options.graphId || 'default',
      similarity_threshold: options.threshold ?? 0.4,
      max_entities: options.maxEntities ?? 10,
      max_relations: options.maxRelations ?? 20,
    });
  }

  // Find entity by name (fuzzy match)
  findEntityByName(name, options = {}) {
    return this.get(`/api/v1/find/entities/by-name/${encodeURIComponent(name)}?graph_id=${encodeURIComponent(options.graphId || 'default')}&threshold=${options.threshold || 0.7}&limit=${options.limit || 5}`);
  }

  // Create entity manually
  createEntity(data, graphId = 'default') {
    return this.post(`/api/v1/find/entities/create?graph_id=${encodeURIComponent(graphId)}`, data);
  }

  // Create relation manually
  createRelation(data, graphId = 'default') {
    return this.post(`/api/v1/find/relations/create?graph_id=${encodeURIComponent(graphId)}`, data);
  }

  // Recent activity
  recentActivity(graphId = 'default', limit = 10) {
    return this.get(`/api/v1/find/recent-activity?graph_id=${encodeURIComponent(graphId)}&limit=${limit}`);
  }

  // Refresh graph edges (after merges etc.)
  refreshGraphEdges(graphId = 'default') {
    return this.post(`/api/v1/find/entities/refresh-edges?graph_id=${encodeURIComponent(graphId)}`, {});
  }

  // Entity profile (entity + relations in one call)
  entityProfile(familyId, graphId = 'default') {
    return this.get(`/api/v1/find/entities/${encodeURIComponent(familyId)}/profile?graph_id=${encodeURIComponent(graphId)}`);
  }

  // Search concepts
  searchConcepts(query, options = {}) {
    return this.post('/api/v1/concepts/search', {
      query,
      graph_id: options.graphId || 'default',
      limit: options.limit || 20,
      role: options.role || '',
    });
  }

  // List concepts
  listConcepts(graphId = 'default', options = {}) {
    let q = `graph_id=${encodeURIComponent(graphId)}`;
    if (options.limit) q += `&limit=${options.limit}`;
    if (options.offset) q += `&offset=${options.offset}`;
    if (options.role) q += `&role=${encodeURIComponent(options.role)}`;
    return this.get(`/api/v1/concepts?${q}`);
  }

  // Batch entity profiles (up to 20)
  batchProfiles(familyIds, graphId = 'default') {
    return this.post('/api/v1/find/batch-profiles', { family_ids: familyIds, graph_id: graphId });
  }
}

// ---- Global State ----
const state = {
  api: new DeepDreamApi(),
  currentGraphId: localStorage.getItem('deepdream_graph_id') || localStorage.getItem('tmg_graph_id') || 'default',
  refreshTimers: {},
  currentPage: null,
  backendType: 'neo4j',
};

function isNeo4j() {
  return state.backendType === 'neo4j';
}

function setGraphId(id) {
  state.currentGraphId = id;
  localStorage.setItem('deepdream_graph_id', id);
  const sel = document.getElementById('graph-selector');
  if (sel) sel.value = id;
  // Detect backend type for new graph
  state.api.health(id).then(h => {
    if (h.data?.storage_backend) state.backendType = h.data.storage_backend;
  }).catch(() => {});
  // Re-render current page with new graph
  handleRoute();
}

// ---- Utilities ----
function getLocale() {
  var lang = (window.I18N && window.I18N.currentLang) || 'zh';
  return lang === 'en' ? 'en-US' : lang === 'ja' ? 'ja-JP' : 'zh-CN';
}

function formatDate(isoStr) {
  if (!isoStr) return '-';
  try {
    const d = new Date(isoStr);
    return d.toLocaleString(getLocale(), {
      year: 'numeric', month: '2-digit', day: '2-digit',
      hour: '2-digit', minute: '2-digit', second: '2-digit',
    });
  } catch { return isoStr; }
}

function formatDateMs(isoStr) {
  if (!isoStr) return '-';
  try {
    const d = new Date(isoStr);
    var ms = String(d.getMilliseconds()).padStart(3, '0');
    return d.toLocaleString(getLocale(), {
      year: 'numeric', month: '2-digit', day: '2-digit',
      hour: '2-digit', minute: '2-digit', second: '2-digit',
    }) + '.' + ms;
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

function renderMarkdown(text) {
  if (!text) return '';
  if (typeof marked === 'undefined') return escapeHtml(text);
  try {
    marked.setOptions({ breaks: true, gfm: true });
    var html = marked.parse(text);
    // Strip <script> tags for safety
    html = html.replace(/<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi, '');
    return html;
  } catch (e) {
    return escapeHtml(text);
  }
}

function escapeAttr(s) {
  return String(s).replace(/&/g,'&amp;').replace(/"/g,'&quot;').replace(/'/g,'&#39;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function getElapsed(startedAt, finishedAt) {
  if (!startedAt) return '-';
  let start = Number(startedAt);
  if (isNaN(start)) return '-';
  if (start < 4102444800000) start *= 1000;

  let end;
  if (finishedAt) {
    end = Number(finishedAt);
    if (end < 4102444800000) end *= 1000;
  } else {
    end = Date.now();
  }

  const diff = Math.max(0, Math.round((end - start) / 1000));
  return formatRelativeTime(diff);
}

function tripleProgressBar(opts) {
  var cols = [
    { pct: opts.smp, color: 'var(--primary)', label: t('dashboard.mainWindow'), text: opts.mainLabel },
    { pct: opts.s6p, color: 'var(--info)', label: t('dashboard.entityAlign'), text: opts.step6Label },
    { pct: opts.s7p, color: 'var(--warning)', label: t('dashboard.relationAlign'), text: opts.step7Label },
  ];
  var html = '<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:4px 12px;">';
  for (var ci = 0; ci < cols.length; ci++) {
    var c = cols[ci];
    html += '<div>'
      + '<div style="font-size:0.65rem;color:' + c.color + ';margin-bottom:2px;">' + c.label + '</div>'
      + '<div class="progress-bar" style="height:3px;"><div class="progress-bar-fill" style="width:' + (c.pct * 100).toFixed(2) + '%;background:' + c.color + ';"></div></div>'
      + '<div style="font-size:0.6rem;color:var(--text-muted);margin-top:1px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">' + escapeHtml(c.text || '-') + '</div>'
      + '</div>';
  }
  html += '</div>';
  if (opts.showOverall) {
    html = '<div style="min-width:240px;">'
      + '<div style="font-size:0.6rem;color:var(--text-muted);margin-bottom:4px;">' + t('memory.overallProgress') + ' ' + (opts.overallP * 100).toFixed(2) + '%</div>'
      + '<div style="margin-bottom:4px;">' + html + '</div>'
      + '</div>';
  }
  return html;
}

function renderVersionTimeline(opts) {
  var sorted = [...opts.versions].sort(function(a, b) {
    var ta = a.processed_time ? new Date(a.processed_time).getTime() : 0;
    var tb = b.processed_time ? new Date(b.processed_time).getTime() : 0;
    return tb - ta;
  });

  var items = sorted.map(function(v, i) {
    var prev = sorted[i + 1];
    var isActive = opts.isActiveCheck ? opts.isActiveCheck(v) : (i === 0);
    var diffHtml = opts.renderDiff ? opts.renderDiff(v, prev) : '';
    var headerHtml = opts.renderHeader ? opts.renderHeader(v, i, sorted, isActive) : '';
    var bodyHtml = opts.renderBody ? opts.renderBody(v) : '';

    return '<div style="position:relative;padding-left:1.5rem;padding-bottom:' + (i < sorted.length - 1 ? '1rem' : '0') + ';">'
      + (i < sorted.length - 1 ? '<div style="position:absolute;left:5px;top:12px;bottom:0;width:1px;background:var(--border-color);"></div>' : '')
      + '<div style="position:absolute;left:0;top:4px;width:11px;height:11px;border-radius:50%;background:' + (isActive ? 'var(--primary)' : 'var(--border-color)') + ';border:2px solid ' + (isActive ? 'var(--primary-hover)' : 'var(--border-hover)') + ';"></div>'
      + '<div style="cursor:pointer;" class="' + opts.toggleClass + '" data-version-idx="' + i + '">'
      + headerHtml
      + diffHtml
      + '</div>'
      + '<div class="' + opts.expandedIdPrefix + '" id="' + opts.expandedIdPrefix + '-' + i + '" style="display:none;margin-top:0.5rem;">'
      + bodyHtml
      + '</div>'
      + '</div>';
  }).join('');

  // Attach expand/collapse behavior
  setTimeout(function() {
    var container = opts.overlay.querySelector('#' + opts.containerId);
    if (!container) return;
    container.querySelectorAll('.' + opts.toggleClass).forEach(function(toggle) {
      toggle.addEventListener('click', function() {
        var idx = toggle.getAttribute('data-version-idx');
        var expanded = opts.overlay.querySelector('#' + opts.expandedIdPrefix + '-' + idx);
        if (expanded) {
          var isHidden = expanded.style.display === 'none';
          expanded.style.display = isHidden ? 'block' : 'none';
        }
      });
    });
  }, 0);

  return items;
}

function truncate(str, maxLen = 80) {
  if (!str) return '';
  return str.length > maxLen ? str.slice(0, maxLen) + '...' : str;
}

function statusBadge(status, phase) {
  if (status === 'running' && phase === 'pausing') {
    return `<span class="badge badge-warning">${escapeHtml(t('memory.statusPausing'))}</span>`;
  }
  if (status === 'running' && phase === 'cancelling') {
    return `<span class="badge badge-error">${escapeHtml(t('memory.statusCancelling'))}</span>`;
  }
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

// Make clickable table rows keyboard-accessible
// Usage: after rendering rows, call bindClickableRows(container)
function bindClickableRows(container) {
  if (!container) return;
  container.querySelectorAll('tr[data-family-id], tr[data-task-id]').forEach(row => {
    if (!row.hasAttribute('tabindex')) row.setAttribute('tabindex', '0');
    row.setAttribute('role', 'button');
    row.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        row.click();
      }
    });
  });
}

// ---- Router ----
const pages = {};
const _pageTitleKeys = {
  chat: 'nav.chat',
  dashboard: 'nav.dashboard',
  graph: 'nav.graph',
  memory: 'nav.memory',
  search: 'nav.search',
  entities: 'nav.entities',
  relations: 'nav.relations',
  episodes: 'nav.episodes',
  communities: 'nav.communities',
  dream: 'nav.dream',
  'api-test': 'nav.apiTest',
};

const _pageSloganKeys = {
  chat: 'slogan.chat',
  dashboard: 'slogan.dashboard',
  graph: 'slogan.graph',
  memory: 'slogan.memory',
  search: 'slogan.search',
  entities: 'slogan.entities',
  relations: 'slogan.relations',
  episodes: 'slogan.episodes',
  communities: 'slogan.communities',
  dream: 'slogan.dream',
  'api-test': 'slogan.apiTest',
};

function registerPage(name, module) {
  pages[name] = module;
}

function navigate(hash) {
  window.location.hash = hash;
}

async function handleRoute() {
  const hash = (window.location.hash || '#chat').slice(1);
  const [page, ...params] = hash.split('/').filter(Boolean);
  const pageName = page || 'chat';
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

  // Set title & breadcrumb
  const breadcrumb = document.getElementById('breadcrumb');
  const pageTitle = t(_pageTitleKeys[pageName] || '') || pageName;
  if (breadcrumb) {
    let bc = `<span id="page-title" class="font-semibold" style="color:var(--text-secondary);">${escapeHtml(pageTitle)}</span>`;
    if (params.length > 0) {
      const paramLabel = decodeURIComponent(params[0]);
      bc = `<a href="#${pageName}" class="breadcrumb-link" style="color:var(--text-muted);text-decoration:none;font-weight:500;">${escapeHtml(pageTitle)}</a>`
        + `<span style="color:var(--text-muted);font-size:0.75rem;">/</span>`
        + `<span id="page-title" class="font-semibold" style="color:var(--text-secondary);">${escapeHtml(truncate(paramLabel, 30))}</span>`;
    }
    breadcrumb.innerHTML = bc;
  } else {
    const titleEl = document.getElementById('page-title');
    if (titleEl) titleEl.textContent = pageTitle;
  }

  // Update sidebar slogan
  const sloganEl = document.getElementById('sidebar-slogan');
  if (sloganEl) sloganEl.textContent = t(_pageSloganKeys[pageName] || _pageSloganKeys['chat']);

  // Call destroy if exists
  if (state.currentPage && pages[state.currentPage] && pages[state.currentPage].destroy) {
    pages[state.currentPage].destroy();
  }

  state.currentPage = pageName;
  container.innerHTML = `<div class="page-enter"><div class="skeleton-card" style="margin-bottom:1rem;"><div class="skeleton skeleton-line w-1/4 h-8"></div><div class="skeleton skeleton-line w-3/4"></div><div class="skeleton skeleton-line w-1/2"></div></div><div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:1rem;"><div class="skeleton-card"><div class="skeleton skeleton-line w-1/2 h-6"></div><div class="skeleton skeleton-line w-full"></div></div><div class="skeleton-card"><div class="skeleton skeleton-line w-1/2 h-6"></div><div class="skeleton skeleton-line w-full"></div></div><div class="skeleton-card"><div class="skeleton skeleton-line w-1/2 h-6"></div><div class="skeleton skeleton-line w-full"></div></div></div></div>`;

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

    // Show/hide delete button: only show when more than one graph exists
    const delBtn = document.getElementById('graph-delete-btn');
    if (delBtn) {
      delBtn.style.display = graphs.length > 1 ? '' : 'none';
      if (window.lucide) lucide.createIcons();
    }
  } catch { /* ignore */ }
}

async function deleteCurrentGraph() {
  const graphId = state.currentGraphId;
  const graphs = Array.from(document.getElementById('graph-selector')?.options || []).map(o => o.value);

  // Safety: don't delete if it's the only graph
  if (graphs.length <= 1) {
    showToast('至少保留一个图谱', 'warning');
    return;
  }

  const confirmed = await showConfirm({
    title: '删除图谱',
    message: `确定要删除图谱 "${graphId}" 吗？此操作将永久删除该图谱的所有实体、关系和文档，不可恢复。`,
    confirmLabel: '删除',
    cancelLabel: '取消',
    destructive: true,
  });
  if (!confirmed) return;

  try {
    await state.api.deleteGraph(graphId);
    showToast(`图谱 "${graphId}" 已删除`, 'success');
    // Switch to first remaining graph
    const remaining = graphs.filter(g => g !== graphId);
    setGraphId(remaining[0] || 'default');
    loadGraphSelector();
  } catch (e) {
    showToast(`删除失败: ${e.message || e}`, 'error');
  }
}

async function clearCurrentGraph() {
  const graphId = state.currentGraphId;

  const confirmed = await showConfirm({
    title: t('graph.clearTitle'),
    message: t('graph.clearMessage', { name: graphId }),
    confirmLabel: t('graph.clearConfirm'),
    cancelLabel: t('common.cancel'),
  });
  if (!confirmed) return;

  try {
    await state.api.clearGraph(graphId);
    showToast(t('graph.clearSuccess', { name: graphId }), 'success');
    handleRoute();
  } catch (e) {
    showToast(t('graph.clearFailed') + `: ${e.message || e}`, 'error');
  }
}

// ---- Theme ----
function toggleTheme() {
  const html = document.documentElement;
  const isDark = html.getAttribute('data-theme') !== 'light';
  const newTheme = isDark ? 'light' : 'dark';
  html.setAttribute('data-theme', newTheme);
  localStorage.setItem('deepdream_theme', newTheme);
  updateThemeIcon(newTheme);
}

function updateThemeIcon(theme) {
  const darkIcon = document.getElementById('theme-icon-dark');
  const lightIcon = document.getElementById('theme-icon-light');
  if (darkIcon) darkIcon.style.display = theme === 'dark' ? '' : 'none';
  if (lightIcon) lightIcon.style.display = theme === 'light' ? '' : 'none';
}

function initTheme() {
  const saved = localStorage.getItem('deepdream_theme') || localStorage.getItem('tmg_theme') || 'dark';
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
  // Setup graph delete button
  const graphDelBtn = document.getElementById('graph-delete-btn');
  if (graphDelBtn) {
    graphDelBtn.addEventListener('click', deleteCurrentGraph);
  }
  // Setup graph clear button
  const graphClearBtn = document.getElementById('graph-clear-btn');
  if (graphClearBtn) {
    graphClearBtn.addEventListener('click', clearCurrentGraph);
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
  const neo4jNavItems = ['nav-episodes', 'nav-communities'];
  neo4jNavItems.forEach(id => {
    const el = document.getElementById(id);
    if (el) el.style.display = isNeo4j() ? '' : 'none';
  });

  // Mobile sidebar toggle with backdrop
  const toggle = document.getElementById('sidebar-toggle');
  const sidebar = document.getElementById('sidebar');
  if (toggle && sidebar) {
    // Create backdrop element for mobile
    const backdrop = document.createElement('div');
    backdrop.className = 'sidebar-backdrop';
    backdrop.id = 'sidebar-backdrop';
    document.body.appendChild(backdrop);

    toggle.addEventListener('click', () => {
      sidebar.classList.toggle('open');
      backdrop.classList.toggle('active', sidebar.classList.contains('open'));
    });
    backdrop.addEventListener('click', () => {
      sidebar.classList.remove('open');
      backdrop.classList.remove('active');
    });

    // Close sidebar on nav link click (mobile)
    sidebar.querySelectorAll('.sidebar-link').forEach(link => {
      link.addEventListener('click', () => {
        if (window.innerWidth < 768) {
          sidebar.classList.remove('open');
          backdrop.classList.remove('active');
        }
      });
    });

    // Restore collapsed state on desktop
    const collapsed = localStorage.getItem('deepdream_sidebar_collapsed') === 'true';
    if (collapsed && window.innerWidth >= 768) {
      sidebar.classList.add('collapsed');
    }
  }
});

// ---- Sidebar collapse (desktop) ----
function _toggleSidebarCollapse() {
  const sidebar = document.getElementById('sidebar');
  if (!sidebar) return;
  sidebar.classList.toggle('collapsed');
  localStorage.setItem('deepdream_sidebar_collapsed', sidebar.classList.contains('collapsed'));
}

// ---- Shared: render episode/doc modal with tab switcher ----
function _renderDocModal(sourceName, eventTime, cache, original) {
  const hasCache = !!cache;
  const hasOriginal = !!original;
  const showTabs = hasCache && hasOriginal;

  let body = `
    <div style="display:flex;flex-direction:column;gap:1rem;">
      <div style="display:grid;grid-template-columns:auto 1fr;gap:0.25rem 0.75rem;font-size:0.85rem;">
        <span style="color:var(--text-secondary);">${t('memory.taskSource')}:</span><span>${escapeHtml(sourceName)}</span>
        <span style="color:var(--text-secondary);">${t('memory.docTime')}:</span><span>${formatDate(eventTime)}</span>
      </div>`;

  if (showTabs) {
    body += `
      <div style="display:flex;gap:0; border-bottom:1px solid var(--border-color); margin-bottom:0;">
        <button class="doc-tab-btn active" data-doc-tab="cache" style="padding:0.5rem 1rem;font-size:0.85rem;border:none;background:none;cursor:pointer;border-bottom:2px solid var(--primary);color:var(--text-primary);font-weight:600;">${t('memory.cacheSummary')}</button>
        <button class="doc-tab-btn" data-doc-tab="original" style="padding:0.5rem 1rem;font-size:0.85rem;border:none;background:none;cursor:pointer;border-bottom:2px solid transparent;color:var(--text-muted);">${t('memory.originalText')}</button>
      </div>
      <div id="doc-tab-cache" class="doc-tab-panel" style="max-height:400px;overflow-y:auto;background:var(--bg-secondary);padding:0.75rem;border-radius:0.5rem;font-size:0.85rem;line-height:1.6;white-space:pre-wrap;word-break:break-word;">${escapeHtml(cache)}</div>
      <div id="doc-tab-original" class="doc-tab-panel" style="display:none;max-height:400px;overflow-y:auto;background:var(--bg-secondary);padding:0.75rem;border-radius:0.5rem;font-size:0.85rem;line-height:1.6;white-space:pre-wrap;word-break:break-word;">${escapeHtml(original)}</div>`;
  } else if (hasCache) {
    body += `
      <div>
        <h4 style="margin-bottom:0.5rem;">${t('memory.cacheSummary')}</h4>
        <div style="max-height:400px;overflow-y:auto;background:var(--bg-secondary);padding:0.75rem;border-radius:0.5rem;font-size:0.85rem;line-height:1.6;white-space:pre-wrap;word-break:break-word;">${escapeHtml(cache)}</div>
      </div>`;
  } else if (hasOriginal) {
    body += `
      <div>
        <h4 style="margin-bottom:0.5rem;">${t('memory.originalText')}</h4>
        <div style="max-height:400px;overflow-y:auto;background:var(--bg-secondary);padding:0.75rem;border-radius:0.5rem;font-size:0.85rem;line-height:1.6;white-space:pre-wrap;word-break:break-word;">${escapeHtml(original)}</div>
      </div>`;
  }

  body += '</div>';
  const { overlay } = showModal({
    title: t('memory.docContent') + ' - ' + escapeHtml(truncate(sourceName, 30)),
    content: body,
    size: 'lg',
  });

  // Tab switching
  if (showTabs) {
    overlay.querySelectorAll('.doc-tab-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const tab = btn.getAttribute('data-doc-tab');
        overlay.querySelectorAll('.doc-tab-btn').forEach(b => {
          b.classList.remove('active');
          b.style.borderBottom = '2px solid transparent';
          b.style.color = 'var(--text-muted)';
          b.style.fontWeight = '400';
        });
        btn.classList.add('active');
        btn.style.borderBottom = '2px solid var(--primary)';
        btn.style.color = 'var(--text-primary)';
        btn.style.fontWeight = '600';
        overlay.querySelectorAll('.doc-tab-panel').forEach(p => p.style.display = 'none');
        const panel = overlay.querySelector('#doc-tab-' + tab);
        if (panel) panel.style.display = '';
      });
    });
  }
}

// ---- Global: Show document content modal (by filename) ----
window.showDocContent = async function(filename) {
  if (!filename) return;
  try {
    const res = await state.api.getDocContent(filename, state.currentGraphId);
    const data = res.data || {};
    const meta = data.meta || {};
    _renderDocModal(
      meta.source_document || meta.doc_name || filename,
      meta.event_time || '-',
      data.cache || '',
      data.original || '',
    );
  } catch (err) {
    showToast(t('memory.loadDocContentFailed') + ': ' + err.message, 'error');
  }
};

// ---- Global: Show episode full detail modal (shared by graph explorer + episodes page) ----
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

    // Build entity name lookup for relation display
    const _entityNameMap = {};
    epEntities.forEach(e => { _entityNameMap[e.absolute_id] = e.name || e.family_id; });

    // Prefetch full relation data for display (batch by absolute_id)
    const _fullRelations = {};
    if (epRelations.length > 0) {
      const fetches = epRelations.map(r =>
        state.api.relationByAbsoluteId(r.absolute_id, state.currentGraphId)
          .then(res => { if (res.data) _fullRelations[r.absolute_id] = res.data; })
          .catch(() => {})
      );
      await Promise.all(fetches);
    }

    function _relationLabel(rel) {
      const full = _fullRelations[rel.absolute_id];
      if (!full) return escapeHtml(rel.family_id || '-');
      const e1 = _entityNameMap[full.entity1_absolute_id] || full.entity1_absolute_id || '?';
      const e2 = _entityNameMap[full.entity2_absolute_id] || full.entity2_absolute_id || '?';
      const content = truncate(full.content || '', 40);
      return `${escapeHtml(e1)} <span style="color:var(--text-muted);">→</span> ${escapeHtml(content)} <span style="color:var(--text-muted);">→</span> ${escapeHtml(e2)}`;
    }

    // ---- Build episode main view HTML ----
    function _buildMainHtml() {
      let body = `<div style="display:flex;flex-direction:column;gap:1rem;">`;
      body += `<div style="display:grid;grid-template-columns:auto 1fr;gap:0.25rem 0.75rem;font-size:0.85rem;">
        <span style="color:var(--text-secondary);">UUID:</span><span class="font-mono text-xs">${escapeHtml(ep.uuid || '')}</span>
        ${ep.episode_type ? `<span style="color:var(--text-secondary);">${t('episodes.episodeType')}:</span><span>${escapeHtml(ep.episode_type)}</span>` : ''}
        <span style="color:var(--text-secondary);">${t('common.source')}:</span><span>${escapeHtml(ep.source_document || '-')}</span>
        <span style="color:var(--text-secondary);">${t('relations.eventTime')}:</span><span class="mono" style="font-size:0.8125rem;">${formatDate(ep.event_time)}</span>
        ${ep.processed_time ? `<span style="color:var(--text-secondary);">${t('relations.processedTime')}:</span><span class="mono" style="font-size:0.8125rem;">${formatDateMs(ep.processed_time)}</span>` : ''}
      </div>`;

      // Build top-level tabs
      const mainTabs = [];
      if (hasCache) mainTabs.push('cache');
      if (hasOriginal) mainTabs.push('original');
      if (hasConcepts) mainTabs.push('concepts');
      if (mainTabs.length === 0) mainTabs.push('cache');
      const useMainTabs = mainTabs.length > 1;

      if (useMainTabs) {
        const tabBtns = mainTabs.map((tab, i) => {
          const labels = {
            cache: t('memory.cacheSummary'),
            original: t('memory.originalText'),
            concepts: `${t('episodes.tabConcepts')} (${epEntities.length + epRelations.length})`,
          };
          const active = i === 0;
          return `<button class="ep-main-tab ${active ? 'active' : ''}" data-ep-main="${tab}" style="padding:0.5rem 1rem;font-size:0.85rem;border:none;background:none;cursor:pointer;border-bottom:2px solid ${active ? 'var(--primary)' : 'transparent'};color:${active ? 'var(--text-primary)' : 'var(--text-muted)'};font-weight:${active ? '600' : '400'};">${labels[tab]}</button>`;
        }).join('');
        body += `<div style="display:flex;gap:0;border-bottom:1px solid var(--border-color);">${tabBtns}</div>`;
      }

      // Cache panel
      if (hasCache) {
        body += `<div id="ep-panel-cache" class="ep-main-panel" style="${useMainTabs && mainTabs[0] !== 'cache' ? 'display:none;' : ''}max-height:500px;overflow-y:auto;background:var(--bg-secondary);padding:0.75rem;border-radius:0.5rem;font-size:0.85rem;line-height:1.6;white-space:pre-wrap;word-break:break-word;">${renderMarkdown(cacheText)}</div>`;
      }

      // Original text panel
      if (hasOriginal) {
        body += `<div id="ep-panel-original" class="ep-main-panel" style="${useMainTabs && mainTabs[0] !== 'original' ? 'display:none;' : ''}max-height:500px;overflow-y:auto;background:var(--bg-secondary);padding:0.75rem;border-radius:0.5rem;font-size:0.85rem;line-height:1.6;white-space:pre-wrap;word-break:break-word;">${escapeHtml(originalText)}</div>`;
      }

      if (!hasCache && !hasOriginal) {
        body += `<div style="color:var(--text-muted);font-size:0.85rem;padding:1rem;">${t('episodes.noContent')}</div>`;
      }

      if (hasConcepts) {
        const entityList = epEntities.length > 0
          ? epEntities.map((ent, i) =>
              `<div class="flex items-center gap-2 p-2 rounded cursor-pointer" style="background:var(--bg-secondary);font-size:0.85rem;margin-bottom:4px;" data-ep-entity-idx="${i}">
                <i data-lucide="circle-dot" style="width:14px;height:14px;color:var(--primary);flex-shrink:0;"></i>
                <span class="font-medium">${escapeHtml(ent.name || ent.family_id || '-')}</span>
              </div>`
            ).join('')
          : `<p style="color:var(--text-muted);font-size:0.85rem;padding:0.5rem;">${t('episodes.noEntities')}</p>`;

        const relationList = epRelations.length > 0
          ? epRelations.map((rel, i) =>
              `<div class="flex items-center gap-2 p-2 rounded cursor-pointer" style="background:var(--bg-secondary);font-size:0.85rem;margin-bottom:4px;" data-ep-relation-idx="${i}">
                <i data-lucide="link" style="width:14px;height:14px;color:var(--warning);flex-shrink:0;"></i>
                <span>${_relationLabel(rel)}</span>
              </div>`
            ).join('')
          : `<p style="color:var(--text-muted);font-size:0.85rem;padding:0.5rem;">${t('episodes.noRelations')}</p>`;

        body += `<div id="ep-panel-concepts" class="ep-main-panel" style="display:none;">
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;">
            <div>
              <h4 style="margin-bottom:0.5rem;font-size:0.85rem;color:var(--text-secondary);">
                <i data-lucide="circle-dot" style="width:14px;height:14px;color:var(--primary);vertical-align:middle;"></i>
                ${t('episodes.entities')} (${epEntities.length})
              </h4>
              ${entityList}
            </div>
            <div>
              <h4 style="margin-bottom:0.5rem;font-size:0.85rem;color:var(--text-secondary);">
                <i data-lucide="link" style="width:14px;height:14px;color:var(--warning);vertical-align:middle;"></i>
                ${t('episodes.relations')} (${epRelations.length})
              </h4>
              ${relationList}
            </div>
          </div>
        </div>`;
      }

      body += '</div>';
      return body;
    }

    // ---- Build inline entity detail HTML ----
    function _buildEntityDetailHtml(entity) {
      let h = `<div style="display:flex;flex-direction:column;gap:0.75rem;">`;
      h += `<div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.25rem;">
        <button class="btn btn-secondary btn-sm" id="ep-back-btn" title="${t('common.back')}"><i data-lucide="arrow-left" style="width:14px;height:14px;"></i></button>
        <span class="badge badge-primary">${t('graph.entityDetail')}</span>
      </div>`;
      h += `<h3 style="font-size:1.1rem;font-weight:600;color:var(--text-primary);word-break:break-word;">${escapeHtml(entity.name || t('graph.unnamedEntity'))}</h3>`;
      h += `<div style="display:grid;grid-template-columns:auto 1fr;gap:0.25rem 0.75rem;font-size:0.85rem;">
        <span style="color:var(--text-secondary);">ID:</span><span class="mono text-xs">${escapeHtml(entity.absolute_id || '')}</span>
        <span style="color:var(--text-secondary);">Family:</span><span class="mono text-xs">${escapeHtml(entity.family_id || '')}</span>
        ${entity.episode_type ? `<span style="color:var(--text-secondary);">${t('episodes.episodeType')}:</span><span>${escapeHtml(entity.episode_type)}</span>` : ''}
        ${entity.event_time ? `<span style="color:var(--text-secondary);">${t('relations.eventTime')}:</span><span class="mono" style="font-size:0.8125rem;">${formatDate(entity.event_time)}</span>` : ''}
      </div>`;
      if (entity.content) {
        h += `<div>
          <span class="form-label" style="margin-bottom:0.25rem;">${t('graph.content')}</span>
          <div class="md-content" style="max-height:350px;overflow-y:auto;background:var(--bg-secondary);padding:0.75rem;border-radius:0.5rem;font-size:0.85rem;line-height:1.6;">${renderMarkdown(entity.content)}</div>
        </div>`;
      }
      h += '</div>';
      return h;
    }

    // ---- Build inline relation detail HTML ----
    function _buildRelationDetailHtml(relation) {
      const e1 = _entityNameMap[relation.entity1_absolute_id] || relation.entity1_absolute_id || '?';
      const e2 = _entityNameMap[relation.entity2_absolute_id] || relation.entity2_absolute_id || '?';
      let h = `<div style="display:flex;flex-direction:column;gap:0.75rem;">`;
      h += `<div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.25rem;">
        <button class="btn btn-secondary btn-sm" id="ep-back-btn" title="${t('common.back')}"><i data-lucide="arrow-left" style="width:14px;height:14px;"></i></button>
        <span class="badge" style="background:var(--info-dim);color:var(--info);">${t('graph.relationDetail')}</span>
      </div>`;
      h += `<h3 style="font-size:1.1rem;font-weight:600;color:var(--text-primary);word-break:break-word;">${escapeHtml(truncate(relation.content || t('graph.unnamedRelation'), 80))}</h3>`;
      h += `<div style="display:grid;grid-template-columns:auto 1fr;gap:0.25rem 0.75rem;font-size:0.85rem;">
        <span style="color:var(--text-secondary);">${t('graph.fromEntity')}:</span><span style="color:var(--info);cursor:pointer;" class="ep-nav-entity" data-nav-abs="${escapeHtml(relation.entity1_absolute_id)}">${escapeHtml(e1)}</span>
        <span style="color:var(--text-secondary);">${t('graph.toEntity')}:</span><span style="color:var(--info);cursor:pointer;" class="ep-nav-entity" data-nav-abs="${escapeHtml(relation.entity2_absolute_id)}">${escapeHtml(e2)}</span>
        <span style="color:var(--text-secondary);">${t('graph.relationId')}:</span><span class="mono text-xs">${escapeHtml(relation.family_id || '-')}</span>
        ${relation.event_time ? `<span style="color:var(--text-secondary);">${t('graph.eventTime')}:</span><span class="mono" style="font-size:0.8125rem;">${formatDate(relation.event_time)}</span>` : ''}
        ${relation.processed_time ? `<span style="color:var(--text-secondary);">${t('graph.processedTime')}:</span><span class="mono" style="font-size:0.8125rem;">${formatDateMs(relation.processed_time)}</span>` : ''}
      </div>`;
      if (relation.content) {
        h += `<div>
          <span class="form-label" style="margin-bottom:0.25rem;">${t('graph.content')}</span>
          <div class="md-content" style="max-height:350px;overflow-y:auto;background:var(--bg-secondary);padding:0.75rem;border-radius:0.5rem;font-size:0.85rem;line-height:1.6;">${renderMarkdown(relation.content)}</div>
        </div>`;
      }
      h += '</div>';
      return h;
    }

    // ---- Navigation stack for back button within modal ----
    const _navStack = []; // each entry = HTML string

    const { overlay, close } = showModal({
      title: t('episodes.detail'),
      content: _buildMainHtml(),
      size: 'lg',
    });
    const _bodyEl = overlay.querySelector('.modal-body');

    if (window.lucide) lucide.createIcons({ nodes: [overlay] });

    function _saveMainAndSwitch(detailHtml) {
      if (_bodyEl) _navStack.push(_bodyEl.innerHTML);
      if (_bodyEl) _bodyEl.innerHTML = detailHtml;
      if (window.lucide) lucide.createIcons({ nodes: [overlay] });
      _bindDetailEvents();
    }

    function _restoreView(html) {
      if (_bodyEl) _bodyEl.innerHTML = html;
      if (window.lucide) lucide.createIcons({ nodes: [overlay] });
      if (_bodyEl?.querySelector('.ep-main-tab')) {
        _bindMainEvents();
      } else {
        _bindDetailEvents();
      }
    }

    function _bindDetailEvents() {
      const backBtn = _bodyEl?.querySelector('#ep-back-btn');
      if (backBtn) {
        backBtn.addEventListener('click', () => {
          const prev = _navStack.pop();
          if (prev) _restoreView(prev);
        });
      }
      _bodyEl?.querySelectorAll('.ep-nav-entity').forEach(el => {
        el.addEventListener('click', () => {
          const absId = el.getAttribute('data-nav-abs');
          if (absId) _navigateToEntity(absId);
        });
      });
    }

    async function _navigateToEntity(absoluteId) {
      try {
        const res = await state.api.entityByAbsoluteId(absoluteId, state.currentGraphId);
        const entity = res.data;
        if (!entity) { showToast('Entity not found', 'error'); return; }
        _saveMainAndSwitch(_buildEntityDetailHtml(entity));
      } catch (err) {
        showToast(err.message, 'error');
      }
    }

    async function _navigateToRelation(absoluteId) {
      const full = _fullRelations[absoluteId];
      if (full) {
        // Also look up entity names if not in map
        if (!_entityNameMap[full.entity1_absolute_id]) {
          try {
            const r = await state.api.entityByAbsoluteId(full.entity1_absolute_id, state.currentGraphId);
            if (r.data) _entityNameMap[full.entity1_absolute_id] = r.data.name || r.data.family_id;
          } catch (_) {}
        }
        if (!_entityNameMap[full.entity2_absolute_id]) {
          try {
            const r = await state.api.entityByAbsoluteId(full.entity2_absolute_id, state.currentGraphId);
            if (r.data) _entityNameMap[full.entity2_absolute_id] = r.data.name || r.data.family_id;
          } catch (_) {}
        }
        _saveMainAndSwitch(_buildRelationDetailHtml(full));
      } else {
        try {
          const res = await state.api.relationByAbsoluteId(absoluteId, state.currentGraphId);
          const rel = res.data;
          if (!rel) { showToast('Relation not found', 'error'); return; }
          _fullRelations[absoluteId] = rel;
          // Lookup entity names
          for (const aid of [rel.entity1_absolute_id, rel.entity2_absolute_id]) {
            if (!_entityNameMap[aid]) {
              try {
                const r = await state.api.entityByAbsoluteId(aid, state.currentGraphId);
                if (r.data) _entityNameMap[aid] = r.data.name || r.data.family_id;
              } catch (_) {}
            }
          }
          _saveMainAndSwitch(_buildRelationDetailHtml(rel));
        } catch (err) {
          showToast(err.message, 'error');
        }
      }
    }

    function _bindMainEvents() {
      // Main tab switching
      overlay.querySelectorAll('.ep-main-tab').forEach(btn => {
        btn.addEventListener('click', () => {
          const tab = btn.getAttribute('data-ep-main');
          overlay.querySelectorAll('.ep-main-tab').forEach(b => {
            b.classList.remove('active');
            b.style.borderBottom = '2px solid transparent';
            b.style.color = 'var(--text-muted)';
            b.style.fontWeight = '400';
          });
          btn.classList.add('active');
          btn.style.borderBottom = '2px solid var(--primary)';
          btn.style.color = 'var(--text-primary)';
          btn.style.fontWeight = '600';
          overlay.querySelectorAll('.ep-main-panel').forEach(p => p.style.display = 'none');
          const panel = overlay.querySelector(`#ep-panel-${tab}`);
          if (panel) panel.style.display = '';
        });
      });

      // Entity click
      overlay.querySelectorAll('[data-ep-entity-idx]').forEach(el => {
        el.addEventListener('click', () => {
          const idx = parseInt(el.getAttribute('data-ep-entity-idx'), 10);
          const ref = epEntities[idx];
          if (ref) _navigateToEntity(ref.absolute_id);
        });
      });

      // Relation click
      overlay.querySelectorAll('[data-ep-relation-idx]').forEach(el => {
        el.addEventListener('click', () => {
          const idx = parseInt(el.getAttribute('data-ep-relation-idx'), 10);
          const ref = epRelations[idx];
          if (ref) _navigateToRelation(ref.absolute_id);
        });
      });
    }

    _bindMainEvents();
  } catch (err) {
    showToast(err.message, 'error');
  }
};

// Backward compat alias: graph explorer uses showEpisodeDoc(cacheId)
window.showEpisodeDoc = window.showEpisodeDetailModal;

// ---- Keyboard Shortcut System ----
const _shortcuts = [];
const _SHORTCUT_IGNORE_TAGS = new Set(['INPUT', 'TEXTAREA', 'SELECT']);

function registerShortcut(key, desc, handler, opts = {}) {
  _shortcuts.push({ key, desc, handler, global: opts.global || false, ctrlKey: opts.ctrlKey, shiftKey: opts.shiftKey, altKey: opts.altKey });
}

function _matchShortcut(e, s) {
  if (s.key !== e.key) return false;
  const needCtrl = s.ctrlKey !== undefined;
  if (needCtrl) {
    const ctrlOk = e.ctrlKey || e.metaKey;
    if (s.ctrlKey && !ctrlOk) return false;
    if (!s.ctrlKey && ctrlOk) return false;
  }
  if (s.shiftKey && !e.shiftKey) return false;
  if (s.altKey && !e.altKey) return false;
  return true;
}

document.addEventListener('keydown', (e) => {
  // Global shortcuts
  for (const s of _shortcuts) {
    if (s.global && _matchShortcut(e, s)) {
      e.preventDefault();
      s.handler(e);
      return;
    }
  }
  // Page-level shortcuts (skip when typing)
  const tag = document.activeElement?.tagName;
  if (_SHORTCUT_IGNORE_TAGS.has(tag)) return;
  for (const s of _shortcuts) {
    if (!s.global && _matchShortcut(e, s)) {
      e.preventDefault();
      s.handler(e);
      return;
    }
  }
});

// Register core shortcuts
registerShortcut('k', 'Open command palette', () => _openCommandPalette(), { ctrlKey: true, global: true });
registerShortcut('/', 'Focus search', () => {
  navigate('#search');
  setTimeout(() => { const el = document.getElementById('search-input'); if (el) el.focus(); }, 100);
}, { global: true });
registerShortcut('b', 'Toggle sidebar', () => {
  const sidebar = document.getElementById('sidebar');
  if (sidebar) sidebar.classList.toggle('collapsed');
}, { ctrlKey: true, global: true });
registerShortcut('?', 'Show shortcuts', () => _showShortcutsHelp(), { ctrlKey: true, global: true });

registerShortcut('1', 'Chat', () => navigate('#chat'), { altKey: true });
registerShortcut('2', 'Dashboard', () => navigate('#dashboard'), { altKey: true });
registerShortcut('3', 'Graph', () => navigate('#graph'), { altKey: true });
registerShortcut('4', 'Memory', () => navigate('#memory'), { altKey: true });
registerShortcut('5', 'Search', () => navigate('#search'), { altKey: true });
registerShortcut('6', 'Entities', () => navigate('#entities'), { altKey: true });
registerShortcut('7', 'Relations', () => navigate('#relations'), { altKey: true });
registerShortcut('8', 'Dream', () => navigate('#dream'), { altKey: true });

// ---- Command Palette (Ctrl+K) ----
function _openCommandPalette() {
  const commands = [
    { label: t('nav.chat') || 'Chat', icon: 'message-circle', action: () => navigate('#chat') },
    { label: t('nav.dashboard') || 'Dashboard', icon: 'layout-dashboard', action: () => navigate('#dashboard') },
    { label: t('nav.graph') || 'Graph', icon: 'network', action: () => navigate('#graph') },
    { label: t('nav.memory') || 'Memory', icon: 'database', action: () => navigate('#memory') },
    { label: t('nav.search') || 'Search', icon: 'search', action: () => navigate('#search') },
    { label: t('nav.entities') || 'Entities', icon: 'circle-dot', action: () => navigate('#entities') },
    { label: t('nav.relations') || 'Relations', icon: 'git-branch', action: () => navigate('#relations') },
    { label: t('nav.dream') || 'Dream', icon: 'moon', action: () => navigate('#dream') },
    { label: t('nav.apiTest') || 'API Test', icon: 'terminal', action: () => navigate('#api-test') },
    { label: t('common.toggleTheme') || 'Toggle Theme', icon: 'sun', action: () => toggleTheme() },
    { label: t('common.writeMemory') || 'Write Memory', icon: 'plus', action: () => navigate('#memory') },
    { label: t('common.healthCheck') || 'Health Check', icon: 'heart', action: async () => {
      try {
        await state.api.health(state.currentGraphId);
        showToast(t('common.statusOk') || 'API Connected', 'success');
      } catch (e) {
        showToast(t('common.statusError') || 'API Error: ' + e.message, 'error');
      }
    }},
  ];

  // Add page-specific commands from current page module
  const currentPageModule = pages[state.currentPage];
  if (currentPageModule && currentPageModule.getCommands) {
    commands.push(...currentPageModule.getCommands());
  }

  const overlay = document.createElement('div');
  overlay.className = 'modal-overlay';
  overlay.setAttribute('role', 'dialog');
  overlay.setAttribute('aria-label', 'Command Palette');
  overlay.style.background = 'rgba(0,0,0,0.5)';
  overlay.style.alignItems = 'flex-start';
  overlay.style.paddingTop = '18vh';
  overlay.innerHTML = `
    <div class="command-palette" style="background:var(--bg-surface);border:1px solid var(--border-color);border-radius:0.75rem;width:90%;max-width:500px;max-height:400px;display:flex;flex-direction:column;box-shadow:var(--shadow-lg);animation:modal-scale-in 0.12s ease;">
      <div style="padding:0.75rem;border-bottom:1px solid var(--border-color);display:flex;align-items:center;gap:0.5rem;">
        <i data-lucide="search" style="width:18px;height:18px;color:var(--text-muted);flex-shrink:0;"></i>
        <input type="text" id="command-palette-input" class="input" placeholder="${t('common.searchCommands') || 'Type a command...'}" style="border:none;background:transparent;box-shadow:none;padding:0;font-size:0.9rem;">
        <kbd style="font-size:0.7rem;padding:2px 6px;border-radius:4px;background:var(--bg-surface-hover);border:1px solid var(--border-color);color:var(--text-muted);font-family:var(--font-mono);">ESC</kbd>
      </div>
      <div id="command-palette-list" style="overflow-y:auto;padding:0.25rem;"></div>
      <div style="padding:0.5rem 0.75rem;border-top:1px solid var(--border-color);display:flex;gap:1rem;font-size:0.7rem;color:var(--text-muted);">
        <span><kbd style="padding:1px 4px;border-radius:3px;background:var(--bg-surface-hover);border:1px solid var(--border-color);">↑↓</kbd> navigate</span>
        <span><kbd style="padding:1px 4px;border-radius:3px;background:var(--bg-surface-hover);border:1px solid var(--border-color);">↵</kbd> select</span>
      </div>
    </div>
  `;

  let selectedIdx = 0;
  let filtered = [...commands];

  function render() {
    const list = overlay.querySelector('#command-palette-list');
    if (!list) return;
    list.innerHTML = filtered.length === 0
      ? '<div style="padding:1rem;text-align:center;color:var(--text-muted);font-size:0.85rem;">No results</div>'
      : filtered.map((cmd, i) => `
        <div class="command-item" data-cmd-idx="${i}" style="display:flex;align-items:center;gap:0.75rem;padding:0.5rem 0.75rem;border-radius:0.5rem;cursor:pointer;transition:background 0.1s ease;${i === selectedIdx ? 'background:var(--primary-dim);' : ''}">
          <i data-lucide="${cmd.icon || 'circle'}" style="width:16px;height:16px;color:var(--text-muted);flex-shrink:0;"></i>
          <span style="font-size:0.85rem;color:${i === selectedIdx ? 'var(--primary-hover)' : 'var(--text-primary)'};">${escapeHtml(cmd.label)}</span>
          ${cmd.desc ? `<span style="font-size:0.7rem;color:var(--text-muted);margin-left:auto;">${escapeHtml(cmd.desc)}</span>` : ''}
        </div>
      `).join('');
    if (window.lucide) lucide.createIcons({ nodes: [overlay] });
  }

  const input = overlay.querySelector('#command-palette-input');
  input.addEventListener('input', () => {
    const q = input.value.toLowerCase().trim();
    filtered = q ? commands.filter(c => c.label.toLowerCase().includes(q) || (c.desc || '').toLowerCase().includes(q)) : [...commands];
    selectedIdx = 0;
    render();
  });

  input.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      selectedIdx = Math.min(selectedIdx + 1, filtered.length - 1);
      render();
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      selectedIdx = Math.max(selectedIdx - 1, 0);
      render();
    } else if (e.key === 'Enter') {
      e.preventDefault();
      if (filtered[selectedIdx]) {
        close();
        filtered[selectedIdx].action();
      }
    } else if (e.key === 'Escape') {
      e.preventDefault();
      close();
    }
  });

  // Event delegation on the list container — works for dynamically rendered items
  const listEl = overlay.querySelector('#command-palette-list');
  listEl.addEventListener('click', (e) => {
    const item = e.target.closest('.command-item');
    if (!item) return;
    const idx = parseInt(item.dataset.cmdIdx);
    if (filtered[idx]) { close(); filtered[idx].action(); }
  });
  listEl.addEventListener('mousemove', (e) => {
    const item = e.target.closest('.command-item');
    if (!item) return;
    const idx = parseInt(item.dataset.cmdIdx);
    if (idx === selectedIdx) return;
    selectedIdx = idx;
    listEl.querySelectorAll('.command-item').forEach((el, i) => {
      el.style.background = i === selectedIdx ? 'var(--primary-dim)' : '';
    });
  });

  const close = () => overlay.remove();
  overlay.addEventListener('click', (e) => { if (e.target === overlay) close(); });

  document.body.appendChild(overlay);
  render();
  requestAnimationFrame(() => input.focus());
}

function _showShortcutsHelp() {
  const shortcutList = _shortcuts.map(s => {
    let keys = [];
    if (s.ctrlKey) keys.push('Ctrl');
    if (s.altKey) keys.push('Alt');
    if (s.shiftKey) keys.push('Shift');
    keys.push(s.key.length === 1 ? s.key.toUpperCase() : s.key);
    return { keys: keys.join(' + '), desc: s.desc };
  });

  const content = `
    <div style="display:flex;flex-direction:column;gap:0.25rem;">
      ${shortcutList.map(s => `
        <div style="display:flex;align-items:center;justify-content:space-between;padding:0.5rem 0;border-bottom:1px solid var(--border-color);">
          <span style="font-size:0.85rem;color:var(--text-secondary);">${escapeHtml(s.desc)}</span>
          <kbd style="font-size:0.75rem;padding:3px 8px;border-radius:4px;background:var(--bg-surface-hover);border:1px solid var(--border-color);font-family:var(--font-mono);color:var(--primary);">${escapeHtml(s.keys)}</kbd>
        </div>
      `).join('')}
    </div>
  `;
  showModal({ title: t('common.keyboardShortcuts') || 'Keyboard Shortcuts', content, size: 'sm' });
}

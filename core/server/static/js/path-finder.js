/* ==========================================
   PathFinder — Shared path finder component
   Used by search.js and relations.js
   ========================================== */

window.PathFinder = (function () {
  'use strict';

  // ---- Internal state ----
  let _container = null;
  let _opts = {};
  let _state = {};

  function _reset() {
    _state = {
      leftEntities: [],
      rightEntities: [],
      leftSelected: 0,
      rightSelected: 0,
      results: null,
      network: null,
      entityMap: {},
      relationMap: {},
      chainEntityMap: {},
      chainRelationMap: {},
      chainIdx: 0,
    };
  }

  // ---- Helpers ----
  function _t(key, params) {
    if (_opts.t) return _opts.t(key, params);
    if (typeof t === 'function') return t(key, params);
    return key;
  }

  // ---- Public API ----

  /**
   * Initialize the PathFinder in the given container.
   * @param {HTMLElement} container
   * @param {Object} options
   *   - api: DeepDreamApi instance
   *   - graphId: current graph ID
   *   - t(key, params): i18n translation function
   *   - onShowEntityDetail(entity): callback for entity click
   *   - onShowRelationDetail(relation, entityLookup): callback for relation click
   */
  function init(container, options) {
    destroy();
    _container = container;
    _opts = options || {};
    _opts.onShowEntityDetail = _opts.onShowEntityDetail || window.showEntityDetail;
    _opts.onShowRelationDetail = _opts.onShowRelationDetail || window.showRelationDetail;
    _reset();
    _render();
    _bindEvents();
  }

  function destroy() {
    if (_state.network) {
      _state.network.destroy();
      _state.network = null;
    }
    _reset();
    if (_container) _container.innerHTML = '';
    _container = null;
    _opts = {};
  }

  // ---- Render ----

  function _render() {
    if (!_container) return;

    _container.innerHTML = `
      <div id="pf-inputs" style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;">
        <div style="flex:1;min-width:160px;position:relative;">
          <span style="font-size:0.75rem;color:var(--text-muted);position:absolute;left:10px;top:50%;transform:translateY(-50%);pointer-events:none;">${_t('search.pathQueryA')}</span>
          <input type="text" id="pf-query-a" class="input" placeholder="${_t('search.pathQueryA')}" style="padding-left:60px;">
        </div>
        <i data-lucide="arrow-right" style="width:16px;height:16px;color:var(--text-muted);flex-shrink:0;"></i>
        <div style="flex:1;min-width:160px;position:relative;">
          <span style="font-size:0.75rem;color:var(--text-muted);position:absolute;left:10px;top:50%;transform:translateY(-50%);pointer-events:none;">${_t('search.pathQueryB')}</span>
          <input type="text" id="pf-query-b" class="input" placeholder="${_t('search.pathQueryB')}" style="padding-left:60px;">
        </div>
        <button class="btn btn-primary btn-sm" id="pf-find-btn">
          <i data-lucide="route" style="width:14px;height:14px;margin-right:4px;"></i>${_t('search.findPath')}
        </button>
      </div>
      <div id="pf-body"></div>
    `;

    if (window.lucide) lucide.createIcons({ nodes: [_container] });
  }

  function _renderBody() {
    const bodyEl = _container ? _container.querySelector('#pf-body') : null;
    if (!bodyEl) return;

    let body = '';

    // Entity pickers
    if (_state.leftEntities.length > 0 || _state.rightEntities.length > 0) {
      body += '<div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:16px;">';
      body += _renderEntityPicker('left', _state.leftEntities, _state.leftSelected);
      body += _renderEntityPicker('right', _state.rightEntities, _state.rightSelected);
      body += '</div>';
    }

    // Path results
    if (_state.results) {
      const data = _state.results;
      const pathLength = data.path_length;
      const totalPaths = data.total_shortest_paths || 0;
      const paths = data.paths || [];

      if (pathLength === -1) {
        body += `<div style="margin-top:16px;">${emptyState(_t('search.noPath'), 'unplug')}</div>`;
      } else if (pathLength === 0) {
        body += `<div style="margin-top:16px;">${emptyState(_t('search.sameEntity'), 'copy')}</div>`;
      } else {
        _state.chainIdx = 0;
        _state.chainEntityMap = {};
        _state.chainRelationMap = {};

        body += `<div style="margin-top:16px;margin-bottom:8px;font-size:0.8125rem;color:var(--text-muted);">
          <span class="badge badge-success">${_t('search.pathLength')}: ${pathLength}</span>
          <span class="badge badge-info" style="margin-left:0.25rem;">${_t('search.pathFound', { count: totalPaths })}</span>
        </div>`;

        // Graph canvas
        body += `<div id="pf-graph-canvas" style="width:100%;height:450px;margin-top:12px;border-radius:8px;background:var(--bg-input);border:1px solid var(--border-color);"></div>`;

        // Path list
        paths.forEach((p, idx) => {
          body += `<div class="card" style="padding:0.75rem 1rem;margin-bottom:0.5rem;">
            ${_buildPathChain(p, idx)}
          </div>`;
        });
      }
    }

    bodyEl.innerHTML = body || '';

    // Render graph after DOM update
    if (_state.results && _state.results.paths && _state.results.paths.length > 0) {
      setTimeout(_renderGraph, 50);
    }
  }

  function _renderEntityPicker(side, entities, selectedIdx) {
    const label = side === 'left' ? _t('search.leftResults') : _t('search.rightResults');
    if (entities.length === 0) {
      return `<div style="padding:12px;border:1px solid var(--border-color);border-radius:8px;">
        <div style="font-size:0.75rem;font-weight:600;color:var(--text-muted);margin-bottom:6px;">${label}</div>
        <div style="font-size:0.8rem;color:var(--text-muted);">${_t('search.selectEntity')}</div>
      </div>`;
    }

    const items = entities.map((item, i) => {
      const isSelected = i === selectedIdx;
      const name = item.entity.name || item.entity.family_id || '?';
      return `<div class="path-entity-item" data-side="${side}" data-index="${i}"
                   style="display:flex;align-items:center;gap:8px;padding:6px 10px;border-radius:6px;cursor:pointer;
                          ${isSelected ? 'background:var(--primary);color:#fff;' : 'background:var(--bg-secondary);color:var(--text-primary);'}
                          transition:background 0.15s;">
        <span class="mono" style="font-size:0.7rem;${isSelected ? 'opacity:0.8;' : 'color:var(--text-muted);'}">#${i + 1}</span>
        <span style="font-size:0.8125rem;flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;" title="${escapeHtml(item.entity.family_id || '')}">${escapeHtml(truncate(name, 20))}</span>
      </div>`;
    }).join('');

    return `<div style="padding:12px;border:1px solid var(--border-color);border-radius:8px;">
      <div style="font-size:0.75rem;font-weight:600;color:var(--text-muted);margin-bottom:8px;">${label} <span style="font-weight:400;">(${_t('search.selectEntity')})</span></div>
      <div style="display:flex;flex-direction:column;gap:4px;">
        ${items}
      </div>
    </div>`;
  }

  function _buildPathChain(path, pathIdx) {
    const entities = path.entities || [];
    const relations = path.relations || [];
    let html = '<div style="display:flex;align-items:center;gap:0.25rem;flex-wrap:wrap;padding:0.25rem 0;">';

    html += `<span style="font-size:0.75rem;color:var(--text-muted);font-weight:600;margin-right:0.25rem;">#${pathIdx + 1}</span>`;

    for (let i = 0; i < entities.length; i++) {
      const e = entities[i];
      const eName = e.name || e.family_id || '?';
      const eIdx = _state.chainIdx++;
      _state.chainEntityMap[eIdx] = e;
      html += `<span class="badge badge-primary" style="cursor:pointer;font-size:0.8125rem;padding:0.25rem 0.5rem;" title="${escapeHtml(e.family_id || '')}&#10;${escapeHtml(truncate(e.content || '', 100))}" data-pce="${eIdx}">${escapeHtml(truncate(eName, 20))}</span>`;
      if (i < relations.length) {
        const r = relations[i];
        const rIdx = _state.chainIdx++;
        _state.chainRelationMap[rIdx] = r;
        html += `<span style="color:var(--text-muted);font-size:0.75rem;padding:0 0.125rem;">→</span>`;
        html += `<span class="badge badge-info" style="cursor:pointer;font-size:0.75rem;padding:0.2rem 0.4rem;max-width:120px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;" title="${escapeHtml(r.content || '')}" data-pcr="${rIdx}">${escapeHtml(truncate(r.content || '—', 18))}</span>`;
        html += `<span style="color:var(--text-muted);font-size:0.75rem;padding:0 0.125rem;">→</span>`;
      }
    }
    html += '</div>';
    return html;
  }

  // ---- Graph ----

  function _renderGraph() {
    if (typeof vis === 'undefined') return;
    const canvas = document.getElementById('pf-graph-canvas');
    if (!canvas || !_state.results || !_state.results.paths || _state.results.paths.length === 0) return;

    if (_state.network) { _state.network.destroy(); _state.network = null; }
    _state.entityMap = {};
    _state.relationMap = {};

    const entityMap = new Map();
    for (const p of _state.results.paths) {
      for (const e of (p.entities || [])) {
        if (!entityMap.has(e.absolute_id)) entityMap.set(e.absolute_id, e);
      }
    }
    const allEntities = Array.from(entityMap.values());

    const edgeList = [];
    const relationMap = {};
    const edgeIdsSeen = new Set();
    for (const p of _state.results.paths) {
      const ents = p.entities || [];
      const rels = p.relations || [];
      for (let i = 0; i < ents.length - 1; i++) {
        const fromId = ents[i].absolute_id;
        const toId = ents[i + 1].absolute_id;
        const edgeKey = fromId < toId ? fromId + '||' + toId : toId + '||' + fromId;
        if (edgeIdsSeen.has(edgeKey)) continue;
        edgeIdsSeen.add(edgeKey);
        const rel = rels[i] || null;
        const edgeId = rel ? rel.absolute_id : ('edge_' + i + '_' + fromId.slice(-6));
        if (rel) relationMap[edgeId] = rel;
        edgeList.push({
          id: edgeId,
          from: fromId,
          to: toId,
          color: { color: '#4b5563', highlight: '#9ca3af', hover: '#6b7280' },
          smooth: { enabled: true, type: 'continuous', roundness: 0.2 },
          width: 1.5,
        });
      }
    }

    // Collect endpoint IDs from all paths (first & last entity of each path)
    const endpointIds = new Set();
    for (const p of _state.results.paths) {
      const ents = p.entities || [];
      if (ents.length > 0) {
        endpointIds.add(ents[0].absolute_id);
        endpointIds.add(ents[ents.length - 1].absolute_id);
      }
    }

    const hopMap = {};
    allEntities.forEach(e => {
      hopMap[e.absolute_id] = endpointIds.has(e.absolute_id) ? 0 : 1;
    });

    const { nodes, entityMap: eMap } = GraphUtils.buildNodes(allEntities, {
      colorMode: 'hop',
      hopMap: hopMap,
      unnamedLabel: _t('graph.unnamedEntity'),
    });
    _state.entityMap = eMap;
    _state.relationMap = relationMap;

    endpointIds.forEach(id => {
      const node = nodes.get(id);
      if (node) nodes.update({ id, size: 28 });
    });

    const edges = new vis.DataSet(edgeList);

    const options = {
      physics: GraphUtils.getPhysicsOptions(),
      interaction: GraphUtils.getInteractionOptions(),
      layout: { improvedLayout: true },
    };

    _state.network = new vis.Network(canvas, { nodes, edges }, options);

    _state.network.once('stabilizationIterationsDone', function () {
      _state.network.setOptions({ physics: { enabled: false } });
    });

    // Allow re-dragging: enable physics during drag so unfixed nodes respond to forces
    _state.network.on('dragStart', function (params) {
      if (params.nodes.length === 0) return;
      params.nodes.forEach(function (nodeId) {
        nodes.update({ id: nodeId, fixed: false });
      });
      _state.network.setOptions({ physics: { enabled: true } });
    });

    _state.network.on('dragEnd', function (params) {
      if (params.nodes.length === 0) return;
      params.nodes.forEach(function (nodeId) {
        var pos = _state.network.getPositions([nodeId])[nodeId];
        if (pos) {
          nodes.update({ id: nodeId, x: pos.x, y: pos.y, fixed: { x: true, y: true } });
        }
      });
      _state.network.setOptions({ physics: { enabled: false } });
    });

    _state.network.on('click', params => {
      const nodeId = params.nodes[0];
      const edgeId = params.edges[0];
      if (nodeId && _state.entityMap[nodeId]) {
        if (_opts.onShowEntityDetail) _opts.onShowEntityDetail(_state.entityMap[nodeId]);
      } else if (edgeId && _state.relationMap[edgeId]) {
        const entityLookup = {};
        allEntities.forEach(e => { entityLookup[e.absolute_id] = e.name || e.family_id || ''; });
        if (_opts.onShowRelationDetail) _opts.onShowRelationDetail(_state.relationMap[edgeId], entityLookup);
      }
    });
  }

  // ---- Search logic ----

  async function _executeSearch() {
    const queryA = (_container ? _container.querySelector('#pf-query-a') : null);
    const queryB = (_container ? _container.querySelector('#pf-query-b') : null);
    const qA = queryA ? queryA.value.trim() : '';
    const qB = queryB ? queryB.value.trim() : '';

    if (!qA || !qB) {
      showToast(_t('search.noQuery'), 'warning');
      return;
    }

    const btn = _container.querySelector('#pf-find-btn');
    if (btn) {
      btn.disabled = true;
      btn.innerHTML = `${spinnerHtml('spinner-sm')} ${_t('search.pathSearching')}`;
    }

    try {
      const api = _opts.api;
      const graphId = _opts.graphId || 'default';

      const [resA, resB] = await Promise.all([
        api.find(qA, { graphId, threshold: 0.3, maxEntities: 5, maxRelations: 0, expand: false }),
        api.find(qB, { graphId, threshold: 0.3, maxEntities: 5, maxRelations: 0, expand: false }),
      ]);

      const entitiesA = (resA.data?.entities || []).slice(0, 5);
      const entitiesB = (resB.data?.entities || []).slice(0, 5);

      _state.leftEntities = entitiesA.map((e, i) => ({ entity: e, rank: i + 1 }));
      _state.rightEntities = entitiesB.map((e, i) => ({ entity: e, rank: i + 1 }));
      _state.leftSelected = 0;
      _state.rightSelected = 0;
      _state.results = null;
      _state.chainIdx = 0;
      _state.chainEntityMap = {};
      _state.chainRelationMap = {};

      if (_state.leftEntities.length === 0 || _state.rightEntities.length === 0) {
        showToast(_t('search.noEntities'), 'warning');
      } else {
        await _findPathBetween();
      }

      _renderBody();

    } catch (err) {
      showToast(`${_t('search.pathSearchFailed')}: ${err.message}`, 'error');
    } finally {
      if (btn) {
        btn.disabled = false;
        btn.innerHTML = `<i data-lucide="route" style="width:14px;height:14px;margin-right:4px;"></i> ${_t('search.findPath')}`;
        if (window.lucide) lucide.createIcons({ nodes: [btn] });
      }
    }
  }

  async function _findPathBetween() {
    if (_state.leftSelected >= _state.leftEntities.length || _state.rightSelected >= _state.rightEntities.length) return;

    const leftEntity = _state.leftEntities[_state.leftSelected].entity;
    const rightEntity = _state.rightEntities[_state.rightSelected].entity;

    if (leftEntity.family_id === rightEntity.family_id) {
      _state.results = { path_length: 0, total_shortest_paths: 0, paths: [] };
      return;
    }

    try {
      const api = _opts.api;
      const graphId = _opts.graphId || 'default';

      // Neo4j: try Cypher fast path first, fall back to standard
      if (isNeo4j()) {
        try {
          const cypherRes = await api.shortestPathCypher(leftEntity.family_id, rightEntity.family_id, graphId, 6);
          const cypherPaths = cypherRes.data?.paths || [];
          if (cypherPaths.length > 0) {
            // Convert name arrays to entity/relation format for graph rendering
            const entityMap = new Map();
            const pathEntities = [];
            const pathRelations = [];
            for (const namePath of cypherPaths) {
              const entities = [];
              const relations = [];
              for (const name of namePath) {
                const absId = 'cypher_' + name;
                if (!entityMap.has(absId)) {
                  const e = { absolute_id: absId, family_id: name, name: name, content: '' };
                  entityMap.set(absId, e);
                }
                entities.push(entityMap.get(absId));
              }
              for (let i = 0; i < entities.length - 1; i++) {
                relations.push({ absolute_id: 'rel_' + i + '_' + entities[i].absolute_id });
              }
              pathEntities.push(entities);
              pathRelations.push(relations);
            }
            _state.results = {
              path_length: cypherPaths[0].length - 1,
              total_shortest_paths: cypherPaths.length,
              paths: cypherPaths.map((np, idx) => ({
                entities: pathEntities[idx],
                relations: pathRelations[idx],
                path_length: np.length - 1,
              })),
            };
            return;
          }
        } catch { /* fall through to standard search */ }
      }

      const res = await api.shortestPaths(leftEntity.family_id, rightEntity.family_id, graphId, {
        maxDepth: 6,
        maxPaths: 10,
      });
      _state.results = res.data || { path_length: -1, total_shortest_paths: 0, paths: [] };
    } catch (err) {
      _state.results = { path_length: -1, total_shortest_paths: 0, paths: [] };
      showToast(`${_t('search.pathSearchFailed')}: ${err.message}`, 'error');
    }
  }

  // ---- Events ----

  function _bindEvents() {
    if (!_container) return;

    const findBtn = _container.querySelector('#pf-find-btn');
    if (findBtn) findBtn.addEventListener('click', _executeSearch);

    ['pf-query-a', 'pf-query-b'].forEach(id => {
      const input = _container.querySelector('#' + id);
      if (input) input.addEventListener('keydown', (e) => { if (e.key === 'Enter') _executeSearch(); });
    });

    // Unified click delegation
    _container.addEventListener('click', (e) => {
      // Entity picker
      const item = e.target.closest('.path-entity-item');
      if (item) {
        const side = item.dataset.side;
        const index = parseInt(item.dataset.index, 10);
        if (side === 'left') _state.leftSelected = index;
        else _state.rightSelected = index;

        (async () => {
          await _findPathBetween();
          _renderBody();
        })();
        return;
      }

      // Path chain entity
      const pceEl = e.target.closest('[data-pce]');
      if (pceEl) {
        const idx = pceEl.dataset.pce;
        if (_state.chainEntityMap[idx] && _opts.onShowEntityDetail) {
          _opts.onShowEntityDetail(_state.chainEntityMap[idx]);
        }
        return;
      }

      // Path chain relation
      const pcrEl = e.target.closest('[data-pcr]');
      if (pcrEl) {
        const idx = pcrEl.dataset.pcr;
        if (_state.chainRelationMap[idx] && _opts.onShowRelationDetail) {
          const entityLookup = {};
          Object.values(_state.chainEntityMap).forEach(ent => {
            entityLookup[ent.absolute_id] = ent.name || ent.family_id || '';
          });
          _opts.onShowRelationDetail(_state.chainRelationMap[idx], entityLookup);
        }
        return;
      }
    });
  }

  return { init, destroy };
})();

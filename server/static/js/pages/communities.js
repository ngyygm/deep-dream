/* ==========================================
   Communities Page - Community Detection
   ========================================== */
registerPage('communities', (function () {
  'use strict';

  let _container = null;
  let _communities = [];
  let _detectResult = null;
  let _loading = false;

  // ---- Detail page (graph-explorer) state ----
  let _detailNetwork = null;
  let _entityMap = {};        // uuid -> entity data
  let _relationMap = {};      // relation_uuid -> relation data
  let _versionCounts = {};    // entity_id -> version count
  let _pinnedNodePositions = {}; // uuid -> { x, y }

  // Community-scoped data
  let _commEntities = [];     // all community member entities
  let _commRelations = [];    // all community internal relations
  let _commEntityMap = {};    // uuid -> entity (from community members)

  // Focus state
  let _focusAbsoluteId = null;
  let _currentVersions = [];
  let _currentVersionIdx = 0;
  let _relationScope = 'accumulated';

  async function _detect() {
    if (_loading) return;
    _loading = true;
    _updateUI();

    try {
      const algorithm = document.getElementById('comm-algorithm')?.value || 'louvain';
      const resolution = parseFloat(document.getElementById('comm-resolution')?.value || '1.0');

      const res = await state.api.detectCommunities(state.currentGraphId, algorithm, resolution);
      _detectResult = res.data || {};
      showToast(t('communities.detectSuccess', { count: _detectResult.total_communities, time: _detectResult.elapsed_seconds }), 'success');
      await _loadCommunities();
    } catch (err) {
      showToast(err.message, 'error');
    } finally {
      _loading = false;
      _updateUI();
    }
  }

  async function _loadCommunities() {
    try {
      const minSize = parseInt(document.getElementById('comm-min-size')?.value || '3');
      const res = await state.api.listCommunities(state.currentGraphId, minSize, 50);
      _communities = res.data?.communities || [];
      _renderCards();
    } catch (err) {
      showToast(err.message, 'error');
    }
  }

  async function _clearCommunities() {
    if (!confirm(t('communities.clearConfirm'))) return;
    try {
      await state.api.clearCommunities(state.currentGraphId);
      _communities = [];
      _detectResult = null;
      showToast(t('communities.clearSuccess'), 'success');
      _renderCards();
      _updateUI();
    } catch (err) {
      showToast(err.message, 'error');
    }
  }

  function _updateUI() {
    const btn = document.getElementById('comm-detect-btn');
    if (btn) {
      btn.disabled = _loading;
      btn.innerHTML = _loading
        ? `<div class="spinner spinner-sm" style="margin-right:6px;"></div>${t('communities.detecting')}`
        : `<i data-lucide="scan" style="width:16px;height:16px;"></i>${t('communities.detect')}`;
      if (window.lucide) lucide.createIcons();
    }
  }

  function _renderCards() {
    const cardsEl = _container.querySelector('#comm-cards');
    const statsEl = _container.querySelector('#comm-stats');

    if (_detectResult) {
      const sizes = _detectResult.community_sizes || [];
      statsEl.innerHTML = `
        <div class="card p-3 flex items-center gap-4 text-sm flex-wrap">
          <span style="color:var(--text-muted);">${t('communities.stats')}:</span>
          <span><strong>${_detectResult.total_communities}</strong> ${t('communities.communitiesCount')}</span>
          <span>${t('communities.largest')}: <strong>${sizes[0] || 0}</strong></span>
          <span>${t('communities.avgSize')}: <strong>${sizes.length ? Math.round(sizes.reduce((a, b) => a + b, 0) / sizes.length) : 0}</strong></span>
          <span>${t('communities.elapsed')}: <strong>${_detectResult.elapsed_seconds}s</strong></span>
        </div>`;
    } else {
      statsEl.innerHTML = '';
    }

    if (_communities.length === 0) {
      cardsEl.innerHTML = emptyState(t('communities.noCommunities'), 'layout-grid');
      if (window.lucide) lucide.createIcons();
      return;
    }

    const palette = [
      '#ef4444', '#f59e0b', '#10b981', '#3b82f6', '#8b5cf6',
      '#ec4899', '#14b8a6', '#f97316', '#6366f1', '#84cc16',
    ];

    let html = '<div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">';
    for (const c of _communities) {
      const color = palette[c.community_id % palette.length];
      const topMembers = (c.members || []).slice(0, 5).map(m => m.name || m.entity_id).join(', ');
      html += `
        <div class="card p-4 cursor-pointer" onclick="window.location.hash='#communities/${c.community_id}'">
          <div class="flex items-center gap-2 mb-2">
            <div style="width:12px;height:12px;border-radius:50%;background:${color};flex-shrink:0;"></div>
            <span class="font-semibold">${t('communities.community')} #${c.community_id}</span>
            <span class="badge badge-info">${c.size} ${t('common.entities')}</span>
          </div>
          <p class="text-xs" style="color:var(--text-muted);line-height:1.5;">${escapeHtml(topMembers)}</p>
        </div>`;
    }
    html += '</div>';
    cardsEl.innerHTML = html;
    if (window.lucide) lucide.createIcons();
  }

  // ================================================
  //  Full-page detail view — graph-explorer style
  // ================================================

  async function _renderDetailPage(container, cid) {
    const palette = [
      '#ef4444', '#f59e0b', '#10b981', '#3b82f6', '#8b5cf6',
      '#ec4899', '#14b8a6', '#f97316', '#6366f1', '#84cc16',
    ];
    const color = palette[cid % palette.length];

    // Render skeleton
    container.innerHTML = `
      <div class="page-enter">
        <!-- Top bar -->
        <div class="card mb-4">
          <div class="card-header">
            <div style="display:flex;align-items:center;gap:0.5rem;">
              <button onclick="window.location.hash='#communities'" class="btn btn-ghost btn-sm">
                <i data-lucide="arrow-left" style="width:16px;height:16px;"></i>${t('common.back')}
              </button>
              <div style="width:12px;height:12px;border-radius:50%;background:${color};"></div>
              <span class="badge badge-primary mono">${t('communities.community')} #${cid}</span>
              <span id="comm-focus-badge" class="badge badge-warning" style="display:none;">
                <i data-lucide="crosshair" style="width:12px;height:12px;margin-right:2px;"></i>
                ${t('graph.focusMode')}
              </span>
            </div>
            <span id="comm-graph-stats" class="mono" style="font-size:0.8125rem;color:var(--text-muted);"></span>
          </div>
          <div style="display:flex;align-items:center;gap:0.5rem;">
            <button class="btn btn-primary btn-sm" id="comm-load-btn">
              <i data-lucide="refresh-cw" style="width:14px;height:14px;"></i>${t('graph.loadGraph')}
            </button>
            <button class="btn btn-secondary btn-sm" id="comm-exit-focus-btn" style="display:none;">
              <i data-lucide="maximize-2" style="width:14px;height:14px;"></i>${t('graph.exitFocus')}
            </button>
          </div>
        </div>

        <!-- Main body: graph + sidebar -->
        <div class="flex gap-4" style="height:calc(100vh - 240px);min-height:400px;">
          <!-- Graph canvas -->
          <div class="flex-1 relative" style="min-width:0;">
            <div id="comm-detail-graph" style="width:100%;height:100%;"></div>
            <div id="comm-graph-loading" class="absolute inset-0 flex items-center justify-center" style="background:var(--bg-input);border-radius:0.5rem;">
              ${spinnerHtml()}
            </div>
          </div>

          <!-- Detail sidebar -->
          <div style="width:30%;min-width:280px;max-width:420px;">
            <div class="card h-full flex flex-col">
              <div class="card-header">
                <span class="card-title">${t('common.detail')}</span>
              </div>
              <div id="comm-detail-content" style="overflow-y:auto;flex:1;">
                ${emptyState(t('common.clickToView'), 'mouse-pointer-click')}
              </div>
            </div>
          </div>
        </div>
      </div>
    `;
    if (window.lucide) lucide.createIcons();

    // Bind buttons
    document.getElementById('comm-load-btn').addEventListener('click', () => _loadCommunityGraph(cid));
    document.getElementById('comm-exit-focus-btn').addEventListener('click', () => _exitFocus());

    // Initial load
    await _loadCommunityGraph(cid);
  }

  // ---- Load community data and build graph ----

  async function _loadCommunityGraph(cid) {
    const loadingEl = document.getElementById('comm-graph-loading');
    const statsEl = document.getElementById('comm-graph-stats');
    if (loadingEl) loadingEl.style.display = 'flex';
    if (statsEl) statsEl.textContent = t('common.loading');

    try {
      let [community, graphData] = await Promise.all([
        state.api.getCommunity(cid, state.currentGraphId),
        state.api.getCommunityGraph(cid, state.currentGraphId),
      ]);

      community = community.data || {};
      graphData = graphData.data || {};

      const members = community.members || [];
      const rawRelations = community.relations || [];

      // Build entity map from community members (uuid = absolute_id)
      _commEntities = members;
      _commEntityMap = {};
      for (const m of members) {
        _commEntityMap[m.uuid] = m;
      }

      // Convert community relations to standard relation format with absolute_id fields
      _commRelations = rawRelations.map(r => ({
        absolute_id: r.relation_uuid || r.source_uuid + '_' + r.target_uuid,
        entity1_absolute_id: r.source_uuid,
        entity2_absolute_id: r.target_uuid,
        content: r.content || '',
      }));

      // Convert graph nodes to entity format
      const graphNodes = (graphData.nodes || []).map(n => ({
        absolute_id: n.uuid,
        entity_id: n.entity_id,
        name: n.name,
        content: _commEntityMap[n.uuid]?.content || '',
      }));

      // Convert graph edges
      const graphEdges = (graphData.edges || []).map(e => ({
        absolute_id: (e.source_uuid || '') + '_' + (e.target_uuid || ''),
        entity1_absolute_id: e.source_uuid,
        entity2_absolute_id: e.target_uuid,
        content: e.content || '',
      }));

      // Build entity map for graph interactions
      _entityMap = {};
      for (const e of graphNodes) {
        _entityMap[e.absolute_id] = e;
      }

      // Fetch version counts
      const allEntityIds = [...new Set(graphNodes.map(e => e.entity_id))];
      try {
        const vcRes = await state.api.entityVersionCounts(allEntityIds, state.currentGraphId);
        _versionCounts = vcRes.data || {};
      } catch (_) { _versionCounts = {}; }

      // Clear focus
      _focusAbsoluteId = null;
      _currentVersions = [];
      _currentVersionIdx = 0;
      const exitBtn = document.getElementById('comm-exit-focus-btn');
      if (exitBtn) exitBtn.style.display = 'none';
      const focusBadge = document.getElementById('comm-focus-badge');
      if (focusBadge) focusBadge.style.display = 'none';

      _buildGraph(graphNodes, graphEdges, null, null);

      if (statsEl) {
        statsEl.textContent = t('graph.loaded', { entities: graphNodes.length, relations: graphEdges.length });
      }

      // Reset detail sidebar
      const detailContent = document.getElementById('comm-detail-content');
      if (detailContent) {
        detailContent.innerHTML = emptyState(t('common.clickToView'), 'mouse-pointer-click');
        if (window.lucide) lucide.createIcons();
      }
    } catch (err) {
      showToast(err.message, 'error');
      if (statsEl) statsEl.textContent = t('common.error');
    } finally {
      if (loadingEl) loadingEl.style.display = 'none';
    }
  }

  // ---- Build vis-network graph (same pattern as graph.js) ----

  function _buildGraph(entities, relations, highlightAbsId, hopMap, inheritedRelationIds) {
    _entityMap = {};
    _relationMap = {};

    const versionLabel = highlightAbsId && _currentVersions.length > 1
      ? { idx: _currentVersionIdx + 1, total: _currentVersions.length }
      : null;

    const colorMode = hopMap ? 'hop' : 'default';
    const { nodes, entityMap: eMap, nodeIds } = GraphUtils.buildNodes(entities, {
      colorMode: colorMode,
      versionCounts: _versionCounts,
      hopMap: hopMap,
      highlightAbsId: highlightAbsId,
      versionLabel: versionLabel,
      unnamedLabel: t('graph.unnamedEntity'),
    });

    const visibleNodeIds = new Set();
    nodes.forEach((node) => {
      visibleNodeIds.add(node.id);
      const pinned = _pinnedNodePositions[node.id];
      if (pinned) {
        nodes.update({
          id: node.id,
          x: pinned.x,
          y: pinned.y,
          fixed: { x: true, y: true },
        });
      }
    });
    for (const nodeId of Object.keys(_pinnedNodePositions)) {
      if (!visibleNodeIds.has(nodeId)) delete _pinnedNodePositions[nodeId];
    }
    _entityMap = eMap;

    const { edges, relationMap: rMap } = GraphUtils.buildEdges(relations, nodeIds, {
      inheritedRelationIds: inheritedRelationIds,
    });
    _relationMap = rMap;

    const graphContainer = document.getElementById('comm-detail-graph');
    if (!graphContainer) return;

    if (_detailNetwork) {
      _detailNetwork.destroy();
      _detailNetwork = null;
    }

    const options = {
      physics: GraphUtils.getPhysicsOptions(),
      interaction: GraphUtils.getInteractionOptions(),
      layout: { improvedLayout: true },
    };

    _detailNetwork = new vis.Network(graphContainer, { nodes, edges }, options);

    if (highlightAbsId) {
      _detailNetwork.once('stabilizationIterationsDone', () => {
        _detailNetwork.focus(highlightAbsId, { scale: 1.2, animation: { duration: 500, easingFunction: 'easeInOutQuad' } });
      });
    }

    _detailNetwork.on('click', (params) => {
      const nodeId = params.nodes[0];
      const edgeId = params.edges[0];
      if (nodeId) {
        _showEntityDetail(nodeId);
      } else if (edgeId) {
        _showRelationDetail(edgeId);
      }
    });

    _detailNetwork.on('dragEnd', (params) => {
      if (!params.nodes || params.nodes.length === 0) return;
      const positions = _detailNetwork.getPositions(params.nodes);
      params.nodes.forEach((nodeId) => {
        const pos = positions[nodeId];
        if (!pos) return;
        _pinnedNodePositions[nodeId] = { x: pos.x, y: pos.y };
        nodes.update({
          id: nodeId,
          x: pos.x,
          y: pos.y,
          fixed: { x: true, y: true },
        });
      });
    });
  }

  // ---- Multi-hop BFS with inherited relation detection (scoped to community) ----
  //   Core principle: can only see the PAST, cannot predict the FUTURE.
  //   - Base set = relations that existed at the viewed version (accumulated up to that point)
  //   - Among base set, those still in latest version → solid (survived)
  //   - Among base set, those NOT in latest version → dashed amber (inherited, lost over time)
  //   - Relations that only exist in newer versions → NOT shown (future, cannot predict)

  async function _fetchMultiHop(startAbsId, startEntityId, hopLevel) {
    const graphId = state.currentGraphId;
    const hopMap = { [startAbsId]: 0 };
    const relationSet = new Map();
    const inheritedRelationIds = new Set();
    let frontier = [{ absId: startAbsId, entityId: startEntityId }];

    const focusedAbsIds = new Set(_currentVersions.map(v => v.absolute_id));
    focusedAbsIds.add(startAbsId);

    // Track absolute_id -> entity_id for all discovered nodes
    const absToEntityId = {};
    if (startEntityId) absToEntityId[startAbsId] = startEntityId;
    for (const v of _currentVersions) absToEntityId[v.absolute_id] = startEntityId;

    for (let h = 1; h <= hopLevel; h++) {
      const nextFrontier = [];

      for (const node of frontier) {
        // Step 1: Fetch relations up to this version (can only see the past)
        let versionRels = [];
        try {
          let res;
          if (node.entityId) {
            res = await state.api.entityRelations(node.entityId, graphId, {
              maxVersionAbsoluteId: node.absId
            });
          } else {
            res = await state.api.entityOneHop(node.absId, graphId);
          }
          versionRels = res.data || [];
        } catch (_) {}

        // Step 2: Fetch latest relations for comparison (only for inherited detection)
        let latestRelIds = new Set();
        if (!_onlyCurrentVersion && node.entityId) {
          try {
            const latestRes = await state.api.entityRelations(node.entityId, graphId);
            latestRelIds = new Set((latestRes.data || []).map(r => r.absolute_id));
          } catch (_) {}
        }

        // Step 3: Add version relations as base set (only past, no future)
        for (const r of versionRels) {
          relationSet.set(r.absolute_id, r);
          const otherAbsId = r.entity1_absolute_id === node.absId
            ? r.entity2_absolute_id : r.entity1_absolute_id;
          if (otherAbsId && !(otherAbsId in hopMap)) {
            hopMap[otherAbsId] = h;
            nextFrontier.push({ absId: otherAbsId, entityId: null });
          }
        }

        // Step 4: Mark relations that existed at this version but NOT in latest as inherited
        // (relations lost over time — inherited from the past)
        if (!_onlyCurrentVersion && node.entityId) {
          for (const r of versionRels) {
            if (!latestRelIds.has(r.absolute_id)) {
              inheritedRelationIds.add(r.absolute_id);
            }
          }
        }

        // Resolve entity_id for new frontier nodes
        for (const item of nextFrontier) {
          if (item.entityId) continue;
          const ent = _entityMap[item.absId] || _commEntityMap[item.absId];
          if (ent) {
            item.entityId = ent.entity_id;
            absToEntityId[item.absId] = ent.entity_id;
          }
        }
      }

      frontier = nextFrontier;
    }

    // ---- Post-BFS: resolve entities, deduplicate, remap relations ----

    // Build entity_id -> latest absId from community data
    const entityIdToLatest = {};
    for (const e of _commEntities) {
      entityIdToLatest[e.entity_id] = e.uuid;
    }
    // Override: focused entity always maps to the version being viewed
    if (startEntityId) entityIdToLatest[startEntityId] = startAbsId;

    // Collect entities
    const rawEntities = [];
    const resolvedIds = new Set();
    for (const absId of Object.keys(hopMap)) {
      if (resolvedIds.has(absId)) continue;
      const ent = _entityMap[absId] || _commEntityMap[absId];
      if (ent) {
        rawEntities.push(ent);
        resolvedIds.add(absId);
        absToEntityId[absId] = ent.entity_id;
      }
    }

    // Deduplicate: keep only latest version per entity_id
    const dedupedEntities = [];
    const seenEntityIds = new Set();
    for (const ent of rawEntities) {
      const latestAbsId = entityIdToLatest[ent.entity_id];
      if (!latestAbsId || ent.absolute_id === latestAbsId) {
        if (!seenEntityIds.has(ent.entity_id)) {
          dedupedEntities.push(ent);
          seenEntityIds.add(ent.entity_id);
        }
      }
    }

    // Resolve unknown endpoints
    const unknownEndpoints = new Set();
    for (const r of relationSet.values()) {
      if (!resolvedIds.has(r.entity1_absolute_id)) unknownEndpoints.add(r.entity1_absolute_id);
      if (!resolvedIds.has(r.entity2_absolute_id)) unknownEndpoints.add(r.entity2_absolute_id);
    }
    if (unknownEndpoints.size > 0) {
      const promises = [...unknownEndpoints].slice(0, 30).map(async (absId) => {
        try {
          const res = await state.api.entityByAbsoluteId(absId, graphId);
          if (res.data) {
            _entityMap[absId] = res.data;
            absToEntityId[absId] = res.data.entity_id;
          }
        } catch (_) {}
      });
      await Promise.all(promises);
    }

    // Remap relation endpoints to latest visible versions
    const relations = [];
    for (const r of relationSet.values()) {
      let e1 = r.entity1_absolute_id;
      let e2 = r.entity2_absolute_id;
      let skip = false;

      const eid1 = absToEntityId[e1];
      if (eid1 && entityIdToLatest[eid1]) {
        e1 = entityIdToLatest[eid1];
      } else {
        skip = true;
      }
      const eid2 = absToEntityId[e2];
      if (eid2 && entityIdToLatest[eid2]) {
        e2 = entityIdToLatest[eid2];
      } else {
        skip = true;
      }

      if (skip) continue;

      // Add remapped entities if not already present
      for (const absId of [e1, e2]) {
        const ent = _commEntityMap[absId] || _entityMap[absId];
        if (ent && !seenEntityIds.has(ent.entity_id)) {
          dedupedEntities.push(ent);
          seenEntityIds.add(ent.entity_id);
        }
      }

      // Fix hop levels when old endpoint is remapped to latest
      const oldHop1 = hopMap[r.entity1_absolute_id];
      if (oldHop1 !== undefined) {
        hopMap[e1] = Math.min(hopMap[e1] ?? Infinity, oldHop1);
      }
      const oldHop2 = hopMap[r.entity2_absolute_id];
      if (oldHop2 !== undefined) {
        hopMap[e2] = Math.min(hopMap[e2] ?? Infinity, oldHop2);
      }

      relations.push({ ...r, entity1_absolute_id: e1, entity2_absolute_id: e2 });
    }

    // Filter: only keep connected entities
    const connectedNodeIds = new Set();
    for (const r of relations) {
      connectedNodeIds.add(r.entity1_absolute_id);
      connectedNodeIds.add(r.entity2_absolute_id);
    }
    const finalEntities = dedupedEntities.filter(e => connectedNodeIds.has(e.absolute_id));

    return { hopMap, entities: finalEntities, relations, inheritedRelationIds };
  }

  // ---- Focus on entity within community scope ----

  async function _focusOnEntity(absoluteId) {
    const loadingEl = document.getElementById('comm-graph-loading');
    if (loadingEl) loadingEl.style.display = 'flex';

    try {
      let entity = _entityMap[absoluteId];
      if (!entity) {
        try {
          const res = await state.api.entityByAbsoluteId(absoluteId, state.currentGraphId);
          entity = res.data;
          if (entity) _entityMap[absoluteId] = entity;
        } catch (_) {}
      }
      if (!entity) {
        showToast(t('graph.loadFailedDetail'), 'error');
        return;
      }

      const { hopMap, entities, relations, inheritedRelationIds } = await _fetchMultiHop(
        absoluteId, entity.entity_id, 1
      );

      if (!entities.find(e => e.absolute_id === absoluteId)) {
        entities.unshift(entity);
      }

      _buildGraph(entities, relations, absoluteId, hopMap, inheritedRelationIds);

      _focusAbsoluteId = absoluteId;
      const exitBtn = document.getElementById('comm-exit-focus-btn');
      if (exitBtn) exitBtn.style.display = '';
      const focusBadge = document.getElementById('comm-focus-badge');
      if (focusBadge) focusBadge.style.display = '';
    } catch (err) {
      showToast(t('graph.loadFailed') + ': ' + err.message, 'error');
    } finally {
      if (loadingEl) loadingEl.style.display = 'none';
    }
  }

  // ---- Exit focus mode ----

  function _exitFocus() {
    _focusAbsoluteId = null;
    _currentVersions = [];
    _currentVersionIdx = 0;
    const exitBtn = document.getElementById('comm-exit-focus-btn');
    if (exitBtn) exitBtn.style.display = 'none';
    const focusBadge = document.getElementById('comm-focus-badge');
    if (focusBadge) focusBadge.style.display = 'none';

    // Rebuild from community data
    const graphNodes = _commEntities.map(m => ({
      absolute_id: m.uuid,
      entity_id: m.entity_id,
      name: m.name,
      content: m.content || '',
    }));
    _buildGraph(graphNodes, _commRelations, null, null, null);

    const detailContent = document.getElementById('comm-detail-content');
    if (detailContent) {
      detailContent.innerHTML = emptyState(t('common.clickToView'), 'mouse-pointer-click');
      if (window.lucide) lucide.createIcons();
    }
  }

  // ---- Show entity detail in sidebar (same as graph.js) ----

  async function _showEntityDetail(absoluteId) {
    let entity = _entityMap[absoluteId];
    if (!entity) {
      try {
        const res = await state.api.entityByAbsoluteId(absoluteId, state.currentGraphId);
        if (res.data) { entity = res.data; _entityMap[absoluteId] = entity; }
      } catch (_) {}
    }
    if (!entity) return;

    const detailContent = document.getElementById('comm-detail-content');
    if (!detailContent) return;

    const entityId = entity.entity_id;

    let versions = [];
    try {
      const vRes = await state.api.entityVersions(entityId, state.currentGraphId);
      versions = vRes.data || [];
    } catch (_) {}

    _currentVersions = versions;
    _currentVersionIdx = versions.findIndex(v => v.absolute_id === absoluteId);
    if (_currentVersionIdx < 0) _currentVersionIdx = 0;

    const totalVersions = versions.length;

    detailContent.innerHTML = `
      <div class="flex items-center justify-between mb-3">
        <span class="badge badge-primary">${t('graph.entityDetail')}</span>
        ${totalVersions > 1 ? `
        <div class="flex items-center gap-1">
          <button class="btn btn-secondary btn-sm" id="comm-prev-ver-btn" ${_currentVersionIdx === 0 ? 'disabled' : ''} title="${t('graph.prevVersion')}">
            <i data-lucide="chevron-left" style="width:14px;height:14px;"></i>
          </button>
          <span class="mono text-xs" style="color:var(--text-muted);min-width:50px;text-align:center;">
            ${_currentVersionIdx + 1}/${totalVersions}
          </span>
          <button class="btn btn-secondary btn-sm" id="comm-next-ver-btn" ${_currentVersionIdx === totalVersions - 1 ? 'disabled' : ''} title="${t('graph.nextVersion')}">
            <i data-lucide="chevron-right" style="width:14px;height:14px;"></i>
          </button>
        </div>
        ` : ''}
      </div>

      <h3 style="font-size:1.1rem;font-weight:600;color:var(--text-primary);margin-bottom:0.75rem;word-break:break-word;">
        ${escapeHtml(entity.name || t('graph.unnamedEntity'))}
        ${totalVersions > 1 ? `<span style="color:var(--text-muted);font-size:0.85rem;font-weight:400;"> [${_currentVersionIdx + 1}/${totalVersions}]</span>` : ''}
      </h3>

      <div class="flex flex-wrap gap-2 mb-3">
        <button class="btn btn-secondary btn-sm" id="comm-view-versions-btn">
          <i data-lucide="git-branch" style="width:14px;height:14px;"></i>
          ${t('graph.versionHistory')}
        </button>
        <button class="btn btn-secondary btn-sm" id="comm-view-relations-btn">
          <i data-lucide="link" style="width:14px;height:14px;"></i>
          ${t('graph.viewRelations')}
        </button>
        <button class="btn btn-primary btn-sm" id="comm-focus-entity-btn">
          <i data-lucide="crosshair" style="width:14px;height:14px;"></i>
          ${t('graph.focusMode')}
        </button>
      </div>

      ${_focusAbsoluteId ? `
      <div style="margin-bottom:0.75rem;">
        <label style="display:flex;align-items:center;gap:0.35rem;font-size:0.8rem;cursor:pointer;color:var(--text-secondary);">
          <input type="checkbox" id="comm-only-current-cb" ${_onlyCurrentVersion ? 'checked' : ''}>
          ${t('graph.onlyCurrentVersion')}
        </label>
      </div>
      ` : ''}

      <div class="divider"></div>

      <div style="display:flex;flex-direction:column;gap:0.75rem;">
        <div>
          <span class="form-label" style="margin-bottom:0.125rem;">${t('graph.content')}</span>
          <p style="font-size:0.8125rem;color:var(--text-secondary);line-height:1.5;word-break:break-word;white-space:pre-wrap;">
            ${escapeHtml(entity.content || '-')}
          </p>
        </div>

        <div>
          <span class="form-label" style="margin-bottom:0.125rem;">${t('graph.entityId')}</span>
          <p class="mono truncate" style="color:var(--text-muted);font-size:0.75rem;" title="${escapeHtml(entity.entity_id || '')}">
            ${escapeHtml(entity.entity_id || '-')}
          </p>
        </div>

        <div>
          <span class="form-label" style="margin-bottom:0.125rem;">${t('graph.absoluteId')}</span>
          <p class="mono truncate" style="color:var(--text-muted);font-size:0.75rem;" title="${escapeHtml(entity.absolute_id || '')}">
            ${escapeHtml(entity.absolute_id || '-')}
          </p>
        </div>

        <div>
          <span class="form-label" style="margin-bottom:0.125rem;">${t('graph.eventTime')}</span>
          <p style="font-size:0.8125rem;color:var(--text-secondary);">
            ${formatDate(entity.event_time)}
          </p>
        </div>

        <div>
          <span class="form-label" style="margin-bottom:0.125rem;">${t('graph.processedTime')}</span>
          <p style="font-size:0.8125rem;color:var(--text-secondary);">
            ${formatDate(entity.processed_time)}
          </p>
        </div>

        ${entity.source_document ? `
        <div>
          <span class="form-label" style="margin-bottom:0.125rem;">${t('graph.sourceDoc')}</span>
          <span class="doc-link mono truncate" style="font-size:0.75rem;"
                data-view-doc="${escapeHtml(entity.source_document)}"
                title="${escapeHtml(entity.source_document)}">
            ${escapeHtml(truncate(entity.source_document, 60))}
          </span>
        </div>
        ` : ''}

        ${entity.memory_cache_id ? `
        <div>
          <span class="form-label" style="margin-bottom:0.125rem;">${t('graph.memoryCacheId')}</span>
          <span class="doc-link mono truncate" style="font-size:0.75rem;"
                data-view-doc="${escapeHtml(entity.memory_cache_id)}"
                title="${t('common.clickToView')}">
            ${escapeHtml(entity.memory_cache_id)}
          </span>
        </div>
        ` : ''}
      </div>
    `;

    if (window.lucide) lucide.createIcons({ nodes: [detailContent] });

    detailContent.querySelectorAll('[data-view-doc]').forEach(el => {
      el.addEventListener('click', () => window.showDocContent(el.getAttribute('data-view-doc')));
    });

    document.getElementById('comm-view-versions-btn').addEventListener('click', () => {
      _openVersionsModal(entity);
    });
    document.getElementById('comm-view-relations-btn').addEventListener('click', () => {
      _openRelationsModal(entity);
    });
    document.getElementById('comm-focus-entity-btn').addEventListener('click', () => {
      _focusOnEntity(absoluteId);
    });

    const onlyCb = document.getElementById('comm-only-current-cb');
    if (onlyCb) {
      onlyCb.addEventListener('change', () => {
        _onlyCurrentVersion = onlyCb.checked;
        _focusOnEntity(absoluteId);
      });
    }

    const prevBtn = document.getElementById('comm-prev-ver-btn');
    const nextBtn = document.getElementById('comm-next-ver-btn');
    if (prevBtn) {
      prevBtn.addEventListener('click', () => {
        if (_currentVersionIdx > 0) _switchVersion(_currentVersionIdx - 1);
      });
    }
    if (nextBtn) {
      nextBtn.addEventListener('click', () => {
        if (_currentVersionIdx < _currentVersions.length - 1) _switchVersion(_currentVersionIdx + 1);
      });
    }
  }

  // ---- Switch version ----

  async function _switchVersion(newIdx) {
    if (!_currentVersions[newIdx]) return;
    _currentVersionIdx = newIdx;
    const version = _currentVersions[newIdx];
    const absoluteId = version.absolute_id;
    if (!_entityMap[absoluteId]) {
      _entityMap[absoluteId] = version;
    }
    // Rebuild graph first, then render sidebar — prevents graph rebuild
    // from interfering with sidebar state (e.g. entityMap reset in buildGraph)
    await _focusOnEntity(absoluteId);
    await _showEntityDetail(absoluteId);
  }

  // ---- Show relation detail in sidebar ----

  function _showRelationDetail(absoluteId) {
    const relation = _relationMap[absoluteId];
    if (!relation) return;

    const detailContent = document.getElementById('comm-detail-content');
    if (!detailContent) return;

    const fromName = _entityMap[relation.entity1_absolute_id]?.name || relation.entity1_absolute_id || '?';
    const toName = _entityMap[relation.entity2_absolute_id]?.name || relation.entity2_absolute_id || '?';

    detailContent.innerHTML = `
      <div class="flex items-center gap-2 mb-3">
        <span class="badge" style="background:var(--info-dim);color:var(--info);">${t('graph.relationDetail')}</span>
      </div>

      <h3 style="font-size:1.1rem;font-weight:600;color:var(--text-primary);margin-bottom:0.75rem;word-break:break-word;">
        ${escapeHtml(truncate(relation.content || t('graph.unnamedRelation'), 60))}
      </h3>

      <div class="divider"></div>

      <div style="display:flex;flex-direction:column;gap:0.75rem;">
        <div>
          <span class="form-label" style="margin-bottom:0.125rem;">${t('graph.content')}</span>
          <p style="font-size:0.8125rem;color:var(--text-secondary);line-height:1.5;word-break:break-word;white-space:pre-wrap;">
            ${escapeHtml(relation.content || '-')}
          </p>
        </div>

        <div>
          <span class="form-label" style="margin-bottom:0.125rem;">${t('graph.fromEntity')}</span>
          <div class="flex items-center gap-2">
            <span class="mono truncate" style="color:var(--info);font-size:0.75rem;cursor:pointer;text-decoration:underline;"
                  data-view-entity="${escapeHtml(relation.entity1_absolute_id)}">${escapeHtml(truncate(fromName, 40))}</span>
            <button class="btn btn-secondary btn-sm" style="padding:0.125rem 0.375rem;" data-focus-entity="${escapeHtml(relation.entity1_absolute_id)}" title="${t('graph.focusMode')}">
              <i data-lucide="crosshair" style="width:12px;height:12px;"></i>
            </button>
          </div>
        </div>

        <div>
          <span class="form-label" style="margin-bottom:0.125rem;">${t('graph.toEntity')}</span>
          <div class="flex items-center gap-2">
            <span class="mono truncate" style="color:var(--info);font-size:0.75rem;cursor:pointer;text-decoration:underline;"
                  data-view-entity="${escapeHtml(relation.entity2_absolute_id)}">${escapeHtml(truncate(toName, 40))}</span>
            <button class="btn btn-secondary btn-sm" style="padding:0.125rem 0.375rem;" data-focus-entity="${escapeHtml(relation.entity2_absolute_id)}" title="${t('graph.focusMode')}">
              <i data-lucide="crosshair" style="width:12px;height:12px;"></i>
            </button>
          </div>
        </div>
      </div>
    `;

    detailContent.querySelectorAll('[data-view-entity]').forEach(el => {
      el.addEventListener('click', () => _showEntityDetail(el.getAttribute('data-view-entity')));
    });
    detailContent.querySelectorAll('[data-focus-entity]').forEach(el => {
      el.addEventListener('click', () => _focusOnEntity(el.getAttribute('data-focus-entity')));
    });

    if (window.lucide) lucide.createIcons();
  }

  // ---- Versions modal ----

  async function _openVersionsModal(entity) {
    const entityId = entity.entity_id || entity.absolute_id;
    const graphId = state.currentGraphId;

    const modal = showModal({
      title: t('graph.versionsTitle', { name: truncate(entity.name || entityId, 40) }),
      content: `<div class="flex justify-center p-6">${spinnerHtml()}</div>`,
      size: 'lg',
    });

    try {
      const res = await state.api.entityVersions(entityId, graphId);
      const versions = res.data || [];

      if (versions.length === 0) {
        modal.overlay.querySelector('.modal-body').innerHTML = emptyState(t('graph.noVersions'));
        return;
      }

      const rows = versions.map(v => `
        <tr>
          <td style="max-width:120px;">${formatDate(v.processed_time)}</td>
          <td style="max-width:200px;" title="${escapeHtml(v.name || '')}">${escapeHtml(truncate(v.name || '-', 30))}</td>
          <td style="max-width:300px;" title="${escapeHtml(v.content || '')}">${escapeHtml(truncate(v.content || '-', 50))}</td>
        </tr>
      `).join('');

      modal.overlay.querySelector('.modal-body').innerHTML = `
        <div class="table-container" style="max-height:50vh;overflow-y:auto;">
          <table class="data-table">
            <thead>
              <tr>
                <th>${t('graph.versionTime')}</th>
                <th>${t('graph.versionName')}</th>
                <th>${t('graph.versionContent')}</th>
              </tr>
            </thead>
            <tbody>${rows}</tbody>
          </table>
        </div>
        <p style="margin-top:0.75rem;font-size:0.75rem;color:var(--text-muted);">${t('graph.versionCount', { count: versions.length })}</p>
      `;
    } catch (err) {
      modal.overlay.querySelector('.modal-body').innerHTML = `
        <div class="empty-state">
          <i data-lucide="alert-triangle"></i>
          <p>${t('graph.loadFailedDetail')}: ${escapeHtml(err.message)}</p>
        </div>
      `;
      if (window.lucide) lucide.createIcons({ nodes: [modal.overlay] });
    }
  }

  // ---- Relations modal (scoped to community) ----

  async function _openRelationsModal(entity) {
    const entityId = entity.entity_id || entity.absolute_id;
    const graphId = state.currentGraphId;

    const modal = showModal({
      title: t('graph.relationsTitle', { name: truncate(entity.name || entityId, 40) }),
      content: `<div class="flex justify-center p-6">${spinnerHtml()}</div>`,
      size: 'lg',
    });

    try {
      const res = await state.api.entityRelations(entityId, graphId);
      let relations = res.data || [];

      // Filter to community scope
      const commUuids = new Set(_commEntities.map(e => e.uuid));
      relations = relations.filter(r =>
        commUuids.has(r.entity1_absolute_id) && commUuids.has(r.entity2_absolute_id)
      );

      if (relations.length === 0) {
        modal.overlay.querySelector('.modal-body').innerHTML = emptyState(t('graph.noRelations'));
        return;
      }

      const rows = relations.map(r => {
        const otherAbsId = r.entity1_absolute_id === entity.absolute_id
          ? r.entity2_absolute_id : r.entity1_absolute_id;
        const otherEntity = _entityMap[otherAbsId] || _commEntityMap[otherAbsId];
        const otherName = otherEntity ? (otherEntity.name || otherEntity.entity_id || '-') : '-';
        return `
        <tr>
          <td style="max-width:250px;" title="${escapeHtml(r.content || '')}">${escapeHtml(truncate(r.content || '-', 40))}</td>
          <td style="max-width:120px;" title="${escapeHtml(otherName)}">${escapeHtml(truncate(otherName, 20))}</td>
          <td class="mono" style="max-width:120px;font-size:0.75rem;color:var(--text-muted);">${formatDate(r.event_time)}</td>
        </tr>
      `;}).join('');

      modal.overlay.querySelector('.modal-body').innerHTML = `
        <div class="table-container" style="max-height:50vh;overflow-y:auto;">
          <table class="data-table">
            <thead>
              <tr>
                <th>${t('graph.content')}</th>
                <th>${t('graph.toEntity')}</th>
                <th>${t('graph.versionTime')}</th>
              </tr>
            </thead>
            <tbody>${rows}</tbody>
          </table>
        </div>
        <p style="margin-top:0.75rem;font-size:0.75rem;color:var(--text-muted);">${t('graph.relationCount', { count: relations.length })}</p>
      `;
    } catch (err) {
      modal.overlay.querySelector('.modal-body').innerHTML = `
        <div class="empty-state">
          <i data-lucide="alert-triangle"></i>
          <p>${t('graph.loadFailedDetail')}: ${escapeHtml(err.message)}</p>
        </div>
      `;
      if (window.lucide) lucide.createIcons({ nodes: [modal.overlay] });
    }
  }

  // Expose
  window._commDetect = _detect;
  window._commClear = _clearCommunities;
  window._commLoad = _loadCommunities;

  async function render(container, params) {
    _container = container;

    // Route: #communities/<cid> → detail page
    if (params && params[0] !== undefined) {
      await _renderDetailPage(container, parseInt(params[0]));
      return;
    }

    // Default: community list
    container.innerHTML = `
      <div class="space-y-4">
        <!-- Controls -->
        <div class="card p-4">
          <div class="flex items-center gap-3 flex-wrap">
            <button id="comm-detect-btn" class="btn btn-primary btn-sm" onclick="window._commDetect()">
              <i data-lucide="scan" style="width:16px;height:16px;"></i>${t('communities.detect')}
            </button>
            <div class="flex items-center gap-2 text-sm">
              <label style="color:var(--text-muted);">${t('communities.algorithm')}:</label>
              <select id="comm-algorithm" class="input text-xs py-1 px-2" style="width:auto;">
                <option value="louvain">Louvain</option>
              </select>
            </div>
            <div class="flex items-center gap-2 text-sm">
              <label style="color:var(--text-muted);">Resolution:</label>
              <input type="number" id="comm-resolution" class="input text-xs py-1 px-2 w-20" value="1.0" min="0.1" max="10" step="0.1">
            </div>
            <div class="flex items-center gap-2 text-sm">
              <label style="color:var(--text-muted);">${t('communities.minSize')}:</label>
              <input type="number" id="comm-min-size" class="input text-xs py-1 px-2 w-16" value="3" min="1" max="100">
            </div>
            <div class="flex-1"></div>
            <button class="btn btn-ghost btn-sm" onclick="window._commClear()" style="color:var(--text-error);">
              <i data-lucide="x-circle" style="width:14px;height:14px;"></i>${t('communities.clear')}
            </button>
          </div>
        </div>

        <!-- Stats -->
        <div id="comm-stats"></div>

        <!-- Community cards -->
        <div id="comm-cards"></div>
      </div>
    `;

    if (window.lucide) lucide.createIcons();

    // Try to load existing communities
    try {
      await _loadCommunities();
    } catch { /* ignore if no communities yet */ }
  }

  function destroy() {
    _container = null;
    _communities = [];
    _detectResult = null;
    if (_detailNetwork) {
      _detailNetwork.destroy();
      _detailNetwork = null;
    }
    _entityMap = {};
    _relationMap = {};
    _versionCounts = {};
    _pinnedNodePositions = {};
    _commEntities = [];
    _commRelations = [];
    _commEntityMap = {};
    _focusAbsoluteId = null;
    _currentVersions = [];
    _currentVersionIdx = 0;
    _onlyCurrentVersion = false;
    delete window._commDetect;
    delete window._commClear;
    delete window._commLoad;
  }

  return { render, destroy };
})());

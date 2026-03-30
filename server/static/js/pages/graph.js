/* ==========================================
   Graph Explorer Page
   ========================================== */

(function () {
  let network = null;
  let entityMap = {};   // absolute_id -> entity data
  let relationMap = {}; // absolute_id -> relation data
  let versionCounts = {}; // entity_id -> version count
  let isFirstRender = true;
  let pinnedNodePositions = {}; // absolute_id -> { x, y }

  // Focus state
  let focusAbsoluteId = null;
  let cachedAllNodes = [];       // default view: all nodes (seed + related)
  let cachedAllEdges = [];       // default view: all edges
  let cachedAllEntities = {};    // absolute_id -> entity data (from full entity list)

  // Version switcher state
  let currentVersions = [];      // versions list for the selected entity
  let currentVersionIdx = 0;     // index into currentVersions

  // Advanced options panel state
  let advancedOptionsOpen = false;

  // Hop & version accumulation state
  let relationScope = 'accumulated';
  let currentHopLevel = 1;
  let cachedInheritedRelationIds = null; // Set of relation absolute_ids inherited in main view
  let cachedAllRawRelations = null; // original raw relations from API (before remapping)
  let cachedRemappedMainRelations = null; // remapped relations for main view

  // Community coloring state
  let communityColoringEnabled = false;
  let communityMap = null; // absolute_id -> community_id

  // Relation strength mode
  let relationStrengthEnabled = false;

  // Time travel snapshot state
  let snapshotMode = false;
  let snapshotTime = null;

  // Color palettes and shared graph builders are in GraphUtils (graph-utils.js)

  // ---- Build the page layout and kick off initial load ----

  async function render(container, params) {
    container.innerHTML = `
      <div class="page-enter">
        <!-- Top control card -->
        <div class="card mb-4">
          <div class="card-header">
            <div style="display:flex;align-items:center;gap:0.5rem;">
              <i data-lucide="git-branch" style="width:16px;height:16px;color:var(--primary);"></i>
              <span class="badge badge-primary mono" id="graph-id-badge">-</span>
              <span id="focus-mode-badge" class="badge badge-warning" style="display:none;">
                <i data-lucide="crosshair" style="width:12px;height:12px;margin-right:2px;"></i>
                ${t('graph.focusMode')}
              </span>
            </div>
            <span id="graph-stats" class="mono" style="font-size:0.8125rem;color:var(--text-muted);"></span>
          </div>

          <div style="display:flex;align-items:center;justify-content:space-between;gap:0.75rem;flex-wrap:wrap;">
            <div style="display:flex;align-items:center;gap:0.5rem;">
              <button class="btn btn-primary" id="load-graph-btn">
                <i data-lucide="refresh-cw" style="width:16px;height:16px;"></i>
                ${t('graph.loadGraph')}
              </button>
              <button class="btn btn-secondary" id="exit-focus-btn" style="display:none;">
                <i data-lucide="maximize-2" style="width:16px;height:16px;"></i>
                ${t('graph.exitFocus')}
              </button>
            </div>
            <button class="btn btn-ghost btn-sm" id="toggle-graph-options-btn" style="color:var(--text-muted);">
              <i data-lucide="sliders-horizontal" style="width:14px;height:14px;margin-right:4px;"></i>
              ${t('graph.displayOptions')}
              <i data-lucide="chevron-down" style="width:14px;height:14px;margin-left:4px;transition:transform 0.2s;" id="graph-options-chevron"></i>
            </button>
          </div>

          <div id="graph-advanced-options" style="display:none;margin-top:12px;padding:14px 16px;background:var(--bg-input);border-radius:8px;border:1px solid var(--border-color);">
            <div style="display:flex;gap:1rem;align-items:flex-end;flex-wrap:wrap;">
              <div>
                <label class="form-label">${t('graph.maxEntities')}</label>
                <input type="number" class="input" id="entity-limit" min="10" max="500" value="50" step="10" style="width:100px;">
              </div>
              <div>
                <label class="form-label">${t('graph.hopLevel')}</label>
                <input type="number" class="input" id="hop-level" min="1" max="3" value="${currentHopLevel}" step="1" style="width:80px;">
              </div>
              <button class="btn btn-secondary" id="apply-hop-btn" title="${t('graph.apply')}">
                <i data-lucide="check" style="width:16px;height:16px;"></i>
                ${t('graph.apply')}
              </button>
            </div>
            ${isNeo4j() ? `
            <div style="margin-top:10px;display:flex;align-items:center;gap:0.5rem;">
              <input type="checkbox" id="community-coloring" style="cursor:pointer;">
              <label for="community-coloring" style="font-size:0.8125rem;cursor:pointer;color:var(--text-secondary);">${t('graph.communityColoring')}</label>
            </div>` : ''}
            <div style="margin-top:10px;display:flex;align-items:center;gap:0.5rem;">
              <input type="checkbox" id="relation-strength-mode" style="cursor:pointer;">
              <label for="relation-strength-mode" style="font-size:0.8125rem;cursor:pointer;color:var(--text-secondary);">${t('graph.relationStrength')}</label>
            </div>
            <div style="margin-top:10px;">
              <label class="form-label" style="font-size:0.8125rem;color:var(--text-secondary);">${t('timeTravel.snapshot')}</label>
              <div style="display:flex;gap:4px;align-items:center;">
                <input type="datetime-local" id="graph-snapshot-time" class="input" style="width:200px;font-size:0.8125rem;">
                <button class="btn btn-sm btn-primary" id="load-snapshot-btn" title="${t('timeTravel.title')}">
                  <i data-lucide="clock" style="width:14px;height:14px;"></i>
                  ${t('timeTravel.title')}
                </button>
                ${snapshotMode ? `
                <button class="btn btn-sm btn-secondary" id="clear-snapshot-btn" title="${t('graph.exitFocus')}">
                  <i data-lucide="rotate-ccw" style="width:14px;height:14px;"></i>
                  ${t('graph.exitFocus')}
                </button>
                ` : ''}
              </div>
            </div>
          </div>
        </div>

        <!-- Main area: canvas + detail sidebar -->
        <div class="flex gap-4" style="height:calc(100vh - 240px);min-height:400px;">
          <!-- Graph canvas -->
          <div class="flex-1 relative" style="min-width:0;">
            <div id="graph-canvas" style="width:100%;height:100%;"></div>
            <div id="graph-loading" class="absolute inset-0 flex items-center justify-center" style="background:var(--bg-input);border-radius:0.5rem;display:none;">
              ${spinnerHtml()}
            </div>
          </div>

          <!-- Detail sidebar -->
          <div id="detail-sidebar" style="width:30%;min-width:280px;max-width:420px;">
            <div class="card h-full flex flex-col">
              <div class="card-header">
                <span class="card-title">${t('common.detail')}</span>
              </div>
              <div id="detail-content" style="overflow-y:auto;flex:1;">
                ${emptyState(t('common.clickToView'), 'mouse-pointer-click')}
              </div>
            </div>
          </div>
        </div>
      </div>
    `;

    if (window.lucide) lucide.createIcons();

    const loadBtn = document.getElementById('load-graph-btn');
    const exitFocusBtn = document.getElementById('exit-focus-btn');
    const graphBadge = document.getElementById('graph-id-badge');

    graphBadge.textContent = state.currentGraphId;

    loadBtn.addEventListener('click', () => loadGraph());
    exitFocusBtn.addEventListener('click', () => exitFocus());

    const applyBtn = document.getElementById('apply-hop-btn');
    if (applyBtn) {
      applyBtn.addEventListener('click', () => {
        const newHop = parseInt(document.getElementById('hop-level').value, 10) || 1;
        currentHopLevel = newHop;
        if (focusAbsoluteId) {
          focusOnEntity(focusAbsoluteId);
        } else {
          // Main view: re-expand with new hop level using cached data
          applyHopToMainView();
        }
      });
    }

    // Advanced options toggle
    const toggleBtn = document.getElementById('toggle-graph-options-btn');
    const optionsPanel = document.getElementById('graph-advanced-options');
    const chevron = document.getElementById('graph-options-chevron');
    if (toggleBtn && optionsPanel) {
      toggleBtn.addEventListener('click', () => {
        const isOpen = optionsPanel.style.display !== 'none';
        optionsPanel.style.display = isOpen ? 'none' : 'block';
        if (chevron) chevron.style.transform = isOpen ? 'rotate(0deg)' : 'rotate(180deg)';
        advancedOptionsOpen = !isOpen;
      });
      if (advancedOptionsOpen) {
        optionsPanel.style.display = 'block';
        if (chevron) chevron.style.transform = 'rotate(180deg)';
      }
    }

    // Community coloring toggle
    const commCheckbox = document.getElementById('community-coloring');
    if (commCheckbox) {
      commCheckbox.addEventListener('change', async () => {
        communityColoringEnabled = commCheckbox.checked;
        if (communityColoringEnabled && !communityMap) {
          await loadCommunityMap();
        }
        if (cachedAllEntities && Object.keys(cachedAllEntities).length > 0) {
          // Rebuild graph with community coloring
          if (focusAbsoluteId) {
            focusOnEntity(focusAbsoluteId);
          } else {
            rebuildMainView();
          }
        }
      });
    }

    // Relation strength toggle
    const strengthCheckbox = document.getElementById('relation-strength-mode');
    if (strengthCheckbox) {
      strengthCheckbox.checked = relationStrengthEnabled;
      strengthCheckbox.addEventListener('change', () => {
        relationStrengthEnabled = strengthCheckbox.checked;
        if (cachedAllEntities && Object.keys(cachedAllEntities).length > 0) {
          if (focusAbsoluteId) {
            focusOnEntity(focusAbsoluteId);
          } else {
            rebuildMainView();
          }
        }
      });
    }

    // Time travel snapshot button
    const snapshotBtn = document.getElementById('load-snapshot-btn');
    if (snapshotBtn) {
      snapshotBtn.addEventListener('click', () => loadGraphSnapshot());
    }

    // Clear snapshot button
    const clearSnapshotBtn = document.getElementById('clear-snapshot-btn');
    if (clearSnapshotBtn) {
      clearSnapshotBtn.addEventListener('click', () => clearSnapshot());
    }

    if (isFirstRender) {
      isFirstRender = false;
      await loadGraph();
    } else if (cachedAllNodes.length > 0) {
      buildGraph(cachedAllNodes, cachedAllEdges, null, null, cachedInheritedRelationIds);
      const exitBtn = document.getElementById('exit-focus-btn');
      if (exitBtn) exitBtn.style.display = focusAbsoluteId ? '' : 'none';
      const focusBadge = document.getElementById('focus-mode-badge');
      if (focusBadge) focusBadge.style.display = focusAbsoluteId ? '' : 'none';
    }
  }

  // ---- Expand N-hop neighbors from seed entities (client-side BFS) ----

  function expandNHops(seedAbsIds, allRelations, hopLevel) {
    const result = new Set(seedAbsIds);
    let frontier = new Set(seedAbsIds);
    for (let h = 1; h <= hopLevel; h++) {
      const nextFrontier = new Set();
      for (const r of allRelations) {
        if (frontier.has(r.entity1_absolute_id) && !result.has(r.entity2_absolute_id)) {
          nextFrontier.add(r.entity2_absolute_id);
        }
        if (frontier.has(r.entity2_absolute_id) && !result.has(r.entity1_absolute_id)) {
          nextFrontier.add(r.entity1_absolute_id);
        }
      }
      for (const id of nextFrontier) result.add(id);
      frontier = nextFrontier;
    }
    return result;
  }

  // ---- Resolve unknown relation endpoints and remap to current entity versions ----
  //   Relations from get_all_relations may reference OLD absolute_ids for entities.
  //   This resolves them to the latest version's absolute_id and detects inherited relations.

  async function resolveAndRemapRelations(entities, relations, graphId) {
    const currentAbsIds = new Set(entities.map(e => e.absolute_id));
    const currentEntityIds = {};
    for (const e of entities) {
      currentEntityIds[e.entity_id] = e.absolute_id;
    }

    // Collect unknown endpoint absolute_ids
    const unknownAbsIds = new Set();
    for (const r of relations) {
      if (!currentAbsIds.has(r.entity1_absolute_id)) unknownAbsIds.add(r.entity1_absolute_id);
      if (!currentAbsIds.has(r.entity2_absolute_id)) unknownAbsIds.add(r.entity2_absolute_id);
    }

    // Batch resolve unknown endpoints (max 30 to avoid overload)
    const resolved = {};
    const toResolve = [...unknownAbsIds].slice(0, 30);
    if (toResolve.length > 0) {
      const promises = toResolve.map(async (absId) => {
        try {
          const res = await state.api.entityByAbsoluteId(absId, graphId);
          if (res.data) resolved[absId] = res.data;
        } catch (_) {}
      });
      await Promise.all(promises);
    }

    // Remap relations: replace old absolute_ids with current versions
    const remapped = [];
    const inheritedRelationIds = new Set();

    for (const r of relations) {
      let e1AbsId = r.entity1_absolute_id;
      let e2AbsId = r.entity2_absolute_id;
      let remapped1 = false, remapped2 = false;

      if (!currentAbsIds.has(e1AbsId)) {
        const oldEntity = resolved[e1AbsId];
        if (oldEntity && currentEntityIds[oldEntity.entity_id]) {
          e1AbsId = currentEntityIds[oldEntity.entity_id];
          remapped1 = true;
        }
      }
      if (!currentAbsIds.has(e2AbsId)) {
        const oldEntity = resolved[e2AbsId];
        if (oldEntity && currentEntityIds[oldEntity.entity_id]) {
          e2AbsId = currentEntityIds[oldEntity.entity_id];
          remapped2 = true;
        }
      }

      if (remapped1 || remapped2) {
        inheritedRelationIds.add(r.absolute_id);
        remapped.push({ ...r, entity1_absolute_id: e1AbsId, entity2_absolute_id: e2AbsId });
      } else {
        remapped.push(r);
      }
    }

    return { relations: remapped, inheritedRelationIds };
  }

  // ---- Apply hop level to main view without full reload ----

  async function applyHopToMainView() {
    if (cachedAllEntities && Object.keys(cachedAllEntities).length > 0) {
      // We need the original full relations set; store it during loadGraph
      // If not available, fall back to loadGraph
      if (!cachedAllRawRelations) {
        await loadGraph();
        return;
      }
      rebuildMainView();
    } else {
      await loadGraph();
    }
  }

  function rebuildMainView() {
    const allKnownEntities = Object.values(cachedAllEntities);
    const entityLimit = parseInt(document.getElementById('entity-limit').value, 10) || 50;
    const hopLevel = currentHopLevel;

    const seedEntities = allKnownEntities.slice(0, entityLimit);
    const seedAbsIds = new Set(seedEntities.map(e => e.absolute_id));

    // Use remapped relations for BFS
    const allRels = cachedRemappedMainRelations || cachedAllRawRelations;
    const visibleAbsIds = expandNHops(seedAbsIds, allRels, hopLevel);

    const visibleRelations = allRels.filter(r =>
      visibleAbsIds.has(r.entity1_absolute_id) && visibleAbsIds.has(r.entity2_absolute_id)
    );

    const allVisible = [...visibleAbsIds]
      .map(aid => cachedAllEntities[aid])
      .filter(Boolean);

    cachedAllNodes = allVisible;
    cachedAllEdges = visibleRelations;

    buildGraph(allVisible, visibleRelations, null, null, cachedInheritedRelationIds);

    const statsEl = document.getElementById('graph-stats');
    if (statsEl) {
      statsEl.textContent = t('graph.loaded', { entities: allVisible.length, relations: visibleRelations.length });
    }
  }

  // ---- Fetch entities, their relations, and build the graph ----

  async function loadGraph() {
    const graphId = state.currentGraphId;
    const entityLimit = parseInt(document.getElementById('entity-limit').value, 10) || 50;
    const hopLevel = parseInt(document.getElementById('hop-level').value, 10) || 1;
    // Store hop level for focus mode
    currentHopLevel = hopLevel;
    const loadingEl = document.getElementById('graph-loading');
    const statsEl = document.getElementById('graph-stats');

    if (loadingEl) loadingEl.style.display = 'flex';
    if (statsEl) statsEl.textContent = t('common.loading');

    try {
      const [entityRes, relationRes] = await Promise.all([
        state.api.listEntities(graphId, 5000),
        state.api.listRelations(graphId, 2000),
      ]);

      const allKnownEntities = entityRes.data || [];
      const allRelations = relationRes.data || [];

      if (allKnownEntities.length === 0) {
        if (statsEl) statsEl.textContent = t('graph.noRelations');
        if (loadingEl) loadingEl.style.display = 'none';
        return;
      }

      // Cache raw relations for Apply button reuse
      cachedAllRawRelations = allRelations;

      const seedEntities = allKnownEntities.slice(0, entityLimit);

      const entityByAbs = {};
      for (const e of allKnownEntities) {
        entityByAbs[e.absolute_id] = e;
      }
      cachedAllEntities = entityByAbs;

      // Resolve and remap inherited relations
      const { relations: remappedRelations, inheritedRelationIds } = await resolveAndRemapRelations(
        allKnownEntities, allRelations, graphId
      );
      cachedInheritedRelationIds = inheritedRelationIds;
      cachedRemappedMainRelations = remappedRelations;

      const seedAbsIds = new Set(seedEntities.map(e => e.absolute_id));

      // Expand N-hop from seeds using client-side BFS (on remapped relations)
      const visibleAbsIds = expandNHops(seedAbsIds, remappedRelations, hopLevel);

      const visibleRelations = remappedRelations.filter(r =>
        visibleAbsIds.has(r.entity1_absolute_id) && visibleAbsIds.has(r.entity2_absolute_id)
      );

      const allVisible = [...visibleAbsIds]
        .map(aid => entityByAbs[aid])
        .filter(Boolean);

      const allEntityIds = [...new Set(allVisible.map(e => e.entity_id))];
      try {
        const vcRes = await state.api.entityVersionCounts(allEntityIds, graphId);
        versionCounts = vcRes.data || {};
      } catch (_) { versionCounts = {}; }

      cachedAllNodes = allVisible;
      cachedAllEdges = visibleRelations;

      buildGraph(allVisible, visibleRelations, null, null, cachedInheritedRelationIds);

      focusAbsoluteId = null;
      currentVersions = [];
      currentVersionIdx = 0;
      const exitBtn = document.getElementById('exit-focus-btn');
      if (exitBtn) exitBtn.style.display = 'none';

      if (statsEl) {
        statsEl.textContent = t('graph.loaded', { entities: allVisible.length, relations: visibleRelations.length });
      }
      showToast(t('graph.loaded', { entities: allVisible.length, relations: visibleRelations.length }), 'success');
    } catch (err) {
      console.error('Failed to load graph:', err);
      showToast(t('graph.loadFailed') + ': ' + err.message, 'error');
      if (statsEl) statsEl.textContent = t('common.error');
    } finally {
      if (loadingEl) loadingEl.style.display = 'none';
    }
  }

  // ---- Load community map from API ----
  async function loadCommunityMap() {
    if (!isNeo4j()) return;
    try {
      const res = await state.api.listCommunities(state.currentGraphId, 1, 200);
      const communities = res.data?.communities || [];
      communityMap = {};
      for (const c of communities) {
        for (const m of (c.members || [])) {
          communityMap[m.uuid] = c.community_id;
        }
      }
    } catch (err) {
      console.warn('Failed to load community map:', err);
      communityMap = null;
    }
  }

  function rebuildMainView() {
    if (cachedAllNodes.length > 0) {
      buildGraph(cachedAllNodes, cachedAllEdges, null, null, cachedInheritedRelationIds);
    }
  }

  // ---- Build vis-network DataSet and initialize the network ----
  //   hopMap: { absoluteId: hopLevel } — optional, for focus mode coloring
  //   inheritedRelationIds: Set of relation absolute_ids that are inherited from older versions
  //   futureRelationIds: Set of relation absolute_ids that are from future versions

  function buildGraph(entities, relations, highlightAbsId, hopMap, inheritedRelationIds, futureRelationIds) {
    entityMap = {};
    relationMap = {};

    // Build version label for focused entity
    const versionLabel = highlightAbsId && currentVersions.length > 1
      ? { idx: currentVersionIdx + 1, total: currentVersions.length }
      : null;

    // Use shared graph builder
    let colorMode = hopMap ? 'hop' : 'default';
    if (communityColoringEnabled && communityMap && !hopMap) {
      colorMode = 'community';
    }
    const { nodes, entityMap: eMap, nodeIds } = GraphUtils.buildNodes(entities, {
      colorMode: colorMode,
      versionCounts: versionCounts,
      hopMap: hopMap,
      highlightAbsId: highlightAbsId,
      versionLabel: versionLabel,
      unnamedLabel: t('graph.unnamedEntity'),
      communityMap: communityMap,
    });
    const visibleNodeIds = new Set();
    nodes.forEach((node) => {
      visibleNodeIds.add(node.id);
      const pinned = pinnedNodePositions[node.id];
      if (pinned) {
        nodes.update({
          id: node.id,
          x: pinned.x,
          y: pinned.y,
          fixed: { x: true, y: true },
        });
      }
    });
    for (const nodeId of Object.keys(pinnedNodePositions)) {
      if (!visibleNodeIds.has(nodeId)) delete pinnedNodePositions[nodeId];
    }
    entityMap = eMap;

    const { edges, relationMap: rMap } = GraphUtils.buildEdges(relations, nodeIds, {
      inheritedRelationIds: inheritedRelationIds,
      futureRelationIds: futureRelationIds,
      weightMode: relationStrengthEnabled ? 'count' : null,
    });
    relationMap = rMap;

    const container = document.getElementById('graph-canvas');
    if (!container) return;

    if (network) {
      network.destroy();
      network = null;
    }

    const options = {
      physics: GraphUtils.getPhysicsOptions(),
      interaction: GraphUtils.getInteractionOptions(),
      layout: { improvedLayout: true },
    };

    network = new vis.Network(container, { nodes, edges }, options);

    if (highlightAbsId) {
      network.once('stabilizationIterationsDone', () => {
        network.focus(highlightAbsId, { scale: 1.2, animation: { duration: 500, easingFunction: 'easeInOutQuad' } });
      });
    }

    network.on('click', (params) => {
      const nodeId = params.nodes[0];
      const edgeId = params.edges[0];

      if (nodeId) {
        showEntityDetail(nodeId);
      } else if (edgeId) {
        showRelationDetail(edgeId);
      }
    });

    network.on('dragEnd', (params) => {
      if (!params.nodes || params.nodes.length === 0) return;
      const positions = network.getPositions(params.nodes);
      params.nodes.forEach((nodeId) => {
        const pos = positions[nodeId];
        if (!pos) return;
        pinnedNodePositions[nodeId] = { x: pos.x, y: pos.y };
        nodes.update({
          id: nodeId,
          x: pos.x,
          y: pos.y,
          fixed: { x: true, y: true },
        });
      });
    });
  }

  // ---- Multi-hop BFS for focus mode ----
  //   Returns: { hopMap, entities, relations, inheritedRelationIds, futureRelationIds }
  //   Uses backend relation_scope for focused entity; neighbors use simple queries.
  //   - Among base set, those NOT in latest version → dashed amber (inherited, lost over time)
  //   - Relations that only exist in newer versions → NOT shown (future, cannot predict)

  async function fetchMultiHop(startAbsId, startEntityId, hopLevel) {
    const graphId = state.currentGraphId;
    const hopMap = { [startAbsId]: 0 };
    const relationSet = new Map();  // absolute_id -> raw relation (before remapping)
    const inheritedRelationIds = new Set();
    const futureRelationIds = new Set();
    let frontier = [{ absId: startAbsId, entityId: startEntityId }];

    const focusedAbsIds = new Set(currentVersions.map(v => v.absolute_id));
    focusedAbsIds.add(startAbsId);

    // Track absolute_id -> entity_id for all discovered nodes
    const absToEntityId = {};
    if (startEntityId) absToEntityId[startAbsId] = startEntityId;
    for (const v of currentVersions) absToEntityId[v.absolute_id] = startEntityId;

    for (let h = 1; h <= hopLevel; h++) {
      const nextFrontier = [];
      const MAX_PER_HOP = 20;

      for (const node of frontier) {
        let versionRels = [];
        const isFocusedNode = focusedAbsIds.has(node.absId);

        try {
          let res;
          if (node.entityId && isFocusedNode) {
            // Focused entity: use backend relation_scope classification
            res = await state.api.entityRelations(node.entityId, graphId, {
              maxVersionAbsoluteId: node.absId,
              relationScope: relationScope
            });
          } else if (node.entityId) {
            // Neighbor entity: simple query without version scope
            res = await state.api.entityRelations(node.entityId, graphId, {
              maxVersionAbsoluteId: node.absId
            });
          } else {
            res = await state.api.entityOneHop(node.absId, graphId);
          }
          versionRels = res.data || [];

          // For focused nodes with relation_scope, extract classification from response
          if (isFocusedNode && node.entityId && relationScope !== 'version_only') {
            for (const r of versionRels) {
              if (relationScope === 'all_versions') {
                if (r._version_scope === 'inherited') inheritedRelationIds.add(r.absolute_id);
                if (r._version_scope === 'future') futureRelationIds.add(r.absolute_id);
              } else {
                // accumulated mode
                if (r._inherited) inheritedRelationIds.add(r.absolute_id);
              }
            }
          }
        } catch (_) {}

        // Add version relations as base set
        for (const r of versionRels) {
          relationSet.set(r.absolute_id, r);
          const otherAbsId = r.entity1_absolute_id === node.absId
            ? r.entity2_absolute_id : r.entity1_absolute_id;
          if (otherAbsId && !(otherAbsId in hopMap)) {
            hopMap[otherAbsId] = h;
            nextFrontier.push({ absId: otherAbsId, entityId: null });
          }
        }

        // Resolve entity_id for new frontier nodes from cache
        for (const item of nextFrontier) {
          if (item.entityId) continue;
          const ent = entityMap[item.absId] || cachedAllEntities[item.absId];
          if (ent) {
            item.entityId = ent.entity_id;
            absToEntityId[item.absId] = ent.entity_id;
          }
        }
      }

      frontier = nextFrontier.slice(0, MAX_PER_HOP * h);
    }

    // ---- Post-BFS: resolve entity data, deduplicate, remap relations ----

    // 1. Resolve entity data for all hopMap entries
    const rawEntities = [];
    const resolvedIds = new Set();
    for (const absId of Object.keys(hopMap)) {
      if (resolvedIds.has(absId)) continue;
      const ent = entityMap[absId] || cachedAllEntities[absId];
      if (ent) {
        rawEntities.push(ent);
        resolvedIds.add(absId);
        absToEntityId[absId] = ent.entity_id;
      }
    }

    // 2. Build entity_id -> latest absId from GLOBAL cache (cachedAllEntities has all latest versions)
    const entityIdToLatest = {};
    for (const [absId, ent] of Object.entries(cachedAllEntities)) {
      entityIdToLatest[ent.entity_id] = absId;
    }
    // Override: focused entity always maps to the version being viewed
    if (startEntityId) entityIdToLatest[startEntityId] = startAbsId;

    // 3. Deduplicate: for each entity_id, keep only the latest version (avoids old-version ghost nodes)
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
      // Old versions (absolute_id !== latestAbsId) are excluded from display
    }

    // 4. Collect unknown endpoints from relations (old absolute_ids not in resolvedIds)
    const unknownEndpoints = new Set();
    for (const r of relationSet.values()) {
      if (!resolvedIds.has(r.entity1_absolute_id)) unknownEndpoints.add(r.entity1_absolute_id);
      if (!resolvedIds.has(r.entity2_absolute_id)) unknownEndpoints.add(r.entity2_absolute_id);
    }

    // Batch resolve unknown endpoints via API (max 30)
    if (unknownEndpoints.size > 0) {
      const promises = [...unknownEndpoints].slice(0, 30).map(async (absId) => {
        try {
          const res = await state.api.entityByAbsoluteId(absId, graphId);
          if (res.data) {
            entityMap[absId] = res.data;
            absToEntityId[absId] = res.data.entity_id;
          }
        } catch (_) {}
      });
      await Promise.all(promises);
    }

    // 5. Remap all relation endpoints to latest visible versions
    const relations = [];
    for (const r of relationSet.values()) {
      let e1 = r.entity1_absolute_id;
      let e2 = r.entity2_absolute_id;
      let skip = false;

      // Resolve e1 — even if in resolvedIds, remap old versions to latest
      {
        const eid = absToEntityId[e1];
        if (eid && entityIdToLatest[eid]) {
          e1 = entityIdToLatest[eid];
        } else {
          skip = true;
        }
      }

      // Resolve e2
      {
        const eid = absToEntityId[e2];
        if (eid && entityIdToLatest[eid]) {
          e2 = entityIdToLatest[eid];
        } else {
          skip = true;
        }
      }

      if (skip) continue;

      // Add remapped entities to visible set if not already present
      for (const absId of [e1, e2]) {
        const ent = cachedAllEntities[absId] || entityMap[absId];
        if (ent && !seenEntityIds.has(ent.entity_id)) {
          dedupedEntities.push(ent);
          seenEntityIds.add(ent.entity_id);
        }
      }

      // Fix hop levels: when old endpoint is remapped to latest, inherit the lower hop level
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

    // 6. Filter: only keep entities that are endpoints of at least one relation
    const connectedNodeIds = new Set();
    for (const r of relations) {
      connectedNodeIds.add(r.entity1_absolute_id);
      connectedNodeIds.add(r.entity2_absolute_id);
    }
    const finalEntities = dedupedEntities.filter(e => connectedNodeIds.has(e.absolute_id));

    return { hopMap, entities: finalEntities, relations, inheritedRelationIds, futureRelationIds };
  }

  // ---- Focus on a specific entity version (multi-hop view) ----

  async function focusOnEntity(absoluteId) {
    const graphId = state.currentGraphId;
    const loadingEl = document.getElementById('graph-loading');
    if (loadingEl) loadingEl.style.display = 'flex';

    try {
      let entity = entityMap[absoluteId];

      if (!entity) {
        try {
          const res = await state.api.entityByAbsoluteId(absoluteId, graphId);
          entity = res.data;
          if (entity) entityMap[absoluteId] = entity;
        } catch (_) {}
      }

      if (!entity) {
        showToast(t('graph.loadFailedDetail'), 'error');
        return;
      }

      // Use stored hop level (set on "Load Graph" click)
      const { hopMap, entities, relations, inheritedRelationIds, futureRelationIds } = await fetchMultiHop(
        absoluteId, entity.entity_id, currentHopLevel
      );

      if (!entities.find(e => e.absolute_id === absoluteId)) {
        entities.unshift(entity);
      }

      // Fetch version counts
      const allEntityIds = [...new Set(entities.map(e => e.entity_id))];
      try {
        const vcRes = await state.api.entityVersionCounts(allEntityIds, graphId);
        versionCounts = vcRes.data || {};
      } catch (_) {}

      buildGraph(entities, relations, absoluteId, hopMap, inheritedRelationIds, futureRelationIds);

      focusAbsoluteId = absoluteId;
      const exitBtn = document.getElementById('exit-focus-btn');
      if (exitBtn) exitBtn.style.display = '';
      const focusBadge = document.getElementById('focus-mode-badge');
      if (focusBadge) focusBadge.style.display = '';

    } catch (err) {
      console.error('Focus failed:', err);
      showToast(t('graph.loadFailed') + ': ' + err.message, 'error');
    } finally {
      if (loadingEl) loadingEl.style.display = 'none';
    }
  }

  // ---- Exit focus mode, restore default view ----

  function exitFocus() {
    focusAbsoluteId = null;
    currentVersions = [];
    currentVersionIdx = 0;
    const exitBtn = document.getElementById('exit-focus-btn');
    if (exitBtn) exitBtn.style.display = 'none';
    const focusBadge = document.getElementById('focus-mode-badge');
    if (focusBadge) focusBadge.style.display = 'none';

    buildGraph(cachedAllNodes, cachedAllEdges, null, null, cachedInheritedRelationIds);

    const detailContent = document.getElementById('detail-content');
    if (detailContent) {
      detailContent.innerHTML = emptyState(t('common.clickToView'), 'mouse-pointer-click');
    }
  }

  // ---- Show entity detail in the sidebar ----

  async function showEntityDetail(absoluteId) {
    let entity = entityMap[absoluteId];
    if (!entity) {
      try {
        const res = await state.api.entityByAbsoluteId(absoluteId, state.currentGraphId);
        if (res.data) { entity = res.data; entityMap[absoluteId] = entity; }
      } catch (_) {}
    }
    if (!entity) return;

    const detailContent = document.getElementById('detail-content');
    if (!detailContent) return;

    const entityId = entity.entity_id;

    let versions = [];
    try {
      const vRes = await state.api.entityVersions(entityId, state.currentGraphId);
      versions = vRes.data || [];
    } catch (_) {}

    currentVersions = versions;
    currentVersionIdx = versions.findIndex(v => v.absolute_id === absoluteId);
    if (currentVersionIdx < 0) currentVersionIdx = 0;

    const totalVersions = versions.length;

    detailContent.innerHTML = `
      <div class="flex items-center justify-between mb-3">
        <span class="badge badge-primary">${t('graph.entityDetail')}</span>
        ${totalVersions > 1 ? `
        <div class="flex items-center gap-1">
          <button class="btn btn-secondary btn-sm" id="prev-ver-btn" ${currentVersionIdx === 0 ? 'disabled' : ''} title="${t('graph.prevVersion')}">
            <i data-lucide="chevron-left" style="width:14px;height:14px;"></i>
          </button>
          <span class="mono text-xs" style="color:var(--text-muted);min-width:50px;text-align:center;">
            ${currentVersionIdx + 1}/${totalVersions}
          </span>
          <button class="btn btn-secondary btn-sm" id="next-ver-btn" ${currentVersionIdx === totalVersions - 1 ? 'disabled' : ''} title="${t('graph.nextVersion')}">
            <i data-lucide="chevron-right" style="width:14px;height:14px;"></i>
          </button>
        </div>
        ` : ''}
      </div>

      <h3 style="font-size:1.1rem;font-weight:600;color:var(--text-primary);margin-bottom:0.75rem;word-break:break-word;">
        ${escapeHtml(entity.name || t('graph.unnamedEntity'))}
        ${totalVersions > 1 ? `<span style="color:var(--text-muted);font-size:0.85rem;font-weight:400;"> [${currentVersionIdx + 1}/${totalVersions}]</span>` : ''}
      </h3>

      <div class="flex flex-wrap gap-2 mb-3">
        <button class="btn btn-secondary btn-sm" id="view-versions-btn">
          <i data-lucide="git-branch" style="width:14px;height:14px;"></i>
          ${t('graph.versionHistory')}
        </button>
        <button class="btn btn-secondary btn-sm" id="view-relations-btn">
          <i data-lucide="link" style="width:14px;height:14px;"></i>
          ${t('graph.viewRelations')}
        </button>
        <button class="btn btn-primary btn-sm" id="focus-entity-btn">
          <i data-lucide="crosshair" style="width:14px;height:14px;"></i>
          ${t('graph.focusMode')}
        </button>
      </div>

      ${focusAbsoluteId ? `
      <div style="margin-bottom:0.75rem;">
        <label style="display:flex;align-items:center;gap:0.35rem;font-size:0.8rem;color:var(--text-secondary);">
          ${t('graph.relationScope')}
          <select id="relation-scope-sel" style="font-size:0.8rem;padding:0.15rem 0.3rem;border-radius:0.25rem;background:var(--bg-secondary);color:var(--text-primary);border:1px solid var(--border-primary);">
            <option value="accumulated" ${relationScope === 'accumulated' ? 'selected' : ''}>${t('graph.scopeAccumulated')}</option>
            <option value="version_only" ${relationScope === 'version_only' ? 'selected' : ''}>${t('graph.scopeVersionOnly')}</option>
            <option value="all_versions" ${relationScope === 'all_versions' ? 'selected' : ''}>${t('graph.scopeAllVersions')}</option>
          </select>
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

    document.getElementById('view-versions-btn').addEventListener('click', () => {
      openVersionsModal(entity);
    });
    document.getElementById('view-relations-btn').addEventListener('click', () => {
      openRelationsModal(entity);
    });
    document.getElementById('focus-entity-btn').addEventListener('click', () => {
      focusOnEntity(absoluteId);
    });

    const prevBtn = document.getElementById('prev-ver-btn');
    const nextBtn = document.getElementById('next-ver-btn');

    if (prevBtn) {
      prevBtn.addEventListener('click', () => {
        if (currentVersionIdx > 0) switchVersion(currentVersionIdx - 1);
      });
    }
    if (nextBtn) {
      nextBtn.addEventListener('click', () => {
        if (currentVersionIdx < currentVersions.length - 1) switchVersion(currentVersionIdx + 1);
      });
    }

    const scopeSel = document.getElementById('relation-scope-sel');
    if (scopeSel) {
      scopeSel.addEventListener('change', () => {
        relationScope = scopeSel.value;
        focusOnEntity(absoluteId);
      });
    }
  }

  // ---- Switch to a different version of the current entity ----

  async function switchVersion(newIdx) {
    if (!currentVersions[newIdx]) return;
    currentVersionIdx = newIdx;

    const version = currentVersions[newIdx];
    const absoluteId = version.absolute_id;

    if (!entityMap[absoluteId]) {
      entityMap[absoluteId] = version;
    }

    // Rebuild graph first, then render sidebar — prevents graph rebuild
    // from interfering with sidebar state (e.g. entityMap reset in buildGraph)
    await focusOnEntity(absoluteId);
    await showEntityDetail(absoluteId);
  }

  // ---- Show relation detail in the sidebar ----

  function showRelationDetail(absoluteId) {
    const relation = relationMap[absoluteId];
    if (!relation) return;

    const detailContent = document.getElementById('detail-content');
    if (!detailContent) return;

    const fromName = entityMap[relation.entity1_absolute_id]?.name || relation.entity1_absolute_id || '?';
    const toName = entityMap[relation.entity2_absolute_id]?.name || relation.entity2_absolute_id || '?';

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

        <div>
          <span class="form-label" style="margin-bottom:0.125rem;">${t('graph.relationId')}</span>
          <p class="mono truncate" style="color:var(--text-muted);font-size:0.75rem;">
            ${escapeHtml(relation.relation_id || '-')}
          </p>
        </div>

        <div>
          <span class="form-label" style="margin-bottom:0.125rem;">${t('graph.eventTime')}</span>
          <p style="font-size:0.8125rem;color:var(--text-secondary);">
            ${formatDate(relation.event_time)}
          </p>
        </div>

        <div>
          <span class="form-label" style="margin-bottom:0.125rem;">${t('graph.processedTime')}</span>
          <p style="font-size:0.8125rem;color:var(--text-secondary);">
            ${formatDate(relation.processed_time)}
          </p>
        </div>

        ${relation.source_document ? `
        <div>
          <span class="form-label" style="margin-bottom:0.125rem;">${t('graph.sourceDoc')}</span>
          <span class="doc-link mono truncate" style="font-size:0.75rem;"
                data-view-doc="${escapeHtml(relation.source_document)}"
                title="${escapeHtml(relation.source_document)}">
            ${escapeHtml(truncate(relation.source_document, 60))}
          </span>
        </div>
        ` : ''}
      </div>
    `;

    // Event handlers for clickable entity endpoints
    detailContent.querySelectorAll('[data-view-entity]').forEach(el => {
      el.addEventListener('click', () => showEntityDetail(el.getAttribute('data-view-entity')));
    });
    detailContent.querySelectorAll('[data-focus-entity]').forEach(el => {
      el.addEventListener('click', () => focusOnEntity(el.getAttribute('data-focus-entity')));
    });
    detailContent.querySelectorAll('[data-view-doc]').forEach(el => {
      el.addEventListener('click', () => window.showDocContent(el.getAttribute('data-view-doc')));
    });

    if (window.lucide) lucide.createIcons({ nodes: [detailContent] });
  }

  // ---- Versions modal ----

  async function openVersionsModal(entity) {
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

      const rows = versions
        .map(
          (v) => `
        <tr>
          <td style="max-width:120px;">${formatDate(v.processed_time)}</td>
          <td style="max-width:200px;" title="${escapeHtml(v.name || '')}">${escapeHtml(truncate(v.name || '-', 30))}</td>
          <td style="max-width:300px;" title="${escapeHtml(v.content || '')}">${escapeHtml(truncate(v.content || '-', 50))}</td>
        </tr>
      `
        )
        .join('');

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

  // ---- Relations modal ----

  async function openRelationsModal(entity) {
    const entityId = entity.entity_id || entity.absolute_id;
    const graphId = state.currentGraphId;

    const modal = showModal({
      title: t('graph.relationsTitle', { name: truncate(entity.name || entityId, 40) }),
      content: `<div class="flex justify-center p-6">${spinnerHtml()}</div>`,
      size: 'lg',
    });

    try {
      const res = await state.api.entityRelations(entityId, graphId);
      const relations = res.data || [];

      if (relations.length === 0) {
        modal.overlay.querySelector('.modal-body').innerHTML = emptyState(t('graph.noRelations'));
        return;
      }

      const rows = relations
        .map(
          (r) => {
            const otherAbsId = r.entity1_absolute_id === entity.absolute_id
              ? r.entity2_absolute_id : r.entity1_absolute_id;
            const otherEntity = entityMap[otherAbsId] || cachedAllEntities[otherAbsId];
            const otherName = otherEntity ? (otherEntity.name || otherEntity.entity_id || '-') : '-';
            return `
        <tr>
          <td style="max-width:250px;" title="${escapeHtml(r.content || '')}">${escapeHtml(truncate(r.content || '-', 40))}</td>
          <td style="max-width:120px;" title="${escapeHtml(otherName)}">${escapeHtml(truncate(otherName, 20))}</td>
          <td class="mono" style="max-width:120px;font-size:0.75rem;color:var(--text-muted);">${formatDate(r.event_time)}</td>
        </tr>
      `;}
        )
        .join('');

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

  // ---- Time Travel: load graph snapshot ----

  async function loadGraphSnapshot() {
    const timeInput = document.getElementById('graph-snapshot-time');
    const time = timeInput ? timeInput.value : null;
    if (!time) {
      showToast(t('timeTravel.time') + ' ' + t('common.required', 'required'), 'warn');
      return;
    }

    const loadingEl = document.getElementById('graph-loading');
    const statsEl = document.getElementById('graph-stats');
    if (loadingEl) loadingEl.style.display = 'flex';
    if (statsEl) statsEl.textContent = t('common.loading');

    try {
      // datetime-local gives "YYYY-MM-DDTHH:MM", append ":00" for seconds
      const timeParam = time + ':00';
      const res = await state.api.getSnapshot(timeParam);

      if (res.error) {
        showToast(res.error, 'error');
        return;
      }

      renderSnapshotData(res);
      snapshotMode = true;
      snapshotTime = time;
      showToast(t('timeTravel.snapshot') + ': ' + time, 'success');
    } catch (err) {
      console.error('Snapshot load failed:', err);
      showToast(t('graph.loadFailed') + ': ' + err.message, 'error');
    } finally {
      if (loadingEl) loadingEl.style.display = 'none';
    }
  }

  function renderSnapshotData(data) {
    const snapshotEntities = data.entities || [];
    const snapshotRelations = data.relations || [];

    if (snapshotEntities.length === 0) {
      const statsEl = document.getElementById('graph-stats');
      if (statsEl) statsEl.textContent = t('graph.noRelations');
      showToast(t('common.noData'), 'warn');
      return;
    }

    // Build entity lookup from snapshot data
    const snapshotEntityMap = {};
    for (const e of snapshotEntities) {
      snapshotEntityMap[e.absolute_id] = e;
    }

    // Store snapshot data as cached for rebuild on option changes
    cachedAllNodes = snapshotEntities;
    cachedAllEdges = snapshotRelations;
    cachedInheritedRelationIds = null;

    // Build version counts
    const snapshotEntityIds = [...new Set(snapshotEntities.map(e => e.entity_id))];
    versionCounts = {};
    // Simple version count from the snapshot entities themselves
    for (const eid of snapshotEntityIds) {
      versionCounts[eid] = snapshotEntities.filter(e => e.entity_id === eid).length;
    }

    // Render using existing buildGraph
    buildGraph(snapshotEntities, snapshotRelations, null, null, null);

    // Update stats
    const statsEl = document.getElementById('graph-stats');
    if (statsEl) {
      statsEl.textContent = t('graph.loaded', {
        entities: snapshotEntities.length,
        relations: snapshotRelations.length,
      }) + ' [' + t('timeTravel.snapshot') + ']';
    }

    // Clear focus mode
    focusAbsoluteId = null;
    const exitBtn = document.getElementById('exit-focus-btn');
    if (exitBtn) exitBtn.style.display = 'none';
    const focusBadge = document.getElementById('focus-mode-badge');
    if (focusBadge) focusBadge.style.display = 'none';
  }

  function clearSnapshot() {
    snapshotMode = false;
    snapshotTime = null;
    const timeInput = document.getElementById('graph-snapshot-time');
    if (timeInput) timeInput.value = '';
    // Reload the graph normally
    loadGraph();
  }

  // ---- Cleanup on page leave ----

  function destroy() {
    if (network) {
      network.destroy();
      network = null;
    }
    entityMap = {};
    relationMap = {};
    focusAbsoluteId = null;
    currentVersions = [];
    currentVersionIdx = 0;
    cachedInheritedRelationIds = null;
    cachedAllRawRelations = null;
    cachedRemappedMainRelations = null;
    relationStrengthEnabled = false;
    snapshotMode = false;
    snapshotTime = null;
  }

  // ---- Register this page ----

  registerPage('graph', { render, destroy });
})();

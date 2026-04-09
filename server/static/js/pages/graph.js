/* ==========================================
   Graph Explorer Page
   ========================================== */

(function () {
  let explorer = null;
  let isFirstRender = true;

  // Focus state
  let focusAbsoluteId = null;
  let cachedAllNodes = [];       // default view: all nodes (seed + related)
  let cachedAllEdges = [];       // default view: all edges
  let cachedAllEntities = {};    // absolute_id -> entity data (from full entity list)

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

    // ---- Create GraphExplorer instance ----
    explorer = GraphExplorer.create({
      canvasId: 'graph-canvas',
      detailContentId: 'detail-content',
      loadingId: 'graph-loading',
      exitFocusBtnId: 'exit-focus-btn',
      focusBadgeId: 'focus-mode-badge',
      idPrefix: '',
      entityCache: cachedAllEntities,
      defaultHopLevel: currentHopLevel,
      maxPerHop: 20,
      communityColoringEnabled: communityColoringEnabled,
      communityMap: communityMap,
      relationStrengthEnabled: relationStrengthEnabled,
      familyIdToLatest: function () {
        var map = {};
        for (var absId in cachedAllEntities) {
          map[cachedAllEntities[absId].family_id] = absId;
        }
        return map;
      },
      onAfterFocus: function () {
        focusAbsoluteId = explorer.getState().focusAbsoluteId;
      },
      onRestoreDefaultView: function () {
        var hubLayout = computeHubLayout(cachedAllEdges);
        return {
          entities: cachedAllNodes,
          relations: cachedAllEdges,
          inheritedRelationIds: cachedInheritedRelationIds,
          hubLayout: hubLayout,
        };
      },
    });

    const loadBtn = document.getElementById('load-graph-btn');
    const graphBadge = document.getElementById('graph-id-badge');

    graphBadge.textContent = state.currentGraphId;

    loadBtn.addEventListener('click', () => loadGraph());
    document.getElementById('exit-focus-btn').addEventListener('click', () => {
      focusAbsoluteId = null;
      explorer.exitFocus();
    });

    const applyBtn = document.getElementById('apply-hop-btn');
    if (applyBtn) {
      applyBtn.addEventListener('click', () => {
        const newHop = parseInt(document.getElementById('hop-level').value, 10) || 1;
        currentHopLevel = newHop;
        explorer.setState('defaultHopLevel', currentHopLevel);
        if (focusAbsoluteId) {
          explorer.focusOnEntity(focusAbsoluteId).then(() => {
            focusAbsoluteId = explorer.getState().focusAbsoluteId;
          });
        } else {
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
        explorer.setState('communityColoringEnabled', communityColoringEnabled);
        if (communityColoringEnabled && !communityMap) {
          await loadCommunityMap();
          explorer.setState('communityMap', communityMap);
        }
        if (cachedAllEntities && Object.keys(cachedAllEntities).length > 0) {
          if (focusAbsoluteId) {
            explorer.focusOnEntity(focusAbsoluteId).then(() => {
              focusAbsoluteId = explorer.getState().focusAbsoluteId;
            });
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
        explorer.setState('relationStrengthEnabled', relationStrengthEnabled);
        if (cachedAllEntities && Object.keys(cachedAllEntities).length > 0) {
          if (focusAbsoluteId) {
            explorer.focusOnEntity(focusAbsoluteId).then(() => {
              focusAbsoluteId = explorer.getState().focusAbsoluteId;
            });
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
      const hubLayout = computeHubLayout(cachedAllEdges);
      explorer.buildGraph(cachedAllNodes, cachedAllEdges, null, null, cachedInheritedRelationIds, undefined, hubLayout);
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

  // ---- Compute top-3 hub entities and their 1-hop neighbors ----

  function computeHubLayout(visibleRelations) {
    if (!visibleRelations || visibleRelations.length === 0) return null;

    const relCounts = {};
    for (const r of visibleRelations) {
      relCounts[r.entity1_absolute_id] = (relCounts[r.entity1_absolute_id] || 0) + 1;
      relCounts[r.entity2_absolute_id] = (relCounts[r.entity2_absolute_id] || 0) + 1;
    }
    const sorted = Object.entries(relCounts).sort((a, b) => b[1] - a[1]);
    const hubIds = sorted.slice(0, 3).map(e => e[0]);
    if (hubIds.length === 0) return null;

    // hubMap: absoluteId -> hubIndex (0/1/2)
    const hubMap = {};
    for (let i = 0; i < hubIds.length; i++) {
      hubMap[hubIds[i]] = i;
    }

    // 1-hop neighbors inherit hub color
    const hubNeighborIds = new Set();
    for (const r of visibleRelations) {
      const e1 = r.entity1_absolute_id, e2 = r.entity2_absolute_id;
      const h1 = hubMap[e1], h2 = hubMap[e2];
      if (h1 !== undefined && h2 === undefined) {
        hubMap[e2] = h1;
        hubNeighborIds.add(e2);
      } else if (h2 !== undefined && h1 === undefined) {
        hubMap[e1] = h2;
        hubNeighborIds.add(e1);
      }
    }

    return { hubMap, hubIds, hubNeighborIds };
  }

  // ---- Resolve unknown relation endpoints and remap to current entity versions ----

  async function resolveAndRemapRelations(entities, relations, graphId) {
    const currentAbsIds = new Set(entities.map(e => e.absolute_id));
    const currentEntityIds = {};
    for (const e of entities) {
      currentEntityIds[e.family_id] = e.absolute_id;
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
        if (oldEntity && currentEntityIds[oldEntity.family_id]) {
          e1AbsId = currentEntityIds[oldEntity.family_id];
          remapped1 = true;
        }
      }
      if (!currentAbsIds.has(e2AbsId)) {
        const oldEntity = resolved[e2AbsId];
        if (oldEntity && currentEntityIds[oldEntity.family_id]) {
          e2AbsId = currentEntityIds[oldEntity.family_id];
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

    const hubLayout = computeHubLayout(visibleRelations);
    explorer.buildGraph(allVisible, visibleRelations, null, null, cachedInheritedRelationIds, undefined, hubLayout);

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
    currentHopLevel = hopLevel;
    if (explorer) explorer.setState('defaultHopLevel', currentHopLevel);
    const loadingEl = document.getElementById('graph-loading');
    const statsEl = document.getElementById('graph-stats');

    if (loadingEl) loadingEl.style.display = 'flex';
    if (statsEl) statsEl.textContent = t('common.loading');

    try {
      const [entityRes, relationRes] = await Promise.all([
        state.api.listEntities(graphId, 5000),
        state.api.listRelations(graphId, 2000),
      ]);

      const allKnownEntities = entityRes.data?.entities || entityRes.data || [];
      const allRelations = relationRes.data?.relations || relationRes.data || [];

      if (allKnownEntities.length === 0) {
        if (statsEl) statsEl.textContent = t('graph.noRelations');
        if (loadingEl) loadingEl.style.display = 'none';
        return;
      }

      cachedAllRawRelations = allRelations;

      const seedEntities = allKnownEntities.slice(0, entityLimit);

      const entityByAbs = {};
      for (const e of allKnownEntities) {
        entityByAbs[e.absolute_id] = e;
      }
      cachedAllEntities = entityByAbs;
      if (explorer) explorer.setEntityCache(cachedAllEntities);

      const { relations: remappedRelations, inheritedRelationIds } = await resolveAndRemapRelations(
        allKnownEntities, allRelations, graphId
      );
      cachedInheritedRelationIds = inheritedRelationIds;
      cachedRemappedMainRelations = remappedRelations;

      const seedAbsIds = new Set(seedEntities.map(e => e.absolute_id));

      const visibleAbsIds = expandNHops(seedAbsIds, remappedRelations, hopLevel);

      const visibleRelations = remappedRelations.filter(r =>
        visibleAbsIds.has(r.entity1_absolute_id) && visibleAbsIds.has(r.entity2_absolute_id)
      );

      const allVisible = [...visibleAbsIds]
        .map(aid => entityByAbs[aid])
        .filter(Boolean);

      const allEntityIds = [...new Set(allVisible.map(e => e.family_id))];
      try {
        const vcRes = await state.api.entityVersionCounts(allEntityIds, graphId);
        const vc = vcRes.data || {};
        if (explorer) explorer.setVersionCounts(vc);
      } catch (_) {}

      cachedAllNodes = allVisible;
      cachedAllEdges = visibleRelations;

      const hubLayout = computeHubLayout(visibleRelations);
      explorer.buildGraph(allVisible, visibleRelations, null, null, cachedInheritedRelationIds, undefined, hubLayout);
      explorer.setMainViewCache(visibleRelations, allVisible, cachedInheritedRelationIds);

      focusAbsoluteId = null;
      const exitBtn = document.getElementById('exit-focus-btn');
      if (exitBtn) exitBtn.style.display = 'none';
      const focusBadge = document.getElementById('focus-mode-badge');
      if (focusBadge) focusBadge.style.display = 'none';

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
      const timeParam = time + ':00';
      const res = await state.api.getSnapshot(timeParam, state.currentGraphId);

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

    cachedAllNodes = snapshotEntities;
    cachedAllEdges = snapshotRelations;
    cachedInheritedRelationIds = null;

    const snapshotEntityIds = [...new Set(snapshotEntities.map(e => e.family_id))];
    var vc = {};
    for (const eid of snapshotEntityIds) {
      vc[eid] = snapshotEntities.filter(e => e.family_id === eid).length;
    }
    if (explorer) explorer.setVersionCounts(vc);

    explorer.buildGraph(snapshotEntities, snapshotRelations, null, null, null);

    const statsEl = document.getElementById('graph-stats');
    if (statsEl) {
      statsEl.textContent = t('graph.loaded', {
        entities: snapshotEntities.length,
        relations: snapshotRelations.length,
      }) + ' [' + t('timeTravel.snapshot') + ']';
    }

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
    loadGraph();
  }

  // ---- Cleanup on page leave ----

  function destroy() {
    if (explorer) {
      explorer.destroy();
      explorer = null;
    }
    focusAbsoluteId = null;
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

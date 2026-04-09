/* ==========================================
   GraphExplorer — Shared graph visualization
   Used by graph.js, communities.js
   Factory pattern: GraphExplorer.create(options)
   ========================================== */

window.GraphExplorer = (function () {
  'use strict';

  function create(options) {
    // ---- Internal state ----
    var _network = null;
    var _entityMap = {};
    var _relationMap = {};
    var _versionCounts = {};
    var _pinnedNodePositions = {};

    var _focusAbsoluteId = null;
    var _currentVersions = [];
    var _currentVersionIdx = 0;
    var _relationScope = 'accumulated';

    // Main view cache (set by graph.js after loadGraph)
    var _mainViewRelations = [];
    var _mainViewEntities = {};
    var _mainViewInheritedRelationIds = null;

    // Focus session (encapsulates accumulation state)
    var _session = null;

    var _opts = options;

    // ---- Helpers ----

    function _el(id) { return document.getElementById(id); }

    // ---- Build vis-network DataSet and initialize the network ----

    function buildGraph(entities, relations, highlightAbsId, hopMap, inheritedRelationIds, futureRelationIds, hubLayout) {
      _entityMap = {};
      _relationMap = {};

      var versionLabel = highlightAbsId && _currentVersions.length > 1
        ? { idx: _currentVersionIdx + 1, total: _currentVersions.length }
        : null;

      // Compute inherited/future entity IDs from relation classification
      var inheritedEntityIds = new Set();
      var futureEntityIds = new Set();
      var hasInherited = inheritedRelationIds && inheritedRelationIds.size > 0;
      var hasFuture = futureRelationIds && futureRelationIds.size > 0;
      if (hasInherited || hasFuture) {
        var entityTypes = {};
        for (var ri = 0; ri < relations.length; ri++) {
          var r = relations[ri];
          var endpoints = [r.entity1_absolute_id, r.entity2_absolute_id];
          for (var ei = 0; ei < endpoints.length; ei++) {
            var eid = endpoints[ei];
            if (eid === highlightAbsId) continue;
            // Skip hop 2+ entities — their colors are determined by propagation (phase 2)
            if (hopMap && hopMap[eid] !== undefined && hopMap[eid] > 1) continue;
            if (!entityTypes[eid]) entityTypes[eid] = new Set();
            if (hasInherited && inheritedRelationIds.has(r.absolute_id)) entityTypes[eid].add('inherited');
            else if (hasFuture && futureRelationIds.has(r.absolute_id)) entityTypes[eid].add('future');
            else entityTypes[eid].add('current');
          }
        }
        var entityTypesKeys = Object.keys(entityTypes);
        for (var ti = 0; ti < entityTypesKeys.length; ti++) {
          var teid = entityTypesKeys[ti];
          var types = entityTypes[teid];
          // Priority: current > inherited > future
          if (types.has('current')) {
            // default blue — don't add to any set
          } else if (types.has('inherited')) {
            inheritedEntityIds.add(teid);
          } else if (types.has('future')) {
            futureEntityIds.add(teid);
          }
        }
      }

      // Propagate entity types from hop 1 to hop 2+ via cascading parent chain
      if (hopMap && (inheritedEntityIds.size > 0 || futureEntityIds.size > 0)) {
        // Build classification map: abs_id → 'inherited' | 'future'
        var entityClassMap = {};
        inheritedEntityIds.forEach(function (id) { entityClassMap[id] = 'inherited'; });
        futureEntityIds.forEach(function (id) { entityClassMap[id] = 'future'; });

        var maxHop = 0;
        for (var mhid in hopMap) {
          if (hopMap[mhid] > maxHop) maxHop = hopMap[mhid];
        }

        // Cascade from hop 2 to maxHop
        for (var ch = 2; ch <= maxHop; ch++) {
          var childTypes = {};
          for (var pi = 0; pi < relations.length; pi++) {
            var pr = relations[pi];
            var ph1 = hopMap[pr.entity1_absolute_id];
            var ph2 = hopMap[pr.entity2_absolute_id];
            if (ph1 === undefined || ph2 === undefined) continue;
            if (Math.abs(ph1 - ph2) !== 1) continue;

            var parent, child;
            if (ph1 === ch - 1 && ph2 === ch) { parent = pr.entity1_absolute_id; child = pr.entity2_absolute_id; }
            else if (ph2 === ch - 1 && ph1 === ch) { parent = pr.entity2_absolute_id; child = pr.entity1_absolute_id; }
            else continue;

            var ptype = entityClassMap[parent];
            if (!ptype) ptype = 'current';

            // Classify connecting edge (only if not already classified by API)
            if (!inheritedRelationIds.has(pr.absolute_id) && !futureRelationIds.has(pr.absolute_id)) {
              if (ptype === 'inherited') inheritedRelationIds.add(pr.absolute_id);
              else if (ptype === 'future') futureRelationIds.add(pr.absolute_id);
            }

            // Collect type for child entity
            if (!childTypes[child]) childTypes[child] = new Set();
            childTypes[child].add(ptype);
          }

          // Apply priority for each child entity
          var childKeys = Object.keys(childTypes);
          for (var ci = 0; ci < childKeys.length; ci++) {
            var ceid = childKeys[ci];
            if (entityClassMap[ceid]) continue; // already classified by earlier hop
            var ctypes = childTypes[ceid];
            if (ctypes.has('current')) { /* default blue */ }
            else if (ctypes.has('inherited')) {
              inheritedEntityIds.add(ceid);
              entityClassMap[ceid] = 'inherited';
            } else if (ctypes.has('future')) {
              futureEntityIds.add(ceid);
              entityClassMap[ceid] = 'future';
            }
          }
        }
      }

      // Determine color mode
      var colorMode = hopMap ? 'hop' : 'default';
      var communityColoringEnabled = _opts.communityColoringEnabled;
      var communityMap = _opts.communityMap;
      if (communityColoringEnabled && communityMap && !hopMap) {
        colorMode = 'community';
      } else if (hubLayout && hubLayout.hubMap && !hopMap) {
        colorMode = 'hub';
      }

      var buildNodesOpts = {
        colorMode: colorMode,
        versionCounts: _versionCounts,
        hopMap: hopMap,
        highlightAbsId: highlightAbsId,
        versionLabel: versionLabel,
        unnamedLabel: t('graph.unnamedEntity'),
        inheritedEntityIds: inheritedEntityIds,
        futureEntityIds: futureEntityIds,
      };
      if (communityMap) buildNodesOpts.communityMap = communityMap;
      if (hubLayout) {
        buildNodesOpts.hubMap = hubLayout.hubMap;
        buildNodesOpts.hubNeighborIds = hubLayout.hubNeighborIds;
      }

      var result = GraphUtils.buildNodes(entities, buildNodesOpts);
      var nodes = result.nodes;
      var eMap = result.entityMap;
      var nodeIds = result.nodeIds;

      var visibleNodeIds = new Set();
      nodes.forEach(function (node) {
        visibleNodeIds.add(node.id);
        var pinned = _pinnedNodePositions[node.id];
        if (pinned) {
          nodes.update({
            id: node.id,
            x: pinned.x,
            y: pinned.y,
            fixed: { x: true, y: true },
          });
        }
      });
      var pinnedKeys = Object.keys(_pinnedNodePositions);
      for (var pi = 0; pi < pinnedKeys.length; pi++) {
        if (!visibleNodeIds.has(pinnedKeys[pi])) delete _pinnedNodePositions[pinnedKeys[pi]];
      }
      _entityMap = eMap;

      // Focus mode: pin the focused entity at canvas center
      if (highlightAbsId && hopMap && nodeIds.has(highlightAbsId)) {
        var focusContainer = _el(_opts.canvasId);
        var fcx = focusContainer.offsetWidth / 2;
        var fcy = focusContainer.offsetHeight / 2;
        nodes.update({ id: highlightAbsId, x: fcx, y: fcy, fixed: { x: true, y: true } });
        _pinnedNodePositions[highlightAbsId] = { x: fcx, y: fcy };
      }

      // Fix top-3 hub nodes in triangle layout
      if (hubLayout && hubLayout.hubIds && hubLayout.hubIds.length > 0) {
        var hubContainer = _el(_opts.canvasId);
        var cx = hubContainer.offsetWidth / 2;
        var cy = hubContainer.offsetHeight / 2;
        var tr = 150;
        var hubPositions = [
          { x: cx, y: cy - tr },
          { x: cx - tr * 0.866, y: cy + tr * 0.5 },
          { x: cx + tr * 0.866, y: cy + tr * 0.5 },
        ];
        for (var hi = 0; hi < hubLayout.hubIds.length && hi < hubPositions.length; hi++) {
          var hubId = hubLayout.hubIds[hi];
          if (nodeIds.has(hubId)) {
            nodes.update({ id: hubId, x: hubPositions[hi].x, y: hubPositions[hi].y, fixed: { x: true, y: true } });
            _pinnedNodePositions[hubId] = hubPositions[hi];
          }
        }
      }

      var buildEdgesOpts = {
        inheritedRelationIds: inheritedRelationIds,
        futureRelationIds: futureRelationIds,
        hopMap: hopMap,
      };
      if (_opts.relationStrengthEnabled) buildEdgesOpts.weightMode = 'count';
      if (hubLayout) buildEdgesOpts.hubMap = hubLayout.hubMap;

      var edgeResult = GraphUtils.buildEdges(relations, nodeIds, buildEdgesOpts);
      var edges = edgeResult.edges;
      var rMap = edgeResult.relationMap;
      _relationMap = rMap;

      var container = _el(_opts.canvasId);
      if (!container) return;

      if (_network) {
        _network.destroy();
        _network = null;
      }

      var visOpts = {
        physics: GraphUtils.getPhysicsOptions(),
        interaction: GraphUtils.getInteractionOptions(),
        layout: { improvedLayout: true },
      };

      _network = new vis.Network(container, { nodes: nodes, edges: edges }, visOpts);

      _network.once('stabilizationIterationsDone', function () {
        // Freeze after stabilization: stop simulation but keep interaction working
        _network.setOptions({ physics: { enabled: false } });
        if (highlightAbsId) {
          _network.focus(highlightAbsId, { scale: 1.2, animation: { duration: 500, easingFunction: 'easeInOutQuad' } });
        }
      });

      _network.on('click', function (params) {
        var nodeId = params.nodes[0];
        var edgeId = params.edges[0];
        if (nodeId) {
          showEntityDetail(nodeId);
        } else if (edgeId) {
          showRelationDetail(edgeId);
        }
      });

      // Allow re-dragging: enable physics during drag so unfixed nodes respond to forces
      _network.on('dragStart', function (params) {
        if (params.nodes.length === 0) return;
        params.nodes.forEach(function (nodeId) {
          nodes.update({ id: nodeId, fixed: false });
        });
        _network.setOptions({ physics: { enabled: true } });
      });

      _network.on('dragEnd', function (params) {
        if (!params.nodes || params.nodes.length === 0) return;
        var positions = _network.getPositions(params.nodes);
        params.nodes.forEach(function (nodeId) {
          var pos = positions[nodeId];
          if (!pos) return;
          _pinnedNodePositions[nodeId] = { x: pos.x, y: pos.y };
          nodes.update({
            id: nodeId,
            x: pos.x,
            y: pos.y,
            fixed: { x: true, y: true },
          });
        });
        _network.setOptions({ physics: { enabled: false } });
      });
    }

    // ---- FocusSession: minimal accumulation tracker ----

    function FocusSession() {
      this.focusFamilyId = null;
      this.accumulatedRelationIds = new Set();
    }

    FocusSession.prototype.reset = function (familyId) {
      this.focusFamilyId = familyId;
      this.accumulatedRelationIds = new Set();
    };

    FocusSession.prototype.merge = function (familyId, currentRelationAbsIds) {
      if (this.focusFamilyId !== familyId) {
        this.reset(familyId);
        return new Set();
      }
      var inherited = new Set();
      var self = this;
      this.accumulatedRelationIds.forEach(function (id) {
        if (!currentRelationAbsIds.has(id)) inherited.add(id);
      });
      currentRelationAbsIds.forEach(function (id) { self.accumulatedRelationIds.add(id); });
      return inherited;
    };

    // ---- Family-ID-Based BFS (no API calls) ----

    function focusBFS(startFamilyId, hopLevel) {
      var entityCache = _opts.entityCache || {};

      // Build abs_id → family_id map
      var absToFid = {};
      for (var absId in entityCache) {
        absToFid[absId] = entityCache[absId].family_id;
      }

      // Build family_id → [relation] index from main view cache
      var familyIndex = {};
      var mainRels = _mainViewRelations || [];
      for (var i = 0; i < mainRels.length; i++) {
        var r = mainRels[i];
        var fid1 = absToFid[r.entity1_absolute_id];
        var fid2 = absToFid[r.entity2_absolute_id];
        if (!fid1 || !fid2) continue;
        if (!familyIndex[fid1]) familyIndex[fid1] = [];
        if (!familyIndex[fid2]) familyIndex[fid2] = [];
        familyIndex[fid1].push(r);
        if (fid1 !== fid2) familyIndex[fid2].push(r);
      }

      // BFS using family_id keys
      var visited = new Set([startFamilyId]);
      var hopMapFid = {};
      hopMapFid[startFamilyId] = 0;
      var discoveredRelations = new Map();
      var frontier = [startFamilyId];

      for (var h = 1; h <= hopLevel; h++) {
        var nextFrontier = [];
        for (var fi = 0; fi < frontier.length; fi++) {
          var fid = frontier[fi];
          var rels = familyIndex[fid] || [];
          for (var ri = 0; ri < rels.length; ri++) {
            var rel = rels[ri];
            discoveredRelations.set(rel.absolute_id, rel);
            var otherFid = absToFid[rel.entity1_absolute_id] === fid
              ? absToFid[rel.entity2_absolute_id]
              : (absToFid[rel.entity2_absolute_id] === fid
                ? absToFid[rel.entity1_absolute_id]
                : null);
            if (otherFid && !visited.has(otherFid)) {
              visited.add(otherFid);
              hopMapFid[otherFid] = h;
              nextFrontier.push(otherFid);
            }
          }
        }
        frontier = nextFrontier;
      }

      var relations = [];
      discoveredRelations.forEach(function (rel) { relations.push(rel); });

      return {
        familyIds: visited,
        hopMapFid: hopMapFid,
        relations: relations,
        absToFid: absToFid
      };
    }

    // ---- Focus on a specific entity version ----

    async function focusOnEntity(absoluteId, opts) {
      opts = opts || {};
      var graphId = state.currentGraphId;
      var loadingEl = _el(_opts.loadingId);
      if (loadingEl) loadingEl.style.display = 'flex';

      try {
        // 1. Resolve entity
        var entity = _entityMap[absoluteId];
        if (!entity) {
          try {
            var res = await state.api.entityByAbsoluteId(absoluteId, graphId);
            entity = res.data;
            if (entity) _entityMap[absoluteId] = entity;
          } catch (_) {}
        }
        if (!entity) {
          showToast(t('graph.loadFailedDetail'), 'error');
          return;
        }

        var familyId = entity.family_id;
        var hopLevel = _opts.defaultHopLevel || 1;

        // 2. BFS topology (family-id based, no API calls)
        var bfs = focusBFS(familyId, hopLevel);

        // 3. Build familyIdToLatest with focus override
        var familyIdToLatest = _opts.familyIdToLatest ? Object.assign({}, _opts.familyIdToLatest()) : {};
        familyIdToLatest[familyId] = absoluteId;

        // 4. Resolve missing absolute_ids (endpoints not in entityCache)
        var absToFid = Object.assign({}, bfs.absToFid);
        var unresolvedAbsIds = [];
        var seenAbs = new Set();
        for (var ui = 0; ui < bfs.relations.length; ui++) {
          var ur = bfs.relations[ui];
          if (!absToFid[ur.entity1_absolute_id] && !seenAbs.has(ur.entity1_absolute_id)) {
            unresolvedAbsIds.push(ur.entity1_absolute_id);
            seenAbs.add(ur.entity1_absolute_id);
          }
          if (!absToFid[ur.entity2_absolute_id] && !seenAbs.has(ur.entity2_absolute_id)) {
            unresolvedAbsIds.push(ur.entity2_absolute_id);
            seenAbs.add(ur.entity2_absolute_id);
          }
        }
        if (unresolvedAbsIds.length > 0) {
          var resolveBatch = unresolvedAbsIds.slice(0, 30);
          var resolvePromises = resolveBatch.map(function (uAbsId) {
            return state.api.entityByAbsoluteId(uAbsId, graphId).then(function (uRes) {
              if (uRes.data) {
                _entityMap[uAbsId] = uRes.data;
                absToFid[uAbsId] = uRes.data.family_id;
              }
            }).catch(function () {});
          });
          await Promise.all(resolvePromises);
        }

        // 5. Collect entity data for all discovered family_ids
        var entityCache = _opts.entityCache || {};
        var entities = [];
        var seenFids = new Set();
        bfs.familyIds.forEach(function (fid) {
          if (seenFids.has(fid)) return;
          var targetAbsId = familyIdToLatest[fid];
          var ent = null;
          if (targetAbsId) ent = _entityMap[targetAbsId] || entityCache[targetAbsId];
          if (!ent) {
            for (var abs in entityCache) {
              if (entityCache[abs].family_id === fid) { ent = entityCache[abs]; break; }
            }
          }
          if (!ent) {
            for (var abs2 in _entityMap) {
              if (_entityMap[abs2].family_id === fid) { ent = _entityMap[abs2]; break; }
            }
          }
          if (ent) {
            entities.push(ent);
            seenFids.add(fid);
          }
        });

        // 6. Remap relation endpoints to target absolute_ids
        var relations = [];
        var relAbsIdSet = new Set();
        for (var ri = 0; ri < bfs.relations.length; ri++) {
          var r = bfs.relations[ri];
          var e1 = r.entity1_absolute_id;
          var e2 = r.entity2_absolute_id;
          var rfid1 = absToFid[e1];
          var rfid2 = absToFid[e2];
          if (rfid1 && familyIdToLatest[rfid1]) e1 = familyIdToLatest[rfid1];
          if (rfid2 && familyIdToLatest[rfid2]) e2 = familyIdToLatest[rfid2];
          relations.push(Object.assign({}, r, { entity1_absolute_id: e1, entity2_absolute_id: e2 }));
          relAbsIdSet.add(r.absolute_id);
        }

        // 7. Convert hopMap: family_id → absolute_id keys
        var hopMap = {};
        var fids = Object.keys(bfs.hopMapFid);
        for (var hi = 0; hi < fids.length; hi++) {
          var hAbsId = familyIdToLatest[fids[hi]];
          if (hAbsId) hopMap[hAbsId] = bfs.hopMapFid[fids[hi]];
        }

        // 8. API classification (parallel) — get _inherited / _future markers
        var inheritedRelationIds = new Set();
        var futureRelationIds = new Set();
        var scope = _relationScope;

        // Optimize: skip API if viewing latest version in accumulated scope
        var latestAbsForFocus = _opts.familyIdToLatest ? _opts.familyIdToLatest()[familyId] : null;
        var isLatestVersion = (absoluteId === latestAbsForFocus);

        if (scope !== 'version_only' && !(isLatestVersion && scope === 'accumulated')) {
          var apiFids = [];
          bfs.familyIds.forEach(function (fid) { apiFids.push(fid); });
          var apiPromises = apiFids.map(function (fid) {
            return state.api.entityRelations(fid, graphId, {
              maxVersionAbsoluteId: absoluteId,
              relationScope: scope
            }).then(function (apiRes) {
              var apiRels = apiRes.data?.relations || apiRes.data || [];
              for (var ai = 0; ai < apiRels.length; ai++) {
                var ar = apiRels[ai];
                if (scope === 'all_versions') {
                  if (ar._version_scope === 'inherited') inheritedRelationIds.add(ar.absolute_id);
                  if (ar._version_scope === 'future') futureRelationIds.add(ar.absolute_id);
                } else {
                  if (ar._inherited) inheritedRelationIds.add(ar.absolute_id);
                }
              }
            }).catch(function () {});
          });
          await Promise.all(apiPromises);
        }

        // 9. Session merge for version switch accumulation
        var isVersionSwitch = _session && _session.focusFamilyId === familyId;
        if (!_session) _session = new FocusSession();
        var sessionInherited = _session.merge(familyId, relAbsIdSet);
        sessionInherited.forEach(function (id) { inheritedRelationIds.add(id); });

        // 10. Filter: only keep connected entities
        var connAbsIds = new Set();
        for (var ci = 0; ci < relations.length; ci++) {
          connAbsIds.add(relations[ci].entity1_absolute_id);
          connAbsIds.add(relations[ci].entity2_absolute_id);
        }
        entities = entities.filter(function (e) { return connAbsIds.has(e.absolute_id); });

        // 11. Fetch version counts
        var allFids = [];
        var vseenIds = new Set();
        for (var vei = 0; vei < entities.length; vei++) {
          if (!vseenIds.has(entities[vei].family_id)) {
            allFids.push(entities[vei].family_id);
            vseenIds.add(entities[vei].family_id);
          }
        }
        try {
          var vcRes = await state.api.entityVersionCounts(allFids, graphId);
          _versionCounts = vcRes.data || {};
        } catch (_) {}

        // 12. Render
        if (!isVersionSwitch) _pinnedNodePositions = {};
        buildGraph(entities, relations, absoluteId, hopMap, inheritedRelationIds, futureRelationIds);

        _focusAbsoluteId = absoluteId;
        var exitBtn = _el(_opts.exitFocusBtnId);
        if (exitBtn) exitBtn.style.display = '';
        var focusBadge = _el(_opts.focusBadgeId);
        if (focusBadge) focusBadge.style.display = '';

        if (_opts.onAfterFocus) _opts.onAfterFocus(entities);
      } catch (err) {
        console.error('Focus failed:', err);
        showToast(t('graph.loadFailed') + ': ' + err.message, 'error');
      } finally {
        if (loadingEl) loadingEl.style.display = 'none';
      }
    }

    // ---- Exit focus mode ----

    function exitFocus() {
      _focusAbsoluteId = null;
      _currentVersions = [];
      _currentVersionIdx = 0;
      _session = null;
      var exitBtn = _el(_opts.exitFocusBtnId);
      if (exitBtn) exitBtn.style.display = 'none';
      var focusBadge = _el(_opts.focusBadgeId);
      if (focusBadge) focusBadge.style.display = 'none';

      if (_opts.onRestoreDefaultView) {
        var view = _opts.onRestoreDefaultView();
        buildGraph(view.entities, view.relations, null, null, view.inheritedRelationIds, undefined, view.hubLayout);
      }

      var detailContent = _el(_opts.detailContentId);
      if (detailContent) {
        detailContent.innerHTML = emptyState(t('common.clickToView'), 'mouse-pointer-click');
      }
    }

    // ---- Show entity detail in the sidebar ----

    async function showEntityDetail(absoluteId) {
      var entity = _entityMap[absoluteId];
      if (!entity) {
        try {
          var res = await state.api.entityByAbsoluteId(absoluteId, state.currentGraphId);
          if (res.data) { entity = res.data; _entityMap[absoluteId] = entity; }
        } catch (_) {}
      }
      if (!entity) return;

      var detailContent = _el(_opts.detailContentId);
      if (!detailContent) return;

      var familyId = entity.family_id;

      var versions = [];
      try {
        var vRes = await state.api.entityVersions(familyId, state.currentGraphId);
        versions = vRes.data || [];
      } catch (_) {}

      _currentVersions = versions;
      _currentVersionIdx = -1;
      for (var vi = 0; vi < versions.length; vi++) {
        if (versions[vi].absolute_id === absoluteId) { _currentVersionIdx = vi; break; }
      }
      if (_currentVersionIdx < 0) _currentVersionIdx = 0;

      var totalVersions = versions.length;
      var prefix = _opts.idPrefix || '';

      detailContent.innerHTML =
        '<div class="flex items-center justify-between mb-3">' +
          '<span class="badge badge-primary">' + t('graph.entityDetail') + '</span>' +
          (totalVersions > 1 ?
            '<div class="flex items-center gap-1">' +
              '<button class="btn btn-secondary btn-sm" id="' + prefix + 'prev-ver-btn" ' + (_currentVersionIdx === 0 ? 'disabled' : '') + ' title="' + t('graph.prevVersion') + '">' +
                '<i data-lucide="chevron-left" style="width:14px;height:14px;"></i>' +
              '</button>' +
              '<span class="mono text-xs" style="color:var(--text-muted);min-width:50px;text-align:center;">' +
                (_currentVersionIdx + 1) + '/' + totalVersions +
              '</span>' +
              '<button class="btn btn-secondary btn-sm" id="' + prefix + 'next-ver-btn" ' + (_currentVersionIdx === totalVersions - 1 ? 'disabled' : '') + ' title="' + t('graph.nextVersion') + '">' +
                '<i data-lucide="chevron-right" style="width:14px;height:14px;"></i>' +
              '</button>' +
            '</div>'
          : '') +
        '</div>' +

        '<h3 style="font-size:1.1rem;font-weight:600;color:var(--text-primary);margin-bottom:0.75rem;word-break:break-word;">' +
          escapeHtml(entity.name || t('graph.unnamedEntity')) +
          (totalVersions > 1 ? ' <span style="color:var(--text-muted);font-size:0.85rem;font-weight:400;"> [' + (_currentVersionIdx + 1) + '/' + totalVersions + ']</span>' : '') +
        '</h3>' +

        '<div class="flex flex-wrap gap-2 mb-3">' +
          '<button class="btn btn-secondary btn-sm" id="' + prefix + 'view-versions-btn">' +
            '<i data-lucide="git-branch" style="width:14px;height:14px;"></i> ' + t('graph.versionHistory') +
          '</button>' +
          '<button class="btn btn-secondary btn-sm" id="' + prefix + 'view-relations-btn">' +
            '<i data-lucide="link" style="width:14px;height:14px;"></i> ' + t('graph.viewRelations') +
          '</button>' +
          '<button class="btn btn-primary btn-sm" id="' + prefix + 'focus-entity-btn">' +
            '<i data-lucide="crosshair" style="width:14px;height:14px;"></i> ' + t('graph.focusMode') +
          '</button>' +
        '</div>' +

        (_focusAbsoluteId ?
          '<div style="margin-bottom:0.75rem;">' +
            '<label style="display:flex;align-items:center;gap:0.35rem;font-size:0.8rem;color:var(--text-secondary);">' +
              t('graph.relationScope') + ' ' +
              '<select id="' + prefix + 'relation-scope-sel" style="font-size:0.8rem;padding:0.15rem 0.3rem;border-radius:0.25rem;background:var(--bg-secondary);color:var(--text-primary);border:1px solid var(--border-primary);">' +
                '<option value="accumulated"' + (_relationScope === 'accumulated' ? ' selected' : '') + '>' + t('graph.scopeAccumulated') + '</option>' +
                '<option value="version_only"' + (_relationScope === 'version_only' ? ' selected' : '') + '>' + t('graph.scopeVersionOnly') + '</option>' +
                '<option value="all_versions"' + (_relationScope === 'all_versions' ? ' selected' : '') + '>' + t('graph.scopeAllVersions') + '</option>' +
              '</select>' +
            '</label>' +
          '</div>'
        : '') +

        '<div class="divider"></div>' +

        '<div style="display:flex;flex-direction:column;gap:0.75rem;">' +
          '<div>' +
            '<span class="form-label" style="margin-bottom:0.125rem;">' + t('graph.content') + '</span>' +
            '<div class="md-content" style="font-size:0.8125rem;color:var(--text-secondary);">' +
              renderMarkdown(entity.content || '-') +
            '</div>' +
          '</div>' +

          '<div>' +
            '<span class="form-label" style="margin-bottom:0.125rem;">' + t('graph.entityId') + '</span>' +
            '<p class="mono truncate" style="color:var(--text-muted);font-size:0.75rem;" title="' + escapeHtml(entity.family_id || '') + '">' +
              escapeHtml(entity.family_id || '-') +
            '</p>' +
          '</div>' +

          '<div>' +
            '<span class="form-label" style="margin-bottom:0.125rem;">' + t('graph.absoluteId') + '</span>' +
            '<p class="mono truncate" style="color:var(--text-muted);font-size:0.75rem;" title="' + escapeHtml(entity.absolute_id || '') + '">' +
              escapeHtml(entity.absolute_id || '-') +
            '</p>' +
          '</div>' +

          '<div>' +
            '<span class="form-label" style="margin-bottom:0.125rem;">' + t('graph.eventTime') + '</span>' +
            '<p style="font-size:0.8125rem;color:var(--text-secondary);">' +
              formatDate(entity.event_time) +
            '</p>' +
          '</div>' +

          '<div>' +
            '<span class="form-label" style="margin-bottom:0.125rem;">' + t('graph.processedTime') + '</span>' +
            '<p style="font-size:0.8125rem;color:var(--text-secondary);">' +
              formatDate(entity.processed_time) +
            '</p>' +
          '</div>' +

          (entity.source_document ?
            '<div>' +
              '<span class="form-label" style="margin-bottom:0.125rem;">' + t('graph.sourceDoc') + '</span>' +
              '<span class="doc-link mono truncate" style="font-size:0.75rem;" data-view-doc="' + escapeHtml(entity.source_document) + '" title="' + escapeHtml(entity.source_document) + '">' +
                escapeHtml(truncate(entity.source_document, 60)) +
              '</span>' +
            '</div>'
          : '') +

          (entity.episode_id ?
            '<div>' +
              '<span class="form-label" style="margin-bottom:0.125rem;">' + t('graph.episodeId') + '</span>' +
              '<span class="doc-link mono truncate" style="font-size:0.75rem;" data-view-doc="' + escapeHtml(entity.episode_id) + '" title="' + t('common.clickToView') + '">' +
                escapeHtml(entity.episode_id) +
              '</span>' +
            '</div>'
          : '') +
        '</div>';

      if (window.lucide) lucide.createIcons({ nodes: [detailContent] });

      detailContent.querySelectorAll('[data-view-doc]').forEach(function (el) {
        el.addEventListener('click', function () { window.showDocContent(el.getAttribute('data-view-doc')); });
      });

      _el(prefix + 'view-versions-btn').addEventListener('click', function () {
        openVersionsModal(entity);
      });
      _el(prefix + 'view-relations-btn').addEventListener('click', function () {
        openRelationsModal(entity);
      });
      _el(prefix + 'focus-entity-btn').addEventListener('click', function () {
        focusOnEntity(absoluteId);
      });

      var scopeSel = _el(prefix + 'relation-scope-sel');
      if (scopeSel) {
        scopeSel.addEventListener('change', function () {
          _relationScope = scopeSel.value;
          // Reset session — scope change is a new focus context
          _session = null;
          focusOnEntity(absoluteId);
        });
      }

      var prevBtn = _el(prefix + 'prev-ver-btn');
      var nextBtn = _el(prefix + 'next-ver-btn');
      if (prevBtn) {
        prevBtn.addEventListener('click', function () {
          if (_currentVersionIdx > 0) switchVersion(_currentVersionIdx - 1);
        });
      }
      if (nextBtn) {
        nextBtn.addEventListener('click', function () {
          if (_currentVersionIdx < _currentVersions.length - 1) switchVersion(_currentVersionIdx + 1);
        });
      }
    }

    // ---- Show relation detail in the sidebar ----

    function showRelationDetail(absoluteId) {
      var relation = _relationMap[absoluteId];
      if (!relation) return;

      var detailContent = _el(_opts.detailContentId);
      if (!detailContent) return;

      var fromName = (_entityMap[relation.entity1_absolute_id] || {}).name || relation.entity1_absolute_id || '?';
      var toName = (_entityMap[relation.entity2_absolute_id] || {}).name || relation.entity2_absolute_id || '?';

      detailContent.innerHTML =
        '<div class="flex items-center gap-2 mb-3">' +
          '<span class="badge" style="background:var(--info-dim);color:var(--info);">' + t('graph.relationDetail') + '</span>' +
        '</div>' +

        '<h3 style="font-size:1.1rem;font-weight:600;color:var(--text-primary);margin-bottom:0.75rem;word-break:break-word;">' +
          escapeHtml(truncate(relation.content || t('graph.unnamedRelation'), 60)) +
        '</h3>' +

        '<div class="divider"></div>' +

        '<div style="display:flex;flex-direction:column;gap:0.75rem;">' +
          '<div>' +
            '<span class="form-label" style="margin-bottom:0.125rem;">' + t('graph.content') + '</span>' +
            '<div class="md-content" style="font-size:0.8125rem;color:var(--text-secondary);">' +
              renderMarkdown(relation.content || '-') +
            '</div>' +
          '</div>' +

          '<div>' +
            '<span class="form-label" style="margin-bottom:0.125rem;">' + t('graph.fromEntity') + '</span>' +
            '<div class="flex items-center gap-2">' +
              '<span class="mono truncate" style="color:var(--info);font-size:0.75rem;cursor:pointer;text-decoration:underline;" data-view-entity="' + escapeHtml(relation.entity1_absolute_id) + '">' + escapeHtml(truncate(fromName, 40)) + '</span>' +
              '<button class="btn btn-secondary btn-sm" style="padding:0.125rem 0.375rem;" data-focus-entity="' + escapeHtml(relation.entity1_absolute_id) + '" title="' + t('graph.focusMode') + '">' +
                '<i data-lucide="crosshair" style="width:12px;height:12px;"></i>' +
              '</button>' +
            '</div>' +
          '</div>' +

          '<div>' +
            '<span class="form-label" style="margin-bottom:0.125rem;">' + t('graph.toEntity') + '</span>' +
            '<div class="flex items-center gap-2">' +
              '<span class="mono truncate" style="color:var(--info);font-size:0.75rem;cursor:pointer;text-decoration:underline;" data-view-entity="' + escapeHtml(relation.entity2_absolute_id) + '">' + escapeHtml(truncate(toName, 40)) + '</span>' +
              '<button class="btn btn-secondary btn-sm" style="padding:0.125rem 0.375rem;" data-focus-entity="' + escapeHtml(relation.entity2_absolute_id) + '" title="' + t('graph.focusMode') + '">' +
                '<i data-lucide="crosshair" style="width:12px;height:12px;"></i>' +
              '</button>' +
            '</div>' +
          '</div>' +

          '<div>' +
            '<span class="form-label" style="margin-bottom:0.125rem;">' + t('graph.relationId') + '</span>' +
            '<p class="mono truncate" style="color:var(--text-muted);font-size:0.75rem;">' +
              escapeHtml(relation.family_id || '-') +
            '</p>' +
          '</div>' +

          '<div>' +
            '<span class="form-label" style="margin-bottom:0.125rem;">' + t('graph.eventTime') + '</span>' +
            '<p style="font-size:0.8125rem;color:var(--text-secondary);">' +
              formatDate(relation.event_time) +
            '</p>' +
          '</div>' +

          '<div>' +
            '<span class="form-label" style="margin-bottom:0.125rem;">' + t('graph.processedTime') + '</span>' +
            '<p style="font-size:0.8125rem;color:var(--text-secondary);">' +
              formatDate(relation.processed_time) +
            '</p>' +
          '</div>' +

          (relation.source_document ?
            '<div>' +
              '<span class="form-label" style="margin-bottom:0.125rem;">' + t('graph.sourceDoc') + '</span>' +
              '<span class="doc-link mono truncate" style="font-size:0.75rem;" data-view-doc="' + escapeHtml(relation.source_document) + '" title="' + escapeHtml(relation.source_document) + '">' +
                escapeHtml(truncate(relation.source_document, 60)) +
              '</span>' +
            '</div>'
          : '') +
        '</div>';

      detailContent.querySelectorAll('[data-view-entity]').forEach(function (el) {
        el.addEventListener('click', function () { showEntityDetail(el.getAttribute('data-view-entity')); });
      });
      detailContent.querySelectorAll('[data-focus-entity]').forEach(function (el) {
        el.addEventListener('click', function () { focusOnEntity(el.getAttribute('data-focus-entity')); });
      });
      detailContent.querySelectorAll('[data-view-doc]').forEach(function (el) {
        el.addEventListener('click', function () { window.showDocContent(el.getAttribute('data-view-doc')); });
      });

      if (window.lucide) lucide.createIcons({ nodes: [detailContent] });
    }

    // ---- Switch to a different version of the current entity ----

    async function switchVersion(newIdx) {
      if (!_currentVersions[newIdx]) return;
      _currentVersionIdx = newIdx;

      var version = _currentVersions[newIdx];
      var absoluteId = version.absolute_id;

      if (!_entityMap[absoluteId]) {
        _entityMap[absoluteId] = version;
      }

      await focusOnEntity(absoluteId, { isVersionSwitch: true });
      await showEntityDetail(absoluteId);
    }

    // ---- Versions modal ----

    async function openVersionsModal(entity) {
      var familyId = entity.family_id || entity.absolute_id;
      var graphId = state.currentGraphId;

      var modal = showModal({
        title: t('graph.versionsTitle', { name: truncate(entity.name || familyId, 40) }),
        content: '<div class="flex justify-center p-6">' + spinnerHtml() + '</div>',
        size: 'lg',
      });

      try {
        var res = await state.api.entityVersions(familyId, graphId);
        var versions = res.data || [];

        if (versions.length === 0) {
          modal.overlay.querySelector('.modal-body').innerHTML = emptyState(t('graph.noVersions'));
          return;
        }

        var rows = versions.map(function (v) {
          return '<tr>' +
            '<td style="max-width:120px;">' + formatDate(v.processed_time) + '</td>' +
            '<td style="max-width:200px;" title="' + escapeHtml(v.name || '') + '">' + escapeHtml(truncate(v.name || '-', 30)) + '</td>' +
            '<td style="max-width:300px;" title="' + escapeHtml(v.content || '') + '">' + escapeHtml(truncate(v.content || '-', 50)) + '</td>' +
          '</tr>';
        }).join('');

        modal.overlay.querySelector('.modal-body').innerHTML =
          '<div class="table-container" style="max-height:50vh;overflow-y:auto;">' +
            '<table class="data-table">' +
              '<thead><tr>' +
                '<th>' + t('graph.versionTime') + '</th>' +
                '<th>' + t('graph.versionName') + '</th>' +
                '<th>' + t('graph.versionContent') + '</th>' +
              '</tr></thead>' +
              '<tbody>' + rows + '</tbody>' +
            '</table>' +
          '</div>' +
          '<p style="margin-top:0.75rem;font-size:0.75rem;color:var(--text-muted);">' + t('graph.versionCount', { count: versions.length }) + '</p>';
      } catch (err) {
        modal.overlay.querySelector('.modal-body').innerHTML =
          '<div class="empty-state">' +
            '<i data-lucide="alert-triangle"></i>' +
            '<p>' + t('graph.loadFailedDetail') + ': ' + escapeHtml(err.message) + '</p>' +
          '</div>';
        if (window.lucide) lucide.createIcons({ nodes: [modal.overlay] });
      }
    }

    // ---- Relations modal ----

    async function openRelationsModal(entity) {
      var familyId = entity.family_id || entity.absolute_id;
      var graphId = state.currentGraphId;

      var modal = showModal({
        title: t('graph.relationsTitle', { name: truncate(entity.name || familyId, 40) }),
        content: '<div class="flex justify-center p-6">' + spinnerHtml() + '</div>',
        size: 'lg',
      });

      try {
        var res = await state.api.entityRelations(familyId, graphId);
        var relations = res.data || [];

        // Apply optional filter callback
        if (_opts.onFilterRelations) {
          relations = _opts.onFilterRelations(relations);
        }

        if (relations.length === 0) {
          modal.overlay.querySelector('.modal-body').innerHTML = emptyState(t('graph.noRelations'));
          return;
        }

        var entityCache = _opts.entityCache || {};
        var rows = relations.map(function (r) {
          var otherAbsId = r.entity1_absolute_id === entity.absolute_id
            ? r.entity2_absolute_id : r.entity1_absolute_id;
          var otherEntity = _entityMap[otherAbsId] || entityCache[otherAbsId];
          var otherName = otherEntity ? (otherEntity.name || otherEntity.family_id || '-') : '-';
          return '<tr>' +
            '<td style="max-width:250px;" title="' + escapeHtml(r.content || '') + '">' + escapeHtml(truncate(r.content || '-', 40)) + '</td>' +
            '<td style="max-width:120px;" title="' + escapeHtml(otherName) + '">' + escapeHtml(truncate(otherName, 20)) + '</td>' +
            '<td class="mono" style="max-width:120px;font-size:0.75rem;color:var(--text-muted);">' + formatDate(r.event_time) + '</td>' +
          '</tr>';
        }).join('');

        modal.overlay.querySelector('.modal-body').innerHTML =
          '<div class="table-container" style="max-height:50vh;overflow-y:auto;">' +
            '<table class="data-table">' +
              '<thead><tr>' +
                '<th>' + t('graph.content') + '</th>' +
                '<th>' + t('graph.toEntity') + '</th>' +
                '<th>' + t('graph.versionTime') + '</th>' +
              '</tr></thead>' +
              '<tbody>' + rows + '</tbody>' +
            '</table>' +
          '</div>' +
          '<p style="margin-top:0.75rem;font-size:0.75rem;color:var(--text-muted);">' + t('graph.relationCount', { count: relations.length }) + '</p>';
      } catch (err) {
        modal.overlay.querySelector('.modal-body').innerHTML =
          '<div class="empty-state">' +
            '<i data-lucide="alert-triangle"></i>' +
            '<p>' + t('graph.loadFailedDetail') + ': ' + escapeHtml(err.message) + '</p>' +
          '</div>';
        if (window.lucide) lucide.createIcons({ nodes: [modal.overlay] });
      }
    }

    // ---- Public API ----

    return {
      buildGraph: buildGraph,
      focusOnEntity: focusOnEntity,
      exitFocus: exitFocus,
      showEntityDetail: showEntityDetail,
      showRelationDetail: showRelationDetail,
      switchVersion: switchVersion,
      openVersionsModal: openVersionsModal,
      openRelationsModal: openRelationsModal,
      setEntityCache: function (cache) { _opts.entityCache = cache; },
      setMainViewCache: function (relations, entities, inheritedIds) {
        _mainViewRelations = relations || [];
        _mainViewEntities = entities || {};
        _mainViewInheritedRelationIds = inheritedIds || null;
      },
      setVersionCounts: function (vc) { _versionCounts = vc; },
      setState: function (key, val) {
        if (key === 'relationScope') _relationScope = val;
        if (key === 'communityColoringEnabled') _opts.communityColoringEnabled = val;
        if (key === 'communityMap') _opts.communityMap = val;
        if (key === 'relationStrengthEnabled') _opts.relationStrengthEnabled = val;
        if (key === 'defaultHopLevel') _opts.defaultHopLevel = val;
      },
      getState: function () {
        return {
          focusAbsoluteId: _focusAbsoluteId,
          currentVersions: _currentVersions,
          currentVersionIdx: _currentVersionIdx,
          relationScope: _relationScope,
          entityMap: _entityMap,
          relationMap: _relationMap,
          versionCounts: _versionCounts,
        };
      },
      destroy: function () {
        if (_network) {
          _network.destroy();
          _network = null;
        }
        _entityMap = {};
        _relationMap = {};
        _versionCounts = {};
        _pinnedNodePositions = {};
        _focusAbsoluteId = null;
        _currentVersions = [];
        _currentVersionIdx = 0;
        _relationScope = 'accumulated';
        _session = null;
      },
    };
  }

  return { create: create };
})();

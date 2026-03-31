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

    // Accumulation state for version-switching
    var _accumEntities = null;       // Map<absolute_id, entity> — accumulated entities
    var _accumRelationsByRid = null; // Map<relation_id, relation> — dedup by relation_id
    var _accumHopMap = null;         // Map<absolute_id, number> — min hop per entity
    var _accumFocusEntityId = null;  // entity_id being focused (detect version switch vs new focus)

    var _focusAbsoluteId = null;
    var _currentVersions = [];
    var _currentVersionIdx = 0;
    var _relationScope = 'accumulated';

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

    // ---- Multi-hop BFS for focus mode ----

    async function fetchMultiHop(startAbsId, startEntityId, hopLevel) {
      var graphId = state.currentGraphId;
      var hopMap = {};
      hopMap[startAbsId] = 0;
      var relationSet = new Map();   // abs_id → raw relation from API
      var inheritedRelationIds = new Set();
      var futureRelationIds = new Set();
      var frontier = [{ absId: startAbsId, entityId: startEntityId }];

      var absToEntityId = {};
      if (startEntityId) absToEntityId[startAbsId] = startEntityId;

      var entityCache = _opts.entityCache || {};
      var scope = _relationScope;    // 'version_only' | 'accumulated' | 'all_versions'

      for (var h = 1; h <= hopLevel; h++) {
        var nextFrontier = [];
        var MAX_PER_HOP = _opts.maxPerHop || 999;

        // ---- Between hops: resolve missing entityIds via API ----
        var unresolved = frontier.filter(function (n) { return !n.entityId; });
        if (unresolved.length > 0) {
          var resolvePromises = unresolved.slice(0, 30).map(function (n) {
            return state.api.entityByAbsoluteId(n.absId, graphId).then(function (uRes) {
              if (uRes.data) {
                _entityMap[n.absId] = uRes.data;
                n.entityId = uRes.data.entity_id;
                absToEntityId[n.absId] = uRes.data.entity_id;
              }
            }).catch(function () {});
          });
          await Promise.all(resolvePromises);
        }

        // ---- Expand each frontier node ----
        for (var ni = 0; ni < frontier.length; ni++) {
          var node = frontier[ni];
          var apiRels = [];

          try {
            var res;
            if (node.entityId) {
              res = await state.api.entityRelations(node.entityId, graphId, {
                maxVersionAbsoluteId: node.absId,
                relationScope: scope
              });
            } else {
              res = await state.api.entityOneHop(node.absId, graphId);
            }
            apiRels = res.data || [];

            // ---- Classify relations from API markers (ALL hops, ALL nodes) ----
            if (node.entityId && scope !== 'version_only') {
              for (var ci = 0; ci < apiRels.length; ci++) {
                var ar = apiRels[ci];
                if (scope === 'all_versions') {
                  // Backend marks each relation with _version_scope: current|inherited|future
                  if (ar._version_scope === 'inherited') inheritedRelationIds.add(ar.absolute_id);
                  if (ar._version_scope === 'future') futureRelationIds.add(ar.absolute_id);
                } else {
                  // accumulated: backend marks _inherited and _future booleans
                  if (ar._inherited) inheritedRelationIds.add(ar.absolute_id);
                  // Do NOT add _future to futureRelationIds — accumulated excludes future
                }
              }
            }
          } catch (_) {}

          // ---- Add to relation set; filter _future for accumulated scope ----
          for (var ri = 0; ri < apiRels.length; ri++) {
            var rr = apiRels[ri];
            // For accumulated scope, completely discard _future relations
            if (scope === 'accumulated' && rr._future) continue;
            relationSet.set(rr.absolute_id, rr);

            // Discover new frontier nodes
            var otherId = rr.entity1_absolute_id === node.absId
              ? rr.entity2_absolute_id : rr.entity1_absolute_id;
            if (otherId && !(otherId in hopMap)) {
              hopMap[otherId] = h;
              nextFrontier.push({ absId: otherId, entityId: null });
            }
          }

          // Quick cache hit for new frontier nodes
          for (var fi = 0; fi < nextFrontier.length; fi++) {
            if (nextFrontier[fi].entityId) continue;
            var ent = _entityMap[nextFrontier[fi].absId] || entityCache[nextFrontier[fi].absId];
            if (ent) {
              nextFrontier[fi].entityId = ent.entity_id;
              absToEntityId[nextFrontier[fi].absId] = ent.entity_id;
            }
          }
        }

        frontier = nextFrontier.slice(0, MAX_PER_HOP * h);
      }

      // ---- Post-BFS: resolve entity data, deduplicate, remap relations ----

      var rawEntities = [];
      var resolvedIds = new Set();
      var hopKeys = Object.keys(hopMap);
      for (var hki = 0; hki < hopKeys.length; hki++) {
        var absId = hopKeys[hki];
        if (resolvedIds.has(absId)) continue;
        var rEnt = _entityMap[absId] || entityCache[absId];
        if (rEnt) {
          rawEntities.push(rEnt);
          resolvedIds.add(absId);
          absToEntityId[absId] = rEnt.entity_id;
        }
      }

      // Build entity_id -> latest absId from entityCache
      var entityIdToLatest = {};
      if (_opts.entityIdToLatest) {
        entityIdToLatest = Object.assign({}, _opts.entityIdToLatest());
      }
      // Override: focused entity always maps to the version being viewed
      if (startEntityId) entityIdToLatest[startEntityId] = startAbsId;

      // Deduplicate: keep only latest version per entity_id
      var dedupedEntities = [];
      var seenEntityIds = new Set();
      for (var dei = 0; dei < rawEntities.length; dei++) {
        var dEnt = rawEntities[dei];
        var latestAbsId = entityIdToLatest[dEnt.entity_id];
        if (!latestAbsId || dEnt.absolute_id === latestAbsId) {
          if (!seenEntityIds.has(dEnt.entity_id)) {
            dedupedEntities.push(dEnt);
            seenEntityIds.add(dEnt.entity_id);
          }
        }
      }

      // Resolve unknown endpoints
      var unknownEndpoints = new Set();
      relationSet.forEach(function (rel) {
        if (!resolvedIds.has(rel.entity1_absolute_id)) unknownEndpoints.add(rel.entity1_absolute_id);
        if (!resolvedIds.has(rel.entity2_absolute_id)) unknownEndpoints.add(rel.entity2_absolute_id);
      });
      if (unknownEndpoints.size > 0) {
        var toResolve = Array.from(unknownEndpoints).slice(0, 30);
        var resolvePromises = toResolve.map(async function (uAbsId) {
          try {
            var uRes = await state.api.entityByAbsoluteId(uAbsId, graphId);
            if (uRes.data) {
              _entityMap[uAbsId] = uRes.data;
              absToEntityId[uAbsId] = uRes.data.entity_id;
            }
          } catch (_) {}
        });
        await Promise.all(resolvePromises);
      }

      // Remap relation endpoints to latest visible versions
      var finalRelations = [];
      relationSet.forEach(function (rel) {
        var e1 = rel.entity1_absolute_id;
        var e2 = rel.entity2_absolute_id;
        var skip = false;

        var eid1 = absToEntityId[e1];
        if (eid1 && entityIdToLatest[eid1]) {
          e1 = entityIdToLatest[eid1];
        } else {
          skip = true;
        }
        var eid2 = absToEntityId[e2];
        if (eid2 && entityIdToLatest[eid2]) {
          e2 = entityIdToLatest[eid2];
        } else {
          skip = true;
        }

        if (skip) return;

        for (var rei = 0; rei < 2; rei++) {
          var rAbsId = rei === 0 ? e1 : e2;
          var rEnt2 = entityCache[rAbsId] || _entityMap[rAbsId];
          if (rEnt2 && !seenEntityIds.has(rEnt2.entity_id)) {
            dedupedEntities.push(rEnt2);
            seenEntityIds.add(rEnt2.entity_id);
          }
        }

        var oldHop1 = hopMap[rel.entity1_absolute_id];
        if (oldHop1 !== undefined) {
          hopMap[e1] = Math.min(hopMap[e1] !== undefined ? hopMap[e1] : Infinity, oldHop1);
        }
        var oldHop2 = hopMap[rel.entity2_absolute_id];
        if (oldHop2 !== undefined) {
          hopMap[e2] = Math.min(hopMap[e2] !== undefined ? hopMap[e2] : Infinity, oldHop2);
        }

        finalRelations.push({ absolute_id: rel.absolute_id, relation_id: rel.relation_id, entity1_absolute_id: e1, entity2_absolute_id: e2, content: rel.content, event_time: rel.event_time, processed_time: rel.processed_time, source_document: rel.source_document, _inherited: rel._inherited, _version_scope: rel._version_scope });
      });

      // Filter out edges that skip hops (e.g., hop 0 directly to hop 2+)
      finalRelations = finalRelations.filter(function (r) {
        var fh1 = hopMap[r.entity1_absolute_id];
        var fh2 = hopMap[r.entity2_absolute_id];
        if (fh1 === undefined || fh2 === undefined) return true;
        return Math.abs(fh1 - fh2) <= 1;
      });

      // Clean hopMap: remove ghost entries for deduplicated entities
      var cleanEntityIds = new Set();
      for (var dci = 0; dci < dedupedEntities.length; dci++) {
        cleanEntityIds.add(dedupedEntities[dci].absolute_id);
      }
      var hopKeysToClean = Object.keys(hopMap);
      for (var hki = 0; hki < hopKeysToClean.length; hki++) {
        if (!cleanEntityIds.has(hopKeysToClean[hki])) delete hopMap[hopKeysToClean[hki]];
      }

      // Filter: only keep connected entities
      var connectedNodeIds = new Set();
      for (var cri = 0; cri < finalRelations.length; cri++) {
        connectedNodeIds.add(finalRelations[cri].entity1_absolute_id);
        connectedNodeIds.add(finalRelations[cri].entity2_absolute_id);
      }
      var finalEntities = dedupedEntities.filter(function (e) { return connectedNodeIds.has(e.absolute_id); });

      return { hopMap: hopMap, entities: finalEntities, relations: finalRelations, inheritedRelationIds: inheritedRelationIds, futureRelationIds: futureRelationIds };
    }

    // ---- Focus on a specific entity version (multi-hop view) ----

    async function focusOnEntity(absoluteId, opts) {
      opts = opts || {};
      var isVersionSwitch = opts.isVersionSwitch || false;

      var graphId = state.currentGraphId;
      var loadingEl = _el(_opts.loadingId);
      if (loadingEl) loadingEl.style.display = 'flex';

      try {
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

        // Detect if this is actually a version switch (same entity_id, accumulation active)
        if (_accumEntities !== null && entity.entity_id === _accumFocusEntityId) {
          isVersionSwitch = true;
        }

        if (!isVersionSwitch) {
          // New focus session — reset accumulation
          _accumEntities = new Map();
          _accumRelationsByRid = new Map();
          _accumHopMap = {};
          _accumFocusEntityId = entity.entity_id;
        }

        var hopLevel = _opts.defaultHopLevel || 1;
        var multiHopResult = await fetchMultiHop(absoluteId, entity.entity_id, hopLevel);

        // Merge into accumulation
        // Safety: ensure accumulation is always initialized before merge
        if (!_accumEntities) {
          _accumEntities = new Map();
          _accumRelationsByRid = new Map();
          _accumHopMap = {};
          _accumFocusEntityId = entity.entity_id;
          isVersionSwitch = false;
        }
        for (var mei = 0; mei < multiHopResult.entities.length; mei++) {
          _accumEntities.set(multiHopResult.entities[mei].absolute_id, multiHopResult.entities[mei]);
        }
        for (var mri = 0; mri < multiHopResult.relations.length; mri++) {
          var mr = multiHopResult.relations[mri];
          var dedupKey = mr.relation_id || mr.absolute_id;
          // Always update: replace old version's relation with remapped version
          _accumRelationsByRid.set(dedupKey, mr);
        }
        for (var mhid in multiHopResult.hopMap) {
          _accumHopMap[mhid] = Math.min(_accumHopMap[mhid] !== undefined ? _accumHopMap[mhid] : Infinity, multiHopResult.hopMap[mhid]);
        }

        // Build classification sets
        var inheritedRelationIds = new Set(multiHopResult.inheritedRelationIds);
        var futureRelationIds = new Set(multiHopResult.futureRelationIds);

        if (isVersionSwitch) {
          // Relations accumulated from previous versions but not in new BFS → inherited
          var newRelAbsIds = new Set(multiHopResult.relations.map(function (r) { return r.absolute_id; }));
          _accumRelationsByRid.forEach(function (rel, rid) {
            if (!newRelAbsIds.has(rel.absolute_id)) {
              inheritedRelationIds.add(rel.absolute_id);
            }
          });
        }

        // Deduplicate accumulated entities by entity_id (version switching can add duplicates)
        var dedupedAccum = {};
        _accumEntities.forEach(function (ent, absId) {
          var eid = ent.entity_id;
          if (!dedupedAccum[eid]) {
            dedupedAccum[eid] = ent;
          } else {
            // For focused entity, prefer the version being viewed
            if (eid === _accumFocusEntityId && absId === absoluteId) {
              dedupedAccum[eid] = ent;
            }
            // Otherwise, keep the later/larger entity (prefer deduped from fetchMultiHop remapping)
          }
        });

        // Build final arrays from deduplicated accumulation
        var entities = Object.values(dedupedAccum);
        var relations = Array.from(_accumRelationsByRid.values());
        var hopMap = Object.assign({}, _accumHopMap);

        // Clean hopMap: remove entries not in final entities
        var finalEntityIds = new Set();
        for (var fei = 0; fei < entities.length; fei++) {
          finalEntityIds.add(entities[fei].absolute_id);
        }
        var hopKeysToRemove = Object.keys(hopMap).filter(function (k) { return !finalEntityIds.has(k); });
        for (var hri = 0; hri < hopKeysToRemove.length; hri++) {
          delete hopMap[hopKeysToRemove[hri]];
        }

        if (!entities.find(function (e) { return e.absolute_id === absoluteId; })) {
          entities.unshift(entity);
        }

        // Fetch version counts
        var allEntityIds = [];
        var seenIds = new Set();
        for (var aei = 0; aei < entities.length; aei++) {
          if (!seenIds.has(entities[aei].entity_id)) {
            allEntityIds.push(entities[aei].entity_id);
            seenIds.add(entities[aei].entity_id);
          }
        }
        try {
          var vcRes = await state.api.entityVersionCounts(allEntityIds, graphId);
          _versionCounts = vcRes.data || {};
        } catch (_) {}

        // Don't clear pinned positions on version switch
        if (!isVersionSwitch) {
          _pinnedNodePositions = {};
        }
        buildGraph(entities, relations, absoluteId, hopMap, inheritedRelationIds, futureRelationIds);

        _focusAbsoluteId = absoluteId;
        var exitBtn = _el(_opts.exitFocusBtnId);
        if (exitBtn) exitBtn.style.display = '';
        var focusBadge = _el(_opts.focusBadgeId);
        if (focusBadge) focusBadge.style.display = '';

        // Callback after focus
        if (_opts.onAfterFocus) {
          _opts.onAfterFocus(entities);
        }
      } catch (err) {
        console.error('Focus failed:', err);
        showToast(t('graph.loadFailed') + ': ' + err.message, 'error');
      } finally {
        if (loadingEl) loadingEl.style.display = 'none';
      }
    }

    // ---- Exit focus mode, restore default view ----

    function exitFocus() {
      _focusAbsoluteId = null;
      _currentVersions = [];
      _currentVersionIdx = 0;
      _accumEntities = null;
      _accumRelationsByRid = null;
      _accumHopMap = null;
      _accumFocusEntityId = null;
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

      var entityId = entity.entity_id;

      var versions = [];
      try {
        var vRes = await state.api.entityVersions(entityId, state.currentGraphId);
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
            '<p style="font-size:0.8125rem;color:var(--text-secondary);line-height:1.5;word-break:break-word;white-space:pre-wrap;">' +
              escapeHtml(entity.content || '-') +
            '</p>' +
          '</div>' +

          '<div>' +
            '<span class="form-label" style="margin-bottom:0.125rem;">' + t('graph.entityId') + '</span>' +
            '<p class="mono truncate" style="color:var(--text-muted);font-size:0.75rem;" title="' + escapeHtml(entity.entity_id || '') + '">' +
              escapeHtml(entity.entity_id || '-') +
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

          (entity.memory_cache_id ?
            '<div>' +
              '<span class="form-label" style="margin-bottom:0.125rem;">' + t('graph.memoryCacheId') + '</span>' +
              '<span class="doc-link mono truncate" style="font-size:0.75rem;" data-view-doc="' + escapeHtml(entity.memory_cache_id) + '" title="' + t('common.clickToView') + '">' +
                escapeHtml(entity.memory_cache_id) +
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
          // Reset accumulation — scope change is a new focus context
          _accumEntities = null;
          _accumRelationsByRid = null;
          _accumHopMap = null;
          _accumFocusEntityId = null;
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
            '<p style="font-size:0.8125rem;color:var(--text-secondary);line-height:1.5;word-break:break-word;white-space:pre-wrap;">' +
              escapeHtml(relation.content || '-') +
            '</p>' +
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
              escapeHtml(relation.relation_id || '-') +
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
      var entityId = entity.entity_id || entity.absolute_id;
      var graphId = state.currentGraphId;

      var modal = showModal({
        title: t('graph.versionsTitle', { name: truncate(entity.name || entityId, 40) }),
        content: '<div class="flex justify-center p-6">' + spinnerHtml() + '</div>',
        size: 'lg',
      });

      try {
        var res = await state.api.entityVersions(entityId, graphId);
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
      var entityId = entity.entity_id || entity.absolute_id;
      var graphId = state.currentGraphId;

      var modal = showModal({
        title: t('graph.relationsTitle', { name: truncate(entity.name || entityId, 40) }),
        content: '<div class="flex justify-center p-6">' + spinnerHtml() + '</div>',
        size: 'lg',
      });

      try {
        var res = await state.api.entityRelations(entityId, graphId);
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
          var otherName = otherEntity ? (otherEntity.name || otherEntity.entity_id || '-') : '-';
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
      fetchMultiHop: fetchMultiHop,
      focusOnEntity: focusOnEntity,
      exitFocus: exitFocus,
      showEntityDetail: showEntityDetail,
      showRelationDetail: showRelationDetail,
      switchVersion: switchVersion,
      openVersionsModal: openVersionsModal,
      openRelationsModal: openRelationsModal,
      setEntityCache: function (cache) { _opts.entityCache = cache; },
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
        _accumEntities = null;
        _accumRelationsByRid = null;
        _accumHopMap = null;
        _accumFocusEntityId = null;
      },
    };
  }

  return { create: create };
})();

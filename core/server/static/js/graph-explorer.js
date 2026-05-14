/* ==========================================
   GraphExplorer — Shared graph visualization
   Used by graph.js, communities.js
   Factory pattern: GraphExplorer.create(options)

   Sub-modules (loaded before this file):
     graph-explorer/diff.js      → window.GraphExplorerDiff
     graph-explorer/versions.js  → window.GraphExplorerVersions
     graph-explorer/detail.js    → window.GraphExplorerDetail
     graph-explorer/focus.js     → window.GraphExplorerFocus
   ========================================== */

window.GraphExplorer = (function () {
  'use strict';

  function create(options) {
    // ---- Internal state ----
    var _network = null;
    var _nodesDataSet = null;
    var _edgesDataSet = null;
    var _entityMap = {};
    var _relationMap = {};
    var _versionCounts = {};
    var _pinnedNodePositions = {};
    var _spiralIdx = 0;
    var _streamingMode = false;  // skip expensive ops during bulk loading

    var _focusAbsoluteId = null;
    var _detailBackStack = [];  // stack of {type:'relation'|'entity', id:string} for back navigation
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

    function buildGraph(entities, relations, highlightAbsId, hopMap, inheritedRelationIds, futureRelationIds, hubLayout, savedPositions) {
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
          if (types.has('current')) {
            // default blue
          } else if (types.has('inherited')) {
            inheritedEntityIds.add(teid);
          } else if (types.has('future')) {
            futureEntityIds.add(teid);
          }
        }
      }

      // Propagate entity types from hop 1 to hop 2+ via cascading parent chain
      if (hopMap && (inheritedEntityIds.size > 0 || futureEntityIds.size > 0)) {
        var entityClassMap = {};
        inheritedEntityIds.forEach(function (id) { entityClassMap[id] = 'inherited'; });
        futureEntityIds.forEach(function (id) { entityClassMap[id] = 'future'; });

        var maxHop = 0;
        for (var mhid in hopMap) {
          if (hopMap[mhid] > maxHop) maxHop = hopMap[mhid];
        }

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

            if (!inheritedRelationIds.has(pr.absolute_id) && !futureRelationIds.has(pr.absolute_id)) {
              if (ptype === 'inherited') inheritedRelationIds.add(pr.absolute_id);
              else if (ptype === 'future') futureRelationIds.add(pr.absolute_id);
            }

            if (!childTypes[child]) childTypes[child] = new Set();
            childTypes[child].add(ptype);
          }

          var childKeys = Object.keys(childTypes);
          for (var ci = 0; ci < childKeys.length; ci++) {
            var ceid = childKeys[ci];
            if (entityClassMap[ceid]) continue;
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

      // Compute relation count per entity for node sizing
      var relationCounts = {};
      for (var ri2 = 0; ri2 < relations.length; ri2++) {
        var r2 = relations[ri2];
        relationCounts[r2.entity1_absolute_id] = (relationCounts[r2.entity1_absolute_id] || 0) + 1;
        relationCounts[r2.entity2_absolute_id] = (relationCounts[r2.entity2_absolute_id] || 0) + 1;
      }
      buildNodesOpts.relationCounts = relationCounts;

      var result = GraphUtils.buildNodes(entities, buildNodesOpts);
      var nodes = result.nodes;
      var eMap = result.entityMap;
      var nodeIds = result.nodeIds;

      var visibleNodeIds = new Set();
      var restorePositions = savedPositions || _pinnedNodePositions;
      var hasRestoredPositions = false;
      nodes.forEach(function (node) {
        visibleNodeIds.add(node.id);
        var pinned = restorePositions[node.id];
        if (pinned) {
          nodes.update({
            id: node.id,
            x: pinned.x,
            y: pinned.y,
            fixed: { x: true, y: true },
          });
          hasRestoredPositions = true;
        }
      });
      if (savedPositions) {
        Object.keys(savedPositions).forEach(function (k) {
          if (visibleNodeIds.has(k)) _pinnedNodePositions[k] = savedPositions[k];
        });
      }
      var pinnedKeys = Object.keys(_pinnedNodePositions);
      for (var pi2 = 0; pi2 < pinnedKeys.length; pi2++) {
        if (!visibleNodeIds.has(pinnedKeys[pi2])) delete _pinnedNodePositions[pinnedKeys[pi2]];
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

      var skipPhysics = hasRestoredPositions && (Object.keys(restorePositions).length >= visibleNodeIds.size * 0.7);

      var visOpts;
      if (skipPhysics) {
        visOpts = {
          physics: { enabled: false },
          interaction: GraphUtils.getInteractionOptions(),
          layout: { improvedLayout: false },
        };
      } else {
        visOpts = {
          physics: GraphUtils.getPhysicsOptions(),
          interaction: GraphUtils.getInteractionOptions(),
          layout: { improvedLayout: true },
        };
      }

      _network = new vis.Network(container, { nodes: nodes, edges: edges }, visOpts);

      // Invalidate hover panel
      _hoverPanel = null;

      if (skipPhysics) {
        renderVersionBadges(nodes);
      } else {
        _network.once('stabilizationIterationsDone', function () {
          _network.setOptions({ physics: { enabled: false } });
          if (highlightAbsId) {
            _network.focus(highlightAbsId, { scale: 1.2, animation: { duration: 500, easingFunction: 'easeInOutQuad' } });
          }
          renderVersionBadges(nodes);
        });
      }

      _network.on('click', function (params) {
        var nodeId = params.nodes[0];
        var edgeId = params.edges[0];
        if (nodeId) {
          hideNodeHover();
          showEntityDetail(nodeId);
        } else if (edgeId) {
          showRelationDetail(edgeId);
        }
      });

      _network.on('hoverNode', function (params) {
        showNodeHover(params.node, params);
      });
      _network.on('hoverEdge', function (params) {
        showEdgeHover(params.edge, params);
      });
      _network.on('blurNode', function () {
        hideNodeHover();
      });
      _network.on('blurEdge', function () {
        hideNodeHover();
      });

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

      _network.on('zoom', function () { updateBadgePositions(); updateNodeHoverPosition(); });
      _network.on('dragEnd', function () {
        setTimeout(updateBadgePositions, 50);
        updateNodeHoverPosition();
      });
      _network.on('viewChanged', function () { updateBadgePositions(); updateNodeHoverPosition(); });
    }

    // ---- Incremental graph building for smooth animations ----

    function initEmptyGraph(hubLayout) {
      _entityMap = {};
      _relationMap = {};
      _spiralIdx = 0;

      var container = _el(_opts.canvasId);
      if (!container) return;

      if (_network) {
        _network.destroy();
        _network = null;
      }

      _nodesDataSet = new vis.DataSet([]);
      _edgesDataSet = new vis.DataSet([]);

      var visOpts = {
        physics: {
          enabled: true,
          solver: 'forceAtlas2Based',
          forceAtlas2Based: {
            gravitationalConstant: -80,
            centralGravity: 0.005,
            springLength: 120,
            springConstant: 0.04,
          },
          stabilization: { enabled: false },
        },
        interaction: GraphUtils.getInteractionOptions(),
        layout: { improvedLayout: true },
      };

      _network = new vis.Network(container, { nodes: _nodesDataSet, edges: _edgesDataSet }, visOpts);

      _hoverPanel = null;

      _network.on('click', function (params) {
        var nodeId = params.nodes[0];
        var edgeId = params.edges[0];
        if (nodeId) {
          hideNodeHover();
          showEntityDetail(nodeId);
        } else if (edgeId) {
          showRelationDetail(edgeId);
        }
      });

      _network.on('hoverNode', function (params) {
        showNodeHover(params.node, params);
      });
      _network.on('hoverEdge', function (params) {
        showEdgeHover(params.edge, params);
      });
      _network.on('blurNode', function () { hideNodeHover(); });
      _network.on('blurEdge', function () { hideNodeHover(); });

      _network.on('dragStart', function (params) {
        if (params.nodes.length === 0) return;
        params.nodes.forEach(function (nodeId) {
          _nodesDataSet.update({ id: nodeId, fixed: false });
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
          _nodesDataSet.update({
            id: nodeId,
            x: pos.x,
            y: pos.y,
            fixed: { x: true, y: true },
          });
        });
        _network.setOptions({ physics: { enabled: false } });
      });

      _network.on('zoom', function () { updateBadgePositions(); updateNodeHoverPosition(); });
      _network.on('dragEnd', function () { setTimeout(updateBadgePositions, 50); updateNodeHoverPosition(); });
      _network.on('viewChanged', function () { updateBadgePositions(); updateNodeHoverPosition(); });
    }

    // Incrementally add entities and relations to the existing network

    function addNodesAndEdges(newEntities, newRelations, hubLayout, totalNodeEstimate) {
      if (!_network || !_nodesDataSet) return;

      var colorMode = hubLayout && hubLayout.hubMap ? 'hub' : 'default';
      var buildNodesOpts = {
        colorMode: colorMode,
        versionCounts: _versionCounts,
        unnamedLabel: t('graph.unnamedEntity'),
      };
      if (hubLayout) {
        buildNodesOpts.hubMap = hubLayout.hubMap;
        buildNodesOpts.hubNeighborIds = hubLayout.hubNeighborIds;
      }

      var allEdges = _edgesDataSet.get();
      var relationCounts = {};
      for (var ri = 0; ri < allEdges.length; ri++) {
        var re = allEdges[ri];
        relationCounts[re.from] = (relationCounts[re.from] || 0) + 1;
        relationCounts[re.to] = (relationCounts[re.to] || 0) + 1;
      }
      for (var ri2 = 0; ri2 < newRelations.length; ri2++) {
        var nr = newRelations[ri2];
        var nFrom = nr.entity1_family_id || nr.entity1_absolute_id;
        var nTo = nr.entity2_family_id || nr.entity2_absolute_id;
        relationCounts[nFrom] = (relationCounts[nFrom] || 0) + 1;
        relationCounts[nTo] = (relationCounts[nTo] || 0) + 1;
      }
      buildNodesOpts.relationCounts = relationCounts;

      var result = GraphUtils.buildNodes(newEntities, buildNodesOpts);
      var newNodes = result.nodes.get();

      for (var ek in result.entityMap) {
        _entityMap[ek] = result.entityMap[ek];
      }

      var existingPositions = _network.getPositions();
      var existingNodeIds = new Set(_nodesDataSet.getIds());

      var newNodeConnections = {};
      for (var ni = 0; ni < newNodes.length; ni++) {
        newNodeConnections[newNodes[ni].id] = [];
      }
      for (var ri3 = 0; ri3 < newRelations.length; ri3++) {
        var rel = newRelations[ri3];
        var e1Id = rel.entity1_family_id || rel.entity1_absolute_id;
        var e2Id = rel.entity2_family_id || rel.entity2_absolute_id;
        if (!existingNodeIds.has(e1Id) && existingNodeIds.has(e2Id)) {
          if (newNodeConnections[e1Id]) newNodeConnections[e1Id].push(e2Id);
        } else if (existingNodeIds.has(e1Id) && !existingNodeIds.has(e2Id)) {
          if (newNodeConnections[e2Id]) newNodeConnections[e2Id].push(e1Id);
        }
      }

      var canvasEl = document.getElementById(_opts.canvasId);
      var cx = canvasEl ? canvasEl.offsetWidth / 2 : 400;
      var cy = canvasEl ? canvasEl.offsetHeight / 2 : 300;
      var maxRadius = Math.min(cx, cy) * 0.85;
      var goldenAngle = Math.PI * (3 - Math.sqrt(5));

      for (var ni2 = 0; ni2 < newNodes.length; ni2++) {
        var node = newNodes[ni2];
        var connections = newNodeConnections[node.id];
        if (connections && connections.length > 0) {
          var sumX = 0, sumY = 0, count = 0;
          for (var ci = 0; ci < connections.length; ci++) {
            var cpos = existingPositions[connections[ci]];
            if (cpos) { sumX += cpos.x; sumY += cpos.y; count++; }
          }
          if (count > 0) {
            node.x = sumX / count + (Math.random() - 0.5) * 60;
            node.y = sumY / count + (Math.random() - 0.5) * 60;
          } else {
            node.x = cx + (Math.random() - 0.5) * 200;
            node.y = cy + (Math.random() - 0.5) * 200;
          }
        } else {
          var idx = _spiralIdx++;
          var totalEst = totalNodeEstimate || 200;
          var r = maxRadius * Math.sqrt((idx + 0.5) / totalEst);
          var theta = idx * goldenAngle;
          node.x = cx + r * Math.cos(theta);
          node.y = cy + r * Math.sin(theta);
        }
      }

      _nodesDataSet.add(newNodes);

      if (newNodes.length > 0 && newNodes.length <= 50) {
        var newNodeIds = newNodes.map(function(n) { return n.id; });
        _network.selectNodes(newNodeIds);
        setTimeout(function() {
          if (_network) _network.unselectAll();
        }, 300);
      }

      var allNodeIds = new Set(_nodesDataSet.getIds());
      var buildEdgesOpts = {};
      if (_opts.relationStrengthEnabled) buildEdgesOpts.weightMode = 'count';
      if (hubLayout) buildEdgesOpts.hubMap = hubLayout.hubMap;

      var edgeResult = GraphUtils.buildEdges(newRelations, allNodeIds, buildEdgesOpts);
      var newEdges = edgeResult.edges.get();

      for (var rk in edgeResult.relationMap) {
        _relationMap[rk] = edgeResult.relationMap[rk];
      }

      _edgesDataSet.add(newEdges);

      if (!_streamingMode) {
        _recalcNodeSizes();
        try { _network.fit({ animation: false }); } catch (_) {}
      }
    }

    /**
     * Incrementally remove nodes and edges from the current network.
     */
    function removeNodesAndEdges(nodeAbsIds, edgeAbsIds) {
      if (!_network || !_nodesDataSet) return;

      if (edgeAbsIds && edgeAbsIds.length > 0) {
        _edgesDataSet.remove(edgeAbsIds);
        for (var i = 0; i < edgeAbsIds.length; i++) {
          delete _relationMap[edgeAbsIds[i]];
        }
      }

      if (nodeAbsIds && nodeAbsIds.length > 0) {
        _nodesDataSet.remove(nodeAbsIds);
        for (var j = 0; j < nodeAbsIds.length; j++) {
          delete _entityMap[nodeAbsIds[j]];
        }
      }

      renderVersionBadges(_nodesDataSet);

      // Recalculate node sizes after edge removal
      var remainEdges = _edgesDataSet.get();
      var remainCounts = {};
      for (var rci = 0; rci < remainEdges.length; rci++) {
        var re2 = remainEdges[rci];
        remainCounts[re2.from] = (remainCounts[re2.from] || 0) + 1;
        remainCounts[re2.to] = (remainCounts[re2.to] || 0) + 1;
      }
      var remainMax = 1;
      var remainKeys = Object.keys(remainCounts);
      for (var rki = 0; rki < remainKeys.length; rki++) {
        if (remainCounts[remainKeys[rki]] > remainMax) remainMax = remainCounts[remainKeys[rki]];
      }
      var remainNodes = _nodesDataSet.get();
      var remainUpdates = [];
      for (var rni = 0; rni < remainNodes.length; rni++) {
        var rn = remainNodes[rni];
        var rc = remainCounts[rn.id] || 0;
        var rs = GraphUtils.computeNodeSize(rc, remainMax);
        if (Math.abs((rn.size || 0) - rs) > 0.5) {
          remainUpdates.push({ id: rn.id, size: rs });
        }
      }
      if (remainUpdates.length > 0) {
        _nodesDataSet.update(remainUpdates);
      }

      try { _network.fit({ animation: false }); } catch (_) {}
    }

    // ---- Focus: delegates to GraphExplorerFocus ----

    async function focusOnEntity(absoluteId, opts) {
      opts = opts || {};
      var isVersionSwitch = opts.isVersionSwitch && _session && _session.focusFamilyId;
      var graphId = state.currentGraphId;
      var loadingEl = _el(_opts.loadingId);
      if (loadingEl && !isVersionSwitch) loadingEl.style.display = 'flex';

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

        // 2. BFS topology
        var bfs = GraphExplorerFocus.focusBFS(familyId, hopLevel, _opts.entityCache || {}, _mainViewRelations || []);

        // 2b. Fetch ALL relations for the focused entity from API
        var _apiRelationAbsIds = new Set();
        if (!isVersionSwitch) {
          try {
            var scope = _relationScope || 'accumulated';
            var apiRelRes = await state.api.entityRelations(familyId, graphId, {
              maxVersionAbsoluteId: absoluteId,
              relationScope: scope
            });
            var apiRels = apiRelRes.data?.relations || apiRelRes.data || [];
            var existingRelAbsIds = new Set();
            bfs.relations.forEach(function(r) { existingRelAbsIds.add(r.absolute_id); });
            for (var ari = 0; ari < apiRels.length; ari++) {
              var apiRel = apiRels[ari];
              if (!existingRelAbsIds.has(apiRel.absolute_id)) {
                bfs.relations.push(apiRel);
                existingRelAbsIds.add(apiRel.absolute_id);
                _apiRelationAbsIds.add(apiRel.entity1_absolute_id);
                _apiRelationAbsIds.add(apiRel.entity2_absolute_id);
              }
            }
          } catch (_) {}
        }

        // 3. Build familyIdToLatest with focus override
        var familyIdToLatest = _opts.familyIdToLatest ? Object.assign({}, _opts.familyIdToLatest()) : {};
        familyIdToLatest[familyId] = absoluteId;

        // 4. Resolve missing absolute_ids
        var absToFid = Object.assign({}, bfs.absToFid);
        if (!isVersionSwitch) {
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
        }

        // 4b. Add family_ids discovered from API relations
        _apiRelationAbsIds.forEach(function(apiAbsId) {
          var fid = absToFid[apiAbsId];
          if (fid && !bfs.familyIds.has(fid)) {
            bfs.familyIds.add(fid);
            bfs.hopMapFid[fid] = 1;
          }
        });

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

        // 7. Convert hopMap: family_id -> absolute_id keys
        var hopMap = {};
        var fids = Object.keys(bfs.hopMapFid);
        for (var hi = 0; hi < fids.length; hi++) {
          var hAbsId = familyIdToLatest[fids[hi]];
          if (hAbsId) hopMap[hAbsId] = bfs.hopMapFid[fids[hi]];
        }

        // 8. API classification
        var inheritedRelationIds = new Set();
        var futureRelationIds = new Set();
        var scope = _relationScope;
        var latestAbsForFocus = _opts.familyIdToLatest ? _opts.familyIdToLatest()[familyId] : null;
        var isLatestVersion = (absoluteId === latestAbsForFocus);

        if (!isVersionSwitch && scope !== 'version_only' && !(isLatestVersion && scope === 'accumulated')) {
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

        // 9. Session merge
        var doSessionMerge = _session && _session.focusFamilyId === familyId;
        if (!_session) _session = new GraphExplorerFocus.FocusSession();
        if (doSessionMerge) {
          var sessionInherited = _session.merge(familyId, relAbsIdSet);
          sessionInherited.forEach(function (id) { inheritedRelationIds.add(id); });
        }

        // 10. Filter: only keep connected entities
        var connAbsIds = new Set();
        for (var ci = 0; ci < relations.length; ci++) {
          connAbsIds.add(relations[ci].entity1_absolute_id);
          connAbsIds.add(relations[ci].entity2_absolute_id);
        }
        entities = entities.filter(function (e) { return connAbsIds.has(e.absolute_id); });

        // 11. Fetch version counts
        if (!isVersionSwitch) {
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
        }

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

    // ---- Hover panel: delegates to GraphExplorerDetail ----

    var _hoverPanel = null;
    var _hoverNodeId = null;

    function showNodeHover(nodeId, params) {
      _hoverPanel = GraphExplorerDetail.showNodeHover({
        nodeId: nodeId,
        network: _network,
        entityMap: _entityMap,
        versionCounts: _versionCounts,
        canvasId: _opts.canvasId,
        hoverPanel: _hoverPanel,
        setHoverPanel: function(p) { _hoverPanel = p; },
        setHoverNodeId: function(id) { _hoverNodeId = id; },
      });
    }

    function hideNodeHover() {
      GraphExplorerDetail.hideNodeHover(_hoverPanel, function(id) { _hoverNodeId = id; });
    }

    function updateNodeHoverPosition() {
      GraphExplorerDetail.updateNodeHoverPosition({
        hoverPanel: _hoverPanel,
        hoverNodeId: _hoverNodeId,
        network: _network,
        canvasId: _opts.canvasId,
      });
    }

    function showEdgeHover(edgeId, params) {
      _hoverPanel = GraphExplorerDetail.showEdgeHover({
        edgeId: edgeId,
        network: _network,
        relationMap: _relationMap,
        entityMap: _entityMap,
        canvasId: _opts.canvasId,
        hoverPanel: _hoverPanel,
        setHoverPanel: function(p) { _hoverPanel = p; },
        setHoverNodeId: function(id) { _hoverNodeId = id; },
      });
    }

    // ---- Entity detail: uses GraphExplorerDiff + GraphExplorerVersions ----

    async function showEntityDetail(absoluteId) {
      var entity = _entityMap[absoluteId];
      if (!entity) {
        try {
          var res = await state.api.entityByAbsoluteId(absoluteId, state.currentGraphId);
          if (res.data) { entity = res.data; _entityMap[absoluteId] = entity; }
        } catch (_) {}
      }
      if (!entity) return;

      var _hasBack = _detailBackStack.length > 0;
      var _backTo = _hasBack ? _detailBackStack[_detailBackStack.length - 1] : null;
      if (_hasBack) {
        _detailBackStack.push({type: 'entity', id: absoluteId});
      } else {
        _detailBackStack = [{type: 'entity', id: absoluteId}];
      }

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

      var versionDiff = null;
      if (totalVersions > 1 && _currentVersionIdx > 0) {
        versionDiff = GraphExplorerDiff.computeInlineDiff(
          versions[_currentVersionIdx - 1].content,
          versions[_currentVersionIdx].content
        );
      }

      detailContent.innerHTML =
        '<div class="flex items-center justify-between mb-3">' +
          '<div class="flex items-center gap-2">' +
            (_hasBack ? '<button class="btn btn-secondary btn-sm" id="detail-back-btn" title="' + t('common.back') + '"><i data-lucide="arrow-left" style="width:14px;height:14px;"></i></button>' : '') +
            '<span class="badge badge-primary">' + t('graph.entityDetail') + '</span>' +
          '</div>' +
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

        (totalVersions > 1 ? GraphExplorerVersions.renderVersionEvolutionSummary(versions) : '') +
        (totalVersions > 1 ? GraphExplorerVersions.renderMiniVersionTimeline(versions, _currentVersionIdx, prefix) : '') +
        (versionDiff ? GraphExplorerDiff.renderDiffPreview(versionDiff) : '') +
        GraphExplorerVersions.renderVersionContext(versions, _currentVersionIdx) +

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

        (totalVersions > 1 ? GraphExplorerVersions.renderKeyboardHints() : '') +

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
              formatDateMs(entity.processed_time) +
            '</p>' +
          '</div>' +

          (entity.source_document ?
            '<div>' +
              '<span class="form-label" style="margin-bottom:0.125rem;">' + t('graph.sourceDoc') + '</span>' +
              '<span class="mono truncate" style="font-size:0.75rem;color:var(--text-secondary);">' +
                escapeHtml(truncate(entity.source_document, 60)) +
              '</span>' +
            '</div>'
          : '') +

          (entity.episode_id ?
            '<div>' +
              '<span class="form-label" style="margin-bottom:0.125rem;">' + t('graph.episodeId') + '</span>' +
              '<span class="doc-link mono truncate" style="font-size:0.75rem;" data-view-episode="' + escapeHtml(entity.episode_id) + '" title="' + t('common.clickToView') + '">' +
                escapeHtml(entity.episode_id) +
              '</span>' +
            '</div>'
          : '') +
        '</div>';

      if (window.lucide) lucide.createIcons({ nodes: [detailContent] });

      detailContent.querySelectorAll('[data-view-doc]').forEach(function (el) {
        el.addEventListener('click', function () { window.showDocContent(el.getAttribute('data-view-doc')); });
      });
      detailContent.querySelectorAll('[data-view-episode]').forEach(function (el) {
        el.addEventListener('click', function () { window.showEpisodeDoc(el.getAttribute('data-view-episode')); });
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

      detailContent.querySelectorAll('.version-mini-dot').forEach(function (dot) {
        dot.addEventListener('click', function () {
          var idx = parseInt(dot.getAttribute('data-ver-idx'), 10);
          if (!isNaN(idx) && idx !== _currentVersionIdx) switchVersion(idx);
        });
      });

      var scopeSel = _el(prefix + 'relation-scope-sel');
      if (scopeSel) {
        scopeSel.addEventListener('change', function () {
          _relationScope = scopeSel.value;
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

      var backBtn = document.getElementById('detail-back-btn');
      if (backBtn && _backTo) {
        backBtn.addEventListener('click', function () {
          while (_detailBackStack.length > 0) {
            var top = _detailBackStack.pop();
            if (top.type === _backTo.type && top.id === _backTo.id) break;
          }
          if (_backTo.type === 'relation') {
            showRelationDetail(_backTo.id);
          } else if (_backTo.type === 'entity') {
            showEntityDetail(_backTo.id);
          }
        });
      }
    }

    // ---- Show relation detail ----

    function showRelationDetail(absoluteId) {
      var relation = _relationMap[absoluteId];
      if (!relation) return;

      _detailBackStack = [];
      _detailBackStack.push({type: 'relation', id: absoluteId});

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
              formatDateMs(relation.processed_time) +
            '</p>' +
          '</div>' +

          (relation.source_document ?
            '<div>' +
              '<span class="form-label" style="margin-bottom:0.125rem;">' + t('graph.sourceDoc') + '</span>' +
              '<span class="mono truncate" style="font-size:0.75rem;color:var(--text-secondary);">' +
                escapeHtml(truncate(relation.source_document, 60)) +
              '</span>' +
            '</div>'
          : '') +

          (relation.episode_id ?
            '<div>' +
              '<span class="form-label" style="margin-bottom:0.125rem;">' + t('graph.episodeId') + '</span>' +
              '<span class="doc-link mono truncate" style="font-size:0.75rem;" data-view-episode="' + escapeHtml(relation.episode_id) + '" title="' + t('common.clickToView') + '">' +
                escapeHtml(relation.episode_id) +
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
      detailContent.querySelectorAll('[data-view-episode]').forEach(function (el) {
        el.addEventListener('click', function () { window.showEpisodeDoc(el.getAttribute('data-view-episode')); });
      });

      if (window.lucide) lucide.createIcons({ nodes: [detailContent] });
    }

    // ---- Switch to a different version ----

    async function switchVersion(newIdx) {
      if (!_currentVersions[newIdx]) return;
      _currentVersionIdx = newIdx;

      var version = _currentVersions[newIdx];
      var absoluteId = version.absolute_id;

      if (!_entityMap[absoluteId]) {
        _entityMap[absoluteId] = version;
      }

      // Fast path: update node label/color in-place via DataSet, skip full rebuild
      if (_nodesDataSet && _network && _focusAbsoluteId) {
        var oldFocusId = _focusAbsoluteId;
        _focusAbsoluteId = absoluteId;
        var familyId = version.family_id;
        var vc = _versionCounts[familyId] || _currentVersions.length;
        var focusColors = GraphUtils.FOCUS_NODE;
        var baseName = version.name || familyId || 'unnamed';
        var newLabel = baseName + ' ' + (newIdx + 1) + '/' + _currentVersions.length;

        if (oldFocusId !== absoluteId) {
          var oldEnt = _entityMap[oldFocusId];
          var oldBaseName = oldEnt ? (oldEnt.name || oldEnt.family_id || 'unnamed') : 'unnamed';
          var oldLabel = oldBaseName;
          if (vc > 1) oldLabel = oldBaseName + ' [v' + vc + ']';
          if (_nodesDataSet.get(oldFocusId)) {
            _nodesDataSet.update({
              id: oldFocusId,
              label: oldLabel,
              color: { background: undefined, border: undefined, highlight: undefined },
            });
          }
        }

        if (_nodesDataSet.get(absoluteId)) {
          _nodesDataSet.update({
            id: absoluteId,
            label: newLabel,
            color: {
              background: focusColors.bg,
              border: focusColors.border,
              highlight: { background: focusColors.bg, border: focusColors.border },
            },
          });
        } else {
          var oldPos = _pinnedNodePositions[oldFocusId] || _network.getPositions([oldFocusId])[oldFocusId];
          var nodeData = {
            id: absoluteId,
            label: newLabel,
            color: {
              background: focusColors.bg,
              border: focusColors.border,
              highlight: { background: focusColors.bg, border: focusColors.border },
            },
            shape: 'dot',
            size: 30,
          };
          if (oldPos) {
            nodeData.x = oldPos.x;
            nodeData.y = oldPos.y;
            nodeData.fixed = { x: true, y: true };
            _pinnedNodePositions[absoluteId] = { x: oldPos.x, y: oldPos.y };
          }
          _nodesDataSet.add(nodeData);
        }

        if (oldFocusId !== absoluteId) {
          var edges = _edgesDataSet.get();
          var edgeUpdates = [];
          for (var ei = 0; ei < edges.length; ei++) {
            var edge = edges[ei];
            var changed = false;
            if (edge.from === oldFocusId) { edge.from = absoluteId; changed = true; }
            if (edge.to === oldFocusId) { edge.to = absoluteId; changed = true; }
            if (changed) edgeUpdates.push(edge);
          }
          if (edgeUpdates.length > 0) _edgesDataSet.update(edgeUpdates);
        }

        try {
          _network.focus(absoluteId, { scale: 1.2, animation: { duration: 400, easingFunction: 'easeInOutQuad' } });
        } catch (_) {}

        await showEntityDetail(absoluteId);
        return;
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

        var sorted = versions.slice().sort(function(a, b) {
          var ta = a.processed_time ? new Date(a.processed_time).getTime() : 0;
          var tb = b.processed_time ? new Date(b.processed_time).getTime() : 0;
          return tb - ta;
        });

        function simpleContentDiff(current, previous) {
          if (!previous) return null;
          return GraphExplorerDiff.computeInlineDiff(previous.content, current.content);
        }

        var items = sorted.map(function(v, i) {
          var prev = sorted[i + 1];
          var diff = simpleContentDiff(v, prev);
          var hasNameChange = prev && v.name !== prev.name;
          var verNum = sorted.length - i;

          var gapHtml = '';
          if (prev) {
            var curTime = v.processed_time ? new Date(v.processed_time).getTime() : 0;
            var prevTime = prev.processed_time ? new Date(prev.processed_time).getTime() : 0;
            var gapMs = curTime - prevTime;
            var gapText = '';
            if (gapMs < 60000) gapText = '< 1m';
            else if (gapMs < 3600000) gapText = Math.round(gapMs / 60000) + 'm';
            else if (gapMs < 86400000) gapText = Math.round(gapMs / 3600000) + 'h';
            else gapText = Math.round(gapMs / 86400000) + 'd';
            gapHtml = '<span style="position:absolute;left:-1.25rem;top:50%;transform:translateY(-50%);font-size:0.625rem;color:var(--text-muted);background:var(--bg-surface);padding:0 0.25rem;white-space:nowrap;z-index:1;">' + gapText + '</span>';
          }

          var sourceLabel = v.source_document || '';
          if (sourceLabel.length > 25) sourceLabel = sourceLabel.substring(0, 22) + '...';

          var episodeHtml = '';
          if (v.episode_id) {
            var shortEpId = v.episode_id.length > 16 ? v.episode_id.substring(0, 8) + '...' + v.episode_id.substring(v.episode_id.length - 4) : v.episode_id;
            episodeHtml = '<span class="doc-link mono" style="font-size:0.6875rem;color:var(--text-muted);cursor:pointer;text-decoration:underline dotted;" data-view-episode="' + escapeHtml(v.episode_id) + '" title="' + escapeHtml(v.episode_id) + '">' + escapeHtml(shortEpId) + '</span>';
          }

          var headerHtml = '<div style="display:flex;align-items:center;gap:0.5rem;flex-wrap:wrap;">'
            + '<span style="display:inline-flex;align-items:center;justify-content:center;width:1.375rem;height:1.375rem;border-radius:50%;font-size:0.6875rem;font-weight:600;font-family:var(--font-mono);background:color-mix(in srgb, #f59e0b 15%, transparent);color:#f59e0b;">v' + verNum + '</span>'
            + '<span class="mono" style="font-size:0.75rem;color:var(--text-muted);">' + formatDate(v.event_time) + '</span>'
            + (i === 0 ? '<span class="badge badge-info" style="font-size:0.6875rem;">' + t('entities.latest') + '</span>' : '')
            + (diff || hasNameChange ? '<span class="badge badge-primary" style="font-size:0.6875rem;">' + t('entities.changed') + '</span>' : '')
            + '</div>'
            + '<div style="margin-top:0.25rem;font-weight:500;font-size:0.875rem;">' + escapeHtml(v.name || '-') + '</div>'
            + '<div style="margin-top:0.125rem;color:var(--text-secondary);font-size:0.8125rem;" class="truncate">' + escapeHtml(truncate(v.content || '', 100)) + '</div>'
            + '<div style="margin-top:0.25rem;display:flex;align-items:center;gap:0.75rem;flex-wrap:wrap;">'
            + (sourceLabel ? '<span style="font-size:0.6875rem;color:var(--text-muted);">source: ' + escapeHtml(sourceLabel) + '</span>' : '')
            + (episodeHtml ? '<span style="font-size:0.6875rem;color:var(--text-muted);">episode: ' + episodeHtml + '</span>' : '')
            + '</div>';

          if (diff) {
            headerHtml += '<div style="margin-top:0.5rem;border-left:3px solid var(--primary);padding:0.375rem 0.5rem;background:var(--bg-input);border-radius:0 0.375rem 0.375rem 0;font-size:0.8125rem;white-space:pre-wrap;word-break:break-all;line-height:1.6;">';
            headerHtml += GraphExplorerDiff._renderDiffSpans(diff);
            headerHtml += '</div>';
          }

          var bodyHtml = '<div class="md-content" style="background:var(--bg-input);border:1px solid var(--border-color);border-radius:0.375rem;padding:0.75rem;">'
            + renderMarkdown(v.content || '')
            + '</div>';

          var toggleId = 'graph-version-toggle-' + i;
          var expandedId = 'graph-version-expanded-' + i;

          return '<div style="position:relative;padding-left:1.5rem;padding-bottom:' + (i < sorted.length - 1 ? '1.5rem' : '0') + ';">'
            + (i < sorted.length - 1 ? '<div style="position:absolute;left:5px;top:12px;bottom:0;width:1px;background:var(--border-color);"></div>' : '')
            + '<div style="position:absolute;left:0;top:4px;width:11px;height:11px;border-radius:50%;background:' + (diff || hasNameChange ? 'var(--primary)' : 'var(--border-color)') + ';border:2px solid ' + (diff || hasNameChange ? 'var(--primary-hover)' : 'var(--border-hover)') + ';"></div>'
            + gapHtml
            + '<div style="cursor:pointer;" id="' + toggleId + '">'
            + headerHtml
            + '</div>'
            + '<div id="' + expandedId + '" style="display:none;margin-top:0.5rem;">'
            + bodyHtml
            + '</div>'
            + '</div>';
        }).join('');

        modal.overlay.querySelector('.modal-body').innerHTML =
          '<div style="max-height:60vh;overflow-y:auto;padding:0.5rem;" data-family-id="' + escapeHtml(familyId) + '">'
          + items
          + '</div>'
          + '<p style="margin-top:0.75rem;font-size:0.75rem;color:var(--text-muted);">' + t('graph.versionCount', { count: versions.length }) + '</p>';

        modal.overlay.querySelectorAll('[id^="graph-version-toggle-"]').forEach(function(toggle) {
          toggle.addEventListener('click', function() {
            var idx = toggle.id.replace('graph-version-toggle-', '');
            var expanded = modal.overlay.querySelector('#graph-version-expanded-' + idx);
            if (expanded) expanded.style.display = expanded.style.display === 'none' ? 'block' : 'none';
          });
        });

        modal.overlay.querySelectorAll('[data-view-episode]').forEach(function(el) {
          el.addEventListener('click', function(e) {
            e.stopPropagation();
            window.showEpisodeDoc(el.getAttribute('data-view-episode'));
          });
        });

        if (window.lucide) lucide.createIcons({ nodes: [modal.overlay] });
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
          var otherName = r.entity1_absolute_id === entity.absolute_id
            ? (r.entity2_name || (otherEntity ? otherEntity.name : '') || otherAbsId || '-')
            : (r.entity1_name || (otherEntity ? otherEntity.name : '') || otherAbsId || '-');
          return '<tr>' +
            '<td style="max-width:250px;" title="' + escapeHtml(r.content || '') + '">' + escapeHtml(truncate(r.content || '-', 40)) + '</td>' +
            '<td style="max-width:120px;" title="' + escapeHtml(otherName) + '">' + escapeHtml(truncate(otherName, 20)) + '</td>' +
            '<td class="mono" style="white-space:nowrap;font-size:0.75rem;color:var(--text-muted);">' + formatDate(r.event_time) + '</td>' +
            '<td class="mono" style="white-space:nowrap;font-size:0.75rem;color:var(--text-muted);">' + formatDateMs(r.processed_time) + '</td>' +
          '</tr>';
        }).join('');

        modal.overlay.querySelector('.modal-body').innerHTML =
          '<div class="table-container" style="max-height:50vh;overflow-y:auto;">' +
            '<table class="data-table">' +
              '<thead><tr>' +
                '<th>' + t('graph.content') + '</th>' +
                '<th>' + t('graph.toEntity') + '</th>' +
                '<th>' + t('graph.eventTime') + '</th>' +
                '<th>' + t('graph.processedTime') + '</th>' +
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

    // ---- Internal helpers ----

    function _recalcNodeSizes() {
      var allUpdatedEdges = _edgesDataSet.get();
      var liveCounts = {};
      for (var i = 0; i < allUpdatedEdges.length; i++) {
        var e = allUpdatedEdges[i];
        liveCounts[e.from] = (liveCounts[e.from] || 0) + 1;
        liveCounts[e.to] = (liveCounts[e.to] || 0) + 1;
      }
      var liveMax = 1;
      var keys = Object.keys(liveCounts);
      for (var k = 0; k < keys.length; k++) {
        if (liveCounts[keys[k]] > liveMax) liveMax = liveCounts[keys[k]];
      }
      var allNodes = _nodesDataSet.get();
      var sizeUpdates = [];
      for (var n = 0; n < allNodes.length; n++) {
        var node = allNodes[n];
        var cnt = liveCounts[node.id] || 0;
        var newSize = GraphUtils.computeNodeSize(cnt, liveMax);
        if (Math.abs((node.size || 0) - newSize) > 0.5) {
          sizeUpdates.push({ id: node.id, size: newSize });
        }
      }
      if (sizeUpdates.length > 0) _nodesDataSet.update(sizeUpdates);
      renderVersionBadges(_nodesDataSet);
    }

    function renderVersionBadges(nodesDataSet) {
      if (!_network) return;
      var container = _el(_opts.canvasId);
      if (!container) return;

      var oldBadges = container.querySelectorAll('.version-badge-overlay');
      for (var oi = 0; oi < oldBadges.length; oi++) oldBadges[oi].remove();

      var oldWatermark = container.querySelector('.graph-deepdream-watermark');
      if (!oldWatermark) {
        var wm = document.createElement('div');
        wm.className = 'graph-deepdream-watermark';
        wm.textContent = 'Deep-Dream';
        container.appendChild(wm);
      }
    }

    function updateBadgePositions() {
      // No-op: version info is in node border styling
    }

    // ---- Public API ----

    return {
      buildGraph: buildGraph,
      initEmptyGraph: initEmptyGraph,
      addNodesAndEdges: addNodesAndEdges,
      removeNodesAndEdges: removeNodesAndEdges,
      setPhysics: function (enabled) {
        if (_network) {
          _network.setOptions({ physics: { enabled: !!enabled } });
        }
      },
      focusOnEntity: focusOnEntity,
      exitFocus: exitFocus,
      showEntityDetail: showEntityDetail,
      showRelationDetail: showRelationDetail,
      switchVersion: switchVersion,
      openVersionsModal: openVersionsModal,
      openRelationsModal: openRelationsModal,
      setEntityCache: function (cache) { _opts.entityCache = cache; },
      setStreamingMode: function (enabled) {
        _streamingMode = !!enabled;
        if (_network) {
          if (enabled) {
            _network.setOptions({
              physics: {
                enabled: true,
                solver: 'forceAtlas2Based',
                forceAtlas2Based: {
                  gravitationalConstant: -50,
                  centralGravity: 0.01,
                  springLength: 50,
                  springConstant: 0.15,
                },
                stabilization: { enabled: false },
              }
            });
          } else {
            var _po = GraphUtils.getPhysicsOptions();
            _po.stabilization = { enabled: false };
            _network.setOptions({ physics: _po });
          }
        }
      },
      fitViewport: function () {
        if (_network) { try { _network.fit({ animation: false }); } catch (_) {} }
      },
      recalcGraphState: function () {
        if (!_network || !_nodesDataSet) return;
        _recalcNodeSizes();
        try { _network.fit({ animation: true }); } catch (_) {}
      },
      applyHubColors: function (hubLayout) {
        if (!_network || !_nodesDataSet || !hubLayout) return;
        var light = isLightTheme();
        var hubMap = hubLayout.hubMap;
        var hubNeighborIds = hubLayout.hubNeighborIds;
        var allNodes = _nodesDataSet.get();
        var colorUpdates = [];
        for (var i = 0; i < allNodes.length; i++) {
          var node = allNodes[i];
          var hubIdx = hubMap[node.id];
          var isNeighbor = hubNeighborIds && hubNeighborIds.has(node.id);
          var bgColor, borderColor;
          if (hubIdx !== undefined) {
            if (isNeighbor) {
              var scheme = GraphUtils.HUB_NEIGHBOR_PALETTE[hubIdx];
              bgColor = scheme.bg; borderColor = scheme.border;
            } else {
              var scheme = GraphUtils.HUB_PALETTE[hubIdx];
              bgColor = scheme.bg; borderColor = scheme.border;
            }
          } else {
            var dc = light ? GraphUtils.DEFAULT_LIGHT : GraphUtils.DEFAULT_DARK;
            bgColor = dc.bg; borderColor = dc.border;
          }
          colorUpdates.push({
            id: node.id,
            color: {
              background: bgColor,
              border: borderColor,
              highlight: { background: bgColor, border: borderColor },
            }
          });
        }
        if (colorUpdates.length > 0) _nodesDataSet.update(colorUpdates);

        var allEdges = _edgesDataSet.get();
        var edgeUpdates = [];
        for (var ei = 0; ei < allEdges.length; ei++) {
          var edge = allEdges[ei];
          var h1 = hubMap[edge.from];
          var h2 = hubMap[edge.to];
          if (h1 !== undefined && h2 !== undefined && h1 === h2) {
            edgeUpdates.push({ id: edge.id, color: GraphUtils.HUB_EDGE_PALETTE[h1] });
          }
        }
        if (edgeUpdates.length > 0) _edgesDataSet.update(edgeUpdates);
      },
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
      updateBadgePositions: updateBadgePositions,
      renderVersionBadges: function () {
        if (_network) {
          var container = _el(_opts.canvasId);
          if (container) {
            var nodes = _network.body.data.nodes;
            renderVersionBadges(nodes);
          }
        }
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
      getCurrentNodes: function () {
        if (!_network || !_network.body || !_network.body.data.nodes) return [];
        return _network.body.data.nodes.get();
      },
      getCurrentEdges: function () {
        if (!_network || !_network.body || !_network.body.data.edges) return [];
        return _network.body.data.edges.get();
      },
      getPinnedPositions: function () {
        if (_network) {
          var allIds = _nodesDataSet ? _nodesDataSet.getIds() : [];
          if (allIds.length > 0) {
            var positions = _network.getPositions(allIds);
            return positions || {};
          }
        }
        var copy = {};
        Object.keys(_pinnedNodePositions).forEach(function (k) { copy[k] = _pinnedNodePositions[k]; });
        return copy;
      },
      destroy: function () {
        hideNodeHover();
        if (_hoverPanel) {
          _hoverPanel.remove();
          _hoverPanel = null;
        }
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

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
    var _nodesDataSet = null;
    var _edgesDataSet = null;
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

      // Compute relation count per entity for node sizing
      var relationCounts = {};
      for (var ri = 0; ri < relations.length; ri++) {
        var r = relations[ri];
        relationCounts[r.entity1_absolute_id] = (relationCounts[r.entity1_absolute_id] || 0) + 1;
        relationCounts[r.entity2_absolute_id] = (relationCounts[r.entity2_absolute_id] || 0) + 1;
      }
      buildNodesOpts.relationCounts = relationCounts;

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

      // Invalidate hover panel — vis.Network may have cleared the container DOM
      _hoverPanel = null;

      _network.once('stabilizationIterationsDone', function () {
        // Freeze after stabilization: stop simulation but keep interaction working
        _network.setOptions({ physics: { enabled: false } });
        if (highlightAbsId) {
          _network.focus(highlightAbsId, { scale: 1.2, animation: { duration: 500, easingFunction: 'easeInOutQuad' } });
        }
        // Render version badges after layout is stable
        renderVersionBadges(nodes);
      });

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
        var nodeId = params.node;
        showNodeHover(nodeId, params);
      });
      _network.on('hoverEdge', function (params) {
        var edgeId = params.edge;
        showEdgeHover(edgeId, params);
      });
      _network.on('blurNode', function () {
        hideNodeHover();
      });
      _network.on('blurEdge', function () {
        hideNodeHover();
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

      // Update version badge positions on zoom/pan/drag
      _network.on('zoom', function () { updateBadgePositions(); updateNodeHoverPosition(); });
      _network.on('dragEnd', function () {
        // The dragEnd above handles node dragging; this is for canvas panning
        setTimeout(updateBadgePositions, 50);
        updateNodeHoverPosition();
      });
      _network.on('viewChanged', function () { updateBadgePositions(); updateNodeHoverPosition(); });
    }

    // ---- Incremental graph building for smooth animations ----
    // Creates an empty network with persistent DataSets for incremental .add()

    function initEmptyGraph(hubLayout) {
      _entityMap = {};
      _relationMap = {};

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

      // Invalidate hover panel — vis.Network may have cleared the container DOM
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
        var nodeId = params.node;
        showNodeHover(nodeId, params);
      });
      _network.on('hoverEdge', function (params) {
        var edgeId = params.edge;
        showEdgeHover(edgeId, params);
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

      // Pin hub nodes in triangle layout if provided
      if (hubLayout && hubLayout.hubIds && hubLayout.hubIds.length > 0) {
        var cx = container.offsetWidth / 2;
        var cy = container.offsetHeight / 2;
        var tr = 150;
        var hubPositions = [
          { x: cx, y: cy - tr },
          { x: cx - tr * 0.866, y: cy + tr * 0.5 },
          { x: cx + tr * 0.866, y: cy + tr * 0.5 },
        ];
        for (var hi = 0; hi < hubLayout.hubIds.length && hi < hubPositions.length; hi++) {
          _pinnedNodePositions[hubLayout.hubIds[hi]] = hubPositions[hi];
        }
      }
    }

    // Incrementally add entities and relations to the existing network
    // Uses DataSet .add() for smooth, non-destructive updates

    function addNodesAndEdges(newEntities, newRelations, hubLayout) {
      if (!_network || !_nodesDataSet) return;

      // Build vis node objects from raw entity data
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

      // Compute relation count from ALL currently visible edges for accurate sizing
      var allEdges = _edgesDataSet.get();
      var relationCounts = {};
      for (var ri = 0; ri < allEdges.length; ri++) {
        var re = allEdges[ri];
        // vis edge 'from'/'to' map to entity absolute_ids (node IDs)
        relationCounts[re.from] = (relationCounts[re.from] || 0) + 1;
        relationCounts[re.to] = (relationCounts[re.to] || 0) + 1;
      }
      // Also count new relations not yet in DataSet
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

      // Update entity map
      for (var ek in result.entityMap) {
        _entityMap[ek] = result.entityMap[ek];
      }

      // Compute smart positions for new nodes based on their connected existing nodes
      var existingPositions = _network.getPositions();
      var existingNodeIds = new Set(_nodesDataSet.getIds());

      // Build a lookup: newEntity absolute_id -> list of connected existing nodeIds
      var newNodeConnections = {};
      for (var ni = 0; ni < newNodes.length; ni++) {
        newNodeConnections[newNodes[ni].id] = [];
      }
      // Check newRelations for connections to existing nodes
      for (var ri = 0; ri < newRelations.length; ri++) {
        var rel = newRelations[ri];
        var e1Id = rel.entity1_family_id || rel.entity1_absolute_id;
        var e2Id = rel.entity2_family_id || rel.entity2_absolute_id;
        // If one end is new and the other is existing, record the connection
        if (!existingNodeIds.has(e1Id) && existingNodeIds.has(e2Id)) {
          if (newNodeConnections[e1Id]) newNodeConnections[e1Id].push(e2Id);
        } else if (existingNodeIds.has(e1Id) && !existingNodeIds.has(e2Id)) {
          if (newNodeConnections[e2Id]) newNodeConnections[e2Id].push(e1Id);
        }
      }

      // Assign positions: new nodes appear near their connected existing neighbors
      var canvasEl = document.getElementById(_opts.canvasId);
      var cx = canvasEl ? canvasEl.offsetWidth / 2 : 400;
      var cy = canvasEl ? canvasEl.offsetHeight / 2 : 300;

      for (var ni = 0; ni < newNodes.length; ni++) {
        var node = newNodes[ni];
        var pinned = _pinnedNodePositions[node.id];
        if (pinned) {
          node.x = pinned.x;
          node.y = pinned.y;
          node.fixed = { x: true, y: true };
          continue;
        }

        var connections = newNodeConnections[node.id];
        if (connections && connections.length > 0) {
          // Average position of connected existing nodes
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
          // No connection to existing nodes — place near center with some spread
          node.x = cx + (Math.random() - 0.5) * 300;
          node.y = cy + (Math.random() - 0.5) * 300;
        }
      }

      // Add to DataSet (triggers smooth network update)
      _nodesDataSet.add(newNodes);

      // Brief highlight pulse for newly added nodes (grow animation effect)
      if (newNodes.length > 0 && newNodes.length <= 50) {
        var newNodeIds = newNodes.map(function(n) { return n.id; });
        _network.selectNodes(newNodeIds);
        setTimeout(function() {
          if (_network) _network.unselectAll();
        }, 300);
      }

      // Build vis edge objects — use ALL visible node IDs as the filter
      var allNodeIds = new Set(_nodesDataSet.getIds());
      var buildEdgesOpts = {};
      if (_opts.relationStrengthEnabled) buildEdgesOpts.weightMode = 'count';
      if (hubLayout) buildEdgesOpts.hubMap = hubLayout.hubMap;

      var edgeResult = GraphUtils.buildEdges(newRelations, allNodeIds, buildEdgesOpts);
      var newEdges = edgeResult.edges.get();

      // Update relation map
      for (var rk in edgeResult.relationMap) {
        _relationMap[rk] = edgeResult.relationMap[rk];
      }

      // Add edges (triggers smooth network update)
      _edgesDataSet.add(newEdges);

      // Recalculate node sizes: existing nodes should grow as new edges connect to them
      var allUpdatedEdges = _edgesDataSet.get();
      var liveCounts = {};
      for (var lci = 0; lci < allUpdatedEdges.length; lci++) {
        var le = allUpdatedEdges[lci];
        liveCounts[le.from] = (liveCounts[le.from] || 0) + 1;
        liveCounts[le.to] = (liveCounts[le.to] || 0) + 1;
      }
      var liveMax = 1;
      var liveKeys = Object.keys(liveCounts);
      for (var lki = 0; lki < liveKeys.length; lki++) {
        if (liveCounts[liveKeys[lki]] > liveMax) liveMax = liveCounts[liveKeys[lki]];
      }
      // Update sizes of ALL existing nodes (not just new ones)
      var allCurrentNodes = _nodesDataSet.get();
      var sizeUpdates = [];
      for (var sni = 0; sni < allCurrentNodes.length; sni++) {
        var sn = allCurrentNodes[sni];
        var newCount = liveCounts[sn.id] || 0;
        var newSize = GraphUtils.computeNodeSize(newCount, liveMax);
        if (Math.abs((sn.size || 0) - newSize) > 0.5) {
          sizeUpdates.push({ id: sn.id, size: newSize });
        }
      }
      if (sizeUpdates.length > 0) {
        _nodesDataSet.update(sizeUpdates);
      }

      // Update version badges
      renderVersionBadges(_nodesDataSet);

      // Auto-fit viewport to keep all nodes visible
      // Use instant fit (no animation) to avoid conflicts with rapid updates during animation
      try { _network.fit({ animation: false }); } catch (_) {}
    }

    /**
     * Incrementally remove nodes and edges from the current network.
     * Used by timeline drag to hide entities/relations outside the time window.
     * @param {string[]} nodeAbsIds - absolute_ids of nodes to remove
     * @param {string[]} edgeAbsIds - absolute_ids of edges to remove
     */
    function removeNodesAndEdges(nodeAbsIds, edgeAbsIds) {
      if (!_network || !_nodesDataSet) return;

      // Remove edges first (to avoid dangling references)
      if (edgeAbsIds && edgeAbsIds.length > 0) {
        _edgesDataSet.remove(edgeAbsIds);
        // Clean up relationMap
        for (var i = 0; i < edgeAbsIds.length; i++) {
          delete _relationMap[edgeAbsIds[i]];
        }
      }

      // Remove nodes
      if (nodeAbsIds && nodeAbsIds.length > 0) {
        _nodesDataSet.remove(nodeAbsIds);
        // Clean up entityMap
        for (var j = 0; j < nodeAbsIds.length; j++) {
          delete _entityMap[nodeAbsIds[j]];
        }
      }

      // Update version badges
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

      // Re-fit viewport
      try { _network.fit({ animation: false }); } catch (_) {}
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

        // 2b. Fetch ALL relations for the focused entity from API (supplement main view cache)
        var _apiRelationAbsIds = new Set();
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
        } catch (_) { /* non-fatal: proceed with main view cache only */ }

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

        // 4b. Add family_ids discovered from API relations to BFS visited set
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

    // ---- Content diff helper ----

    function computeContentDiff(currentContent, previousContent) {
      if (!previousContent) return null;
      var curLines = (currentContent || '').split('\n').filter(function(l) { return l.trim(); });
      var prevLines = (previousContent || '').split('\n').filter(function(l) { return l.trim(); });
      if (curLines.join('\n') === prevLines.join('\n')) return null;

      var added = [], removed = [];
      curLines.forEach(function(line) { if (prevLines.indexOf(line) === -1) added.push(line); });
      prevLines.forEach(function(line) { if (curLines.indexOf(line) === -1) removed.push(line); });

      if (added.length === 0 && removed.length === 0) return null;
      return { added: added, removed: removed };
    }

    function renderDiffPreview(diff) {
      if (!diff) return '';
      var html = '<div class="version-diff-container">';
      html += '<div class="version-diff-header"><span>' + t('entities.changes') + '</span><span style="color:var(--success);">+' + diff.added.length + '</span>/<span style="color:var(--error);">-' + diff.removed.length + '</span></div>';
      html += '<div class="version-diff-body">';
      diff.removed.forEach(function(line) {
        html += '<div class="version-diff-line removed">- ' + escapeHtml(line) + '</div>';
      });
      diff.added.forEach(function(line) {
        html += '<div class="version-diff-line added">+ ' + escapeHtml(line) + '</div>';
      });
      html += '</div></div>';
      return html;
    }

    // ---- Version evolution summary ----

    function renderVersionContext(versions, currentIdx) {
      var v = versions[currentIdx];
      if (!v) return '';
      var html = '<div class="version-context-card">';

      // Version creation time
      html += '<div class="version-ctx-row">';
      html += '<span class="version-ctx-label">' + t('graph.processedTime') + '</span>';
      html += '<span class="version-ctx-value">' + (v.processed_time ? formatDateMs(v.processed_time) : '-') + '</span>';
      html += '</div>';

      // Source document
      if (v.source_document) {
        html += '<div class="version-ctx-row">';
        html += '<span class="version-ctx-label">' + t('graph.sourceDoc') + '</span>';
        html += '<span class="version-ctx-value mono" style="font-size:0.6875rem;">' + escapeHtml(v.source_document) + '</span>';
        html += '</div>';
      }

      // Episode ID (clickable)
      if (v.episode_id) {
        html += '<div class="version-ctx-row">';
        html += '<span class="version-ctx-label">' + t('graph.episodeId') + '</span>';
        html += '<span class="doc-link version-ctx-value mono" style="font-size:0.6875rem;" data-view-episode="' + escapeHtml(v.episode_id) + '">' + escapeHtml(v.episode_id) + '</span>';
        html += '</div>';
      }

      // Change indicator: first version vs update
      if (currentIdx === 0) {
        html += '<div class="version-ctx-tag tag-created">Created</div>';
      } else {
        var hasContentChange = (v.content || '') !== (versions[currentIdx - 1].content || '');
        var hasNameChange = (v.name || '') !== (versions[currentIdx - 1].name || '');
        var tags = [];
        if (hasContentChange) tags.push('Content updated');
        if (hasNameChange) tags.push('Renamed');
        if (tags.length === 0) tags.push('Metadata update');
        html += '<div class="version-ctx-tag tag-updated">' + tags.join(' + ') + '</div>';
      }

      html += '</div>';
      return html;
    }

    function renderVersionEvolutionSummary(versions) {
      if (versions.length < 2) return '';
      var changeCount = 0;
      for (var i = 1; i < versions.length; i++) {
        if ((versions[i].content || '') !== (versions[i - 1].content || '') ||
            (versions[i].name || '') !== (versions[i - 1].name || '')) {
          changeCount++;
        }
      }

      var times = versions.map(function(v) { return v.processed_time ? new Date(v.processed_time).getTime() : 0; }).filter(function(t) { return t > 0; });
      var timeSpan = '';
      if (times.length >= 2) {
        var diff = Math.max.apply(null, times) - Math.min.apply(null, times);
        if (diff < 3600000) timeSpan = Math.round(diff / 60000) + 'm';
        else if (diff < 86400000) timeSpan = (diff / 3600000).toFixed(1) + 'h';
        else timeSpan = (diff / 86400000).toFixed(1) + 'd';
      }

      var nameChanges = 0;
      var contentChanges = 0;
      for (var j = 1; j < versions.length; j++) {
        if (versions[j].name !== versions[j - 1].name) nameChanges++;
        if (versions[j].content !== versions[j - 1].content) contentChanges++;
      }

      return '<div class="version-evolution-summary">' +
        '<div class="version-evo-stat"><div class="version-evo-stat-value">' + versions.length + '</div><div class="version-evo-stat-label">' + t('graph.versions') + '</div></div>' +
        '<div class="version-evo-stat"><div class="version-evo-stat-value">' + changeCount + '</div><div class="version-evo-stat-label">' + t('entities.changes') + '</div></div>' +
        (timeSpan ? '<div class="version-evo-stat"><div class="version-evo-stat-value">' + timeSpan + '</div><div class="version-evo-stat-label">' + t('graph.timeSpan') + '</div></div>' : '') +
        (nameChanges > 0 ? '<div class="version-evo-stat"><div class="version-evo-stat-value" style="color:var(--warning);">' + nameChanges + '</div><div class="version-evo-stat-label">' + t('graph.nameChanges') + '</div></div>' : '') +
      '</div>';
    }

    // ---- Keyboard shortcut hints ----

    function renderKeyboardHints() {
      return '<div class="keyboard-hints">' +
        '<span class="kb-hint"><span class="kb-key">&larr;</span><span class="kb-key">&rarr;</span> ' + t('graph.switchVersion') + '</span>' +
        '<span class="kb-hint"><span class="kb-key">Space</span> ' + t('graph.playPause') + '</span>' +
        '<span class="kb-hint"><span class="kb-key">Esc</span> ' + t('graph.exitFocus') + '</span>' +
      '</div>';
    }

    // ---- Mini version timeline widget ----

    function renderMiniVersionTimeline(versions, currentIdx, prefix) {
      if (versions.length < 2) return '';
      var html = '<div class="version-mini-timeline" style="margin-bottom:0.75rem;">';

      // Time range
      var times = versions.map(function (v) {
        return v.processed_time ? new Date(v.processed_time).getTime() : 0;
      });
      var validTimes = times.filter(function (t) { return t > 0; });
      var minT = validTimes.length > 0 ? Math.min.apply(null, validTimes) : 0;
      var maxT = validTimes.length > 0 ? Math.max.apply(null, validTimes) : 0;
      var range = maxT - minT || 1;

      // Version dots with gap indicators
      html += '<div class="version-mini-track">';
      for (var i = 0; i < versions.length; i++) {
        var t = times[i] || 0;
        var pct = t > 0 && range > 0 ? ((t - minT) / range * 100) : (i / Math.max(versions.length - 1, 1) * 100);
        var isCurrent = i === currentIdx;
        var cls = 'version-mini-dot' + (isCurrent ? ' current' : '');
        var timeStr = versions[i].processed_time ? formatDateMs(versions[i].processed_time) : '';

        // Source label (episode source or version number)
        var sourceLabel = '';
        if (versions[i].source_document) {
          sourceLabel = versions[i].source_document.replace(/^document:/, '').substring(0, 20);
        } else {
          sourceLabel = 'v' + (i + 1);
        }

        // Time gap from previous version
        var gapLabel = '';
        if (i > 0 && times[i] > 0 && times[i - 1] > 0) {
          var gapMs = times[i] - times[i - 1];
          if (gapMs < 60000) gapLabel = gapMs + 'ms';
          else if (gapMs < 3600000) gapLabel = Math.round(gapMs / 60000) + 'm';
          else if (gapMs < 86400000) gapLabel = (gapMs / 3600000).toFixed(1) + 'h';
          else gapLabel = (gapMs / 86400000).toFixed(1) + 'd';
        }

        html += '<div class="' + cls + '" style="left:' + pct.toFixed(1) + '%;" data-ver-idx="' + i + '" title="v' + (i + 1) + ' — ' + timeStr + '">';
        // Version number badge inside dot
        html += '<span class="version-mini-dot-num">' + (i + 1) + '</span>';
        html += '</div>';

        // Source label below the dot
        html += '<div class="version-mini-source" style="left:' + pct.toFixed(1) + '%;">' + escapeHtml(sourceLabel) + '</div>';

        // Gap indicator between dots
        if (gapLabel && i > 0) {
          var prevPct = times[i - 1] > 0 && range > 0 ? ((times[i - 1] - minT) / range * 100) : ((i - 1) / Math.max(versions.length - 1, 1) * 100);
          var midPct = (prevPct + pct) / 2;
          html += '<div class="version-mini-gap" style="left:' + midPct.toFixed(1) + '%;">' + gapLabel + '</div>';
        }
      }
      // Connecting line
      html += '<div class="version-mini-line"></div>';
      html += '</div>';

      // Timestamp labels
      if (versions.length >= 2) {
        html += '<div class="version-mini-labels">';
        html += '<span class="version-mini-label">' + (versions[0].processed_time ? formatDateMs(versions[0].processed_time) : 'v1') + '</span>';
        html += '<span class="version-mini-label">' + (versions[versions.length - 1].processed_time ? formatDateMs(versions[versions.length - 1].processed_time) : 'v' + versions.length) + '</span>';
        html += '</div>';
      }

      html += '</div>';
      return html;
    }

    // ---- Version badge overlays on vis-network canvas ----
    // Version information shown via node border styling (amber border + glow)
    // No DOM overlays — version count visible in tooltip and detail sidebar

    function renderVersionBadges(nodesDataSet) {
      if (!_network) return;
      var container = _el(_opts.canvasId);
      if (!container) return;

      // Remove any legacy overlay badges
      var oldBadges = container.querySelectorAll('.version-badge-overlay');
      for (var oi = 0; oi < oldBadges.length; oi++) oldBadges[oi].remove();

      // Add watermark
      var oldWatermark = container.querySelector('.graph-deepdream-watermark');
      if (!oldWatermark) {
        var wm = document.createElement('div');
        wm.className = 'graph-deepdream-watermark';
        wm.textContent = 'Deep-Dream';
        container.appendChild(wm);
      }
    }

    function updateBadgePositions() {
      // No-op: version info is in node border styling, no overlay to update
    }

    // ---- Node hover info panel ----
    var _hoverPanel = null;
    var _hoverNodeId = null;

    function showNodeHover(nodeId, params) {
      if (!_network) return;
      var entity = _entityMap[nodeId];
      if (!entity) return;

      var container = _el(_opts.canvasId);
      if (!container) return;

      // Create panel if needed (also recreate if detached from DOM by vis-network rebuild)
      if (!_hoverPanel || !_hoverPanel.parentElement) {
        _hoverPanel = document.createElement('div');
        _hoverPanel.className = 'node-hover-info';
        _hoverPanel.style.opacity = '0';
        container.appendChild(_hoverPanel);
      }

      // Build content
      var name = entity.name || entity.family_id || nodeId;
      var vc = _versionCounts[entity.family_id] || _versionCounts[nodeId] || 0;
      var summary = entity.summary || '';
      var content = entity.content || '';

      var html = '<div class="nhv-name">' + escapeHtml(name);
      if (vc > 1) {
        html += ' <span class="nhv-version">[v' + vc + ']</span>';
      }
      html += '</div>';
      // Prefer summary for preview, fall back to content
      var preview = summary || content || '';
      if (preview) {
        if (preview.length > 150) preview = preview.substring(0, 150) + '...';
        html += '<div class="nhv-content">' + escapeHtml(preview) + '</div>';
      }
      if (entity.processed_time) {
        html += '<div style="font-size:0.6875rem;color:var(--text-muted);margin-top:0.25rem;">' + formatDate(entity.processed_time) + '</div>';
      }

      _hoverPanel.innerHTML = html;
      _hoverNodeId = nodeId;

      // Position near the node
      updateNodeHoverPosition();

      // Show with slight delay for smooth appearance
      requestAnimationFrame(function () {
        if (_hoverPanel) _hoverPanel.style.opacity = '1';
      });
    }

    function hideNodeHover() {
      if (_hoverPanel) {
        _hoverPanel.style.opacity = '0';
      }
      _hoverNodeId = null;
    }

    function updateNodeHoverPosition() {
      if (!_hoverPanel || !_hoverNodeId || !_network) return;

      var container = _el(_opts.canvasId);
      if (!container) return;

      // Edge hover — reposition at edge midpoint
      if (_hoverNodeId.indexOf('__edge__') === 0) {
        var edgeId = _hoverNodeId.replace('__edge__', '');
        var edgeEnds = _network.getConnectedNodes(edgeId);
        if (edgeEnds && edgeEnds.length === 2) {
          var positions = _network.getPositions(edgeEnds);
          var p1 = positions[edgeEnds[0]];
          var p2 = positions[edgeEnds[1]];
          if (p1 && p2) {
            var midCanvas = { x: (p1.x + p2.x) / 2, y: (p1.y + p2.y) / 2 };
            var domPos = _network.canvasToDOM(midCanvas);
            _hoverPanel.style.left = domPos.x + 'px';
            _hoverPanel.style.top = (domPos.y - 30) + 'px';
          }
        }
        return;
      }

      var canvasPos = _network.getPositions([_hoverNodeId]);
      if (!canvasPos[_hoverNodeId]) return;

      var domPos = _network.canvasToDOM({ x: canvasPos[_hoverNodeId].x, y: canvasPos[_hoverNodeId].y });

      // Get node size to offset positioning
      var node = _network.body.nodes[_hoverNodeId];
      var nodeSize = node ? (node.options.size || 20) : 20;

      var left = domPos.x + nodeSize + 12;
      var top = domPos.y - 20;

      // Keep panel within container bounds
      var panelRect = _hoverPanel.getBoundingClientRect();
      var containerRect = container.getBoundingClientRect();
      if (left + 240 > containerRect.width) {
        left = domPos.x - nodeSize - 252;
      }
      if (top + 100 > containerRect.height) {
        top = containerRect.height - 110;
      }
      if (top < 10) top = 10;
      if (left < 10) left = 10;

      _hoverPanel.style.left = left + 'px';
      _hoverPanel.style.top = top + 'px';
    }

    function escapeHtml(text) {
      var div = document.createElement('div');
      div.appendChild(document.createTextNode(text));
      return div.innerHTML;
    }

    function showEdgeHover(edgeId, params) {
      if (!_network) return;
      var relation = _relationMap[edgeId];
      if (!relation) return;

      var container = _el(_opts.canvasId);
      if (!container) return;

      // Reuse or create panel (also recreate if detached from DOM)
      if (!_hoverPanel || !_hoverPanel.parentElement) {
        _hoverPanel = document.createElement('div');
        _hoverPanel.className = 'node-hover-info';
        _hoverPanel.style.opacity = '0';
        container.appendChild(_hoverPanel);
      }

      // Look up endpoint entity names
      var e1Name = '';
      var e2Name = '';
      if (relation.entity1_absolute_id && _entityMap[relation.entity1_absolute_id]) {
        e1Name = _entityMap[relation.entity1_absolute_id].name || '';
      }
      if (relation.entity2_absolute_id && _entityMap[relation.entity2_absolute_id]) {
        e2Name = _entityMap[relation.entity2_absolute_id].name || '';
      }

      var html = '';
      if (e1Name || e2Name) {
        html += '<div class="nhv-name" style="font-size:0.75rem;">' + escapeHtml(e1Name || '?') + ' <span style="color:var(--text-muted);margin:0 0.25rem;">&harr;</span> ' + escapeHtml(e2Name || '?') + '</div>';
      }
      var content = relation.content || relation.summary || '';
      if (content) {
        var preview = content.length > 150 ? content.substring(0, 150) + '...' : content;
        html += '<div class="nhv-content" style="margin-top:0.25rem;">' + escapeHtml(preview) + '</div>';
      }
      if (relation.processed_time) {
        html += '<div style="font-size:0.6875rem;color:var(--text-muted);margin-top:0.25rem;">' + formatDate(relation.processed_time) + '</div>';
      }

      if (!html) return; // Don't show empty panel

      _hoverPanel.innerHTML = html;
      _hoverNodeId = '__edge__' + edgeId;

      // Position at edge midpoint
      var edgeEnds = _network.getConnectedNodes(edgeId);
      if (edgeEnds && edgeEnds.length === 2) {
        var positions = _network.getPositions(edgeEnds);
        var p1 = positions[edgeEnds[0]];
        var p2 = positions[edgeEnds[1]];
        if (p1 && p2) {
          var midCanvas = { x: (p1.x + p2.x) / 2, y: (p1.y + p2.y) / 2 };
          var domPos = _network.canvasToDOM(midCanvas);
          _hoverPanel.style.left = domPos.x + 'px';
          _hoverPanel.style.top = (domPos.y - 30) + 'px';
        }
      }

      requestAnimationFrame(function () {
        if (_hoverPanel) _hoverPanel.style.opacity = '1';
      });
    }

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

      // Compute diff against previous version
      var versionDiff = null;
      if (totalVersions > 1 && _currentVersionIdx > 0) {
        versionDiff = computeContentDiff(
          versions[_currentVersionIdx].content,
          versions[_currentVersionIdx - 1].content
        );
      }

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

        // Version evolution summary (if > 1 version)
        (totalVersions > 1 ? renderVersionEvolutionSummary(versions) : '') +

        // Mini version timeline (if > 1 version)
        (totalVersions > 1 ? renderMiniVersionTimeline(versions, _currentVersionIdx, prefix) : '') +

        // Content diff preview (if version changed from previous)
        (versionDiff ? renderDiffPreview(versionDiff) : '') +

        // Version context: which episode/source created this version
        renderVersionContext(versions, _currentVersionIdx) +

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

        // Keyboard shortcut hints
        (totalVersions > 1 ? renderKeyboardHints() : '') +

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

      // Mini version timeline dot clicks
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
        '</div>';

      detailContent.querySelectorAll('[data-view-entity]').forEach(function (el) {
        el.addEventListener('click', function () { showEntityDetail(el.getAttribute('data-view-entity')); });
      });
      detailContent.querySelectorAll('[data-focus-entity]').forEach(function (el) {
        el.addEventListener('click', function () { focusOnEntity(el.getAttribute('data-focus-entity')); });
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

        // Build rich version timeline with inline diff
        var sorted = versions.slice().sort(function(a, b) {
          var ta = a.processed_time ? new Date(a.processed_time).getTime() : 0;
          var tb = b.processed_time ? new Date(b.processed_time).getTime() : 0;
          return tb - ta;
        });

        function simpleContentDiff(current, previous) {
          if (!previous) return null;
          var curLines = (current.content || '').split('\n').filter(function(l) { return l.trim(); });
          var prevLines = (previous.content || '').split('\n').filter(function(l) { return l.trim(); });
          if (curLines.join('\n') === prevLines.join('\n')) return null;
          var added = [], removed = [];
          curLines.forEach(function(line) { if (prevLines.indexOf(line) === -1) added.push(line); });
          prevLines.forEach(function(line) { if (curLines.indexOf(line) === -1) removed.push(line); });
          return { added: added, removed: removed };
        }

        var items = sorted.map(function(v, i) {
          var prev = sorted[i + 1];
          var diff = simpleContentDiff(v, prev);
          var hasNameChange = prev && v.name !== prev.name;
          var verNum = sorted.length - i;

          // Time gap indicator
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

          // Source label
          var sourceLabel = v.source_document || '';
          if (sourceLabel.length > 25) sourceLabel = sourceLabel.substring(0, 22) + '...';

          // Episode ID (clickable to view source episode)
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

          // Inline diff
          if (diff) {
            headerHtml += '<div style="margin-top:0.5rem;border-left:3px solid var(--primary);padding:0.375rem 0.5rem;background:var(--bg-input);border-radius:0 0.375rem 0.375rem 0;font-size:0.8125rem;">';
            diff.removed.forEach(function(line) {
              headerHtml += '<div style="color:var(--error);text-decoration:line-through;opacity:0.7;padding:1px 0;">- ' + escapeHtml(line) + '</div>';
            });
            diff.added.forEach(function(line) {
              headerHtml += '<div style="color:var(--success);padding:1px 0;">+ ' + escapeHtml(line) + '</div>';
            });
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

        // Attach expand/collapse
        modal.overlay.querySelectorAll('[id^="graph-version-toggle-"]').forEach(function(toggle) {
          toggle.addEventListener('click', function() {
            var idx = toggle.id.replace('graph-version-toggle-', '');
            var expanded = modal.overlay.querySelector('#graph-version-expanded-' + idx);
            if (expanded) expanded.style.display = expanded.style.display === 'none' ? 'block' : 'none';
          });
        });

        // Attach episode ID click handlers
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

    // ---- Public API ----

    return {
      buildGraph: buildGraph,
      initEmptyGraph: initEmptyGraph,
      addNodesAndEdges: addNodesAndEdges,
      removeNodesAndEdges: removeNodesAndEdges,
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

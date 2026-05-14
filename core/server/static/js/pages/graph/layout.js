/* ==========================================
   Graph Page — Layout utilities
   Hub layout computation, hop expansion
   ========================================== */

window.GraphPageLayout = (function () {
  'use strict';

  /** Expand N-hop neighbors from seed entities (client-side BFS) */
  function expandNHops(seedAbsIds, allRelations, hopLevel) {
    var result = new Set(seedAbsIds);
    var frontier = new Set(seedAbsIds);
    for (var h = 1; h <= hopLevel; h++) {
      var nextFrontier = new Set();
      for (var ri = 0; ri < allRelations.length; ri++) {
        var r = allRelations[ri];
        if (frontier.has(r.entity1_absolute_id) && !result.has(r.entity2_absolute_id)) {
          nextFrontier.add(r.entity2_absolute_id);
        }
        if (frontier.has(r.entity2_absolute_id) && !result.has(r.entity1_absolute_id)) {
          nextFrontier.add(r.entity1_absolute_id);
        }
      }
      nextFrontier.forEach(function(id) { result.add(id); });
      frontier = nextFrontier;
    }
    return result;
  }

  /** Compute top-3 hub entities and their 1-hop neighbors */
  function computeHubLayout(visibleRelations) {
    if (!visibleRelations || visibleRelations.length === 0) return null;

    var relCounts = {};
    for (var ri = 0; ri < visibleRelations.length; ri++) {
      relCounts[visibleRelations[ri].entity1_absolute_id] = (relCounts[visibleRelations[ri].entity1_absolute_id] || 0) + 1;
      relCounts[visibleRelations[ri].entity2_absolute_id] = (relCounts[visibleRelations[ri].entity2_absolute_id] || 0) + 1;
    }
    var sorted = Object.entries(relCounts).sort(function(a, b) { return b[1] - a[1]; });
    var hubIds = sorted.slice(0, 3).map(function(e) { return e[0]; });
    if (hubIds.length === 0) return null;

    // hubMap: absoluteId -> hubIndex (0/1/2)
    var hubMap = {};
    for (var i = 0; i < hubIds.length; i++) {
      hubMap[hubIds[i]] = i;
    }

    // 1-hop neighbors inherit hub color
    var hubNeighborIds = new Set();
    for (var rj = 0; rj < visibleRelations.length; rj++) {
      var r = visibleRelations[rj];
      var e1 = r.entity1_absolute_id, e2 = r.entity2_absolute_id;
      var h1 = hubMap[e1], h2 = hubMap[e2];
      if (h1 !== undefined && h2 === undefined) {
        hubMap[e2] = h1;
        hubNeighborIds.add(e2);
      } else if (h2 !== undefined && h1 === undefined) {
        hubMap[e1] = h2;
        hubNeighborIds.add(e1);
      }
    }

    return { hubMap: hubMap, hubIds: hubIds, hubNeighborIds: hubNeighborIds };
  }

  return {
    expandNHops: expandNHops,
    computeHubLayout: computeHubLayout,
  };
})();

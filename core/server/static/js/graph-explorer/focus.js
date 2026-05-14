/* ==========================================
   GraphExplorer — Focus session & BFS
   FocusSession class, focusBFS, focusOnEntity
   ========================================== */

window.GraphExplorerFocus = (function () {
  'use strict';

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

  function focusBFS(startFamilyId, hopLevel, entityCache, mainViewRelations) {
    // Build abs_id -> family_id map
    var absToFid = {};
    for (var absId in entityCache) {
      absToFid[absId] = entityCache[absId].family_id;
    }

    // Build family_id -> [relation] index from main view cache
    var familyIndex = {};
    var mainRels = mainViewRelations || [];
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

  return {
    FocusSession: FocusSession,
    focusBFS: focusBFS,
  };
})();

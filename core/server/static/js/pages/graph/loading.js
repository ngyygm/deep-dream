/* ==========================================
   Graph Page — Loading functions
   SSE loading, incremental updates, version poll
   ========================================== */

// This module is used via globals — the main graph.js IIFE calls these functions.
// We export them on window.GraphPageLoading so the IIFE can destructure them.

window.GraphPageLoading = (function () {
  'use strict';

  function consumeSSE(url, handlers) {
    return SSEClient.get(url, {
      onEvent: function (eventType, data) {
        if (handlers[eventType]) handlers[eventType](data);
      },
      onDone: function () {
        if (handlers.done_event) handlers.done_event();
      },
    });
  }

  /** Check /graph/version against cached last_modified.
   *  Returns true if server has newer data, false if unchanged, null on error. */
  async function checkGraphVersion(graphId, graphLastModified) {
    if (!graphLastModified) return true;
    try {
      var res = await fetch('/api/v1/find/graph/version?graph_id=' + encodeURIComponent(graphId));
      if (!res.ok) return null;
      var data = (await res.json()).data;
      if (!data || !data.last_modified) return true;
      return data.last_modified !== graphLastModified;
    } catch (_) {
      return null;
    }
  }

  /** Incremental update: fetch only entities/relations since graphLastModified, merge into cache. */
  async function loadGraphIncremental(opts) {
    // opts: { graphId, graphLastModified, cachedAllEntities, cachedAllRawRelations,
    //         cachedAllEdges, cachedInheritedRelationIds, explorer, state,
    //         updateCachedData, consumeSSE }
    var since = opts.graphLastModified;
    var graphId = opts.graphId;
    var _consumeSSE = opts.consumeSSE || consumeSSE;
    var statsEl = opts.statsEl;
    var loadingEl = opts.loadingEl;

    if (loadingEl) loadingEl.style.display = 'flex';
    if (statsEl) statsEl.textContent = t('common.loading');

    var newEntities = [];
    var newRelations = [];

    var streamBase = '/api/v1/find/graph/stream/';
    var streamParams = '?graph_id=' + encodeURIComponent(graphId) + '&since=' + encodeURIComponent(since);

    var entPromise = _consumeSSE(streamBase + 'entities' + streamParams, {
      entity: function (d) { if (d) newEntities.push(d); },
      error: function () {}
    }).catch(function () {});

    var relPromise = _consumeSSE(streamBase + 'relations' + streamParams, {
      relation: function (d) { if (d) newRelations.push(d); },
      error: function () {}
    }).catch(function () {});

    await Promise.all([entPromise, relPromise]);

    // Return results for the caller to merge
    return {
      newEntities: newEntities,
      newRelations: newRelations,
      hasChanges: newEntities.length > 0 || newRelations.length > 0,
    };
  }

  /** Full SSE-based graph load: fetch entities and relations via streaming. */
  async function loadGraphSSE(opts) {
    // opts: { graphId, statsEl, loadingEl, state, consumeSSE,
    //         onEntity, onRelation, onEntityMeta, onRelationMeta, onEntityError, onRelationError }
    var graphId = opts.graphId;
    var _consumeSSE = opts.consumeSSE || consumeSSE;
    var statsEl = opts.statsEl;

    var totalEntities = 0;
    var totalRelations = 0;

    var streamBase = '/api/v1/find/graph/stream/';
    var streamParams = '?graph_id=' + encodeURIComponent(graphId);

    var entityFailed = false;
    var relFailed = false;

    var entityPromise = _consumeSSE(streamBase + 'entities' + streamParams, {
      meta: function (d) { if (d) totalEntities = d.total || 0; if (opts.onEntityMeta) opts.onEntityMeta(d); },
      entity: function (d) {
        if (!d) return;
        if (opts.onEntity) opts.onEntity(d, totalEntities);
      },
      error: function (d) {
        console.error('Entity stream error:', (d || {}).message);
        entityFailed = true;
        if (opts.onEntityError) opts.onEntityError(d);
      }
    }).catch(function (err) {
      console.error('Entity stream failed:', err);
      entityFailed = true;
    });

    var relPromise = _consumeSSE(streamBase + 'relations' + streamParams, {
      meta: function (d) { if (d) totalRelations = d.total || 0; if (opts.onRelationMeta) opts.onRelationMeta(d); },
      relation: function (d) {
        if (!d) return;
        if (opts.onRelation) opts.onRelation(d, totalRelations);
      },
      error: function (d) {
        console.error('Relation stream error:', (d || {}).message);
        relFailed = true;
        if (opts.onRelationError) opts.onRelationError(d);
      }
    }).catch(function (err) {
      console.error('Relation stream failed:', err);
      relFailed = true;
    });

    await Promise.all([entityPromise, relPromise]);

    return {
      entityFailed: entityFailed,
      relFailed: relFailed,
      bothFailed: entityFailed && relFailed,
    };
  }

  // ---- Resolve unknown relation endpoints and remap to current entity versions ----

  async function resolveAndRemapRelations(entities, relations, graphId, api) {
    var currentAbsIds = new Set(entities.map(function(e) { return e.absolute_id; }));
    var currentEntityIds = {};
    for (var ei = 0; ei < entities.length; ei++) {
      currentEntityIds[entities[ei].family_id] = entities[ei].absolute_id;
    }

    // Collect unknown endpoint absolute_ids
    var unknownAbsIds = new Set();
    for (var ri = 0; ri < relations.length; ri++) {
      if (!currentAbsIds.has(relations[ri].entity1_absolute_id)) unknownAbsIds.add(relations[ri].entity1_absolute_id);
      if (!currentAbsIds.has(relations[ri].entity2_absolute_id)) unknownAbsIds.add(relations[ri].entity2_absolute_id);
    }

    // Batch resolve unknown endpoints in concurrent batches of 50
    var resolved = {};
    var toResolve = Array.from(unknownAbsIds);
    var batchSize = 50;
    for (var i = 0; i < toResolve.length; i += batchSize) {
      var batch = toResolve.slice(i, i + batchSize);
      var promises = batch.map(function(absId) {
        return api.entityByAbsoluteId(absId, graphId).then(function(res) {
          if (res.data) resolved[absId] = res.data;
        }).catch(function() {});
      });
      await Promise.all(promises);
    }

    // Remap relations: replace old absolute_ids with current versions
    var remapped = [];
    var inheritedRelationIds = new Set();

    for (var rj = 0; rj < relations.length; rj++) {
      var r = relations[rj];
      var e1AbsId = r.entity1_absolute_id;
      var e2AbsId = r.entity2_absolute_id;
      var remapped1 = false, remapped2 = false;

      if (!currentAbsIds.has(e1AbsId)) {
        var oldEntity = resolved[e1AbsId];
        if (oldEntity && currentEntityIds[oldEntity.family_id]) {
          e1AbsId = currentEntityIds[oldEntity.family_id];
          remapped1 = true;
        }
      }
      if (!currentAbsIds.has(e2AbsId)) {
        var oldEntity2 = resolved[e2AbsId];
        if (oldEntity2 && currentEntityIds[oldEntity2.family_id]) {
          e2AbsId = currentEntityIds[oldEntity2.family_id];
          remapped2 = true;
        }
      }

      if (remapped1 || remapped2) {
        inheritedRelationIds.add(r.absolute_id);
        remapped.push(Object.assign({}, r, { entity1_absolute_id: e1AbsId, entity2_absolute_id: e2AbsId }));
      } else {
        remapped.push(r);
      }
    }

    return { relations: remapped, inheritedRelationIds: inheritedRelationIds };
  }

  return {
    consumeSSE: consumeSSE,
    checkGraphVersion: checkGraphVersion,
    loadGraphIncremental: loadGraphIncremental,
    loadGraphSSE: loadGraphSSE,
    resolveAndRemapRelations: resolveAndRemapRelations,
  };
})();

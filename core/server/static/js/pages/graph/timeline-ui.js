/* ==========================================
   Graph Page — Timeline UI
   Timeline rendering, playback, drag, density visualization
   ========================================== */

// This module provides timeline functions used by the graph page.
// Exported on window.GraphPageTimeline so the IIFE in graph.js can use them.

window.GraphPageTimeline = (function () {
  'use strict';

  // ---- Canvas time indicator overlay ----

  function showCanvasTimeOverlay(timeStr, progressStr) {
    var overlay = document.getElementById('canvas-time-overlay');
    var textEl = document.getElementById('canvas-time-text');
    var progressEl = document.getElementById('canvas-time-progress');
    if (!overlay) return;
    if (textEl) textEl.textContent = timeStr || '';
    if (progressEl) progressEl.textContent = progressStr || '';
    overlay.style.display = '';
  }

  function hideCanvasTimeOverlay() {
    var overlay = document.getElementById('canvas-time-overlay');
    if (overlay) overlay.style.display = 'none';
  }

  function formatTimeShort(ts) {
    if (!ts) return '';
    var d = new Date(ts);
    return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' }) + ' ' +
      d.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' });
  }

  // ---- Render timeline bar ----

  function renderTimeline(container, opts) {
    // opts: { tlEpisodes, tlMinTime, tlMaxTime, tlCurrentPos, tlIsLive, tlIsDragging,
    //         cachedAllNodes, tlPlaybackTimer, tlPlaybackMode, escapeHtml }
    var tlEpisodes = opts.tlEpisodes;
    var tlMinTime = opts.tlMinTime;
    var tlMaxTime = opts.tlMaxTime;
    var tlCurrentPos = opts.tlCurrentPos;
    var tlIsLive = opts.tlIsLive;
    var tlIsDragging = opts.tlIsDragging;
    var cachedAllNodes = opts.cachedAllNodes;
    var tlPlaybackTimer = opts.tlPlaybackTimer;
    var tlPlaybackMode = opts.tlPlaybackMode || 'grow';
    var _escapeHtml = opts.escapeHtml || escapeHtml;

    var timeRange = tlMaxTime - tlMinTime;
    var posPercent = (tlCurrentPos * 100).toFixed(2);

    // Build version density bars
    var densityHtml = '';
    if (cachedAllNodes.length > 0 && timeRange > 0) {
      var BINS = 50;
      var bins = new Array(BINS).fill(0);
      for (var bi = 0; bi < cachedAllNodes.length; bi++) {
        var e = cachedAllNodes[bi];
        var bt = e.processed_time ? new Date(e.processed_time).getTime() : 0;
        if (bt >= tlMinTime && bt <= tlMaxTime) {
          var idx = Math.min(Math.floor((bt - tlMinTime) / timeRange * BINS), BINS - 1);
          bins[idx]++;
        }
      }
      var maxBin = Math.max.apply(null, bins.concat([1]));
      var barWidth = (100 / BINS).toFixed(3);
      densityHtml = '<div class="timeline-density-bar">';
      for (var b = 0; b < BINS; b++) {
        var h = Math.max(bins[b] / maxBin * 100, 0).toFixed(1);
        var opacity = Math.max(bins[b] / maxBin, 0.05).toFixed(2);
        var binPct = ((b + 0.5) / BINS * 100).toFixed(2);
        densityHtml += '<div class="timeline-density-col" style="width:' + barWidth + '%;height:' + h + '%;opacity:' + opacity + ';" data-bin-pct="' + binPct + '" title="' + bins[b] + ' entities"></div>';
      }
      densityHtml += '</div>';
    }

    var markerHtml = tlEpisodes.map(function(ep, i) {
      if (timeRange === 0) return '';
      var pct = ((ep.time - tlMinTime) / timeRange * 100).toFixed(2);
      var statsHtml = (ep.entityCount || ep.relationCount)
        ? '<br><span style="font-size:0.625rem;">E:' + (ep.entityCount || '?') + ' R:' + (ep.relationCount || '?') + '</span>'
        : '';
      var typeIcon = ep.type === 'dream'
        ? '<span style="color:var(--warning);font-size:0.6875rem;">&#9728;</span> '
        : '<span style="color:var(--primary);font-size:0.6875rem;">&#9679;</span> ';
      return '<div class="timeline-marker type-' + ep.type + '" style="left:' + pct + '%;" data-ep-idx="' + i + '">' +
        '<div class="timeline-tooltip">' + typeIcon + _escapeHtml(ep.label) + '<br><span style="color:var(--text-muted);">' + formatDate(new Date(ep.time).toISOString()) + '</span>' + statsHtml + '</div>' +
      '</div>';
    }).join('');

    var minLabel = tlMinTime ? new Date(tlMinTime).toLocaleDateString() : '-';
    var maxLabel = tlMaxTime ? new Date(tlMaxTime).toLocaleDateString() : '-';
    var currentTime = tlCurrentPos < 1
      ? new Date(tlMinTime + tlCurrentPos * timeRange)
      : null;
    var currentTimeStr = currentTime
      ? currentTime.toLocaleString()
      : '';

    container.innerHTML =
      '<div class="timeline-bar" style="display:flex;align-items:center;gap:0.5rem;padding:0.375rem 0.75rem;">' +
        '<div style="display:flex;align-items:center;gap:0.25rem;flex-shrink:0;">' +
          (tlIsLive
            ? '<span class="timeline-live-dot"></span>'
            : '<span class="timeline-live-dot snapshot"></span>') +
          '<button class="timeline-btn" id="tl-step-back" title="' + t('timeline.stepBack') + '" ' + (tlIsLive ? 'disabled' : '') + '>' +
            '<i data-lucide="skip-back" style="width:11px;height:11px;"></i>' +
          '</button>' +
          '<button class="timeline-btn' + (tlPlaybackTimer ? ' active' : '') + '" id="tl-play-btn" title="' + (tlPlaybackTimer ? t('timeline.pause') : t('timeline.play')) + '">' +
            '<i data-lucide="' + (tlPlaybackTimer ? 'pause' : 'play') + '" style="width:11px;height:11px;"></i>' +
          '</button>' +
          '<button class="timeline-btn" id="tl-step-forward" title="' + t('timeline.stepForward') + '">' +
            '<i data-lucide="skip-forward" style="width:11px;height:11px;"></i>' +
          '</button>' +
          '<button class="timeline-btn" id="tl-reset-live" title="' + t('timeline.resetToLive') + '" ' + (tlIsLive ? 'disabled' : '') + '>' +
            '<i data-lucide="zap" style="width:11px;height:11px;"></i>' +
          '</button>' +
          '<button class="timeline-btn tl-mode-btn' + (tlPlaybackMode === 'grow' ? ' active' : '') + '" id="tl-mode-grow" title="Grow">' +
            '<i data-lucide="sprout" style="width:11px;height:11px;"></i>' +
          '</button>' +
          '<button class="timeline-btn tl-mode-btn' + (tlPlaybackMode === 'snapshot' ? ' active' : '') + '" id="tl-mode-snapshot" title="Snapshot">' +
            '<i data-lucide="camera" style="width:11px;height:11px;"></i>' +
          '</button>' +
        '</div>' +
        '<div style="flex:1;min-width:0;position:relative;">' +
          densityHtml +
          '<div class="timeline-track" id="tl-track" style="position:relative;">' +
            '<div class="timeline-track-fill" style="width:' + posPercent + '%;"></div>' +
            markerHtml +
            '<div class="timeline-thumb' + (tlIsDragging ? ' dragging' : '') + '" id="tl-thumb" style="left:' + posPercent + '%;"></div>' +
          '</div>' +
        '</div>' +
        '<div style="flex-shrink:0;text-align:right;min-width:120px;">' +
          '<span class="mono" style="font-size:0.6875rem;color:var(--text-secondary);" id="tl-current-time">' +
            (currentTimeStr || maxLabel) +
          '</span>' +
        '</div>' +
      '</div>';

    if (window.lucide) lucide.createIcons({ nodes: [container] });
  }

  function updateThumbPosition(tlCurrentPos, tlMinTime, tlMaxTime) {
    var fill = document.querySelector('.timeline-track-fill');
    var thumb = document.getElementById('tl-thumb');
    var curTime = document.getElementById('tl-current-time');
    if (fill) fill.style.width = (tlCurrentPos * 100).toFixed(2) + '%';
    if (thumb) thumb.style.left = (tlCurrentPos * 100).toFixed(2) + '%';

    var timeRange = tlMaxTime - tlMinTime;
    if (curTime && timeRange > 0 && tlCurrentPos < 1) {
      var t = new Date(tlMinTime + tlCurrentPos * timeRange);
      curTime.textContent = t.toLocaleString();
    } else if (curTime) {
      curTime.textContent = '';
    }
  }

  // ---- Drag time filter helpers ----

  function prepareDragSortedData(cachedAllNodes, cachedAllEdges, cachedAllEntities) {
    var sortedNodes = cachedAllNodes.slice().sort(function(a, b) {
      var ta = a.processed_time ? new Date(a.processed_time).getTime() : 0;
      var tb = b.processed_time ? new Date(b.processed_time).getTime() : 0;
      return ta - tb;
    });
    var sortedEdges = cachedAllEdges.slice().sort(function(a, b) {
      var ta1 = (cachedAllEntities[a.entity1_absolute_id] || {}).processed_time;
      var ta2 = (cachedAllEntities[a.entity2_absolute_id] || {}).processed_time;
      var tb1 = (cachedAllEntities[b.entity1_absolute_id] || {}).processed_time;
      var tb2 = (cachedAllEntities[b.entity2_absolute_id] || {}).processed_time;
      var taMax = Math.max(ta1 ? new Date(ta1).getTime() : 0, ta2 ? new Date(ta2).getTime() : 0);
      var tbMax = Math.max(tb1 ? new Date(tb1).getTime() : 0, tb2 ? new Date(tb2).getTime() : 0);
      return taMax - tbMax;
    });
    return { sortedNodes: sortedNodes, sortedEdges: sortedEdges };
  }

  function applyDragTimeFilter(targetTimeMs, opts) {
    // opts: { tlDragSortedNodes, tlDragSortedEdges, tlDragVisibleNodeIds, tlDragVisibleEdgeIds,
    //         tlDragLastTime, cachedAllEntities, explorer, computeHubLayout, cachedAllEdges }
    var tlDragSortedNodes = opts.tlDragSortedNodes;
    var tlDragSortedEdges = opts.tlDragSortedEdges;
    var cachedAllEntities = opts.cachedAllEntities;

    // Debounce
    if (Math.abs(targetTimeMs - opts.tlDragLastTime) < 1000 && opts.tlDragVisibleNodeIds.size > 0) return null;
    var tlDragLastTime = targetTimeMs;

    // Find nodes with processed_time <= targetTimeMs
    var nodeEndIdx = tlDragSortedNodes.length;
    for (var i = 0; i < tlDragSortedNodes.length; i++) {
      var nt = tlDragSortedNodes[i].processed_time ? new Date(tlDragSortedNodes[i].processed_time).getTime() : 0;
      if (nt > targetTimeMs) { nodeEndIdx = i; break; }
    }
    // Find edges with max endpoint time <= targetTimeMs
    var edgeEndIdx = tlDragSortedEdges.length;
    for (var j = 0; j < tlDragSortedEdges.length; j++) {
      var ea1 = (cachedAllEntities[tlDragSortedEdges[j].entity1_absolute_id] || {}).processed_time;
      var ea2 = (cachedAllEntities[tlDragSortedEdges[j].entity2_absolute_id] || {}).processed_time;
      var eMaxT = Math.max(ea1 ? new Date(ea1).getTime() : 0, ea2 ? new Date(ea2).getTime() : 0);
      if (eMaxT > targetTimeMs) { edgeEndIdx = j; break; }
    }

    // Determine target visible sets
    var targetNodeIds = new Set();
    var targetNodeData = [];
    for (var ni = 0; ni < nodeEndIdx; ni++) {
      targetNodeIds.add(tlDragSortedNodes[ni].absolute_id);
      targetNodeData.push(tlDragSortedNodes[ni]);
    }
    var targetEdgeIds = new Set();
    var targetEdgeData = [];
    for (var ei = 0; ei < edgeEndIdx; ei++) {
      var edge = tlDragSortedEdges[ei];
      if (targetNodeIds.has(edge.entity1_absolute_id) && targetNodeIds.has(edge.entity2_absolute_id)) {
        targetEdgeIds.add(edge.absolute_id);
        targetEdgeData.push(edge);
      }
    }

    return {
      targetNodeIds: targetNodeIds,
      targetEdgeIds: targetEdgeIds,
      targetNodeData: targetNodeData,
      targetEdgeData: targetEdgeData,
      tlDragLastTime: tlDragLastTime,
    };
  }

  // ---- Snapshot transition animation ----

  function playSnapshotTransition() {
    var canvasParent = document.getElementById('graph-canvas');
    if (!canvasParent) return;

    var oldOverlay = canvasParent.querySelector('.snapshot-transition-overlay');
    if (oldOverlay) oldOverlay.remove();

    var overlay = document.createElement('div');
    overlay.className = 'snapshot-transition-overlay';

    var flash = document.createElement('div');
    flash.className = 'snapshot-flash';
    overlay.appendChild(flash);

    setTimeout(function() {
      var scanline = document.createElement('div');
      scanline.className = 'snapshot-scanline';
      overlay.appendChild(scanline);
    }, 80);

    setTimeout(function() {
      var ripple = document.createElement('div');
      ripple.className = 'snapshot-ripple';
      overlay.appendChild(ripple);
    }, 150);

    canvasParent.appendChild(overlay);

    setTimeout(function() { if (overlay.parentNode) overlay.remove(); }, 1500);
  }

  // ---- Snapshot overlay badge ----

  function updateSnapshotOverlay(timeStr, entityCount, relationCount) {
    var overlay = document.getElementById('snapshot-overlay');
    var timeEl = document.getElementById('snapshot-overlay-time');
    var statsEl = document.getElementById('snapshot-overlay-stats');
    if (!overlay) return;
    if (timeStr) {
      overlay.style.display = '';
      if (timeEl) timeEl.textContent = timeStr;
      if (statsEl && entityCount !== undefined) {
        statsEl.textContent = 'E:' + entityCount + ' R:' + relationCount;
      }
      if (window.lucide) lucide.createIcons({ nodes: [overlay] });
    } else {
      overlay.style.display = 'none';
    }
  }

  // ---- Grow animation helpers ----

  function startGrowAnimation(opts) {
    // opts: { cachedAllNodes, cachedAllEdges, explorer, tlPlaybackSpeed,
    //         computeHubLayout, playSnapshotTransition, renderTimeline }
    var cachedAllNodes = opts.cachedAllNodes;
    var cachedAllEdges = opts.cachedAllEdges;

    // Group entities by episode_id
    var epEntMap = {};
    var noEpNodes = [];
    for (var i = 0; i < cachedAllNodes.length; i++) {
      var node = cachedAllNodes[i];
      if (node.episode_id) {
        if (!epEntMap[node.episode_id]) epEntMap[node.episode_id] = [];
        epEntMap[node.episode_id].push(node);
      } else {
        noEpNodes.push(node);
      }
    }

    // Group relations by episode_id
    var epRelMap = {};
    var noEpRels = [];
    for (var ri = 0; ri < cachedAllEdges.length; ri++) {
      var rel = cachedAllEdges[ri];
      if (rel.episode_id) {
        if (!epRelMap[rel.episode_id]) epRelMap[rel.episode_id] = [];
        epRelMap[rel.episode_id].push(rel);
      } else {
        noEpRels.push(rel);
      }
    }

    // Build episode groups
    var allEpIds = new Set([].concat(Object.keys(epEntMap), Object.keys(epRelMap)));
    var groups = [];
    allEpIds.forEach(function(epId) {
      var ents = epEntMap[epId] || [];
      var rels = epRelMap[epId] || [];
      var minTime = Infinity;
      var maxTime = 0;
      for (var k = 0; k < ents.length; k++) {
        var t = ents[k].processed_time ? new Date(ents[k].processed_time).getTime() : 0;
        if (t > 0 && t < minTime) minTime = t;
        if (t > maxTime) maxTime = t;
      }
      if (minTime === Infinity) minTime = 0;
      groups.push({ episode_id: epId, entities: ents, relations: rels, minTime: minTime, maxTime: maxTime });
    });
    groups.sort(function(a, b) { return a.minTime - b.minTime; });

    if (noEpNodes.length > 0 || noEpRels.length > 0) {
      var noEpMax = 0;
      for (var ni = 0; ni < noEpNodes.length; ni++) {
        var nt = noEpNodes[ni].processed_time ? new Date(noEpNodes[ni].processed_time).getTime() : 0;
        if (nt > noEpMax) noEpMax = nt;
      }
      groups.unshift({ episode_id: null, entities: noEpNodes, relations: noEpRels, minTime: 0, maxTime: noEpMax });
    }

    return {
      groups: groups,
      hubLayout: opts.computeHubLayout(cachedAllEdges),
    };
  }

  return {
    showCanvasTimeOverlay: showCanvasTimeOverlay,
    hideCanvasTimeOverlay: hideCanvasTimeOverlay,
    formatTimeShort: formatTimeShort,
    renderTimeline: renderTimeline,
    updateThumbPosition: updateThumbPosition,
    prepareDragSortedData: prepareDragSortedData,
    applyDragTimeFilter: applyDragTimeFilter,
    playSnapshotTransition: playSnapshotTransition,
    updateSnapshotOverlay: updateSnapshotOverlay,
    startGrowAnimation: startGrowAnimation,
  };
})();

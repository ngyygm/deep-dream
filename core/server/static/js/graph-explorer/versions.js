/* ==========================================
   GraphExplorer — Version management
   Version evolution, switching, modals
   ========================================== */

window.GraphExplorerVersions = (function () {
  'use strict';

  var escapeHtml = window.escapeHtml || function escapeHtml(text) {
    var div = document.createElement('div');
    div.appendChild(document.createTextNode(text));
    return div.innerHTML;
  };

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
      var vt = times[i] || 0;
      var pct = vt > 0 && range > 0 ? ((vt - minT) / range * 100) : (i / Math.max(versions.length - 1, 1) * 100);
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

  return {
    renderVersionContext: renderVersionContext,
    renderVersionEvolutionSummary: renderVersionEvolutionSummary,
    renderKeyboardHints: renderKeyboardHints,
    renderMiniVersionTimeline: renderMiniVersionTimeline,
  };
})();

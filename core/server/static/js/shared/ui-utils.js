/* ==========================================
   Shared UI utility functions
   Progress bars, spinners, status badges, etc.
   ========================================== */

window.UIUtils = (function () {
  'use strict';

  function tripleProgressBar(opts) {
    var cols = [
      { pct: opts.smp, color: 'var(--primary)', label: t('dashboard.mainWindow'), text: opts.mainLabel },
      { pct: opts.s9p, color: 'var(--info)', label: t('dashboard.entityAlign'), text: opts.step9Label },
      { pct: opts.s10p, color: 'var(--warning)', label: t('dashboard.relationAlign'), text: opts.step10Label },
    ];
    var html = '<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:4px 12px;">';
    for (var ci = 0; ci < cols.length; ci++) {
      var c = cols[ci];
      html += '<div>'
        + '<div style="font-size:0.65rem;color:' + c.color + ';margin-bottom:2px;">' + c.label + '</div>'
        + '<div class="progress-bar" style="height:3px;"><div class="progress-bar-fill" style="width:' + (c.pct * 100).toFixed(2) + '%;background:' + c.color + ';"></div></div>'
        + '<div style="font-size:0.6rem;color:var(--text-muted);margin-top:1px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">' + escapeHtml(c.text || '-') + '</div>'
        + '</div>';
    }
    html += '</div>';
    if (opts.showOverall) {
      html = '<div style="min-width:240px;">'
        + '<div style="font-size:0.6rem;color:var(--text-muted);margin-bottom:4px;">' + t('memory.overallProgress') + ' ' + (opts.overallP * 100).toFixed(2) + '%</div>'
        + '<div style="margin-bottom:4px;">' + html + '</div>'
        + '</div>';
    }
    return html;
  }

  function progressBar(pct, cls) {
    cls = cls || '';
    var w = Math.min(100, Math.max(0, (pct || 0) * 100));
    return '<div class="progress-bar"><div class="progress-bar-fill ' + cls + '" style="width:' + w.toFixed(1) + '%"></div></div>';
  }

  function spinnerHtml(cls) {
    cls = cls || '';
    return '<div class="spinner ' + cls + '"></div>';
  }

  function emptyState(text, icon) {
    icon = icon || 'inbox';
    return '<div class="empty-state"><i data-lucide="' + icon + '"></i><p>' + escapeHtml(text) + '</p></div>';
  }

  function statusBadge(status, phase) {
    if (status === 'running' && phase === 'pausing') {
      return '<span class="badge badge-warning">' + escapeHtml(t('memory.statusPausing')) + '</span>';
    }
    if (status === 'running' && phase === 'cancelling') {
      return '<span class="badge badge-error">' + escapeHtml(t('memory.statusCancelling')) + '</span>';
    }
    var map = {
      queued: 'badge-warning',
      running: 'badge-info',
      paused: 'badge-warning',
      completed: 'badge-success',
      failed: 'badge-error',
    };
    return '<span class="badge ' + (map[status] || 'badge-primary') + '">' + escapeHtml(status) + '</span>';
  }

  function renderVersionTimeline(opts) {
    var sorted = [].concat(opts.versions).sort(function(a, b) {
      var ta = a.processed_time ? new Date(a.processed_time).getTime() : 0;
      var tb = b.processed_time ? new Date(b.processed_time).getTime() : 0;
      return tb - ta;
    });

    var items = sorted.map(function(v, i) {
      var prev = sorted[i + 1];
      var isActive = opts.isActiveCheck ? opts.isActiveCheck(v) : (i === 0);
      var diffHtml = opts.renderDiff ? opts.renderDiff(v, prev) : '';
      var headerHtml = opts.renderHeader ? opts.renderHeader(v, i, sorted, isActive) : '';
      var bodyHtml = opts.renderBody ? opts.renderBody(v) : '';

      return '<div style="position:relative;padding-left:1.5rem;padding-bottom:' + (i < sorted.length - 1 ? '1rem' : '0') + ';">'
        + (i < sorted.length - 1 ? '<div style="position:absolute;left:5px;top:12px;bottom:0;width:1px;background:var(--border-color);"></div>' : '')
        + '<div style="position:absolute;left:0;top:4px;width:11px;height:11px;border-radius:50%;background:' + (isActive ? 'var(--primary)' : 'var(--border-color)') + ';border:2px solid ' + (isActive ? 'var(--primary-hover)' : 'var(--border-hover)') + ';"></div>'
        + '<div style="cursor:pointer;" class="' + opts.toggleClass + '" data-version-idx="' + i + '">'
        + headerHtml
        + diffHtml
        + '</div>'
        + '<div class="' + opts.expandedIdPrefix + '" id="' + opts.expandedIdPrefix + '-' + i + '" style="display:none;margin-top:0.5rem;">'
        + bodyHtml
        + '</div>'
        + '</div>';
    }).join('');

    // Attach expand/collapse behavior
    setTimeout(function() {
      var container = opts.overlay.querySelector('#' + opts.containerId);
      if (!container) return;
      container.querySelectorAll('.' + opts.toggleClass).forEach(function(toggle) {
        toggle.addEventListener('click', function() {
          var idx = toggle.getAttribute('data-version-idx');
          var expanded = opts.overlay.querySelector('#' + opts.expandedIdPrefix + '-' + idx);
          if (expanded) {
            var isHidden = expanded.style.display === 'none';
            expanded.style.display = isHidden ? 'block' : 'none';
          }
        });
      });
    }, 0);

    return items;
  }

  // Make clickable table rows keyboard-accessible
  function bindClickableRows(container) {
    if (!container) return;
    container.querySelectorAll('tr[data-family-id], tr[data-task-id]').forEach(function(row) {
      if (!row.hasAttribute('tabindex')) row.setAttribute('tabindex', '0');
      row.setAttribute('role', 'button');
      row.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          row.click();
        }
      });
    });
  }

  return {
    tripleProgressBar: tripleProgressBar,
    progressBar: progressBar,
    spinnerHtml: spinnerHtml,
    emptyState: emptyState,
    statusBadge: statusBadge,
    renderVersionTimeline: renderVersionTimeline,
    bindClickableRows: bindClickableRows,
  };
})();

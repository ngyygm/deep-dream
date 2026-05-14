/* ==========================================
   GraphExplorer — Diff utilities
   LCS-based inline diff for version comparison
   ========================================== */

window.GraphExplorerDiff = (function () {
  'use strict';

  var escapeHtml = window.escapeHtml || function escapeHtml(text) {
    var div = document.createElement('div');
    div.appendChild(document.createTextNode(text));
    return div.innerHTML;
  };

  /**
   * Shared LCS engine: builds DP table once, backtracks to produce diff ops.
   * oldArr/newArr are arrays of tokens (chars or lines).
   * Returns [{type:'equal'|'del'|'ins', text: string}]
   */
  function _lcsDiff(oldArr, newArr) {
    var n = oldArr.length, m = newArr.length;
    var dp = [new Array(m + 1).fill(0)];
    for (var i = 1; i <= n; i++) {
      var row = [0];
      for (var j = 1; j <= m; j++) {
        if (oldArr[i - 1] === newArr[j - 1]) row[j] = dp[i - 1][j - 1] + 1;
        else row[j] = Math.max(dp[i - 1][j], row[j - 1]);
      }
      dp.push(row);
    }
    var ops = [];
    var ci = n, cj = m;
    while (ci > 0 || cj > 0) {
      if (ci > 0 && cj > 0 && oldArr[ci - 1] === newArr[cj - 1]) {
        ops.push({ type: 'equal', text: oldArr[ci - 1] });
        ci--; cj--;
      } else if (cj > 0 && (ci === 0 || dp[ci][cj - 1] >= dp[ci - 1][cj])) {
        ops.push({ type: 'ins', text: newArr[cj - 1] });
        cj--;
      } else {
        ops.push({ type: 'del', text: oldArr[ci - 1] });
        ci--;
      }
    }
    ops.reverse();
    return ops;
  }

  function _renderDiffSpans(runs) {
    var html = '';
    for (var i = 0; i < runs.length; i++) {
      var r = runs[i];
      var escaped = escapeHtml(r.text);
      if (r.type === 'equal') {
        html += '<span style="color:var(--text-secondary);">' + escaped + '</span>';
      } else if (r.type === 'del') {
        html += '<span style="color:var(--error);text-decoration:line-through;">' + escaped + '</span>';
      } else {
        html += '<span style="color:var(--success);font-weight:500;">' + escaped + '</span>';
      }
    }
    return html;
  }

  function computeInlineDiff(oldText, newText) {
    oldText = oldText || '';
    newText = newText || '';
    if (oldText === newText) return null;

    var n = oldText.length, m = newText.length;
    // For very long texts, fall back to line-level diff
    if (n * m > 4000000) {
      var oldLines = oldText.split('\n');
      var newLines = newText.split('\n');
      return _lcsDiff(oldLines, newLines);
    }

    // Character-level diff
    var chars = _lcsDiff(oldText.split(''), newText.split(''));
    // Merge consecutive same-type ops into runs
    var runs = [];
    for (var k = 0; k < chars.length; k++) {
      var op = chars[k];
      if (runs.length > 0 && runs[runs.length - 1].type === op.type) {
        runs[runs.length - 1].text += op.text;
      } else {
        runs.push({ type: op.type, text: op.text });
      }
    }
    return runs;
  }

  function renderDiffPreview(runs) {
    if (!runs || runs.length === 0) return '';
    var delCount = 0, insCount = 0;
    for (var i = 0; i < runs.length; i++) {
      if (runs[i].type === 'del') delCount += runs[i].text.length;
      if (runs[i].type === 'ins') insCount += runs[i].text.length;
    }
    var html = '<div class="version-diff-container">';
    html += '<div class="version-diff-header"><span>' + t('entities.changes') + '</span><span style="color:var(--success);">+' + insCount + '</span>/<span style="color:var(--error);">-' + delCount + '</span></div>';
    html += '<div class="version-diff-body" style="white-space:pre-wrap;word-break:break-all;line-height:1.6;">';
    html += _renderDiffSpans(runs);
    html += '</div></div>';
    return html;
  }

  return {
    _lcsDiff: _lcsDiff,
    _renderDiffSpans: _renderDiffSpans,
    computeInlineDiff: computeInlineDiff,
    renderDiffPreview: renderDiffPreview,
  };
})();

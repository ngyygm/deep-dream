/* ==========================================
   Shared formatting utilities
   Date, time, number, HTML formatting
   ========================================== */

window.Format = (function () {
  'use strict';

  function getLocale() {
    var lang = (window.I18N && window.I18N.currentLang) || 'zh';
    return lang === 'en' ? 'en-US' : lang === 'ja' ? 'ja-JP' : 'zh-CN';
  }

  function formatDate(isoStr) {
    if (!isoStr) return '-';
    try {
      var d = new Date(isoStr);
      return d.toLocaleString(getLocale(), {
        year: 'numeric', month: '2-digit', day: '2-digit',
        hour: '2-digit', minute: '2-digit', second: '2-digit',
      });
    } catch { return isoStr; }
  }

  function formatDateMs(isoStr) {
    if (!isoStr) return '-';
    try {
      var d = new Date(isoStr);
      var ms = String(d.getMilliseconds()).padStart(3, '0');
      return d.toLocaleString(getLocale(), {
        year: 'numeric', month: '2-digit', day: '2-digit',
        hour: '2-digit', minute: '2-digit', second: '2-digit',
      }) + '.' + ms;
    } catch { return isoStr; }
  }

  function formatRelativeTime(seconds) {
    if (seconds == null) return '-';
    seconds = Math.max(0, Math.round(seconds));
    if (seconds < 60) return seconds + 's';
    var m = Math.floor(seconds / 60);
    var s = seconds % 60;
    if (m < 60) return m + 'm' + String(s).padStart(2, '0') + 's';
    var h = Math.floor(m / 60);
    var rm = m % 60;
    return h + 'h' + String(rm).padStart(2, '0') + 'm';
  }

  function formatNumber(n) {
    if (n == null) return '0';
    return n.toLocaleString();
  }

  function getElapsed(startedAt, finishedAt) {
    if (!startedAt) return '-';
    var start = Number(startedAt);
    if (isNaN(start)) return '-';
    if (start < 4102444800000) start *= 1000;

    var end;
    if (finishedAt) {
      end = Number(finishedAt);
      if (end < 4102444800000) end *= 1000;
    } else {
      end = Date.now();
    }

    var diff = Math.max(0, Math.round((end - start) / 1000));
    return formatRelativeTime(diff);
  }

  function truncate(str, maxLen) {
    maxLen = maxLen || 80;
    if (!str) return '';
    return str.length > maxLen ? str.slice(0, maxLen) + '...' : str;
  }

  function escapeHtml(str) {
    if (!str) return '';
    var div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
  }

  function escapeAttr(s) {
    return String(s).replace(/&/g,'&amp;').replace(/"/g,'&quot;').replace(/'/g,'&#39;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  }

  return {
    getLocale: getLocale,
    formatDate: formatDate,
    formatDateMs: formatDateMs,
    formatRelativeTime: formatRelativeTime,
    formatNumber: formatNumber,
    getElapsed: getElapsed,
    truncate: truncate,
    escapeHtml: escapeHtml,
    escapeAttr: escapeAttr,
  };
})();

// Also expose escapeHtml globally for backward compatibility
window.escapeHtml = window.Format.escapeHtml;

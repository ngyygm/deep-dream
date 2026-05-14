/**
 * ApiHelpers — Shared frontend utilities for button loading states,
 * unified API calls with toast feedback, and AbortController management.
 *
 * Usage:
 *   ApiHelpers.withLoading(btn, async () => { ... });
 *   ApiHelpers.withApiCall(() => fetch(...), { successMsg: 'Done!', errorMsg: 'Failed' });
 *   const { signal, abort } = ApiHelpers.createAbortController('search');
 */
window.ApiHelpers = (() => {
  // Module-level map for abort controller management
  const _abortControllers = new Map();

  /**
   * Wrap an async operation with button loading state.
   * Disables button, shows spinner, re-enables on completion.
   *
   * @param {HTMLElement} btn - button element
   * @param {Function} asyncFn - async operation to run
   * @returns {Promise<*>} result of asyncFn
   */
  async function withLoading(btn, asyncFn) {
    const origHtml = btn.innerHTML;
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner spinner-sm"></span> ' + (btn.dataset.loadingText || '');
    try {
      return await asyncFn();
    } finally {
      btn.disabled = false;
      btn.innerHTML = origHtml;
      if (window.lucide) lucide.createIcons({ nodes: [btn] });
    }
  }

  /**
   * Unified API call with success/error toast feedback.
   *
   * @param {Function} apiFn - async function returning the API result
   * @param {object} [opts] - { successMsg, errorMsg }
   * @returns {Promise<*>} result of apiFn
   */
  async function withApiCall(apiFn, { successMsg, errorMsg } = {}) {
    try {
      const result = await apiFn();
      if (successMsg && typeof showToast === 'function') showToast(successMsg, 'success');
      return result;
    } catch (e) {
      const msg = (errorMsg || 'Operation failed') + ': ' + e.message;
      if (typeof showToast === 'function') showToast(msg, 'error');
      throw e;
    }
  }

  /**
   * Abort controller manager — abort any previous request with the same key,
   * then return a fresh { signal, abort } for the current request.
   *
   * @param {string} key - unique key identifying the request slot
   * @returns {{ signal: AbortSignal, abort: Function }}
   */
  function createAbortController(key) {
    // Abort previous controller for this key, if any
    const prev = _abortControllers.get(key);
    if (prev) prev.abort();

    const controller = new AbortController();
    _abortControllers.set(key, controller);

    return {
      signal: controller.signal,
      abort() {
        controller.abort();
        _abortControllers.delete(key);
      },
    };
  }

  return { withLoading, withApiCall, createAbortController };
})();

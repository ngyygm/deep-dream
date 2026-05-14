/**
 * SSEClient — Unified SSE stream reader using fetch + ReadableStream.
 *
 * Provides both POST- and GET-based SSE consumption with proper frame
 * parsing, multi-line data support, and AbortController integration.
 *
 * Usage:
 *   SSEClient.post('/api/stream', { q: 'hello' }, {
 *     onEvent(type, data) { ... },
 *     onDone()           { ... },
 *     onError(err)       { ... },
 *   }, { signal: myAbortController.signal });
 *
 *   SSEClient.get('/api/stream?x=1', { ... });
 */
window.SSEClient = {
  /**
   * Internal: read SSE frames from a fetch Response and dispatch events.
   * @param {Response} res - fetch Response with a ReadableStream body
   * @param {object} handlers - { onEvent(eventType, data), onDone(), onError(error) }
   */
  async _consume(res, handlers) {
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let curEvent = '';
    let curData = '';

    function dispatch() {
      if (!curEvent || !curData) return;
      let data;
      try {
        data = JSON.parse(curData);
      } catch {
        data = { text: curData };
      }
      if (handlers.onEvent) handlers.onEvent(curEvent, data);
      curEvent = '';
      curData = '';
    }

    function parseLines(text) {
      const lines = text.split('\n');
      for (const line of lines) {
        if (line.startsWith('event: ')) {
          curEvent = line.slice(7).trim();
        } else if (line.startsWith('data: ')) {
          // Multi-line data: concatenate with newline
          if (curData) curData += '\n';
          curData += line.slice(6);
        } else if (line.startsWith('id: ')) {
          // SSE id field — recognized but not used currently
        } else if (line === '') {
          // Empty line dispatches the accumulated event
          dispatch();
        }
      }
    }

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        // Split on newlines, keep the last incomplete fragment in buffer
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';
        parseLines(lines.join('\n') + '\n');
      }

      // Flush any remaining buffered data
      if (buffer.trim()) {
        parseLines(buffer + '\n');
      }

      if (handlers.onDone) handlers.onDone();
    } catch (err) {
      if (err.name === 'AbortError') return;
      if (handlers.onError) handlers.onError(err);
    }
  },

  /**
   * POST to an SSE endpoint and consume events.
   * @param {string} url - endpoint URL
   * @param {object} body - POST body (JSON)
   * @param {object} handlers - { onEvent(eventType, data), onDone(), onError(error) }
   * @param {object} [options] - { signal: AbortSignal }
   */
  async post(url, body, handlers, options = {}) {
    try {
      const res = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
        signal: options.signal || undefined,
      });

      if (!res.ok) {
        const text = await res.text().catch(() => '');
        let msg = text;
        try { msg = JSON.parse(text).error || text; } catch { /* keep raw */ }
        throw new Error(msg || `HTTP ${res.status}`);
      }

      await this._consume(res, handlers);
    } catch (err) {
      if (err.name === 'AbortError') return;
      if (handlers.onError) handlers.onError(err);
    }
  },

  /**
   * GET from an SSE endpoint and consume events.
   * @param {string} url - endpoint URL
   * @param {object} handlers - { onEvent(eventType, data), onDone(), onError(error) }
   * @param {object} [options] - { signal: AbortSignal }
   * @returns {Promise} resolves when the stream ends or errors
   */
  async get(url, handlers, options = {}) {
    try {
      const res = await fetch(url, {
        signal: options.signal || undefined,
      });

      if (!res.ok) {
        throw new Error('HTTP ' + res.status);
      }

      await this._consume(res, handlers);
    } catch (err) {
      if (err.name === 'AbortError') return;
      if (handlers.onError) handlers.onError(err);
    }
  },
};

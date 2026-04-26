/**
 * DreamSSEClient — SSE stream reader using fetch + ReadableStream.
 *
 * Usage:
 *   const client = new DreamSSEClient('/api/v1/find/ask/stream', { question: 'hello' });
 *   client.onEvent = (type, data) => { ... };
 *   client.onDone  = (data)  => { ... };
 *   client.onError = (err)   => { ... };
 *   client.start();
 */
class DreamSSEClient {
  constructor(url, body, options = {}) {
    this.url = url;
    this.body = body;
    this.abortController = new AbortController();
    this._started = false;

    // Callbacks
    this.onEvent = options.onEvent || null;   // (event_type, data) => void
    this.onDone  = options.onDone  || null;   // (data) => void
    this.onError = options.onError || null;   // (Error) => void
  }

  async start() {
    if (this._started) return;
    this._started = true;

    try {
      const res = await fetch(this.url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(this.body),
        signal: this.abortController.signal,
      });

      if (!res.ok) {
        const text = await res.text().catch(() => '');
        let msg = text;
        try { msg = JSON.parse(text).error || text; } catch { /* keep raw */ }
        throw new Error(msg || `HTTP ${res.status}`);
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        // Parse SSE frames from buffer
        const lines = buffer.split('\n');
        buffer = lines.pop() || ''; // keep incomplete line

        let currentEvent = '';
        let currentData = '';

        for (const line of lines) {
          if (line.startsWith('event: ')) {
            currentEvent = line.slice(7).trim();
          } else if (line.startsWith('data: ')) {
            currentData = line.slice(6);
          } else if (line === '' && currentEvent && currentData) {
            // Empty line = end of event
            this._dispatch(currentEvent, currentData);
            currentEvent = '';
            currentData = '';
          }
        }
      }

      // Dispatch any remaining buffered event
      if (buffer.trim()) {
        const remaining = buffer + '\n';
        const rLines = remaining.split('\n');
        let ev = '', dt = '';
        for (const line of rLines) {
          if (line.startsWith('event: ')) ev = line.slice(7).trim();
          else if (line.startsWith('data: ')) dt = line.slice(6);
          else if (line === '' && ev && dt) {
            this._dispatch(ev, dt);
            ev = ''; dt = '';
          }
        }
      }

      // Stream body closed — ensure onDone fires even if server didn't send event: done
      if (this.onDone) this.onDone();
    } catch (err) {
      if (err.name === 'AbortError') return; // cancelled — silent
      if (this.onError) this.onError(err);
    }
  }

  stop() {
    this.abortController.abort();
  }

  _dispatch(eventType, rawData) {
    let data;
    try {
      data = JSON.parse(rawData);
    } catch {
      data = { text: rawData };
    }

    if (eventType === 'done') {
      if (this.onDone) this.onDone(data);
    } else if (eventType === 'error') {
      if (this.onError) this.onError(new Error(data.message || data.text || 'Stream error'));
    }

    if (this.onEvent) this.onEvent(eventType, data);
  }
}

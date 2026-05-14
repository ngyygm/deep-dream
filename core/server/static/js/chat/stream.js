/**
 * DreamSSEClient — SSE stream reader, now backed by the shared SSEClient.
 *
 * Usage (unchanged):
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

    await SSEClient.post(this.url, this.body, {
      onEvent: (eventType, data) => this._dispatch(eventType, data),
      onDone: () => {
        if (this.onDone) this.onDone();
      },
      onError: (err) => {
        if (this.onError) this.onError(err);
      },
    }, { signal: this.abortController.signal });
  }

  stop() {
    this.abortController.abort();
  }

  _dispatch(eventType, rawData) {
    if (eventType === 'done') {
      if (this.onDone) this.onDone(rawData);
    } else if (eventType === 'error') {
      if (this.onError) this.onError(new Error(rawData.message || rawData.text || 'Stream error'));
    }

    if (this.onEvent) this.onEvent(eventType, rawData);
  }
}

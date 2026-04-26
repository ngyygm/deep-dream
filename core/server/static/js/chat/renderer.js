/**
 * Chat message renderer — creates DOM elements for each message type.
 */
const ChatRenderer = {
  /**
   * Render a user message bubble (right-aligned).
   */
  renderUserMessage(text) {
    const wrap = document.createElement('div');
    wrap.className = 'chat-message chat-message-user';
    var sender = document.createElement('div');
    sender.className = 'chat-msg-sender';
    sender.textContent = t('chat.user') || 'You';
    var bubble = document.createElement('div');
    bubble.className = 'chat-msg-bubble';
    // Render markdown for user messages too (preserves line breaks, lists, etc.)
    if (text && typeof marked !== 'undefined') {
      bubble.innerHTML = marked.parse(text);
    } else {
      bubble.textContent = text;
    }
    wrap.appendChild(sender);
    wrap.appendChild(bubble);
    return wrap;
  },

  /**
   * Start an assistant message (left-aligned). Returns the content element
   * that can be incrementally populated.
   */
  renderAssistantMessage() {
    var wrap = document.createElement('div');
    wrap.className = 'chat-message chat-message-assistant';
    var sender = document.createElement('div');
    sender.className = 'chat-msg-sender';
    sender.textContent = t('chat.assistant') || 'Assistant';
    var bubble = document.createElement('div');
    bubble.className = 'chat-msg-bubble';
    wrap.appendChild(sender);
    wrap.appendChild(bubble);
    return { wrap: wrap, bubble: bubble };
  },

  /**
   * Render a system/status message (centered, left-border colored).
   */
  renderSystemMessage(text, type) {
    const wrap = document.createElement('div');
    wrap.className = 'chat-message chat-message-system' + (type === 'error' ? ' chat-message-error' : '');
    const bubble = document.createElement('div');
    bubble.className = 'chat-msg-bubble';
    bubble.textContent = text;
    wrap.appendChild(bubble);
    return wrap;
  },

  /**
   * Render a collapsible tool-call block with args + result.
   * Returns { wrap, setSuccess, setResult, toggle }.
   */
  renderToolCall(toolName, args, result, success) {
    const block = document.createElement('div');
    block.className = 'chat-tool-block';

    // Header
    const header = document.createElement('div');
    header.className = 'chat-tool-header';

    const chevron = document.createElement('span');
    chevron.className = 'chat-tool-chevron';
    chevron.textContent = '\u25B6'; // ▶

    const nameEl = document.createElement('span');
    nameEl.className = 'chat-tool-name';
    nameEl.textContent = toolName;

    const argsPreview = document.createElement('span');
    argsPreview.className = 'chat-tool-args-preview';
    argsPreview.textContent = ChatRenderer._summarizeArgs(args);

    const statusEl = document.createElement('span');
    statusEl.className = 'chat-tool-status ' + (success !== false ? 'chat-tool-status-ok' : 'chat-tool-status-err');
    statusEl.textContent = success !== false ? 'OK' : 'ERR';

    header.appendChild(chevron);
    header.appendChild(nameEl);
    header.appendChild(argsPreview);
    header.appendChild(statusEl);

    // Body
    const body = document.createElement('div');
    body.className = 'chat-tool-body';

    const argsLabel = document.createElement('div');
    argsLabel.className = 'chat-tool-section-label';
    argsLabel.textContent = 'Args';
    const argsPre = document.createElement('pre');
    argsPre.textContent = typeof args === 'string' ? args : JSON.stringify(args, null, 2);
    body.appendChild(argsLabel);
    body.appendChild(argsPre);

    if (result !== undefined && result !== null) {
      const resLabel = document.createElement('div');
      resLabel.className = 'chat-tool-section-label';
      resLabel.textContent = success !== false ? 'Result' : 'Error';
      const resPre = document.createElement('pre');
      resPre.textContent = typeof result === 'string' ? result : JSON.stringify(result, null, 2);
      body.appendChild(resLabel);
      body.appendChild(resPre);
    }

    block.appendChild(header);
    block.appendChild(body);

    // Toggle expand
    header.addEventListener('click', () => {
      block.classList.toggle('expanded');
    });

    return { wrap: block, statusEl, body };
  },

  /**
   * Render a dream cycle status indicator.
   */
  renderDreamStatus(cycle, totalCycles, strategy, stats) {
    const div = document.createElement('div');
    div.className = 'chat-dream-cycle';

    let html = '<span class="cycle-label">' +
      escapeHtml(t('chat.cycleLabel') || 'Cycle') + ' ' + cycle + '/' + totalCycles +
      '</span>';

    if (strategy) {
      html += ' <span style="color:var(--text-muted);">|</span> ' + escapeHtml(strategy);
    }

    if (stats) {
      const parts = [];
      if (stats.entities_examined) parts.push(stats.entities_examined + ' ' + (t('chat.entities') || 'entities'));
      if (stats.relations_saved) parts.push(stats.relations_saved + ' ' + (t('chat.relations') || 'relations'));
      if (stats.tool_calls_made) parts.push(stats.tool_calls_made + ' ' + (t('chat.toolCalls') || 'tool calls'));
      if (parts.length) {
        html += ' <span style="color:var(--text-muted);">|</span> ' + parts.join(', ');
      }
    }

    div.innerHTML = html;
    return div;
  },

  /**
   * Render final summary with markdown support.
   */
  renderSummary(text) {
    const div = document.createElement('div');
    div.className = 'chat-summary';
    if (typeof marked !== 'undefined') {
      div.innerHTML = marked.parse(text || '');
    } else {
      div.textContent = text || '';
    }
    return div;
  },

  /**
   * Append a text token to an element (for streaming).
   */
  appendToken(element, token) {
    element.textContent += token;
  },

  /**
   * Add streaming cursor class to element.
   */
  showStreamingCursor(element) {
    element.classList.add('streaming-cursor');
  },

  /**
   * Remove streaming cursor class from element.
   */
  hideStreamingCursor(element) {
    element.classList.remove('streaming-cursor');
  },

  // --- Helpers ---

  _summarizeArgs(args) {
    if (!args) return '';
    if (typeof args === 'string') {
      return args.length > 60 ? args.slice(0, 60) + '...' : args;
    }
    try {
      const s = JSON.stringify(args);
      return s.length > 80 ? s.slice(0, 80) + '...' : s;
    } catch {
      return '';
    }
  },
};

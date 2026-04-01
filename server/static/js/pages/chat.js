registerPage('chat', (function () {
  'use strict';

  let _container = null;
  let _messagesEl = null;
  let _inputEl = null;
  let _sendBtn = null;
  let _sseClient = null;
  let _isStreaming = false;
  let _currentAssistantEl = null;
  let _currentToolStatusEl = null;
  let _currentToolBodyEl = null;
  let _dreamRunning = false;

  // ---- Render ----

  async function render(container) {
    _container = container;

    container.innerHTML =
      '<div class="chat-page">' +
        '<div class="chat-toolbar">' +
          '<span class="chat-toolbar-title">DeepDream Chat</span>' +
          '<button id="chat-dream-btn" class="btn btn-primary btn-sm">' +
            '<i data-lucide="sparkles" style="width:14px;height:14px;"></i> ' +
            '<span>' + (t('chat.startDream') || 'Start Dream') + '</span>' +
          '</button>' +
          '<button id="chat-stop-btn" class="btn btn-stop btn-sm" style="display:none;">' +
            '<i data-lucide="square" style="width:14px;height:14px;"></i> ' +
            '<span>' + (t('chat.stop') || 'Stop') + '</span>' +
          '</button>' +
          '<button id="chat-clear-btn" class="btn btn-ghost btn-sm" title="' + (t('chat.clear') || 'Clear') + '">' +
            '<i data-lucide="trash-2" style="width:14px;height:14px;"></i>' +
          '</button>' +
        '</div>' +

        '<div id="chat-messages" class="chat-messages">' +
          _buildWelcomeHTML() +
        '</div>' +

        '<div class="chat-input-area">' +
          '<div class="chat-input-wrap">' +
            '<textarea id="chat-input" rows="1" placeholder="' +
              (t('chat.placeholder') || 'Ask a question or type /dream to start dreaming...') +
            '"></textarea>' +
          '</div>' +
          '<button id="chat-send-btn" class="chat-send-btn btn btn-primary">' +
            '<i data-lucide="send" style="width:16px;height:16px;"></i>' +
          '</button>' +
        '</div>' +
      '</div>';

    if (window.lucide) lucide.createIcons({ nodes: [container] });

    _messagesEl = container.querySelector('#chat-messages');
    _inputEl = container.querySelector('#chat-input');
    _sendBtn = container.querySelector('#chat-send-btn');

    // Bind events
    _sendBtn.addEventListener('click', _handleSend);
    _inputEl.addEventListener('keydown', function (e) {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        _handleSend();
      }
    });

    // Auto-resize textarea
    _inputEl.addEventListener('input', function () {
      _inputEl.style.height = 'auto';
      _inputEl.style.height = Math.min(_inputEl.scrollHeight, 160) + 'px';
    });

    container.querySelector('#chat-dream-btn').addEventListener('click', _handleDreamStart);
    container.querySelector('#chat-stop-btn').addEventListener('click', _handleStop);
    container.querySelector('#chat-clear-btn').addEventListener('click', _handleClear);

    _loadWelcomeStats();
  }

  function destroy() {
    if (_sseClient) {
      _sseClient.stop();
      _sseClient = null;
    }
    _container = null;
    _messagesEl = null;
    _inputEl = null;
  }

  // ---- Welcome ----

  function _buildWelcomeHTML() {
    return '<div id="chat-welcome" class="chat-welcome">' +
      '<div class="chat-welcome-icon"><i data-lucide="message-circle" style="width:24px;height:24px;color:var(--primary);"></i></div>' +
      '<h2>' + (t('chat.welcomeTitle') || 'DeepDream Chat') + '</h2>' +
      '<p>' + (t('chat.welcomeDesc') || 'Ask questions about your knowledge graph or start a dream session to discover hidden connections.') + '</p>' +
      '<div id="chat-welcome-stats" class="chat-welcome-stats"></div>' +
    '</div>';
  }

  function _hideWelcome() {
    var w = _container && _container.querySelector('#chat-welcome');
    if (w) w.style.display = 'none';
  }

  async function _loadWelcomeStats() {
    try {
      var res = await state.api.getCounts(state.currentGraphId);
      var d = res.data || {};
      var statsEl = _container && _container.querySelector('#chat-welcome-stats');
      if (!statsEl) return;
      statsEl.innerHTML =
        '<div class="chat-welcome-stat"><div class="stat-value">' + formatNumber(d.entity_count || 0) + '</div><div class="stat-label">' + (t('common.entities') || 'Entities') + '</div></div>' +
        '<div class="chat-welcome-stat"><div class="stat-value">' + formatNumber(d.relation_count || 0) + '</div><div class="stat-label">' + (t('common.relations') || 'Relations') + '</div></div>';
    } catch (e) { /* ignore */ }
  }

  // ---- Actions ----

  function _handleSend() {
    var text = (_inputEl.value || '').trim();
    if (!text || _isStreaming) return;

    _inputEl.value = '';
    _inputEl.style.height = 'auto';

    if (text === '/dream') { _handleDreamStart(); return; }
    if (text === '/stop')  { _handleStop(); return; }
    if (text === '/clear') { _handleClear(); return; }

    _hideWelcome();
    _addUserMessage(text);
    _streamAsk(text);
  }

  function _handleDreamStart() {
    if (_isStreaming || _dreamRunning) return;

    var configHtml =
      '<div class="chat-dream-config">' +
        '<label><span>' + (t('chat.dreamCycles') || 'Cycles') + '</span>' +
          '<input type="number" id="dream-cfg-cycles" value="3" min="1" max="50"></label>' +
        '<label><span>' + (t('chat.dreamStrategy') || 'Strategy Mode') + '</span>' +
          '<select id="dream-cfg-strategy">' +
            '<option value="round_robin">Round Robin</option>' +
            '<option value="random">Random</option>' +
          '</select></label>' +
        '<label><span>' + (t('chat.dreamMaxToolCalls') || 'Max Tool Calls / Cycle') + '</span>' +
          '<input type="number" id="dream-cfg-max-tools" value="15" min="1" max="50"></label>' +
      '</div>';

    showModal({
      title: t('chat.startDream') || 'Start Dream',
      content: configHtml,
      size: 'md',
    });

    // Add confirm/cancel buttons to the modal
    var overlay = document.querySelector('.modal-overlay:last-child');
    if (!overlay) return;

    var contentEl = overlay.querySelector('.modal-body') || overlay.querySelector('.modal-content');
    if (!contentEl) return;

    var actions = document.createElement('div');
    actions.style.cssText = 'display:flex;gap:0.5rem;justify-content:flex-end;margin-top:1rem;';
    actions.innerHTML =
      '<button class="btn btn-ghost" id="dream-cfg-cancel">' + (t('common.cancel') || 'Cancel') + '</button>' +
      '<button class="btn btn-primary" id="dream-cfg-confirm">' + (t('common.confirm') || 'Confirm') + '</button>';
    contentEl.appendChild(actions);

    overlay.querySelector('#dream-cfg-cancel').addEventListener('click', function () {
      overlay.remove();
    });
    overlay.querySelector('#dream-cfg-confirm').addEventListener('click', function () {
      var cycles = parseInt(overlay.querySelector('#dream-cfg-cycles').value) || 3;
      var strategyMode = overlay.querySelector('#dream-cfg-strategy').value;
      var maxTools = parseInt(overlay.querySelector('#dream-cfg-max-tools').value) || 15;
      overlay.remove();
      _startDream({
        max_cycles: cycles,
        strategy_mode: strategyMode,
        max_tool_calls_per_cycle: maxTools,
      });
    });
  }

  function _startDream(config) {
    _hideWelcome();
    _dreamRunning = true;
    _setStreaming(true);

    _addSystemMessage((t('chat.dreamStarted') || 'Dream session started') + '...');

    _sseClient = new DreamSSEClient(
      '/api/v1/find/dream/agent/stream',
      Object.assign({}, config, { graph_id: state.currentGraphId }),
    );

    _sseClient.onEvent = _handleDreamEvent;
    _sseClient.onDone = function () {
      _dreamRunning = false;
      _setStreaming(false);
      _sseClient = null;
    };
    _sseClient.onError = function (err) {
      _addSystemMessage(err.message, 'error');
      _dreamRunning = false;
      _setStreaming(false);
      _sseClient = null;
    };

    _sseClient.start();
  }

  function _streamAsk(question) {
    _setStreaming(true);

    _sseClient = new DreamSSEClient(
      '/api/v1/find/ask/stream',
      { question: question, graph_id: state.currentGraphId },
    );

    _sseClient.onEvent = _handleAskEvent;
    _sseClient.onDone = function () {
      _finishCurrentAssistant();
      _setStreaming(false);
      _sseClient = null;
    };
    _sseClient.onError = function (err) {
      _addSystemMessage(err.message, 'error');
      _finishCurrentAssistant();
      _setStreaming(false);
      _sseClient = null;
    };

    _sseClient.start();
  }

  // ---- Event Handlers ----

  function _handleDreamEvent(type, data) {
    switch (type) {
      case 'cycle_start':
        _finishCurrentAssistant();
        _appendDreamCycle(data.cycle, data.total_cycles, data.strategy);
        break;

      case 'thought':
        _appendThought(data.text || '');
        break;

      case 'tool_call':
        _appendToolCall(data.tool || 'unknown', data.arguments || {});
        break;

      case 'tool_result':
        _updateToolResult(data.success, data.data || data.error);
        break;

      case 'relation_created':
        _appendSystemLine(
          (t('chat.relationCreated') || 'New relation') + ': ' +
          (data.entity1_id || '?') + ' \u2194 ' + (data.entity2_id || '?')
        );
        break;

      case 'cycle_end':
        _finishCurrentAssistant();
        if (data) {
          _appendSystemLine(
            (t('chat.cycleComplete') || 'Cycle complete') + ': ' +
            (data.entities_examined || 0) + ' ' + (t('chat.entities') || 'entities') + ', ' +
            (data.relations_saved || 0) + ' ' + (t('chat.saved') || 'saved')
          );
        }
        break;

      case 'episode_saved':
        if (data && data.episode_content) {
          _appendSystemLine((t('chat.episodeSaved') || 'Episode saved'));
        }
        break;

      case 'summary':
        _finishCurrentAssistant();
        _appendSummary(data.narrative || data.text || '');
        break;
    }
  }

  function _handleAskEvent(type, data) {
    switch (type) {
      case 'thought':
        if (data.text) {
          _appendThought(data.text);
        }
        break;

      case 'tool_call':
        _appendToolCall(data.tool || 'search', data.arguments || {});
        break;

      case 'tool_result':
        _updateToolResult(data.success, data.data || data.error);
        break;

      case 'summary':
        _finishCurrentAssistant();
        _appendSummary(data.answer || data.text || '');
        break;
    }
  }

  // ---- DOM Construction ----

  function _addUserMessage(text) {
    var el = ChatRenderer.renderUserMessage(text);
    _messagesEl.appendChild(el);
    _scrollToBottom();
  }

  function _addSystemMessage(text, type) {
    _finishCurrentAssistant();
    var el = ChatRenderer.renderSystemMessage(text, type);
    _messagesEl.appendChild(el);
    _scrollToBottom();
  }

  function _appendThought(text) {
    if (!_currentAssistantEl) {
      var result = ChatRenderer.renderAssistantMessage();
      _messagesEl.appendChild(result.wrap);
      _currentAssistantEl = result.bubble;
      ChatRenderer.showStreamingCursor(_currentAssistantEl);
    }
    ChatRenderer.appendToken(_currentAssistantEl, text);
    _scrollToBottom();
  }

  function _appendToolCall(toolName, args) {
    _finishCurrentAssistant();

    var block = document.createElement('div');
    block.className = 'chat-tool-block';

    // Header
    var header = document.createElement('div');
    header.className = 'chat-tool-header';

    var chevron = document.createElement('span');
    chevron.className = 'chat-tool-chevron';
    chevron.textContent = '\u25B6';

    var nameEl = document.createElement('span');
    nameEl.className = 'chat-tool-name';
    nameEl.textContent = toolName;

    var argsPreview = document.createElement('span');
    argsPreview.className = 'chat-tool-args-preview';
    argsPreview.textContent = ChatRenderer._summarizeArgs(args);

    var statusEl = document.createElement('span');
    statusEl.className = 'chat-tool-status chat-tool-status-ok';
    statusEl.textContent = '...';

    header.appendChild(chevron);
    header.appendChild(nameEl);
    header.appendChild(argsPreview);
    header.appendChild(statusEl);

    // Body
    var body = document.createElement('div');
    body.className = 'chat-tool-body';

    var argsLabel = document.createElement('div');
    argsLabel.className = 'chat-tool-section-label';
    argsLabel.textContent = 'Args';
    var argsPre = document.createElement('pre');
    argsPre.textContent = typeof args === 'string' ? args : JSON.stringify(args, null, 2);
    body.appendChild(argsLabel);
    body.appendChild(argsPre);

    block.appendChild(header);
    block.appendChild(body);

    header.addEventListener('click', function () {
      block.classList.toggle('expanded');
    });

    _messagesEl.appendChild(block);
    _currentToolStatusEl = statusEl;
    _currentToolBodyEl = body;
    _scrollToBottom();
  }

  function _updateToolResult(success, data) {
    if (_currentToolStatusEl) {
      _currentToolStatusEl.className = 'chat-tool-status ' + (success ? 'chat-tool-status-ok' : 'chat-tool-status-err');
      _currentToolStatusEl.textContent = success ? 'OK' : 'ERR';

      var resLabel = document.createElement('div');
      resLabel.className = 'chat-tool-section-label';
      resLabel.textContent = success ? 'Result' : 'Error';

      var resPre = document.createElement('pre');
      var resultText = typeof data === 'string' ? data : JSON.stringify(data, null, 2);
      resPre.textContent = resultText.length > 2000 ? resultText.slice(0, 2000) + '\n...' : resultText;

      _currentToolBodyEl.appendChild(resLabel);
      _currentToolBodyEl.appendChild(resPre);

      _currentToolStatusEl = null;
      _currentToolBodyEl = null;
      _scrollToBottom();
    }
  }

  function _appendDreamCycle(cycle, totalCycles, strategy) {
    var div = document.createElement('div');
    div.className = 'chat-dream-cycle';

    var html = '<span class="cycle-label">' +
      escapeHtml(t('chat.cycleLabel') || 'Cycle') + ' ' + cycle + '/' + totalCycles +
      '</span>';

    if (strategy) {
      html += ' <span style="color:var(--text-muted);">|</span> ' + escapeHtml(strategy);
    }

    div.innerHTML = html;
    _messagesEl.appendChild(div);
    _scrollToBottom();
  }

  function _appendSystemLine(text) {
    var el = ChatRenderer.renderSystemMessage(text);
    _messagesEl.appendChild(el);
    _scrollToBottom();
  }

  function _appendSummary(text) {
    var el = ChatRenderer.renderSummary(text);
    _messagesEl.appendChild(el);
    _scrollToBottom();
  }

  function _finishCurrentAssistant() {
    if (_currentAssistantEl) {
      ChatRenderer.hideStreamingCursor(_currentAssistantEl);
      var raw = _currentAssistantEl.textContent;
      if (raw && typeof marked !== 'undefined') {
        _currentAssistantEl.innerHTML = marked.parse(raw);
      }
      _currentAssistantEl = null;
    }
    _currentToolStatusEl = null;
    _currentToolBodyEl = null;
  }

  // ---- UI State ----

  function _scrollToBottom() {
    if (!_messagesEl) return;
    requestAnimationFrame(function () {
      _messagesEl.scrollTop = _messagesEl.scrollHeight;
    });
  }

  function _setStreaming(val) {
    _isStreaming = val;
    if (_sendBtn) _sendBtn.disabled = val;
    if (_inputEl) _inputEl.disabled = val;

    var stopBtn = _container && _container.querySelector('#chat-stop-btn');
    if (stopBtn) stopBtn.style.display = val ? '' : 'none';

    var dreamBtn = _container && _container.querySelector('#chat-dream-btn');
    if (dreamBtn) dreamBtn.disabled = val || _dreamRunning;
  }

  function _handleStop() {
    if (_sseClient) {
      _sseClient.stop();
      _sseClient = null;
    }
    _finishCurrentAssistant();
    _addSystemMessage(t('chat.streamStopped') || 'Stream stopped');
    _dreamRunning = false;
    _setStreaming(false);
  }

  function _handleClear() {
    _currentAssistantEl = null;
    _currentToolStatusEl = null;
    _currentToolBodyEl = null;
    if (_messagesEl) {
      _messagesEl.innerHTML = _buildWelcomeHTML();
      if (window.lucide) lucide.createIcons({ nodes: [_messagesEl] });
      _loadWelcomeStats();
    }
  }

  return { render: render, destroy: destroy };
})());

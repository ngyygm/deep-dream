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

  // Session state
  let _sessions = [];           // [{session_id, title, graph_id, status, ...}]
  let _activeSessionId = null;  // currently active session
  let _sidebarEl = null;

  // ---- localStorage message persistence ----
  var _CHAT_STORAGE_PREFIX = 'deepdream_chat_';

  function _getSessionStorageKey(sid) {
    return _CHAT_STORAGE_PREFIX + sid;
  }

  function _saveSessionHistory(sid) {
    if (!_messagesEl) return;
    var messages = [];
    var children = _messagesEl.children;
    for (var i = 0; i < children.length; i++) {
      var el = children[i];
      if (el.classList.contains('chat-welcome')) continue;
      if (el.classList.contains('chat-message-user')) {
        var bubble = el.querySelector('.chat-msg-bubble');
        // Prefer raw markdown from data-raw attribute, fallback to textContent
        var content = bubble ? (bubble.getAttribute('data-raw') || bubble.textContent) : '';
        messages.push({ role: 'user', content: content });
      } else if (el.classList.contains('chat-message-assistant')) {
        var bubble2 = el.querySelector('.chat-msg-bubble');
        var content2 = bubble2 ? (bubble2.getAttribute('data-raw') || bubble2.textContent) : '';
        messages.push({ role: 'assistant', content: content2 });
      } else if (el.classList.contains('chat-message-system')) {
        var bubble3 = el.querySelector('.chat-msg-bubble');
        if (bubble3) messages.push({ role: 'system', content: bubble3.textContent });
      } else if (el.classList.contains('chat-tool-block')) {
        var nameEl = el.querySelector('.chat-tool-name');
        var argsPre = el.querySelector('.chat-tool-body pre');
        var resultPre = el.querySelectorAll('.chat-tool-body pre');
        var resultText = resultPre.length > 1 ? resultPre[1].textContent : '';
        var statusEl = el.querySelector('.chat-tool-status');
        messages.push({
          role: 'tool',
          name: nameEl ? nameEl.textContent : 'tool',
          args: argsPre ? argsPre.textContent : '',
          result: resultText,
          success: statusEl ? statusEl.classList.contains('chat-tool-status-ok') : true,
        });
      } else if (el.classList.contains('chat-dream-cycle')) {
        messages.push({ role: 'dream_cycle', content: el.textContent });
      } else if (el.classList.contains('chat-summary')) {
        messages.push({ role: 'summary', content: el.getAttribute('data-raw') || el.textContent });
      }
    }
    try {
      localStorage.setItem(_getSessionStorageKey(sid), JSON.stringify(messages));
    } catch (e) { /* storage full — ignore */ }
  }

  function _loadSessionHistory(sid) {
    try {
      var raw = localStorage.getItem(_getSessionStorageKey(sid));
      if (!raw) return [];
      return JSON.parse(raw);
    } catch (e) { return []; }
  }

  function _removeSessionHistory(sid) {
    try { localStorage.removeItem(_getSessionStorageKey(sid)); } catch (e) { /* ignore */ }
  }

  function _renderHistoryMessages(messages) {
    for (var i = 0; i < messages.length; i++) {
      var msg = messages[i];
      if (msg.role === 'user') {
        var userEl = ChatRenderer.renderUserMessage(msg.content);
        // Store raw markdown for re-saving
        var userBubble = userEl.querySelector('.chat-msg-bubble');
        if (userBubble && msg.content) userBubble.setAttribute('data-raw', msg.content);
        _messagesEl.appendChild(userEl);
      } else if (msg.role === 'assistant') {
        var result = ChatRenderer.renderAssistantMessage();
        if (msg.content && typeof marked !== 'undefined') {
          result.bubble.innerHTML = marked.parse(msg.content);
          result.bubble.setAttribute('data-raw', msg.content);
        } else {
          result.bubble.textContent = msg.content;
        }
        _messagesEl.appendChild(result.wrap);
      } else if (msg.role === 'system') {
        _messagesEl.appendChild(ChatRenderer.renderSystemMessage(msg.content));
      } else if (msg.role === 'tool') {
        var toolResult = ChatRenderer.renderToolCall(
          msg.name || 'tool',
          msg.args || '',
          msg.result || undefined,
          msg.success !== false
        );
        _messagesEl.appendChild(toolResult.wrap);
      } else if (msg.role === 'dream_cycle') {
        var div = document.createElement('div');
        div.className = 'chat-dream-cycle';
        div.textContent = msg.content;
        _messagesEl.appendChild(div);
      } else if (msg.role === 'summary') {
        var sumEl = ChatRenderer.renderSummary(msg.content);
        if (msg.content) sumEl.setAttribute('data-raw', msg.content);
        _messagesEl.appendChild(sumEl);
      }
    }
    _scrollToBottom();
  }

  // ---- API helpers ----

  async function _api(method, path, body) {
    var opts = { method: method, headers: { 'Content-Type': 'application/json' } };
    if (body !== undefined) opts.body = JSON.stringify(body);
    var res = await fetch('/api/v1/chat' + path, opts);
    var json = await res.json();
    if (!res.ok) throw new Error((json.data && json.data.error) || json.error || 'Request failed');
    return json.data;
  }

  // ---- Session management ----

  async function _loadSessions() {
    try {
      _sessions = await _api('GET', '/sessions?include_closed=0') || [];
    } catch (e) {
      _sessions = [];
    }
    _renderSidebar();
  }

  async function _createSession() {
    try {
      var session = await _api('POST', '/sessions', {
        graph_id: state.currentGraphId || 'default',
        title: null,
      });
      _activeSessionId = session.session_id;
      await _loadSessions();
      _clearMessages();
      _hideWelcome();
      _inputEl.focus();
    } catch (e) {
      _addSystemMessage('Failed to create session: ' + e.message, 'error');
    }
  }

  async function _switchSession(sid) {
    if (_isStreaming || _dreamRunning) return;
    // Save current session history before switching
    if (_activeSessionId) {
      _saveSessionHistory(_activeSessionId);
    }
    if (sid === _activeSessionId) return;
    _activeSessionId = sid;
    _clearMessages();
    _hideWelcome();
    // Load history from localStorage
    var history = _loadSessionHistory(sid);
    if (history.length) {
      _renderHistoryMessages(history);
    } else {
      _showWelcome();
    }
    _renderSidebar();
    _inputEl.focus();
  }

  async function _deleteSession(sid) {
    if (_isStreaming || _dreamRunning) return;
    var session = _sessions.find(function (s) { return s.session_id === sid; });
    var title = session ? (session.title || sid.slice(0, 8)) : sid.slice(0, 8);
    showModal({
      title: t('chat.deleteTitle') || 'Delete Session',
      content: '<p>' + escapeHtml(t('chat.deleteConfirm') || 'Are you sure you want to delete this session? This cannot be undone.') + '</p>',
      size: 'sm',
    });
    var overlay = document.querySelector('.modal-overlay:last-child');
    if (!overlay) return;
    var contentEl = overlay.querySelector('.modal-body') || overlay.querySelector('.modal-content');
    if (!contentEl) return;
    var actions = document.createElement('div');
    actions.style.cssText = 'display:flex;gap:0.5rem;justify-content:flex-end;margin-top:1rem;';
    actions.innerHTML =
      '<button class="btn btn-ghost" id="del-cancel">' + (t('common.cancel') || 'Cancel') + '</button>' +
      '<button class="btn btn-primary" id="del-confirm" style="background:var(--error,#e53e3e);border-color:var(--error,#e53e3e);">' + (t('common.confirm') || 'Confirm') + '</button>';
    contentEl.appendChild(actions);
    overlay.querySelector('#del-cancel').addEventListener('click', function () { overlay.remove(); });
    overlay.querySelector('#del-confirm').addEventListener('click', function () {
      overlay.remove();
      _doDeleteSession(sid);
    });
  }

  async function _doDeleteSession(sid) {
    try {
      await _api('DELETE', '/sessions/' + sid);
      _removeSessionHistory(sid);
      if (_activeSessionId === sid) {
        _activeSessionId = null;
        _clearMessages();
        _showWelcome();
      }
      await _loadSessions();
    } catch (e) { /* ignore */ }
  }

  async function _closeSession(sid) {
    if (_isStreaming || _dreamRunning) return;
    // Save history before closing so localStorage backup is preserved
    if (sid === _activeSessionId) _saveSessionHistory(sid);
    try {
      await _api('POST', '/sessions/' + sid + '/close');
      await _loadSessions();
    } catch (e) { /* ignore */ }
  }

  // ---- Render ----

  async function render(container) {
    _container = container;

    container.innerHTML =
      '<div class="chat-layout">' +
        '<div class="chat-sidebar" id="chat-sidebar">' +
          '<div class="chat-sidebar-header">' +
            '<span class="chat-sidebar-title">' + t('chat.sessions') + '</span>' +
            '<button id="chat-new-session-btn" class="btn btn-primary btn-sm" title="' + t('chat.newSession') + '">' +
              '<i data-lucide="plus" style="width:14px;height:14px;"></i>' +
            '</button>' +
          '</div>' +
          '<div class="chat-sidebar-list" id="chat-sidebar-list"></div>' +
        '</div>' +
        '<div class="chat-main">' +
          '<div class="chat-toolbar">' +
            '<span class="chat-toolbar-title" id="chat-toolbar-title">' + t('chat.welcomeTitle') + '</span>' +
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
                (t('chat.placeholder') || 'Type a message... (auto-creates a session)') +
              '"></textarea>' +
            '</div>' +
            '<button id="chat-send-btn" class="chat-send-btn btn btn-primary">' +
              '<i data-lucide="send" style="width:16px;height:16px;"></i>' +
            '</button>' +
          '</div>' +
        '</div>' +
      '</div>';

    if (window.lucide) lucide.createIcons({ nodes: [container] });

    _messagesEl = container.querySelector('#chat-messages');
    _inputEl = container.querySelector('#chat-input');
    _sendBtn = container.querySelector('#chat-send-btn');
    _sidebarEl = container.querySelector('#chat-sidebar-list');

    // Bind events
    _sendBtn.addEventListener('click', _handleSend);
    _inputEl.addEventListener('keydown', function (e) {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        _handleSend();
      }
    });

    _inputEl.addEventListener('input', function () {
      _inputEl.style.height = 'auto';
      _inputEl.style.height = Math.min(_inputEl.scrollHeight, 160) + 'px';
    });

    container.querySelector('#chat-dream-btn').addEventListener('click', _handleDreamStart);
    container.querySelector('#chat-stop-btn').addEventListener('click', _handleStop);
    container.querySelector('#chat-clear-btn').addEventListener('click', _handleClear);
    container.querySelector('#chat-new-session-btn').addEventListener('click', _createSession);

    await _loadSessions();
    _loadWelcomeStats();
  }

  function destroy() {
    // Save current session history before leaving
    if (_activeSessionId) _saveSessionHistory(_activeSessionId);
    if (_sseClient) {
      _sseClient.stop();
      _sseClient = null;
    }
    _container = null;
    _messagesEl = null;
    _inputEl = null;
    _sidebarEl = null;
  }

  // ---- Sidebar rendering ----

  function _renderSidebar() {
    if (!_sidebarEl) return;

    var html = '';
    for (var i = 0; i < _sessions.length; i++) {
      var s = _sessions[i];
      var active = s.session_id === _activeSessionId ? ' active' : '';
      var title = escapeHtml(s.title || (t('chat.sessionLabel') + ' ' + s.session_id.slice(0, 8)));
      html += '<div class="chat-sidebar-item' + active + '" data-sid="' + s.session_id + '">' +
        '<span class="chat-sidebar-item-title" title="' + escapeHtml(s.title || '') + '">' + title + '</span>' +
        '<div class="chat-sidebar-item-actions">' +
          '<button class="chat-sidebar-btn chat-sidebar-rename" data-sid="' + s.session_id + '" title="' + (t('chat.rename') || 'Rename') + '">' +
            '<i data-lucide="pencil" style="width:12px;height:12px;"></i>' +
          '</button>' +
          '<button class="chat-sidebar-btn chat-sidebar-close" data-sid="' + s.session_id + '" title="' + t('chat.close') + '">' +
            '<i data-lucide="x" style="width:12px;height:12px;"></i>' +
          '</button>' +
          '<button class="chat-sidebar-btn chat-sidebar-delete" data-sid="' + s.session_id + '" title="' + (t('chat.deleteTitle') || 'Delete') + '">' +
            '<i data-lucide="trash-2" style="width:12px;height:12px;"></i>' +
          '</button>' +
        '</div>' +
      '</div>';
    }

    if (!_sessions.length) {
      html = '<div class="chat-sidebar-empty">' + t('chat.noSessions') + '</div>';
    }

    _sidebarEl.innerHTML = html;
    if (window.lucide) lucide.createIcons({ nodes: [_sidebarEl] });

    // Bind click events
    var items = _sidebarEl.querySelectorAll('.chat-sidebar-item');
    items.forEach(function (item) {
      item.addEventListener('click', function (e) {
        // Don't switch if clicking action button
        if (e.target.closest('.chat-sidebar-item-actions')) return;
        _switchSession(item.dataset.sid);
      });
      // Double-click title to rename inline
      var titleSpan = item.querySelector('.chat-sidebar-item-title');
      if (titleSpan) {
        titleSpan.addEventListener('dblclick', function (e) {
          e.stopPropagation();
          _startRename(item.dataset.sid, titleSpan);
        });
      }
    });

    var renameBtns = _sidebarEl.querySelectorAll('.chat-sidebar-rename');
    renameBtns.forEach(function (btn) {
      btn.addEventListener('click', function (e) {
        e.stopPropagation();
        var titleSpan = btn.closest('.chat-sidebar-item').querySelector('.chat-sidebar-item-title');
        if (titleSpan) _startRename(btn.dataset.sid, titleSpan);
      });
    });

    var closeBtns = _sidebarEl.querySelectorAll('.chat-sidebar-close');
    closeBtns.forEach(function (btn) {
      btn.addEventListener('click', function (e) {
        e.stopPropagation();
        _closeSession(btn.dataset.sid);
      });
    });

    var deleteBtns = _sidebarEl.querySelectorAll('.chat-sidebar-delete');
    deleteBtns.forEach(function (btn) {
      btn.addEventListener('click', function (e) {
        e.stopPropagation();
        _deleteSession(btn.dataset.sid);
      });
    });

    // Update toolbar title
    var titleEl = _container && _container.querySelector('#chat-toolbar-title');
    if (titleEl) {
      var activeSession = _sessions.find(function (s) { return s.session_id === _activeSessionId; });
      titleEl.textContent = activeSession
        ? (activeSession.title || (t('chat.sessionLabel') + ' ' + activeSession.session_id.slice(0, 8)))
        : t('chat.welcomeTitle');
    }
  }

  function _startRename(sid, titleSpan) {
    var session = _sessions.find(function (s) { return s.session_id === sid; });
    if (!session) return;
    var currentTitle = session.title || '';

    var input = document.createElement('input');
    input.type = 'text';
    input.className = 'chat-sidebar-rename-input';
    input.value = currentTitle;
    input.maxLength = 100;

    titleSpan.style.display = 'none';
    titleSpan.parentNode.insertBefore(input, titleSpan);
    input.focus();
    input.select();

    function finishRename() {
      var newTitle = input.value.trim();
      input.remove();
      titleSpan.style.display = '';
      if (newTitle && newTitle !== currentTitle) {
        _doRenameSession(sid, newTitle);
      }
    }

    input.addEventListener('blur', finishRename);
    input.addEventListener('keydown', function (e) {
      if (e.key === 'Enter') { e.preventDefault(); input.blur(); }
      if (e.key === 'Escape') { input.value = currentTitle; input.blur(); }
    });
  }

  async function _doRenameSession(sid, newTitle) {
    try {
      await _api('PUT', '/sessions/' + sid, { title: newTitle });
      // Update local session data
      var session = _sessions.find(function (s) { return s.session_id === sid; });
      if (session) session.title = newTitle;
      _renderSidebar();
    } catch (e) { /* ignore */ }
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

  function _showWelcome() {
    if (!_messagesEl) return;
    _messagesEl.innerHTML = _buildWelcomeHTML();
    if (window.lucide) lucide.createIcons({ nodes: [_messagesEl] });
    _loadWelcomeStats();
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

  async function _handleSend() {
    var text = (_inputEl.value || '').trim();
    if (!text || _isStreaming) return;

    _inputEl.value = '';
    _inputEl.style.height = 'auto';

    if (text === '/dream') { _handleDreamStart(); return; }
    if (text === '/stop')  { _handleStop(); return; }
    if (text === '/clear') { _handleClear(); return; }

    _hideWelcome();

    // Auto-create session if none active
    if (!_activeSessionId) {
      try {
        var session = await _api('POST', '/sessions', {
          graph_id: state.currentGraphId || 'default',
          title: text.slice(0, 50),
        });
        _activeSessionId = session.session_id;
        await _loadSessions();
      } catch (e) {
        _addSystemMessage('Failed to create session: ' + e.message, 'error');
        return;
      }
    }

    _addUserMessage(text);
    _streamChat(text);
  }

  async function _handleDreamStart() {
    if (_isStreaming || _dreamRunning) return;

    // Auto-create session if none active
    if (!_activeSessionId) {
      try {
        var session = await _api('POST', '/sessions', {
          graph_id: state.currentGraphId || 'default',
          title: t('chat.startDream') || 'Dream Session',
        });
        _activeSessionId = session.session_id;
        await _loadSessions();
        _hideWelcome();
      } catch (e) {
        _addSystemMessage('Failed to create session: ' + e.message, 'error');
        return;
      }
    }

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
    if (!_activeSessionId) return;
    _dreamRunning = true;
    _setStreaming(true);

    _addSystemMessage((t('chat.dreamStarted') || 'Dream session started') + '...');

    // Send dream command as a chat message
    var dreamMessage = '/dream ' + JSON.stringify(config);

    _sseClient = new DreamSSEClient(
      '/api/v1/chat/sessions/' + _activeSessionId + '/stream',
      { message: dreamMessage },
    );

    _sseClient.onEvent = _handleDreamEvent;
    _sseClient.onDone = function () {
      _dreamRunning = false;
      _setStreaming(false);
      _sseClient = null;
      if (_activeSessionId) _saveSessionHistory(_activeSessionId);
    };
    _sseClient.onError = function (err) {
      _addSystemMessage(err.message, 'error');
      _dreamRunning = false;
      _setStreaming(false);
      _sseClient = null;
      if (_activeSessionId) _saveSessionHistory(_activeSessionId);
    };

    _sseClient.start();
  }

  function _streamChat(text) {
    if (!_activeSessionId) return;
    _setStreaming(true);

    _sseClient = new DreamSSEClient(
      '/api/v1/chat/sessions/' + _activeSessionId + '/stream',
      { message: text },
    );

    _sseClient.onEvent = _handleStreamEvent;
    _sseClient.onDone = function () {
      _finishCurrentAssistant();
      _setStreaming(false);
      _sseClient = null;
      if (_activeSessionId) _saveSessionHistory(_activeSessionId);
    };
    _sseClient.onError = function (err) {
      _addSystemMessage(err.message, 'error');
      _finishCurrentAssistant();
      _setStreaming(false);
      _sseClient = null;
      if (_activeSessionId) _saveSessionHistory(_activeSessionId);
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
        _appendToolCall(data.tool || 'unknown', data.args || data.arguments || {});
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

  function _handleStreamEvent(type, data) {
    switch (type) {
      case 'text':
        // Main text content from claude
        _appendThought(data.content || '');
        break;

      case 'tool_call':
        _appendToolCall(data.tool || 'unknown', data.args || data.arguments || {});
        break;

      case 'tool_result':
        var success = !data.is_error;
        var resultData = data.result || data.content || '';
        _updateToolResult(success, resultData);
        break;

      case 'system':
        // Skip system init events (already filtered server-side, but be safe)
        if (data && data.subtype === 'init') break;
        // Show only a brief label for other system events
        if (data && data.message) {
          _appendSystemLine(data.message);
        }
        break;

      case 'error':
        _addSystemMessage(data.error || 'Stream error', 'error');
        break;

      // Legacy dream events (if skill emits them)
      case 'thought':
        if (data.text) _appendThought(data.text);
        break;

      case 'summary':
        _finishCurrentAssistant();
        _appendSummary(data.answer || data.text || data.content || '');
        break;
    }
  }

  // ---- DOM Construction ----

  function _clearMessages() {
    _currentAssistantEl = null;
    _currentToolStatusEl = null;
    _currentToolBodyEl = null;
    if (_messagesEl) _messagesEl.innerHTML = '';
  }

  function _addUserMessage(text) {
    var el = ChatRenderer.renderUserMessage(text);
    // Store raw markdown for history persistence
    var bubble = el.querySelector('.chat-msg-bubble');
    if (bubble) bubble.setAttribute('data-raw', text);
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

    // Use ChatRenderer for consistent DOM structure, override status to "..."
    var result = ChatRenderer.renderToolCall(toolName, args, undefined, true);
    result.statusEl.textContent = '...';

    _messagesEl.appendChild(result.wrap);
    _currentToolStatusEl = result.statusEl;
    _currentToolBodyEl = result.body;
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
    var div = ChatRenderer.renderDreamStatus(cycle, totalCycles, strategy);
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
    if (text) el.setAttribute('data-raw', text);
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
      // Store raw markdown for history persistence
      if (raw) _currentAssistantEl.setAttribute('data-raw', raw);
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
    if (_activeSessionId) _saveSessionHistory(_activeSessionId);
  }

  function _handleClear() {
    if (_isStreaming || _dreamRunning) return;
    // Check if there are actual messages (beyond welcome)
    var hasMessages = false;
    if (_messagesEl) {
      var children = _messagesEl.children;
      for (var i = 0; i < children.length; i++) {
        if (!children[i].classList.contains('chat-welcome')) { hasMessages = true; break; }
      }
    }
    if (!hasMessages) return;

    showModal({
      title: t('chat.clear') || 'Clear',
      content: '<p>' + escapeHtml(t('chat.clearConfirm') || 'Clear all messages in this session? The session itself will be kept.') + '</p>',
      size: 'sm',
    });
    var overlay = document.querySelector('.modal-overlay:last-child');
    if (!overlay) return;
    var contentEl = overlay.querySelector('.modal-body') || overlay.querySelector('.modal-content');
    if (!contentEl) return;
    var actions = document.createElement('div');
    actions.style.cssText = 'display:flex;gap:0.5rem;justify-content:flex-end;margin-top:1rem;';
    actions.innerHTML =
      '<button class="btn btn-ghost" id="clear-cancel">' + (t('common.cancel') || 'Cancel') + '</button>' +
      '<button class="btn btn-primary" id="clear-confirm">' + (t('common.confirm') || 'Confirm') + '</button>';
    contentEl.appendChild(actions);
    overlay.querySelector('#clear-cancel').addEventListener('click', function () { overlay.remove(); });
    overlay.querySelector('#clear-confirm').addEventListener('click', function () {
      overlay.remove();
      _currentAssistantEl = null;
      _currentToolStatusEl = null;
      _currentToolBodyEl = null;
      if (_messagesEl) {
        _messagesEl.innerHTML = _buildWelcomeHTML();
        if (window.lucide) lucide.createIcons({ nodes: [_messagesEl] });
        _loadWelcomeStats();
      }
      // Save empty history after clearing (overwrites old messages in localStorage)
      if (_activeSessionId) _saveSessionHistory(_activeSessionId);
    });
  }

  return { render: render, destroy: destroy };
})());

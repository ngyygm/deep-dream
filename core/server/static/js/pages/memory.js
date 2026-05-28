/* ==========================================
   Memory Management Page
   Upload text/files, monitor tasks, browse docs
   ========================================== */

(function() {
  // ---- Smart refresh state ----
  let _hasActiveTasks = false;

  // ---- Helpers ----

  function progressClass(status) {
    if (status === 'completed') return 'success';
    if (status === 'failed') return 'error';
    return '';
  }

  function formatDocSize(bytes) {
    const n = Number(bytes || 0);
    if (!Number.isFinite(n) || n <= 0) return '0 B';
    if (n < 1024) return `${n} B`;
    if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
    if (n < 1024 * 1024 * 1024) return `${(n / (1024 * 1024)).toFixed(1)} MB`;
    return `${(n / (1024 * 1024 * 1024)).toFixed(1)} GB`;
  }

  // ---- Upload Section ----

  function renderUploadSection() {
    return `
      <div class="card" style="margin-bottom:1rem;">
        <div class="card-header">
          <span class="card-title">${t('memory.addMemory')}</span>
        </div>
        <div class="tabs" id="upload-tabs">
          <div class="tab active" data-tab="text">${t('memory.textInput')}</div>
          <div class="tab" data-tab="file">${t('memory.fileUpload')}</div>
        </div>

        <!-- Text Input Tab -->
        <div id="upload-tab-text">
          <textarea class="input" id="memory-text" placeholder="${t('memory.textPlaceholder')}" style="min-height:200px;"></textarea>
          <div id="text-counter" style="display:flex;justify-content:flex-end;font-size:0.75rem;color:var(--text-muted);padding:0.25rem 0.5rem 0;">${t('memory.charCount') || 'Characters'}: 0</div>
          <div style="display:flex;gap:1rem;align-items:flex-end;margin-top:0.75rem;flex-wrap:wrap;">
            <div style="flex:1;min-width:180px;">
              <label class="form-label">${t('memory.sourceName')}</label>
              <input class="input" type="text" id="text-source-name" placeholder="${t('memory.sourcePlaceholder')}">
            </div>
            <div style="flex:1;min-width:200px;">
              <label class="form-label">${t('memory.eventTime')}</label>
              <input class="input" type="datetime-local" id="text-event-time">
            </div>
            <div style="display:flex;align-items:center;gap:0.5rem;padding-bottom:2px;">
              <div class="toggle active" id="text-load-cache-toggle">
                <input type="checkbox" id="text-load-cache-input" checked>
              </div>
              <label class="form-label" style="margin:0;cursor:pointer;" for="text-load-cache-input">${t('memory.loadCache')}</label>
            </div>
            <div style="display:flex;align-items:flex-end;">
              <button class="btn btn-primary" id="btn-submit-text">
                <i data-lucide="send" style="width:16px;height:16px;"></i>
                ${t('memory.submitMemory')}
              </button>
            </div>
          </div>
        </div>

        <!-- File Upload Tab -->
        <div id="upload-tab-file" style="display:none;">
          <div class="drop-zone" id="file-drop-zone">
            <i data-lucide="upload-cloud" style="width:40px;height:40px;color:var(--text-muted);margin-bottom:0.5rem;"></i>
            <p style="color:var(--text-secondary);margin:0 0 0.25rem;">${t('memory.dragDrop')}</p>
            <div id="file-list-area">
              <p style="color:var(--text-muted);font-size:0.75rem;margin:0;" id="file-status-text">${t('memory.noFiles')}</p>
            </div>
            <input type="file" id="file-input" multiple accept=".md,.markdown,.txt,.text,.json,.html,.htm,.csv,.log,.pdf,.doc,.docx" style="display:none;">
          </div>
          <div style="display:flex;gap:1rem;align-items:flex-end;margin-top:0.75rem;flex-wrap:wrap;">
            <div style="flex:1;min-width:180px;">
              <label class="form-label">${t('memory.sourceName')}</label>
              <input class="input" type="text" id="file-source-name" placeholder="${t('memory.sourcePlaceholder')}">
            </div>
            <div style="flex:1;min-width:200px;">
              <label class="form-label">${t('memory.eventTime')}</label>
              <input class="input" type="datetime-local" id="file-event-time">
            </div>
            <div style="display:flex;align-items:center;gap:0.5rem;padding-bottom:2px;">
              <div class="toggle active" id="file-load-cache-toggle">
                <input type="checkbox" id="file-load-cache-input" checked>
              </div>
              <label class="form-label" style="margin:0;cursor:pointer;" for="file-load-cache-input">${t('memory.loadCache')}</label>
            </div>
            <button class="btn btn-secondary btn-sm" id="btn-clear-files" style="display:none;">
              <i data-lucide="x" style="width:14px;height:14px;"></i>
              ${t('memory.clearFiles')}
            </button>
            <button class="btn btn-primary" id="btn-submit-file" disabled>
              <i data-lucide="upload" style="width:16px;height:16px;"></i>
              ${t('memory.uploadProcess')}
            </button>
          </div>
        </div>
      </div>
    `;
  }

  function bindUploadEvents() {
    function bindLoadCacheToggle(toggleId, inputId) {
      const toggle = document.getElementById(toggleId);
      const input = document.getElementById(inputId);
      if (!toggle || !input) return;

      const sync = () => {
        toggle.classList.toggle('active', !!input.checked);
      };

      toggle.addEventListener('click', (e) => {
        if (e.target === input) return;
        input.checked = !input.checked;
        sync();
      });
      input.addEventListener('change', sync);
      sync();
    }

    // Tab switching
    const tabs = document.querySelectorAll('#upload-tabs .tab');
    tabs.forEach(tab => {
      tab.addEventListener('click', () => {
        tabs.forEach(tabEl => tabEl.classList.remove('active'));
        tab.classList.add('active');
        const target = tab.getAttribute('data-tab');
        document.getElementById('upload-tab-text').style.display = target === 'text' ? '' : 'none';
        document.getElementById('upload-tab-file').style.display = target === 'file' ? '' : 'none';
      });
    });

    // Toggle switches
    bindLoadCacheToggle('text-load-cache-toggle', 'text-load-cache-input');
    bindLoadCacheToggle('file-load-cache-toggle', 'file-load-cache-input');

    // Text submit
    document.getElementById('btn-submit-text').addEventListener('click', submitText);
    // Ctrl+Enter / Cmd+Enter to submit text
    const memoryTextEl = document.getElementById('memory-text');
    memoryTextEl.addEventListener('keydown', (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault();
        submitText();
      }
    });
    // Character counter
    const counterEl = document.getElementById('text-counter');
    memoryTextEl.addEventListener('input', () => {
      const len = (memoryTextEl.value || '').length;
      if (counterEl) counterEl.textContent = (t('memory.charCount') || 'Characters') + ': ' + len.toLocaleString();
    });

    // ---- Multi-file handling ----
    const dropZone = document.getElementById('file-drop-zone');
    const fileInput = document.getElementById('file-input');
    const fileListArea = document.getElementById('file-list-area');
    const btnFile = document.getElementById('btn-submit-file');
    const btnClear = document.getElementById('btn-clear-files');
    let selectedFiles = [];

    function formatFileSize(bytes) {
      if (bytes < 1024) return bytes + ' B';
      if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
      return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
    }

    function renderFileList() {
      const count = selectedFiles.length;
      btnFile.disabled = count === 0;
      btnClear.style.display = count > 0 ? '' : 'none';

      if (count === 0) {
        fileListArea.innerHTML = `<p style="color:var(--text-muted);font-size:0.75rem;margin:0;">${t('memory.noFiles')}</p>`;
        return;
      }

      let html = `<div style="margin-top:0.5rem;text-align:left;max-height:200px;overflow-y:auto;">`;
      html += `<p style="color:var(--text-secondary);font-size:0.8125rem;margin:0 0 0.375rem;font-weight:500;">${t('memory.fileCount', { count: count })}</p>`;
      selectedFiles.forEach((file, idx) => {
        html += `<div style="display:flex;align-items:center;gap:0.5rem;padding:0.25rem 0;font-size:0.8125rem;" data-file-idx="${idx}">
          <i data-lucide="file-text" style="width:14px;height:14px;flex-shrink:0;color:var(--text-muted);"></i>
          <span style="flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;" title="${escapeHtml(file.name)}">${escapeHtml(file.name)}</span>
          <span class="mono" style="color:var(--text-muted);font-size:0.75rem;flex-shrink:0;">${formatFileSize(file.size)}</span>
          <button class="btn-remove-file" data-idx="${idx}" style="background:none;border:none;color:var(--text-muted);cursor:pointer;padding:2px;display:flex;align-items:center;">
            <i data-lucide="x" style="width:14px;height:14px;"></i>
          </button>
        </div>`;
      });
      html += `</div>`;
      fileListArea.innerHTML = html;

      // Bind remove buttons
      fileListArea.querySelectorAll('.btn-remove-file').forEach(btn => {
        btn.addEventListener('click', (e) => {
          e.stopPropagation();
          const idx = parseInt(btn.getAttribute('data-idx'), 10);
          selectedFiles.splice(idx, 1);
          renderFileList();
        });
      });

      if (window.lucide) lucide.createIcons({ nodes: [fileListArea] });
    }

    function removeFile(index) {
      selectedFiles.splice(index, 1);
      renderFileList();
    }

    function addFiles(fileList) {
      const newFiles = Array.from(fileList);
      if (newFiles.length === 0) return;
      selectedFiles = selectedFiles.concat(newFiles);
      renderFileList();
    }

    dropZone.addEventListener('click', (e) => {
      if (e.target.closest('.btn-remove-file')) return;
      fileInput.click();
    });

    dropZone.addEventListener('dragover', (e) => {
      e.preventDefault();
      dropZone.classList.add('drag-over');
    });

    dropZone.addEventListener('dragleave', () => {
      dropZone.classList.remove('drag-over');
    });

    dropZone.addEventListener('drop', (e) => {
      e.preventDefault();
      dropZone.classList.remove('drag-over');
      addFiles(e.dataTransfer.files);
    });

    fileInput.addEventListener('change', () => {
      addFiles(fileInput.files);
      fileInput.value = '';
    });

    // Clear all files
    btnClear.addEventListener('click', (e) => {
      e.stopPropagation();
      selectedFiles = [];
      renderFileList();
    });

    // File submit — send each file as a separate request
    btnFile.addEventListener('click', async () => {
      if (selectedFiles.length === 0) return;
      const sourceName = document.getElementById('file-source-name').value.trim();
      const eventTime = document.getElementById('file-event-time').value;
      const loadCache = !!document.getElementById('file-load-cache-input')?.checked;

      const filesToUpload = [...selectedFiles];
      const total = filesToUpload.length;

      btnFile.disabled = true;
      btnClear.style.display = 'none';

      let successCount = 0;
      let failCount = 0;

      for (let i = 0; i < total; i++) {
        const file = filesToUpload[i];
        btnFile.innerHTML = `${spinnerHtml('spinner-sm')} ${t('memory.uploadProgress', { current: i + 1, total: total })}`;

        try {
          await state.api.rememberFile(state.currentGraphId, file, {
            source_name: sourceName || file.name,
            event_time: eventTime,
            load_cache: loadCache,
          });
          successCount++;
        } catch (err) {
          failCount++;
          console.error(`File upload failed: ${file.name}`, err);
        }
      }

      // Show result
      if (failCount === 0) {
        showToast(t('memory.uploadSuccess'), 'success');
      } else {
        showToast(t('memory.uploadPartialSuccess', { success: successCount, fail: failCount }), failCount === total ? 'error' : 'warning');
      }
      state.events.dispatchEvent(new CustomEvent('graph-changed', { detail: { graphId: state.currentGraphId } }));

      // Reset
      selectedFiles = [];
      renderFileList();
      btnFile.innerHTML = `<i data-lucide="upload" style="width:16px;height:16px;"></i> ${t('memory.uploadProcess')}`;
      if (window.lucide) lucide.createIcons({ nodes: [btnFile] });
      refreshTasks();
    });
  }

  async function submitText() {
    const text = document.getElementById('memory-text').value.trim();
    if (!text) {
      showToast(t('memory.noText'), 'warning');
      return;
    }

    const sourceName = document.getElementById('text-source-name').value.trim();
    const eventTime = document.getElementById('text-event-time').value;
    const loadCache = !!document.getElementById('text-load-cache-input')?.checked;

    const btn = document.getElementById('btn-submit-text');
    btn.disabled = true;
    btn.innerHTML = `${spinnerHtml('spinner-sm')} ${t('memory.submitting')}`;

    try {
      await state.api.rememberText(state.currentGraphId, text, {
        source_name: sourceName,
        event_time: eventTime,
        load_cache: loadCache,
      });
      showToast(t('memory.submitSuccess'), 'success');
      document.getElementById('memory-text').value = '';
      state.events.dispatchEvent(new CustomEvent('graph-changed', { detail: { graphId: state.currentGraphId } }));
      btn.disabled = false;
      btn.innerHTML = `<i data-lucide="send" style="width:16px;height:16px;"></i> ${t('memory.submitMemory')}`;
      if (window.lucide) lucide.createIcons({ nodes: [btn] });
      refreshTasks();
    } catch (err) {
      showToast(t('memory.submitFailed') + ': ' + err.message, 'error');
      btn.disabled = false;
      btn.innerHTML = `<i data-lucide="send" style="width:16px;height:16px;"></i> ${t('memory.submitMemory')}`;
      if (window.lucide) lucide.createIcons({ nodes: [btn] });
    }
  }

  // ---- Task Queue Section ----

  function renderTaskSection(tasks, count) {
    const badge = `<span class="badge badge-primary" style="margin-left:0.5rem;">${escapeHtml(String(count ?? 0))}</span>`;
    const activeTasks = (tasks || []).filter(task => ['running', 'queued', 'paused'].includes(task.status) || ['pausing', 'cancelling'].includes(task.phase));
    const historyTasks = (tasks || []).filter(task => !activeTasks.includes(task));

    function taskTable(items, emptyText) {
      if (!items || items.length === 0) return emptyState(emptyText);
      const rows = items.map(task => {
        const pCls = progressClass(task.status);
        const elapsed = taskTimingText(task);
        const docSize = formatDocSize(task.document_size_bytes ?? task.text_size_bytes ?? task.size_bytes ?? 0);
        const isPaused = task.status === 'paused';
        const isPausePending = task.phase === 'pausing';
        const loadCacheLabel = task.load_cache_memory ? t('memory.loadCacheOn') : t('memory.loadCacheOff');
        const canDelete = task.status === 'queued' || task.status === 'running' || task.status === 'paused';
        const canPause = task.status === 'running' && !isPausePending;
        const canResume = isPaused;
        const progressCell = renderTaskProgress(task, { progressClass: pCls });
        const repairCount = Number(task.repair_window_count || task.repair_window_indices?.length || task.failed_window_indices?.length || 0);
        const repairHint = repairCount > 0
          ? `<div style="font-size:0.7rem;color:var(--warning);margin-top:0.2rem;">只补跑 ${escapeHtml(String(repairCount))} 个缺失/失败窗口</div>`
          : '';
        return `
          <tr data-task-id="${escapeHtml(task.task_id)}" title="${t('memory.taskDetail')}">
            <td>${escapeHtml(truncate(task.source_name || '-', 24))}</td>
            <td><span class="mono" title="${escapeHtml(String(task.document_size_bytes || 0))} bytes">${escapeHtml(docSize)}</span></td>
            <td>${escapeHtml(loadCacheLabel)}</td>
            <td>${statusBadge(task.status, task.phase)}</td>
            <td>${progressCell}${repairHint}</td>
            <td>${escapeHtml(task.phase_label || '-')}</td>
            <td>${elapsed}</td>
            <td>
              ${isPausePending ? `
                <button class="btn btn-secondary btn-sm" disabled style="margin-right:0.35rem;opacity:0.75;cursor:not-allowed;">
                  ${spinnerHtml('spinner-sm')}
                  ${t('memory.pausePending')} · ${escapeHtml(task.phase_label || '')}
                </button>
              ` : ''}
              ${task.phase === 'cancelling' ? `
                <button class="btn btn-secondary btn-sm" disabled style="margin-right:0.35rem;opacity:0.75;cursor:not-allowed;color:var(--error);">
                  ${spinnerHtml('spinner-sm')}
                  ${t('memory.deleting')} · ${escapeHtml(task.phase_label || '')}
                </button>
              ` : ''}
              ${canPause ? `
                <button class="btn btn-secondary btn-sm btn-pause-task" data-task-id="${escapeHtml(task.task_id)}" title="${t('memory.pauseTask')}" style="margin-right:0.35rem;">
                  <i data-lucide="pause" style="width:14px;height:14px;"></i>
                  ${t('memory.pauseTask')}
                </button>
              ` : ''}
              ${canResume ? `
                <button class="btn btn-secondary btn-sm btn-resume-task" data-task-id="${escapeHtml(task.task_id)}" title="${t('memory.startTask')}" style="margin-right:0.35rem;">
                  <i data-lucide="play" style="width:14px;height:14px;"></i>
                  ${t('memory.startTask')}
                </button>
              ` : ''}
              ${canDelete ? `
                <button class="btn btn-secondary btn-sm btn-delete-task" data-task-id="${escapeHtml(task.task_id)}" title="${t('memory.deleteTask')}">
                  <i data-lucide="trash-2" style="width:14px;height:14px;"></i>
                  ${t('memory.deleteTask')}
                </button>
              ` : '<span style="color:var(--text-muted);">-</span>'}
            </td>
          </tr>
        `;
      }).join('');
      return `
        <div class="table-container">
          <table class="data-table">
            <thead>
              <tr>
                <th>${t('memory.taskSource')}</th>
                <th>${t('memory.documentSize')}</th>
                <th>${t('memory.taskLoadCache')}</th>
                <th>${t('memory.taskStatus')}</th>
                <th>${t('memory.taskProgress')}</th>
                <th>${t('memory.taskPhase')}</th>
                <th>${t('memory.taskElapsed')}</th>
                <th>${t('memory.taskActions')}</th>
              </tr>
            </thead>
            <tbody>${rows}</tbody>
          </table>
        </div>
      `;
    }

    let tableHtml;
    if (!tasks || tasks.length === 0) {
      tableHtml = emptyState(t('memory.noTasks'));
    } else {
      tableHtml = `
        <div style="margin-bottom:1rem;">
          ${taskTable(activeTasks, t('memory.noActiveTasks'))}
        </div>
        <details style="border-top:1px solid var(--border-color);padding-top:0.75rem;">
          <summary style="cursor:pointer;color:var(--text-secondary);font-weight:600;">${t('memory.historyTasks')} (${historyTasks.length})</summary>
          <div style="margin-top:0.75rem;">${taskTable(historyTasks, t('memory.noHistoryTasks'))}</div>
        </details>
      `;
    }

    return `
      <div class="card" style="margin-bottom:1rem;">
        <div class="card-header">
          <span class="card-title">${t('memory.processQueue')}${badge}</span>
          <button class="btn btn-secondary btn-sm" id="btn-resume-all-tasks">
            <i data-lucide="play" style="width:14px;height:14px;"></i>
            ${t('memory.resumeAll')}
          </button>
        </div>
        <div id="task-list">${tableHtml}</div>
      </div>
    `;
  }

  async function loadTasks() {
    try {
      const res = await state.api.rememberTasks(state.currentGraphId);
      const tasks = res.data?.tasks || [];
      const count = res.data?.count ?? tasks.length;

      // Track active tasks for smart refresh
      const hadActive = _hasActiveTasks;
      _hasActiveTasks = tasks.some(t => t.status === 'running' || t.status === 'queued' || t.status === 'paused');
      if (hadActive !== _hasActiveTasks && typeof scheduleRefresh === 'function') scheduleRefresh();

      const el = document.getElementById('task-list-wrapper');
      if (!el) return;

      el.innerHTML = renderTaskSection(tasks, count);

      // Bind click handlers for task detail modal
      el.querySelectorAll('tr[data-task-id]').forEach(row => {
        row.addEventListener('click', () => {
          const taskId = row.getAttribute('data-task-id');
          showTaskDetail(taskId);
        });
      });

      el.querySelectorAll('.btn-delete-task').forEach(btn => {
        btn.addEventListener('click', async (e) => {
          e.stopPropagation();
          const taskId = btn.getAttribute('data-task-id');
          await deleteQueuedTask(taskId);
        });
      });
      el.querySelectorAll('.btn-pause-task').forEach(btn => {
        btn.addEventListener('click', async (e) => {
          e.stopPropagation();
          const taskId = btn.getAttribute('data-task-id');
          await pauseTask(taskId);
        });
      });
      el.querySelectorAll('.btn-resume-task').forEach(btn => {
        btn.addEventListener('click', async (e) => {
          e.stopPropagation();
          const taskId = btn.getAttribute('data-task-id');
          await resumeTask(taskId);
        });
      });
      el.querySelector('#btn-resume-all-tasks')?.addEventListener('click', async (e) => {
        e.stopPropagation();
        await resumeAllTasks();
      });

      if (window.lucide) lucide.createIcons({ nodes: [el] });
      bindClickableRows(el);
    } catch (err) {
      const el = document.getElementById('task-list-wrapper');
      if (el) {
        el.innerHTML = `<div class="card" style="margin-bottom:1rem;"><div class="empty-state"><p style="color:var(--error);">${t('memory.loadTasksFailed')}: ${escapeHtml(err.message)}</p></div></div>`;
      }
    }
  }

  function refreshTasks() {
    loadTasks();
  }

  async function deleteQueuedTask(taskId) {
    if (!taskId) return;
    const ok = await showConfirm({ message: t('memory.deleteTaskConfirm'), destructive: true });
    if (!ok) return;
    const btn = document.querySelector(`.btn-delete-task[data-task-id="${taskId}"]`);
    if (btn) { btn.disabled = true; btn.innerHTML = `${spinnerHtml('spinner-sm')} ${t('memory.deleting')}`; }
    try {
      const res = await state.api.rememberDelete(taskId, state.currentGraphId);
      showToast(res.data?.message || t('memory.deleteTaskSuccess'), 'success');
      refreshTasks();
    } catch (err) {
      showToast(t('memory.deleteTaskFailed') + ': ' + err.message, 'error');
      refreshTasks();
    }
  }

  async function pauseTask(taskId) {
    if (!taskId) return;
    const btn = document.querySelector(`.btn-pause-task[data-task-id="${taskId}"]`);
    if (btn) { btn.disabled = true; btn.innerHTML = `${spinnerHtml('spinner-sm')} ${t('memory.pausing')}`; }
    try {
      const res = await state.api.rememberPause(taskId, state.currentGraphId);
      showToast(res.data?.message || t('memory.pauseTaskSuccess'), 'success');
      refreshTasks();
    } catch (err) {
      showToast(t('memory.pauseTaskFailed') + ': ' + err.message, 'error');
      refreshTasks();
    }
  }

  async function resumeTask(taskId) {
    if (!taskId) return;
    const btn = document.querySelector(`.btn-resume-task[data-task-id="${taskId}"]`);
    if (btn) { btn.disabled = true; btn.innerHTML = `${spinnerHtml('spinner-sm')} ${t('memory.resuming')}`; }
    try {
      const res = await state.api.rememberResume(taskId, state.currentGraphId);
      showToast(res.data?.message || t('memory.resumeTaskSuccess'), 'success');
      refreshTasks();
    } catch (err) {
      showToast(t('memory.resumeTaskFailed') + ': ' + err.message, 'error');
      refreshTasks();
    }
  }

  async function resumeAllTasks() {
    try {
      const res = await state.api.rememberResumeAll(state.currentGraphId);
      showToast(t('memory.resumeAllSuccess', { count: res.data?.count || 0 }), 'success');
      refreshTasks();
    } catch (err) {
      showToast(t('memory.resumeAllFailed') + ': ' + err.message, 'error');
      refreshTasks();
    }
  }

  async function showTaskDetail(taskId) {
    try {
      const res = await state.api.rememberStatus(taskId, state.currentGraphId);
      const task = res.data;

      const pCls = progressClass(task.status);
      const isRunning = task.status === 'running';
      const isPausePending = task.phase === 'pausing';
      const loadCacheLabel = task.load_cache_memory ? t('memory.loadCacheOn') : t('memory.loadCacheOff');
      const progressDetail = renderTaskProgress(task, { progressClass: pCls });
      const chainDetail = renderTaskChainDetails(task);
      const docSize = formatDocSize(task.document_size_bytes ?? task.text_size_bytes ?? task.size_bytes ?? 0);
      const repairCount = Number(task.repair_window_count || task.repair_window_indices?.length || task.failed_window_indices?.length || 0);
      let body = `
        <div style="margin-bottom:1rem;">
          <div style="display:grid;grid-template-columns:auto 1fr;gap:0.5rem 1rem;font-size:0.8125rem;">
            <span style="color:var(--text-muted);">${t('memory.taskSource')}</span>
            <span>${escapeHtml(task.source_name || '-')}</span>
            <span style="color:var(--text-muted);">${t('memory.documentSize')}</span>
            <span class="mono">${escapeHtml(docSize)}</span>
            <span style="color:var(--text-muted);">${t('memory.taskStatus')}</span>
            <span>${statusBadge(task.status, task.phase)}</span>
            <span style="color:var(--text-muted);">${t('memory.taskLoadCache')}</span>
            <span>${escapeHtml(loadCacheLabel)}</span>
            <span style="color:var(--text-muted);">${t('memory.taskProgress')}</span>
            ${progressDetail}
            <span style="color:var(--text-muted);">${t('memory.elapsedEstimated')}</span>
            <span>${escapeHtml(taskTimingText(task))}</span>
            <span style="color:var(--text-muted);">${t('memory.taskPhase')}</span>
            <span>${escapeHtml(task.phase_label || '-')}</span>
            <span style="color:var(--text-muted);">${t('memory.taskCreated')}</span>
            <span>${formatDate(task.created_at)}</span>
            <span style="color:var(--text-muted);">${t('memory.taskStarted')}</span>
            <span>${formatDate(task.started_at)}</span>
            ${repairCount > 0 ? `
              <span style="color:var(--text-muted);">补跑窗口</span>
              <span style="color:var(--warning);">只处理 ${escapeHtml(String(repairCount))} 个缺失/失败窗口：${escapeHtml((task.repair_window_indices || task.failed_window_indices || []).slice(0, 20).join(', '))}${repairCount > 20 ? ' ...' : ''}</span>
            ` : ''}
          </div>
        </div>
      `;

      if (chainDetail) {
        body += `
          <div class="divider"></div>
          <div>
            <span class="form-label" style="margin-bottom:0.5rem;">${t('memory.phaseProgress')}</span>
            ${chainDetail}
          </div>
        `;
      }

      if (task.result) {
        body += `
          <div class="divider"></div>
          <div>
            <span class="form-label" style="margin-bottom:0.5rem;">${t('memory.taskResult')}</span>
            <pre class="mono" style="background:var(--bg-input);border:1px solid var(--border-color);border-radius:0.5rem;padding:0.75rem;overflow-x:auto;white-space:pre-wrap;word-break:break-word;font-size:0.8125rem;max-height:300px;overflow-y:auto;">${escapeHtml(typeof task.result === 'string' ? task.result : JSON.stringify(task.result, null, 2))}</pre>
          </div>
        `;
      }

      if (task.error) {
        body += `
          <div class="divider"></div>
          <div>
            <span class="form-label" style="margin-bottom:0.5rem;color:var(--error);">${t('memory.taskError')}</span>
            <pre class="mono" style="background:var(--error-dim);border:1px solid var(--error);border-radius:0.5rem;padding:0.75rem;overflow-x:auto;white-space:pre-wrap;word-break:break-word;font-size:0.8125rem;color:var(--error);max-height:200px;overflow-y:auto;">${escapeHtml(typeof task.error === 'string' ? task.error : JSON.stringify(task.error, null, 2))}</pre>
          </div>
        `;
      }

      const footerParts = [
        `<button class="btn btn-secondary btn-sm task-detail-close">${t('common.close')}</button>`,
      ];
      if (isPausePending) {
        footerParts.unshift(`
          <button class="btn btn-secondary btn-sm" disabled style="opacity:0.75;cursor:not-allowed;">
            ${spinnerHtml('spinner-sm')}
            ${t('memory.pausePending')} · ${escapeHtml(task.phase_label || '')}
          </button>
        `);
      }
      if (task.status === 'running' && task.phase === 'cancelling') {
        footerParts.unshift(`
          <button class="btn btn-secondary btn-sm" disabled style="opacity:0.75;cursor:not-allowed;color:var(--error);">
            ${spinnerHtml('spinner-sm')}
            ${t('memory.deleting')} · ${escapeHtml(task.phase_label || '')}
          </button>
        `);
      }
      if (task.status === 'running' && !isPausePending) {
        footerParts.unshift(`
          <button class="btn btn-secondary btn-sm task-detail-pause">
            <i data-lucide="pause" style="width:14px;height:14px;"></i>
            ${t('memory.pauseTask')}
          </button>
        `);
      }
      if (task.status === 'paused') {
        footerParts.unshift(`
          <button class="btn btn-secondary btn-sm task-detail-resume">
            <i data-lucide="play" style="width:14px;height:14px;"></i>
            ${t('memory.startTask')}
          </button>
        `);
      }
      if (task.status === 'queued' || task.status === 'running' || task.status === 'paused') {
        footerParts.unshift(`
          <button class="btn btn-secondary btn-sm task-detail-delete">
            <i data-lucide="trash-2" style="width:14px;height:14px;"></i>
            ${t('memory.deleteTask')}
          </button>
        `);
      }

      const modal = showModal({
        title: t('memory.taskDetail'),
        content: body,
        footer: `<div style="display:flex;justify-content:flex-end;gap:0.5rem;flex-wrap:wrap;">${footerParts.join('')}</div>`,
        size: 'lg',
      });

      const { overlay, close } = modal;
      const closeBtn = overlay.querySelector('.task-detail-close');
      if (closeBtn) {
        closeBtn.addEventListener('click', () => close());
      }

      const deleteBtn = overlay.querySelector('.task-detail-delete');
      if (deleteBtn) {
        deleteBtn.addEventListener('click', async () => {
          close();
          await deleteQueuedTask(task.task_id);
        });
      }

      const pauseBtn = overlay.querySelector('.task-detail-pause');
      if (pauseBtn) {
        pauseBtn.addEventListener('click', async () => {
          close();
          await pauseTask(task.task_id);
        });
      }

      const resumeBtn = overlay.querySelector('.task-detail-resume');
      if (resumeBtn) {
        resumeBtn.addEventListener('click', async () => {
          close();
          await resumeTask(task.task_id);
        });
      }
    } catch (err) {
      showToast(t('memory.taskDetailFailed') + ': ' + err.message, 'error');
    }
  }

  // ---- Documents Section ----

  let _allDocs = [];
  let _docsPage = 1;
  let _docsPageSize = 10;
  let _deletingDocIds = new Set();
  let _selectedDocIds = new Set();

  // ---- Multi-select helpers ----

  function toggleSelectAll() {
    const total = _allDocs.length;
    if (_selectedDocIds.size === total) {
      _selectedDocIds.clear();
    } else {
      _selectedDocIds = new Set(_allDocs.map(d => d.document_version_id).filter(Boolean));
    }
    updateDocsTable();
    updateSelectionUI();
  }

  function toggleSelectDoc(docVersionId) {
    if (!docVersionId) return;
    if (_selectedDocIds.has(docVersionId)) {
      _selectedDocIds.delete(docVersionId);
    } else {
      _selectedDocIds.add(docVersionId);
    }
    updateSelectionUI();
  }

  function updateSelectionUI() {
    const total = _allDocs.length;
    const validTotal = _allDocs.filter(d => d.document_version_id).length;
    const selectedCount = _selectedDocIds.size;

    // Update select-all checkbox
    const selectAllCheckbox = document.getElementById('docs-select-all');
    if (selectAllCheckbox) {
      selectAllCheckbox.checked = selectedCount === validTotal && validTotal > 0;
      selectAllCheckbox.indeterminate = selectedCount > 0 && selectedCount < validTotal;
    }

    // Update batch delete button
    const batchBtn = document.getElementById('docs-batch-delete');
    if (batchBtn) {
      batchBtn.disabled = selectedCount === 0;
    }

    // Update batch count badge
    const batchCount = document.getElementById('docs-batch-count');
    if (batchCount) {
      batchCount.textContent = selectedCount > 0 ? ` (${selectedCount})` : '';
    }

    // Update row highlights and per-row checkboxes
    document.querySelectorAll('.doc-row-checkbox').forEach(cb => {
      const id = cb.getAttribute('data-doc-id');
      if (!id) return;
      cb.checked = _selectedDocIds.has(id);
      const row = cb.closest('tr');
      if (row) {
        row.style.background = _selectedDocIds.has(id) ? 'rgba(59,130,246,0.08)' : '';
      }
    });
  }

  async function batchDeleteDocuments() {
    const ids = Array.from(_selectedDocIds);
    const count = ids.length;
    if (!count) return;
    const confirmed = await showConfirm({
      message: t('documents.batchDeleteConfirm').replace('{count}', String(count)),
      destructive: true,
    });
    if (!confirmed) return;
    try {
      _selectedDocIds.clear();
      const res = await state.api.batchDeleteDocuments(ids, state.currentGraphId);
      const data = res.data;
      if (res.success || data) {
        const deleted = data?.deleted || 0;
        const failed = count - deleted;
        if (failed > 0) {
          showToast(t('documents.batchPartialFail').replace('{failed}', String(failed)), 'warning');
        } else {
          showToast(t('documents.batchDeleted').replace('{count}', String(deleted)), 'success');
        }
      } else {
        showToast(res.error || t('documents.batchDeleteFailed'), 'error');
      }
      await loadDocs();
    } catch (e) {
      showToast(t('documents.batchDeleteFailed') + ': ' + e.message, 'error');
    }
  }

  function renderDocsTableHtml() {
    const total = _allDocs.length;
    if (total === 0) return emptyState(t('memory.noDocs'));

    const totalPages = Math.max(1, Math.ceil(total / _docsPageSize));
    if (_docsPage > totalPages) _docsPage = totalPages;
    const start = (_docsPage - 1) * _docsPageSize;
    const pageDocs = _allDocs.slice(start, start + _docsPageSize);

    const rows = pageDocs.map((d, i) => {
      const idx = start + i;
      const title = d.version_title || d.title || d.source_document || d.doc_name || '-';
      const processed = d.processed_time || d.updated_at || d.created_at;
      const size = formatDocSize(Number(d.byte_size || d.size || d.blob_size || 0));
      const entityCount = Number(d.entity_count || 0).toLocaleString();
      const relationCount = Number(d.relation_count || 0).toLocaleString();
      const docId = d.document_version_id || '';
      const invalid = !docId;
      const deleting = _deletingDocIds.has(docId);
      const selected = _selectedDocIds.has(docId);
      const integrity = d.integrity || {};
      const missingWindows = Number(integrity.missing_windows || 0);
      const integrityHtml = integrity.complete === false
        ? `<div style="font-size:0.7rem;color:var(--warning);margin-top:0.2rem;">缺失 ${escapeHtml(String(missingWindows))} 个窗口</div>`
        : integrity.complete === true
          ? `<div style="font-size:0.7rem;color:var(--success);margin-top:0.2rem;">完整</div>`
          : '';
      return `
        <tr style="${(deleting || invalid) ? 'opacity:0.55;' : ''}${selected && !deleting && !invalid ? 'background:rgba(59,130,246,0.08);' : ''}">
          <td style="width:2rem;text-align:center;">
            <input type="checkbox" class="doc-row-checkbox" data-doc-id="${escapeHtml(docId)}" ${selected ? 'checked' : ''} ${(deleting || invalid) ? 'disabled' : ''}>
          </td>
          <td>${escapeHtml(truncate(title, 32))}${integrityHtml}</td>
          <td class="mono">${escapeHtml(size)}</td>
          <td>${entityCount} / ${relationCount}</td>
          <td>${formatDate(d.created_at)}</td>
          <td>${invalid ? t('memory.invalidRecord') : deleting ? t('memory.deleting') : formatDateMs(processed)}</td>
          <td style="display:flex;gap:0.35rem;flex-wrap:wrap;">
            <button class="btn btn-secondary btn-sm doc-detail-btn" data-doc-idx="${idx}" ${(deleting || invalid) ? 'disabled' : ''}>${t('common.detail')}</button>
            <button class="btn btn-secondary btn-sm doc-integrity-btn" data-doc-idx="${idx}" ${(deleting || invalid) ? 'disabled' : ''}>检查</button>
            ${missingWindows > 0 ? `<button class="btn btn-secondary btn-sm doc-repair-btn" data-doc-idx="${idx}" ${(deleting || invalid) ? 'disabled' : ''}>修复</button>` : ''}
            <button class="btn btn-secondary btn-sm doc-delete-btn" data-doc-idx="${idx}" style="color:var(--error);" ${(deleting || invalid) ? 'disabled' : ''}>
              ${deleting ? spinnerHtml('spinner-sm') : '<i data-lucide="trash-2" style="width:14px;height:14px;"></i>'}
            </button>
          </td>
        </tr>
      `;
    }).join('');

    const pageSizeOptions = [10, 20, 30, 50];
    const pageSizeSelect = pageSizeOptions.map(n =>
      `<option value="${n}" ${n === _docsPageSize ? 'selected' : ''}>${n}</option>`
    ).join('');
    const pageInfo = `${start + 1}-${Math.min(start + _docsPageSize, total)} / ${total}`;

    return `
      <div class="table-container">
        <table class="data-table">
          <thead>
            <tr>
              <th style="width:2rem;text-align:center;"><input type="checkbox" id="docs-select-all"></th>
              <th>${t('memory.taskSource')}</th>
              <th>${t('memory.docSize')}</th>
              <th>${t('memory.entityRelation')}</th>
              <th>${t('memory.docTime')}</th>
              <th>${t('memory.processedTime')}</th>
              <th>${t('common.detail')}</th>
            </tr>
          </thead>
          <tbody>${rows}</tbody>
        </table>
      </div>
      <div style="display:flex;align-items:center;justify-content:space-between;padding:0.75rem 0.5rem;font-size:0.85rem;flex-wrap:wrap;gap:0.5rem;">
        <span style="color:var(--text-secondary);">${pageInfo}</span>
        <div style="display:flex;align-items:center;gap:0.5rem;">
          <button class="btn btn-secondary btn-sm" id="docs-prev" ${_docsPage <= 1 ? 'disabled' : ''}>&laquo;</button>
          <span style="color:var(--text-secondary);">${_docsPage} / ${totalPages}</span>
          <button class="btn btn-secondary btn-sm" id="docs-next" ${_docsPage >= totalPages ? 'disabled' : ''}>&raquo;</button>
          <select id="docs-page-size" style="margin-left:0.5rem;">${pageSizeSelect}</select>
        </div>
      </div>
    `;
  }

  function renderDocsSection(count) {
    const badge = `<span class="badge badge-primary" style="margin-left:0.5rem;">${escapeHtml(String(count ?? 0))}</span>`;
    return `
      <div class="card">
        <div class="card-header">
          <span class="card-title">${t('memory.docs')}${badge}</span>
          <button class="btn btn-secondary btn-sm" id="docs-batch-delete" disabled style="color:var(--error);">
            <i data-lucide="trash-2" style="width:14px;height:14px;"></i>
            ${t('documents.batchDelete')}<span id="docs-batch-count"></span>
          </button>
        </div>
        <div id="docs-list">${renderDocsTableHtml()}</div>
      </div>
    `;
  }

  function updateDocsTable() {
    const el = document.getElementById('docs-list');
    if (!el) return;
    el.innerHTML = renderDocsTableHtml();
    bindDocsEvents();
  }

  function bindDocsPagination() {
    const prevBtn = document.getElementById('docs-prev');
    const nextBtn = document.getElementById('docs-next');
    const sizeSelect = document.getElementById('docs-page-size');

    if (prevBtn) {
      prevBtn.addEventListener('click', () => {
        if (_docsPage > 1) { _docsPage--; updateDocsTable(); }
      });
    }
    if (nextBtn) {
      nextBtn.addEventListener('click', () => {
        const totalPages = Math.ceil(_allDocs.length / _docsPageSize);
        if (_docsPage < totalPages) { _docsPage++; updateDocsTable(); }
      });
    }
    if (sizeSelect) {
      sizeSelect.addEventListener('change', () => {
        _docsPageSize = parseInt(sizeSelect.value, 10);
        _docsPage = 1;
        updateDocsTable();
      });
    }
  }

  async function copyTextToClipboard(text) {
    const value = String(text || '');
    if (!value) return false;
    try {
      if (navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(value);
        return true;
      }
    } catch (_) {}
    const ta = document.createElement('textarea');
    ta.value = value;
    ta.style.position = 'fixed';
    ta.style.left = '-9999px';
    document.body.appendChild(ta);
    ta.focus();
    ta.select();
    let ok = false;
    try { ok = document.execCommand('copy'); } catch (_) {}
    ta.remove();
    return ok;
  }

  function formatPathValue(path) {
    return path ? `<span class="mono" style="word-break:break-all;">${escapeHtml(path)}</span>` : '<span style="color:var(--text-muted);">-</span>';
  }

  async function showDocPreview(doc) {
    const name = doc.version_title || doc.title || doc.source_document || doc.doc_name || '-';
    const size = formatDocSize(doc.byte_size || doc.size || doc.blob_size || 0);
    const entityCount = Number(doc.entity_count || 0).toLocaleString();
    const relationCount = Number(doc.relation_count || 0).toLocaleString();
    const eventTime = formatDate(doc.created_at);
    const procTime = formatDateMs(doc.processed_time || doc.updated_at || doc.created_at);
    const hash = doc.content_hash || doc.doc_hash || doc.source_id || '-';
    const readPath = doc.read_path || doc.absolute_path || doc.managed_path || doc.snapshot_path || doc.blob_path || '';
    const charCount = Number(doc.char_count || 0).toLocaleString();
    const lineCount = Number(doc.line_count || 0).toLocaleString();
    const integrity = doc.integrity || {};
    const missingWindows = Number(integrity.missing_windows || 0);
    const integrityText = integrity.complete === false
      ? `缺失 ${missingWindows} / ${Number(integrity.total_windows || 0)} 个窗口`
      : integrity.complete === true
        ? `完整 ${Number(integrity.complete_windows || 0)} / ${Number(integrity.total_windows || 0)}`
        : '未检查';

    const rows = [
      [t('memory.taskSource'), escapeHtml(truncate(name, 60))],
      [t('memory.documentSize'), `<span class="mono">${escapeHtml(size)}</span>`],
      ['来源模式', escapeHtml(doc.source_mode || '-')],
      ['可读路径', formatPathValue(readPath)],
      ['Managed', formatPathValue(doc.managed_path || '')],
      ['External', formatPathValue(doc.absolute_path || '')],
      ['Snapshot', formatPathValue(doc.snapshot_path || doc.blob_path || '')],
      ['字符 / 行数', `<span class="mono">${escapeHtml(charCount)} / ${escapeHtml(lineCount)}</span>`],
      [t('memory.entityRelation'), `${entityCount} / ${relationCount}`],
      [t('memory.docHash'), `<span class="mono">${escapeHtml(hash)}</span>`],
      ['完整性', escapeHtml(integrityText)],
      [t('memory.docTime'), eventTime],
      [t('memory.processedTime'), procTime],
    ];
    const grid = rows.map(([k, v]) =>
      `<span style="color:var(--text-secondary);white-space:nowrap;">${k}:</span><span>${v}</span>`
    ).join('');

    const modal = showModal({
      title: t('memory.docPreview'),
      size: 'lg',
      content: `
        <div style="display:grid;grid-template-columns:auto 1fr;gap:0.4rem 0.75rem;font-size:0.85rem;">${grid}</div>
        <div style="display:flex;gap:0.5rem;flex-wrap:wrap;margin-top:1rem;">
          <button class="btn btn-secondary btn-sm doc-copy-path-btn" ${readPath ? '' : 'disabled'}>复制路径</button>
          <button class="btn btn-secondary btn-sm doc-fulltext-btn" ${doc.document_version_id ? '' : 'disabled'}>查看全文</button>
          <button class="btn btn-secondary btn-sm modal-integrity-btn" ${doc.document_version_id ? '' : 'disabled'}>检查完整性</button>
          ${missingWindows > 0 ? `<button class="btn btn-secondary btn-sm modal-repair-btn">修复文档</button>` : ''}
        </div>
        <pre class="doc-fulltext-box" style="display:none;margin-top:0.75rem;max-height:50vh;overflow:auto;padding:0.75rem;border:1px solid var(--border);border-radius:8px;background:var(--bg-secondary);white-space:pre-wrap;word-break:break-word;font-size:0.8rem;"></pre>
      `,
    });
    const copyBtn = modal.overlay.querySelector('.doc-copy-path-btn');
    if (copyBtn) {
      copyBtn.addEventListener('click', async () => {
        const ok = await copyTextToClipboard(readPath);
        showToast(ok ? '已复制路径' : '复制失败', ok ? 'success' : 'error');
      });
    }
    const fullBtn = modal.overlay.querySelector('.doc-fulltext-btn');
    const fullBox = modal.overlay.querySelector('.doc-fulltext-box');
    if (fullBtn && fullBox) {
      fullBtn.addEventListener('click', async () => {
        fullBtn.disabled = true;
        fullBtn.innerHTML = `${spinnerHtml('spinner-sm')} 加载中`;
        try {
          const res = await state.api.documentContent(doc.document_version_id, state.currentGraphId, { offset: 0, limit: 200000 });
          const data = res.data || {};
          fullBox.style.display = '';
          fullBox.textContent = (data.content || '') + (data.truncated ? '\n\n[内容较大，已显示前 200000 字符]' : '');
          fullBtn.innerHTML = '已加载全文';
        } catch (err) {
          fullBtn.disabled = false;
          fullBtn.textContent = '查看全文';
          showToast('加载全文失败: ' + err.message, 'error');
        }
      });
    }
    const integrityBtn = modal.overlay.querySelector('.modal-integrity-btn');
    if (integrityBtn) {
      integrityBtn.addEventListener('click', async () => {
        integrityBtn.disabled = true;
        integrityBtn.innerHTML = spinnerHtml('spinner-sm');
        try {
          await checkDocumentIntegrity(doc);
          modal.close?.();
          await loadDocs();
        } catch (err) {
          integrityBtn.disabled = false;
          integrityBtn.textContent = '检查完整性';
        }
      });
    }
    const repairBtn = modal.overlay.querySelector('.modal-repair-btn');
    if (repairBtn) {
      repairBtn.addEventListener('click', async () => {
        repairBtn.disabled = true;
        repairBtn.innerHTML = spinnerHtml('spinner-sm');
        try {
          await repairDocument(doc);
          modal.close?.();
          await Promise.all([loadDocs(), refreshTasks()]);
        } catch (err) {
          repairBtn.disabled = false;
          repairBtn.textContent = '修复文档';
        }
      });
    }
  }

  function bindDocsEvents() {
    if (window.lucide) lucide.createIcons({ nodes: [document.getElementById('docs-list-wrapper')] });
    bindDocsPagination();

    // Select-all checkbox
    const selectAllCb = document.getElementById('docs-select-all');
    if (selectAllCb) {
      selectAllCb.addEventListener('change', () => {
        toggleSelectAll();
      });
    }

    // Per-row checkboxes
    document.querySelectorAll('.doc-row-checkbox').forEach(cb => {
      cb.addEventListener('change', () => {
        const docId = cb.getAttribute('data-doc-id');
        toggleSelectDoc(docId);
      });
    });

    // Batch delete button
    const batchBtn = document.getElementById('docs-batch-delete');
    if (batchBtn) {
      batchBtn.addEventListener('click', () => {
        batchDeleteDocuments();
      });
    }

    document.querySelectorAll('.doc-detail-btn').forEach(btn => {
      btn.addEventListener('click', async () => {
        const idx = parseInt(btn.getAttribute('data-doc-idx'), 10);
        if (isNaN(idx) || idx < 0 || idx >= _allDocs.length) return;
        await showDocPreview(_allDocs[idx]);
      });
    });
    document.querySelectorAll('.doc-integrity-btn').forEach(btn => {
      btn.addEventListener('click', async () => {
        const idx = parseInt(btn.getAttribute('data-doc-idx'), 10);
        if (isNaN(idx) || idx < 0 || idx >= _allDocs.length) return;
        await checkDocumentIntegrity(_allDocs[idx]);
      });
    });
    document.querySelectorAll('.doc-repair-btn').forEach(btn => {
      btn.addEventListener('click', async () => {
        const idx = parseInt(btn.getAttribute('data-doc-idx'), 10);
        if (isNaN(idx) || idx < 0 || idx >= _allDocs.length) return;
        await repairDocument(_allDocs[idx]);
      });
    });
    document.querySelectorAll('.doc-delete-btn').forEach(btn => {
      btn.addEventListener('click', async () => {
        const idx = parseInt(btn.getAttribute('data-doc-idx'), 10);
        if (isNaN(idx) || idx < 0 || idx >= _allDocs.length) return;
        await deleteDocument(_allDocs[idx]);
      });
    });

    // Initialize selection UI state
    updateSelectionUI();
  }

  async function deleteDocument(doc) {
    const id = doc.document_version_id;
    if (!id) return;
    const name = doc.version_title || doc.title || doc.relative_path || id;
    const ok = await showConfirm({
      message: t('memory.deleteDocConfirm', { name: truncate(name, 40) }),
      destructive: true,
    });
    if (!ok) return;
    _deletingDocIds.add(id);
    updateDocsTable();
    const pendingToast = showToast(t('memory.deletingDoc', { name: truncate(name, 40) }), 'info', 0);
    try {
      await state.api.deleteDocument(id, state.currentGraphId);
      _deletingDocIds.delete(id);
      _allDocs = _allDocs.filter(d => d.document_version_id !== id);
      updateDocsTable();
      if (pendingToast) pendingToast.remove();
      showToast(t('memory.docDeleted'), 'success');
      state.events.dispatchEvent(new CustomEvent('graph-changed', { detail: { graphId: state.currentGraphId } }));
    } catch (err) {
      _deletingDocIds.delete(id);
      if (pendingToast) pendingToast.remove();
      showToast(t('memory.docDeleteFailed') + ': ' + err.message, 'error');
      await loadDocs();
    }
  }

  async function checkDocumentIntegrity(doc) {
    const id = doc.document_version_id;
    if (!id) return;
    try {
      const res = await state.api.documentIntegrity(id, state.currentGraphId);
      doc.integrity = res.data || {};
      updateDocsTable();
      const missing = Number(doc.integrity.missing_windows || 0);
      showToast(missing > 0 ? `发现 ${missing} 个缺失/不完整窗口` : '文档完整性正常', missing > 0 ? 'warning' : 'success');
    } catch (err) {
      showToast('完整性检查失败: ' + err.message, 'error');
    }
  }

  async function repairDocument(doc) {
    const id = doc.document_version_id;
    if (!id) return;
    try {
      const res = await state.api.repairDocument(id, state.currentGraphId);
      showToast(res.data?.message || '已提交修复任务', res.data?.submitted === false ? 'info' : 'success');
      await Promise.all([loadDocs(), refreshTasks()]);
    } catch (err) {
      showToast('文档修复失败: ' + err.message, 'error');
    }
  }

  async function loadDocs() {
    try {
      const res = await state.api.listDocs(state.currentGraphId);
      _allDocs = res.data?.docs || [];
      _deletingDocIds = new Set([..._deletingDocIds].filter(id => _allDocs.some(d => d.document_version_id === id)));
      _selectedDocIds = new Set([..._selectedDocIds].filter(id => _allDocs.some(d => d.document_version_id === id)));
      _docsPage = 1;

      const el = document.getElementById('docs-list-wrapper');
      if (!el) return;

      el.innerHTML = renderDocsSection(_allDocs.length);
      bindDocsEvents();
    } catch (err) {
      const el = document.getElementById('docs-list-wrapper');
      if (el) {
        el.innerHTML = `<div class="card"><div class="empty-state"><p style="color:var(--error);">${t('memory.loadDocsFailed')}: ${escapeHtml(err.message)}</p></div></div>`;
      }
    }
  }

  async function showDocContent(filename) {
    try {
      const res = await state.api.getDocContent(filename, state.currentGraphId);
      const data = res.data || {};
      const meta = data.meta || {};
      _renderDocModal(
        meta.source_document || meta.doc_name || filename,
        meta.event_time || '-',
        data.cache || '',
        data.original || '',
      );
    } catch (err) {
      showToast(t('memory.loadDocContentFailed') + ': ' + err.message, 'error');
    }
  }

  // ---- Page Lifecycle ----

  async function render(container, params) {
    container.innerHTML = `
      <div class="page-enter">
        ${renderUploadSection()}
        <div id="task-list-wrapper">${renderTaskSection([], 0)}</div>
        <div id="docs-list-wrapper">${spinnerHtml()}</div>
      </div>
    `;

    bindUploadEvents();

    // Initial data loads in parallel
    loadTasks();
    loadDocs();

    // Smart auto-refresh: fast (3s) when tasks active, slow (15s) when idle
    function scheduleRefresh() {
      if (state.refreshTimers.memory) clearInterval(state.refreshTimers.memory);
      const interval = _hasActiveTasks ? 3000 : 15000;
      state.refreshTimers.memory = setInterval(() => {
        refreshTasks();
      }, interval);
    }
    scheduleRefresh();

    // Visibility detection: pause polling when tab is hidden, resume when visible
    const _visHandler = () => {
      if (document.hidden) {
        if (state.refreshTimers.memory) {
          clearInterval(state.refreshTimers.memory);
          state.refreshTimers.memory = null;
        }
      } else {
        refreshTasks();
        scheduleRefresh();
      }
    };
    document.addEventListener('visibilitychange', _visHandler);
    // Store handler for cleanup
    state._memoryVisHandler = _visHandler;

    // Re-render icons
    if (window.lucide) lucide.createIcons();
  }

  function destroy() {
    // Timers are cleared by the router's handleRoute, but we can also clear our own
    if (state.refreshTimers.memory) {
      clearInterval(state.refreshTimers.memory);
      delete state.refreshTimers.memory;
    }
    if (state._memoryVisHandler) {
      document.removeEventListener('visibilitychange', state._memoryVisHandler);
      delete state._memoryVisHandler;
    }
  }

  registerPage('memory', { render, destroy });
})();

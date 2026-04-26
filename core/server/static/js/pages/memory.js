/* ==========================================
   Memory Management Page
   Upload text/files, monitor tasks, browse docs
   ========================================== */

(function() {
  // ---- Smart refresh state ----
  let _hasActiveTasks = true;

  // ---- Helpers ----

  function progressClass(status) {
    if (status === 'completed') return 'success';
    if (status === 'failed') return 'error';
    return '';
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
            <input type="file" id="file-input" multiple style="display:none;">
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

    let tableHtml;
    if (!tasks || tasks.length === 0) {
      tableHtml = emptyState(t('memory.noTasks'));
    } else {
      const rows = tasks.map(task => {
        const pCls = progressClass(task.status);
        const elapsed = getElapsed(task.started_at || task.created_at, task.finished_at);
        const isRunning = task.status === 'running';
        const isPaused = task.status === 'paused';
        const isPausePending = task.phase === 'pausing';
        const loadCacheLabel = task.load_cache_memory ? t('memory.loadCacheOn') : t('memory.loadCacheOff');
        const canDelete = task.status === 'queued' || task.status === 'running' || task.status === 'paused';
        const canPause = task.status === 'running' && !isPausePending;
        const canResume = task.status === 'paused';
        const hasTriple = isRunning;
        const s6p = Math.min(1, Math.max(0, task.step6_progress ?? 0));
        const s7p = Math.min(1, Math.max(0, task.step7_progress ?? 0));
        const smp = Math.min(1, Math.max(0, task.main_progress ?? 0));
        const overallP = Math.min(1, Math.max(0, (smp + s6p + s7p) / 3));
        let progressCell;
        if (hasTriple) {
          progressCell = tripleProgressBar({
            smp, s6p, s7p,
            mainLabel: task.main_label || '-',
            step6Label: task.step6_label || '-',
            step7Label: task.step7_label || '-',
            showOverall: true,
            overallP: overallP,
          });
        } else {
          progressCell = `<div style="min-width:100px;">${progressBar(task.progress, pCls)}</div>`;
        }
        return `
          <tr data-task-id="${escapeHtml(task.task_id)}" title="${t('memory.taskDetail')}">
            <td><span class="mono" title="${escapeHtml(task.task_id)}">${escapeHtml(truncate(task.task_id, 12))}</span></td>
            <td>${escapeHtml(truncate(task.source_name || '-', 24))}</td>
            <td>${escapeHtml(loadCacheLabel)}</td>
            <td>${statusBadge(task.status)}</td>
            <td>${progressCell}</td>
            <td>${escapeHtml(task.phase_label || '-')}</td>
            <td>${elapsed}</td>
            <td>
              ${isPausePending ? `
                <button class="btn btn-secondary btn-sm" disabled style="margin-right:0.35rem;opacity:0.75;cursor:not-allowed;">
                  <i data-lucide="pause" style="width:14px;height:14px;"></i>
                  ${t('memory.pausePending')}
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

      tableHtml = `
        <div class="table-container">
          <table class="data-table">
            <thead>
              <tr>
                <th>${t('memory.taskId')}</th>
                <th>${t('memory.taskSource')}</th>
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

    return `
      <div class="card" style="margin-bottom:1rem;">
        <div class="card-header">
          <span class="card-title">${t('memory.processQueue')}${badge}</span>
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
    try {
      const res = await state.api.rememberDelete(taskId, state.currentGraphId);
      showToast(res.data?.message || t('memory.deleteTaskSuccess'), 'success');
      refreshTasks();
    } catch (err) {
      showToast(t('memory.deleteTaskFailed') + ': ' + err.message, 'error');
    }
  }

  async function pauseTask(taskId) {
    if (!taskId) return;
    try {
      const res = await state.api.rememberPause(taskId, state.currentGraphId);
      showToast(res.data?.message || t('memory.pauseTaskSuccess'), 'success');
      refreshTasks();
    } catch (err) {
      showToast(t('memory.pauseTaskFailed') + ': ' + err.message, 'error');
    }
  }

  async function resumeTask(taskId) {
    if (!taskId) return;
    try {
      const res = await state.api.rememberResume(taskId, state.currentGraphId);
      showToast(res.data?.message || t('memory.resumeTaskSuccess'), 'success');
      refreshTasks();
    } catch (err) {
      showToast(t('memory.resumeTaskFailed') + ': ' + err.message, 'error');
    }
  }

  async function showTaskDetail(taskId) {
    try {
      const res = await state.api.rememberStatus(taskId, state.currentGraphId);
      const task = res.data;

      const pCls = progressClass(task.status);
      const isRunning = task.status === 'running';
      const isPausePending = task.phase === 'pausing';
      const hasTriple = isRunning;
      const loadCacheLabel = task.load_cache_memory ? t('memory.loadCacheOn') : t('memory.loadCacheOff');
      const s6p = Math.min(1, Math.max(0, task.step6_progress ?? 0));
      const s7p = Math.min(1, Math.max(0, task.step7_progress ?? 0));
      const smp = Math.min(1, Math.max(0, task.main_progress ?? 0));
      const overallPd = Math.min(1, Math.max(0, (smp + s6p + s7p) / 3));
      let progressDetail;
      if (hasTriple) {
        progressDetail = tripleProgressBar({
          smp, s6p, s7p,
          mainLabel: task.main_label || '-',
          step6Label: task.step6_label || '-',
          step7Label: task.step7_label || '-',
          showOverall: true,
          overallP: overallPd,
        });
      } else {
        progressDetail = `<div style="min-width:120px;">${progressBar(task.progress, pCls)}</div>`;
      }
      let body = `
        <div style="margin-bottom:1rem;">
          <div style="display:grid;grid-template-columns:auto 1fr;gap:0.5rem 1rem;font-size:0.8125rem;">
            <span style="color:var(--text-muted);">${t('memory.taskId')}</span>
            <span class="mono">${escapeHtml(task.task_id)}</span>
            <span style="color:var(--text-muted);">${t('memory.taskStatus')}</span>
            <span>${statusBadge(task.status)}</span>
            <span style="color:var(--text-muted);">${t('memory.taskLoadCache')}</span>
            <span>${escapeHtml(loadCacheLabel)}</span>
            <span style="color:var(--text-muted);">${t('memory.taskProgress')}</span>
            ${progressDetail}
            <span style="color:var(--text-muted);">${t('memory.taskPhase')}</span>
            <span>${escapeHtml(task.phase_label || '-')}</span>
            <span style="color:var(--text-muted);">${t('memory.taskCreated')}</span>
            <span>${formatDate(task.created_at)}</span>
            <span style="color:var(--text-muted);">${t('memory.taskStarted')}</span>
            <span>${formatDate(task.started_at)}</span>
          </div>
        </div>
      `;

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
            <i data-lucide="pause" style="width:14px;height:14px;"></i>
            ${t('memory.pausePending')}
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

  function renderDocsTableHtml() {
    const total = _allDocs.length;
    if (total === 0) return emptyState(t('memory.noDocs'));

    const totalPages = Math.max(1, Math.ceil(total / _docsPageSize));
    if (_docsPage > totalPages) _docsPage = totalPages;
    const start = (_docsPage - 1) * _docsPageSize;
    const pageDocs = _allDocs.slice(start, start + _docsPageSize);

    const rows = pageDocs.map(d => {
      const filename = d.filename || d.doc_hash || '';
      return `
        <tr>
          <td><span class="mono" title="${escapeHtml(d.doc_hash)}">${escapeHtml(d.doc_hash || '-')}</span></td>
          <td>${escapeHtml(truncate(d.source_document || d.doc_name || '-', 32))}</td>
          <td>${formatDate(d.event_time)}</td>
          <td>${formatDateMs(d.processed_time)}</td>
          <td class="mono">${d.text_length != null ? d.text_length.toLocaleString() : '-'}</td>
          <td><button class="btn btn-secondary btn-sm doc-detail-btn" data-filename="${escapeHtml(filename)}" ${!filename ? 'disabled' : ''}>${t('common.detail')}</button></td>
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
              <th>${t('memory.docHash')}</th>
              <th>${t('memory.taskSource')}</th>
              <th>${t('memory.docTime')}</th>
              <th>${t('memory.processedTime')}</th>
              <th>${t('memory.docTextLength')}</th>
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

  function bindDocsEvents() {
    if (window.lucide) lucide.createIcons({ nodes: [document.getElementById('docs-list-wrapper')] });
    bindDocsPagination();
    document.querySelectorAll('.doc-detail-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const filename = btn.getAttribute('data-filename');
        if (filename) showDocContent(filename);
      });
    });
  }

  async function loadDocs() {
    try {
      const res = await state.api.listDocs(state.currentGraphId);
      _allDocs = res.data?.docs || [];
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

      const sourceName = meta.source_document || meta.doc_name || filename;
      const eventTime = meta.event_time || '-';
      const original = data.original || '';
      const cache = data.cache || '';

      let body = `
        <div style="display:flex;flex-direction:column;gap:1rem;">
          <div style="display:grid;grid-template-columns:auto 1fr;gap:0.25rem 0.75rem;font-size:0.85rem;">
            <span style="color:var(--text-secondary);">${t('memory.taskSource')}:</span><span>${escapeHtml(sourceName)}</span>
            <span style="color:var(--text-secondary);">${t('memory.docTime')}:</span><span>${formatDate(eventTime)}</span>
          </div>
      `;

      if (cache) {
        body += `
          <div>
            <h4 style="margin-bottom:0.5rem;">${t('memory.cacheSummary')}</h4>
            <div style="max-height:400px;overflow-y:auto;background:var(--bg-secondary);padding:0.75rem;border-radius:0.5rem;font-size:0.85rem;line-height:1.6;white-space:pre-wrap;word-break:break-word;">${escapeHtml(cache)}</div>
          </div>
        `;
      }

      if (original) {
        body += `
          <div>
            <h4 style="margin-bottom:0.5rem;">${t('memory.originalText')}</h4>
            <div style="max-height:400px;overflow-y:auto;background:var(--bg-secondary);padding:0.75rem;border-radius:0.5rem;font-size:0.85rem;line-height:1.6;white-space:pre-wrap;word-break:break-word;">${escapeHtml(original)}</div>
          </div>
        `;
      }

      body += '</div>';

      showModal({
        title: t('memory.docContent') + ' - ' + escapeHtml(truncate(sourceName, 30)),
        content: body,
        size: 'lg',
      });
    } catch (err) {
      showToast(t('memory.loadDocContentFailed') + ': ' + err.message, 'error');
    }
  }

  // ---- Page Lifecycle ----

  async function render(container, params) {
    container.innerHTML = `
      <div class="page-enter">
        ${renderUploadSection()}
        <div id="task-list-wrapper">${spinnerHtml()}</div>
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

    // Re-render icons
    if (window.lucide) lucide.createIcons();
  }

  function destroy() {
    // Timers are cleared by the router's handleRoute, but we can also clear our own
    if (state.refreshTimers.memory) {
      clearInterval(state.refreshTimers.memory);
      delete state.refreshTimers.memory;
    }
  }

  registerPage('memory', { render, destroy });
})();

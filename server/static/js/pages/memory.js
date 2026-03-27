/* ==========================================
   Memory Management Page
   Upload text/files, monitor tasks, browse docs
   ========================================== */

(function() {
  // ---- Helpers ----

  function getElapsed(startedAt, finishedAt) {
    if (!startedAt) return '-';
    let start = Number(startedAt);
    if (isNaN(start)) return '-';
    // Unix timestamp in seconds vs ISO string in ms
    if (start < 4102444800000) start *= 1000;

    let end;
    if (finishedAt) {
      end = Number(finishedAt);
      if (end < 4102444800000) end *= 1000;
    } else {
      end = Date.now();
    }

    const diff = Math.max(0, Math.round((end - start) / 1000));
    return formatRelativeTime(diff);
  }

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
              <div class="toggle active" id="text-load-cache-toggle"></div>
              <label class="form-label" style="margin:0;cursor:pointer;" for="text-load-cache-toggle">${t('memory.loadCache')}</label>
            </div>
          </div>
          <div style="margin-top:1rem;display:flex;justify-content:flex-end;">
            <button class="btn btn-primary" id="btn-submit-text">
              <i data-lucide="send" style="width:16px;height:16px;"></i>
              ${t('memory.submitMemory')}
            </button>
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
              <div class="toggle active" id="file-load-cache-toggle"></div>
              <label class="form-label" style="margin:0;cursor:pointer;" for="file-load-cache-toggle">${t('memory.loadCache')}</label>
            </div>
          </div>
          <div style="margin-top:1rem;display:flex;justify-content:space-between;align-items:center;">
            <button class="btn btn-secondary btn-sm" id="btn-clear-files" style="display:none;">
              <i data-lucide="x" style="width:14px;height:14px;"></i>
              ${t('memory.clearFiles')}
            </button>
            <div style="flex:1;"></div>
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
    document.getElementById('text-load-cache-toggle').addEventListener('click', function() {
      this.classList.toggle('active');
    });
    document.getElementById('file-load-cache-toggle').addEventListener('click', function() {
      this.classList.toggle('active');
    });

    // Text submit
    document.getElementById('btn-submit-text').addEventListener('click', submitText);

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
      const loadCache = document.getElementById('file-load-cache-toggle').classList.contains('active');

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
    const loadCache = document.getElementById('text-load-cache-toggle').classList.contains('active');

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
        const hasDual = isRunning && (task.step6_label || task.step7_label);
        const s6p = Math.min(1, Math.max(0, task.step6_progress ?? 0));
        const s7p = Math.min(1, Math.max(0, task.step7_progress ?? 0));
        let progressCell;
        if (hasDual) {
          progressCell = `
            <div style="min-width:140px;">
              <div style="display:grid;grid-template-columns:1fr 1fr;gap:2px 6px;">
                <div>
                  <div style="font-size:0.6rem;color:var(--info);">${t('dashboard.entityAlign')}</div>
                  <div class="progress-bar" style="height:2px;"><div class="progress-bar-fill" style="width:${(s6p*100).toFixed(1)}%;background:var(--info);"></div></div>
                </div>
                <div>
                  <div style="font-size:0.6rem;color:var(--warning);">${t('dashboard.relationAlign')}</div>
                  <div class="progress-bar" style="height:2px;"><div class="progress-bar-fill" style="width:${(s7p*100).toFixed(1)}%;background:var(--warning);"></div></div>
                </div>
              </div>
            </div>`;
        } else {
          progressCell = `<div style="min-width:100px;">${progressBar(task.progress, pCls)}</div>`;
        }
        return `
          <tr data-task-id="${escapeHtml(task.task_id)}" title="${t('memory.taskDetail')}">
            <td><span class="mono" title="${escapeHtml(task.task_id)}">${escapeHtml(truncate(task.task_id, 12))}</span></td>
            <td>${escapeHtml(truncate(task.source_name || '-', 24))}</td>
            <td>${statusBadge(task.status)}</td>
            <td>${progressCell}</td>
            <td>${escapeHtml(task.phase_label || '-')}</td>
            <td>${elapsed}</td>
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
                <th>${t('memory.taskStatus')}</th>
                <th>${t('memory.taskProgress')}</th>
                <th>${t('memory.taskPhase')}</th>
                <th>${t('memory.taskElapsed')}</th>
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

      if (window.lucide) lucide.createIcons({ nodes: [el] });
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

  async function showTaskDetail(taskId) {
    try {
      const res = await state.api.rememberStatus(taskId, state.currentGraphId);
      const task = res.data;

      const pCls = progressClass(task.status);
      const isRunning = task.status === 'running';
      const hasDual = isRunning && (task.step6_label || task.step7_label);
      const s6p = Math.min(1, Math.max(0, task.step6_progress ?? 0));
      const s7p = Math.min(1, Math.max(0, task.step7_progress ?? 0));
      let progressDetail;
      if (hasDual) {
        progressDetail = `
          <div style="min-width:200px;">
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:4px 12px;">
              <div>
                <div style="font-size:0.7rem;color:var(--info);margin-bottom:2px;">${t('dashboard.entityAlign')}</div>
                <div class="progress-bar" style="height:3px;"><div class="progress-bar-fill" style="width:${(s6p*100).toFixed(1)}%;background:var(--info);"></div></div>
                <div style="font-size:0.65rem;color:var(--text-muted);margin-top:1px;">${escapeHtml(task.step6_label || '-')}</div>
              </div>
              <div>
                <div style="font-size:0.7rem;color:var(--warning);margin-bottom:2px;">${t('dashboard.relationAlign')}</div>
                <div class="progress-bar" style="height:3px;"><div class="progress-bar-fill" style="width:${(s7p*100).toFixed(1)}%;background:var(--warning);"></div></div>
                <div style="font-size:0.65rem;color:var(--text-muted);margin-top:1px;">${escapeHtml(task.step7_label || '-')}</div>
              </div>
            </div>
          </div>`;
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

      showModal({
        title: t('memory.taskDetail'),
        content: body,
        size: 'lg',
      });
    } catch (err) {
      showToast(t('memory.taskDetailFailed') + ': ' + err.message, 'error');
    }
  }

  // ---- Documents Section ----

  function renderDocsSection(docs, count) {
    const badge = `<span class="badge badge-primary" style="margin-left:0.5rem;">${escapeHtml(String(count ?? 0))}</span>`;

    let tableHtml;
    if (!docs || docs.length === 0) {
      tableHtml = emptyState(t('memory.noDocs'));
    } else {
      const rows = docs.map(d => {
        return `
          <tr>
            <td><span class="mono" title="${escapeHtml(d.doc_hash)}">${escapeHtml(truncate(d.doc_hash, 16))}</span></td>
            <td>${escapeHtml(truncate(d.source_document || d.doc_name || '-', 32))}</td>
            <td>${formatDate(d.physical_time)}</td>
            <td class="mono">${d.text_length != null ? d.text_length.toLocaleString() : '-'}</td>
            <td>${escapeHtml(d.activity_type || '-')}</td>
          </tr>
        `;
      }).join('');

      tableHtml = `
        <div class="table-container">
          <table class="data-table">
            <thead>
              <tr>
                <th>${t('memory.docHash')}</th>
                <th>${t('memory.taskSource')}</th>
                <th>${t('memory.docTime')}</th>
                <th>${t('memory.docTextLength')}</th>
                <th>${t('memory.docActivity')}</th>
              </tr>
            </thead>
            <tbody>${rows}</tbody>
          </table>
        </div>
      `;
    }

    return `
      <div class="card">
        <div class="card-header">
          <span class="card-title">${t('memory.docs')}${badge}</span>
        </div>
        <div id="docs-list">${tableHtml}</div>
      </div>
    `;
  }

  async function loadDocs() {
    try {
      const res = await state.api.listDocs(state.currentGraphId);
      const docs = res.data?.docs || [];
      const count = res.data?.count ?? docs.length;

      const el = document.getElementById('docs-list-wrapper');
      if (!el) return;

      el.innerHTML = renderDocsSection(docs, count);
      if (window.lucide) lucide.createIcons({ nodes: [el] });
    } catch (err) {
      const el = document.getElementById('docs-list-wrapper');
      if (el) {
        el.innerHTML = `<div class="card"><div class="empty-state"><p style="color:var(--error);">${t('memory.loadDocsFailed')}: ${escapeHtml(err.message)}</p></div></div>`;
      }
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

    // Auto-refresh task list every 1 second
    state.refreshTimers.memory = setInterval(() => {
      refreshTasks();
    }, 1000);

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

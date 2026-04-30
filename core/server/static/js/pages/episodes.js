/* ==========================================
   Episodes Page - Processing Records
   ========================================== */
registerPage('episodes', (function () {
  'use strict';

  let _container = null;
  let _episodes = [];
  let _total = 0;
  let _offset = 0;
  let _limit = 50;
  let _loading = false;
  let _allLoaded = false;
  let _batchRows = [{ content: '', source_document: '', episode_type: '' }];

  async function _loadEpisodes(offset, append) {
    if (_loading) return;
    _loading = true;
    if (!append) {
      _offset = offset || 0;
      _episodes = [];
      _allLoaded = false;
      _renderList();
    }
    try {
      const res = await state.api.listEpisodes(state.currentGraphId, _limit, offset || 0);
      const data = res.data || {};
      const newEpisodes = data.episodes || [];
      _total = data.total || 0;

      if (append) {
        _episodes = _episodes.concat(newEpisodes);
      } else {
        _episodes = newEpisodes;
      }
      _offset = (offset || 0) + newEpisodes.length;
      _allLoaded = _episodes.length >= _total;
      _renderList();
    } catch (err) {
      const listEl = _container.querySelector('#ep-list');
      if (listEl) {
        listEl.innerHTML =
          `<div class="empty-state"><i data-lucide="alert-triangle"></i><p>${escapeHtml(err.message)}</p></div>`;
      }
    } finally {
      _loading = false;
    }
  }

  async function _searchEpisodes(query) {
    if (!query || _loading) return;
    _loading = true;
    _allLoaded = true; // search mode: disable infinite scroll
    _renderList();
    try {
      const res = await state.api.searchEpisodes(query, state.currentGraphId, 50);
      _episodes = res.data?.episodes || [];
      _total = _episodes.length;
      _offset = 0;
      _renderList();
    } catch (err) {
      const listEl = _container.querySelector('#ep-list');
      if (listEl) {
        listEl.innerHTML =
          `<div class="empty-state"><i data-lucide="alert-triangle"></i><p>${escapeHtml(err.message)}</p></div>`;
      }
    } finally {
      _loading = false;
    }
  }

  function _renderList() {
    const listEl = _container.querySelector('#ep-list');
    if (!listEl) return;

    if (_loading && _episodes.length === 0) {
      listEl.innerHTML = `<div class="flex items-center justify-center p-8">${spinnerHtml()}</div>`;
      if (window.lucide) lucide.createIcons();
      return;
    }

    if (_episodes.length === 0) {
      listEl.innerHTML = emptyState(t('episodes.noEpisodes'), 'film');
      if (window.lucide) lucide.createIcons();
      return;
    }

    let html = '<div class="space-y-3">';
    for (const ep of _episodes) {
      html += `
        <div class="card p-4 cursor-pointer" data-uuid="${escapeHtml(ep.uuid)}" onclick="window._epShowDetail('${escapeHtml(ep.uuid)}')">
          <div class="flex items-start justify-between gap-3">
            <div class="flex-1 min-w-0">
              <div class="flex items-center gap-2 mb-1">
                ${ep.episode_type ? `<span class="badge badge-secondary">${escapeHtml(ep.episode_type)}</span>` : ''}
                ${ep.source_document ? `<span class="badge badge-info">${escapeHtml(truncate(ep.source_document, 30))}</span>` : ''}
                ${ep.event_time ? `<span class="text-xs mono" style="color:var(--text-muted);">${formatDate(ep.event_time)}</span>` : ''}
                ${ep.processed_time ? `<span class="text-xs mono" style="color:var(--text-muted);margin-left:0.25rem;">${formatDateMs(ep.processed_time)}</span>` : ''}
              </div>
              <p class="text-sm" style="color:var(--text-primary);line-height:1.5;">${escapeHtml(truncate(ep.content, 200))}</p>
            </div>
            <div class="flex items-center gap-1 flex-shrink-0">
              <button class="btn btn-ghost btn-sm" title="${t('common.detail')}" onclick="event.stopPropagation();window._epShowDetail('${escapeHtml(ep.uuid)}')">
                <i data-lucide="eye" style="width:16px;height:16px;"></i>
              </button>
              <button class="btn btn-ghost btn-sm" title="${t('episodes.delete')}" onclick="event.stopPropagation();window._epDelete('${escapeHtml(ep.uuid)}')">
                <i data-lucide="trash-2" style="width:16px;height:16px;"></i>
              </button>
            </div>
          </div>
        </div>`;
    }
    html += '</div>';

    // Load-more sentinel for infinite scroll
    if (!_allLoaded) {
      html += `<div id="ep-sentinel" style="padding:1rem;text-align:center;color:var(--text-muted);font-size:0.85rem;">
        ${_loading ? `${spinnerHtml('spinner-sm')} ${t('common.loading')}` : ''}
      </div>`;
    } else {
      html += `<div style="padding:0.75rem;text-align:center;color:var(--text-muted);font-size:0.8125rem;">
        ${t('episodes.showing')} ${_episodes.length} / ${_total}
      </div>`;
    }

    listEl.innerHTML = html;
    if (window.lucide) lucide.createIcons();

    // Observe sentinel for infinite scroll
    const sentinel = listEl.querySelector('#ep-sentinel');
    if (sentinel) {
      _observeSentinel(sentinel);
    }
  }

  let _sentinelObserver = null;

  function _observeSentinel(sentinel) {
    if (_sentinelObserver) _sentinelObserver.disconnect();
    _sentinelObserver = new IntersectionObserver((entries) => {
      if (entries[0].isIntersecting && !_loading && !_allLoaded) {
        _loadEpisodes(_offset, true);
      }
    }, { rootMargin: '200px' });
    _sentinelObserver.observe(sentinel);
  }

  async function _deleteEpisode(uuid) {
    const ok = await showConfirm({ message: t('episodes.deleteConfirm'), destructive: true });
    if (!ok) return;
    try {
      await state.api.deleteEpisode(uuid, state.currentGraphId);
      showToast(t('episodes.deleteSuccess'), 'success');
      _episodes = _episodes.filter(e => e.uuid !== uuid);
      _total--;
      _allLoaded = false;
      _renderList();
    } catch (err) {
      showToast(t('episodes.deleteFailed') + ': ' + err.message, 'error');
    }
  }

  // ---- Batch Import ----

  function _renderBatchImportModal() {
    const rows = _batchRows.map((row, i) => `
      <div class="batch-import-row" data-row="${i}" style="display:grid;grid-template-columns:1fr 200px 120px auto;gap:8px;align-items:start;margin-bottom:8px;">
        <textarea class="input batch-content" rows="2" placeholder="${t('episodes.contentPlaceholder')}">${escapeHtml(row.content)}</textarea>
        <input type="text" class="input batch-source" placeholder="${t('episodes.sourcePlaceholder')}" value="${escapeHtml(row.source_document)}">
        <select class="input batch-type">
          <option value="" ${!row.episode_type ? 'selected' : ''}>--</option>
          <option value="narrative" ${row.episode_type === 'narrative' ? 'selected' : ''}>Narrative</option>
          <option value="fact" ${row.episode_type === 'fact' ? 'selected' : ''}>Fact</option>
          <option value="conversation" ${row.episode_type === 'conversation' ? 'selected' : ''}>Conversation</option>
          <option value="dream" ${row.episode_type === 'dream' ? 'selected' : ''}>Dream</option>
        </select>
        ${_batchRows.length > 1 ? `<button class="btn btn-ghost btn-sm" onclick="window._epRemoveBatchRow(${i})" title="${t('common.remove')}"><i data-lucide="x" style="width:14px;height:14px;"></i></button>` : '<div style="width:32px;"></div>'}
      </div>
    `).join('');

    const html = `
      <div>
        <p style="font-size:0.85rem;color:var(--text-muted);margin-bottom:12px;">${t('episodes.batchImportHint')}</p>
        <div id="batch-import-rows">${rows}</div>
        <div style="margin-top:8px;">
          <button class="btn btn-secondary btn-sm" onclick="window._epAddBatchRow()">
            <i data-lucide="plus" style="width:14px;height:14px;margin-right:4px;"></i>${t('episodes.addEpisode')}
          </button>
        </div>
      </div>
    `;

    const footer = `
      <button class="btn btn-secondary modal-cancel-btn">${t('common.cancel')}</button>
      <button class="btn btn-primary modal-save-btn">${t('common.submit')}</button>
    `;

    const { overlay, close } = showModal({
      title: t('episodes.batchImport'),
      content: html,
      footer: footer,
      size: 'lg',
    });

    if (window.lucide) lucide.createIcons({ nodes: [overlay] });

    overlay.querySelector('.modal-cancel-btn').addEventListener('click', close);
    overlay.querySelector('.modal-save-btn').addEventListener('click', async () => {
      _syncBatchRows(overlay);
      const episodes = _batchRows
        .filter(r => r.content.trim())
        .map(r => ({
          content: r.content.trim(),
          source_document: r.source_document.trim() || undefined,
          episode_type: r.episode_type || undefined,
        }));
      if (episodes.length === 0) {
        showToast(t('episodes.noContent'), 'warning');
        return;
      }
      const saveBtn = overlay.querySelector('.modal-save-btn');
      saveBtn.disabled = true;
      saveBtn.innerHTML = `${spinnerHtml('spinner-sm')} ${t('common.loading')}`;
      try {
        const res = await state.api.batchIngestEpisodes(episodes, state.currentGraphId);
        const imported = res.data?.imported || res.data?.count || episodes.length;
        showToast(t('episodes.importSuccess', { count: imported }), 'success');
        close();
        _loadEpisodes(0, false);
      } catch (err) {
        showToast(t('episodes.importFailed') + ': ' + err.message, 'error');
      } finally {
        saveBtn.disabled = false;
        saveBtn.textContent = t('common.submit');
      }
    });
  }

  function _syncBatchRows(overlay) {
    if (!overlay) return;
    overlay.querySelectorAll('.batch-import-row').forEach((rowEl, i) => {
      if (_batchRows[i]) {
        _batchRows[i].content = rowEl.querySelector('.batch-content')?.value || '';
        _batchRows[i].source_document = rowEl.querySelector('.batch-source')?.value || '';
        _batchRows[i].episode_type = rowEl.querySelector('.batch-type')?.value || '';
      }
    });
  }

  // Expose to onclick handlers
  window._epShowDetail = function(uuid) { window.showEpisodeDetailModal(uuid); };
  window._epDelete = _deleteEpisode;
  window._epLoad = function(offset) { _loadEpisodes(offset, false); };
  window._epDoSearch = null;
  window._epAddBatchRow = function () {
    _batchRows.push({ content: '', source_document: '', episode_type: '' });
    // Re-render the batch modal content
    const rowsContainer = document.querySelector('#batch-import-rows');
    if (rowsContainer) {
      const i = _batchRows.length - 1;
      const newRow = document.createElement('div');
      newRow.className = 'batch-import-row';
      newRow.dataset.row = i;
      newRow.style.cssText = 'display:grid;grid-template-columns:1fr 200px 120px auto;gap:8px;align-items:start;margin-bottom:8px;';
      newRow.innerHTML = `
        <textarea class="input batch-content" rows="2" placeholder="${t('episodes.contentPlaceholder')}"></textarea>
        <input type="text" class="input batch-source" placeholder="${t('episodes.sourcePlaceholder')}">
        <select class="input batch-type">
          <option value="">--</option>
          <option value="narrative">Narrative</option>
          <option value="fact">Fact</option>
          <option value="conversation">Conversation</option>
          <option value="dream">Dream</option>
        </select>
        <button class="btn btn-ghost btn-sm" onclick="window._epRemoveBatchRow(${i})" title="${t('common.remove')}"><i data-lucide="x" style="width:14px;height:14px;"></i></button>
      `;
      rowsContainer.appendChild(newRow);
      if (window.lucide) lucide.createIcons({ nodes: [newRow] });
      newRow.querySelector('.batch-content').focus();
    }
  };
  window._epRemoveBatchRow = function (idx) {
    if (_batchRows.length <= 1) return;
    _batchRows.splice(idx, 1);
    // Close and reopen modal
    _renderBatchImportModal();
  };

  async function render(container, params) {
    _container = container;
    container.innerHTML = `
      <div class="space-y-4">
        <!-- Search bar -->
        <div class="flex gap-2">
          <div class="flex-1 relative">
            <input type="text" id="ep-search" class="input w-full" placeholder="${t('episodes.searchPlaceholder')}"
              onkeydown="if(event.key==='Enter'){window._epDoSearch();}">
          </div>
          <button class="btn btn-primary btn-sm" onclick="window._epDoSearch()">
            <i data-lucide="search" style="width:16px;height:16px;"></i>${t('common.search')}
          </button>
          <button class="btn btn-secondary btn-sm" onclick="window._epLoad(0)">
            <i data-lucide="list" style="width:16px;height:16px;"></i>${t('episodes.listAll')}
          </button>
          <button class="btn btn-secondary btn-sm" onclick="window._epOpenBatchImport()">
            <i data-lucide="upload" style="width:16px;height:16px;"></i>${t('episodes.batchImport')}
          </button>
        </div>
        <!-- Episode list -->
        <div id="ep-list"></div>
      </div>
    `;

    window._epDoSearch = function () {
      const q = document.getElementById('ep-search')?.value?.trim();
      if (q) _searchEpisodes(q);
    };

    window._epOpenBatchImport = function () {
      _batchRows = [{ content: '', source_document: '', episode_type: '' }];
      _renderBatchImportModal();
    };

    await _loadEpisodes(0, false);
  }

  function destroy() {
    if (_sentinelObserver) {
      _sentinelObserver.disconnect();
      _sentinelObserver = null;
    }
    _container = null;
    _episodes = [];
    _batchRows = [{ content: '', source_document: '', episode_type: '' }];
    delete window._epShowDetail;
    delete window._epDelete;
    delete window._epLoad;
    delete window._epDoSearch;
    delete window._epOpenBatchImport;
    delete window._epAddBatchRow;
    delete window._epRemoveBatchRow;
  }

  return { render, destroy };
})());

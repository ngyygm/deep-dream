/* ==========================================
   Episodes Page - Episode Management
   ========================================== */
registerPage('episodes', (function () {
  'use strict';

  let _container = null;
  let _episodes = [];
  let _total = 0;
  let _offset = 0;
  let _limit = 20;
  let _loading = false;
  let _batchRows = [{ content: '', source_document: '', episode_type: '' }];

  async function _loadEpisodes(offset) {
    if (_loading) return;
    _loading = true;
    _offset = offset || 0;
    _renderList();
    try {
      const res = await state.api.listEpisodes(state.currentGraphId, _limit, _offset);
      const data = res.data || {};
      _episodes = data.episodes || [];
      _total = data.total || 0;
      _renderList();
    } catch (err) {
      _container.querySelector('#ep-list').innerHTML =
        `<div class="empty-state"><i data-lucide="alert-triangle"></i><p>${escapeHtml(err.message)}</p></div>`;
    } finally {
      _loading = false;
    }
  }

  async function _searchEpisodes(query) {
    if (!query || _loading) return;
    _loading = true;
    _renderList();
    try {
      const res = await state.api.searchEpisodes(query, state.currentGraphId, 50);
      _episodes = res.data?.episodes || [];
      _total = _episodes.length;
      _offset = 0;
      _renderList();
    } catch (err) {
      _container.querySelector('#ep-list').innerHTML =
        `<div class="empty-state"><i data-lucide="alert-triangle"></i><p>${escapeHtml(err.message)}</p></div>`;
    } finally {
      _loading = false;
    }
  }

  function _renderList() {
    const listEl = _container.querySelector('#ep-list');
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
                ${ep.event_time ? `<span class="text-xs" style="color:var(--text-muted);">${formatDate(ep.event_time)}</span>` : ''}
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

    // Pagination
    if (_total > _limit) {
      html += `<div class="flex items-center justify-between mt-4 text-sm" style="color:var(--text-muted);">
        <span>${t('episodes.showing')} ${_offset + 1}-${Math.min(_offset + _limit, _total)} / ${_total}</span>
        <div class="flex gap-2">
          <button class="btn btn-secondary btn-sm" ${_offset <= 0 ? 'disabled' : ''} onclick="window._epLoad(${_offset - _limit})">${t('common.loadMore')}</button>
          <button class="btn btn-secondary btn-sm" ${_offset + _limit >= _total ? 'disabled' : ''} onclick="window._epLoad(${_offset + _limit})">${t('episodes.next')}</button>
        </div>
      </div>`;
    }

    listEl.innerHTML = html;
    if (window.lucide) lucide.createIcons();
  }

  async function _showDetail(uuid) {
    try {
      const [epRes, entRes] = await Promise.all([
        state.api.getEpisode(uuid, state.currentGraphId),
        state.api.getEpisodeEntities(uuid, state.currentGraphId),
      ]);
      const ep = epRes.data || {};
      const entities = entRes.data?.entities || [];

      let body = `<div style="display:flex;flex-direction:column;gap:1rem;">`;
      body += `<div style="display:grid;grid-template-columns:auto 1fr;gap:0.25rem 0.75rem;font-size:0.85rem;">
        <span style="color:var(--text-secondary);">UUID:</span><span class="font-mono text-xs">${escapeHtml(ep.uuid || '')}</span>
        ${ep.episode_type ? `<span style="color:var(--text-secondary);">${t('episodes.episodeType')}:</span><span>${escapeHtml(ep.episode_type)}</span>` : ''}
        <span style="color:var(--text-secondary);">${t('common.source')}:</span><span>${escapeHtml(ep.source_document || '-')}</span>
        <span style="color:var(--text-secondary);">${t('common.time')}:</span><span>${formatDate(ep.event_time)}</span>
      </div>`;

      body += `<div>
        <h4 style="margin-bottom:0.5rem;">${t('common.content')}</h4>
        <div style="max-height:300px;overflow-y:auto;background:var(--bg-secondary);padding:0.75rem;border-radius:0.5rem;font-size:0.85rem;line-height:1.6;white-space:pre-wrap;word-break:break-word;">${escapeHtml(ep.content || '')}</div>
      </div>`;

      if (entities.length > 0) {
        body += `<div>
          <h4 style="margin-bottom:0.5rem;">${t('episodes.entities')} (${entities.length})</h4>
          <div class="space-y-1">`;
        for (const ent of entities) {
          body += `<div class="flex items-center gap-2 p-2 rounded cursor-pointer" style="background:var(--bg-secondary);font-size:0.85rem;" onclick="window.showEntityDetail(${JSON.stringify(ent).replace(/"/g, '&quot;')})">
            <i data-lucide="circle-dot" style="width:14px;height:14px;color:var(--primary);flex-shrink:0;"></i>
            <span class="font-medium">${escapeHtml(ent.name || ent.entity_id || '-')}</span>
            ${ent.content ? `<span class="text-xs" style="color:var(--text-muted);">${escapeHtml(truncate(ent.content, 60))}</span>` : ''}
          </div>`;
        }
        body += '</div></div>';
      } else {
        body += `<div><p style="color:var(--text-muted);font-size:0.85rem;">${t('episodes.noEntities')}</p></div>`;
      }

      body += '</div>';

      showModal({
        title: t('episodes.detail'),
        content: body,
        size: 'lg',
      });
      if (window.lucide) lucide.createIcons();
    } catch (err) {
      showToast(err.message, 'error');
    }
  }

  async function _deleteEpisode(uuid) {
    if (!confirm(t('episodes.deleteConfirm'))) return;
    try {
      await state.api.deleteEpisode(uuid, state.currentGraphId);
      showToast(t('episodes.deleteSuccess'), 'success');
      _loadEpisodes(_offset);
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
        const res = await state.api.batchIngestEpisodes(episodes);
        const imported = res.data?.imported || res.data?.count || episodes.length;
        showToast(t('episodes.importSuccess', { count: imported }), 'success');
        close();
        _loadEpisodes(0);
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
  window._epShowDetail = _showDetail;
  window._epDelete = _deleteEpisode;
  window._epLoad = _loadEpisodes;
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

    await _loadEpisodes(0);
  }

  function destroy() {
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

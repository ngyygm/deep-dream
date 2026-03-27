/* ==========================================
   Entities Page - Entity Browser
   ========================================== */

(function() {
  const PAGE_SIZE = 50;
  let debounceTimer = null;
  let allEntities = [];
  let displayedCount = 0;
  let isSearchMode = false;
  let isSearchAllMode = false;

  // ---- Search & Filter Bar ----

  function buildSearchBar() {
    return `
      <div class="card">
        <div class="card-header">
          <div style="display:flex;align-items:center;gap:0.75rem;flex:1;">
            <input
              type="text"
              class="input"
              id="entity-search-input"
              placeholder="${t('entities.searchPlaceholder')}"
              style="max-width:400px;"
              autocomplete="off"
            />
            <button class="btn btn-secondary" id="entity-list-all-btn">
              <i data-lucide="list" style="width:16px;height:16px;"></i>
              ${t('entities.listAll')}
            </button>
          </div>
          <span id="entity-count" class="mono" style="font-size:0.8125rem;color:var(--text-muted);">${t('entities.entityCount', { count: 0 })}</span>
        </div>
      </div>
    `;
  }

  // ---- Entity Table ----

  function buildEntityTable(entities) {
    if (!entities || entities.length === 0) {
      return emptyState(t('entities.noEntities'), 'box');
    }

    const rows = entities.map(e => `
      <tr data-entity-id="${escapeHtml(e.entity_id)}" data-absolute-id="${escapeHtml(e.absolute_id)}">
        <td style="max-width:180px;font-weight:500;">${escapeHtml(e.name || '-')}</td>
        <td style="max-width:300px;" class="truncate" title="${escapeHtml(e.content || '')}">${escapeHtml(truncate(e.content || '', 60))}</td>
        <td style="white-space:nowrap;">${formatDate(e.physical_time)}</td>
        <td style="max-width:120px;" class="truncate" title="${escapeHtml(e.doc_name || e.source_document || '')}">${escapeHtml(e.doc_name || e.source_document || '-')}</td>
        <td style="text-align:center;">
          <span class="badge badge-info">${escapeHtml(String(e.version_count || '?'))}</span>
        </td>
      </tr>
    `).join('');

    return `
      <div class="card" style="margin-top:0.75rem;">
        <div class="table-container">
          <table class="data-table">
            <thead>
              <tr>
                <th>${t('entities.name')}</th>
                <th>${t('entities.content')}</th>
                <th>${t('entities.physicalTime')}</th>
                <th>${t('entities.source')}</th>
                <th style="text-align:center;">${t('entities.version')}</th>
              </tr>
            </thead>
            <tbody>${rows}</tbody>
          </table>
        </div>
        ${buildLoadMore()}
      </div>
    `;
  }

  function buildLoadMore() {
    if (displayedCount >= allEntities.length) return '';
    const remaining = allEntities.length - displayedCount;
    return `
      <div style="display:flex;justify-content:center;padding-top:0.75rem;">
        <button class="btn btn-ghost" id="entity-load-more-btn">
          ${t('common.loadMore')} (${t('common.remaining')} ${remaining} ${t('common.records')})
        </button>
      </div>
    `;
  }

  // ---- Entity Detail Modal ----

  async function openEntityDetail(entity) {
    const modalContent = document.createElement('div');
    modalContent.innerHTML = `
      <div style="display:flex;flex-direction:column;gap:0.75rem;">
        <div>
          <span style="font-size:0.75rem;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.05em;">${t('graph.entityId')}</span>
          <div class="mono" style="margin-top:0.125rem;">${escapeHtml(entity.entity_id)}</div>
        </div>
        <div>
          <span style="font-size:0.75rem;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.05em;">${t('graph.absoluteId')}</span>
          <div class="mono" style="margin-top:0.125rem;">${escapeHtml(entity.absolute_id)}</div>
        </div>
        <div>
          <span style="font-size:0.75rem;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.05em;">${t('common.name')}</span>
          <div style="margin-top:0.125rem;font-weight:600;">${escapeHtml(entity.name || '-')}</div>
        </div>
        <div>
          <span style="font-size:0.75rem;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.05em;">${t('common.content')}</span>
          <div style="margin-top:0.125rem;line-height:1.6;white-space:pre-wrap;word-break:break-word;">${escapeHtml(entity.content || '-')}</div>
        </div>
        <div style="display:flex;gap:2rem;">
          <div>
            <span style="font-size:0.75rem;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.05em;">${t('graph.physicalTime')}</span>
            <div class="mono" style="margin-top:0.125rem;">${formatDate(entity.physical_time)}</div>
          </div>
          <div>
            <span style="font-size:0.75rem;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.05em;">${t('graph.sourceDoc')}</span>
            <div style="margin-top:0.125rem;">${escapeHtml(entity.doc_name || entity.source_document || '-')}</div>
          </div>
        </div>
        ${entity.memory_cache_id ? `
        <div>
          <span style="font-size:0.75rem;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.05em;">${t('graph.memoryCacheId')}</span>
          <div class="mono" style="margin-top:0.125rem;">${escapeHtml(entity.memory_cache_id)}</div>
        </div>
        ` : ''}
      </div>

      <div class="divider"></div>

      <div id="entity-versions-section">
        <div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.75rem;">
          <i data-lucide="git-branch" style="width:16px;height:16px;color:var(--text-muted);"></i>
          <span style="font-size:0.875rem;font-weight:600;">${t('entities.versionHistory')}</span>
          <div class="spinner spinner-sm" id="versions-spinner"></div>
        </div>
        <div id="versions-container"></div>
      </div>

      <div class="divider"></div>

      <div id="entity-relations-section">
        <div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.75rem;">
          <i data-lucide="link" style="width:16px;height:16px;color:var(--text-muted);"></i>
          <span style="font-size:0.875rem;font-weight:600;">${t('entities.relations')}</span>
          <div class="spinner spinner-sm" id="relations-spinner"></div>
        </div>
        <div id="relations-container"></div>
      </div>
    `;

    const { overlay } = showModal({
      title: entity.name || entity.entity_id,
      content: modalContent.innerHTML,
      size: 'lg',
    });

    if (window.lucide) lucide.createIcons({ nodes: [overlay] });

    // Fetch versions and relations in parallel
    const graphId = state.currentGraphId;
    const entityId = entity.entity_id;

    try {
      const [versionsRes, relationsRes] = await Promise.all([
        state.api.entityVersions(entityId, graphId),
        state.api.entityRelations(entityId, graphId),
      ]);

      const vSpinner = overlay.querySelector('#versions-spinner');
      if (vSpinner) vSpinner.remove();
      const rSpinner = overlay.querySelector('#relations-spinner');
      if (rSpinner) rSpinner.remove();

      const versions = versionsRes.data || [];
      const relations = relationsRes.data || [];

      const versionsContainer = overlay.querySelector('#versions-container');
      if (versionsContainer) {
        versionsContainer.innerHTML = versions.length > 0
          ? buildVersionTimeline(versions)
          : `<div style="color:var(--text-muted);font-size:0.8125rem;">${t('entities.noVersionHistory')}</div>`;
      }

      const relationsContainer = overlay.querySelector('#relations-container');
      if (relationsContainer) {
        relationsContainer.innerHTML = relations.length > 0
          ? buildRelationsList(relations, entityId)
          : `<div style="color:var(--text-muted);font-size:0.8125rem;">${t('entities.noRelations')}</div>`;
      }

      if (window.lucide) lucide.createIcons({ nodes: [overlay] });
    } catch (err) {
      const vSpinner = overlay.querySelector('#versions-spinner');
      if (vSpinner) vSpinner.remove();
      const rSpinner = overlay.querySelector('#relations-spinner');
      if (rSpinner) rSpinner.remove();
      showToast(t('entities.loadVersionsFailed') + '：' + err.message, 'error');
    }
  }

  // ---- Version Timeline ----

  function buildVersionTimeline(versions) {
    // Sort versions by physical_time descending (newest first)
    const sorted = [...versions].sort((a, b) => {
      const ta = a.physical_time ? new Date(a.physical_time).getTime() : 0;
      const tb = b.physical_time ? new Date(b.physical_time).getTime() : 0;
      return tb - ta;
    });

    const items = sorted.map((v, i) => {
      const prev = sorted[i + 1];
      const nameChanged = prev && v.name !== prev.name;
      const nameDiffHtml = nameChanged ? `
        <div style="display:flex;gap:0.75rem;align-items:center;margin-top:0.5rem;padding:0.375rem 0.5rem;background:var(--bg-input);border-radius:0.375rem;font-size:0.8125rem;">
          <span style="color:var(--error);text-decoration:line-through;">${escapeHtml(prev.name)}</span>
          <i data-lucide="arrow-right" style="width:14px;height:14px;color:var(--text-muted);flex-shrink:0;"></i>
          <span style="color:var(--success);">${escapeHtml(v.name)}</span>
        </div>
      ` : '';

      return `
        <div style="position:relative;padding-left:1.5rem;padding-bottom:${i < sorted.length - 1 ? '1rem' : '0'};">
          ${i < sorted.length - 1 ? '<div style="position:absolute;left:5px;top:12px;bottom:0;width:1px;background:var(--border-color);"></div>' : ''}
          <div style="position:absolute;left:0;top:4px;width:11px;height:11px;border-radius:50%;background:${i === 0 ? 'var(--primary)' : 'var(--border-color)'};border:2px solid ${i === 0 ? 'var(--primary-hover)' : 'var(--border-hover)'};"></div>
          <div style="cursor:pointer;" class="version-expand-toggle" data-version-idx="${i}">
            <div style="display:flex;align-items:center;gap:0.5rem;">
              <span class="mono" style="font-size:0.75rem;color:var(--text-muted);">${formatDate(v.physical_time)}</span>
              ${i === 0 ? '<span class="badge badge-info" style="font-size:0.6875rem;">' + t('entities.latest') + '</span>' : ''}
            </div>
            <div style="margin-top:0.25rem;font-weight:500;font-size:0.875rem;">${escapeHtml(v.name || '-')}</div>
            <div style="margin-top:0.125rem;color:var(--text-secondary);font-size:0.8125rem;" class="truncate">${escapeHtml(truncate(v.content || '', 100))}</div>
            ${nameDiffHtml}
          </div>
          <div class="version-expanded-content" id="version-expanded-${i}" style="display:none;margin-top:0.5rem;">
            <div style="background:var(--bg-input);border:1px solid var(--border-color);border-radius:0.375rem;padding:0.75rem;font-size:0.8125rem;line-height:1.6;white-space:pre-wrap;word-break:break-word;">
              ${escapeHtml(v.content || '')}
            </div>
          </div>
        </div>
      `;
    }).join('');

    // Attach expand/collapse behavior after render
    setTimeout(() => {
      const container = document.getElementById('versions-container');
      if (!container) return;
      container.querySelectorAll('.version-expand-toggle').forEach(toggle => {
        toggle.addEventListener('click', () => {
          const idx = toggle.getAttribute('data-version-idx');
          const expanded = document.getElementById('version-expanded-' + idx);
          if (expanded) {
            const isHidden = expanded.style.display === 'none';
            expanded.style.display = isHidden ? 'block' : 'none';
          }
        });
      });
    }, 0);

    return items;
  }

  // ---- Relations List ----

  function buildRelationsList(relations, currentEntityId) {
    const items = relations.map(r => {
      const isEntity1 = r.entity1_absolute_id === currentEntityId || r.entity1_entity_id === currentEntityId;
      const otherId = isEntity1
        ? (r.entity2_absolute_id || r.entity2_entity_id)
        : (r.entity1_absolute_id || r.entity1_entity_id);
      const direction = isEntity1 ? t('entities.to') : t('entities.from');

      return `
        <div style="padding:0.5rem 0;border-bottom:1px solid var(--border-color);">
          <div style="display:flex;align-items:flex-start;gap:0.5rem;">
            <i data-lucide="arrow-right" style="width:14px;height:14px;color:var(--text-muted);flex-shrink:0;margin-top:2px;"></i>
            <div style="flex:1;min-width:0;">
              <div style="font-size:0.8125rem;color:var(--text-primary);white-space:pre-wrap;word-break:break-word;">${escapeHtml(r.content || '-')}</div>
              <div style="margin-top:0.25rem;display:flex;align-items:center;gap:0.5rem;">
                <span class="badge badge-primary" style="font-size:0.6875rem;">${escapeHtml(direction)}</span>
                <span class="mono" style="font-size:0.75rem;color:var(--text-muted);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${escapeHtml(otherId || t('entities.unknown'))}</span>
                <span class="mono" style="font-size:0.6875rem;color:var(--text-muted);">${formatDate(r.physical_time)}</span>
              </div>
            </div>
          </div>
        </div>
      `;
    }).join('');

    return `<div>${items}</div>`;
  }

  // ---- Data Loading ----

  async function loadAllEntities() {
    const graphId = state.currentGraphId;
    const res = await state.api.listEntities(graphId);
    allEntities = res.data || [];
    displayedCount = 0;
    isSearchMode = false;
    isSearchAllMode = true;
  }

  async function searchEntities(query) {
    const graphId = state.currentGraphId;
    const res = await state.api.searchEntities(query, graphId);
    allEntities = res.data || [];
    displayedCount = 0;
    isSearchMode = true;
    isSearchAllMode = false;
  }

  function renderCurrentSlice() {
    const slice = allEntities.slice(0, displayedCount + PAGE_SIZE);
    displayedCount = slice.length;

    const tableContainer = document.getElementById('entity-table-container');
    if (tableContainer) {
      tableContainer.innerHTML = buildEntityTable(slice);
      bindTableEvents(tableContainer);
    }

    const countEl = document.getElementById('entity-count');
    if (countEl) {
      countEl.textContent = isSearchMode
        ? t('entities.resultCount', { count: allEntities.length })
        : t('entities.entityCount', { count: allEntities.length });
    }
  }

  // ---- Event Binding ----

  function bindTableEvents(container) {
    container.querySelectorAll('tr[data-entity-id]').forEach(row => {
      row.addEventListener('click', (e) => {
        // Don't trigger if clicking load-more button
        if (e.target.closest('#entity-load-more-btn')) return;
        const entityId = row.getAttribute('data-entity-id');
        const entity = allEntities.find(en => en.entity_id === entityId);
        if (entity) openEntityDetail(entity);
      });
    });

    const loadMoreBtn = container.querySelector('#entity-load-more-btn');
    if (loadMoreBtn) {
      loadMoreBtn.addEventListener('click', () => {
        renderCurrentSlice();
      });
    }
  }

  function bindSearchEvents(container) {
    const searchInput = container.querySelector('#entity-search-input');
    const listAllBtn = container.querySelector('#entity-list-all-btn');

    if (searchInput) {
      searchInput.addEventListener('input', () => {
        const query = searchInput.value.trim();
        clearTimeout(debounceTimer);
        if (!query) {
          debounceTimer = setTimeout(() => {
            resetToListAll();
          }, 300);
          return;
        }
        debounceTimer = setTimeout(async () => {
          try {
            const tableContainer = container.querySelector('#entity-table-container');
            if (tableContainer) {
              tableContainer.innerHTML = `<div style="display:flex;justify-content:center;padding:2rem;">${spinnerHtml()}</div>`;
            }
            await searchEntities(query);
            renderCurrentSlice();
          } catch (err) {
            showToast(t('entities.searchFailed') + '：' + err.message, 'error');
            const tableContainer = container.querySelector('#entity-table-container');
            if (tableContainer) {
              tableContainer.innerHTML = emptyState(t('entities.searchFailed'), 'search-x');
            }
          }
        }, 500);
      });
    }

    if (listAllBtn) {
      listAllBtn.addEventListener('click', () => {
        resetToListAll();
      });
    }
  }

  async function resetToListAll() {
    const tableContainer = document.getElementById('entity-table-container');
    if (tableContainer) {
      tableContainer.innerHTML = `<div style="display:flex;justify-content:center;padding:2rem;">${spinnerHtml()}</div>`;
    }
    try {
      await loadAllEntities();
      renderCurrentSlice();
    } catch (err) {
      showToast(t('entities.loadFailed') + '：' + err.message, 'error');
      if (tableContainer) {
        tableContainer.innerHTML = emptyState(t('entities.loadFailed'), 'alert-triangle');
      }
    }
    const searchInput = document.getElementById('entity-search-input');
    if (searchInput) searchInput.value = '';
  }

  // ---- Page Render ----

  async function render(container) {
    container.innerHTML = `
      <div class="page-enter">
        ${buildSearchBar()}
        <div id="entity-table-container">
          <div style="display:flex;justify-content:center;padding:2rem;">${spinnerHtml()}</div>
        </div>
      </div>
    `;

    if (window.lucide) lucide.createIcons({ nodes: [container] });

    bindSearchEvents(container);

    try {
      await loadAllEntities();
      renderCurrentSlice();
    } catch (err) {
      const tableContainer = container.querySelector('#entity-table-container');
      if (tableContainer) {
        tableContainer.innerHTML = emptyState(t('entities.loadFailed') + '：' + err.message, 'alert-triangle');
      }
      showToast(t('entities.loadFailed') + '：' + err.message, 'error');
    }

    if (window.lucide) lucide.createIcons({ nodes: [container] });
  }

  function destroy() {
    clearTimeout(debounceTimer);
    debounceTimer = null;
    allEntities = [];
    displayedCount = 0;
    isSearchMode = false;
    isSearchAllMode = false;
  }

  registerPage('entities', { render, destroy });
})();

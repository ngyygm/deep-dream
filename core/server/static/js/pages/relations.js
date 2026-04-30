/* ==========================================
   Relations Page - 关系浏览器
   ========================================== */

(function() {
  // ---- Module-level cache (persists across render/destroy) ----
  let entityMap = {};          // absolute_id -> { name, family_id }
  let relationMap = {};        // family_id -> relation object
  let allRelations = [];       // accumulated for "全部关系" tab
  let allOffset = 0;
  let allLoading = false;
  let allHasMore = true;
  let activeTab = 'all';
  let _cachedGraphId = null;

  // ---- UI state only (cleared on destroy) ----
  let searchTimer = null;

  // ---- Helpers ----
  function entityName(absoluteId, fallbackName) {
    if (!absoluteId) return '-';
    if (fallbackName) return fallbackName;
    const e = entityMap[absoluteId];
    return e ? (e.name || e.family_id || absoluteId.slice(0, 8) + '...') : absoluteId.slice(0, 8) + '...';
  }

  function familyId(absoluteId) {
    if (!absoluteId) return '-';
    const e = entityMap[absoluteId];
    return e ? e.family_id : '-';
  }

  async function loadEntityMap() {
    try {
      const graphId = state.currentGraphId;
      const res = await state.api.listEntities(graphId, 5000);
      const entities = res.data?.entities || res.data || [];
      entityMap = {};
      entities.forEach(e => {
        entityMap[e.absolute_id] = { name: e.name, family_id: e.family_id };
      });
      _cachedGraphId = graphId;
    } catch (err) {
      console.error('Failed to load entity map:', err);
      showToast(t('relations.loadEntityMapFailed'), 'warning');
    }
  }

  // ---- Table builder ----
  function buildRelationTable(relations) {
    if (!relations || relations.length === 0) {
      return emptyState(t('relations.noRelations'));
    }
    let rows = relations.map(r => {
      if (r.family_id) relationMap[r.family_id] = r;
      return `<tr data-relation-family-id="${escapeAttr(r.family_id || '')}">
        <td title="${escapeHtml(r.content || '')}">${escapeHtml(truncate(r.content || '-', 60))}</td>
        <td title="${escapeHtml(entityName(r.entity1_absolute_id, r.entity1_name))}">${escapeHtml(truncate(entityName(r.entity1_absolute_id, r.entity1_name), 24))}</td>
        <td title="${escapeHtml(entityName(r.entity2_absolute_id, r.entity2_name))}">${escapeHtml(truncate(entityName(r.entity2_absolute_id, r.entity2_name), 24))}</td>
        <td class="mono" style="white-space:nowrap;">${formatDate(r.event_time)}</td>
        <td class="mono" style="white-space:nowrap;">${formatDateMs(r.processed_time)}</td>
        <td title="${escapeHtml(r.source_document || r.doc_name || '')}">${escapeHtml(truncate(r.source_document || r.doc_name || '-', 20))}</td>
        <td>
          <button class="btn btn-sm btn-primary btn-edit-relation" data-family-id="${escapeAttr(r.family_id || '')}" data-i18n="relations.edit">Edit</button>
          <button class="btn btn-sm btn-danger btn-delete-relation" data-family-id="${escapeAttr(r.family_id || '')}" data-i18n="relations.delete">Delete</button>
        </td>
      </tr>`;
    }).join('');

    return `<div class="table-container">
      <table class="data-table">
        <thead>
          <tr>
            <th>${t('relations.content')}</th>
            <th>${t('relations.entity1')}</th>
            <th>${t('relations.entity2')}</th>
            <th>${t('relations.eventTime')}</th>
            <th>${t('relations.processedTime')}</th>
            <th>${t('relations.source')}</th>
            <th data-i18n="relations.actions">Actions</th>
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    </div>`;
  }

  // ---- Detail modal with version history ----
  async function showRelationDetail(r) {
    const e1Name = entityName(r.entity1_absolute_id, r.entity1_name);
    const e1Id = familyId(r.entity1_absolute_id);
    const e2Name = entityName(r.entity2_absolute_id, r.entity2_name);
    const e2Id = familyId(r.entity2_absolute_id);
    const needsE1Resolve = !r.entity1_name && !entityMap[r.entity1_absolute_id];
    const needsE2Resolve = !r.entity2_name && !entityMap[r.entity2_absolute_id];

    // Build static summary content
    const modalContent = document.createElement('div');
    modalContent.innerHTML = `
      <div style="display:flex;flex-direction:column;gap:1rem;">
        <div>
          <div class="form-label">${t('relations.content')}</div>
          <div id="relation-detail-content" class="md-content" style="font-size:0.875rem;color:var(--text-primary);">${renderMarkdown(r.content || '-')}</div>
        </div>

        <div class="divider"></div>

        <div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;">
          <div class="card" style="padding:0.75rem 1rem;">
            <div class="form-label" style="margin-bottom:0.5rem;">${t('relations.entity1')}</div>
            <div class="entity-link" id="rel-e1-name" data-entity-abs="${escapeHtml(r.entity1_absolute_id || '')}" style="font-size:0.875rem;font-weight:600;" title="${escapeHtml(r.entity1_absolute_id || '')}">${escapeHtml(e1Name)}</div>
            <div class="mono" id="rel-e1-id" style="font-size:0.75rem;color:var(--text-muted);margin-top:0.25rem;">${escapeHtml(e1Id)}</div>
            <div class="mono" style="font-size:0.7rem;color:var(--text-muted);margin-top:0.125rem;">${escapeHtml(r.entity1_absolute_id || '-')}</div>
          </div>
          <div class="card" style="padding:0.75rem 1rem;">
            <div class="form-label" style="margin-bottom:0.5rem;">${t('relations.entity2')}</div>
            <div class="entity-link" id="rel-e2-name" data-entity-abs="${escapeHtml(r.entity2_absolute_id || '')}" style="font-size:0.875rem;font-weight:600;" title="${escapeHtml(r.entity2_absolute_id || '')}">${escapeHtml(e2Name)}</div>
            <div class="mono" id="rel-e2-id" style="font-size:0.75rem;color:var(--text-muted);margin-top:0.25rem;">${escapeHtml(e2Id)}</div>
            <div class="mono" style="font-size:0.7rem;color:var(--text-muted);margin-top:0.125rem;">${escapeHtml(r.entity2_absolute_id || '-')}</div>
          </div>
        </div>

        <div class="divider"></div>

        <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.75rem 1.5rem;">
          <div>
            <div class="form-label">${t('relations.relationId')}</div>
            <div class="mono" style="font-size:0.8125rem;color:var(--text-primary);">${escapeHtml(r.family_id || '-')}</div>
          </div>
          <div>
            <div class="form-label">${t('relations.absoluteId')}</div>
            <div class="mono" id="relation-detail-abs-id" style="font-size:0.8125rem;color:var(--text-primary);">${escapeHtml(r.absolute_id || '-')}</div>
          </div>
          <div>
            <div class="form-label">${t('relations.eventTime')}</div>
            <div class="mono" id="relation-detail-event-time" style="font-size:0.8125rem;color:var(--text-primary);">${formatDate(r.event_time)}</div>
          </div>
          <div>
            <div class="form-label">${t('relations.processedTime')}</div>
            <div class="mono" id="relation-detail-processed-time" style="font-size:0.8125rem;color:var(--text-primary);">${formatDateMs(r.processed_time)}</div>
          </div>
          <div>
            <div class="form-label">${t('relations.sourceLabel')}</div>
            <div style="font-size:0.8125rem;color:var(--text-primary);">${escapeHtml(r.source_document || r.doc_name || '-')}</div>
          </div>
          <div>
            <div class="form-label">${t('relations.episodeId')}</div>
            ${r.episode_id ? `<div class="mono doc-link" data-cache-id="${escapeHtml(r.episode_id)}" style="font-size:0.8125rem;">${escapeHtml(r.episode_id)}</div>` : `<div style="font-size:0.8125rem;color:var(--text-primary);">-</div>`}
          </div>
        </div>

        <div class="divider"></div>

        <div id="relation-versions-section">
          <div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.75rem;">
            <i data-lucide="git-commit" style="width:16px;height:16px;color:var(--text-muted);"></i>
            <span style="font-size:0.875rem;font-weight:600;">${t('relations.versionHistory')}</span>
            <div class="spinner spinner-sm" id="relation-versions-spinner"></div>
          </div>
          <div id="relation-versions-container"></div>
        </div>

        <div class="divider"></div>

        <div id="relation-embedding-section">
          <span style="font-size:0.75rem;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.05em;">${t('relations.embedding')}</span>
          <div id="relation-embedding-values" style="margin-top:0.25rem;font-size:0.75rem;color:var(--text-muted);">-</div>
        </div>
      </div>
    `;

    const { overlay } = showModal({
      title: t('relations.detail'),
      content: modalContent.innerHTML,
      size: 'lg',
    });

    if (window.lucide) lucide.createIcons({ nodes: [overlay] });

    // Bind doc link clicks
    overlay.querySelectorAll('.doc-link').forEach(el => {
      el.addEventListener('click', () => {
        const cacheId = el.getAttribute('data-cache-id');
        if (cacheId) window.showEpisodeDoc(cacheId);
      });
    });

    // Bind entity link clicks
    overlay.querySelectorAll('.entity-link').forEach(el => {
      el.addEventListener('click', () => {
        const absId = el.getAttribute('data-entity-abs');
        if (!absId) return;
        state.api.entityByAbsoluteId(absId, state.currentGraphId)
          .then(res => { if (res.data) window.showEntityDetail(res.data); })
          .catch(err => showToast(err.message, 'error'));
      });
    });

    // Resolve missing entity names from backend
    if (needsE1Resolve || needsE2Resolve) {
      const promises = [];
      if (needsE1Resolve && r.entity1_absolute_id) {
        promises.push(
          state.api.entityByAbsoluteId(r.entity1_absolute_id, state.currentGraphId)
            .then(res => {
              if (res.data) {
                const el = overlay.querySelector('#rel-e1-name');
                const idEl = overlay.querySelector('#rel-e1-id');
                if (el) el.textContent = res.data.name || res.data.family_id || el.textContent;
                if (idEl) idEl.textContent = res.data.family_id || idEl.textContent;
                entityMap[r.entity1_absolute_id] = { name: res.data.name, family_id: res.data.family_id };
              }
            }).catch(() => {})
        );
      }
      if (needsE2Resolve && r.entity2_absolute_id) {
        promises.push(
          state.api.entityByAbsoluteId(r.entity2_absolute_id, state.currentGraphId)
            .then(res => {
              if (res.data) {
                const el = overlay.querySelector('#rel-e2-name');
                const idEl = overlay.querySelector('#rel-e2-id');
                if (el) el.textContent = res.data.name || res.data.family_id || el.textContent;
                if (idEl) idEl.textContent = res.data.family_id || idEl.textContent;
                entityMap[r.entity2_absolute_id] = { name: res.data.name, family_id: res.data.family_id };
              }
            }).catch(() => {})
        );
      }
      if (promises.length) await Promise.all(promises);
    }

    // Fetch versions + embedding preview
    const graphId = state.currentGraphId;
    const relFamilyId = r.family_id;

    // Load embedding preview
    state.api.relationEmbeddingPreview(r.absolute_id, 5, graphId)
      .then(embRes => {
        const embEl = overlay.querySelector('#relation-embedding-values');
        if (embEl && embRes && embRes.data && embRes.data.values) {
          const vals = embRes.data.values;
          embEl.innerHTML = vals.map(v => `<span class="mono" style="display:inline-block;padding:2px 6px;margin:2px;border-radius:4px;background:var(--bg-secondary);font-size:0.7rem;">${v.toFixed(4)}</span>`).join('');
        } else {
          const embEl = overlay.querySelector('#relation-embedding-values');
          if (embEl) embEl.textContent = t('relations.noEmbedding');
        }
      })
      .catch(() => {
        const embEl = overlay.querySelector('#relation-embedding-values');
        if (embEl) embEl.textContent = t('relations.noEmbedding');
      });

    state.api.relationVersions(relFamilyId, graphId)
      .then(res => {
        const spinner = overlay.querySelector('#relation-versions-spinner');
        if (spinner) spinner.remove();

        const versions = res.data || [];
        const container = overlay.querySelector('#relation-versions-container');
        if (!container) return;

        if (versions.length <= 1) {
          container.innerHTML = `<div style="color:var(--text-muted);font-size:0.8125rem;">${t('relations.noVersionHistory')}</div>`;
          return;
        }

        container.innerHTML = buildRelationVersionTimeline(versions, overlay);

        if (window.lucide) lucide.createIcons({ nodes: [overlay] });
      })
      .catch(err => {
        const spinner = overlay.querySelector('#relation-versions-spinner');
        if (spinner) spinner.remove();
        const container = overlay.querySelector('#relation-versions-container');
        if (container) container.innerHTML = `<div style="color:var(--error);font-size:0.8125rem;">${t('relations.loadVersionsFailed')}</div>`;
      });
  }

  // ---- Relation Version Timeline ----

  function buildRelationVersionTimeline(versions, overlay) {
    var currentAbsId = overlay.querySelector('#relation-detail-abs-id')?.textContent || '';

    return renderVersionTimeline({
      versions: versions,
      overlay: overlay,
      containerId: 'relation-versions-container',
      toggleClass: 'relation-version-toggle',
      expandedIdPrefix: 'relation-version-expanded',
      isActiveCheck: function(v) { return v.absolute_id === currentAbsId; },
      renderHeader: function(v, i, sorted, isActive) {
        return '<div style="display:flex;align-items:center;gap:0.5rem;">'
          + '<span class="mono" style="font-size:0.75rem;color:var(--text-muted);">' + t('relations.eventTime') + ' ' + formatDate(v.event_time) + '</span>'
          + '<span class="mono" style="font-size:0.7rem;color:var(--text-muted);">' + t('relations.processedTime') + ' ' + formatDateMs(v.processed_time) + '</span>'
          + (i === 0 ? '<span class="badge badge-info" style="font-size:0.6875rem;">' + t('relations.latest') + '</span>' : '')
          + (isActive && i !== 0 ? '<span class="badge badge-primary" style="font-size:0.6875rem;">' + t('relations.current') + '</span>' : '')
          + '</div>'
          + '<div style="margin-top:0.125rem;color:var(--text-secondary);font-size:0.8125rem;" class="truncate">' + escapeHtml(truncate(v.content || '', 100)) + '</div>';
      },
      renderDiff: function(v, prev) {
        if (!prev) return '';
        var contentChanged = v.content !== prev.content;
        var e1Changed = v.entity1_absolute_id !== prev.entity1_absolute_id;
        var e2Changed = v.entity2_absolute_id !== prev.entity2_absolute_id;
        if (!contentChanged && !e1Changed && !e2Changed) return '';
        return '<div style="margin-top:0.5rem;padding:0.375rem 0.5rem;background:var(--bg-input);border-radius:0.375rem;font-size:0.8125rem;">'
          + (contentChanged ? '<div style="margin-bottom:0.25rem;"><span style="color:var(--text-muted);font-size:0.75rem;">' + t('relations.contentChanged') + '</span></div>' : '')
          + ((e1Changed || e2Changed) ? '<div><span style="color:var(--text-muted);font-size:0.75rem;">' + t('relations.entityChanged') + '</span></div>' : '')
          + '</div>';
      },
      renderBody: function(v) {
        return '<div class="md-content" style="background:var(--bg-input);border:1px solid var(--border-color);border-radius:0.375rem;padding:0.75rem;">'
          + renderMarkdown(v.content || '')
          + '</div>';
      },
    });
  }

  function bindTableClicks(rootEl) {
    rootEl.querySelectorAll('tr[data-relation-family-id]').forEach(tr => {
      tr.addEventListener('click', (e) => {
        if (e.target.closest('button')) return;
        const familyId = tr.getAttribute('data-relation-family-id');
        const r = relationMap[familyId];
        if (r) showRelationDetail(r);
      });
    });
    // Edit / Delete button delegation
    rootEl.querySelectorAll('.btn-edit-relation').forEach(btn => {
      btn.addEventListener('click', (e) => {
        e.stopPropagation();
        const familyId = btn.getAttribute('data-family-id');
        const r = relationMap[familyId];
        if (r && window.openEditRelationModal) {
          window.openEditRelationModal(familyId, r.content || '');
        }
      });
    });
    rootEl.querySelectorAll('.btn-delete-relation').forEach(btn => {
      btn.addEventListener('click', (e) => {
        e.stopPropagation();
        const familyId = btn.getAttribute('data-family-id');
        if (window.confirmDeleteRelation) {
          window.confirmDeleteRelation(familyId);
        }
      });
    });
  }

  // ---- Tab 1: All Relations ----
  async function loadAllRelations() {
    if (allLoading || !allHasMore) return;
    allLoading = true;
    const container = document.getElementById('relations-all-body');
    const loadBtn = document.getElementById('relations-load-more');
    if (loadBtn) loadBtn.disabled = true;

    try {
      const res = await state.api.listRelations(state.currentGraphId, 50, allOffset);
      const relations = res.data?.relations || res.data || [];
      if (relations.length === 0) {
        allHasMore = false;
        if (allOffset === 0) {
          container.innerHTML = emptyState(t('relations.noRelations'));
        } else {
          if (loadBtn) loadBtn.style.display = 'none';
        }
      } else {
        allRelations = allRelations.concat(relations);
        allOffset += relations.length;
        container.innerHTML = buildRelationTable(allRelations);
        bindTableClicks(container);
        if (loadBtn) {
          loadBtn.style.display = '';
          loadBtn.textContent = t('relations.loaded', { count: allRelations.length });
        }
        if (relations.length < 50) {
          allHasMore = false;
          if (loadBtn) {
            loadBtn.disabled = true;
            loadBtn.textContent = t('relations.allLoaded', { count: allRelations.length });
          }
        }
      }
    } catch (err) {
      console.error('Failed to load relations:', err);
      showToast(t('relations.loadFailed') + '：' + err.message, 'error');
      if (allOffset === 0) {
        container.innerHTML = emptyState(t('relations.loadFailed'));
      }
    } finally {
      allLoading = false;
      if (loadBtn) loadBtn.disabled = false;
    }
  }

  // Refresh: invalidate cache and reload from scratch
  async function refreshRelations() {
    allRelations = [];
    allOffset = 0;
    allHasMore = true;
    allLoading = false;
    entityMap = {};
    const container = document.getElementById('relations-all-body');
    if (container) {
      container.innerHTML = `<div class="flex items-center justify-center p-8"><div class="spinner"></div></div>`;
    }
    await loadEntityMap();
    await loadAllRelations();
  }

  // ---- Tab 2: Search Relations ----

  async function doSearch(query) {
    const resultsEl = document.getElementById('relations-search-results');
    if (!query || query.trim().length === 0) {
      resultsEl.innerHTML = emptyState(t('relations.searchInput'));
      return;
    }

    query = query.trim();
    resultsEl.innerHTML = `<div class="flex items-center justify-center p-8"><div class="spinner"></div></div>`;

    // Relation ID direct lookup
    if (query.startsWith('rel_') || query.startsWith('relation_')) {
      try {
        let relations;
        if (query.startsWith('relation_')) {
          // absolute_id lookup
          const res = await state.api.relationByAbsoluteId(query, state.currentGraphId);
          relations = res.data ? [res.data] : [];
        } else {
          // family_id → get all versions
          const res = await state.api.relationVersions(query, state.currentGraphId);
          relations = res.data || [];
        }
        if (relations.length === 0) {
          resultsEl.innerHTML = emptyState(t('relations.searchNoMatch'));
        } else {
          _renderSearchResults(resultsEl, relations);
        }
        return;
      } catch (err) {
        // ID lookup failed, fall through to semantic search
      }
    }

    // Semantic search
    try {
      const res = await state.api.searchRelations(query, state.currentGraphId, {
        threshold: 0.3,
        maxResults: 50,
      });
      const relations = res.data?.relations || res.data || [];
      if (relations.length === 0) {
        resultsEl.innerHTML = emptyState(t('relations.searchNoMatch'));
      } else {
        _renderSearchResults(resultsEl, relations);
      }
    } catch (err) {
      console.error('Search failed:', err);
      showToast(t('relations.searchFailed') + '：' + err.message, 'error');
      resultsEl.innerHTML = emptyState(t('relations.searchFailed'));
    }
  }

  function _renderSearchResults(resultsEl, relations) {
    resultsEl.innerHTML = `<div style="margin-bottom:0.5rem;font-size:0.8125rem;color:var(--text-muted);">
      <span class="badge badge-info">${relations.length}</span> ${t('relations.resultCount', { count: relations.length })}
    </div>` + buildRelationTable(relations);
    bindTableClicks(resultsEl);
  }

  function initSearch() {
    const input = document.getElementById('relations-search-input');
    if (!input) return;

    const debouncedSearch = debounce(() => {
      doSearch(input.value);
    }, 500);

    input.addEventListener('input', debouncedSearch);

    const searchBtn = document.getElementById('relations-search-btn');
    if (searchBtn) {
      searchBtn.addEventListener('click', () => {
        if (searchTimer) clearTimeout(searchTimer);
        doSearch(input.value);
      });
    }

    input.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        if (searchTimer) clearTimeout(searchTimer);
        doSearch(input.value);
      }
    });
  }

  // ---- Tab 3: Between Entities ----
  async function loadBetweenEntities() {
    const resultsEl = document.getElementById('relations-between-results');
    const inputA = document.getElementById('relations-entity-a');
    const inputB = document.getElementById('relations-entity-b');

    const entityA = inputA ? inputA.value.trim() : '';
    const entityB = inputB ? inputB.value.trim() : '';

    if (!entityA || !entityB) {
      showToast(t('relations.noTwoEntities'), 'warning');
      return;
    }

    resultsEl.innerHTML = `<div class="flex items-center justify-center p-8"><div class="spinner"></div></div>`;

    try {
      const res = await state.api.relationsBetween(entityA, entityB, state.currentGraphId);
      const relations = res.data || [];
      if (relations.length === 0) {
        const nameA = entityName(entityA);
        const nameB = entityName(entityB);
        resultsEl.innerHTML = emptyState(t('relations.noRelationsBetween', { a: nameA, b: nameB }));
      } else {
        resultsEl.innerHTML = `<div style="margin-bottom:0.5rem;font-size:0.8125rem;color:var(--text-muted);">
          <span class="badge badge-info">${relations.length}</span> ${t('relations.betweenResultCount', { count: relations.length })}
        </div>` + buildRelationTable(relations);
        bindTableClicks(resultsEl);
      }
    } catch (err) {
      console.error('Failed to find relations between entities:', err);
      showToast(t('relations.queryFailed') + '：' + err.message, 'error');
      resultsEl.innerHTML = emptyState(t('relations.queryFailed'));
    }
  }

  function initBetween() {
    const findBtn = document.getElementById('relations-find-between-btn');
    if (findBtn) {
      findBtn.addEventListener('click', loadBetweenEntities);
    }

    const inputB = document.getElementById('relations-entity-b');
    if (inputB) {
      inputB.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') loadBetweenEntities();
      });
    }
  }

  function buildEntityDatalist() {
    const datalistA = document.getElementById('entity-datalist');
    if (!datalistA) return;

    const options = Object.values(entityMap).map(e =>
      `<option value="${escapeHtml(e.family_id)}">${escapeHtml(e.name)}</option>`
    ).join('');

    document.querySelectorAll('datalist#entity-datalist').forEach(dl => {
      dl.innerHTML = options;
    });
  }

  // ---- Tab switching ----
  function switchTab(tabName) {
    activeTab = tabName;
    document.querySelectorAll('#relations-tabs .tab').forEach(tabEl => {
      tabEl.classList.toggle('active', tabEl.getAttribute('data-tab') === tabName);
    });
    document.querySelectorAll('[id^="relations-tab-"]').forEach(panel => {
      panel.style.display = panel.id === `relations-tab-${tabName}` ? '' : 'none';
    });

    // Lazy-load search / between tab content on first visit
    if (tabName === 'search') {
      const input = document.getElementById('relations-search-input');
      if (input && input.value.trim()) doSearch(input.value);
    }

    // Lazy-init PathFinder when path tab is activated
    if (tabName === 'paths') {
      const pfContainer = document.getElementById('path-finder-container');
      if (pfContainer && pfContainer.children.length === 0) {
        PathFinder.init(pfContainer, {
          api: state.api,
          graphId: state.currentGraphId,
          t: t,
          onShowRelationDetail: showRelationDetail,
        });
      }
    }
  }

  // ---- Main render ----
  async function render(container, params) {
    // Determine initial tab from hash params
    const initialTab = params[0] || 'all';

    // Load entity map (cached — if already populated AND graph hasn't changed, skip)
    if (Object.keys(entityMap).length === 0 || _cachedGraphId !== state.currentGraphId) {
      // Graph changed — invalidate all caches
      if (_cachedGraphId !== state.currentGraphId) {
        entityMap = {};
        relationMap = {};
        allRelations = [];
        allOffset = 0;
        allHasMore = true;
        allLoading = false;
      }
      await loadEntityMap();
    }

    container.innerHTML = `
      <div class="page-enter">
        <div class="card">
          <div class="card-header">
            <h2 class="card-title">${t('relations.title')}</h2>
            <button class="btn btn-ghost btn-sm" id="relations-refresh-btn" title="${t('common.refresh')}">
              <i data-lucide="refresh-cw" style="width:14px;height:14px;"></i>
            </button>
          </div>

          <!-- Tabs -->
          <div id="relations-tabs" class="tabs">
            <div class="tab active" data-tab="all">
              <i data-lucide="list" style="width:14px;height:14px;vertical-align:-2px;margin-right:4px;"></i>${t('relations.allRelations')}
            </div>
            <div class="tab" data-tab="search">
              <i data-lucide="search" style="width:14px;height:14px;vertical-align:-2px;margin-right:4px;"></i>${t('relations.searchRelations')}
            </div>
            <div class="tab" data-tab="between">
              <i data-lucide="git-branch" style="width:14px;height:14px;vertical-align:-2px;margin-right:4px;"></i>${t('relations.betweenEntities')}
            </div>
            <div class="tab" data-tab="paths">
              <i data-lucide="route" style="width:14px;height:14px;vertical-align:-2px;margin-right:4px;"></i>${t('relations.pathQuery')}
            </div>
          </div>

          <!-- Tab 1: All Relations -->
          <div id="relations-tab-all">
            <div id="relations-all-body">
              ${allRelations.length > 0
                ? buildRelationTable(allRelations)
                : '<div class="flex items-center justify-center p-8"><div class="spinner"></div></div>'}
            </div>
            <div style="margin-top:1rem;text-align:center;">
              <button id="relations-load-more" class="btn btn-secondary btn-sm" ${!allHasMore && allRelations.length > 0 ? 'disabled' : ''}>${allHasMore ? t('relations.loadMore') : t('relations.allLoaded', { count: allRelations.length })}</button>
            </div>
          </div>

          <!-- Tab 2: Search Relations -->
          <div id="relations-tab-search" style="display:none;">
            <div style="display:flex;gap:0.5rem;margin-bottom:1rem;">
              <input id="relations-search-input" class="input" type="text" placeholder="${t('relations.searchPlaceholder')}" style="flex:1;">
              <button id="relations-search-btn" class="btn btn-primary btn-sm">
                <i data-lucide="search" style="width:14px;height:14px;"></i>${t('relations.searchBtn')}
              </button>
            </div>
            <div id="relations-search-results">
              ${emptyState(t('relations.searchInput'))}
            </div>
          </div>

          <!-- Tab 3: Between Entities -->
          <div id="relations-tab-between" style="display:none;">
            <div style="display:flex;gap:0.75rem;align-items:flex-end;margin-bottom:1rem;flex-wrap:wrap;">
              <div style="flex:1;min-width:200px;">
                <label class="form-label">${t('relations.entityA')}</label>
                <input id="relations-entity-a" class="input" type="text" list="entity-datalist" placeholder="${t('relations.entityPlaceholder')}">
              </div>
              <div style="flex:1;min-width:200px;">
                <label class="form-label">${t('relations.entityB')}</label>
                <input id="relations-entity-b" class="input" type="text" list="entity-datalist" placeholder="${t('relations.entityPlaceholder')}">
              </div>
              <button id="relations-find-between-btn" class="btn btn-primary btn-sm" style="height:36px;">
                <i data-lucide="arrow-right-left" style="width:14px;height:14px;"></i>${t('relations.findRelations')}
              </button>
            </div>
            <datalist id="entity-datalist"></datalist>
            <div id="relations-between-results">
              ${emptyState(t('relations.selectTwo'))}
            </div>
          </div>

          <!-- Tab 4: Path Query -->
          <div id="relations-tab-paths" style="display:none;">
            <div id="path-finder-container" style="min-height:400px;"></div>
          </div>
        </div>
      </div>
    `;

    // Re-render lucide icons for the newly injected HTML
    if (window.lucide) lucide.createIcons({ nodes: [container] });

    // Build entity datalist for the "Between" and "Paths" tabs
    buildEntityDatalist();

    // Bind tab clicks
    container.querySelectorAll('#relations-tabs .tab').forEach(tab => {
      tab.addEventListener('click', () => switchTab(tab.getAttribute('data-tab')));
    });

    // Bind refresh button
    const refreshBtn = document.getElementById('relations-refresh-btn');
    if (refreshBtn) {
      refreshBtn.addEventListener('click', refreshRelations);
    }

    // Load initial tab
    switchTab(initialTab);

    // Initialize tab-specific logic
    initSearch();
    initBetween();

    // Load "All Relations" tab data only if cache is empty
    if (allRelations.length === 0) {
      await loadAllRelations();
      // Bind table clicks for the freshly loaded data
      const allBody = document.getElementById('relations-all-body');
      if (allBody) bindTableClicks(allBody);
    }

    // Bind load-more button
    const loadBtn = document.getElementById('relations-load-more');
    if (loadBtn) {
      loadBtn.addEventListener('click', async () => {
        await loadAllRelations();
        const allBody = document.getElementById('relations-all-body');
        if (allBody) bindTableClicks(allBody);
      });
    }
  }

  function destroy() {
    // Only clear UI state, preserve cache (allRelations, allOffset, allHasMore, entityMap)
    if (searchTimer) {
      clearTimeout(searchTimer);
      searchTimer = null;
    }
    activeTab = 'all';
    if (typeof PathFinder !== 'undefined') PathFinder.destroy();
  }

  // ---- Edit relation modal ----
  function openEditRelationModal(familyId, currentContent) {
    const t = (key) => window.I18N ? window.I18N.t(key) : key;
    const bodyContent = `
      <div class="form-group">
        <label class="form-label" data-i18n="relations.content">Content</label>
        <textarea id="editRelationContent" class="input" rows="4">${escapeAttr(currentContent)}</textarea>
      </div>`;
    const footerContent = `
      <button class="btn btn-secondary" id="editRelationCancel" data-i18n="common.cancel">Cancel</button>
      <button class="btn btn-primary" id="editRelationSave" data-i18n="common.save">Save</button>`;

    const { close } = showModal({
      title: t('relations.editTitle'),
      content: bodyContent,
      footer: footerContent,
    });

    document.getElementById('editRelationCancel').addEventListener('click', close);
    document.getElementById('editRelationSave').addEventListener('click', () => {
      window._submitEditRelationWithClose(familyId, close);
    });
  }

  async function _submitEditRelationWithClose(familyId, closeFn) {
    const t = (key) => window.I18N ? window.I18N.t(key) : key;
    const saveBtn = document.getElementById('editRelationSave');
    if (saveBtn) { saveBtn.disabled = true; }
    const content = document.getElementById('editRelationContent').value.trim();
    if (!content) { showToast(t('relations.contentRequired'), 'error'); if (saveBtn) saveBtn.disabled = false; return; }
    try {
      const res = await state.api.updateRelation(familyId, { content }, state.currentGraphId);
      if (res.error) { showToast(res.error, 'error'); return; }
      showToast(t('relations.updateSuccess'), 'success');
      closeFn();
      refreshRelations();
    } catch (e) { showToast(t('relations.updateFailed') + ': ' + e.message, 'error'); }
    finally { if (saveBtn) { saveBtn.disabled = false; } }
  }

  // ---- Delete relation ----
  async function confirmDeleteRelation(familyId) {
    const t = (key) => window.I18N ? window.I18N.t(key) : key;
    const ok = await showConfirm({ message: t('relations.deleteConfirm'), destructive: true });
    if (!ok) return;
    state.api.deleteRelation(familyId, state.currentGraphId).then(res => {
      if (res.error) { showToast(res.error, 'error'); return; }
      showToast(t('relations.deleteSuccess'), 'success');
      refreshRelations();
    }).catch(e => showToast(t('relations.deleteFailed') + ': ' + e.message, 'error'));
  }

  // Expose globally for use by other pages (search, path-finder) and inline onclick handlers
  window.showRelationDetail = showRelationDetail;
  window.openEditRelationModal = openEditRelationModal;
  window.confirmDeleteRelation = confirmDeleteRelation;

  registerPage('relations', { render, destroy });
})();

/* ==========================================
   Relations Page - 关系浏览器
   ========================================== */

(function() {
  // ---- State ----
  let entityMap = {};          // absolute_id -> { name, entity_id }
  let allRelations = [];       // accumulated for "全部关系" tab
  let allOffset = 0;
  let allLoading = false;
  let allHasMore = true;
  let activeTab = 'all';

  // ---- Helpers ----
  function entityName(absoluteId) {
    if (!absoluteId) return '-';
    const e = entityMap[absoluteId];
    return e ? e.name : absoluteId.slice(0, 8) + '...';
  }

  function entityId(absoluteId) {
    if (!absoluteId) return '-';
    const e = entityMap[absoluteId];
    return e ? e.entity_id : '-';
  }

  async function loadEntityMap() {
    try {
      const res = await state.api.listEntities(state.currentGraphId, 5000);
      const entities = res.data || [];
      entityMap = {};
      entities.forEach(e => {
        entityMap[e.absolute_id] = { name: e.name, entity_id: e.entity_id };
      });
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
      return `<tr data-relation='${escapeHtml(JSON.stringify(r))}'>
        <td title="${escapeHtml(r.content || '')}">${escapeHtml(truncate(r.content || '-', 60))}</td>
        <td title="${escapeHtml(entityName(r.entity1_absolute_id))}">${escapeHtml(truncate(entityName(r.entity1_absolute_id), 24))}</td>
        <td title="${escapeHtml(entityName(r.entity2_absolute_id))}">${escapeHtml(truncate(entityName(r.entity2_absolute_id), 24))}</td>
        <td class="mono" style="white-space:nowrap;">${formatDate(r.physical_time)}</td>
        <td title="${escapeHtml(r.source_document || r.doc_name || '')}">${escapeHtml(truncate(r.source_document || r.doc_name || '-', 20))}</td>
      </tr>`;
    }).join('');

    return `<div class="table-container">
      <table class="data-table">
        <thead>
          <tr>
            <th>${t('relations.content')}</th>
            <th>${t('relations.entity1')}</th>
            <th>${t('relations.entity2')}</th>
            <th>${t('relations.time')}</th>
            <th>${t('relations.source')}</th>
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    </div>`;
  }

  // ---- Detail modal ----
  function showRelationDetail(r) {
    const e1Name = entityName(r.entity1_absolute_id);
    const e1Id = entityId(r.entity1_absolute_id);
    const e2Name = entityName(r.entity2_absolute_id);
    const e2Id = entityId(r.entity2_absolute_id);

    const content = `
      <div style="display:flex;flex-direction:column;gap:1rem;">
        <div>
          <div class="form-label">${t('relations.content')}</div>
          <div style="font-size:0.875rem;color:var(--text-primary);line-height:1.6;">${escapeHtml(r.content || '-')}</div>
        </div>

        <div class="divider"></div>

        <div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;">
          <div class="card" style="padding:0.75rem 1rem;">
            <div class="form-label" style="margin-bottom:0.5rem;">${t('relations.entity1')}</div>
            <div style="font-size:0.875rem;font-weight:600;color:var(--text-primary);">${escapeHtml(e1Name)}</div>
            <div class="mono" style="font-size:0.75rem;color:var(--text-muted);margin-top:0.25rem;">${escapeHtml(e1Id)}</div>
            <div class="mono" style="font-size:0.7rem;color:var(--text-muted);margin-top:0.125rem;">${escapeHtml(r.entity1_absolute_id || '-')}</div>
          </div>
          <div class="card" style="padding:0.75rem 1rem;">
            <div class="form-label" style="margin-bottom:0.5rem;">${t('relations.entity2')}</div>
            <div style="font-size:0.875rem;font-weight:600;color:var(--text-primary);">${escapeHtml(e2Name)}</div>
            <div class="mono" style="font-size:0.75rem;color:var(--text-muted);margin-top:0.25rem;">${escapeHtml(e2Id)}</div>
            <div class="mono" style="font-size:0.7rem;color:var(--text-muted);margin-top:0.125rem;">${escapeHtml(r.entity2_absolute_id || '-')}</div>
          </div>
        </div>

        <div class="divider"></div>

        <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.75rem 1.5rem;">
          <div>
            <div class="form-label">${t('relations.relationId')}</div>
            <div class="mono" style="font-size:0.8125rem;color:var(--text-primary);">${escapeHtml(r.relation_id || '-')}</div>
          </div>
          <div>
            <div class="form-label">${t('relations.absoluteId')}</div>
            <div class="mono" style="font-size:0.8125rem;color:var(--text-primary);">${escapeHtml(r.absolute_id || '-')}</div>
          </div>
          <div>
            <div class="form-label">${t('relations.physicalTime')}</div>
            <div class="mono" style="font-size:0.8125rem;color:var(--text-primary);">${formatDate(r.physical_time)}</div>
          </div>
          <div>
            <div class="form-label">${t('relations.sourceLabel')}</div>
            <div style="font-size:0.8125rem;color:var(--text-primary);">${escapeHtml(r.source_document || r.doc_name || '-')}</div>
          </div>
          <div>
            <div class="form-label">${t('relations.memoryCacheId')}</div>
            <div class="mono" style="font-size:0.8125rem;color:var(--text-primary);">${escapeHtml(r.memory_cache_id || '-')}</div>
          </div>
        </div>
      </div>
    `;

    showModal({
      title: t('relations.detail'),
      content,
      size: 'lg',
    });
  }

  function bindTableClicks(rootEl) {
    rootEl.querySelectorAll('tr[data-relation]').forEach(tr => {
      tr.addEventListener('click', () => {
        try {
          const r = JSON.parse(tr.getAttribute('data-relation'));
          showRelationDetail(r);
        } catch (e) {
          console.error('Failed to parse relation data', e);
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
      const relations = res.data || [];
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

  // ---- Tab 2: Search Relations ----
  let searchTimer = null;

  async function doSearch(query) {
    const resultsEl = document.getElementById('relations-search-results');
    if (!query || query.trim().length === 0) {
      resultsEl.innerHTML = emptyState(t('relations.searchInput'));
      return;
    }

    resultsEl.innerHTML = `<div class="flex items-center justify-center p-8"><div class="spinner"></div></div>`;

    try {
      const res = await state.api.searchRelations(query.trim(), state.currentGraphId, {
        threshold: 0.3,
        maxResults: 50,
      });
      const relations = res.data || [];
      if (relations.length === 0) {
        resultsEl.innerHTML = emptyState(t('relations.searchNoMatch'));
      } else {
        resultsEl.innerHTML = `<div style="margin-bottom:0.5rem;font-size:0.8125rem;color:var(--text-muted);">
          <span class="badge badge-info">${relations.length}</span> ${t('relations.resultCount', { count: relations.length })}
        </div>` + buildRelationTable(relations);
        bindTableClicks(resultsEl);
      }
    } catch (err) {
      console.error('Search failed:', err);
      showToast(t('relations.searchFailed') + '：' + err.message, 'error');
      resultsEl.innerHTML = emptyState(t('relations.searchFailed'));
    }
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
    const datalistB = document.getElementById('entity-datalist');
    if (!datalistA) return;

    const options = Object.values(entityMap).map(e =>
      `<option value="${escapeHtml(e.entity_id)}">${escapeHtml(e.name)}</option>`
    ).join('');

    // Both selectors share the same datalist
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
  }

  // ---- Main render ----
  async function render(container, params) {
    // Determine initial tab from hash params
    const initialTab = params[0] || 'all';

    // Load entity map first
    await loadEntityMap();

    container.innerHTML = `
      <div class="page-enter">
        <div class="card">
          <div class="card-header">
            <h2 class="card-title">${t('relations.title')}</h2>
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
          </div>

          <!-- Tab 1: All Relations -->
          <div id="relations-tab-all">
            <div id="relations-all-body">
              <div class="flex items-center justify-center p-8"><div class="spinner"></div></div>
            </div>
            <div style="margin-top:1rem;text-align:center;">
              <button id="relations-load-more" class="btn btn-secondary btn-sm">${t('relations.loadMore')}</button>
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
        </div>
      </div>
    `;

    // Re-render lucide icons for the newly injected HTML
    if (window.lucide) lucide.createIcons({ nodes: [container] });

    // Build entity datalist for the "Between" tab
    buildEntityDatalist();

    // Bind tab clicks
    container.querySelectorAll('#relations-tabs .tab').forEach(tab => {
      tab.addEventListener('click', () => switchTab(tab.getAttribute('data-tab')));
    });

    // Load initial tab
    switchTab(initialTab);

    // Initialize tab-specific logic
    initSearch();
    initBetween();

    // Load "All Relations" tab data
    allRelations = [];
    allOffset = 0;
    allHasMore = true;
    allLoading = false;
    await loadAllRelations();

    // Bind load-more button
    const loadBtn = document.getElementById('relations-load-more');
    if (loadBtn) {
      loadBtn.addEventListener('click', loadAllRelations);
    }
  }

  function destroy() {
    allRelations = [];
    allOffset = 0;
    allHasMore = true;
    allLoading = false;
    activeTab = 'all';
    entityMap = {};
    if (searchTimer) {
      clearTimeout(searchTimer);
      searchTimer = null;
    }
  }

  registerPage('relations', { render, destroy });
})();

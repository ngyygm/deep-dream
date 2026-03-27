/* ==========================================
   Search Page - Semantic Search
   ========================================== */

(function() {
  const HISTORY_KEY = 'tmg_search_history';
  const MAX_HISTORY = 10;

  // ---- Local state ----
  let multiQueryMode = false;
  let multiQueries = [''];
  let batchResults = {};   // { queryIndex: { entities, relations } }
  let activeBatchTab = 0;
  let currentResults = null;
  let activeTab = 'entities';
  let searchNetwork = null;
  let searchEntityMap = {};
  let searchRelationMap = {};
  let hopLevel = 0;
  let lastSearchQuery = '';

  // ---- History helpers ----
  function loadHistory() {
    try {
      return JSON.parse(localStorage.getItem(HISTORY_KEY)) || [];
    } catch { return []; }
  }

  function saveHistory(query) {
    if (!query || !query.trim()) return;
    let history = loadHistory();
    history = history.filter(h => h !== query.trim());
    history.unshift(query.trim());
    if (history.length > MAX_HISTORY) history = history.slice(0, MAX_HISTORY);
    localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
  }

  function removeHistory(query) {
    let history = loadHistory();
    history = history.filter(h => h !== query);
    localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
  }

  // ---- Read filter values from DOM ----
  function getFilters() {
    const threshold = parseFloat(document.getElementById('search-threshold')?.value || '0.5');
    const maxEntities = parseInt(document.getElementById('search-max-entities')?.value || '20', 10);
    const maxRelations = parseInt(document.getElementById('search-max-relations')?.value || '50', 10);
    const timeAfter = document.getElementById('search-time-after')?.value || '';
    const timeBefore = document.getElementById('search-time-before')?.value || '';
    const expandEl = document.getElementById('search-expand');
    const expand = expandEl ? expandEl.checked : true;
    return { threshold, maxEntities, maxRelations, timeAfter, timeBefore, expand };
  }

  // ---- Build entity name lookup from results ----
  function buildEntityLookup(entities) {
    const map = {};
    if (!Array.isArray(entities)) return map;
    entities.forEach(e => {
      map[e.absolute_id] = e.name || e.entity_id || '';
    });
    return map;
  }

  // ---- Render search bar section ----
  function renderSearchBar() {
    const history = loadHistory();
    const historyChips = history.length > 0
      ? `<div class="search-history" id="search-history" style="margin-top:10px;">
           <div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap;">
             <span style="font-size:0.75rem;color:var(--text-muted);white-space:nowrap;">${t('search.recent')}</span>
             ${history.map(q => `
               <span class="badge badge-primary" style="cursor:pointer;display:inline-flex;align-items:center;gap:4px;padding:2px 10px;font-size:0.75rem;"
                     data-history-query="${escapeHtml(q)}">
                 ${escapeHtml(truncate(q, 30))}
                 <i data-lucide="x" style="width:10px;height:10px;opacity:0.6;" data-history-remove="${escapeHtml(q)}"></i>
               </span>
             `).join('')}
           </div>
         </div>`
      : '';

    return `
      <div class="card">
        <div class="card-header">
          <h2 class="card-title" style="margin:0;">
            <i data-lucide="search" style="width:18px;height:18px;margin-right:6px;"></i>
            ${t('search.title')}
          </h2>
        </div>
        <div style="padding:20px;">
          <div style="display:flex;gap:10px;align-items:stretch;">
            <div style="flex:1;position:relative;">
              <i data-lucide="search" style="position:absolute;left:12px;top:50%;transform:translateY(-50%);width:18px;height:18px;color:var(--text-muted);pointer-events:none;"></i>
              <input type="text" id="search-input" class="input" placeholder="${t('search.placeholder')}"
                     style="padding-left:38px;font-size:0.95rem;height:42px;" value="">
            </div>
            <button class="btn btn-primary" id="search-btn" style="height:42px;white-space:nowrap;">
              <i data-lucide="search" style="width:16px;height:16px;margin-right:6px;"></i>
              ${t('search.searchBtn')}
            </button>
          </div>

          <div style="margin-top:12px;">
            <button class="btn btn-ghost btn-sm" id="toggle-advanced-btn" style="color:var(--text-muted);">
              <i data-lucide="sliders-horizontal" style="width:14px;height:14px;margin-right:4px;"></i>
              ${t('search.advancedFilters')}
              <i data-lucide="chevron-down" style="width:14px;height:14px;margin-left:4px;" id="advanced-chevron"></i>
            </button>
          </div>

          <div id="advanced-filters" style="display:none;margin-top:14px;padding:16px;background:var(--bg-secondary);border-radius:8px;border:1px solid var(--border-color);">
            <div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:16px;">
              <div>
                <label class="form-label">${t('search.threshold')}</label>
                <div style="display:flex;align-items:center;gap:8px;">
                  <input type="range" id="search-threshold" min="0" max="1" step="0.05" value="0.5"
                         style="flex:1;" oninput="document.getElementById('threshold-value').textContent=this.value">
                  <span id="threshold-value" class="mono" style="font-size:0.85rem;min-width:32px;text-align:right;">0.5</span>
                </div>
              </div>
              <div>
                <label class="form-label">${t('search.maxEntities')}</label>
                <input type="number" id="search-max-entities" class="input" value="20" min="1" max="500">
              </div>
              <div>
                <label class="form-label">${t('search.maxRelations')}</label>
                <input type="number" id="search-max-relations" class="input" value="50" min="1" max="500">
              </div>
              <div>
                <label class="form-label">${t('search.timeAfter')}</label>
                <input type="datetime-local" id="search-time-after" class="input">
              </div>
              <div>
                <label class="form-label">${t('search.timeBefore')}</label>
                <input type="datetime-local" id="search-time-before" class="input">
              </div>
              <div>
                <label class="form-label">${t('search.expandNeighbors')}</label>
                <label class="toggle" style="margin-top:4px;">
                  <input type="checkbox" id="search-expand" checked>
                  <span class="toggle-slider"></span>
                  <span style="margin-left:8px;font-size:0.85rem;">${t('search.enabled')}</span>
                </label>
              </div>
            </div>
          </div>

          ${historyChips}
        </div>
      </div>
    `;
  }

  // ---- Render multi-query mode section ----
  function renderMultiQuerySection() {
    return `
      <div class="card" style="margin-top:16px;" id="multi-query-card">
        <div style="padding:16px 20px;display:flex;align-items:center;justify-content:space-between;">
          <div style="display:flex;align-items:center;gap:10px;">
            <label class="toggle">
              <input type="checkbox" id="multi-query-toggle" ${multiQueryMode ? 'checked' : ''}>
              <span class="toggle-slider"></span>
            </label>
            <span style="font-size:0.9rem;font-weight:500;">${t('search.multiQuery')}</span>
          </div>
          ${multiQueryMode ? `
            <button class="btn btn-primary btn-sm" id="batch-search-btn">
              <i data-lucide="play" style="width:14px;height:14px;margin-right:4px;"></i>
              ${t('search.batchSearch')}
            </button>
          ` : ''}
        </div>
        ${multiQueryMode ? `
          <div style="padding:0 20px 16px;">
            <div class="divider"></div>
            <div id="multi-query-list" style="margin-top:12px;display:flex;flex-direction:column;gap:8px;">
              ${multiQueries.map((q, i) => `
                <div class="multi-query-row" data-index="${i}" style="display:flex;gap:8px;align-items:center;">
                  <span class="mono" style="font-size:0.75rem;color:var(--text-muted);min-width:20px;">${i + 1}.</span>
                  <input type="text" class="input multi-query-input" value="${escapeHtml(q)}"
                         placeholder="${t('search.query')} ${i + 1}" style="flex:1;">
                  ${multiQueries.length > 1 ? `
                    <button class="btn btn-ghost btn-sm remove-query-btn" data-index="${i}" title="移除查询">
                      <i data-lucide="x" style="width:14px;height:14px;"></i>
                    </button>
                  ` : ''}
                </div>
              `).join('')}
            </div>
            <button class="btn btn-secondary btn-sm" id="add-query-btn" style="margin-top:10px;">
              <i data-lucide="plus" style="width:14px;height:14px;margin-right:4px;"></i>
              ${t('search.addQuery')}
            </button>
          </div>
        ` : ''}
      </div>
    `;
  }

  // ---- Render results section ----
  function renderResultsSection() {
    if (!currentResults) {
      return `
        <div class="card" style="margin-top:16px;">
          <div style="padding:40px;">
            ${emptyState(t('search.noResults'), 'search')}
          </div>
        </div>
      `;
    }

    const { entities, relations } = currentResults;
    const entityCount = Array.isArray(entities) ? entities.length : 0;
    const relationCount = Array.isArray(relations) ? relations.length : 0;

    if (entityCount === 0 && relationCount === 0) {
      return `
        <div class="card" style="margin-top:16px;">
          <div style="padding:40px;">
            ${emptyState(t('search.noMatch'), 'search-x')}
          </div>
        </div>
      `;
    }

    return `
      <div class="card" style="margin-top:16px;">
        <div style="padding:14px 20px;display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:10px;">
          <div style="display:flex;align-items:center;gap:12px;">
            <span style="font-size:0.9rem;font-weight:600;">${t('search.results')}</span>
            <span class="badge badge-success">${t('search.entityCount', { count: entityCount })}</span>
            <span class="badge badge-info">${t('search.relationCount', { count: relationCount })}</span>
          </div>
        </div>

        <div class="tabs" style="padding:0 20px;">
          <button class="tab ${activeTab === 'entities' ? 'active' : ''}" data-result-tab="entities">
            <i data-lucide="box" style="width:14px;height:14px;margin-right:4px;"></i>
            ${t('common.entities')}
            <span class="badge badge-secondary" style="margin-left:4px;">${entityCount}</span>
          </button>
          <button class="tab ${activeTab === 'relations' ? 'active' : ''}" data-result-tab="relations">
            <i data-lucide="git-branch" style="width:14px;height:14px;margin-right:4px;"></i>
            ${t('common.relations')}
            <span class="badge badge-secondary" style="margin-left:4px;">${relationCount}</span>
          </button>
          <button class="tab ${activeTab === 'visualize' ? 'active' : ''}" data-result-tab="visualize">
            <i data-lucide="network" style="width:14px;height:14px;margin-right:4px;"></i>
            ${t('search.visualize')}
          </button>
        </div>

        <div style="padding:0 20px 20px;" id="results-tab-content">
          ${activeTab === 'entities' ? renderEntitiesTable(entities)
            : activeTab === 'relations' ? renderRelationsTable(relations, entities)
            : renderVisualizeTab(entities, relations)}
        </div>
      </div>
    `;
  }

  // ---- Render entities table ----
  function renderEntitiesTable(entities) {
    if (!Array.isArray(entities) || entities.length === 0) {
      return emptyState(t('search.noEntities'));
    }
    return `
      <div class="table-container">
        <table class="data-table">
          <thead>
            <tr>
              <th style="width:20%;">${t('common.name')}</th>
              <th style="width:40%;">${t('common.content')}</th>
              <th style="width:20%;">${t('common.time')}</th>
              <th style="width:20%;">${t('common.source')}</th>
            </tr>
          </thead>
          <tbody>
            ${entities.map((e, i) => `
              <tr class="entity-row" data-entity-index="${i}" style="cursor:pointer;">
                <td><strong>${escapeHtml(e.name || '-')}</strong></td>
                <td class="truncate" title="${escapeHtml(e.content || '')}">${escapeHtml(truncate(e.content, 100))}</td>
                <td class="mono" style="font-size:0.8rem;">${formatDate(e.physical_time)}</td>
                <td class="truncate" title="${escapeHtml(e.source_document || '')}">${escapeHtml(truncate(e.source_document, 40))}</td>
              </tr>
            `).join('')}
          </tbody>
        </table>
      </div>
    `;
  }

  // ---- Render relations table ----
  function renderRelationsTable(relations, entities) {
    if (!Array.isArray(relations) || relations.length === 0) {
      return emptyState(t('search.noRelations'));
    }
    const entityLookup = buildEntityLookup(entities);
    return `
      <div class="table-container">
        <table class="data-table">
          <thead>
            <tr>
              <th style="width:15%;">${t('search.entity1')}</th>
              <th style="width:40%;">${t('common.content')}</th>
              <th style="width:15%;">${t('search.entity2')}</th>
              <th style="width:15%;">${t('common.time')}</th>
              <th style="width:15%;">${t('common.source')}</th>
            </tr>
          </thead>
          <tbody>
            ${relations.map((r, i) => `
              <tr class="relation-row" data-relation-index="${i}" style="cursor:pointer;">
                <td>
                  <span class="badge badge-primary" style="font-size:0.75rem;">
                    ${escapeHtml(entityLookup[r.entity1_absolute_id] || truncate(r.entity1_absolute_id || '-', 20))}
                  </span>
                </td>
                <td class="truncate" title="${escapeHtml(r.content || '')}">${escapeHtml(truncate(r.content, 100))}</td>
                <td>
                  <span class="badge badge-info" style="font-size:0.75rem;">
                    ${escapeHtml(entityLookup[r.entity2_absolute_id] || truncate(r.entity2_absolute_id || '-', 20))}
                  </span>
                </td>
                <td class="mono" style="font-size:0.8rem;">${formatDate(r.physical_time)}</td>
                <td class="truncate" title="${escapeHtml(r.source_document || '')}">${escapeHtml(truncate(r.source_document, 40))}</td>
              </tr>
            `).join('')}
          </tbody>
        </table>
      </div>
    `;
  }

  // ---- Render batch results section ----
  function renderBatchResultsSection() {
    const keys = Object.keys(batchResults);
    if (keys.length === 0) {
      return `
        <div class="card" style="margin-top:16px;">
          <div style="padding:40px;">
            ${emptyState(t('search.noBatchResults'), 'layers')}
          </div>
        </div>
      `;
    }

    return `
      <div class="card" style="margin-top:16px;">
        <div style="padding:14px 20px;">
          <span style="font-size:0.9rem;font-weight:600;">${t('search.batchResult')}</span>
        </div>
        <div class="tabs" style="padding:0 20px;">
          ${keys.map(idx => {
            const res = batchResults[idx];
            const ec = Array.isArray(res.entities) ? res.entities.length : 0;
            const rc = Array.isArray(res.relations) ? res.relations.length : 0;
            return `
              <button class="tab ${activeBatchTab === parseInt(idx) ? 'active' : ''}" data-batch-tab="${idx}">
                ${escapeHtml(truncate(res.query || `${t('search.query')} ${parseInt(idx) + 1}`, 20))}
                <span class="badge badge-secondary" style="margin-left:4px;">${ec + rc}</span>
              </button>
            `;
          }).join('')}
        </div>
        <div style="padding:0 20px 20px;" id="batch-results-content">
          ${renderBatchResultContent(keys[activeBatchTab] || keys[0])}
        </div>
      </div>
    `;
  }

  function renderBatchResultContent(idx) {
    const res = batchResults[idx];
    if (!res) return emptyState(t('common.noData'));
    const entities = res.entities || [];
    const relations = res.relations || [];
    const entityCount = entities.length;
    const relationCount = relations.length;

    if (entityCount === 0 && relationCount === 0) {
      return `
        <div style="margin-top:12px;">
          <div style="margin-bottom:10px;font-size:0.85rem;color:var(--text-muted);">
            ${t('search.query')}: <strong>${escapeHtml(res.query || '-')}</strong>
          </div>
          ${emptyState(t('search.queryNoResults'))}
        </div>
      `;
    }

    return `
      <div style="margin-top:12px;">
        <div style="margin-bottom:10px;font-size:0.85rem;color:var(--text-muted);">
          ${t('search.query')}: <strong>${escapeHtml(res.query || '-')}</strong>
          &nbsp;&mdash;&nbsp;
          <span class="badge badge-success">${t('search.entityCount', { count: entityCount })}</span>
          <span class="badge badge-info" style="margin-left:4px;">${t('search.relationCount', { count: relationCount })}</span>
        </div>
        <div class="tabs" style="margin-bottom:12px;">
          <button class="tab ${activeTab === 'entities' ? 'active' : ''}" data-result-tab="entities">${t('common.entities')}</button>
          <button class="tab ${activeTab === 'relations' ? 'active' : ''}" data-result-tab="relations">${t('common.relations')}</button>
        </div>
        ${activeTab === 'entities' ? renderEntitiesTable(entities) : renderRelationsTable(relations, entities)}
      </div>
    `;
  }

  // ---- Render visualize tab with hop slider + graph canvas ----
  function renderVisualizeTab(entities, relations) {
    if (!Array.isArray(entities) || entities.length === 0) {
      return emptyState(t('search.graphEmpty'), 'network');
    }

    return `
      <div style="margin-top:12px;">
        <div style="display:flex;align-items:center;gap:16px;margin-bottom:12px;flex-wrap:wrap;">
          <div style="display:flex;align-items:center;gap:8px;">
            <label class="form-label" style="margin-bottom:0;white-space:nowrap;">${t('search.hopLevel')}:</label>
            <input type="range" id="hop-level-slider" min="0" max="3" step="1" value="${hopLevel}"
                   style="width:120px;" oninput="document.getElementById('hop-level-value').textContent=this.value">
            <span id="hop-level-value" class="mono" style="font-size:0.85rem;min-width:16px;text-align:center;">${hopLevel}</span>
          </div>
          <span style="font-size:0.75rem;color:var(--text-muted);">${t('search.hopDesc')}</span>
          <div style="display:flex;align-items:center;gap:8px;margin-left:auto;">
            <span style="width:10px;height:10px;border-radius:50%;background:#f59e0b;display:inline-block;"></span>
            <span style="font-size:0.75rem;color:var(--text-muted);">${t('search.searchResult')}</span>
            <span style="width:10px;height:10px;border-radius:50%;background:#6366f1;display:inline-block;margin-left:8px;"></span>
            <span style="font-size:0.75rem;color:var(--text-muted);">${t('search.expandedNode')}</span>
          </div>
        </div>
        <div id="search-graph-canvas" style="width:100%;height:500px;border-radius:8px;background:var(--bg-input);border:1px solid var(--border-color);"></div>
      </div>
    `;
  }

  // ---- Build vis-network graph from search results ----
  function renderSearchGraph(entities, relations) {
    if (typeof vis === 'undefined') return;

    const canvas = document.getElementById('search-graph-canvas');
    if (!canvas) return;

    // Destroy previous network
    if (searchNetwork) {
      searchNetwork.destroy();
      searchNetwork = null;
    }

    searchEntityMap = {};
    searchRelationMap = {};

    // Track which entity IDs are direct search results
    const directResultIds = new Set();
    entities.forEach(e => directResultIds.add(e.absolute_id));

    const nodeIds = new Set();

    // Build nodes: search result entities in amber, others in indigo
    const nodes = new vis.DataSet(
      entities.map(e => {
        searchEntityMap[e.absolute_id] = e;
        nodeIds.add(e.absolute_id);
        const isDirect = directResultIds.has(e.absolute_id);
        return {
          id: e.absolute_id,
          label: e.name || e.entity_id || t('graph.unnamedEntity'),
          title: escapeHtml(truncate(e.content || e.name || '', 80)),
          color: isDirect ? {
            background: '#f59e0b',
            border: '#fbbf24',
            highlight: { background: '#fbbf24', border: '#fcd34d' },
            hover: { background: '#fbbf24', border: '#fcd34d' },
          } : {
            background: '#6366f1',
            border: '#818cf8',
            highlight: { background: '#818cf8', border: '#a5b4fc' },
            hover: { background: '#818cf8', border: '#a5b4fc' },
          },
          size: isDirect ? 24 : 16,
          shape: 'dot',
          font: { color: '#e2e8f0', size: 11, face: 'Inter, sans-serif' },
        };
      })
    );

    // Build edges
    const edges = new vis.DataSet(
      relations
        .filter(r => nodeIds.has(r.entity1_absolute_id) && nodeIds.has(r.entity2_absolute_id))
        .map(r => {
          searchRelationMap[r.absolute_id] = r;
          return {
            id: r.absolute_id,
            from: r.entity1_absolute_id,
            to: r.entity2_absolute_id,
            label: truncate(r.content || '', 30),
            color: { color: '#4b5563', highlight: '#9ca3af', hover: '#6b7280' },
            font: { color: '#94a3b8', size: 9, face: 'Inter, sans-serif', strokeWidth: 3, strokeColor: '#141620' },
            arrows: 'to',
            smooth: { enabled: true, type: 'continuous', roundness: 0.2 },
          };
        })
    );

    const options = {
      physics: {
        enabled: true,
        solver: 'forceAtlas2Based',
        forceAtlas2Based: {
          gravitationalConstant: -80,
          centralGravity: 0.008,
          springLength: 120,
          springConstant: 0.04,
          damping: 0.6,
          avoidOverlap: 0.4,
        },
        stabilization: { enabled: true, iterations: 150, updateInterval: 25 },
      },
      interaction: {
        hover: true,
        tooltipDelay: 200,
        zoomView: true,
        dragView: true,
        navigationButtons: false,
        keyboard: false,
      },
      layout: { improvedLayout: true },
    };

    searchNetwork = new vis.Network(canvas, { nodes, edges }, options);

    // Click handler: show detail modal
    searchNetwork.on('click', params => {
      const nodeId = params.nodes[0];
      const edgeId = params.edges[0];
      if (nodeId && searchEntityMap[nodeId]) {
        showEntityDetail(searchEntityMap[nodeId]);
      } else if (edgeId && searchRelationMap[edgeId]) {
        const entityLookup = buildEntityLookup(entities);
        showRelationDetail(searchRelationMap[edgeId], entityLookup);
      }
    });
  }

  // ---- Show entity detail modal ----
  function showEntityDetail(entity) {
    const fields = [
      [t('common.name'), entity.name],
      [t('graph.entityId'), entity.entity_id],
      [t('graph.absoluteId'), entity.absolute_id],
      [t('common.content'), entity.content],
      [t('graph.physicalTime'), formatDate(entity.physical_time)],
      [t('graph.sourceDoc'), entity.source_document],
    ].filter(([, v]) => v);

    const content = `
      <div style="display:flex;flex-direction:column;gap:12px;">
        ${fields.map(([label, value]) => `
          <div>
            <div style="font-size:0.75rem;font-weight:600;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.5px;margin-bottom:2px;">${escapeHtml(label)}</div>
            <div style="font-size:0.9rem;color:var(--text-primary);word-break:break-word;white-space:pre-wrap;">${escapeHtml(value)}</div>
          </div>
        `).join('')}
      </div>
    `;

    showModal({ title: t('search.entityDetail'), content, size: 'lg' });
  }

  // ---- Show relation detail modal ----
  function showRelationDetail(relation, entityLookup) {
    const e1Name = entityLookup[relation.entity1_absolute_id] || relation.entity1_absolute_id || '-';
    const e2Name = entityLookup[relation.entity2_absolute_id] || relation.entity2_absolute_id || '-';

    const fields = [
      [t('common.content'), relation.content],
      [t('graph.relationId'), relation.relation_id],
      [t('graph.absoluteId'), relation.absolute_id],
      [t('search.entity1'), `${e1Name} (${relation.entity1_absolute_id || '-'})`],
      [t('search.entity2'), `${e2Name} (${relation.entity2_absolute_id || '-'})`],
      [t('graph.physicalTime'), formatDate(relation.physical_time)],
      [t('graph.sourceDoc'), relation.source_document],
    ].filter(([, v]) => v);

    const content = `
      <div style="display:flex;flex-direction:column;gap:12px;">
        ${fields.map(([label, value]) => `
          <div>
            <div style="font-size:0.75rem;font-weight:600;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.5px;margin-bottom:2px;">${escapeHtml(label)}</div>
            <div style="font-size:0.9rem;color:var(--text-primary);word-break:break-word;white-space:pre-wrap;">${escapeHtml(value)}</div>
          </div>
        `).join('')}
      </div>
    `;

    showModal({ title: t('search.relationDetail'), content, size: 'lg' });
  }

  // ---- Execute single search ----
  async function executeSearch() {
    const input = document.getElementById('search-input');
    const query = input ? input.value.trim() : '';
    if (!query) {
      showToast(t('search.noQuery'), 'warning');
      return;
    }

    const filters = getFilters();
    const btn = document.getElementById('search-btn');
    const originalBtnHtml = btn ? btn.innerHTML : '';

    if (btn) {
      btn.disabled = true;
      btn.innerHTML = `${spinnerHtml('spinner-sm')} ${t('search.searching')}`;
    }

    try {
      const res = await state.api.find(query, {
        graphId: state.currentGraphId,
        threshold: filters.threshold,
        maxEntities: filters.maxEntities,
        maxRelations: filters.maxRelations,
        timeAfter: filters.timeAfter || undefined,
        timeBefore: filters.timeBefore || undefined,
        expand: filters.expand,
      });

      currentResults = res.data || { entities: [], relations: [] };
      activeTab = 'entities';
      batchResults = {};
      lastSearchQuery = query;

      saveHistory(query);
      refreshResults();
      showToast(t('search.found', { e: currentResults.entities.length, r: currentResults.relations.length }), 'success');
    } catch (err) {
      showToast(`${t('search.searchFailed')}: ${err.message}`, 'error');
    } finally {
      if (btn) {
        btn.disabled = false;
        btn.innerHTML = originalBtnHtml;
        if (window.lucide) lucide.createIcons({ nodes: [btn] });
      }
    }
  }

  // ---- Execute batch search ----
  async function executeBatchSearch() {
    const inputs = document.querySelectorAll('.multi-query-input');
    const queries = Array.from(inputs).map(inp => inp.value.trim()).filter(Boolean);

    if (queries.length === 0) {
      showToast(t('search.noQueryInput'), 'warning');
      return;
    }

    const filters = getFilters();
    const btn = document.getElementById('batch-search-btn');
    const originalBtnHtml = btn ? btn.innerHTML : '';

    if (btn) {
      btn.disabled = true;
      btn.innerHTML = `${spinnerHtml('spinner-sm')} ${t('search.executing')}`;
    }

    batchResults = {};
    activeBatchTab = 0;
    activeTab = 'entities';

    try {
      const promises = queries.map(async (q, i) => {
        try {
          const res = await state.api.find(q, {
            graphId: state.currentGraphId,
            threshold: filters.threshold,
            maxEntities: filters.maxEntities,
            maxRelations: filters.maxRelations,
            timeAfter: filters.timeAfter || undefined,
            timeBefore: filters.timeBefore || undefined,
            expand: filters.expand,
          });
          batchResults[i] = { query: q, entities: res.data?.entities || [], relations: res.data?.relations || [] };
        } catch (err) {
          batchResults[i] = { query: q, entities: [], relations: [], error: err.message };
        }
      });

      await Promise.all(promises);

      queries.forEach(q => saveHistory(q));
      currentResults = null;
      refreshResults();
      showToast(t('search.batchComplete', { count: queries.length }), 'success');
    } catch (err) {
      showToast(`${t('search.batchFailed')}: ${err.message}`, 'error');
    } finally {
      if (btn) {
        btn.disabled = false;
        btn.innerHTML = originalBtnHtml;
        if (window.lucide) lucide.createIcons({ nodes: [btn] });
      }
    }
  }

  // ---- Refresh the results area ----
  function refreshResults() {
    const resultsContainer = document.getElementById('results-area');
    if (!resultsContainer) return;

    if (multiQueryMode && Object.keys(batchResults).length > 0) {
      resultsContainer.innerHTML = renderBatchResultsSection();
    } else {
      resultsContainer.innerHTML = renderResultsSection();
    }

    if (window.lucide) lucide.createIcons({ nodes: [resultsContainer] });
    bindResultEvents(resultsContainer);

    // If visualize tab is active, render the graph
    if (activeTab === 'visualize' && currentResults && !multiQueryMode) {
      const { entities, relations } = currentResults;
      renderSearchGraph(entities || [], relations || []);
      bindHopSlider(resultsContainer);
    }
  }

  // ---- Bind hop level slider to re-execute search ----
  function bindHopSlider(container) {
    const slider = container.querySelector('#hop-level-slider');
    if (!slider || !lastSearchQuery) return;

    slider.addEventListener('change', () => {
      hopLevel = parseInt(slider.value, 10);
      const valueLabel = container.querySelector('#hop-level-value');
      if (valueLabel) valueLabel.textContent = hopLevel;
      // Re-execute search with updated expand and relation limits
      executeSearchWithHop();
    });
  }

  // ---- Execute search with hop level ----
  async function executeSearchWithHop() {
    if (!lastSearchQuery) return;

    const filters = getFilters();
    const canvas = document.getElementById('search-graph-canvas');
    if (canvas) canvas.innerHTML = `<div class="flex items-center justify-center h-full">${spinnerHtml()}<span style="margin-left:8px;color:var(--text-muted);">${t('search.graphLoading')}</span></div>`;

    try {
      // Hop 0 = only direct results (no expand), hop 1+ = expand with increasing relation limits
      const expand = hopLevel > 0;
      const maxRelations = hopLevel === 0 ? filters.maxRelations : filters.maxRelations * (hopLevel + 1);

      const res = await state.api.find(lastSearchQuery, {
        graphId: state.currentGraphId,
        threshold: filters.threshold,
        maxEntities: filters.maxEntities,
        maxRelations: maxRelations,
        timeAfter: filters.timeAfter || undefined,
        timeBefore: filters.timeBefore || undefined,
        expand,
      });

      currentResults = res.data || { entities: [], relations: [] };

      if (canvas) {
        const { entities, relations } = currentResults;
        renderSearchGraph(entities || [], relations || []);
      }

      // Update result counts in the header
      const resultsHeader = document.querySelector('#results-area .card .flex');
      if (resultsHeader) {
        const ec = (currentResults.entities || []).length;
        const rc = (currentResults.relations || []).length;
        const badges = resultsHeader.querySelectorAll('.badge');
        if (badges.length >= 2) {
          badges[0].textContent = t('search.entityCount', { count: ec });
          badges[1].textContent = t('search.relationCount', { count: rc });
        }
      }
    } catch (err) {
      showToast(`${t('search.searchFailed')}: ${err.message}`, 'error');
      if (canvas) {
        canvas.innerHTML = `<div class="flex items-center justify-center h-full" style="color:var(--text-muted);">${escapeHtml(err.message)}</div>`;
      }
    }
  }

  // ---- Refresh history chips ----
  function refreshHistory() {
    const historyEl = document.getElementById('search-history');
    if (!historyEl) return;
    const history = loadHistory();
    if (history.length === 0) {
      historyEl.remove();
      return;
    }
    historyEl.innerHTML = `
      <div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap;">
        <span style="font-size:0.75rem;color:var(--text-muted);white-space:nowrap;">${t('search.recent')}</span>
        ${history.map(q => `
          <span class="badge badge-primary" style="cursor:pointer;display:inline-flex;align-items:center;gap:4px;padding:2px 10px;font-size:0.75rem;"
                data-history-query="${escapeHtml(q)}">
            ${escapeHtml(truncate(q, 30))}
            <i data-lucide="x" style="width:10px;height:10px;opacity:0.6;" data-history-remove="${escapeHtml(q)}"></i>
          </span>
        `).join('')}
      </div>
    `;
    if (window.lucide) lucide.createIcons({ nodes: [historyEl] });
  }

  // ---- Bind events on the main page ----
  function bindEvents(container) {
    // Search button
    const searchBtn = container.querySelector('#search-btn');
    if (searchBtn) searchBtn.addEventListener('click', executeSearch);

    // Enter key on search input
    const searchInput = container.querySelector('#search-input');
    if (searchInput) {
      searchInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') executeSearch();
      });
      // Auto-focus
      setTimeout(() => searchInput.focus(), 100);
    }

    // Advanced filters toggle
    const toggleBtn = container.querySelector('#toggle-advanced-btn');
    const filtersPanel = container.querySelector('#advanced-filters');
    const chevron = container.querySelector('#advanced-chevron');
    if (toggleBtn && filtersPanel) {
      toggleBtn.addEventListener('click', () => {
        const open = filtersPanel.style.display !== 'none';
        filtersPanel.style.display = open ? 'none' : 'block';
        if (chevron) {
          chevron.style.transform = open ? '' : 'rotate(180deg)';
          chevron.style.transition = 'transform 0.2s ease';
        }
      });
    }

    // Expand neighbors toggle label update
    const expandToggle = container.querySelector('#search-expand');
    if (expandToggle) {
      expandToggle.addEventListener('change', () => {
        const label = expandToggle.closest('label').querySelector('span:last-child');
        if (label) label.textContent = expandToggle.checked ? t('search.enabled') : t('search.disabled');
      });
    }

    // History chip clicks
    const searchBar = container.querySelector('#search-history');
    if (searchBar) {
      searchBar.addEventListener('click', (e) => {
        const removeBtn = e.target.closest('[data-history-remove]');
        const queryChip = e.target.closest('[data-history-query]');

        if (removeBtn) {
          e.stopPropagation();
          removeHistory(removeBtn.getAttribute('data-history-remove'));
          refreshHistory();
          return;
        }

        if (queryChip) {
          const q = queryChip.getAttribute('data-history-query');
          const input = container.querySelector('#search-input');
          if (input) input.value = q;
          executeSearch();
        }
      });
    }

    // Multi-query toggle
    const multiToggle = container.querySelector('#multi-query-toggle');
    if (multiToggle) {
      multiToggle.addEventListener('change', () => {
        multiQueryMode = multiToggle.checked;
        if (multiQueryMode && multiQueries.length === 0) {
          multiQueries = [''];
        }
        const multiCard = container.querySelector('#multi-query-card');
        if (multiCard) {
          multiCard.outerHTML = renderMultiQuerySection();
        } else {
          const searchBarCard = container.querySelector('#multi-query-card') ||
            container.querySelector('.card + .card');
          // Re-render the multi-query area after search bar
          const placeholder = document.createElement('div');
          placeholder.id = 'multi-query-area';
          const resultsArea = container.querySelector('#results-area');
          if (resultsArea) {
            container.insertBefore(placeholder, resultsArea);
          }
          placeholder.outerHTML = renderMultiQuerySection();
        }
        if (window.lucide) lucide.createIcons({ nodes: [container] });
        bindMultiQueryEvents(container);
      });
    }

    // Bind multi-query events initially (in case mode is on)
    bindMultiQueryEvents(container);
  }

  // ---- Bind multi-query specific events ----
  function bindMultiQueryEvents(container) {
    // Add query button
    const addBtn = container.querySelector('#add-query-btn');
    if (addBtn) {
      addBtn.addEventListener('click', () => {
        multiQueries.push('');
        const listEl = container.querySelector('#multi-query-list');
        if (listEl) {
          const idx = multiQueries.length - 1;
          const row = document.createElement('div');
          row.className = 'multi-query-row';
          row.dataset.index = idx;
          row.style.cssText = 'display:flex;gap:8px;align-items:center;';
          row.innerHTML = `
            <span class="mono" style="font-size:0.75rem;color:var(--text-muted);min-width:20px;">${idx + 1}.</span>
            <input type="text" class="input multi-query-input" value="" placeholder="${t('search.query')} ${idx + 1}" style="flex:1;">
            <button class="btn btn-ghost btn-sm remove-query-btn" data-index="${idx}" title="移除查询">
              <i data-lucide="x" style="width:14px;height:14px;"></i>
            </button>
          `;
          listEl.appendChild(row);
          if (window.lucide) lucide.createIcons({ nodes: [row] });
          row.querySelector('.multi-query-input').focus();
        }
      });
    }

    // Remove query buttons
    container.querySelectorAll('.remove-query-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const idx = parseInt(btn.dataset.index);
        multiQueries.splice(idx, 1);
        // Re-read current values before re-rendering
        syncMultiQueryInputs(container);
        const multiCard = container.querySelector('#multi-query-card');
        if (multiCard) {
          multiCard.outerHTML = renderMultiQuerySection();
          if (window.lucide) lucide.createIcons({ nodes: [container] });
          bindMultiQueryEvents(container);
        }
      });
    });

    // Batch search button
    const batchBtn = container.querySelector('#batch-search-btn');
    if (batchBtn) {
      batchBtn.addEventListener('click', () => {
        syncMultiQueryInputs(container);
        executeBatchSearch();
      });
    }
  }

  // ---- Sync multi-query input values back to the array ----
  function syncMultiQueryInputs(container) {
    const inputs = container.querySelectorAll('.multi-query-input');
    inputs.forEach((inp, i) => {
      if (multiQueries[i] !== undefined) {
        multiQueries[i] = inp.value;
      }
    });
  }

  // ---- Bind result section events (tabs, row clicks) ----
  function bindResultEvents(container) {
    // Entity/Relation/Visualize tabs in single results
    container.querySelectorAll('[data-result-tab]').forEach(tab => {
      tab.addEventListener('click', () => {
        activeTab = tab.dataset.resultTab;
        // Re-render results tab content
        const tabContent = container.querySelector('#results-tab-content') ||
                           container.querySelector('#batch-results-content');
        if (tabContent) {
          const batchKeys = Object.keys(batchResults);
          if (batchKeys.length > 0) {
            tabContent.innerHTML = renderBatchResultContent(batchKeys[activeBatchTab] || batchKeys[0]);
          } else if (currentResults) {
            const { entities, relations } = currentResults;
            if (activeTab === 'visualize') {
              tabContent.innerHTML = renderVisualizeTab(entities, relations);
            } else {
              tabContent.innerHTML = activeTab === 'entities'
                ? renderEntitiesTable(entities)
                : renderRelationsTable(relations, entities);
            }
          }
          if (window.lucide) lucide.createIcons({ nodes: [tabContent] });
          bindResultEvents(container);

          // If switching to visualize tab, render the graph
          if (activeTab === 'visualize' && currentResults && batchKeys.length === 0) {
            renderSearchGraph(entities || [], relations || []);
            bindHopSlider(container);
          }
        }
        // Update active tab styling
        container.querySelectorAll('[data-result-tab]').forEach(tb => {
          tb.classList.toggle('active', tb.dataset.resultTab === activeTab);
        });
      });
    });

    // Batch tabs
    container.querySelectorAll('[data-batch-tab]').forEach(tab => {
      tab.addEventListener('click', () => {
        activeBatchTab = parseInt(tab.dataset.batchTab);
        const tabContent = container.querySelector('#batch-results-content');
        if (tabContent) {
          tabContent.innerHTML = renderBatchResultContent(activeBatchTab);
          if (window.lucide) lucide.createIcons({ nodes: [tabContent] });
          bindResultEvents(container);
        }
        container.querySelectorAll('[data-batch-tab]').forEach(tb => {
          tb.classList.toggle('active', parseInt(tb.dataset.batchTab) === activeBatchTab);
        });
      });
    });

    // Entity row clicks
    const entities = currentResults?.entities || [];
    container.querySelectorAll('.entity-row').forEach(row => {
      row.addEventListener('click', () => {
        const idx = parseInt(row.dataset.entityIndex);
        if (entities[idx]) {
          // Check if in batch mode
          const batchKeys = Object.keys(batchResults);
          if (batchKeys.length > 0) {
            const batchRes = batchResults[batchKeys[activeBatchTab] || batchKeys[0]];
            if (batchRes && batchRes.entities[idx]) {
              showEntityDetail(batchRes.entities[idx]);
              return;
            }
          }
          showEntityDetail(entities[idx]);
        }
      });
    });

    // Relation row clicks
    const entityLookup = buildEntityLookup(entities);
    const relations = currentResults?.relations || [];
    container.querySelectorAll('.relation-row').forEach(row => {
      row.addEventListener('click', () => {
        const idx = parseInt(row.dataset.relationIndex);
        if (relations[idx]) {
          const batchKeys = Object.keys(batchResults);
          if (batchKeys.length > 0) {
            const batchRes = batchResults[batchKeys[activeBatchTab] || batchKeys[0]];
            if (batchRes && batchRes.relations[idx]) {
              const batchLookup = buildEntityLookup(batchRes.entities || []);
              showRelationDetail(batchRes.relations[idx], batchLookup);
              return;
            }
          }
          showRelationDetail(relations[idx], entityLookup);
        }
      });
    });
  }

  // ---- Main render ----
  async function render(container, params) {
    currentResults = null;
    batchResults = {};
    activeTab = 'entities';
    activeBatchTab = 0;
    multiQueryMode = false;
    multiQueries = [''];
    lastSearchQuery = '';
    hopLevel = 0;
    if (searchNetwork) {
      searchNetwork.destroy();
      searchNetwork = null;
    }
    searchEntityMap = {};
    searchRelationMap = {};

    container.innerHTML = `
      <div class="page-enter">
        <div style="max-width:1200px;margin:0 auto;">
          <div id="search-bar-area"></div>
          <div id="multi-query-area"></div>
          <div id="results-area"></div>
        </div>
      </div>
    `;

    // Render search bar
    const searchBarArea = container.querySelector('#search-bar-area');
    searchBarArea.innerHTML = renderSearchBar();

    // Render multi-query section
    const multiQueryArea = container.querySelector('#multi-query-area');
    multiQueryArea.innerHTML = renderMultiQuerySection();

    // Render results (empty state)
    const resultsArea = container.querySelector('#results-area');
    resultsArea.innerHTML = renderResultsSection();

    // Initialize icons
    if (window.lucide) lucide.createIcons({ nodes: [container] });

    // Bind all events
    bindEvents(container);
    bindResultEvents(container);
  }

  // ---- Cleanup ----
  function destroy() {
    currentResults = null;
    batchResults = {};
    lastSearchQuery = '';
    hopLevel = 0;
    if (searchNetwork) {
      searchNetwork.destroy();
      searchNetwork = null;
    }
    searchEntityMap = {};
    searchRelationMap = {};
  }

  registerPage('search', { render, destroy });
})();

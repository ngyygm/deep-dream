/* ==========================================
   Graph Explorer Page
   ========================================== */

(function () {
  let network = null;
  let entityMap = {};   // absolute_id -> entity data
  let relationMap = {}; // absolute_id -> relation data
  let isFirstRender = true;

  // ---- Build the page layout and kick off initial load ----

  async function render(container, params) {
    container.innerHTML = `
      <div class="page-enter">
        <!-- Top control bar -->
        <div class="card mb-4">
          <div class="flex flex-wrap items-end gap-4">
            <div class="flex-shrink-0" style="min-width:140px;">
              <label class="form-label">${t('graph.graphId')}</label>
              <div class="badge badge-primary mono text-sm" id="graph-id-badge">-</div>
            </div>

            <div class="flex-1" style="min-width:160px;max-width:260px;">
              <label class="form-label">${t('graph.maxEntities')}: <span id="entity-limit-val" class="mono">100</span></label>
              <input type="range" id="entity-limit" min="10" max="500" value="100" step="10">
            </div>

            <div class="flex-1" style="min-width:160px;max-width:260px;">
              <label class="form-label">${t('graph.maxRelations')}: <span id="relation-limit-val" class="mono">500</span></label>
              <input type="range" id="relation-limit" min="10" max="2000" value="500" step="10">
            </div>

            <button class="btn btn-primary" id="load-graph-btn">
              <i data-lucide="refresh-cw" style="width:16px;height:16px;"></i>
              ${t('graph.loadGraph')}
            </button>

            <span id="graph-stats" class="text-sm" style="color:var(--text-muted);"></span>
          </div>
        </div>

        <!-- Main area: canvas + detail sidebar -->
        <div class="flex gap-4" style="height:calc(100vh - 240px);min-height:400px;">
          <!-- Graph canvas -->
          <div class="flex-1 relative" style="min-width:0;">
            <div id="graph-canvas" style="width:100%;height:100%;"></div>
            <div id="graph-loading" class="absolute inset-0 flex items-center justify-center" style="background:var(--bg-input);border-radius:0.5rem;display:none;">
              ${spinnerHtml()}
            </div>
          </div>

          <!-- Detail sidebar -->
          <div id="detail-sidebar" style="width:30%;min-width:280px;max-width:420px;">
            <div class="card h-full">
              <div class="card-header">
                <span class="card-title">${t('common.detail')}</span>
              </div>
              <div id="detail-content">
                ${emptyState(t('common.clickToView'), 'mouse-pointer-click')}
              </div>
            </div>
          </div>
        </div>
      </div>
    `;

    // Re-render lucide icons for this page
    if (window.lucide) lucide.createIcons();

    // Wire up controls
    const entitySlider = document.getElementById('entity-limit');
    const entityVal = document.getElementById('entity-limit-val');
    const relationSlider = document.getElementById('relation-limit');
    const relationVal = document.getElementById('relation-limit-val');
    const loadBtn = document.getElementById('load-graph-btn');
    const graphBadge = document.getElementById('graph-id-badge');

    graphBadge.textContent = state.currentGraphId;

    entitySlider.addEventListener('input', () => {
      entityVal.textContent = entitySlider.value;
    });

    relationSlider.addEventListener('input', () => {
      relationVal.textContent = relationSlider.value;
    });

    loadBtn.addEventListener('click', () => loadGraph());

    // Auto-load on first render
    if (isFirstRender) {
      isFirstRender = false;
      await loadGraph();
    }
  }

  // ---- Fetch entities & relations, then build the vis-network graph ----

  async function loadGraph() {
    const graphId = state.currentGraphId;
    const entityLimit = parseInt(document.getElementById('entity-limit').value, 10);
    const relationLimit = parseInt(document.getElementById('relation-limit').value, 10);
    const loadingEl = document.getElementById('graph-loading');
    const statsEl = document.getElementById('graph-stats');

    if (loadingEl) loadingEl.style.display = 'flex';
    if (statsEl) statsEl.textContent = t('common.loading');

    try {
      const [entityRes, relationRes] = await Promise.all([
        state.api.listEntities(graphId, entityLimit),
        state.api.listRelations(graphId, relationLimit),
      ]);

      const entities = entityRes.data || [];
      const relations = relationRes.data || [];

      buildGraph(entities, relations);

      if (statsEl) {
        statsEl.textContent = t('graph.loaded', { entities: entities.length, relations: relations.length });
      }

      showToast(t('graph.loaded', { entities: entities.length, relations: relations.length }), 'success');
    } catch (err) {
      console.error('Failed to load graph:', err);
      showToast(t('graph.loadFailed') + ': ' + err.message, 'error');
      if (statsEl) statsEl.textContent = t('common.error');
    } finally {
      if (loadingEl) loadingEl.style.display = 'none';
    }
  }

  // ---- Build vis-network DataSet and initialize the network ----

  function buildGraph(entities, relations) {
    // Reset maps
    entityMap = {};
    relationMap = {};

    const nodeIds = new Set();

    // Detect current theme for canvas colors (vis-network uses canvas, not DOM)
    const isLight = document.documentElement.getAttribute('data-theme') === 'light';
    const nodeFontColor = isLight ? '#1e293b' : '#e2e8f0';

    // Build nodes from entities
    const nodes = new vis.DataSet(
      entities.map((e) => {
        entityMap[e.absolute_id] = e;
        nodeIds.add(e.absolute_id);
        return {
          id: e.absolute_id,
          label: e.name || e.entity_id || t('graph.unnamedEntity'),
          title: escapeHtml(truncate(e.content || e.name || '', 80)),
          color: {
            background: '#6366f1',
            border: '#818cf8',
            highlight: {
              background: '#818cf8',
              border: '#a5b4fc',
            },
            hover: {
              background: '#818cf8',
              border: '#a5b4fc',
            },
          },
          size: 20,
          shape: 'dot',
          font: {
            color: nodeFontColor,
            size: 11,
            face: 'Inter, sans-serif',
          },
        };
      })
    );

    // Build edges from relations; only include edges where both endpoints are known nodes
    const edges = new vis.DataSet(
      relations
        .filter((r) => nodeIds.has(r.entity1_absolute_id) && nodeIds.has(r.entity2_absolute_id))
        .map((r) => {
          relationMap[r.absolute_id] = r;
          return {
            id: r.absolute_id,
            from: r.entity1_absolute_id,
            to: r.entity2_absolute_id,
            color: {
              color: '#4b5563',
              highlight: '#9ca3af',
              hover: '#6b7280',
            },
            smooth: {
              enabled: true,
              type: 'continuous',
              roundness: 0.2,
            },
          };
        })
    );

    // Container
    const container = document.getElementById('graph-canvas');
    if (!container) return;

    // Destroy previous network
    if (network) {
      network.destroy();
      network = null;
    }

    // vis-network options
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
        stabilization: {
          enabled: true,
          iterations: 150,
          updateInterval: 25,
        },
      },
      interaction: {
        hover: true,
        tooltipDelay: 200,
        zoomView: true,
        dragView: true,
        navigationButtons: false,
        keyboard: false,
      },
      layout: {
        improvedLayout: true,
      },
    };

    const data = { nodes, edges };
    network = new vis.Network(container, data, options);

    // Click handler: open detail sidebar
    network.on('click', (params) => {
      const nodeId = params.nodes[0];
      const edgeId = params.edges[0];

      if (nodeId) {
        showEntityDetail(nodeId);
      } else if (edgeId) {
        showRelationDetail(edgeId);
      }
    });
  }

  // ---- Show entity detail in the sidebar ----

  function showEntityDetail(absoluteId) {
    const entity = entityMap[absoluteId];
    if (!entity) return;

    const detailContent = document.getElementById('detail-content');
    if (!detailContent) return;

    detailContent.innerHTML = `
      <div class="flex items-center gap-2 mb-3">
        <span class="badge badge-primary">${t('graph.entityDetail')}</span>
      </div>

      <h3 style="font-size:1.1rem;font-weight:600;color:var(--text-primary);margin-bottom:0.75rem;word-break:break-word;">
        ${escapeHtml(entity.name || t('graph.unnamedEntity'))}
      </h3>

      <div class="divider"></div>

      <div style="display:flex;flex-direction:column;gap:0.75rem;">
        <div>
          <span class="form-label" style="margin-bottom:0.125rem;">${t('graph.content')}</span>
          <p style="font-size:0.8125rem;color:var(--text-secondary);line-height:1.5;word-break:break-word;white-space:pre-wrap;">
            ${escapeHtml(entity.content || '-')}
          </p>
        </div>

        <div>
          <span class="form-label" style="margin-bottom:0.125rem;">${t('graph.entityId')}</span>
          <p class="mono truncate" style="color:var(--text-muted);font-size:0.75rem;" title="${escapeHtml(entity.entity_id || '')}">
            ${escapeHtml(entity.entity_id || '-')}
          </p>
        </div>

        <div>
          <span class="form-label" style="margin-bottom:0.125rem;">${t('graph.absoluteId')}</span>
          <p class="mono truncate" style="color:var(--text-muted);font-size:0.75rem;" title="${escapeHtml(entity.absolute_id || '')}">
            ${escapeHtml(entity.absolute_id || '-')}
          </p>
        </div>

        <div>
          <span class="form-label" style="margin-bottom:0.125rem;">${t('graph.physicalTime')}</span>
          <p style="font-size:0.8125rem;color:var(--text-secondary);">
            ${formatDate(entity.physical_time)}
          </p>
        </div>

        ${entity.source_document ? `
        <div>
          <span class="form-label" style="margin-bottom:0.125rem;">${t('graph.sourceDoc')}</span>
          <p class="mono truncate" style="color:var(--text-muted);font-size:0.75rem;" title="${escapeHtml(entity.source_document)}">
            ${escapeHtml(truncate(entity.source_document, 60))}
          </p>
        </div>
        ` : ''}

        ${entity.memory_cache_id ? `
        <div>
          <span class="form-label" style="margin-bottom:0.125rem;">${t('graph.memoryCacheId')}</span>
          <p class="mono truncate" style="color:var(--text-muted);font-size:0.75rem;">
            ${escapeHtml(entity.memory_cache_id)}
          </p>
        </div>
        ` : ''}
      </div>

      <div class="divider"></div>

      <div class="flex flex-wrap gap-2">
        <button class="btn btn-secondary btn-sm" id="view-versions-btn">
          <i data-lucide="git-branch" style="width:14px;height:14px;"></i>
          ${t('graph.versionHistory')}
        </button>
        <button class="btn btn-secondary btn-sm" id="view-relations-btn">
          <i data-lucide="link" style="width:14px;height:14px;"></i>
          ${t('graph.viewRelations')}
        </button>
      </div>
    `;

    if (window.lucide) lucide.createIcons({ nodes: [detailContent] });

    // Wire buttons
    document.getElementById('view-versions-btn').addEventListener('click', () => {
      openVersionsModal(entity);
    });

    document.getElementById('view-relations-btn').addEventListener('click', () => {
      openRelationsModal(entity);
    });
  }

  // ---- Show relation detail in the sidebar ----

  function showRelationDetail(absoluteId) {
    const relation = relationMap[absoluteId];
    if (!relation) return;

    const detailContent = document.getElementById('detail-content');
    if (!detailContent) return;

    const fromName = entityMap[relation.entity1_absolute_id]?.name || relation.entity1_absolute_id || '?';
    const toName = entityMap[relation.entity2_absolute_id]?.name || relation.entity2_absolute_id || '?';

    detailContent.innerHTML = `
      <div class="flex items-center gap-2 mb-3">
        <span class="badge" style="background:var(--info-dim);color:var(--info);">${t('graph.relationDetail')}</span>
      </div>

      <h3 style="font-size:1.1rem;font-weight:600;color:var(--text-primary);margin-bottom:0.75rem;word-break:break-word;">
        ${escapeHtml(truncate(relation.content || t('graph.unnamedRelation'), 60))}
      </h3>

      <div class="divider"></div>

      <div style="display:flex;flex-direction:column;gap:0.75rem;">
        <div>
          <span class="form-label" style="margin-bottom:0.125rem;">${t('graph.content')}</span>
          <p style="font-size:0.8125rem;color:var(--text-secondary);line-height:1.5;word-break:break-word;white-space:pre-wrap;">
            ${escapeHtml(relation.content || '-')}
          </p>
        </div>

        <div>
          <span class="form-label" style="margin-bottom:0.125rem;">${t('graph.fromEntity')}</span>
          <p class="mono truncate" style="color:var(--text-muted);font-size:0.75rem;" title="${escapeHtml(fromName)}">
            ${escapeHtml(truncate(fromName, 40))}
          </p>
        </div>

        <div>
          <span class="form-label" style="margin-bottom:0.125rem;">${t('graph.toEntity')}</span>
          <p class="mono truncate" style="color:var(--text-muted);font-size:0.75rem;" title="${escapeHtml(toName)}">
            ${escapeHtml(truncate(toName, 40))}
          </p>
        </div>

        <div>
          <span class="form-label" style="margin-bottom:0.125rem;">${t('graph.relationId')}</span>
          <p class="mono truncate" style="color:var(--text-muted);font-size:0.75rem;">
            ${escapeHtml(relation.relation_id || '-')}
          </p>
        </div>

        <div>
          <span class="form-label" style="margin-bottom:0.125rem;">${t('graph.physicalTime')}</span>
          <p style="font-size:0.8125rem;color:var(--text-secondary);">
            ${formatDate(relation.physical_time)}
          </p>
        </div>

        ${relation.source_document ? `
        <div>
          <span class="form-label" style="margin-bottom:0.125rem;">${t('graph.sourceDoc')}</span>
          <p class="mono truncate" style="color:var(--text-muted);font-size:0.75rem;" title="${escapeHtml(relation.source_document)}">
            ${escapeHtml(truncate(relation.source_document, 60))}
          </p>
        </div>
        ` : ''}
      </div>
    `;

    if (window.lucide) lucide.createIcons({ nodes: [detailContent] });
  }

  // ---- Versions modal: fetch and display entity version history ----

  async function openVersionsModal(entity) {
    const entityId = entity.entity_id || entity.absolute_id;
    const graphId = state.currentGraphId;

    const modal = showModal({
      title: t('graph.versionsTitle', { name: truncate(entity.name || entityId, 40) }),
      content: `<div class="flex justify-center p-6">${spinnerHtml()}</div>`,
      size: 'lg',
    });

    try {
      const res = await state.api.entityVersions(entityId, graphId);
      const versions = res.data || [];

      if (versions.length === 0) {
        modal.overlay.querySelector('.modal-body').innerHTML = emptyState(t('graph.noVersions'));
        return;
      }

      const rows = versions
        .map(
          (v) => `
        <tr>
          <td style="max-width:120px;">${formatDate(v.physical_time)}</td>
          <td style="max-width:200px;" title="${escapeHtml(v.name || '')}">${escapeHtml(truncate(v.name || '-', 30))}</td>
          <td style="max-width:300px;" title="${escapeHtml(v.content || '')}">${escapeHtml(truncate(v.content || '-', 50))}</td>
        </tr>
      `
        )
        .join('');

      modal.overlay.querySelector('.modal-body').innerHTML = `
        <div class="table-container" style="max-height:50vh;overflow-y:auto;">
          <table class="data-table">
            <thead>
              <tr>
                <th>${t('graph.versionTime')}</th>
                <th>${t('graph.versionName')}</th>
                <th>${t('graph.versionContent')}</th>
              </tr>
            </thead>
            <tbody>${rows}</tbody>
          </table>
        </div>
        <p style="margin-top:0.75rem;font-size:0.75rem;color:var(--text-muted);">${t('graph.versionCount', { count: versions.length })}</p>
      `;
    } catch (err) {
      modal.overlay.querySelector('.modal-body').innerHTML = `
        <div class="empty-state">
          <i data-lucide="alert-triangle"></i>
          <p>${t('graph.loadFailedDetail')}: ${escapeHtml(err.message)}</p>
        </div>
      `;
      if (window.lucide) lucide.createIcons({ nodes: [modal.overlay] });
    }
  }

  // ---- Relations modal: fetch and display relations for an entity ----

  async function openRelationsModal(entity) {
    const entityId = entity.entity_id || entity.absolute_id;
    const graphId = state.currentGraphId;

    const modal = showModal({
      title: t('graph.relationsTitle', { name: truncate(entity.name || entityId, 40) }),
      content: `<div class="flex justify-center p-6">${spinnerHtml()}</div>`,
      size: 'lg',
    });

    try {
      const res = await state.api.entityRelations(entityId, graphId);
      const relations = res.data || [];

      if (relations.length === 0) {
        modal.overlay.querySelector('.modal-body').innerHTML = emptyState(t('graph.noRelations'));
        return;
      }

      const rows = relations
        .map(
          (r) => `
        <tr>
          <td style="max-width:250px;" title="${escapeHtml(r.content || '')}">${escapeHtml(truncate(r.content || '-', 40))}</td>
          <td class="mono" style="max-width:120px;font-size:0.75rem;color:var(--text-muted);">${formatDate(r.physical_time)}</td>
        </tr>
      `
        )
        .join('');

      modal.overlay.querySelector('.modal-body').innerHTML = `
        <div class="table-container" style="max-height:50vh;overflow-y:auto;">
          <table class="data-table">
            <thead>
              <tr>
                <th>${t('graph.content')}</th>
                <th>${t('graph.versionTime')}</th>
              </tr>
            </thead>
            <tbody>${rows}</tbody>
          </table>
        </div>
        <p style="margin-top:0.75rem;font-size:0.75rem;color:var(--text-muted);">${t('graph.relationCount', { count: relations.length })}</p>
      `;
    } catch (err) {
      modal.overlay.querySelector('.modal-body').innerHTML = `
        <div class="empty-state">
          <i data-lucide="alert-triangle"></i>
          <p>${t('graph.loadFailedDetail')}: ${escapeHtml(err.message)}</p>
        </div>
      `;
      if (window.lucide) lucide.createIcons({ nodes: [modal.overlay] });
    }
  }

  // ---- Cleanup on page leave ----

  function destroy() {
    if (network) {
      network.destroy();
      network = null;
    }
    entityMap = {};
    relationMap = {};
  }

  // ---- Register this page ----

  registerPage('graph', { render, destroy });
})();

/* ==========================================
   Entities Page - Entity Browser
   ========================================== */

(function() {
  const BATCH_SIZE = 50;
  let debounceTimer = null;

  // ---- Module-level cache (persists across render/destroy) ----
  let allEntities = [];
  let allOffset = 0;
  let allHasMore = true;
  let allLoading = false;
  let totalCount = null;
  let isSearchMode = false;
  let _currentModalClose = null;
  let _searchSeq = 0;
  let _cachedGraphId = null;

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
          <span id="entity-count" class="mono" style="font-size:0.8125rem;color:var(--text-muted);"></span>
          <button class="btn btn-ghost btn-sm" id="entity-refresh-btn" title="${t('common.refresh')}">
            <i data-lucide="refresh-cw" style="width:14px;height:14px;"></i>
          </button>
        </div>
      </div>
    `;
  }

  // ---- Entity Table ----

  function buildEntityTable(entities) {
    if (!entities || entities.length === 0) {
      return emptyState(t('entities.noEntities'), 'box');
    }

    const rows = entities.filter(e => e.family_id).map(e => `
      <tr data-family-id="${escapeHtml(e.family_id)}" data-absolute-id="${escapeHtml(e.absolute_id)}">
        <td><input type="checkbox" class="entity-checkbox" value="${escapeAttr(e.family_id)}"></td>
        <td style="max-width:180px;font-weight:500;">${escapeHtml(e.name || '-')}</td>
        <td style="max-width:300px;" class="truncate" title="${escapeHtml(e.content || '')}">${escapeHtml(truncate(e.content || '', 60))}</td>
        <td style="white-space:nowrap;">${formatDate(e.event_time)}</td>
        <td style="white-space:nowrap;">${formatDateMs(e.processed_time)}</td>
        <td style="max-width:120px;" class="truncate" title="${escapeHtml(e.doc_name || e.source_document || '')}">${escapeHtml(e.doc_name || e.source_document || '-')}</td>
        <td style="text-align:center;">
          <span class="badge badge-info">${escapeHtml(String(e.version_count || '?'))}</span>
        </td>
        <td>
          <button class="btn btn-sm btn-primary btn-edit-entity" data-family-id="${escapeAttr(e.family_id)}" data-i18n="entities.edit">Edit</button>
          <button class="btn btn-sm btn-danger btn-delete-entity" data-family-id="${escapeAttr(e.family_id)}" data-i18n="entities.delete">Delete</button>
        </td>
      </tr>
    `).join('');

    return `
      <div class="card" style="margin-top:0.75rem;">
        <div style="display:flex;align-items:center;gap:0.5rem;padding:0.5rem 0.75rem;border-bottom:1px solid var(--border-color);">
          <button class="btn btn-sm btn-danger" id="batch-delete-entities-btn" data-i18n="entities.batchDelete">Batch Delete</button>
          <button class="btn btn-sm btn-primary" id="merge-entities-btn" data-i18n="entities.merge">Merge</button>
        </div>
        <div class="table-container">
          <table class="data-table">
            <thead>
              <tr>
                <th><input type="checkbox" id="selectAllEntities" onchange="toggleAllEntityCheckboxes(this)"></th>
                <th>${t('entities.name')}</th>
                <th>${t('entities.content')}</th>
                <th>${t('entities.eventTime')}</th>
                <th>${t('graph.processedTime')}</th>
                <th>${t('entities.source')}</th>
                <th style="text-align:center;">${t('entities.version')}</th>
                <th data-i18n="entities.actions">Actions</th>
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
    if (!allHasMore) {
      if (allEntities.length > 0) {
        return `
          <div style="display:flex;justify-content:center;padding-top:0.75rem;">
            <span style="font-size:0.8125rem;color:var(--text-muted);">${t('entities.allLoaded', { count: allEntities.length })}</span>
          </div>`;
      }
      return '';
    }
    return `
      <div style="display:flex;justify-content:center;padding-top:0.75rem;">
        <button class="btn btn-ghost" id="entity-load-more-btn">
          ${t('common.loadMore')}
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
          <div class="mono" style="margin-top:0.125rem;">${escapeHtml(entity.family_id)}</div>
        </div>
        <div>
          <span style="font-size:0.75rem;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.05em;">${t('graph.absoluteId')}</span>
          <div class="mono" style="margin-top:0.125rem;">${escapeHtml(entity.absolute_id)}</div>
        </div>
        <div>
          <span style="font-size:0.75rem;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.05em;">${t('common.name')}</span>
          <div style="margin-top:0.125rem;font-weight:600;display:flex;align-items:center;gap:0.5rem;">
            ${escapeHtml(entity.name || '-')}
            <span class="badge badge-info" style="font-size:0.625rem;">v${escapeHtml(String(entity.version_count || '?'))}</span>
          </div>
        </div>
        <div>
          <span style="font-size:0.75rem;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.05em;">${t('common.content')}</span>
          <div class="md-content" style="margin-top:0.125rem;">${renderMarkdown(entity.content || '-')}</div>
        </div>
        <div style="display:flex;gap:2rem;">
          <div>
            <span style="font-size:0.75rem;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.05em;">${t('graph.eventTime')}</span>
            <div class="mono" style="margin-top:0.125rem;">${formatDate(entity.event_time)}</div>
          </div>
          <div>
            <span style="font-size:0.75rem;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.05em;">${t('graph.processedTime')}</span>
            <div class="mono" style="margin-top:0.125rem;">${formatDateMs(entity.processed_time)}</div>
          </div>
          <div>
            <span style="font-size:0.75rem;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.05em;">${t('graph.sourceDoc')}</span>
            <div style="margin-top:0.125rem;">${escapeHtml(entity.doc_name || entity.source_document || '-')}</div>
          </div>
        </div>
        ${entity.episode_id ? `
        <div>
          <span style="font-size:0.75rem;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.05em;">${t('graph.episodeId')}</span>
          <div class="mono doc-link" data-cache-id="${escapeHtml(entity.episode_id)}" style="margin-top:0.125rem;">${escapeHtml(entity.episode_id)}</div>
        </div>
        ` : ''}
        ${entity.summary ? `
        <div>
          <span style="font-size:0.75rem;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.05em;">${t('entities.summary')}</span>
          <div class="md-content" style="margin-top:0.125rem;font-size:0.85rem;">${renderMarkdown(entity.summary)}</div>
        </div>
        ` : ''}
        ${entity.attributes && Object.keys(entity.attributes).length > 0 ? `
        <div>
          <span style="font-size:0.75rem;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.05em;">${t('entities.attributes')}</span>
          <div style="margin-top:0.25rem;display:flex;flex-wrap:wrap;gap:0.375rem;">
            ${Object.entries(entity.attributes).map(([k, v]) => `
              <span class="badge badge-secondary" style="font-size:0.75rem;">${escapeHtml(k)}: ${escapeHtml(String(v))}</span>
            `).join('')}
          </div>
        </div>
        ` : ''}
        <div id="entity-embedding-section">
          <span style="font-size:0.75rem;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.05em;">${t('entities.embedding')}</span>
          <div id="entity-embedding-values" style="margin-top:0.25rem;font-size:0.75rem;color:var(--text-muted);">-</div>
        </div>
        <div style="display:flex;gap:0.5rem;">
          <button class="btn btn-primary btn-sm" id="evolve-summary-btn">
            <i data-lucide="sparkles" style="width:14px;height:14px;margin-right:4px;"></i>${t('entities.evolveSummary')}
          </button>
          <button class="btn btn-secondary btn-sm" id="view-provenance-btn">
            <i data-lucide="git-commit" style="width:14px;height:14px;margin-right:4px;"></i>${t('entities.provenance')}
          </button>
        </div>
      </div>

      <div class="divider"></div>

      <div id="entity-versions-section">
        <div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.75rem;">
          <i data-lucide="git-branch" style="width:16px;height:16px;color:var(--text-muted);"></i>
          <span style="font-size:0.875rem;font-weight:600;">${t('entities.versionHistory')}</span>
          <div class="spinner spinner-sm" id="versions-spinner"></div>
        </div>
        <div id="versions-container" data-family-id="${familyId}"></div>
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
      ${isNeo4j() ? `
      <div class="divider"></div>
      <div id="entity-neighbors-section">
        <div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.75rem;">
          <i data-lucide="share-2" style="width:16px;height:16px;color:var(--text-muted);"></i>
          <span style="font-size:0.875rem;font-weight:600;">${t('communities.neighborGraph')}</span>
        </div>
        <button class="btn btn-secondary btn-sm" id="load-neighbors-btn">
          <i data-lucide="network" style="width:14px;height:14px;"></i>${t('graph.loadGraph')}
        </button>
        <div id="neighbors-graph" style="height:300px;margin-top:0.5rem;border:1px solid var(--border-color);border-radius:0.5rem;"></div>
      </div>` : ''}

      <div class="divider"></div>
      <div id="entity-contradictions-section">
        <div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.75rem;">
          <i data-lucide="alert-triangle" style="width:16px;height:16px;color:var(--text-muted);"></i>
          <span style="font-size:0.875rem;font-weight:600;">${t('entities.contradictions')}</span>
          <div class="spinner spinner-sm" id="contradictions-spinner"></div>
        </div>
        <div id="contradictions-container"></div>
      </div>
    `;

    const { overlay } = showModal({
      title: entity.name || entity.family_id,
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

    // Evolve summary button
    const evolveBtn = overlay.querySelector('#evolve-summary-btn');
    if (evolveBtn) {
      evolveBtn.addEventListener('click', async () => {
        evolveBtn.disabled = true;
        evolveBtn.innerHTML = `${spinnerHtml('spinner-sm')} ${t('entities.evolveSummaryRunning')}`;
        try {
          const res = await state.api.evolveEntitySummary(entity.family_id, state.currentGraphId);
          showToast(t('entities.evolveSummarySuccess'), 'success');
          // Refresh entity detail
          if (res.data) {
            entity.summary = res.data.summary || entity.summary;
            entity.attributes = res.data.attributes || entity.attributes;
          }
          invalidateAndReload();
        } catch (err) {
          showToast(t('entities.evolveSummaryFailed') + ': ' + err.message, 'error');
        } finally {
          evolveBtn.disabled = false;
          evolveBtn.innerHTML = `<i data-lucide="sparkles" style="width:14px;height:14px;margin-right:4px;"></i>${t('entities.evolveSummary')}`;
          if (window.lucide) lucide.createIcons({ nodes: [evolveBtn] });
        }
      });
    }

    // View provenance button
    const provenanceBtn = overlay.querySelector('#view-provenance-btn');
    if (provenanceBtn) {
      provenanceBtn.addEventListener('click', async () => {
        provenanceBtn.disabled = true;
        try {
          const res = await state.api.entityProvenance(entity.family_id, state.currentGraphId);
          const prov = res.data || {};
          let body = `<div style="display:flex;flex-direction:column;gap:0.75rem;">`;
          if (prov.source_document || prov.source) {
            body += `<div><span style="font-size:0.75rem;color:var(--text-muted);">${t('entities.provenanceSource')}</span><div style="margin-top:0.125rem;">${escapeHtml(prov.source_document || prov.source || '-')}</div></div>`;
          }
          if (prov.extracted_at || prov.created_at) {
            body += `<div><span style="font-size:0.75rem;color:var(--text-muted);">${t('entities.provenanceExtractedAt')}</span><div class="mono" style="margin-top:0.125rem;">${formatDate(prov.extracted_at || prov.created_at)}</div></div>`;
          }
          if (prov.confidence != null) {
            body += `<div><span style="font-size:0.75rem;color:var(--text-muted);">${t('entities.provenanceConfidence')}</span><div class="mono" style="margin-top:0.125rem;">${prov.confidence}</div></div>`;
          }
          if (body === `<div style="display:flex;flex-direction:column;gap:0.75rem;">`) {
            body += `<div style="color:var(--text-muted);">${t('entities.noProvenance')}</div>`;
          }
          body += '</div>';
          showModal({ title: t('entities.provenance'), content: body, size: 'sm' });
        } catch (err) {
          showToast(t('entities.loadProvenanceFailed') + ': ' + err.message, 'error');
        } finally {
          provenanceBtn.disabled = false;
        }
      });
    }

    // Neo4j: Bind neighbors graph button
    const loadNeighborsBtn = overlay.querySelector('#load-neighbors-btn');
    if (loadNeighborsBtn && isNeo4j()) {
      let neighborNetwork = null;
      loadNeighborsBtn.addEventListener('click', async () => {
        const graphCanvas = overlay.querySelector('#neighbors-graph');
        if (!graphCanvas) return;
        if (neighborNetwork) { neighborNetwork.destroy(); neighborNetwork = null; }
        graphCanvas.innerHTML = `<div class="flex items-center justify-center h-full">${spinnerHtml()}</div>`;
        try {
          const res = await state.api.entityNeighbors(entity.absolute_id, graphId, 1);
          const data = res.data || {};
          const centerEntity = data.entity;
          const tier1 = GraphUtils.TIER_1;
          const nodes = [{ id: centerEntity.uuid, label: centerEntity.name || centerEntity.family_id || '?', font: { size: 14, bold: true }, shape: 'dot', size: 25, color: { background: tier1.bg, border: tier1.border } }];
          const nodeIds = new Set([centerEntity.uuid]);
          for (const n of (data.nodes || [])) {
            if (!nodeIds.has(n.uuid)) {
              nodes.push({ id: n.uuid, label: n.name || n.family_id || '?', font: { size: 12 }, shape: 'dot', size: 18 });
              nodeIds.add(n.uuid);
            }
          }
          const edges = (data.edges || []).map(e => ({
            from: e.source_uuid,
            to: e.target_uuid,
            label: e.content ? truncate(e.content, 25) : '',
            font: { size: 9, color: '#999' },
            arrows: 'to',
            smooth: { type: 'continuous' },
          }));
          if (neighborNetwork) neighborNetwork.destroy();
          neighborNetwork = new vis.Network(graphCanvas,
            { nodes: new vis.DataSet(nodes), edges: new vis.DataSet(edges) },
            GraphUtils.getPhysicsOptions()
          );
        } catch (err) {
          graphCanvas.innerHTML = `<div class="flex items-center justify-center h-full text-sm" style="color:var(--text-muted);">${escapeHtml(err.message)}</div>`;
        }
      });
    }

    // Fetch versions and relations in parallel
    const graphId = state.currentGraphId;
    const familyId = entity.family_id;

    try {
      const [versionsRes, relationsRes, contradictionsRes, embRes] = await Promise.all([
        state.api.entityVersions(familyId, graphId),
        state.api.entityRelations(familyId, graphId),
        state.api.entityContradictions(familyId, state.currentGraphId).catch(() => ({ data: [] })),
        state.api.entityEmbeddingPreview(entity.absolute_id, 5, graphId).catch(() => null),
      ]);

      const vSpinner = overlay.querySelector('#versions-spinner');
      if (vSpinner) vSpinner.remove();
      const rSpinner = overlay.querySelector('#relations-spinner');
      if (rSpinner) rSpinner.remove();
      const cSpinner = overlay.querySelector('#contradictions-spinner');
      if (cSpinner) cSpinner.remove();

      const versions = versionsRes.data || [];
      const relations = relationsRes.data || [];
      const contradictions = contradictionsRes.data || [];

      // Render embedding preview
      const embEl = overlay.querySelector('#entity-embedding-values');
      if (embEl) {
        if (embRes && embRes.data && embRes.data.values) {
          const vals = embRes.data.values;
          embEl.innerHTML = vals.map(v => `<span class="mono" style="display:inline-block;padding:2px 6px;margin:2px;border-radius:4px;background:var(--bg-secondary);font-size:0.7rem;">${v.toFixed(4)}</span>`).join('');
        } else {
          embEl.textContent = t('entities.noEmbedding');
        }
      }

      const versionsContainer = overlay.querySelector('#versions-container');
      if (versionsContainer) {
        versionsContainer.innerHTML = versions.length > 0
          ? buildVersionTimeline(versions, overlay)
          : `<div style="color:var(--text-muted);font-size:0.8125rem;">${t('entities.noVersionHistory')}</div>`;
        // Bind episode link clicks inside version timeline
        versionsContainer.querySelectorAll('.doc-link[data-cache-id]').forEach(el => {
          el.addEventListener('click', () => {
            const cacheId = el.getAttribute('data-cache-id');
            if (cacheId) window.showEpisodeDoc(cacheId);
          });
        });
      }

      const relationsContainer = overlay.querySelector('#relations-container');
      if (relationsContainer) {
        relationsContainer.innerHTML = relations.length > 0
          ? buildRelationsList(relations, familyId)
          : `<div style="color:var(--text-muted);font-size:0.8125rem;">${t('entities.noRelations')}</div>`;
      }

      const contradictionsContainer = overlay.querySelector('#contradictions-container');
      if (contradictionsContainer) {
        contradictionsContainer.innerHTML = renderContradictions(contradictions, familyId, overlay);
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

  // ---- Contradictions ----

  function renderContradictions(contradictions, familyId, overlay) {
    if (!Array.isArray(contradictions) || contradictions.length === 0) {
      return `<div style="color:var(--text-muted);font-size:0.8125rem;">${t('entities.noContradictions')}</div>`;
    }
    const items = contradictions.map((c, i) => {
      const severity = c.severity || 'medium';
      const severityColor = severity === 'high' ? 'var(--error)' : severity === 'low' ? 'var(--success)' : 'var(--warning)';
      return `
        <div style="padding:0.5rem 0;border-bottom:1px solid var(--border-color);">
          <div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.25rem;">
            <span style="width:8px;height:8px;border-radius:50%;background:${severityColor};"></span>
            <span style="font-size:0.8125rem;font-weight:500;">${escapeHtml(c.description || t('entities.contradictionBetween'))}</span>
          </div>
          ${c.version_a ? `<div style="font-size:0.8125rem;color:var(--text-secondary);margin-left:1rem;">${escapeHtml(truncate(c.version_a, 100))}</div>` : ''}
          ${c.version_b ? `<div style="font-size:0.8125rem;color:var(--text-secondary);margin-left:1rem;">${escapeHtml(truncate(c.version_b, 100))}</div>` : ''}
          ${c.contradiction_id ? `<button class="btn btn-ghost btn-sm" style="margin-top:0.25rem;margin-left:1rem;" onclick="event.stopPropagation();window._resolveContradiction('${escapeAttr(familyId)}',${i})">${t('entities.resolveContradiction')}</button>` : ''}
        </div>`;
    }).join('');
    // 存储矛盾数据以便后续使用
    if (!window._contradictionData) window._contradictionData = {};
    window._contradictionData[familyId] = contradictions;
    return `<div>${items}</div>`;
  }

  // Expose resolve contradiction handler
  window._resolveContradiction = async function(familyId, contradictionIndex) {
    try {
      const contradictions = (window._contradictionData && window._contradictionData[familyId]) || [];
      const contradiction = typeof contradictionIndex === 'number' ? contradictions[contradictionIndex] : null;
      if (!contradiction) {
        showToast(t('entities.resolveFailed') + ': contradiction not found', 'error');
        return;
      }
      await state.api.resolveContradiction(familyId, { contradiction }, state.currentGraphId);
      showToast(t('entities.resolveSuccess'), 'success');
    } catch (err) {
      showToast(t('entities.resolveFailed') + ': ' + err.message, 'error');
    }
  };

  // ---- Version Timeline ----

  function buildVersionTimeline(versions, overlay) {
    var familyId = versions.length > 0 && versions[0].family_id;

    // Build a simple content diff between adjacent versions
    function simpleContentDiff(current, previous) {
      if (!previous) return null;
      var curLines = (current.content || '').split('\n').filter(function(l) { return l.trim(); });
      var prevLines = (previous.content || '').split('\n').filter(function(l) { return l.trim(); });
      if (curLines.join('\n') === prevLines.join('\n')) return null;

      var added = [];
      var removed = [];
      curLines.forEach(function(line) {
        if (prevLines.indexOf(line) === -1) added.push(line);
      });
      prevLines.forEach(function(line) {
        if (curLines.indexOf(line) === -1) removed.push(line);
      });
      return { added: added, removed: removed };
    }

    function renderContentDiffInline(current, previous) {
      var diff = simpleContentDiff(current, previous);
      if (!diff) return '';
      var html = '<div style="margin-top:0.5rem;border-left:3px solid var(--primary);padding:0.375rem 0.5rem;background:var(--bg-input);border-radius:0 0.375rem 0.375rem 0;font-size:0.8125rem;">';
      diff.removed.forEach(function(line) {
        html += '<div style="color:var(--error);text-decoration:line-through;opacity:0.7;padding:1px 0;">- ' + escapeHtml(line) + '</div>';
      });
      diff.added.forEach(function(line) {
        html += '<div style="color:var(--success);padding:1px 0;">+ ' + escapeHtml(line) + '</div>';
      });
      html += '</div>';
      return html;
    }

    // Lazy diff loader using the backend API
    function loadFullDiff(versionIdx, sorted, diffContainerId) {
      var current = sorted[versionIdx];
      var prev = sorted[versionIdx + 1];
      if (!prev || !familyId) return;
      var container = overlay.querySelector('#' + diffContainerId);
      if (!container || container.dataset.loaded) return;
      container.dataset.loaded = 'true';
      container.innerHTML = '<div style="padding:0.5rem;color:var(--text-muted);font-size:0.75rem;">' + t('common.loading') + '...</div>';

      state.api.entityVersionDiff(familyId, prev.absolute_id, current.absolute_id, state.currentGraphId).then(function(res) {
        var sections = res.data && res.data.sections;
        if (!sections) { container.innerHTML = ''; return; }
        var sectionHtml = '';
        Object.keys(sections).forEach(function(key) {
          var s = sections[key];
          if (!s.changed) return;
          sectionHtml += '<div style="margin-bottom:0.5rem;">'
            + '<div style="font-size:0.75rem;font-weight:600;color:var(--text-primary);margin-bottom:0.125rem;">' + escapeHtml(key) + '</div>';
          if (s.old) {
            sectionHtml += '<div style="color:var(--error);font-size:0.8125rem;text-decoration:line-through;opacity:0.7;padding:2px 0.5rem;background:rgba(239,68,68,0.06);border-radius:0.25rem;margin-bottom:2px;">' + escapeHtml(s.old) + '</div>';
          }
          if (s.new_val) {
            sectionHtml += '<div style="color:var(--success);font-size:0.8125rem;padding:2px 0.5rem;background:rgba(34,197,94,0.06);border-radius:0.25rem;">' + escapeHtml(s.new_val) + '</div>';
          }
          sectionHtml += '</div>';
        });
        container.innerHTML = sectionHtml
          ? '<div style="border-left:3px solid var(--primary);padding:0.5rem;background:var(--bg-input);border-radius:0 0.375rem 0.375rem 0;">' + sectionHtml + '</div>'
          : '<div style="color:var(--text-muted);font-size:0.8125rem;padding:0.25rem 0;">' + t('entities.noContentChange') + '</div>';
      }).catch(function() {
        container.innerHTML = '<div style="color:var(--text-muted);font-size:0.8125rem;">Diff load failed</div>';
      });
    }

    return renderVersionTimeline({
      versions: versions,
      overlay: overlay,
      containerId: 'versions-container',
      toggleClass: 'version-expand-toggle',
      expandedIdPrefix: 'version-expanded',
      isActiveCheck: function(v) { return false; },
      renderHeader: function(v, i, sorted, isActive) {
        var prev = sorted[i + 1];
        var hasDiff = prev && (v.name !== prev.name || v.content !== prev.content);
        // Episode ID display
        var episodeLine = '';
        if (v.episode_id) {
          var shortEpId = v.episode_id.length > 16 ? v.episode_id.substring(0, 8) + '...' + v.episode_id.substring(v.episode_id.length - 4) : v.episode_id;
          episodeLine = '<span style="font-size:0.6875rem;color:var(--text-muted);">episode: <span class="doc-link mono" style="cursor:pointer;text-decoration:underline dotted;" data-cache-id="' + escapeHtml(v.episode_id) + '" title="' + escapeHtml(v.episode_id) + '">' + escapeHtml(shortEpId) + '</span></span>';
        }
        // Source label
        var sourceLine = '';
        if (v.source_document) {
          var src = v.source_document.length > 25 ? v.source_document.substring(0, 22) + '...' : v.source_document;
          sourceLine = '<span style="font-size:0.6875rem;color:var(--text-muted);">source: ' + escapeHtml(src) + '</span>';
        }
        return '<div style="display:flex;align-items:center;gap:0.5rem;">'
          + '<span class="mono" style="font-size:0.75rem;color:var(--text-muted);">' + t('graph.eventTime') + ' ' + formatDate(v.event_time) + '</span>'
          + '<span class="mono" style="font-size:0.7rem;color:var(--text-muted);">' + t('graph.processedTime') + ' ' + formatDateMs(v.processed_time) + '</span>'
          + (i === 0 ? '<span class="badge badge-info" style="font-size:0.6875rem;">' + t('entities.latest') + '</span>' : '')
          + (hasDiff ? '<span class="badge badge-primary" style="font-size:0.6875rem;">' + t('entities.changed') + '</span>' : '')
          + '</div>'
          + '<div style="margin-top:0.25rem;font-weight:500;font-size:0.875rem;">' + escapeHtml(v.name || '-') + '</div>'
          + '<div style="margin-top:0.125rem;color:var(--text-secondary);font-size:0.8125rem;" class="truncate">' + escapeHtml(truncate(v.content || '', 100)) + '</div>'
          + '<div style="margin-top:0.25rem;display:flex;align-items:center;gap:0.75rem;flex-wrap:wrap;">'
          + sourceLine + episodeLine
          + '</div>'
          + renderContentDiffInline(v, prev);
      },
      renderDiff: function(v, prev) {
        if (!prev || v.name === prev.name) return '';
        return '<div style="display:flex;gap:0.75rem;align-items:center;margin-top:0.5rem;padding:0.375rem 0.5rem;background:var(--bg-input);border-radius:0.375rem;font-size:0.8125rem;">'
          + '<span style="color:var(--error);text-decoration:line-through;">' + escapeHtml(prev.name) + '</span>'
          + '<i data-lucide="arrow-right" style="width:14px;height:14px;color:var(--text-muted);flex-shrink:0;"></i>'
          + '<span style="color:var(--success);">' + escapeHtml(v.name) + '</span>'
          + '</div>';
      },
      renderBody: function(v) {
        var idx = versions.indexOf(v);
        var sortedIdx = -1;
        var sorted = versions.slice().sort(function(a, b) {
          return (b.processed_time ? new Date(b.processed_time).getTime() : 0) - (a.processed_time ? new Date(a.processed_time).getTime() : 0);
        });
        sortedIdx = sorted.indexOf(v);
        var diffContainerId = 'version-fulldiff-' + sortedIdx;
        var hasPrev = sortedIdx < sorted.length - 1;

        return '<div class="md-content" style="background:var(--bg-input);border:1px solid var(--border-color);border-radius:0.375rem;padding:0.75rem;">'
          + renderMarkdown(v.content || '')
          + '</div>'
          + (hasPrev
            ? '<button class="btn btn-secondary" style="margin-top:0.5rem;font-size:0.75rem;padding:0.25rem 0.5rem;" onclick="window.__loadVersionDiff(\'' + diffContainerId + '\',' + sortedIdx + ')">'
              + '<i data-lucide="git-compare" style="width:12px;height:12px;vertical-align:middle;margin-right:4px;"></i>'
              + t('entities.showDiff')
              + '</button>'
              + '<div id="' + diffContainerId + '" style="margin-top:0.5rem;"></div>'
            : '');
      },
      onExpand: function(versionIdx, sorted) {
        // Auto-load diff on first expand
      },
    });
  }

  // Expose diff loader for onclick
  window.__loadVersionDiff = function(containerId, sortedIdx) {
    var container = document.getElementById(containerId);
    if (!container || container.dataset.loaded) return;
    // Find the overlay by walking up
    var versionsContainer = container.closest('#versions-container');
    if (!versionsContainer) return;
    var overlayEl = versionsContainer.closest('.modal-overlay') || versionsContainer.closest('.side-panel');
    if (!overlayEl) return;
    // Reconstruct sorted versions from DOM
    var familyIdEl = overlayEl.querySelector('[data-family-id]');
    if (!familyIdEl) return;
    var famId = familyIdEl.getAttribute('data-family-id');
    var graphId = state.currentGraphId;

    state.api.entityVersions(famId, graphId).then(function(res) {
      var versions = res.data || [];
      var sorted = versions.slice().sort(function(a, b) {
        return (b.processed_time ? new Date(b.processed_time).getTime() : 0) - (a.processed_time ? new Date(a.processed_time).getTime() : 0);
      });
      var current = sorted[sortedIdx];
      var prev = sorted[sortedIdx + 1];
      if (!prev) { container.innerHTML = ''; return; }

      container.dataset.loaded = 'true';
      container.innerHTML = '<div style="padding:0.5rem;color:var(--text-muted);font-size:0.75rem;">Loading diff...</div>';

      state.api.entityVersionDiff(famId, prev.absolute_id, current.absolute_id, graphId).then(function(diffRes) {
        var sections = diffRes.data && diffRes.data.sections;
        if (!sections) { container.innerHTML = ''; return; }
        var sectionHtml = '';
        Object.keys(sections).forEach(function(key) {
          var s = sections[key];
          if (!s.changed) return;
          sectionHtml += '<div style="margin-bottom:0.5rem;">'
            + '<div style="font-size:0.75rem;font-weight:600;color:var(--text-primary);margin-bottom:0.125rem;">' + escapeHtml(key) + '</div>';
          if (s.old) {
            sectionHtml += '<div style="color:var(--error);font-size:0.8125rem;text-decoration:line-through;opacity:0.7;padding:2px 0.5rem;background:rgba(239,68,68,0.06);border-radius:0.25rem;margin-bottom:2px;">' + escapeHtml(s.old) + '</div>';
          }
          if (s['new']) {
            sectionHtml += '<div style="color:var(--success);font-size:0.8125rem;padding:2px 0.5rem;background:rgba(34,197,94,0.06);border-radius:0.25rem;">' + escapeHtml(s['new']) + '</div>';
          }
          sectionHtml += '</div>';
        });
        container.innerHTML = sectionHtml
          ? '<div style="border-left:3px solid var(--primary);padding:0.5rem;background:var(--bg-input);border-radius:0 0.375rem 0.375rem 0;">' + sectionHtml + '</div>'
          : '<div style="color:var(--text-muted);font-size:0.8125rem;padding:0.25rem 0;">No content changes</div>';
      });
    });
  };

  // ---- Relations List ----

  function buildRelationsList(relations, currentFamilyId) {
    const items = relations.map(r => {
      const isEntity1 = r.entity1_absolute_id === currentFamilyId || r.entity1_family_id === currentFamilyId;
      const otherId = isEntity1
        ? (r.entity2_absolute_id || r.entity2_family_id)
        : (r.entity1_absolute_id || r.entity1_family_id);
      const direction = isEntity1 ? t('entities.to') : t('entities.from');

      return `
        <div style="padding:0.5rem 0;border-bottom:1px solid var(--border-color);">
          <div style="display:flex;align-items:flex-start;gap:0.5rem;">
            <i data-lucide="arrow-right" style="width:14px;height:14px;color:var(--text-muted);flex-shrink:0;margin-top:2px;"></i>
            <div style="flex:1;min-width:0;">
              <div class="md-content" style="font-size:0.8125rem;color:var(--text-primary);">${renderMarkdown(r.content || '-')}</div>
              <div style="margin-top:0.25rem;display:flex;align-items:center;gap:0.5rem;">
                <span class="badge badge-primary" style="font-size:0.6875rem;">${escapeHtml(direction)}</span>
                <span class="mono" style="font-size:0.75rem;color:var(--text-muted);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${escapeHtml(otherId || t('entities.unknown'))}</span>
                <span class="mono" style="font-size:0.6875rem;color:var(--text-muted);">${formatDate(r.event_time)}</span>
                <span class="mono" style="font-size:0.65rem;color:var(--text-muted);">${formatDateMs(r.processed_time)}</span>
              </div>
            </div>
          </div>
        </div>
      `;
    }).join('');

    return `<div>${items}</div>`;
  }

  // ---- Data Loading (server-side pagination) ----

  async function loadEntityBatch() {
    if (allLoading || !allHasMore) return;
    allLoading = true;
    const graphId = state.currentGraphId;
    try {
      const res = await state.api.listEntities(graphId, BATCH_SIZE, allOffset);
      const entities = res.data?.entities || res.data || [];
      if (entities.length === 0) {
        allHasMore = false;
      } else {
        allEntities = allEntities.concat(entities);
        allOffset += entities.length;
        if (entities.length < BATCH_SIZE) {
          allHasMore = false;
        }
        _cachedGraphId = graphId;
      }
    } finally {
      allLoading = false;
    }
  }

  async function fetchTotalCount() {
    try {
      const res = await state.api.getCounts(state.currentGraphId);
      totalCount = res.data ? (res.data.entity_count || null) : null;
    } catch (e) {
      totalCount = null;
    }
  }

  function updateCountDisplay() {
    const countEl = document.getElementById('entity-count');
    if (!countEl) return;
    if (isSearchMode) {
      countEl.textContent = t('entities.resultCount', { count: allEntities.length });
    } else if (totalCount !== null) {
      countEl.textContent = t('entities.showing', { shown: allEntities.length, total: totalCount });
    } else {
      countEl.textContent = t('entities.entityCount', { count: allEntities.length });
    }
  }

  function renderEntityTable() {
    const tableContainer = document.getElementById('entity-table-container');
    if (tableContainer) {
      tableContainer.innerHTML = buildEntityTable(allEntities);
      bindTableEvents(tableContainer);
      bindClickableRows(tableContainer);
    }
    updateCountDisplay();
  }

  async function searchEntities(query) {
    const seq = ++_searchSeq;
    const graphId = state.currentGraphId;
    const res = await state.api.searchEntities(query, graphId);
    if (seq !== _searchSeq) return; // stale result, discard
    allEntities = res.data || [];
    isSearchMode = true;
  }

  // ---- Event Binding ----

  function bindTableEvents(container) {
    container.querySelectorAll('tr[data-family-id]').forEach(row => {
      row.addEventListener('click', (e) => {
        // Don't trigger if clicking load-more button, checkboxes, or action buttons
        if (e.target.closest('#entity-load-more-btn')) return;
        if (e.target.closest('input[type="checkbox"]')) return;
        if (e.target.closest('button')) return;
        const familyId = row.getAttribute('data-family-id');
        const entity = allEntities.find(en => en.family_id === familyId);
        if (entity) openEntityDetail(entity);
      });
    });

    const loadMoreBtn = container.querySelector('#entity-load-more-btn');
    if (loadMoreBtn) {
      loadMoreBtn.addEventListener('click', async () => {
        loadMoreBtn.disabled = true;
        loadMoreBtn.textContent = t('common.loadMore') + '...';
        await loadEntityBatch();
        renderEntityTable();
      });
    }

    const batchDeleteBtn = container.querySelector('#batch-delete-entities-btn');
    if (batchDeleteBtn) {
      batchDeleteBtn.addEventListener('click', openBatchDeleteEntities);
    }

    const mergeBtn = container.querySelector('#merge-entities-btn');
    if (mergeBtn) {
      mergeBtn.addEventListener('click', openMergeEntities);
    }

    // Edit / Delete button delegation
    container.querySelectorAll('.btn-edit-entity').forEach(btn => {
      btn.addEventListener('click', (e) => {
        e.stopPropagation();
        const familyId = btn.getAttribute('data-family-id');
        const entity = allEntities.find(en => en.family_id === familyId);
        if (entity) openEditEntityModal(familyId, entity.name, entity.content || '', entity.summary || '', entity.attributes);
      });
    });
    container.querySelectorAll('.btn-delete-entity').forEach(btn => {
      btn.addEventListener('click', (e) => {
        e.stopPropagation();
        const familyId = btn.getAttribute('data-family-id');
        confirmDeleteEntity(familyId);
      });
    });
  }

  function bindSearchEvents(container) {
    const searchInput = container.querySelector('#entity-search-input');
    const listAllBtn = container.querySelector('#entity-list-all-btn');
    const refreshBtn = container.querySelector('#entity-refresh-btn');

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
            renderEntityTable();
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

    if (refreshBtn) {
      refreshBtn.addEventListener('click', () => {
        invalidateAndReload();
      });
    }
  }

  // Reset cache and reload from scratch
  async function invalidateAndReload() {
    allEntities = [];
    allOffset = 0;
    allHasMore = true;
    totalCount = null;
    isSearchMode = false;
    const searchInput = document.getElementById('entity-search-input');
    if (searchInput) searchInput.value = '';
    const tableContainer = document.getElementById('entity-table-container');
    if (tableContainer) {
      tableContainer.innerHTML = `<div style="display:flex;justify-content:center;padding:2rem;">${spinnerHtml()}</div>`;
    }
    try {
      await Promise.all([loadEntityBatch(), fetchTotalCount()]);
      renderEntityTable();
    } catch (err) {
      showToast(t('entities.loadFailed') + '：' + err.message, 'error');
      if (tableContainer) {
        tableContainer.innerHTML = emptyState(t('entities.loadFailed'), 'alert-triangle');
      }
    }
  }

  async function resetToListAll() {
    // If we have cached data AND graph hasn't changed, just re-render from cache (no API call)
    if (allEntities.length > 0 && !isSearchMode && _cachedGraphId === state.currentGraphId) {
      renderEntityTable();
      const searchInput = document.getElementById('entity-search-input');
      if (searchInput) searchInput.value = '';
      return;
    }
    await invalidateAndReload();
  }

  // ---- Edit Entity ----

  function openEditEntityModal(familyId, currentName, currentContent, currentSummary, currentAttributes) {
    const html = `
      <div class="form-group">
        <label class="form-label" data-i18n="entities.name">${t('entities.name')}</label>
        <input type="text" id="editEntityName" class="input" value="${escapeAttr(currentName)}">
      </div>
      <div class="form-group">
        <label class="form-label" data-i18n="entities.content">${t('entities.content')}</label>
        <textarea id="editEntityContent" class="input" rows="4">${escapeAttr(currentContent)}</textarea>
      </div>
      <div class="form-group">
        <label class="form-label" data-i18n="entities.summary">${t('entities.summary')}</label>
        <textarea id="editEntitySummary" class="input" rows="3">${escapeAttr(currentSummary || '')}</textarea>
      </div>
      <div class="form-group">
        <label class="form-label" data-i18n="entities.attributes">${t('entities.attributes')}</label>
        <textarea id="editEntityAttributes" class="input" rows="2" placeholder="key1: value1, key2: value2">${escapeAttr(currentAttributes ? JSON.stringify(currentAttributes) : '')}</textarea>
      </div>`;

    const footer = `
      <button class="btn btn-secondary modal-cancel-btn">${t('common.cancel')}</button>
      <button class="btn btn-primary modal-save-btn">${t('common.save')}</button>`;

    const { overlay, close } = showModal({
      title: t('entities.editTitle'),
      content: html,
      footer: footer,
      size: 'sm',
    });
    _currentModalClose = close;

    overlay.querySelector('.modal-cancel-btn').addEventListener('click', close);
    overlay.querySelector('.modal-save-btn').addEventListener('click', () => submitEditEntity(familyId, close));
  }

  async function submitEditEntity(familyId, close) {
    const name = document.getElementById('editEntityName').value.trim();
    const content = document.getElementById('editEntityContent').value.trim();
    const summary = document.getElementById('editEntitySummary').value.trim();
    const attributesStr = document.getElementById('editEntityAttributes').value.trim();
    let attributes = undefined;
    if (attributesStr) {
      try { attributes = JSON.parse(attributesStr); } catch { /* ignore invalid JSON */ }
    }
    if (!name && !content) { showToast(t('entities.nameRequired'), 'error'); return; }
    try {
      const data = { name: name || undefined, content: content || undefined };
      if (summary) data.summary = summary;
      if (attributes) data.attributes = attributes;
      const res = await state.api.updateEntity(familyId, data, state.currentGraphId);
      if (res.error) { showToast(res.error, 'error'); return; }
      showToast(t('entities.updateSuccess'), 'success');
      close();
      invalidateAndReload();
    } catch (e) { showToast(t('entities.updateFailed') + ': ' + e.message, 'error'); }
  }

  // ---- Delete Entity ----

  function confirmDeleteEntity(familyId) {
    const html = `
      <p>${t('entities.deleteConfirm')}</p>
      <label class="checkbox-label" style="display:flex;align-items:center;gap:0.5rem;margin-top:0.5rem;">
        <input type="checkbox" id="deleteCascade"> ${t('entities.cascadeDelete')}
      </label>`;

    const footer = `
      <button class="btn btn-secondary modal-cancel-btn">${t('common.cancel')}</button>
      <button class="btn btn-danger modal-confirm-btn">${t('common.confirm')}</button>`;

    const { overlay, close } = showModal({
      title: t('entities.deleteTitle'),
      content: html,
      footer: footer,
      size: 'sm',
    });
    _currentModalClose = close;

    overlay.querySelector('.modal-cancel-btn').addEventListener('click', close);
    overlay.querySelector('.modal-confirm-btn').addEventListener('click', () => executeDeleteEntity(familyId, close));
  }

  async function executeDeleteEntity(familyId, close) {
    const confirmBtn = document.querySelector('.modal-confirm-btn');
    if (confirmBtn) { confirmBtn.disabled = true; confirmBtn.textContent = '...'; }
    const cascade = document.getElementById('deleteCascade')?.checked || false;
    try {
      const res = await state.api.deleteEntity(familyId, cascade, state.currentGraphId);
      if (res.error) { showToast(res.error, 'error'); return; }
      showToast(t('entities.deleteSuccess'), 'success');
      close();
      invalidateAndReload();
    } catch (e) { showToast(t('entities.deleteFailed') + ': ' + e.message, 'error'); }
    finally { if (confirmBtn) { confirmBtn.disabled = false; } }
  }

  // ---- Batch Delete & Merge ----

  function toggleAllEntityCheckboxes(el) {
    document.querySelectorAll('.entity-checkbox').forEach(cb => cb.checked = el.checked);
  }

  function getSelectedEntityIds() {
    return [...document.querySelectorAll('.entity-checkbox:checked')].map(cb => cb.value);
  }

  async function openBatchDeleteEntities() {
    const ids = getSelectedEntityIds();
    if (!ids.length) { showToast(t('entities.selectEntities'), 'warning'); return; }
    const ok = await showConfirm({ message: t('entities.deleteConfirm') + ' (' + ids.length + ')', destructive: true });
    if (!ok) return;
    state.api.batchDeleteEntities(ids, false, state.currentGraphId).then(res => {
      if (res.error) { showToast(res.error, 'error'); return; }
      showToast(t('entities.batchDeleteSuccess').replace('{count}', ids.length), 'success');
      invalidateAndReload();
    }).catch(e => showToast(t('entities.deleteFailed') + ': ' + e.message, 'error'));
  }

  async function openMergeEntities() {
    const ids = getSelectedEntityIds();
    if (ids.length < 2) { showToast(t('entities.selectEntities') + ' (>=2)', 'warning'); return; }
    const target = ids[0];
    const sources = ids.slice(1);
    const ok = await showConfirm({ message: t('entities.mergeConfirm') });
    if (!ok) return;
    state.api.mergeEntities(target, sources, state.currentGraphId).then(res => {
      if (res.error) { showToast(res.error, 'error'); return; }
      showToast(t('entities.mergeSuccess'), 'success');
      invalidateAndReload();
    }).catch(e => showToast(t('entities.mergeFailed') + ': ' + e.message, 'error'));
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

    // If cache has data AND graph hasn't changed, render immediately without API call
    if (allEntities.length > 0 && !isSearchMode && _cachedGraphId === state.currentGraphId) {
      renderEntityTable();
      if (window.lucide) lucide.createIcons({ nodes: [container] });
      return;
    }

    // First load or after search mode — fetch from server
    try {
      if (isSearchMode) {
        // Coming back from search, reset to list mode
        isSearchMode = false;
        allEntities = [];
        allOffset = 0;
        allHasMore = true;
      }
      await Promise.all([loadEntityBatch(), fetchTotalCount()]);
      renderEntityTable();
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
    // Only clear UI state, preserve cache (allEntities, allOffset, allHasMore, totalCount)
    clearTimeout(debounceTimer);
    debounceTimer = null;
    _currentModalClose = null;
  }

  // Expose globally for use by other pages (search, relations, path-finder) and inline onclick handlers
  window.showEntityDetail = openEntityDetail;
  window.openEditEntityModal = openEditEntityModal;
  window.confirmDeleteEntity = confirmDeleteEntity;
  window.toggleAllEntityCheckboxes = toggleAllEntityCheckboxes;

  registerPage('entities', { render, destroy });
})();

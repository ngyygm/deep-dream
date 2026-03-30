/* ==========================================
   Entities Page - Entity Browser
   ========================================== */

function escapeAttr(s) {
  return String(s).replace(/&/g,'&amp;').replace(/"/g,'&quot;').replace(/'/g,'&#39;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

(function() {
  const PAGE_SIZE = 50;
  let debounceTimer = null;
  let allEntities = [];
  let displayedCount = 0;
  let isSearchMode = false;
  let isSearchAllMode = false;
  let _currentModalClose = null;

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
        <td><input type="checkbox" class="entity-checkbox" value="${escapeAttr(e.entity_id)}"></td>
        <td style="max-width:180px;font-weight:500;">${escapeHtml(e.name || '-')}</td>
        <td style="max-width:300px;" class="truncate" title="${escapeHtml(e.content || '')}">${escapeHtml(truncate(e.content || '', 60))}</td>
        <td style="white-space:nowrap;">${formatDate(e.event_time)}</td>
        <td style="max-width:120px;" class="truncate" title="${escapeHtml(e.doc_name || e.source_document || '')}">${escapeHtml(e.doc_name || e.source_document || '-')}</td>
        <td style="text-align:center;">
          <span class="badge badge-info">${escapeHtml(String(e.version_count || '?'))}</span>
        </td>
        <td>
          <button class="btn btn-sm btn-primary" onclick="event.stopPropagation(); openEditEntityModal('${escapeAttr(e.entity_id)}', '${escapeAttr(e.name)}', '${escapeAttr(e.content || '')}', '${escapeAttr(e.summary || '')}', '${escapeAttr(e.attributes ? JSON.stringify(e.attributes).replace(/'/g, "\\'") : '')}')" data-i18n="entities.edit">Edit</button>
          <button class="btn btn-sm btn-danger" onclick="event.stopPropagation(); confirmDeleteEntity('${escapeAttr(e.entity_id)}')" data-i18n="entities.delete">Delete</button>
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
            <span style="font-size:0.75rem;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.05em;">${t('graph.eventTime')}</span>
            <div class="mono" style="margin-top:0.125rem;">${formatDate(entity.event_time)}</div>
          </div>
          <div>
            <span style="font-size:0.75rem;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.05em;">${t('graph.processedTime')}</span>
            <div class="mono" style="margin-top:0.125rem;">${formatDate(entity.processed_time)}</div>
          </div>
          <div>
            <span style="font-size:0.75rem;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.05em;">${t('graph.sourceDoc')}</span>
            <div style="margin-top:0.125rem;">${escapeHtml(entity.doc_name || entity.source_document || '-')}</div>
          </div>
        </div>
        ${entity.memory_cache_id ? `
        <div>
          <span style="font-size:0.75rem;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.05em;">${t('graph.memoryCacheId')}</span>
          <div class="mono doc-link" data-cache-id="${escapeHtml(entity.memory_cache_id)}" style="margin-top:0.125rem;">${escapeHtml(entity.memory_cache_id)}</div>
        </div>
        ` : ''}
        ${entity.summary ? `
        <div>
          <span style="font-size:0.75rem;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.05em;">${t('entities.summary')}</span>
          <div style="margin-top:0.125rem;line-height:1.6;font-size:0.85rem;">${escapeHtml(entity.summary)}</div>
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
      title: entity.name || entity.entity_id,
      content: modalContent.innerHTML,
      size: 'lg',
    });

    if (window.lucide) lucide.createIcons({ nodes: [overlay] });

    // Bind doc link clicks
    overlay.querySelectorAll('.doc-link').forEach(el => {
      el.addEventListener('click', () => {
        const cacheId = el.getAttribute('data-cache-id');
        if (cacheId) window.showDocContent(cacheId);
      });
    });

    // Evolve summary button
    const evolveBtn = overlay.querySelector('#evolve-summary-btn');
    if (evolveBtn) {
      evolveBtn.addEventListener('click', async () => {
        evolveBtn.disabled = true;
        evolveBtn.innerHTML = `${spinnerHtml('spinner-sm')} ${t('entities.evolveSummaryRunning')}`;
        try {
          const res = await state.api.evolveEntitySummary(entity.entity_id);
          showToast(t('entities.evolveSummarySuccess'), 'success');
          // Refresh entity detail
          if (res.data) {
            entity.summary = res.data.summary || entity.summary;
            entity.attributes = res.data.attributes || entity.attributes;
          }
          resetToListAll();
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
          const res = await state.api.entityProvenance(entity.entity_id);
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
          const nodes = [{ id: centerEntity.uuid, label: centerEntity.name || centerEntity.entity_id || '?', font: { size: 14, bold: true }, shape: 'dot', size: 25, color: { background: '#ef4444', border: '#f87171' } }];
          const nodeIds = new Set([centerEntity.uuid]);
          for (const n of (data.nodes || [])) {
            if (!nodeIds.has(n.uuid)) {
              nodes.push({ id: n.uuid, label: n.name || n.entity_id || '?', font: { size: 12 }, shape: 'dot', size: 18 });
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
    const entityId = entity.entity_id;

    try {
      const [versionsRes, relationsRes, contradictionsRes] = await Promise.all([
        state.api.entityVersions(entityId, graphId),
        state.api.entityRelations(entityId, graphId),
        state.api.entityContradictions(entityId).catch(() => ({ data: [] })),
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

      const versionsContainer = overlay.querySelector('#versions-container');
      if (versionsContainer) {
        versionsContainer.innerHTML = versions.length > 0
          ? buildVersionTimeline(versions, overlay)
          : `<div style="color:var(--text-muted);font-size:0.8125rem;">${t('entities.noVersionHistory')}</div>`;
      }

      const relationsContainer = overlay.querySelector('#relations-container');
      if (relationsContainer) {
        relationsContainer.innerHTML = relations.length > 0
          ? buildRelationsList(relations, entityId)
          : `<div style="color:var(--text-muted);font-size:0.8125rem;">${t('entities.noRelations')}</div>`;
      }

      const contradictionsContainer = overlay.querySelector('#contradictions-container');
      if (contradictionsContainer) {
        contradictionsContainer.innerHTML = renderContradictions(contradictions, entityId, overlay);
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

  function renderContradictions(contradictions, entityId, overlay) {
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
          ${c.contradiction_id ? `<button class="btn btn-ghost btn-sm" style="margin-top:0.25rem;margin-left:1rem;" onclick="event.stopPropagation();window._resolveContradiction('${escapeAttr(entityId)}','${escapeAttr(c.contradiction_id)}')">${t('entities.resolveContradiction')}</button>` : ''}
        </div>`;
    }).join('');
    return `<div>${items}</div>`;
  }

  // Expose resolve contradiction handler
  window._resolveContradiction = async function(entityId, contradictionId) {
    try {
      await state.api.resolveContradiction(entityId, { contradiction_id: contradictionId });
      showToast(t('entities.resolveSuccess'), 'success');
    } catch (err) {
      showToast(t('entities.resolveFailed') + ': ' + err.message, 'error');
    }
  };

  // ---- Version Timeline ----

  function buildVersionTimeline(versions, overlay) {
    // Sort versions by processed_time descending (newest first)
    const sorted = [...versions].sort((a, b) => {
      const ta = a.processed_time ? new Date(a.processed_time).getTime() : 0;
      const tb = b.processed_time ? new Date(b.processed_time).getTime() : 0;
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
              <span class="mono" style="font-size:0.75rem;color:var(--text-muted);">${formatDate(v.processed_time)}</span>
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
      if (!overlay) return;
      const container = overlay.querySelector('#versions-container');
      if (!container) return;
      container.querySelectorAll('.version-expand-toggle').forEach(toggle => {
        toggle.addEventListener('click', () => {
          const idx = toggle.getAttribute('data-version-idx');
          const expanded = overlay.querySelector('#version-expanded-' + idx);
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
                <span class="mono" style="font-size:0.6875rem;color:var(--text-muted);">${formatDate(r.event_time)}</span>
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
        // Don't trigger if clicking load-more button, checkboxes, or action buttons
        if (e.target.closest('#entity-load-more-btn')) return;
        if (e.target.closest('input[type="checkbox"]')) return;
        if (e.target.closest('button')) return;
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

    const batchDeleteBtn = container.querySelector('#batch-delete-entities-btn');
    if (batchDeleteBtn) {
      batchDeleteBtn.addEventListener('click', openBatchDeleteEntities);
    }

    const mergeBtn = container.querySelector('#merge-entities-btn');
    if (mergeBtn) {
      mergeBtn.addEventListener('click', openMergeEntities);
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

  // ---- Edit Entity ----

  function openEditEntityModal(entityId, currentName, currentContent, currentSummary, currentAttributes) {
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
    overlay.querySelector('.modal-save-btn').addEventListener('click', () => submitEditEntity(entityId, close));
  }

  async function submitEditEntity(entityId, close) {
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
      const res = await state.api.updateEntity(entityId, data);
      if (res.error) { showToast(res.error, 'error'); return; }
      showToast(t('entities.updateSuccess'), 'success');
      close();
      resetToListAll();
    } catch (e) { showToast(t('entities.updateFailed') + ': ' + e.message, 'error'); }
  }

  // ---- Delete Entity ----

  function confirmDeleteEntity(entityId) {
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
    overlay.querySelector('.modal-confirm-btn').addEventListener('click', () => executeDeleteEntity(entityId, close));
  }

  async function executeDeleteEntity(entityId, close) {
    const cascade = document.getElementById('deleteCascade')?.checked || false;
    try {
      const res = await state.api.deleteEntity(entityId, cascade);
      if (res.error) { showToast(res.error, 'error'); return; }
      showToast(t('entities.deleteSuccess'), 'success');
      close();
      resetToListAll();
    } catch (e) { showToast(t('entities.deleteFailed') + ': ' + e.message, 'error'); }
  }

  // ---- Batch Delete & Merge ----

  function toggleAllEntityCheckboxes(el) {
    document.querySelectorAll('.entity-checkbox').forEach(cb => cb.checked = el.checked);
  }

  function getSelectedEntityIds() {
    return [...document.querySelectorAll('.entity-checkbox:checked')].map(cb => cb.value);
  }

  function openBatchDeleteEntities() {
    const ids = getSelectedEntityIds();
    if (!ids.length) { showToast(t('entities.selectEntities'), 'warn'); return; }
    if (!confirm(t('entities.deleteConfirm') + ' (' + ids.length + ')')) return;
    state.api.batchDeleteEntities(ids).then(res => {
      if (res.error) { showToast(res.error, 'error'); return; }
      showToast(t('entities.batchDeleteSuccess').replace('{count}', ids.length), 'success');
      resetToListAll();
    }).catch(e => showToast(t('entities.deleteFailed') + ': ' + e.message, 'error'));
  }

  function openMergeEntities() {
    const ids = getSelectedEntityIds();
    if (ids.length < 2) { showToast(t('entities.selectEntities') + ' (>=2)', 'warn'); return; }
    const target = ids[0];
    const sources = ids.slice(1);
    if (!confirm(t('entities.mergeConfirm'))) return;
    state.api.mergeEntities(target, sources).then(res => {
      if (res.error) { showToast(res.error, 'error'); return; }
      showToast(t('entities.mergeSuccess'), 'success');
      resetToListAll();
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

  // Expose globally for use by other pages (search, relations, path-finder) and inline onclick handlers
  window.showEntityDetail = openEntityDetail;
  window.openEditEntityModal = openEditEntityModal;
  window.confirmDeleteEntity = confirmDeleteEntity;
  window.toggleAllEntityCheckboxes = toggleAllEntityCheckboxes;

  registerPage('entities', { render, destroy });
})();

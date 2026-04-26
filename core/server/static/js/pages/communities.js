/* ==========================================
   Communities Page - Community Detection
   ========================================== */
registerPage('communities', (function () {
  'use strict';

  let _container = null;
  let _communities = [];
  let _total = 0;
  let _offset = 0;
  let _limit = 50;
  let _detectResult = null;
  let _loading = false;
  let _allLoaded = false;

  // Community-scoped data
  let _commEntities = [];
  let _commRelations = [];
  let _commEntityMap = {};

  // GraphExplorer instance (created per detail page)
  let explorer = null;

  async function _detect() {
    if (_loading) return;
    _loading = true;
    _updateUI();

    try {
      const algorithm = document.getElementById('comm-algorithm')?.value || 'louvain';
      const resolution = parseFloat(document.getElementById('comm-resolution')?.value || '1.0');

      const res = await state.api.detectCommunities(state.currentGraphId, algorithm, resolution);
      _detectResult = res.data || {};
      showToast(t('communities.detectSuccess', { count: _detectResult.total_communities, time: _detectResult.elapsed_seconds }), 'success');
      _loadCommunities(true);
    } catch (err) {
      showToast(err.message, 'error');
    } finally {
      _loading = false;
      _updateUI();
    }
  }

  async function _loadCommunities(reset) {
    if (_loading) return;
    _loading = true;

    if (reset) {
      _offset = 0;
      _communities = [];
      _allLoaded = false;
      _renderCards();
    }

    try {
      const minSize = parseInt(document.getElementById('comm-min-size')?.value || '3');
      const res = await state.api.listCommunities(state.currentGraphId, minSize, _limit, _offset);
      const data = res.data || {};
      const newCommunities = data.communities || [];
      _total = data.total || 0;

      if (reset) {
        _communities = newCommunities;
      } else {
        _communities = _communities.concat(newCommunities);
      }
      _offset += newCommunities.length;
      _allLoaded = (_total > 0 && _communities.length >= _total) || (newCommunities.length < _limit);
      _renderCards();
    } catch (err) {
      showToast(err.message, 'error');
    } finally {
      _loading = false;
    }
  }

  async function _clearCommunities() {
    const ok = await showConfirm({ message: t('communities.clearConfirm'), destructive: true });
    if (!ok) return;
    try {
      await state.api.clearCommunities(state.currentGraphId);
      _communities = [];
      _total = 0;
      _offset = 0;
      _allLoaded = false;
      _detectResult = null;
      showToast(t('communities.clearSuccess'), 'success');
      _renderCards();
      _updateUI();
    } catch (err) {
      showToast(err.message, 'error');
    }
  }

  function _updateUI() {
    const btn = document.getElementById('comm-detect-btn');
    if (btn) {
      btn.disabled = _loading;
      btn.innerHTML = _loading
        ? `<div class="spinner spinner-sm" style="margin-right:6px;"></div>${t('communities.detecting')}`
        : `<i data-lucide="scan" style="width:16px;height:16px;"></i>${t('communities.detect')}`;
      if (window.lucide) lucide.createIcons();
    }
  }

  function _renderCards() {
    const cardsEl = _container.querySelector('#comm-cards');
    const statsEl = _container.querySelector('#comm-stats');
    if (!cardsEl) return;

    if (_detectResult) {
      const sizes = _detectResult.community_sizes || [];
      statsEl.innerHTML = `
        <div class="card p-3 flex items-center gap-4 text-sm flex-wrap">
          <span style="color:var(--text-muted);">${t('communities.stats')}:</span>
          <span><strong>${_detectResult.total_communities}</strong> ${t('communities.communitiesCount')}</span>
          <span>${t('communities.largest')}: <strong>${sizes[0] || 0}</strong></span>
          <span>${t('communities.avgSize')}: <strong>${sizes.length ? Math.round(sizes.reduce((a, b) => a + b, 0) / sizes.length) : 0}</strong></span>
          <span>${t('communities.elapsed')}: <strong>${_detectResult.elapsed_seconds}s</strong></span>
        </div>`;
    } else {
      statsEl.innerHTML = '';
    }

    if (_communities.length === 0 && !_loading) {
      cardsEl.innerHTML = emptyState(t('communities.noCommunities'), 'layout-grid');
      if (window.lucide) lucide.createIcons();
      return;
    }

    const palette = GraphUtils.COMMUNITY_PALETTE;

    let html = '<div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">';
    for (const c of _communities) {
      const color = palette[c.community_id % palette.length].bg;
      const topMembers = (c.members || []).slice(0, 5).map(m => m.name || m.family_id).join(', ');
      html += `
        <div class="card p-4 cursor-pointer" onclick="navigate('#communities/${c.community_id}')">
          <div class="flex items-center gap-2 mb-2">
            <div style="width:12px;height:12px;border-radius:50%;background:${color};flex-shrink:0;"></div>
            <span class="font-semibold">${t('communities.community')} #${c.community_id}</span>
            <span class="badge badge-info">${c.size} ${t('common.entities')}</span>
          </div>
          <p class="text-xs" style="color:var(--text-muted);line-height:1.5;">${escapeHtml(topMembers)}</p>
        </div>`;
    }
    html += '</div>';

    // Load-more sentinel for infinite scroll
    if (!_allLoaded) {
      html += `<div id="comm-sentinel" style="padding:1rem;text-align:center;color:var(--text-muted);font-size:0.85rem;">
        ${_loading ? `${spinnerHtml('spinner-sm')} ${t('common.loading')}` : ''}
      </div>`;
    } else if (_communities.length > 0) {
      const displayTotal = _total > 0 ? _total : _communities.length;
      html += `<div style="padding:0.75rem;text-align:center;color:var(--text-muted);font-size:0.8125rem;">
        ${t('communities.loadedCount', {loaded: _communities.length, total: displayTotal})}
      </div>`;
    }

    cardsEl.innerHTML = html;
    if (window.lucide) lucide.createIcons();

    // Observe sentinel for infinite scroll
    const sentinel = cardsEl.querySelector('#comm-sentinel');
    if (sentinel) {
      _observeSentinel(sentinel);
    }
  }

  let _sentinelObserver = null;

  function _observeSentinel(sentinel) {
    if (_sentinelObserver) _sentinelObserver.disconnect();
    _sentinelObserver = new IntersectionObserver((entries) => {
      if (entries[0].isIntersecting && !_loading && !_allLoaded) {
        _loadCommunities(false);
      }
    }, { rootMargin: '200px' });
    _sentinelObserver.observe(sentinel);
  }

  // ================================================
  //  Full-page detail view — graph-explorer style
  // ================================================

  async function _renderDetailPage(container, cid) {
    const palette = GraphUtils.COMMUNITY_PALETTE;
    const color = palette[cid % palette.length].bg;

    // Render skeleton
    container.innerHTML = `
      <div class="page-enter">
        <!-- Top bar -->
        <div class="card mb-4">
          <div class="card-header">
            <div style="display:flex;align-items:center;gap:0.5rem;">
              <button onclick="navigate('#communities')" class="btn btn-ghost btn-sm">
                <i data-lucide="arrow-left" style="width:16px;height:16px;"></i>${t('common.back')}
              </button>
              <div style="width:12px;height:12px;border-radius:50%;background:${color};"></div>
              <span class="badge badge-primary mono">${t('communities.community')} #${cid}</span>
              <span id="comm-focus-badge" class="badge badge-warning" style="display:none;">
                <i data-lucide="crosshair" style="width:12px;height:12px;margin-right:2px;"></i>
                ${t('graph.focusMode')}
              </span>
            </div>
            <span id="comm-graph-stats" class="mono" style="font-size:0.8125rem;color:var(--text-muted);"></span>
          </div>
          <div style="display:flex;align-items:center;gap:0.5rem;">
            <button class="btn btn-primary btn-sm" id="comm-load-btn">
              <i data-lucide="refresh-cw" style="width:14px;height:14px;"></i>${t('graph.loadGraph')}
            </button>
            <button class="btn btn-secondary btn-sm" id="comm-exit-focus-btn" style="display:none;">
              <i data-lucide="maximize-2" style="width:14px;height:14px;"></i>${t('graph.exitFocus')}
            </button>
          </div>
        </div>

        <!-- Main body: graph + sidebar -->
        <div class="flex gap-4" style="height:calc(100vh - 240px);min-height:400px;">
          <!-- Graph canvas -->
          <div class="flex-1 relative" style="min-width:0;">
            <div id="comm-detail-graph" style="width:100%;height:100%;"></div>
            <div id="comm-graph-loading" class="absolute inset-0 flex items-center justify-center" style="background:var(--bg-input);border-radius:0.5rem;">
              ${spinnerHtml()}
            </div>
          </div>

          <!-- Detail sidebar -->
          <div style="width:30%;min-width:280px;max-width:420px;">
            <div class="card h-full flex flex-col">
              <div class="card-header">
                <span class="card-title">${t('common.detail')}</span>
              </div>
              <div id="comm-detail-content" style="overflow-y:auto;flex:1;">
                ${emptyState(t('common.clickToView'), 'mouse-pointer-click')}
              </div>
            </div>
          </div>
        </div>
      </div>
    `;
    if (window.lucide) lucide.createIcons();

    // ---- Create GraphExplorer instance for community detail ----
    explorer = GraphExplorer.create({
      canvasId: 'comm-detail-graph',
      detailContentId: 'comm-detail-content',
      loadingId: 'comm-graph-loading',
      exitFocusBtnId: 'comm-exit-focus-btn',
      focusBadgeId: 'comm-focus-badge',
      idPrefix: 'comm-',
      entityCache: _commEntityMap,
      defaultHopLevel: 1,
      familyIdToLatest: function () {
        var map = {};
        for (var i = 0; i < _commEntities.length; i++) {
          map[_commEntities[i].family_id] = _commEntities[i].uuid;
        }
        return map;
      },
      onRestoreDefaultView: function () {
        var graphNodes = _commEntities.map(function (m) {
          return {
            absolute_id: m.uuid,
            family_id: m.family_id,
            name: m.name,
            content: m.content || '',
          };
        });
        return {
          entities: graphNodes,
          relations: _commRelations,
          inheritedRelationIds: null,
        };
      },
      onFilterRelations: function (relations) {
        var commUuids = new Set(_commEntities.map(function (e) { return e.uuid; }));
        return relations.filter(function (r) {
          return commUuids.has(r.entity1_absolute_id) && commUuids.has(r.entity2_absolute_id);
        });
      },
    });

    // Bind buttons
    document.getElementById('comm-load-btn').addEventListener('click', function () { _loadCommunityGraph(cid); });
    document.getElementById('comm-exit-focus-btn').addEventListener('click', function () { explorer.exitFocus(); });

    // Initial load
    await _loadCommunityGraph(cid);
  }

  // ---- Load community data and build graph ----

  async function _loadCommunityGraph(cid) {
    const loadingEl = document.getElementById('comm-graph-loading');
    const statsEl = document.getElementById('comm-graph-stats');
    if (loadingEl) loadingEl.style.display = 'flex';
    if (statsEl) statsEl.textContent = t('common.loading');

    try {
      let [community, graphData] = await Promise.all([
        state.api.getCommunity(cid, state.currentGraphId),
        state.api.getCommunityGraph(cid, state.currentGraphId),
      ]);

      community = community.data || {};
      graphData = graphData.data || {};

      const members = community.members || [];
      const rawRelations = community.relations || [];

      // Build entity map from community members (uuid = absolute_id)
      _commEntities = members;
      _commEntityMap = {};
      for (const m of members) {
        _commEntityMap[m.uuid] = m;
      }

      // Convert community relations to standard relation format
      _commRelations = rawRelations.map(function (r) {
        return {
          absolute_id: r.relation_uuid || r.source_uuid + '_' + r.target_uuid,
          entity1_absolute_id: r.source_uuid,
          entity2_absolute_id: r.target_uuid,
          content: r.content || '',
        };
      });

      // Convert graph nodes to entity format
      const graphNodes = (graphData.nodes || []).map(function (n) {
        return {
          absolute_id: n.uuid,
          family_id: n.family_id,
          name: n.name,
          content: _commEntityMap[n.uuid] ? (_commEntityMap[n.uuid].content || '') : '',
        };
      });

      // Convert graph edges
      const graphEdges = (graphData.edges || []).map(function (e) {
        return {
          absolute_id: (e.source_uuid || '') + '_' + (e.target_uuid || ''),
          entity1_absolute_id: e.source_uuid,
          entity2_absolute_id: e.target_uuid,
          content: e.content || '',
        };
      });

      // Fetch version counts
      const allEntityIds = [];
      var seenIds = new Set();
      for (var i = 0; i < graphNodes.length; i++) {
        if (!seenIds.has(graphNodes[i].family_id)) {
          allEntityIds.push(graphNodes[i].family_id);
          seenIds.add(graphNodes[i].family_id);
        }
      }
      try {
        const vcRes = await state.api.entityVersionCounts(allEntityIds, state.currentGraphId);
        explorer.setVersionCounts(vcRes.data || {});
      } catch (_) {}

      // Update entity cache for explorer
      explorer.setEntityCache(_commEntityMap);

      explorer.buildGraph(graphNodes, graphEdges, null, null, null);

      // Clear focus UI
      var exitBtn = document.getElementById('comm-exit-focus-btn');
      if (exitBtn) exitBtn.style.display = 'none';
      var focusBadge = document.getElementById('comm-focus-badge');
      if (focusBadge) focusBadge.style.display = 'none';

      if (statsEl) {
        statsEl.textContent = t('graph.loaded', { entities: graphNodes.length, relations: graphEdges.length });
      }

      // Reset detail sidebar
      const detailContent = document.getElementById('comm-detail-content');
      if (detailContent) {
        detailContent.innerHTML = emptyState(t('common.clickToView'), 'mouse-pointer-click');
        if (window.lucide) lucide.createIcons();
      }
    } catch (err) {
      showToast(err.message, 'error');
      if (statsEl) statsEl.textContent = t('common.error');
    } finally {
      if (loadingEl) loadingEl.style.display = 'none';
    }
  }

  // Expose
  window._commDetect = _detect;
  window._commClear = _clearCommunities;

  async function render(container, params) {
    _container = container;

    // Route: #communities/<cid> → detail page
    if (params && params[0] !== undefined) {
      await _renderDetailPage(container, parseInt(params[0]));
      return;
    }

    // Default: community list
    container.innerHTML = `
      <div class="space-y-4">
        <!-- Controls -->
        <div class="card p-4">
          <div class="flex items-center gap-3 flex-wrap">
            <button id="comm-detect-btn" class="btn btn-primary btn-sm" onclick="window._commDetect()">
              <i data-lucide="scan" style="width:16px;height:16px;"></i>${t('communities.detect')}
            </button>
            <div class="flex items-center gap-2 text-sm">
              <label style="color:var(--text-muted);">${t('communities.algorithm')}:</label>
              <select id="comm-algorithm" class="input text-xs py-1 px-2" style="width:auto;">
                <option value="louvain">Louvain</option>
              </select>
            </div>
            <div class="flex items-center gap-2 text-sm">
              <label style="color:var(--text-muted);">Resolution:</label>
              <input type="number" id="comm-resolution" class="input text-xs py-1 px-2 w-20" value="1.0" min="0.1" max="10" step="0.1">
            </div>
            <div class="flex items-center gap-2 text-sm">
              <label style="color:var(--text-muted);">${t('communities.minSize')}:</label>
              <input type="number" id="comm-min-size" class="input text-xs py-1 px-2 w-16" value="3" min="1" max="100">
            </div>
            <div class="flex-1"></div>
            <button class="btn btn-ghost btn-sm" onclick="window._commClear()" style="color:var(--text-error);">
              <i data-lucide="x-circle" style="width:14px;height:14px;"></i>${t('communities.clear')}
            </button>
          </div>
        </div>

        <!-- Stats -->
        <div id="comm-stats"></div>

        <!-- Community cards -->
        <div id="comm-cards"></div>
      </div>
    `;

    if (window.lucide) lucide.createIcons();

    // Try to load existing communities
    try {
      await _loadCommunities(true);
    } catch { /* ignore if no communities yet */ }
  }

  function destroy() {
    if (_sentinelObserver) {
      _sentinelObserver.disconnect();
      _sentinelObserver = null;
    }
    _container = null;
    _communities = [];
    _total = 0;
    _offset = 0;
    _allLoaded = false;
    _detectResult = null;
    if (explorer) {
      explorer.destroy();
      explorer = null;
    }
    _commEntities = [];
    _commRelations = [];
    _commEntityMap = {};
    delete window._commDetect;
    delete window._commClear;
    delete window._commLoad;
  }

  return { render, destroy };
})());

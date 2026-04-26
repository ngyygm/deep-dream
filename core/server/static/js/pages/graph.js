/* ==========================================
   Graph Explorer Page
   ========================================== */

(function () {
  let explorer = null;
  let isFirstRender = true;

  // Focus state
  let focusAbsoluteId = null;
  let cachedAllNodes = [];       // default view: all nodes (seed + related)
  let cachedAllEdges = [];       // default view: all edges
  let cachedAllEntities = {};    // absolute_id -> entity data (from full entity list)

  // Advanced options panel state
  let advancedOptionsOpen = false;

  // Hop & version accumulation state
  let relationScope = 'accumulated';
  let currentHopLevel = 1;
  let cachedInheritedRelationIds = null; // Set of relation absolute_ids inherited in main view
  let cachedAllRawRelations = null; // original raw relations from API (before remapping)
  let cachedRemappedMainRelations = null; // remapped relations for main view

  // Community coloring state
  let communityColoringEnabled = false;
  let communityMap = null; // absolute_id -> community_id

  // Keyboard shortcut handler reference (for cleanup)
  let _graphKeyHandler = null;

  // Relation strength mode
  let relationStrengthEnabled = false;

  // Time travel snapshot state
  let snapshotMode = false;
  let snapshotTime = null;

  // Color palettes and shared graph builders are in GraphUtils (graph-utils.js)

  // ---- Build the page layout and kick off initial load ----

  async function render(container, params) {
    container.innerHTML = `
      <div class="page-enter" style="display:flex;gap:1rem;height:100%;overflow:hidden;">
        <!-- Left column: toolbar + canvas + timeline (aligns together) -->
        <div class="flex-1 flex flex-col" style="min-width:0;">
          <!-- Toolbar card — only canvas width -->
          <div class="card" style="flex-shrink:0;padding:0.5rem 0.75rem;margin-bottom:0.375rem;">
            <div style="display:flex;align-items:center;gap:0.5rem;flex-wrap:nowrap;">
              <!-- Left: graph ID + stats -->
              <div style="display:flex;align-items:center;gap:0.5rem;flex-shrink:0;">
                <i data-lucide="git-branch" style="width:14px;height:14px;color:var(--primary);"></i>
                <span class="badge badge-primary mono" id="graph-id-badge">-</span>
                <span id="focus-mode-badge" class="badge badge-warning" style="display:none;">
                  <i data-lucide="crosshair" style="width:12px;height:12px;margin-right:2px;"></i>
                  ${t('graph.focusMode')}
                </span>
                <span id="graph-stats" class="mono" style="font-size:0.75rem;color:var(--text-muted);"></span>
              </div>
              <!-- Center: search -->
              <div style="flex:1;display:flex;justify-content:center;">
                <div style="position:relative;width:100%;max-width:280px;">
                  <input type="text" class="input" id="graph-entity-search" placeholder="${t('graph.searchEntity') || 'Find entity...'}" style="width:100%;font-size:0.75rem;padding-left:24px;padding-top:2px;padding-bottom:2px;">
                  <i data-lucide="search" style="width:12px;height:12px;position:absolute;left:6px;top:50%;transform:translateY(-50%);color:var(--text-muted);pointer-events:none;"></i>
                  <div id="graph-search-dropdown" style="display:none;position:absolute;top:100%;left:0;width:100%;min-width:260px;max-height:200px;overflow-y:auto;background:var(--bg-surface);border:1px solid var(--border-color);border-radius:0.5rem;z-index:100;margin-top:4px;box-shadow:0 4px 12px rgba(0,0,0,0.3);"></div>
                </div>
              </div>
              <!-- Right: actions -->
              <div style="display:flex;align-items:center;gap:0.375rem;flex-shrink:0;">
                <button class="btn btn-primary btn-sm" id="load-graph-btn">
                  <i data-lucide="refresh-cw" style="width:14px;height:14px;"></i>
                  ${t('graph.loadGraph')}
                </button>
                <button class="btn btn-secondary btn-sm" id="exit-focus-btn" style="display:none;">
                  <i data-lucide="maximize-2" style="width:14px;height:14px;"></i>
                  ${t('graph.exitFocus')}
                </button>
                ${isNeo4j() ? `
                <button class="btn btn-ghost btn-sm" id="community-coloring-toggle" title="${t('graph.communityColoring') || 'Community coloring'}" style="color:var(--text-muted);">
                  <i data-lucide="layers" style="width:14px;height:14px;"></i>
                  <span id="community-coloring-status" style="font-size:0.65rem;margin-left:2px;opacity:0.5;">OFF</span>
                </button>
                ` : ''}
                <button class="btn btn-ghost btn-sm" id="toggle-graph-options-btn" style="color:var(--text-muted);">
                  <i data-lucide="sliders-horizontal" style="width:14px;height:14px;"></i>
                  <i data-lucide="chevron-down" style="width:12px;height:12px;transition:transform 0.2s;" id="graph-options-chevron"></i>
                </button>
                <button class="btn btn-ghost btn-sm" id="graph-shortcuts-btn" title="Keyboard shortcuts" style="color:var(--text-muted);">
                  <i data-lucide="keyboard" style="width:14px;height:14px;"></i>
                </button>
              </div>
            </div>

            <div id="graph-advanced-options" style="display:none;margin-top:6px;padding:8px 12px;background:var(--bg-input);border-radius:6px;border:1px solid var(--border-color);">
              <div style="display:flex;gap:0.75rem;align-items:center;flex-wrap:wrap;">
                <div style="display:flex;align-items:center;gap:4px;">
                  <label class="form-label" style="margin:0;font-size:0.75rem;">${t('graph.maxEntities')}</label>
                  <input type="number" class="input" id="entity-limit" min="0" max="500" value="" placeholder="Auto" step="10" style="width:80px;font-size:0.75rem;padding:2px 6px;">
                </div>
                <div style="display:flex;align-items:center;gap:4px;">
                  <label class="form-label" style="margin:0;font-size:0.75rem;">${t('graph.hopLevel')}</label>
                  <input type="number" class="input" id="hop-level" min="1" max="3" value="${currentHopLevel}" step="1" style="width:60px;font-size:0.75rem;padding:2px 6px;">
                </div>
                <button class="btn btn-secondary btn-sm" id="apply-hop-btn" title="${t('graph.apply')}">
                  <i data-lucide="check" style="width:14px;height:14px;"></i>
                  ${t('graph.apply')}
                </button>
                ${isNeo4j() ? `
                <label style="display:flex;align-items:center;gap:4px;font-size:0.75rem;cursor:pointer;color:var(--text-secondary);">
                  <input type="checkbox" id="community-coloring" style="cursor:pointer;">
                  ${t('graph.communityColoring')}
                </label>` : ''}
                <label style="display:flex;align-items:center;gap:4px;font-size:0.75rem;cursor:pointer;color:var(--text-secondary);">
                  <input type="checkbox" id="relation-strength-mode" style="cursor:pointer;">
                  ${t('graph.relationStrength')}
                </label>
                <div style="display:flex;align-items:center;gap:4px;">
                  <input type="datetime-local" id="graph-snapshot-time" class="input" style="width:180px;font-size:0.75rem;padding:2px 6px;">
                  <button class="btn btn-sm btn-primary" id="load-snapshot-btn" title="${t('timeTravel.title')}">
                    <i data-lucide="clock" style="width:14px;height:14px;"></i>
                  </button>
                  ${snapshotMode ? `
                  <button class="btn btn-sm btn-secondary" id="clear-snapshot-btn" title="${t('graph.exitFocus')}">
                    <i data-lucide="rotate-ccw" style="width:14px;height:14px;"></i>
                  </button>
                  ` : ''}
                </div>
              </div>
            </div>
          </div>

          <!-- Graph canvas -->
          <div class="relative" style="flex:1;min-height:0;">
              <div id="graph-canvas" style="width:100%;height:100%;"></div>
              <div id="graph-loading" class="absolute inset-0 flex items-center justify-center" style="background:var(--bg-input);border-radius:0.5rem;display:none;">
                ${spinnerHtml()}
              </div>
              <div id="snapshot-overlay" style="display:none;position:absolute;top:0.75rem;left:0.75rem;z-index:10;">
                <div class="snapshot-badge-overlay">
                  <i data-lucide="clock" style="width:12px;height:12px;"></i>
                  <span id="snapshot-overlay-time"></span>
                  <span id="snapshot-overlay-stats" style="font-size:0.6875rem;color:var(--text-muted);margin-left:0.25rem;"></span>
                  <button class="snapshot-return-btn" id="snapshot-return-live" title="${t('timeline.resetToLive')}">
                    <i data-lucide="zap" style="width:10px;height:10px;"></i>
                    ${t('timeline.live')}
                  </button>
                </div>
              </div>
              <div id="canvas-time-overlay" style="display:none;position:absolute;bottom:1.25rem;left:50%;transform:translateX(-50%);z-index:10;">
                <div class="canvas-time-badge">
                  <span class="canvas-time-label" id="canvas-time-text"></span>
                  <span class="canvas-time-progress" id="canvas-time-progress"></span>
                </div>
              </div>
            </div>
            <!-- Timeline slider bar — aligned with canvas width only -->
            <div id="timeline-container" style="flex-shrink:0;"></div>
          </div>

          <!-- Detail sidebar — full height, aligned to top -->
          <div id="detail-sidebar" style="width:30%;min-width:280px;max-width:420px;">
            <div class="card h-full flex flex-col">
              <div class="card-header">
                <span class="card-title">${t('common.detail')}</span>
              </div>
              <div id="detail-content" style="overflow-y:auto;flex:1;">
                ${emptyState(t('common.clickToView'), 'mouse-pointer-click')}
              </div>
            </div>
          </div>
      </div>
    `;

    if (window.lucide) lucide.createIcons();

    // Snapshot overlay return-to-live button
    const snapshotReturnBtn = document.getElementById('snapshot-return-live');
    if (snapshotReturnBtn) {
      snapshotReturnBtn.addEventListener('click', function () {
        resetToLive();
        renderTimeline(document.getElementById('timeline-container'));
      });
    }

    // ---- Keyboard shortcuts for time+version navigation ----
    _graphKeyHandler = function (e) {
      // Only handle when graph page is visible
      if (!container.offsetParent) return;

      const key = e.key;

      // Space — play/pause timeline
      if (key === ' ' && !e.target.matches('input, textarea, select')) {
        e.preventDefault();
        if (tlPlaybackTimer) {
          stopTimelinePlayback();
        } else {
          startTimelinePlayback();
        }
        renderTimeline(document.getElementById('timeline-container'));
        return;
      }

      // Left/Right arrows — version navigation (only when detail sidebar is showing entity)
      if ((key === 'ArrowLeft' || key === 'ArrowRight') && !e.target.matches('input, textarea, select')) {
        if (explorer && explorer.getState().focusAbsoluteId) {
          e.preventDefault();
          var st = explorer.getState();
          if (key === 'ArrowLeft' && st.currentVersionIdx > 0) {
            explorer.switchVersion(st.currentVersionIdx - 1);
          } else if (key === 'ArrowRight' && st.currentVersionIdx < st.currentVersions.length - 1) {
            explorer.switchVersion(st.currentVersionIdx + 1);
          }
        } else if (key === 'ArrowLeft') {
          // Step timeline backward
          e.preventDefault();
          stepTimeline(-1);
        } else if (key === 'ArrowRight') {
          // Step timeline forward
          e.preventDefault();
          stepTimeline(1);
        }
        return;
      }

      // Escape — close shortcuts overlay, then exit snapshot/focus mode
      if (key === 'Escape') {
        var shortcutsOverlay = document.getElementById('graph-shortcuts-overlay');
        if (shortcutsOverlay) {
          shortcutsOverlay.remove();
          return;
        }
        if (focusAbsoluteId) {
          focusAbsoluteId = null;
          explorer.exitFocus();
        } else if (snapshotMode) {
          resetToLive();
          renderTimeline(document.getElementById('timeline-container'));
        }
        return;
      }
    };
    document.addEventListener('keydown', _graphKeyHandler);

    // ---- Create GraphExplorer instance ----
    explorer = GraphExplorer.create({
      canvasId: 'graph-canvas',
      detailContentId: 'detail-content',
      loadingId: 'graph-loading',
      exitFocusBtnId: 'exit-focus-btn',
      focusBadgeId: 'focus-mode-badge',
      idPrefix: '',
      entityCache: cachedAllEntities,
      defaultHopLevel: currentHopLevel,
      maxPerHop: 20,
      communityColoringEnabled: communityColoringEnabled,
      communityMap: communityMap,
      relationStrengthEnabled: relationStrengthEnabled,
      familyIdToLatest: function () {
        var map = {};
        for (var absId in cachedAllEntities) {
          map[cachedAllEntities[absId].family_id] = absId;
        }
        return map;
      },
      onAfterFocus: function () {
        focusAbsoluteId = explorer.getState().focusAbsoluteId;
      },
      onRestoreDefaultView: function () {
        var hubLayout = computeHubLayout(cachedAllEdges);
        return {
          entities: cachedAllNodes,
          relations: cachedAllEdges,
          inheritedRelationIds: cachedInheritedRelationIds,
          hubLayout: hubLayout,
        };
      },
    });

    const loadBtn = document.getElementById('load-graph-btn');
    const graphBadge = document.getElementById('graph-id-badge');

    graphBadge.textContent = state.currentGraphId;

    loadBtn.addEventListener('click', () => loadGraph());

    // Graph entity search with dropdown
    const searchInput = document.getElementById('graph-entity-search');
    const searchDropdown = document.getElementById('graph-search-dropdown');
    if (searchInput && searchDropdown) {
      let _searchTimeout = null;
      searchInput.addEventListener('input', () => {
        clearTimeout(_searchTimeout);
        const query = (searchInput.value || '').trim().toLowerCase();
        if (!query || query.length < 1) { searchDropdown.style.display = 'none'; return; }
        _searchTimeout = setTimeout(() => {
          const matches = [];
          for (const absId in cachedAllEntities) {
            const e = cachedAllEntities[absId];
            if ((e.name || '').toLowerCase().includes(query)) {
              matches.push(e);
              if (matches.length >= 8) break;
            }
          }
          if (matches.length === 0) {
            searchDropdown.innerHTML = '<div style="padding:0.5rem 0.75rem;font-size:0.8rem;color:var(--text-muted);">No matches</div>';
          } else {
            // Build version counts index for search results
            var vcMap = explorer.getState().versionCounts || {};
            searchDropdown.innerHTML = matches.map(e => {
              var vc = vcMap[e.family_id] || 1;
              var versionTag = vc > 1
                ? ' <span style="display:inline-block;padding:0 0.25rem;background:color-mix(in srgb, #f59e0b 15%, transparent);color:#f59e0b;border-radius:3px;font-size:0.65rem;font-weight:500;font-family:var(--font-mono);">[v' + vc + ']</span>'
                : '';
              return '<div class="graph-search-item" data-abs-id="' + escapeHtml(e.absolute_id) + '" style="padding:0.5rem 0.75rem;cursor:pointer;font-size:0.8rem;border-bottom:1px solid var(--border-color);" onmouseover="this.style.background=\'var(--bg-input)\'" onmouseout="this.style.background=\'transparent\'">' +
              '<strong>' + escapeHtml(e.name || '-') + '</strong>' + versionTag +
              '<div style="font-size:0.7rem;color:var(--text-muted);margin-top:2px;">' + escapeHtml((e.content || '').slice(0, 60)) + '</div>' +
              '</div>';
            }).join('');
          }
          searchDropdown.style.display = 'block';
        }, 200);
      });
      searchDropdown.addEventListener('click', (ev) => {
        const item = ev.target.closest('.graph-search-item');
        if (!item) return;
        const absId = item.getAttribute('data-abs-id');
        if (absId && explorer) {
          explorer.focusOnEntity(absId).then(() => {
            focusAbsoluteId = explorer.getState().focusAbsoluteId;
          });
        }
        searchDropdown.style.display = 'none';
        searchInput.value = '';
      });
      document.addEventListener('click', (ev) => {
        if (!searchInput.contains(ev.target) && !searchDropdown.contains(ev.target)) {
          searchDropdown.style.display = 'none';
        }
      });
    }
    document.getElementById('exit-focus-btn').addEventListener('click', () => {
      focusAbsoluteId = null;
      explorer.exitFocus();
    });

    const applyBtn = document.getElementById('apply-hop-btn');
    if (applyBtn) {
      applyBtn.addEventListener('click', () => {
        const newHop = parseInt(document.getElementById('hop-level').value, 10) || 1;
        currentHopLevel = newHop;
        explorer.setState('defaultHopLevel', currentHopLevel);
        if (focusAbsoluteId) {
          explorer.focusOnEntity(focusAbsoluteId).then(() => {
            focusAbsoluteId = explorer.getState().focusAbsoluteId;
          });
        } else {
          applyHopToMainView();
        }
      });
    }

    // Advanced options toggle
    const toggleBtn = document.getElementById('toggle-graph-options-btn');
    const optionsPanel = document.getElementById('graph-advanced-options');
    const chevron = document.getElementById('graph-options-chevron');
    if (toggleBtn && optionsPanel) {
      toggleBtn.addEventListener('click', () => {
        const isOpen = optionsPanel.style.display !== 'none';
        optionsPanel.style.display = isOpen ? 'none' : 'block';
        if (chevron) chevron.style.transform = isOpen ? 'rotate(0deg)' : 'rotate(180deg)';
        advancedOptionsOpen = !isOpen;
      });
      if (advancedOptionsOpen) {
        optionsPanel.style.display = 'block';
        if (chevron) chevron.style.transform = 'rotate(180deg)';
      }
    }

    // Keyboard shortcuts overlay
    const shortcutsBtn = document.getElementById('graph-shortcuts-btn');
    if (shortcutsBtn) {
      var shortcutsOverlay = null;
      shortcutsBtn.addEventListener('click', function () {
        if (shortcutsOverlay) {
          shortcutsOverlay.remove();
          shortcutsOverlay = null;
          return;
        }
        shortcutsOverlay = document.createElement('div');
        shortcutsOverlay.id = 'graph-shortcuts-overlay';
        shortcutsOverlay.innerHTML =
          '<div style="position:fixed;inset:0;background:rgba(0,0,0,0.4);z-index:1000;display:flex;align-items:center;justify-content:center;">' +
            '<div style="background:var(--bg-surface);border:1px solid var(--border-color);border-radius:0.75rem;padding:1.25rem 1.5rem;max-width:360px;width:90%;box-shadow:0 8px 32px rgba(0,0,0,0.3);">' +
              '<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:0.75rem;">' +
                '<h3 style="font-size:0.9375rem;font-weight:600;">' + t('common.keyboardShortcuts') + '</h3>' +
                '<button id="graph-shortcuts-close" style="background:none;border:none;color:var(--text-muted);cursor:pointer;font-size:1.25rem;line-height:1;">&times;</button>' +
              '</div>' +
              '<div style="display:grid;grid-template-columns:auto 1fr;gap:0.375rem 1rem;font-size:0.8125rem;">' +
                '<kbd style="background:var(--bg-input);border:1px solid var(--border-color);border-radius:4px;padding:0.125rem 0.5rem;font-family:var(--font-mono);font-size:0.75rem;text-align:center;">Space</kbd>' +
                '<span style="color:var(--text-secondary);">' + t('graph.sc_playPause') + '</span>' +
                '<kbd style="background:var(--bg-input);border:1px solid var(--border-color);border-radius:4px;padding:0.125rem 0.5rem;font-family:var(--font-mono);font-size:0.75rem;text-align:center;">&larr; &rarr;</kbd>' +
                '<span style="color:var(--text-secondary);">' + t('graph.sc_stepTimeline') + '</span>' +
                '<kbd style="background:var(--bg-input);border:1px solid var(--border-color);border-radius:4px;padding:0.125rem 0.5rem;font-family:var(--font-mono);font-size:0.75rem;text-align:center;">Esc</kbd>' +
                '<span style="color:var(--text-secondary);">' + t('graph.sc_exitFocus') + '</span>' +
                '<kbd style="background:var(--bg-input);border:1px solid var(--border-color);border-radius:4px;padding:0.125rem 0.5rem;font-family:var(--font-mono);font-size:0.75rem;text-align:center;">Click</kbd>' +
                '<span style="color:var(--text-secondary);">' + t('graph.sc_entityDetail') + '</span>' +
                '<kbd style="background:var(--bg-input);border:1px solid var(--border-color);border-radius:4px;padding:0.125rem 0.5rem;font-family:var(--font-mono);font-size:0.75rem;text-align:center;">Hover</kbd>' +
                '<span style="color:var(--text-secondary);">' + t('graph.sc_hoverPreview') + '</span>' +
              '</div>' +
              '<div style="margin-top:0.75rem;padding-top:0.5rem;border-top:1px solid var(--border-color);font-size:0.75rem;color:var(--text-muted);">' +
                '<span style="color:#f59e0b;">&#9679;</span> ' + t('graph.sc_amberGlow') + ' &nbsp; ' +
                '<span style="color:var(--text-muted);">' + t('graph.sc_versionBadge') + '</span>' +
              '</div>' +
            '</div>' +
          '</div>';
        document.body.appendChild(shortcutsOverlay);
        // Close handlers
        var closeBtn = document.getElementById('graph-shortcuts-close');
        if (closeBtn) closeBtn.addEventListener('click', function () {
          if (shortcutsOverlay) { shortcutsOverlay.remove(); shortcutsOverlay = null; }
        });
        shortcutsOverlay.addEventListener('click', function (ev) {
          if (ev.target === shortcutsOverlay || ev.target === shortcutsOverlay.firstElementChild.parentElement) {
            shortcutsOverlay.remove();
            shortcutsOverlay = null;
          }
        });
      });
    }

    // Community coloring toggle
    const commCheckbox = document.getElementById('community-coloring');
    if (commCheckbox) {
      commCheckbox.addEventListener('change', async () => {
        communityColoringEnabled = commCheckbox.checked;
        explorer.setState('communityColoringEnabled', communityColoringEnabled);
        if (communityColoringEnabled && !communityMap) {
          await loadCommunityMap();
          explorer.setState('communityMap', communityMap);
        }
        if (cachedAllEntities && Object.keys(cachedAllEntities).length > 0) {
          if (focusAbsoluteId) {
            explorer.focusOnEntity(focusAbsoluteId).then(() => {
              focusAbsoluteId = explorer.getState().focusAbsoluteId;
            });
          } else {
            rebuildMainView();
          }
        }
        updateCommunityColoringUI();
      });
    }

    // Community coloring toggle button (toolbar button)
    const commToggleBtn = document.getElementById('community-coloring-toggle');
    if (commToggleBtn) {
      commToggleBtn.addEventListener('click', async () => {
        communityColoringEnabled = !communityColoringEnabled;
        explorer.setState('communityColoringEnabled', communityColoringEnabled);
        if (communityColoringEnabled && !communityMap) {
          await loadCommunityMap();
          explorer.setState('communityMap', communityMap);
        }
        if (cachedAllEntities && Object.keys(cachedAllEntities).length > 0) {
          if (focusAbsoluteId) {
            explorer.focusOnEntity(focusAbsoluteId).then(() => {
              focusAbsoluteId = explorer.getState().focusAbsoluteId;
            });
          } else {
            rebuildMainView();
          }
        }
        updateCommunityColoringUI();
      });
    }

    // Initialize community coloring UI state
    updateCommunityColoringUI();

    // Relation strength toggle
    const strengthCheckbox = document.getElementById('relation-strength-mode');
    if (strengthCheckbox) {
      strengthCheckbox.checked = relationStrengthEnabled;
      strengthCheckbox.addEventListener('change', () => {
        relationStrengthEnabled = strengthCheckbox.checked;
        explorer.setState('relationStrengthEnabled', relationStrengthEnabled);
        if (cachedAllEntities && Object.keys(cachedAllEntities).length > 0) {
          if (focusAbsoluteId) {
            explorer.focusOnEntity(focusAbsoluteId).then(() => {
              focusAbsoluteId = explorer.getState().focusAbsoluteId;
            });
          } else {
            rebuildMainView();
          }
        }
      });
    }

    // Time travel snapshot button
    const snapshotBtn = document.getElementById('load-snapshot-btn');
    if (snapshotBtn) {
      snapshotBtn.addEventListener('click', () => loadGraphSnapshot());
    }

    // Clear snapshot button
    const clearSnapshotBtn = document.getElementById('clear-snapshot-btn');
    if (clearSnapshotBtn) {
      clearSnapshotBtn.addEventListener('click', () => clearSnapshot());
    }

    if (isFirstRender) {
      isFirstRender = false;
      await loadGraph();
    } else if (cachedAllNodes.length > 0) {
      const hubLayout = computeHubLayout(cachedAllEdges);
      explorer.buildGraph(cachedAllNodes, cachedAllEdges, null, null, cachedInheritedRelationIds, undefined, hubLayout);
      const exitBtn = document.getElementById('exit-focus-btn');
      if (exitBtn) exitBtn.style.display = focusAbsoluteId ? '' : 'none';
      const focusBadge = document.getElementById('focus-mode-badge');
      if (focusBadge) focusBadge.style.display = focusAbsoluteId ? '' : 'none';
    }
  }

  // ---- Expand N-hop neighbors from seed entities (client-side BFS) ----

  function expandNHops(seedAbsIds, allRelations, hopLevel) {
    const result = new Set(seedAbsIds);
    let frontier = new Set(seedAbsIds);
    for (let h = 1; h <= hopLevel; h++) {
      const nextFrontier = new Set();
      for (const r of allRelations) {
        if (frontier.has(r.entity1_absolute_id) && !result.has(r.entity2_absolute_id)) {
          nextFrontier.add(r.entity2_absolute_id);
        }
        if (frontier.has(r.entity2_absolute_id) && !result.has(r.entity1_absolute_id)) {
          nextFrontier.add(r.entity1_absolute_id);
        }
      }
      for (const id of nextFrontier) result.add(id);
      frontier = nextFrontier;
    }
    return result;
  }

  // ---- Compute top-3 hub entities and their 1-hop neighbors ----

  function computeHubLayout(visibleRelations) {
    if (!visibleRelations || visibleRelations.length === 0) return null;

    const relCounts = {};
    for (const r of visibleRelations) {
      relCounts[r.entity1_absolute_id] = (relCounts[r.entity1_absolute_id] || 0) + 1;
      relCounts[r.entity2_absolute_id] = (relCounts[r.entity2_absolute_id] || 0) + 1;
    }
    const sorted = Object.entries(relCounts).sort((a, b) => b[1] - a[1]);
    const hubIds = sorted.slice(0, 3).map(e => e[0]);
    if (hubIds.length === 0) return null;

    // hubMap: absoluteId -> hubIndex (0/1/2)
    const hubMap = {};
    for (let i = 0; i < hubIds.length; i++) {
      hubMap[hubIds[i]] = i;
    }

    // 1-hop neighbors inherit hub color
    const hubNeighborIds = new Set();
    for (const r of visibleRelations) {
      const e1 = r.entity1_absolute_id, e2 = r.entity2_absolute_id;
      const h1 = hubMap[e1], h2 = hubMap[e2];
      if (h1 !== undefined && h2 === undefined) {
        hubMap[e2] = h1;
        hubNeighborIds.add(e2);
      } else if (h2 !== undefined && h1 === undefined) {
        hubMap[e1] = h2;
        hubNeighborIds.add(e1);
      }
    }

    return { hubMap, hubIds, hubNeighborIds };
  }

  // ---- Resolve unknown relation endpoints and remap to current entity versions ----

  async function resolveAndRemapRelations(entities, relations, graphId) {
    const currentAbsIds = new Set(entities.map(e => e.absolute_id));
    const currentEntityIds = {};
    for (const e of entities) {
      currentEntityIds[e.family_id] = e.absolute_id;
    }

    // Collect unknown endpoint absolute_ids
    const unknownAbsIds = new Set();
    for (const r of relations) {
      if (!currentAbsIds.has(r.entity1_absolute_id)) unknownAbsIds.add(r.entity1_absolute_id);
      if (!currentAbsIds.has(r.entity2_absolute_id)) unknownAbsIds.add(r.entity2_absolute_id);
    }

    // Batch resolve unknown endpoints in concurrent batches of 50
    const resolved = {};
    const toResolve = [...unknownAbsIds];
    const batchSize = 50;
    for (let i = 0; i < toResolve.length; i += batchSize) {
      const batch = toResolve.slice(i, i + batchSize);
      const promises = batch.map(async (absId) => {
        try {
          const res = await state.api.entityByAbsoluteId(absId, graphId);
          if (res.data) resolved[absId] = res.data;
        } catch (_) {}
      });
      await Promise.all(promises);
    }

    // Remap relations: replace old absolute_ids with current versions
    const remapped = [];
    const inheritedRelationIds = new Set();

    for (const r of relations) {
      let e1AbsId = r.entity1_absolute_id;
      let e2AbsId = r.entity2_absolute_id;
      let remapped1 = false, remapped2 = false;

      if (!currentAbsIds.has(e1AbsId)) {
        const oldEntity = resolved[e1AbsId];
        if (oldEntity && currentEntityIds[oldEntity.family_id]) {
          e1AbsId = currentEntityIds[oldEntity.family_id];
          remapped1 = true;
        }
      }
      if (!currentAbsIds.has(e2AbsId)) {
        const oldEntity = resolved[e2AbsId];
        if (oldEntity && currentEntityIds[oldEntity.family_id]) {
          e2AbsId = currentEntityIds[oldEntity.family_id];
          remapped2 = true;
        }
      }

      if (remapped1 || remapped2) {
        inheritedRelationIds.add(r.absolute_id);
        remapped.push({ ...r, entity1_absolute_id: e1AbsId, entity2_absolute_id: e2AbsId });
      } else {
        remapped.push(r);
      }
    }

    return { relations: remapped, inheritedRelationIds };
  }

  // ---- Apply hop level to main view without full reload ----

  async function applyHopToMainView() {
    if (cachedAllEntities && Object.keys(cachedAllEntities).length > 0) {
      if (!cachedAllRawRelations) {
        await loadGraph();
        return;
      }
      rebuildMainView();
    } else {
      await loadGraph();
    }
  }

  function rebuildMainView() {
    const allKnownEntities = Object.values(cachedAllEntities);
    const hopLevel = currentHopLevel;

    // Dynamic seed limit (same logic as loadGraph)
    var entityLimit = parseInt(document.getElementById('entity-limit').value, 10);
    if (!entityLimit || entityLimit <= 0) {
      if (allKnownEntities.length <= 200) {
        entityLimit = Math.min(allKnownEntities.length, 200);
      } else if (allKnownEntities.length <= 1000) {
        entityLimit = 200;
      } else {
        entityLimit = 300;
      }
    }
    const seedEntities = allKnownEntities.slice(0, entityLimit);
    const seedAbsIds = new Set(seedEntities.map(e => e.absolute_id));

    const allRels = cachedRemappedMainRelations || cachedAllRawRelations;
    const visibleAbsIds = expandNHops(seedAbsIds, allRels, hopLevel);

    const visibleRelations = allRels.filter(r =>
      visibleAbsIds.has(r.entity1_absolute_id) && visibleAbsIds.has(r.entity2_absolute_id)
    );

    const allVisible = [...visibleAbsIds]
      .map(aid => cachedAllEntities[aid])
      .filter(Boolean);

    cachedAllNodes = allVisible;
    cachedAllEdges = visibleRelations;

    const hubLayout = computeHubLayout(visibleRelations);
    explorer.buildGraph(allVisible, visibleRelations, null, null, cachedInheritedRelationIds, undefined, hubLayout);

    const statsEl = document.getElementById('graph-stats');
    if (statsEl) {
      statsEl.textContent = t('graph.loaded', { entities: allVisible.length, relations: visibleRelations.length });
    }
  }

  // ---- Fetch entities, their relations, and build the graph ----

  async function loadGraph() {
    const graphId = state.currentGraphId;
    const hopLevel = parseInt(document.getElementById('hop-level').value, 10) || 1;
    currentHopLevel = hopLevel;
    if (explorer) explorer.setState('defaultHopLevel', currentHopLevel);
    const loadingEl = document.getElementById('graph-loading');
    const statsEl = document.getElementById('graph-stats');

    if (loadingEl) loadingEl.style.display = 'flex';
    if (statsEl) statsEl.textContent = t('common.loading');

    try {
      const [entityRes, relationRes] = await Promise.all([
        state.api.listEntities(graphId, 5000),
        state.api.listRelations(graphId, 2000),
      ]);

      const allKnownEntities = entityRes.data?.entities || entityRes.data || [];
      const allRelations = relationRes.data?.relations || relationRes.data || [];

      if (allKnownEntities.length === 0) {
        if (statsEl) statsEl.textContent = t('graph.noEntities');
        if (loadingEl) loadingEl.style.display = 'none';
        return;
      }

      cachedAllRawRelations = allRelations;

      // Prioritize connected entities as seeds (entities appearing in relations)
      const relationAbsIds = new Set();
      for (const r of allRelations) {
        relationAbsIds.add(r.entity1_absolute_id);
        relationAbsIds.add(r.entity2_absolute_id);
      }
      const connectedEntities = allKnownEntities.filter(e => relationAbsIds.has(e.absolute_id));
      const isolatedEntities = allKnownEntities.filter(e => !relationAbsIds.has(e.absolute_id));

      // Dynamic seed limit: auto-scale based on graph size
      // Small graphs (<200 entities): show all connected + some isolated
      // Medium graphs (200-1000): show up to 200 connected
      // Large graphs (>1000): show up to 300 connected
      var entityLimit = parseInt(document.getElementById('entity-limit').value, 10);
      if (!entityLimit || entityLimit <= 0) {
        if (connectedEntities.length <= 200) {
          entityLimit = Math.min(allKnownEntities.length, 200);
        } else if (connectedEntities.length <= 1000) {
          entityLimit = 200;
        } else {
          entityLimit = 300;
        }
      }
      const seedEntities = [...connectedEntities, ...isolatedEntities].slice(0, entityLimit);

      const entityByAbs = {};
      for (const e of allKnownEntities) {
        entityByAbs[e.absolute_id] = e;
      }
      cachedAllEntities = entityByAbs;
      if (explorer) explorer.setEntityCache(cachedAllEntities);

      const { relations: remappedRelations, inheritedRelationIds } = await resolveAndRemapRelations(
        allKnownEntities, allRelations, graphId
      );
      cachedInheritedRelationIds = inheritedRelationIds;
      cachedRemappedMainRelations = remappedRelations;

      const seedAbsIds = new Set(seedEntities.map(e => e.absolute_id));

      const visibleAbsIds = expandNHops(seedAbsIds, remappedRelations, hopLevel);

      const visibleRelations = remappedRelations.filter(r =>
        visibleAbsIds.has(r.entity1_absolute_id) && visibleAbsIds.has(r.entity2_absolute_id)
      );

      // Remove isolated entities: keep only entities that have at least one visible relation
      const connectedAbsIds = new Set();
      for (const r of visibleRelations) {
        connectedAbsIds.add(r.entity1_absolute_id);
        connectedAbsIds.add(r.entity2_absolute_id);
      }

      const allVisible = [...connectedAbsIds]
        .map(aid => entityByAbs[aid])
        .filter(Boolean);

      const allEntityIds = [...new Set(allVisible.map(e => e.family_id))];
      try {
        const vcRes = await state.api.entityVersionCounts(allEntityIds, graphId);
        const vc = vcRes.data || {};
        if (explorer) explorer.setVersionCounts(vc);
      } catch (_) {}

      cachedAllNodes = allVisible;
      cachedAllEdges = visibleRelations;

      const hubLayout = computeHubLayout(visibleRelations);
      explorer.buildGraph(allVisible, visibleRelations, null, null, cachedInheritedRelationIds, undefined, hubLayout);
      explorer.setMainViewCache(visibleRelations, allVisible, cachedInheritedRelationIds);

      focusAbsoluteId = null;
      const exitBtn = document.getElementById('exit-focus-btn');
      if (exitBtn) exitBtn.style.display = 'none';
      const focusBadge = document.getElementById('focus-mode-badge');
      if (focusBadge) focusBadge.style.display = 'none';

      if (statsEl) {
        statsEl.textContent = t('graph.loaded', { entities: allVisible.length, relations: visibleRelations.length });
      }
      showToast(t('graph.loaded', { entities: allVisible.length, relations: visibleRelations.length }), 'success');

      // Initialize timeline after graph loads
      initTimeline();
    } catch (err) {
      console.error('Failed to load graph:', err);
      showToast(t('graph.loadFailed') + ': ' + err.message, 'error');
      if (statsEl) statsEl.textContent = t('common.error');
    } finally {
      if (loadingEl) loadingEl.style.display = 'none';
    }
  }

  // ---- Load community map from API ----
  async function loadCommunityMap() {
    if (!isNeo4j()) return;
    try {
      const res = await state.api.listCommunities(state.currentGraphId, 1, 200);
      const communities = res.data?.communities || [];
      communityMap = {};
      for (const c of communities) {
        for (const m of (c.members || [])) {
          communityMap[m.uuid] = c.community_id;
        }
      }
    } catch (err) {
      console.warn('Failed to load community map:', err);
      communityMap = null;
    }
  }

  // ---- Update community coloring UI state ----
  function updateCommunityColoringUI() {
    const commCheckbox = document.getElementById('community-coloring');
    const commStatus = document.getElementById('community-coloring-status');
    const commToggleBtn = document.getElementById('community-coloring-toggle');

    if (commCheckbox) {
      commCheckbox.checked = communityColoringEnabled;
    }
    if (commStatus) {
      commStatus.textContent = communityColoringEnabled ? 'ON' : 'OFF';
      commStatus.style.opacity = communityColoringEnabled ? '1' : '0.5';
      commStatus.style.color = communityColoringEnabled ? 'var(--primary)' : 'inherit';
    }
    if (commToggleBtn) {
      commToggleBtn.style.color = communityColoringEnabled ? 'var(--primary)' : 'var(--text-muted)';
    }
  }

  // ---- Time Travel: load graph snapshot ----

  async function loadGraphSnapshot() {
    const timeInput = document.getElementById('graph-snapshot-time');
    const time = timeInput ? timeInput.value : null;
    if (!time) {
      showToast(t('timeTravel.time') + ' ' + t('common.required'), 'warning');
      return;
    }

    const loadingEl = document.getElementById('graph-loading');
    const statsEl = document.getElementById('graph-stats');
    if (loadingEl) loadingEl.style.display = 'flex';
    if (statsEl) statsEl.textContent = t('common.loading');

    try {
      const timeParam = time + ':00';
      const res = await state.api.getSnapshot(timeParam, state.currentGraphId);

      renderSnapshotData(res.data || res);
      snapshotMode = true;
      snapshotTime = time;
      showToast(t('timeTravel.snapshot') + ': ' + time, 'success');
    } catch (err) {
      console.error('Snapshot load failed:', err);
      showToast(t('graph.loadFailed') + ': ' + err.message, 'error');
    } finally {
      if (loadingEl) loadingEl.style.display = 'none';
    }
  }

  // ---- Snapshot transition animation ----

  function playSnapshotTransition() {
    const canvasParent = document.getElementById('graph-canvas');
    if (!canvasParent) return;

    // Remove any existing overlay
    const oldOverlay = canvasParent.querySelector('.snapshot-transition-overlay');
    if (oldOverlay) oldOverlay.remove();

    const overlay = document.createElement('div');
    overlay.className = 'snapshot-transition-overlay';

    // Flash effect
    const flash = document.createElement('div');
    flash.className = 'snapshot-flash';
    overlay.appendChild(flash);

    // Scanline (delayed slightly for layered effect)
    setTimeout(function () {
      const scanline = document.createElement('div');
      scanline.className = 'snapshot-scanline';
      overlay.appendChild(scanline);
    }, 80);

    // Ripple (delayed)
    setTimeout(function () {
      const ripple = document.createElement('div');
      ripple.className = 'snapshot-ripple';
      overlay.appendChild(ripple);
    }, 150);

    canvasParent.appendChild(overlay);

    // Clean up after animation
    setTimeout(function () { if (overlay.parentNode) overlay.remove(); }, 1500);
  }

  function renderSnapshotData(data, isTimelineDriven) {
    const snapshotEntities = data.entities || [];
    const snapshotRelations = data.relations || [];

    // Filter out isolated entities — only keep entities that have at least one relation
    const connectedAbsIds = new Set();
    for (const r of snapshotRelations) {
      if (r.entity1_absolute_id) connectedAbsIds.add(r.entity1_absolute_id);
      if (r.entity2_absolute_id) connectedAbsIds.add(r.entity2_absolute_id);
    }
    const filteredEntities = snapshotEntities.filter(e => connectedAbsIds.has(e.absolute_id));
    const filteredRelations = snapshotRelations.filter(r =>
      connectedAbsIds.has(r.entity1_absolute_id) && connectedAbsIds.has(r.entity2_absolute_id)
    );

    if (filteredEntities.length === 0) {
      const statsEl = document.getElementById('graph-stats');
      if (statsEl) statsEl.textContent = t('graph.noEntities');
      // Only show toast for explicit snapshot button clicks, not timeline scrubbing/playback
      // During timeline playback, early time points legitimately have no connected entities
      if (!isTimelineDriven) {
        showToast(t('common.noData'), 'warning');
      }
      return;
    }

    cachedAllNodes = filteredEntities;
    cachedAllEdges = filteredRelations;
    cachedInheritedRelationIds = null;

    // Play time warp transition animation
    playSnapshotTransition();

    const snapshotEntityIds = [...new Set(filteredEntities.map(e => e.family_id))];
    var vc = {};
    for (const eid of snapshotEntityIds) {
      vc[eid] = filteredEntities.filter(e => e.family_id === eid).length;
    }
    if (explorer) explorer.setVersionCounts(vc);

    explorer.buildGraph(filteredEntities, filteredRelations, null, null, null);

    const statsEl = document.getElementById('graph-stats');
    if (statsEl) {
      statsEl.textContent = t('graph.loaded', {
        entities: filteredEntities.length,
        relations: filteredRelations.length,
      }) + ' [' + t('timeTravel.snapshot') + ']';
    }
    updateSnapshotOverlay(snapshotTime ? new Date(snapshotTime).toLocaleString() : t('timeline.snapshot'), filteredEntities.length, filteredRelations.length);

    focusAbsoluteId = null;
    const exitBtn = document.getElementById('exit-focus-btn');
    if (exitBtn) exitBtn.style.display = 'none';
    const focusBadge = document.getElementById('focus-mode-badge');
    if (focusBadge) focusBadge.style.display = 'none';
  }

  function clearSnapshot() {
    snapshotMode = false;
    snapshotTime = null;
    const timeInput = document.getElementById('graph-snapshot-time');
    if (timeInput) timeInput.value = '';
    updateSnapshotOverlay(null);
    loadGraph();
  }

  // ---- Snapshot overlay badge ----

  function updateSnapshotOverlay(timeStr, entityCount, relationCount) {
    const overlay = document.getElementById('snapshot-overlay');
    const timeEl = document.getElementById('snapshot-overlay-time');
    const statsEl = document.getElementById('snapshot-overlay-stats');
    if (!overlay) return;
    if (timeStr) {
      overlay.style.display = '';
      if (timeEl) timeEl.textContent = timeStr;
      if (statsEl && entityCount !== undefined) {
        statsEl.textContent = 'E:' + entityCount + ' R:' + relationCount;
      }
      if (window.lucide) lucide.createIcons({ nodes: [overlay] });
    } else {
      overlay.style.display = 'none';
    }
  }

  // ---- Cleanup on page leave ----

  function destroy() {
    if (explorer) {
      explorer.destroy();
      explorer = null;
    }
    stopTimelinePlayback();
    // Remove keyboard handler
    if (_graphKeyHandler) {
      document.removeEventListener('keydown', _graphKeyHandler);
    }
    focusAbsoluteId = null;
    cachedAllNodes = [];
    cachedAllEdges = [];
    cachedAllEntities = {};
    cachedInheritedRelationIds = null;
    cachedAllRawRelations = null;
    cachedRemappedMainRelations = null;
    relationStrengthEnabled = false;
    snapshotMode = false;
    snapshotTime = null;
    isFirstRender = true;
  }

  // ==================================================================
  // Timeline Slider — Deep-Dream Time + Version Visualization
  // ==================================================================

  // ---- Canvas time indicator overlay ----

  function showCanvasTimeOverlay(timeStr, progressStr) {
    const overlay = document.getElementById('canvas-time-overlay');
    const textEl = document.getElementById('canvas-time-text');
    const progressEl = document.getElementById('canvas-time-progress');
    if (!overlay) return;
    if (textEl) textEl.textContent = timeStr || '';
    if (progressEl) progressEl.textContent = progressStr || '';
    overlay.style.display = '';
  }

  function hideCanvasTimeOverlay() {
    const overlay = document.getElementById('canvas-time-overlay');
    if (overlay) overlay.style.display = 'none';
  }

  function formatTimeShort(ts) {
    if (!ts) return '';
    const d = new Date(ts);
    return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' }) + ' ' +
      d.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' });
  }

  // Timeline state
  let tlEpisodes = [];      // sorted episode list [{time, label, type, data}]
  let tlMinTime = 0;
  let tlMaxTime = 0;
  let tlCurrentPos = 1.0;   // 0..1 where 1 = live (rightmost)
  let tlIsLive = true;
  let tlIsDragging = false;
  let tlPlaybackTimer = null;
  let tlPlaybackIndex = -1;
  let tlPlaybackSpeed = 1;  // seconds between steps
  let tlPlaybackMode = 'grow';  // 'grow' = entity-by-entity animation, 'snapshot' = episode snapshots

  // Grow-mode state
  let growSortedNodes = [];   // nodes sorted by processed_time
  let growSortedEdges = [];   // edges sorted by processed_time
  let growAnimTimer = null;   // animation interval
  let growIndex = 0;          // current index in grow animation
  let growEdgeIndex = 0;      // current edge frontier index (for time-ordered edge addition)
  let growActive = false;
  let growVisibleNodes = [];  // accumulated visible nodes during animation
  let growVisibleEdges = [];  // accumulated visible edges during animation

  // Timeline drag animation state
  let tlDragSortedNodes = [];   // nodes sorted by processed_time (for drag filtering)
  let tlDragSortedEdges = [];   // edges sorted by max endpoint processed_time
  let tlDragVisibleNodeIds = new Set();  // currently visible node absolute_ids during drag
  let tlDragVisibleEdgeIds = new Set();  // currently visible edge absolute_ids during drag
  let tlDragLastTime = 0;       // last target time during drag (to skip redundant updates)
  let tlDragRafPending = false; // requestAnimationFrame throttle
  let tlDragHubLayout = null;   // cached hub layout for drag session

  async function initTimeline() {
    const container = document.getElementById('timeline-container');
    if (!container) return;

    // Fetch episodes for event markers
    try {
      const res = await state.api.listEpisodes(state.currentGraphId, 100, 0);
      const episodes = res.data?.episodes || [];
      tlEpisodes = episodes
        .filter(e => e.created_at || e.processed_time)
        .map((e, i) => ({
          time: new Date(e.created_at || e.processed_time).getTime(),
          label: e.source_document || t('timeline.episode', { index: i + 1 }),
          type: (e.source_document || '').includes('dream') ? 'dream' : 'remember',
          data: e,
          entityCount: (e.entity_count || (e.data && e.data.entity_count)),
          relationCount: (e.relation_count || (e.data && e.data.relation_count)),
        }))
        .sort((a, b) => a.time - b.time);
    } catch (_) {
      tlEpisodes = [];
    }

    // Compute time range from entities
    if (cachedAllNodes.length > 0) {
      const times = cachedAllNodes
        .map(e => e.processed_time ? new Date(e.processed_time).getTime() : 0)
        .filter(t => t > 0);
      if (times.length > 0) {
        tlMinTime = Math.min(...times);
        tlMaxTime = Math.max(...times);
      }
    }
    if (tlMinTime === tlMaxTime) {
      tlMinTime = tlMaxTime - 86400000; // fallback: 1 day range
    }

    tlIsLive = !snapshotMode;
    tlCurrentPos = 1.0;

    renderTimeline(container);

    // Auto-load recent changes if episodes exist
    if (tlEpisodes.length > 0) {
      loadTimelineChanges();
    }
  }

  function renderTimeline(container) {
    const timeRange = tlMaxTime - tlMinTime;
    const posPercent = (tlCurrentPos * 100).toFixed(2);

    // Build version density bars — shows entity activity distribution
    let densityHtml = '';
    if (cachedAllNodes.length > 0 && timeRange > 0) {
      const BINS = 50;
      const bins = new Array(BINS).fill(0);
      for (const e of cachedAllNodes) {
        const t = e.processed_time ? new Date(e.processed_time).getTime() : 0;
        if (t >= tlMinTime && t <= tlMaxTime) {
          const idx = Math.min(Math.floor((t - tlMinTime) / timeRange * BINS), BINS - 1);
          bins[idx]++;
        }
      }
      const maxBin = Math.max(...bins, 1);
      const barWidth = (100 / BINS).toFixed(3);
      densityHtml = '<div class="timeline-density-bar">';
      for (let b = 0; b < BINS; b++) {
        const h = Math.max(bins[b] / maxBin * 100, 0).toFixed(1);
        const opacity = Math.max(bins[b] / maxBin, 0.05).toFixed(2);
        const binPct = ((b + 0.5) / BINS * 100).toFixed(2);
        densityHtml += '<div class="timeline-density-col" style="width:' + barWidth + '%;height:' + h + '%;opacity:' + opacity + ';" data-bin-pct="' + binPct + '" title="' + bins[b] + ' entities"></div>';
      }
      densityHtml += '</div>';
    }

    const markerHtml = tlEpisodes.map((ep, i) => {
      if (timeRange === 0) return '';
      const pct = ((ep.time - tlMinTime) / timeRange * 100).toFixed(2);
      const statsHtml = (ep.entityCount || ep.relationCount)
        ? '<br><span style="font-size:0.625rem;">E:' + (ep.entityCount || '?') + ' R:' + (ep.relationCount || '?') + '</span>'
        : '';
      const typeIcon = ep.type === 'dream'
        ? '<span style="color:var(--warning);font-size:0.6875rem;">&#9728;</span> '
        : '<span style="color:var(--primary);font-size:0.6875rem;">&#9679;</span> ';
      return '<div class="timeline-marker type-' + ep.type + '" style="left:' + pct + '%;" data-ep-idx="' + i + '">' +
        '<div class="timeline-tooltip">' + typeIcon + escapeHtml(ep.label) + '<br><span style="color:var(--text-muted);">' + formatDate(new Date(ep.time).toISOString()) + '</span>' + statsHtml + '</div>' +
      '</div>';
    }).join('');

    const minLabel = tlMinTime ? new Date(tlMinTime).toLocaleDateString() : '-';
    const maxLabel = tlMaxTime ? new Date(tlMaxTime).toLocaleDateString() : '-';
    const currentTime = tlCurrentPos < 1
      ? new Date(tlMinTime + tlCurrentPos * timeRange)
      : null;
    const currentTimeStr = currentTime
      ? currentTime.toLocaleString()
      : '';

    container.innerHTML =
      '<div class="timeline-bar" style="display:flex;align-items:center;gap:0.5rem;padding:0.375rem 0.75rem;">' +
        // Left: live/snapshot indicator + controls
        '<div style="display:flex;align-items:center;gap:0.25rem;flex-shrink:0;">' +
          (tlIsLive
            ? '<span class="timeline-live-dot"></span>'
            : '<span class="timeline-live-dot snapshot"></span>') +
          '<button class="timeline-btn" id="tl-step-back" title="' + t('timeline.stepBack') + '" ' + (tlIsLive ? 'disabled' : '') + '>' +
            '<i data-lucide="skip-back" style="width:11px;height:11px;"></i>' +
          '</button>' +
          '<button class="timeline-btn' + (tlPlaybackTimer ? ' active' : '') + '" id="tl-play-btn" title="' + (tlPlaybackTimer ? t('timeline.pause') : t('timeline.play')) + '">' +
            '<i data-lucide="' + (tlPlaybackTimer ? 'pause' : 'play') + '" style="width:11px;height:11px;"></i>' +
          '</button>' +
          '<button class="timeline-btn" id="tl-step-forward" title="' + t('timeline.stepForward') + '">' +
            '<i data-lucide="skip-forward" style="width:11px;height:11px;"></i>' +
          '</button>' +
          '<button class="timeline-btn" id="tl-reset-live" title="' + t('timeline.resetToLive') + '" ' + (tlIsLive ? 'disabled' : '') + '>' +
            '<i data-lucide="zap" style="width:11px;height:11px;"></i>' +
          '</button>' +
          '<button class="timeline-btn tl-mode-btn' + (tlPlaybackMode === 'grow' ? ' active' : '') + '" id="tl-mode-grow" title="Grow">' +
            '<i data-lucide="sprout" style="width:11px;height:11px;"></i>' +
          '</button>' +
          '<button class="timeline-btn tl-mode-btn' + (tlPlaybackMode === 'snapshot' ? ' active' : '') + '" id="tl-mode-snapshot" title="Snapshot">' +
            '<i data-lucide="camera" style="width:11px;height:11px;"></i>' +
          '</button>' +
        '</div>' +
        // Center: density + track + thumb (flex:1 fills remaining space)
        '<div style="flex:1;min-width:0;position:relative;">' +
          densityHtml +
          '<div class="timeline-track" id="tl-track" style="position:relative;">' +
            '<div class="timeline-track-fill" style="width:' + posPercent + '%;"></div>' +
            markerHtml +
            '<div class="timeline-thumb' + (tlIsDragging ? ' dragging' : '') + '" id="tl-thumb" style="left:' + posPercent + '%;"></div>' +
          '</div>' +
        '</div>' +
        // Right: time label
        '<div style="flex-shrink:0;text-align:right;min-width:120px;">' +
          '<span class="mono" style="font-size:0.6875rem;color:var(--text-secondary);" id="tl-current-time">' +
            (currentTimeStr || maxLabel) +
          '</span>' +
        '</div>' +
      '</div>';

    if (window.lucide) lucide.createIcons({ nodes: [container] });

    // Wire up interactions
    wireTimelineEvents(container);
  }

  function wireTimelineEvents(container) {
    const track = document.getElementById('tl-track');
    const thumb = document.getElementById('tl-thumb');
    if (!track || !thumb) return;

    // Drag handling
    function onDragStart(e) {
      e.preventDefault();
      tlIsDragging = true;
      thumb.classList.add('dragging');
      document.addEventListener('mousemove', onDragMove);
      document.addEventListener('mouseup', onDragEnd);
      document.addEventListener('touchmove', onDragMove, { passive: false });
      document.addEventListener('touchend', onDragEnd);
    }

    function onDragMove(e) {
      if (!tlIsDragging) return;
      e.preventDefault();
      const rect = track.getBoundingClientRect();
      const clientX = e.touches ? e.touches[0].clientX : e.clientX;
      let pct = (clientX - rect.left) / rect.width;
      pct = Math.max(0, Math.min(1, pct));
      tlCurrentPos = pct;
      updateThumbPosition();
      // Show canvas time indicator during drag
      const timeRange = tlMaxTime - tlMinTime;
      if (timeRange > 0) {
        const t = tlMinTime + tlCurrentPos * timeRange;
        showCanvasTimeOverlay(formatTimeShort(t), '');
        // Throttle graph updates to animation frames
        if (!tlDragRafPending) {
          tlDragRafPending = true;
          requestAnimationFrame(function () {
            tlDragRafPending = false;
            applyDragTimeFilter(t);
          });
        }
      }
    }

    function onDragEnd() {
      if (!tlIsDragging) return;
      tlIsDragging = false;
      thumb.classList.remove('dragging');
      document.removeEventListener('mousemove', onDragMove);
      document.removeEventListener('mouseup', onDragEnd);
      document.removeEventListener('touchmove', onDragMove);
      document.removeEventListener('touchend', onDragEnd);
      hideCanvasTimeOverlay();
      // If near right edge, reset to live mode (full graph, no snapshot)
      if (tlCurrentPos >= 0.995) {
        resetToLive();
        renderTimeline(document.getElementById('timeline-container'));
        return;
      }
      // Finalize drag animation state
      snapshotMode = true;
      tlIsLive = false;
      const timeRange = tlMaxTime - tlMinTime;
      if (timeRange > 0) {
        snapshotTime = new Date(tlMinTime + tlCurrentPos * timeRange).toISOString().replace('Z', '');
      }
      // Show stats overlay
      const entityCount = tlDragVisibleNodeIds.size;
      const relationCount = tlDragVisibleEdgeIds.size;
      const targetTime = timeRange > 0 ? new Date(tlMinTime + tlCurrentPos * timeRange) : null;
      if (targetTime) {
        showCanvasTimeOverlay(formatTimeShort(targetTime.getTime()), entityCount + 'E ' + relationCount + 'R');
        setTimeout(hideCanvasTimeOverlay, 2500);
      }
      renderTimeline(document.getElementById('timeline-container'));
      showToast(t('timeline.snapshotAt', { time: targetTime ? targetTime.toLocaleString() : '' }), 'info');
    }

    thumb.addEventListener('mousedown', onDragStart);
    thumb.addEventListener('touchstart', onDragStart);

    // Click on track to jump
    track.addEventListener('click', function (e) {
      if (e.target.classList.contains('timeline-marker')) {
        // Click on marker — jump to that event's time
        const idx = parseInt(e.target.getAttribute('data-ep-idx'), 10);
        if (!isNaN(idx) && tlEpisodes[idx]) {
          const timeRange = tlMaxTime - tlMinTime;
          if (timeRange > 0) {
            tlCurrentPos = (tlEpisodes[idx].time - tlMinTime) / timeRange;
            tlCurrentPos = Math.max(0, Math.min(1, tlCurrentPos));
            updateThumbPosition();
            applyTimelinePosition();
          }
        }
        return;
      }
      if (e.target === thumb) return;
      const rect = track.getBoundingClientRect();
      let pct = (e.clientX - rect.left) / rect.width;
      pct = Math.max(0, Math.min(1, pct));
      tlCurrentPos = pct;
      updateThumbPosition();
      applyTimelinePosition();
    });

    // Playback controls
    const playBtn = document.getElementById('tl-play-btn');
    if (playBtn) {
      playBtn.addEventListener('click', function () {
        if (tlPlaybackTimer || growActive) {
          stopTimelinePlayback();
          stopGrowAnimation();
        } else if (tlPlaybackMode === 'grow') {
          startGrowAnimation();
        } else {
          startTimelinePlayback();
        }
        renderTimeline(container);
      });
    }

    const stepBackBtn = document.getElementById('tl-step-back');
    if (stepBackBtn) {
      stepBackBtn.addEventListener('click', function () {
        stepTimeline(-1);
      });
    }

    const stepFwdBtn = document.getElementById('tl-step-forward');
    if (stepFwdBtn) {
      stepFwdBtn.addEventListener('click', function () {
        stepTimeline(1);
      });
    }

    const resetBtn = document.getElementById('tl-reset-live');
    if (resetBtn) {
      resetBtn.addEventListener('click', function () {
        resetToLive();
        renderTimeline(container);
      });
    }

    const speedBtn = document.getElementById('tl-speed');
    if (speedBtn) {
      speedBtn.addEventListener('click', function () {
        const speeds = [0.5, 1, 2, 3, 5];
        const curIdx = speeds.indexOf(tlPlaybackSpeed);
        tlPlaybackSpeed = speeds[(curIdx + 1) % speeds.length];
        speedBtn.textContent = tlPlaybackSpeed + 's';
        // Update grow animation speed if running
        if (growAnimTimer) {
          clearInterval(growAnimTimer);
          growAnimTimer = setInterval(growStep, Math.max(tlPlaybackSpeed * 100, 50));
        }
      });
    }

    // Mode toggle buttons
    const growModeBtn = document.getElementById('tl-mode-grow');
    const snapModeBtn = document.getElementById('tl-mode-snapshot');
    if (growModeBtn) {
      growModeBtn.addEventListener('click', function () {
        tlPlaybackMode = 'grow';
        stopTimelinePlayback();
        stopGrowAnimation();
        renderTimeline(container);
      });
    }
    if (snapModeBtn) {
      snapModeBtn.addEventListener('click', function () {
        tlPlaybackMode = 'snapshot';
        stopTimelinePlayback();
        stopGrowAnimation();
        renderTimeline(container);
      });
    }

    // Density bar click to navigate time
    const densityCols = container.querySelectorAll('.timeline-density-col');
    densityCols.forEach(function (col) {
      col.style.cursor = 'pointer';
      col.addEventListener('click', function (e) {
        e.stopPropagation();
        var pct = parseFloat(col.getAttribute('data-bin-pct')) / 100;
        if (!isNaN(pct)) {
          tlCurrentPos = Math.max(0, Math.min(1, pct));
          updateThumbPosition();
          applyTimelinePosition();
        }
      });
      col.addEventListener('mouseenter', function () {
        var pct = parseFloat(col.getAttribute('data-bin-pct')) / 100;
        if (!isNaN(pct)) {
          var timeRange = tlMaxTime - tlMinTime;
          var t = tlMinTime + pct * timeRange;
          showCanvasTimeOverlay(formatTimeShort(t), col.getAttribute('title') || '');
        }
      });
      col.addEventListener('mouseleave', function () {
        if (!growActive) hideCanvasTimeOverlay();
      });
    });
  }

  function updateThumbPosition() {
    const fill = document.querySelector('.timeline-track-fill');
    const thumb = document.getElementById('tl-thumb');
    const curTime = document.getElementById('tl-current-time');
    if (fill) fill.style.width = (tlCurrentPos * 100).toFixed(2) + '%';
    if (thumb) thumb.style.left = (tlCurrentPos * 100).toFixed(2) + '%';

    const timeRange = tlMaxTime - tlMinTime;
    if (curTime && timeRange > 0 && tlCurrentPos < 1) {
      const t = new Date(tlMinTime + tlCurrentPos * timeRange);
      curTime.textContent = t.toLocaleString();
    } else if (curTime) {
      curTime.textContent = '';
    }
  }

  async function applyTimelinePosition() {
    if (tlCurrentPos >= 0.995) {
      // Close enough to right edge = live
      if (!tlIsLive) {
        resetToLive();
        renderTimeline(document.getElementById('timeline-container'));
      }
      return;
    }

    const timeRange = tlMaxTime - tlMinTime;
    const targetTime = new Date(tlMinTime + tlCurrentPos * timeRange);
    const isoTime = targetTime.toISOString().replace('Z', '');

    // Use local cache for instant response (no API call needed)
    if (cachedAllNodes.length > 0) {
      snapshotMode = true;
      snapshotTime = isoTime;
      tlIsLive = false;

      // Reset drag state and use applyDragTimeFilter for smooth transition
      tlDragSortedNodes = [];
      tlDragSortedEdges = [];
      tlDragVisibleNodeIds = new Set();
      tlDragVisibleEdgeIds = new Set();
      tlDragLastTime = 0;

      applyDragTimeFilter(targetTime.getTime());

      const entityCount = tlDragVisibleNodeIds.size;
      const relationCount = tlDragVisibleEdgeIds.size;
      showCanvasTimeOverlay(
        formatTimeShort(targetTime.getTime()),
        entityCount + 'E ' + relationCount + 'R'
      );
      setTimeout(hideCanvasTimeOverlay, 2500);

      renderTimeline(document.getElementById('timeline-container'));
      showToast(t('timeline.snapshotAt', { time: targetTime.toLocaleString() }), 'info');
      return;
    }

    // Fallback: use API if no cached data
    const loadingEl = document.getElementById('graph-loading');
    if (loadingEl) loadingEl.style.display = 'flex';

    try {
      const res = await state.api.getSnapshot(isoTime, state.currentGraphId);
      renderSnapshotData(res.data || res, true);
      snapshotMode = true;
      snapshotTime = isoTime;
      tlIsLive = false;

      const snapData = res.data || res;
      const snapEntities = (snapData.entities || []).length;
      const snapRelations = (snapData.relations || []).length;
      showCanvasTimeOverlay(
        formatTimeShort(targetTime.getTime()),
        snapEntities + 'E ' + snapRelations + 'R'
      );
      setTimeout(hideCanvasTimeOverlay, 2500);

      renderTimeline(document.getElementById('timeline-container'));
      showToast(t('timeline.snapshotAt', { time: targetTime.toLocaleString() }), 'info');
    } catch (err) {
      console.error('Timeline snapshot failed:', err);
      showToast(t('graph.loadFailed') + ': ' + err.message, 'error');
    } finally {
      if (loadingEl) loadingEl.style.display = 'none';
    }
  }

  function stepTimeline(direction) {
    // Step through episode markers
    if (tlEpisodes.length === 0) return;
    const timeRange = tlMaxTime - tlMinTime;
    if (timeRange === 0) return;

    // Find nearest episode index based on current position
    const currentTime = tlMinTime + tlCurrentPos * timeRange;
    let nearestIdx = -1;

    if (direction > 0) {
      // Step forward: find next episode after current time
      for (let i = 0; i < tlEpisodes.length; i++) {
        if (tlEpisodes[i].time > currentTime + 1000) {
          nearestIdx = i;
          break;
        }
      }
      if (nearestIdx < 0) nearestIdx = tlEpisodes.length - 1;
    } else {
      // Step backward: find previous episode before current time
      for (let i = tlEpisodes.length - 1; i >= 0; i--) {
        if (tlEpisodes[i].time < currentTime - 1000) {
          nearestIdx = i;
          break;
        }
      }
      if (nearestIdx < 0) {
        // Before first episode — reset to live
        resetToLive();
        renderTimeline(document.getElementById('timeline-container'));
        return;
      }
    }

    const ep = tlEpisodes[nearestIdx];
    tlCurrentPos = (ep.time - tlMinTime) / timeRange;
    tlCurrentPos = Math.max(0, Math.min(1, tlCurrentPos));
    tlIsLive = false;
    updateThumbPosition();
    applyTimelinePosition();
  }

  // ---- Timeline drag animation: filter graph by time during drag ----

  function prepareDragSortedData() {
    if (tlDragSortedNodes.length > 0) return; // already prepared
    // Sort nodes by processed_time
    tlDragSortedNodes = cachedAllNodes.slice().sort(function (a, b) {
      var ta = a.processed_time ? new Date(a.processed_time).getTime() : 0;
      var tb = b.processed_time ? new Date(b.processed_time).getTime() : 0;
      return ta - tb;
    });
    // Sort edges by the later of their two endpoint times
    tlDragSortedEdges = cachedAllEdges.slice().sort(function (a, b) {
      var ta1 = (cachedAllEntities[a.entity1_absolute_id] || {}).processed_time;
      var ta2 = (cachedAllEntities[a.entity2_absolute_id] || {}).processed_time;
      var tb1 = (cachedAllEntities[b.entity1_absolute_id] || {}).processed_time;
      var tb2 = (cachedAllEntities[b.entity2_absolute_id] || {}).processed_time;
      var taMax = Math.max(ta1 ? new Date(ta1).getTime() : 0, ta2 ? new Date(ta2).getTime() : 0);
      var tbMax = Math.max(tb1 ? new Date(tb1).getTime() : 0, tb2 ? new Date(tb2).getTime() : 0);
      return taMax - tbMax;
    });
  }

  function applyDragTimeFilter(targetTimeMs) {
    if (!explorer || cachedAllNodes.length === 0) return;

    // Debounce: skip if time hasn't changed meaningfully (within 1 second)
    if (Math.abs(targetTimeMs - tlDragLastTime) < 1000 && tlDragVisibleNodeIds.size > 0) return;
    tlDragLastTime = targetTimeMs;

    // Stop any running grow animation
    if (growActive) stopGrowAnimation();

    // Prepare sorted data on first drag
    prepareDragSortedData();

    // Find nodes with processed_time <= targetTimeMs using binary search
    var nodeEndIdx = tlDragSortedNodes.length;
    for (var i = 0; i < tlDragSortedNodes.length; i++) {
      var nt = tlDragSortedNodes[i].processed_time ? new Date(tlDragSortedNodes[i].processed_time).getTime() : 0;
      if (nt > targetTimeMs) { nodeEndIdx = i; break; }
    }
    // Find edges with max endpoint time <= targetTimeMs
    var edgeEndIdx = tlDragSortedEdges.length;
    for (var j = 0; j < tlDragSortedEdges.length; j++) {
      var ea1 = (cachedAllEntities[tlDragSortedEdges[j].entity1_absolute_id] || {}).processed_time;
      var ea2 = (cachedAllEntities[tlDragSortedEdges[j].entity2_absolute_id] || {}).processed_time;
      var eMaxT = Math.max(ea1 ? new Date(ea1).getTime() : 0, ea2 ? new Date(ea2).getTime() : 0);
      if (eMaxT > targetTimeMs) { edgeEndIdx = j; break; }
    }

    // Determine target visible sets
    var targetNodeIds = new Set();
    var targetNodeData = [];
    for (var ni = 0; ni < nodeEndIdx; ni++) {
      targetNodeIds.add(tlDragSortedNodes[ni].absolute_id);
      targetNodeData.push(tlDragSortedNodes[ni]);
    }
    // Edges: only include if both endpoints are visible
    var targetEdgeIds = new Set();
    var targetEdgeData = [];
    for (var ei = 0; ei < edgeEndIdx; ei++) {
      var edge = tlDragSortedEdges[ei];
      if (targetNodeIds.has(edge.entity1_absolute_id) && targetNodeIds.has(edge.entity2_absolute_id)) {
        targetEdgeIds.add(edge.absolute_id);
        targetEdgeData.push(edge);
      }
    }

    // If first drag frame, seed from current visible state
    if (tlDragVisibleNodeIds.size === 0) {
      tlDragHubLayout = computeHubLayout(cachedAllEdges);
      // Seed from current graph state instead of clearing to empty
      var currentVisNodes = explorer.getCurrentNodes();
      var currentVisEdges = explorer.getCurrentEdges();
      if (currentVisNodes.length > 0) {
        for (var si = 0; si < currentVisNodes.length; si++) {
          tlDragVisibleNodeIds.add(currentVisNodes[si].id);
        }
        for (var sj = 0; sj < currentVisEdges.length; sj++) {
          tlDragVisibleEdgeIds.add(currentVisEdges[sj].id);
        }
      } else {
        // Empty graph — start fresh
        explorer.initEmptyGraph(tlDragHubLayout);
      }
    }

    // Compute diffs: what to add and what to remove
    var nodesToAdd = [];
    for (var ai = 0; ai < targetNodeData.length; ai++) {
      if (!tlDragVisibleNodeIds.has(targetNodeData[ai].absolute_id)) {
        nodesToAdd.push(targetNodeData[ai]);
      }
    }
    var nodesToRemove = [];
    tlDragVisibleNodeIds.forEach(function (absId) {
      if (!targetNodeIds.has(absId)) nodesToRemove.push(absId);
    });
    var edgesToAdd = [];
    for (var aei = 0; aei < targetEdgeData.length; aei++) {
      if (!tlDragVisibleEdgeIds.has(targetEdgeData[aei].absolute_id)) {
        edgesToAdd.push(targetEdgeData[aei]);
      }
    }
    var edgesToRemove = [];
    tlDragVisibleEdgeIds.forEach(function (absId) {
      if (!targetEdgeIds.has(absId)) edgesToRemove.push(absId);
    });

    // Update state before modifying graph
    tlDragVisibleNodeIds = targetNodeIds;
    tlDragVisibleEdgeIds = targetEdgeIds;

    // Apply incremental changes (use cached hub layout)
    if (nodesToAdd.length > 0 || edgesToAdd.length > 0) {
      explorer.addNodesAndEdges(nodesToAdd, edgesToAdd, tlDragHubLayout);
    }
    if (nodesToRemove.length > 0 || edgesToRemove.length > 0) {
      explorer.removeNodesAndEdges(nodesToRemove, edgesToRemove);
    }
  }

  // ---- Grow Animation: entity-by-entity graph building ----

  function startGrowAnimation() {
    if (cachedAllNodes.length === 0) return;

    // Sort nodes by processed_time
    growSortedNodes = cachedAllNodes.slice().sort(function (a, b) {
      var ta = a.processed_time ? new Date(a.processed_time).getTime() : 0;
      var tb = b.processed_time ? new Date(b.processed_time).getTime() : 0;
      return ta - tb;
    });

    // Sort edges by the later of their two endpoint times
    growSortedEdges = cachedAllEdges.slice().sort(function (a, b) {
      var ta1 = (cachedAllEntities[a.entity1_absolute_id] || {}).processed_time;
      var ta2 = (cachedAllEntities[a.entity2_absolute_id] || {}).processed_time;
      var tb1 = (cachedAllEntities[b.entity1_absolute_id] || {}).processed_time;
      var tb2 = (cachedAllEntities[b.entity2_absolute_id] || {}).processed_time;
      var taMax = Math.max(ta1 ? new Date(ta1).getTime() : 0, ta2 ? new Date(ta2).getTime() : 0);
      var tbMax = Math.max(tb1 ? new Date(tb1).getTime() : 0, tb2 ? new Date(tb2).getTime() : 0);
      return taMax - tbMax;
    });

    growIndex = 0;
    growEdgeIndex = 0;
    growActive = true;
    growVisibleNodes = [];
    growVisibleEdges = [];
    tlIsLive = false;

    // Precompute hub layout from ALL edges (stable throughout animation)
    var hubLayout = computeHubLayout(cachedAllEdges);

    // Start with empty graph using persistent DataSets
    explorer.initEmptyGraph(hubLayout);

    // Play snapshot transition effect
    playSnapshotTransition();

    // Start animation interval — target ~15-40 seconds total regardless of speed
    var interval = Math.max(tlPlaybackSpeed * 100, 50);
    growAnimTimer = setInterval(growStep, interval);

    renderTimeline(document.getElementById('timeline-container'));
  }

  function growStep() {
    if (!growActive || !explorer) {
      stopGrowAnimation();
      return;
    }

    // Target ~200 total frames for a smooth animation regardless of graph size
    var totalSteps = 200;
    var nodeBatchSize = Math.max(1, Math.ceil(growSortedNodes.length / totalSteps));
    var addedNodes = [];
    var currentNodeTime = 0;

    for (var i = 0; i < nodeBatchSize && growIndex < growSortedNodes.length; i++) {
      var node = growSortedNodes[growIndex];
      addedNodes.push(node);
      currentNodeTime = node.processed_time ? new Date(node.processed_time).getTime() : 0;
      growIndex++;
    }

    if (addedNodes.length === 0) {
      // All nodes added — continue to add remaining edges
      growAddRemainingEdges();
      return;
    }

    // Accumulate nodes
    growVisibleNodes = growVisibleNodes.concat(addedNodes);
    var visibleAbsIds = new Set(growVisibleNodes.map(function (n) { return n.absolute_id; }));

    // Advance the edge frontier in time order — edges appear when their time <= current node time
    var newlyConnectable = [];
    var currentEdgeIds = new Set(growVisibleEdges.map(function (e) { return e.absolute_id; }));

    // Move edge frontier forward: advance growEdgeIndex up to currentNodeTime
    while (growEdgeIndex < growSortedEdges.length) {
      var edge = growSortedEdges[growEdgeIndex];
      var eTime1 = (cachedAllEntities[edge.entity1_absolute_id] || {}).processed_time;
      var eTime2 = (cachedAllEntities[edge.entity2_absolute_id] || {}).processed_time;
      var edgeTime = Math.max(
        eTime1 ? new Date(eTime1).getTime() : 0,
        eTime2 ? new Date(eTime2).getTime() : 0
      );
      if (edgeTime > currentNodeTime) break;
      // Edge is within current time window — add if both endpoints visible
      if (visibleAbsIds.has(edge.entity1_absolute_id) && visibleAbsIds.has(edge.entity2_absolute_id)
          && !currentEdgeIds.has(edge.absolute_id)) {
        newlyConnectable.push(edge);
        currentEdgeIds.add(edge.absolute_id);
      }
      growEdgeIndex++;
    }

    // Also check previously-seen edges that might now be connectable (one endpoint was added this frame)
    var addedAbsIds = new Set(addedNodes.map(function (n) { return n.absolute_id; }));
    for (var ei = 0; ei < growEdgeIndex; ei++) {
      var edge2 = growSortedEdges[ei];
      // Skip if already in growVisibleEdges or already found
      if (currentEdgeIds.has(edge2.absolute_id)) continue;
      // Only consider edges where at least one endpoint was just added
      if (!addedAbsIds.has(edge2.entity1_absolute_id) && !addedAbsIds.has(edge2.entity2_absolute_id)) continue;
      if (visibleAbsIds.has(edge2.entity1_absolute_id) && visibleAbsIds.has(edge2.entity2_absolute_id)) {
        newlyConnectable.push(edge2);
        currentEdgeIds.add(edge2.absolute_id);
      }
    }

    growVisibleEdges = growVisibleEdges.concat(newlyConnectable);

    // Incrementally add new nodes and edges (no full rebuild!)
    var hubLayout = computeHubLayout(cachedAllEdges);
    explorer.addNodesAndEdges(addedNodes, newlyConnectable, hubLayout);

    // Update timeline position
    var timeRange = tlMaxTime - tlMinTime;
    if (timeRange > 0 && currentNodeTime > 0) {
      tlCurrentPos = Math.min((currentNodeTime - tlMinTime) / timeRange, 1);
      updateThumbPosition();
    }

    // Update stats with progress
    var statsEl = document.getElementById('graph-stats');
    if (statsEl) {
      var pct = Math.round(growIndex / growSortedNodes.length * 100);
      statsEl.textContent = t('graph.loaded', { entities: growVisibleNodes.length, relations: growVisibleEdges.length })
        + ' (' + pct + '%)';
    }

    // Update canvas time indicator
    if (currentNodeTime > 0) {
      showCanvasTimeOverlay(
        formatTimeShort(currentNodeTime),
        growVisibleNodes.length + 'E ' + growVisibleEdges.length + 'R'
      );
    }
  }

  function growAddRemainingEdges() {
    if (!growActive || !explorer) {
      stopGrowAnimation();
      return;
    }

    // Advance edge frontier to the end (all remaining edges are now fair game)
    var visibleAbsIds = new Set(growVisibleNodes.map(function (n) { return n.absolute_id; }));
    var currentEdgeIds = new Set(growVisibleEdges.map(function (e) { return e.absolute_id; }));

    // First, process any edges still in the frontier beyond growEdgeIndex
    var newlyConnectable = [];
    while (growEdgeIndex < growSortedEdges.length) {
      var edge = growSortedEdges[growEdgeIndex];
      if (visibleAbsIds.has(edge.entity1_absolute_id) && visibleAbsIds.has(edge.entity2_absolute_id)
          && !currentEdgeIds.has(edge.absolute_id)) {
        newlyConnectable.push(edge);
        currentEdgeIds.add(edge.absolute_id);
      }
      growEdgeIndex++;
    }

    // Also catch any previously-seen edges that are now connectable
    var allRemaining = newlyConnectable;
    for (var ei = 0; ei < growSortedEdges.length; ei++) {
      var edge2 = growSortedEdges[ei];
      if (currentEdgeIds.has(edge2.absolute_id)) continue;
      if (visibleAbsIds.has(edge2.entity1_absolute_id) && visibleAbsIds.has(edge2.entity2_absolute_id)) {
        allRemaining.push(edge2);
        currentEdgeIds.add(edge2.absolute_id);
      }
    }

    if (allRemaining.length === 0) {
      finishGrowAnimation();
      return;
    }

    // Add a batch of the remaining edges
    var batchSize = Math.max(1, Math.ceil(allRemaining.length / 10));
    var batch = allRemaining.slice(0, batchSize);
    growVisibleEdges = growVisibleEdges.concat(batch);

    // Incrementally add edges only
    var hubLayout = computeHubLayout(cachedAllEdges);
    explorer.addNodesAndEdges([], batch, hubLayout);

    var statsEl = document.getElementById('graph-stats');
    if (statsEl) {
      statsEl.textContent = t('graph.loaded', { entities: growVisibleNodes.length, relations: growVisibleEdges.length })
        + ' (edges ' + Math.round(growVisibleEdges.length / growSortedEdges.length * 100) + '%)';
    }

    if (batch.length >= allRemaining.length) {
      finishGrowAnimation();
    }
  }

  function finishGrowAnimation() {
    growActive = false;
    if (growAnimTimer) {
      clearInterval(growAnimTimer);
      growAnimTimer = null;
    }
    hideCanvasTimeOverlay();
    tlIsLive = true;
    tlCurrentPos = 1.0;
    updateThumbPosition();

    // No full rebuild needed — all data was already added incrementally via addNodesAndEdges.
    // Just update version badges and stats for the final state.
    try {
      var allEntityIds = [...new Set(cachedAllNodes.map(function(e) { return e.family_id; }))];
      if (allEntityIds.length > 0) {
        state.api.entityVersionCounts(allEntityIds, state.currentGraphId).then(function(vcRes) {
          var vc = vcRes.data || {};
          if (explorer) explorer.setVersionCounts(vc);
        }).catch(function() {});
      }
    } catch (_) {}

    var statsEl = document.getElementById('graph-stats');
    if (statsEl) {
      statsEl.textContent = t('graph.loaded', { entities: cachedAllNodes.length, relations: cachedAllEdges.length });
    }

    renderTimeline(document.getElementById('timeline-container'));
    showToast('Graph growth animation complete!', 'success');
  }

  function stopGrowAnimation() {
    growActive = false;
    if (growAnimTimer) {
      clearInterval(growAnimTimer);
      growAnimTimer = null;
    }
    hideCanvasTimeOverlay();
  }

  function startTimelinePlayback() {
    if (tlEpisodes.length === 0) return;

    // Start from first episode if currently live
    if (tlIsLive) {
      const timeRange = tlMaxTime - tlMinTime;
      if (timeRange > 0 && tlEpisodes.length > 0) {
        tlPlaybackIndex = 0;
        tlCurrentPos = (tlEpisodes[0].time - tlMinTime) / timeRange;
      }
    } else {
      // Find current position in episode list
      const timeRange = tlMaxTime - tlMinTime;
      const currentTime = tlMinTime + tlCurrentPos * timeRange;
      tlPlaybackIndex = 0;
      for (let i = 0; i < tlEpisodes.length; i++) {
        if (tlEpisodes[i].time >= currentTime) {
          tlPlaybackIndex = i;
          break;
        }
      }
    }

    function playStep() {
      if (tlPlaybackIndex >= tlEpisodes.length) {
        stopTimelinePlayback();
        resetToLive();
        renderTimeline(document.getElementById('timeline-container'));
        return;
      }

      const ep = tlEpisodes[tlPlaybackIndex];
      const timeRange = tlMaxTime - tlMinTime;
      if (timeRange > 0) {
        tlCurrentPos = (ep.time - tlMinTime) / timeRange;
        tlCurrentPos = Math.max(0, Math.min(1, tlCurrentPos));
      }

      // Apply snapshot
      const isoTime = new Date(ep.time).toISOString().replace('Z', '');
      state.api.getSnapshot(isoTime, state.currentGraphId).then(function (res) {
        renderSnapshotData(res.data || res, true);
        snapshotMode = true;
        tlIsLive = false;
        updateThumbPosition();

        // Update current time display
        const curTime = document.getElementById('tl-current-time');
        if (curTime) curTime.textContent = new Date(ep.time).toLocaleString();

        tlPlaybackIndex++;
        tlPlaybackTimer = setTimeout(playStep, tlPlaybackSpeed * 1000);
      }).catch(function () {
        tlPlaybackIndex++;
        tlPlaybackTimer = setTimeout(playStep, tlPlaybackSpeed * 1000);
      });
    }

    playStep();
  }

  function stopTimelinePlayback() {
    if (tlPlaybackTimer) {
      clearTimeout(tlPlaybackTimer);
      tlPlaybackTimer = null;
    }
    tlPlaybackIndex = -1;
  }

  function resetToLive() {
    stopTimelinePlayback();
    tlIsLive = true;
    tlCurrentPos = 1.0;
    snapshotMode = false;
    snapshotTime = null;
    updateSnapshotOverlay(null);
    // Clear drag animation state
    tlDragSortedNodes = [];
    tlDragSortedEdges = [];
    tlDragVisibleNodeIds = new Set();
    tlDragVisibleEdgeIds = new Set();
    tlDragLastTime = 0;
    tlDragHubLayout = null;
    // Rebuild graph from cache
    if (cachedAllNodes.length > 0) {
      const hubLayout = computeHubLayout(cachedAllEdges);
      explorer.buildGraph(cachedAllNodes, cachedAllEdges, null, null, cachedInheritedRelationIds, undefined, hubLayout);
    }
  }

  async function loadTimelineChanges() {
    // Load recent changes to enrich timeline markers
    try {
      if (!tlMaxTime) return;
      const since = new Date(tlMinTime).toISOString();
      const res = await state.api.getChanges(since, null, state.currentGraphId);
      const changes = res.data?.changes || res.data || [];
      // Add change events as additional markers (if not already covered by episodes)
      // We don't need to do much here — episodes are the primary markers
    } catch (_) {}
  }

  // ---- Register this page ----

  registerPage('graph', { render, destroy });
})();

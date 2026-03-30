/* ==========================================
   Graph Utilities — Shared vis-network helpers
   Used by graph.js and search.js
   ========================================== */

window.GraphUtils = (function () {
  'use strict';

  // ---- Unified 4-tier color system ----
  // Used across search ranking, graph hop levels, and path finder
  //   Tier 1: Red    — Primary / Focus / #1
  //   Tier 2: Amber  — Secondary / #2~5
  //   Tier 3: Teal   — Tertiary  / #6~10
  //   Tier 4: Slate  — Low       / #11+

  const TIER_1    = { bg: '#ef4444', border: '#f87171' };
  const TIER_2    = { bg: '#f59e0b', border: '#fbbf24' };
  const TIER_3    = { bg: '#14b8a6', border: '#2dd4bf' };
  const TIER_4    = { bg: '#64748b', border: '#94a3b8' };

  const HOP_PALETTE = [TIER_1, TIER_2, TIER_3, TIER_4];

  const DEFAULT_LIGHT = { bg: '#a5b4fc', border: '#818cf8' };
  const DEFAULT_DARK  = { bg: '#6366f1', border: '#818cf8' };

  // Search page: rank-based entity colors (same 4 tiers)
  const RANK_1       = TIER_1;
  const RANK_2_5     = TIER_2;
  const RANK_6_10    = TIER_3;
  const RANK_OTHER   = TIER_4;

  // Expanded neighbor (not in rankMap) — distinct neutral color
  const SEARCH_EXPANDED_LIGHT = { bg: '#a5b4fc', border: '#818cf8' };
  const SEARCH_EXPANDED_DARK  = { bg: '#6366f1', border: '#818cf8' };

  // Community color palette (20 distinct colors)
  const COMMUNITY_PALETTE = [
    { bg: '#ef4444', border: '#f87171' }, // red
    { bg: '#f59e0b', border: '#fbbf24' }, // amber
    { bg: '#10b981', border: '#34d399' }, // emerald
    { bg: '#3b82f6', border: '#60a5fa' }, // blue
    { bg: '#8b5cf6', border: '#a78bfa' }, // violet
    { bg: '#ec4899', border: '#f472b6' }, // pink
    { bg: '#14b8a6', border: '#2dd4bf' }, // teal
    { bg: '#f97316', border: '#fb923c' }, // orange
    { bg: '#06b6d4', border: '#22d3ee' }, // cyan
    { bg: '#84cc16', border: '#a3e635' }, // lime
    { bg: '#6366f1', border: '#818cf8' }, // indigo
    { bg: '#d946ef', border: '#e879f9' }, // fuchsia
    { bg: '#0ea5e9', border: '#38bdf8' }, // sky
    { bg: '#a855f7', border: '#c084fc' }, // purple
    { bg: '#e11d48', border: '#fb7185' }, // rose
    { bg: '#65a30d', border: '#84cc16' }, // green
    { bg: '#7c3aed', border: '#8b5cf6' }, // violet-dark
    { bg: '#0891b2', border: '#06b6d4' }, // teal-dark
    { bg: '#c2410c', border: '#ea580c' }, // orange-dark
    { bg: '#4f46e5', border: '#6366f1' }, // indigo-dark
  ];

  function getRankColor(rank) {
    if (rank === 1) return TIER_1;
    if (rank <= 5) return TIER_2;
    if (rank <= 10) return TIER_3;
    return TIER_4;
  }

  // ---- Edge colors ----

  const EDGE_CURRENT  = { color: '#4b5563', highlight: '#9ca3af', hover: '#6b7280' };
  const EDGE_INHERITED = { color: '#d97706', highlight: '#fbbf24', hover: '#b45309' };
  const EDGE_FUTURE   = { color: '#3b82f6', highlight: '#60a5fa', hover: '#2563eb' };

  // ---- Theme detection ----

  function isLightTheme() {
    return document.documentElement.getAttribute('data-theme') === 'light';
  }

  // ---- Build nodes ----
  //   entities: array of entity objects
  //   options:
  //     colorMode: 'hop' | 'search' | 'community' | 'default'
  //     versionCounts: { entity_id: count }
  //     hopMap: { absoluteId: hopLevel }          (for 'hop' mode)
  //     highlightAbsId: string                    (focused entity id)
  //     rankMap: { absoluteId: 1-based-rank }     (for 'search' mode, rank-based coloring)
  //     communityMap: { absoluteId: communityId }  (for 'community' mode)
  //     versionLabel: { idx: number, total: number }  (focused entity version label)
  //     unnamedLabel: string                      (fallback for unnamed entities)

  function buildNodes(entities, options) {
    options = options || {};
    const light = isLightTheme();

    // Font colors
    const labelFontColor = light ? '#1e293b' : '#e2e8f0';
    const highlightFontColor = light ? '#1e40af' : '#ffffff';

    const versionCounts = options.versionCounts || {};
    const hopMap = options.hopMap || null;
    const highlightAbsId = options.highlightAbsId || null;
    const rankMap = options.rankMap || null;
    const communityMap = options.communityMap || null;
    const versionLabel = options.versionLabel || null;
    const unnamedLabel = options.unnamedLabel || 'unnamed';

    const entityMap = {};
    const nodeIds = new Set();

    const nodes = new vis.DataSet(
      entities.map(function (e) {
        entityMap[e.absolute_id] = e;
        nodeIds.add(e.absolute_id);

        const baseName = e.name || e.entity_id || unnamedLabel;
        const isHighlight = highlightAbsId && e.absolute_id === highlightAbsId;
        const hopLevel = hopMap ? hopMap[e.absolute_id] : undefined;

        // Label formatting
        let label;
        if (isHighlight && versionLabel && versionLabel.total > 1) {
          label = baseName + ' [' + versionLabel.idx + '/' + versionLabel.total + ']';
        } else {
          const vc = versionCounts[e.entity_id] || 1;
          label = vc > 1 ? baseName + ' [' + vc + ']' : baseName;
        }

        // Color selection
        let bgColor, borderColor;
        if (options.colorMode === 'hop' && hopMap && hopLevel !== undefined) {
          var palette = HOP_PALETTE[Math.min(hopLevel, HOP_PALETTE.length - 1)];
          bgColor = palette.bg;
          borderColor = palette.border;
        } else if (options.colorMode === 'search') {
          var rank = rankMap ? rankMap[e.absolute_id] : undefined;
          if (rank !== undefined) {
            var rankScheme = getRankColor(rank);
            bgColor = rankScheme.bg;
            borderColor = rankScheme.border;
          } else {
            // Expanded neighbor (not in rankMap)
            var expandedScheme = light ? SEARCH_EXPANDED_LIGHT : SEARCH_EXPANDED_DARK;
            bgColor = expandedScheme.bg;
            borderColor = expandedScheme.border;
          }
        } else if (options.colorMode === 'community' && communityMap) {
          var cid = communityMap[e.absolute_id];
          if (cid !== undefined && cid !== null) {
            var commColor = COMMUNITY_PALETTE[cid % COMMUNITY_PALETTE.length];
            bgColor = commColor.bg;
            borderColor = commColor.border;
          } else {
            var defaultColor = light ? DEFAULT_LIGHT : DEFAULT_DARK;
            bgColor = defaultColor.bg;
            borderColor = defaultColor.border;
          }
        } else {
          // default
          var defaultColor = light ? DEFAULT_LIGHT : DEFAULT_DARK;
          bgColor = defaultColor.bg;
          borderColor = defaultColor.border;
        }

        var nodeFontColor = isHighlight ? highlightFontColor : labelFontColor;
        var nodeSize = isHighlight ? 25 : (hopLevel === 0 ? 25 : 20);

        return {
          id: e.absolute_id,
          label: label,
          title: typeof escapeHtml !== 'undefined' ? escapeHtml(typeof truncate !== 'undefined' ? truncate(e.content || e.name || '', 80) : (e.content || e.name || '')) : (e.content || e.name || ''),
          color: {
            background: bgColor,
            border: borderColor,
            highlight: { background: borderColor, border: '#a5b4fc' },
            hover: { background: borderColor, border: '#a5b4fc' },
          },
          size: options.colorMode === 'search' && rankMap
            ? (rankMap[e.absolute_id] === 1 ? 28 : (rankMap[e.absolute_id] <= 5 ? 22 : (rankMap[e.absolute_id] <= 10 ? 18 : 14)))
            : nodeSize,
          shape: 'dot',
          font: {
            color: nodeFontColor,
            size: isHighlight ? 12 : 11,
            face: 'Inter, sans-serif',
            bold: isHighlight ? { color: nodeFontColor, size: 12, face: 'Inter, sans-serif' } : undefined,
          },
        };
      })
    );

    return { nodes: nodes, entityMap: entityMap, nodeIds: nodeIds };
  }

  // ---- Build edges ----
  //   relations: array of relation objects
  //   nodeIds: Set<absoluteId> — visible node IDs
  //   options:
  //     inheritedRelationIds: Set<absoluteId>

  function buildEdges(relations, nodeIds, options) {
    options = options || {};
    var inheritedRelationIds = options.inheritedRelationIds || null;
    var hasInherited = inheritedRelationIds && inheritedRelationIds.size > 0;
    var futureRelationIds = options.futureRelationIds || null;
    var hasFuture = futureRelationIds && futureRelationIds.size > 0;
    var weightMode = options.weightMode || null;

    var relationMap = {};

    var edges = new vis.DataSet(
      relations
        .filter(function (r) {
          return nodeIds.has(r.entity1_absolute_id) && nodeIds.has(r.entity2_absolute_id);
        })
        .map(function (r) {
          relationMap[r.absolute_id] = r;
          var isInherited = hasInherited && inheritedRelationIds.has(r.absolute_id);
          var isFuture = hasFuture && futureRelationIds.has(r.absolute_id);
          var edgeColor = isFuture ? EDGE_FUTURE : (isInherited ? EDGE_INHERITED : EDGE_CURRENT);
          var dashes = isFuture ? [2, 4] : (isInherited ? [5, 5] : false);

          return {
            id: r.absolute_id,
            from: r.entity1_absolute_id,
            to: r.entity2_absolute_id,
            color: edgeColor,
            dashes: dashes,
            smooth: {
              enabled: true,
              type: 'continuous',
              roundness: 0.2,
            },
          };
        })
    );

    // If weightMode is 'count', adjust edge width based on number of relations per entity pair
    if (weightMode === 'count') {
      var pairCount = {};
      var pairMaxCount = 1;
      edges.forEach(function (e) {
        var key = [e.from, e.to].sort().join('|');
        pairCount[key] = (pairCount[key] || 0) + 1;
        if (pairCount[key] > pairMaxCount) pairMaxCount = pairCount[key];
      });

      edges.forEach(function (e) {
        var key = [e.from, e.to].sort().join('|');
        var count = pairCount[key] || 1;
        e.width = Math.min(1 + count * 1.5, 8);
        // Adjust opacity: more relations = darker/more opaque
        var baseColor = e.color.color || '#4b5563';
        var opacity = 0.3 + (count / pairMaxCount) * 0.7;
        e.color = {
          color: baseColor,
          opacity: Math.round(opacity * 100) / 100,
          highlight: e.color.highlight,
          hover: e.color.hover,
        };
      });
    }

    return { edges: edges, relationMap: relationMap };
  }

  // ---- Physics options ----

  function getPhysicsOptions() {
    return {
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
    };
  }

  // ---- Interaction options ----

  function getInteractionOptions() {
    return {
      hover: true,
      tooltipDelay: 200,
      zoomView: true,
      dragView: true,
      navigationButtons: false,
      keyboard: false,
    };
  }

  // ---- Public API ----

  return {
    // Unified tier constants
    TIER_1: TIER_1,
    TIER_2: TIER_2,
    TIER_3: TIER_3,
    TIER_4: TIER_4,

    // Color palettes
    HOP_PALETTE: HOP_PALETTE,
    DEFAULT_LIGHT: DEFAULT_LIGHT,
    DEFAULT_DARK: DEFAULT_DARK,
    RANK_1: RANK_1,
    RANK_2_5: RANK_2_5,
    RANK_6_10: RANK_6_10,
    RANK_OTHER: RANK_OTHER,
    SEARCH_EXPANDED_LIGHT: SEARCH_EXPANDED_LIGHT,
    SEARCH_EXPANDED_DARK: SEARCH_EXPANDED_DARK,
    COMMUNITY_PALETTE: COMMUNITY_PALETTE,
    EDGE_CURRENT: EDGE_CURRENT,
    EDGE_INHERITED: EDGE_INHERITED,
    EDGE_FUTURE: EDGE_FUTURE,

    // Functions
    buildNodes: buildNodes,
    buildEdges: buildEdges,
    getPhysicsOptions: getPhysicsOptions,
    getInteractionOptions: getInteractionOptions,
  };
})();

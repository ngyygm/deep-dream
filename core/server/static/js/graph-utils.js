/* ==========================================
   Graph Utilities — Shared vis-network helpers
   Used by graph.js (physics/interaction), search.js (nodes/edges)
   ========================================== */

window.GraphUtils = (function () {
  'use strict';

  // ---- 4-tier color system (search rank coloring) ----
  //   Tier 1: Red    — Primary / Focus / #1
  //   Tier 2: Amber  — Secondary / #2~5
  //   Tier 3: Teal   — Tertiary  / #6~10
  //   Tier 4: Slate  — Low       / #11+

  var TIER_1    = { bg: '#ef4444', border: '#f87171' };
  var TIER_2    = { bg: '#f59e0b', border: '#fbbf24' };
  var TIER_3    = { bg: '#14b8a6', border: '#2dd4bf' };
  var TIER_4    = { bg: '#64748b', border: '#94a3b8' };

  var DEFAULT_LIGHT = { bg: '#f9a8d4', border: '#f472b6' };
  var DEFAULT_DARK  = { bg: '#ec4899', border: '#f472b6' };

  // Search page: rank-based entity colors (same 4 tiers)
  var RANK_1       = TIER_1;
  var RANK_2_5     = TIER_2;
  var RANK_6_10    = TIER_3;
  var RANK_OTHER   = TIER_4;

  // Expanded neighbor (not in rankMap) — distinct neutral color
  var SEARCH_EXPANDED_LIGHT = { bg: '#f9a8d4', border: '#f472b6' };
  var SEARCH_EXPANDED_DARK  = { bg: '#ec4899', border: '#f472b6' };

  function getRankColor(rank) {
    if (rank === 1) return TIER_1;
    if (rank <= 5) return TIER_2;
    if (rank <= 10) return TIER_3;
    return TIER_4;
  }

  // ---- Legacy edge colors ----

  var EDGE_DEFAULT  = { color: '#4b5563', highlight: '#9ca3af', hover: '#6b7280' };

  // ---- Theme detection (shared) ----

  var isLightTheme = Format.isLightTheme;

  // ---- Build nodes ----
  //   entities: array of entity objects
  //   options:
  //     colorMode: 'search' | 'default'
  //     rankMap: { absoluteId: 1-based-rank }     (for 'search' mode)
  //     unnamedLabel: string

  function buildNodes(entities, options) {
    options = options || {};
    var light = isLightTheme();

    // Font colors
    var labelFontColor = light ? '#1e293b' : '#e2e8f0';

    var rankMap = options.rankMap || null;
    var unnamedLabel = options.unnamedLabel || 'unnamed';

    var entityMap = {};
    var nodeIds = new Set();

    var nodes = new vis.DataSet(
      entities.map(function (e) {
        entityMap[e.absolute_id] = e;
        nodeIds.add(e.absolute_id);

        var baseName = e.name || e.family_id || unnamedLabel;
        var label = baseName;

        // ---- Color selection ----
        var bgColor, borderColor;

        if (options.colorMode === 'search') {
          var rank = rankMap ? rankMap[e.absolute_id] : undefined;
          if (rank !== undefined) {
            var rankScheme = getRankColor(rank);
            bgColor = rankScheme.bg;
            borderColor = rankScheme.border;
          } else {
            var expandedScheme = light ? SEARCH_EXPANDED_LIGHT : SEARCH_EXPANDED_DARK;
            bgColor = expandedScheme.bg;
            borderColor = expandedScheme.border;
          }
        } else {
          var defaultColor = light ? DEFAULT_LIGHT : DEFAULT_DARK;
          bgColor = defaultColor.bg;
          borderColor = defaultColor.border;
        }

        // Node size — rank-based in search mode, fixed otherwise
        var nodeSize;
        if (options.colorMode === 'search' && rankMap) {
          var rank2 = rankMap[e.absolute_id];
          nodeSize = rank2 === 1 ? 28
            : (rank2 <= 5 ? 22 : (rank2 <= 10 ? 18 : 14));
        } else {
          nodeSize = 14;
        }

        // Concept role-based border styling
        // entity: solid, relation: dashed, observation: dotted
        var borderDashes = false;
        var borderWidth = 1;
        if (e.role === 'relation') {
          borderDashes = [5, 5];
          borderWidth = Math.max(borderWidth, 1.5);
        } else if (e.role === 'observation' || e.role === ':Episode') {
          borderDashes = [2, 2];
          borderWidth = Math.max(borderWidth, 1.2);
        } else if (e.confidence && e.confidence < 0.7) {
          // Low confidence: thinner, more transparent border
          borderWidth = Math.max(borderWidth - 0.5, 0.5);
        }

        // Shape: 'dot' by default (label outside = entity name always visible)
        return {
          id: e.absolute_id,
          label: label,
          color: {
            background: bgColor,
            border: borderColor,
            highlight: { background: borderColor, border: '#a5b4fc' },
            hover: { background: borderColor, border: '#a5b4fc' },
          },
          borderWidth: borderWidth,
          borderWidthSelected: 2,
          borderDashes: borderDashes,
          size: nodeSize,
          shape: 'dot',
          font: {
            color: labelFontColor,
            size: 11,
            face: 'Inter, sans-serif',
          },
        };
      })
    );

    return { nodes: nodes, entityMap: entityMap, nodeIds: nodeIds };
  }

  // ---- Build edges ----
  //   relations: array of relation objects
  //   nodeIds: Set<absoluteId> — visible node IDs

  function buildEdges(relations, nodeIds) {
    var relationMap = {};

    var edges = new vis.DataSet(
      relations
        .filter(function (r) {
          return nodeIds.has(r.entity1_absolute_id) && nodeIds.has(r.entity2_absolute_id);
        })
        .map(function (r) {
          relationMap[r.absolute_id] = r;

          return {
            id: r.absolute_id,
            from: r.entity1_absolute_id,
            to: r.entity2_absolute_id,
            color: EDGE_DEFAULT,
            dashes: false,
            smooth: {
              enabled: true,
              type: 'continuous',
              roundness: 0.2,
            },
          };
        })
    );

    return { edges: edges, relationMap: relationMap };
  }

  // ---- Physics options ----

  function getPhysicsOptions() {
    return {
      enabled: true,
      solver: 'forceAtlas2Based',
      forceAtlas2Based: {
        gravitationalConstant: -150,
        centralGravity: 0.005,
        springLength: 160,
        springConstant: 0.04,
        damping: 0.6,
        avoidOverlap: 0.8,
      },
      stabilization: {
        enabled: true,
        iterations: 300,
        updateInterval: 25,
      },
    };
  }

  /** Gentle "swimming" physics — nodes drift slowly, dense clusters spread apart. */
  function getSwimmingPhysicsOptions(nodeCount) {
    var scale = nodeCount > 2000 ? 0.6 : nodeCount > 800 ? 0.8 : 1.0;
    return {
      enabled: true,
      solver: 'forceAtlas2Based',
      forceAtlas2Based: {
        gravitationalConstant: -50 * scale,
        centralGravity: 0.001,
        springLength: 220,
        springConstant: 0.012,
        damping: 0.92,
        avoidOverlap: 1.0,
      },
      timestep: 0.25,
      minVelocity: 0.1,
      stabilization: { enabled: false },
    };
  }

  // ---- Interaction options ----

  function getInteractionOptions() {
    return {
      hover: true,
      tooltipDelay: 0,
      hideTooltipOnDragMove: false,
      zoomView: true,
      dragView: true,
      navigationButtons: false,
      keyboard: false,
    };
  }

  // ---- Public API ----

  return {
    // Tier constants + search rank colors
    TIER_1: TIER_1,
    TIER_2: TIER_2,
    TIER_3: TIER_3,
    TIER_4: TIER_4,
    DEFAULT_LIGHT: DEFAULT_LIGHT,
    DEFAULT_DARK: DEFAULT_DARK,
    RANK_1: RANK_1,
    RANK_2_5: RANK_2_5,
    RANK_6_10: RANK_6_10,
    RANK_OTHER: RANK_OTHER,
    SEARCH_EXPANDED_LIGHT: SEARCH_EXPANDED_LIGHT,
    SEARCH_EXPANDED_DARK: SEARCH_EXPANDED_DARK,
    EDGE_DEFAULT: EDGE_DEFAULT,

    // Functions
    buildNodes: buildNodes,
    buildEdges: buildEdges,
    getPhysicsOptions: getPhysicsOptions,
    getSwimmingPhysicsOptions: getSwimmingPhysicsOptions,
    getInteractionOptions: getInteractionOptions,
  };
})();

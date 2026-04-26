/* ==========================================
   Graph Utilities — Shared vis-network helpers
   Used by graph.js, search.js, communities.js
   ========================================== */

window.GraphUtils = (function () {
  'use strict';

  // ---- Focus mode: type-based coloring with hop fading ----
  //   Focus (hop 0): Red
  //   Current: Blue node + dark solid edge, fades with hop
  //   Inherited (history): Orange node + orange dashed edge, fades with hop
  //   Future: Purple node + purple dotted edge, fades with hop

  // Node palettes — index 0 unused, 1-3 = hop levels (3+ clamped to 3)
  var FOCUS_NODE = { bg: '#ef4444', border: '#f87171' };

  var CURRENT_NODE = [
    null,
    { bg: '#3b82f6', border: '#60a5fa' },  // hop 1: blue-500
    { bg: '#93c5fd', border: '#bfdbfe' },  // hop 2: blue-300
    { bg: '#bfdbfe', border: '#dbeafe' },  // hop 3+: blue-200
  ];

  var INHERITED_NODE = [
    null,
    { bg: '#f97316', border: '#fb923c' },  // hop 1: orange-500
    { bg: '#fdba74', border: '#fed7aa' },  // hop 2: orange-300
    { bg: '#fed7aa', border: '#ffedd5' },  // hop 3+: orange-200
  ];

  var FUTURE_NODE = [
    null,
    { bg: '#a855f7', border: '#c084fc' },  // hop 1: purple-500
    { bg: '#d8b4fe', border: '#e9d5ff' },  // hop 2: purple-300
    { bg: '#e9d5ff', border: '#f3e8ff' },  // hop 3+: purple-200
  ];

  // Node sizes per hop level
  var NODE_SIZES = [25, 20, 16, 13];

  // Edge palettes — same hop fading
  var EDGE_CURRENT = [
    null,
    { color: '#2563eb', highlight: '#3b82f6', hover: '#1d4ed8' },  // hop 1: blue-600
    { color: '#60a5fa', highlight: '#93c5fd', hover: '#3b82f6' },  // hop 2: blue-400
    { color: '#93c5fd', highlight: '#bfdbfe', hover: '#60a5fa' },  // hop 3+: blue-300
  ];

  var EDGE_INHERITED = [
    null,
    { color: '#ea580c', highlight: '#f97316', hover: '#c2410c' },  // hop 1: orange-600
    { color: '#fb923c', highlight: '#fdba74', hover: '#f97316' },  // hop 2: orange-400
    { color: '#fdba74', highlight: '#fed7aa', hover: '#fb923c' },  // hop 3+: orange-300
  ];

  var EDGE_FUTURE = [
    null,
    { color: '#9333ea', highlight: '#a855f7', hover: '#7e22ce' },  // hop 1: purple-600
    { color: '#c084fc', highlight: '#d8b4fe', hover: '#a855f7' },  // hop 2: purple-400
    { color: '#d8b4fe', highlight: '#e9d5ff', hover: '#c084fc' },  // hop 3+: purple-300
  ];

  // Helper: pick palette entry by hop (clamped 1-3)
  function hopPalette(palette, hop) {
    return palette[Math.min(Math.max(hop || 1, 1), palette.length - 1)];
  }

  // ---- Legacy 4-tier color system (search, path-finder) ----
  //   Tier 1: Red    — Primary / Focus / #1
  //   Tier 2: Amber  — Secondary / #2~5
  //   Tier 3: Teal   — Tertiary  / #6~10
  //   Tier 4: Slate  — Low       / #11+

  var TIER_1    = { bg: '#ef4444', border: '#f87171' };
  var TIER_2    = { bg: '#f59e0b', border: '#fbbf24' };
  var TIER_3    = { bg: '#14b8a6', border: '#2dd4bf' };
  var TIER_4    = { bg: '#64748b', border: '#94a3b8' };

  var HOP_PALETTE = [TIER_1, TIER_2, TIER_3, TIER_4];

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

  // Hub color palette — 3 groups of saturated + light colors for top-3 hubs
  var HUB_PALETTE = [
    { bg: '#ef4444', border: '#f87171' },  // red
    { bg: '#3b82f6', border: '#60a5fa' },  // blue
    { bg: '#10b981', border: '#34d399' },  // green
  ];

  var HUB_NEIGHBOR_PALETTE = [
    { bg: '#fca5a5', border: '#f87171' },  // light red
    { bg: '#93c5fd', border: '#60a5fa' },  // light blue
    { bg: '#6ee7b7', border: '#34d399' },  // light green
  ];

  var HUB_EDGE_PALETTE = [
    { color: '#f87171', highlight: '#fca5a5', hover: '#ef4444' },  // red
    { color: '#60a5fa', highlight: '#93c5fd', hover: '#3b82f6' },  // blue
    { color: '#34d399', highlight: '#6ee7b7', hover: '#10b981' },  // green
  ];

  // Community color palette (20 distinct colors)
  var COMMUNITY_PALETTE = [
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
    { bg: '#ec4899', border: '#f472b6' }, // pink (was indigo)
    { bg: '#d946ef', border: '#e879f9' }, // fuchsia
    { bg: '#0ea5e9', border: '#38bdf8' }, // sky
    { bg: '#a855f7', border: '#c084fc' }, // purple
    { bg: '#e11d48', border: '#fb7185' }, // rose
    { bg: '#65a30d', border: '#84cc16' }, // green
    { bg: '#7c3aed', border: '#8b5cf6' }, // violet-dark
    { bg: '#0891b2', border: '#06b6d4' }, // teal-dark
    { bg: '#c2410c', border: '#ea580c' }, // orange-dark
    { bg: '#db2777', border: '#ec4899' }, // pink-dark
  ];

  function getRankColor(rank) {
    if (rank === 1) return TIER_1;
    if (rank <= 5) return TIER_2;
    if (rank <= 10) return TIER_3;
    return TIER_4;
  }

  // Node sizing: sqrt scaling for balanced visual hierarchy
  var NODE_MIN_SIZE = 14;
  var NODE_MAX_SIZE = 38;

  function computeNodeSize(relationCount, maxRelationCount) {
    if (!relationCount || relationCount === 0) return NODE_MIN_SIZE - 4; // isolated: smaller
    if (!maxRelationCount || maxRelationCount <= 0) return NODE_MIN_SIZE;
    // Square-root scaling: big nodes are clearly larger, but small nodes aren't invisible
    var ratio = Math.sqrt(relationCount / maxRelationCount);
    return NODE_MIN_SIZE + ratio * (NODE_MAX_SIZE - NODE_MIN_SIZE);
  }

  // ---- Legacy edge colors (non-focus-mode) ----

  var EDGE_DEFAULT  = { color: '#4b5563', highlight: '#9ca3af', hover: '#6b7280' };

  // ---- Theme detection ----

  function isLightTheme() {
    return document.documentElement.getAttribute('data-theme') === 'light';
  }

  // ---- Time-based color palette for knowledge age visualization ----
  //   Cold colors (blue/purple) = older knowledge
  //   Warm colors (orange/red) = newer knowledge
  //   Gradient based on processed_time relative to time range

  var TIME_PALETTE = [
    { bg: '#3b82f6', border: '#60a5fa' },  // oldest: blue-500
    { bg: '#8b5cf6', border: '#a78bfa' },  // older: purple-500
    { bg: '#06b6d4', border: '#22d3ee' },  // mid-old: cyan-500
    { bg: '#10b981', border: '#34d399' },  // mid: emerald-500
    { bg: '#f59e0b', border: '#fbbf24' },  // newer: amber-500
    { bg: '#ef4444', border: '#f87171' },  // newest: red-500
  ];

  function getTimeBasedColor(processedTime, minTime, maxTime) {
    if (!processedTime || minTime === maxTime) {
      return TIME_PALETTE[3]; // default to middle color
    }
    var time = new Date(processedTime).getTime();
    var normalizedTime = (time - minTime) / (maxTime - minTime);
    normalizedTime = Math.max(0, Math.min(1, normalizedTime));
    var idx = Math.floor(normalizedTime * (TIME_PALETTE.length - 1));
    return TIME_PALETTE[idx];
  }

  // ---- Build nodes ----
  //   entities: array of entity objects
  //   options:
  //     colorMode: 'hop' | 'search' | 'community' | 'default' | 'time'
  //     versionCounts: { family_id: count }
  //     hopMap: { absoluteId: hopLevel }          (for focus/search modes)
  //     highlightAbsId: string                    (focused entity id)
  //     rankMap: { absoluteId: 1-based-rank }     (for 'search' mode)
  //     communityMap: { absoluteId: communityId }  (for 'community' mode)
  //     versionLabel: { idx: number, total: number }
  //     unnamedLabel: string
  //     inheritedEntityIds: Set<absoluteId>        (focus mode: inherited entities)
  //     futureEntityIds: Set<absoluteId>           (focus mode: future entities)
  //     timeRange: { min: number, max: number }   (for 'time' mode: ms timestamps)

  function buildNodes(entities, options) {
    options = options || {};
    var light = isLightTheme();

    // Font colors
    var labelFontColor = light ? '#1e293b' : '#e2e8f0';
    var highlightFontColor = light ? '#1e40af' : '#ffffff';

    var versionCounts = options.versionCounts || {};
    var hopMap = options.hopMap || null;
    var highlightAbsId = options.highlightAbsId || null;
    var rankMap = options.rankMap || null;
    var communityMap = options.communityMap || null;
    var versionLabel = options.versionLabel || null;
    var unnamedLabel = options.unnamedLabel || 'unnamed';
    var inheritedEntityIds = options.inheritedEntityIds || null;
    var futureEntityIds = options.futureEntityIds || null;

    // Focus mode: type-based coloring with hop fading
    var hasTypeColoring = hopMap && (inheritedEntityIds || futureEntityIds);

    // Compute max relation count for linear node sizing
    var maxRelCount = 1;
    if (options.relationCounts) {
      var rcKeys = Object.keys(options.relationCounts);
      for (var rci = 0; rci < rcKeys.length; rci++) {
        if (options.relationCounts[rcKeys[rci]] > maxRelCount) {
          maxRelCount = options.relationCounts[rcKeys[rci]];
        }
      }
    }

    var entityMap = {};
    var nodeIds = new Set();

    var nodes = new vis.DataSet(
      entities.map(function (e) {
        entityMap[e.absolute_id] = e;
        nodeIds.add(e.absolute_id);

        var baseName = e.name || e.family_id || unnamedLabel;
        var isHighlight = highlightAbsId && e.absolute_id === highlightAbsId;
        var hopLevel = hopMap ? hopMap[e.absolute_id] : undefined;

        // Version count — needed early for label and styling decisions
        var vc = versionCounts[e.family_id] || 1;
        var isMultiVersion = vc > 1;

        // Label: entity name + version count [v2] for multi-version entities
        // In focus mode version-switching, show "2/3" inside the focused node
        var label = baseName;
        if (isMultiVersion) {
          label = baseName + ' [v' + vc + ']';
        }
        var showVersionInside = false;
        if (isHighlight && versionLabel && versionLabel.total > 1) {
          label = versionLabel.idx + '/' + versionLabel.total;
          showVersionInside = true;
        }

        // ---- Color selection ----
        var bgColor, borderColor;

        if (hasTypeColoring) {
          // Focus mode: type + hop based coloring
          if (isHighlight) {
            bgColor = FOCUS_NODE.bg;
            borderColor = FOCUS_NODE.border;
          } else {
            var isInh = inheritedEntityIds && inheritedEntityIds.has(e.absolute_id);
            var isFut = futureEntityIds && futureEntityIds.has(e.absolute_id);
            var h = hopLevel || 1;

            if (isFut) {
              var scheme = hopPalette(FUTURE_NODE, h);
              bgColor = scheme.bg;
              borderColor = scheme.border;
            } else if (isInh) {
              var scheme = hopPalette(INHERITED_NODE, h);
              bgColor = scheme.bg;
              borderColor = scheme.border;
            } else {
              var scheme = hopPalette(CURRENT_NODE, h);
              bgColor = scheme.bg;
              borderColor = scheme.border;
            }
          }
        } else if (options.colorMode === 'hop' && hopMap && hopLevel !== undefined) {
          // Legacy hop coloring (non-focus)
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
        } else if (options.colorMode === 'hub' && options.hubMap) {
          var hubIdx = options.hubMap[e.absolute_id];
          var isNeighbor = options.hubNeighborIds && options.hubNeighborIds.has(e.absolute_id);
          if (hubIdx !== undefined) {
            if (isNeighbor) {
              var scheme = HUB_NEIGHBOR_PALETTE[hubIdx];
              bgColor = scheme.bg;
              borderColor = scheme.border;
            } else {
              var scheme = HUB_PALETTE[hubIdx];
              bgColor = scheme.bg;
              borderColor = scheme.border;
            }
          } else {
            var defaultColor = light ? DEFAULT_LIGHT : DEFAULT_DARK;
            bgColor = defaultColor.bg;
            borderColor = defaultColor.border;
          }
        } else if (options.colorMode === 'time' && options.timeRange) {
          // Time-based coloring: cold (old) -> warm (new)
          var timeColor = getTimeBasedColor(e.processed_time, options.timeRange.min, options.timeRange.max);
          bgColor = timeColor.bg;
          borderColor = timeColor.border;
        } else {
          var defaultColor = light ? DEFAULT_LIGHT : DEFAULT_DARK;
          bgColor = defaultColor.bg;
          borderColor = defaultColor.border;
        }

        // Node size — linear scaling by relation count
        var relationCount = options.relationCounts ? (options.relationCounts[e.absolute_id] || 0) : 0;
        var nodeSize;
        if (hasTypeColoring) {
          nodeSize = isHighlight ? 30 : NODE_SIZES[Math.min(hopLevel || 1, NODE_SIZES.length - 1)];
        } else if (options.colorMode === 'search' && rankMap) {
          nodeSize = rankMap[e.absolute_id] === 1 ? 28
            : (rankMap[e.absolute_id] <= 5 ? 22 : (rankMap[e.absolute_id] <= 10 ? 18 : 14));
        } else {
          nodeSize = computeNodeSize(relationCount, maxRelCount);
        }

        // Version badge: gold border + glow for multi-version entities
        var borderWidth = 1;
        var shadow = { enabled: false };
        var versionBorderColor = borderColor;
        if (isMultiVersion) {
          borderWidth = 1.5 + Math.min(vc * 0.3, 2.5);
          versionBorderColor = light ? '#d97706' : '#fbbf24'; // amber-600 / amber-400
          shadow = {
            enabled: true,
            color: 'rgba(251, 191, 36, 0.3)',
            size: Math.min(4 + vc * 1, 14),
            x: 0,
            y: 0,
          };
        }

        // Concept role-based border styling
        // entity: solid, relation: dashed, observation: dotted
        var borderDashes = false;
        var finalBorderWidth = borderWidth;
        if (e.role === 'relation') {
          borderDashes = [5, 5];
          finalBorderWidth = Math.max(borderWidth, 1.5);
        } else if (e.role === 'observation' || e.role === ':Episode') {
          borderDashes = [2, 2];
          finalBorderWidth = Math.max(borderWidth, 1.2);
        } else if (e.confidence && e.confidence < 0.7) {
          // Low confidence: thinner, more transparent border
          finalBorderWidth = Math.max(borderWidth - 0.5, 0.5);
        }

        var nodeFontColor = isHighlight ? highlightFontColor : labelFontColor;

        // Shape: 'dot' by default (label outside = entity name always visible)
        // Only focused entity in version-switch mode uses 'circle' (version label inside)
        var nodeShape = 'dot';
        var borderWidthSelected = 2;
        var nodeFontSize = isHighlight ? 12 : 11;

        if (showVersionInside) {
          nodeShape = 'circle';
          borderWidthSelected = 3;
          nodeFontSize = Math.max(10, Math.min(14, nodeSize * 0.5));
          nodeFontColor = light ? '#1e293b' : '#ffffff';
        }

        // Build rich HTML tooltip
        var tooltipHtml = '<div style="max-width:280px;padding:4px 0;">';
        tooltipHtml += '<div style="font-weight:600;font-size:0.85rem;margin-bottom:4px;">' + (typeof escapeHtml !== 'undefined' ? escapeHtml(baseName) : baseName) + '</div>';
        if (isMultiVersion) {
          tooltipHtml += '<div style="font-size:0.75rem;color:#d97706;margin-bottom:4px;">⏱ ' + vc + ' versions</div>';
        }
        if (e.content) {
          var contentPreview = e.content.substring(0, 150).replace(/\n/g, ' ');
          if (e.content.length > 150) contentPreview += '...';
          tooltipHtml += '<div style="font-size:0.75rem;color:#64748b;line-height:1.4;">' + (typeof escapeHtml !== 'undefined' ? escapeHtml(contentPreview) : contentPreview) + '</div>';
        }
        if (e.processed_time) {
          var dateStr = new Date(e.processed_time).toLocaleDateString();
          tooltipHtml += '<div style="font-size:0.6875rem;color:#94a3b8;margin-top:4px;">📅 ' + dateStr + '</div>';
        }
        tooltipHtml += '</div>';

        return {
          id: e.absolute_id,
          label: label,
          // title: tooltipHtml,  // Disabled: custom hover panel (graph-explorer.js) replaces vis-network tooltip
          color: {
            background: bgColor,
            border: isMultiVersion ? versionBorderColor : borderColor,
            highlight: { background: isMultiVersion ? versionBorderColor : borderColor, border: '#a5b4fc' },
            hover: { background: isMultiVersion ? versionBorderColor : borderColor, border: '#a5b4fc' },
          },
          borderWidth: finalBorderWidth,
          borderWidthSelected: borderWidthSelected,
          borderDashes: borderDashes,
          shadow: shadow,
          size: nodeSize,
          shape: nodeShape,
          font: {
            color: nodeFontColor,
            size: nodeFontSize,
            face: 'Inter, sans-serif',
            bold: isHighlight || isMultiVersion ? { color: nodeFontColor, size: nodeFontSize, face: 'Inter, sans-serif' } : undefined,
          },
          // Custom metadata
          _isMultiVersion: isMultiVersion,
          _baseColor: bgColor,
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
  //     futureRelationIds: Set<absoluteId>
  //     hopMap: { absoluteId: hopLevel }       (for focus mode hop fading)
  //     weightMode: 'count' | null

  function buildEdges(relations, nodeIds, options) {
    options = options || {};
    var inheritedRelationIds = options.inheritedRelationIds || null;
    var hasInherited = inheritedRelationIds && inheritedRelationIds.size > 0;
    var futureRelationIds = options.futureRelationIds || null;
    var hasFuture = futureRelationIds && futureRelationIds.size > 0;
    var hopMap = options.hopMap || null;
    var weightMode = options.weightMode || null;

    // Focus mode: type-based edge coloring with hop fading
    var hasTypeColoring = hopMap && (hasInherited || hasFuture);

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

          var edgeColor, dashes;

          if (hasTypeColoring) {
            // Compute edge hop from endpoints
            var h1 = hopMap[r.entity1_absolute_id];
            var h2 = hopMap[r.entity2_absolute_id];
            var edgeHop = Math.max(h1 || 1, h2 || 1);
            edgeHop = Math.min(edgeHop, 3);

            if (isFuture) {
              edgeColor = EDGE_FUTURE[edgeHop];
              dashes = [2, 4];
            } else if (isInherited) {
              edgeColor = EDGE_INHERITED[edgeHop];
              dashes = [5, 5];
            } else {
              edgeColor = EDGE_CURRENT[edgeHop];
              dashes = false;
            }
          } else {
            // Non-focus mode: simple inherited/future colors
            if (isFuture) {
              edgeColor = EDGE_FUTURE[1];
              dashes = [2, 4];
            } else if (isInherited) {
              edgeColor = EDGE_INHERITED[1];
              dashes = [5, 5];
            } else if (options.hubMap) {
              var hh1 = options.hubMap[r.entity1_absolute_id];
              var hh2 = options.hubMap[r.entity2_absolute_id];
              if (hh1 !== undefined && hh2 !== undefined && hh1 === hh2) {
                edgeColor = HUB_EDGE_PALETTE[hh1];
              } else {
                edgeColor = EDGE_DEFAULT;
              }
              dashes = false;
            } else {
              edgeColor = EDGE_DEFAULT;
              dashes = false;
            }
          }

          // Edge tooltip handled by custom hover panel in graph-explorer.js
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
        iterations: 300,
        updateInterval: 25,
      },
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
    // Focus mode type-based palettes
    FOCUS_NODE: FOCUS_NODE,
    CURRENT_NODE: CURRENT_NODE,
    INHERITED_NODE: INHERITED_NODE,
    FUTURE_NODE: FUTURE_NODE,
    EDGE_CURRENT: EDGE_CURRENT,
    EDGE_INHERITED: EDGE_INHERITED,
    EDGE_FUTURE: EDGE_FUTURE,

    // Legacy tier constants
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
    HUB_PALETTE: HUB_PALETTE,
    HUB_NEIGHBOR_PALETTE: HUB_NEIGHBOR_PALETTE,
    HUB_EDGE_PALETTE: HUB_EDGE_PALETTE,
    EDGE_DEFAULT: EDGE_DEFAULT,
    TIME_PALETTE: TIME_PALETTE,

    // Functions
    buildNodes: buildNodes,
    buildEdges: buildEdges,
    getPhysicsOptions: getPhysicsOptions,
    getInteractionOptions: getInteractionOptions,
    computeNodeSize: computeNodeSize,
    getTimeBasedColor: getTimeBasedColor,
    NODE_MIN_SIZE: NODE_MIN_SIZE,
    NODE_MAX_SIZE: NODE_MAX_SIZE,
  };
})();

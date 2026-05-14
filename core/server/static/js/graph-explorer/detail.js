/* ==========================================
   GraphExplorer — Detail sidebar rendering
   Entity detail, relation detail, hover panels
   ========================================== */

window.GraphExplorerDetail = (function () {
  'use strict';

  var escapeHtml = window.escapeHtml || function escapeHtml(text) {
    var div = document.createElement('div');
    div.appendChild(document.createTextNode(text));
    return div.innerHTML;
  };

  // ---- Node hover info panel ----

  function createHoverPanel(canvasId, network) {
    var container = document.getElementById(canvasId);
    if (!container) return null;
    var panel = document.createElement('div');
    panel.className = 'node-hover-info';
    panel.style.opacity = '0';
    container.appendChild(panel);
    return panel;
  }

  function showNodeHover(opts) {
    var nodeId = opts.nodeId;
    var network = opts.network;
    var entityMap = opts.entityMap;
    var versionCounts = opts.versionCounts;
    var canvasId = opts.canvasId;
    var hoverPanel = opts.hoverPanel;
    var setHoverPanel = opts.setHoverPanel;
    var setHoverNodeId = opts.setHoverNodeId;

    if (!network) return hoverPanel;
    var entity = entityMap[nodeId];
    if (!entity) return hoverPanel;

    var container = document.getElementById(canvasId);
    if (!container) return hoverPanel;

    // Create panel if needed (also recreate if detached from DOM by vis-network rebuild)
    if (!hoverPanel || !hoverPanel.parentElement) {
      hoverPanel = createHoverPanel(canvasId, network);
      if (setHoverPanel) setHoverPanel(hoverPanel);
    }

    // Build content
    var name = entity.name || entity.family_id || nodeId;
    var vc = versionCounts[entity.family_id] || versionCounts[nodeId] || 0;
    var summary = entity.summary || '';
    var content = entity.content || '';

    var html = '<div class="nhv-name">' + escapeHtml(name);
    if (vc > 1) {
      html += ' <span class="nhv-version">[v' + vc + ']</span>';
    }
    html += '</div>';
    // Prefer summary for preview, fall back to content
    var preview = summary || content || '';
    if (preview) {
      if (preview.length > 150) preview = preview.substring(0, 150) + '...';
      html += '<div class="nhv-content">' + escapeHtml(preview) + '</div>';
    }
    if (entity.processed_time) {
      html += '<div style="font-size:0.6875rem;color:var(--text-muted);margin-top:0.25rem;">' + formatDate(entity.processed_time) + '</div>';
    }

    hoverPanel.innerHTML = html;
    if (setHoverNodeId) setHoverNodeId(nodeId);

    // Position near the node
    updateNodeHoverPosition({
      hoverPanel: hoverPanel,
      hoverNodeId: nodeId,
      network: network,
      canvasId: canvasId,
    });

    // Show with slight delay for smooth appearance
    requestAnimationFrame(function () {
      if (hoverPanel) hoverPanel.style.opacity = '1';
    });

    return hoverPanel;
  }

  function hideNodeHover(hoverPanel, setHoverNodeId) {
    if (hoverPanel) {
      hoverPanel.style.opacity = '0';
    }
    if (setHoverNodeId) setHoverNodeId(null);
  }

  function updateNodeHoverPosition(opts) {
    var hoverPanel = opts.hoverPanel;
    var hoverNodeId = opts.hoverNodeId;
    var network = opts.network;
    var canvasId = opts.canvasId;

    if (!hoverPanel || !hoverNodeId || !network) return;

    var container = document.getElementById(canvasId);
    if (!container) return;

    // Edge hover — reposition at edge midpoint
    if (hoverNodeId.indexOf('__edge__') === 0) {
      var edgeId = hoverNodeId.replace('__edge__', '');
      var edgeEnds = network.getConnectedNodes(edgeId);
      if (edgeEnds && edgeEnds.length === 2) {
        var positions = network.getPositions(edgeEnds);
        var p1 = positions[edgeEnds[0]];
        var p2 = positions[edgeEnds[1]];
        if (p1 && p2) {
          var midCanvas = { x: (p1.x + p2.x) / 2, y: (p1.y + p2.y) / 2 };
          var domPos = network.canvasToDOM(midCanvas);
          hoverPanel.style.left = domPos.x + 'px';
          hoverPanel.style.top = (domPos.y - 30) + 'px';
        }
      }
      return;
    }

    var canvasPos = network.getPositions([hoverNodeId]);
    if (!canvasPos[hoverNodeId]) return;

    var domPos = network.canvasToDOM({ x: canvasPos[hoverNodeId].x, y: canvasPos[hoverNodeId].y });

    // Get node size to offset positioning
    var node = network.body.nodes[hoverNodeId];
    var nodeSize = node ? (node.options.size || 20) : 20;

    var left = domPos.x + nodeSize + 12;
    var top = domPos.y - 20;

    // Keep panel within container bounds
    var panelRect = hoverPanel.getBoundingClientRect();
    var containerRect = container.getBoundingClientRect();
    if (left + 240 > containerRect.width) {
      left = domPos.x - nodeSize - 252;
    }
    if (top + 100 > containerRect.height) {
      top = containerRect.height - 110;
    }
    if (top < 10) top = 10;
    if (left < 10) left = 10;

    hoverPanel.style.left = left + 'px';
    hoverPanel.style.top = top + 'px';
  }

  function showEdgeHover(opts) {
    var edgeId = opts.edgeId;
    var network = opts.network;
    var relationMap = opts.relationMap;
    var entityMap = opts.entityMap;
    var canvasId = opts.canvasId;
    var hoverPanel = opts.hoverPanel;
    var setHoverPanel = opts.setHoverPanel;
    var setHoverNodeId = opts.setHoverNodeId;

    if (!network) return hoverPanel;
    var relation = relationMap[edgeId];
    if (!relation) return hoverPanel;

    var container = document.getElementById(canvasId);
    if (!container) return hoverPanel;

    // Reuse or create panel (also recreate if detached from DOM)
    if (!hoverPanel || !hoverPanel.parentElement) {
      hoverPanel = createHoverPanel(canvasId, network);
      if (setHoverPanel) setHoverPanel(hoverPanel);
    }

    // Look up endpoint entity names
    var e1Name = '';
    var e2Name = '';
    if (relation.entity1_absolute_id && entityMap[relation.entity1_absolute_id]) {
      e1Name = entityMap[relation.entity1_absolute_id].name || '';
    }
    if (relation.entity2_absolute_id && entityMap[relation.entity2_absolute_id]) {
      e2Name = entityMap[relation.entity2_absolute_id].name || '';
    }

    var html = '';
    if (e1Name || e2Name) {
      html += '<div class="nhv-name" style="font-size:0.75rem;">' + escapeHtml(e1Name || '?') + ' <span style="color:var(--text-muted);margin:0 0.25rem;">&harr;</span> ' + escapeHtml(e2Name || '?') + '</div>';
    }
    var content = relation.content || relation.summary || '';
    if (content) {
      var preview = content.length > 150 ? content.substring(0, 150) + '...' : content;
      html += '<div class="nhv-content" style="margin-top:0.25rem;">' + escapeHtml(preview) + '</div>';
    }
    if (relation.processed_time) {
      html += '<div style="font-size:0.6875rem;color:var(--text-muted);margin-top:0.25rem;">' + formatDate(relation.processed_time) + '</div>';
    }

    if (!html) return hoverPanel; // Don't show empty panel

    hoverPanel.innerHTML = html;
    if (setHoverNodeId) setHoverNodeId('__edge__' + edgeId);

    // Position at edge midpoint
    var edgeEnds = network.getConnectedNodes(edgeId);
    if (edgeEnds && edgeEnds.length === 2) {
      var positions = network.getPositions(edgeEnds);
      var p1 = positions[edgeEnds[0]];
      var p2 = positions[edgeEnds[1]];
      if (p1 && p2) {
        var midCanvas = { x: (p1.x + p2.x) / 2, y: (p1.y + p2.y) / 2 };
        var domPos = network.canvasToDOM(midCanvas);
        hoverPanel.style.left = domPos.x + 'px';
        hoverPanel.style.top = (domPos.y - 30) + 'px';
      }
    }

    requestAnimationFrame(function () {
      if (hoverPanel) hoverPanel.style.opacity = '1';
    });

    return hoverPanel;
  }

  return {
    createHoverPanel: createHoverPanel,
    showNodeHover: showNodeHover,
    hideNodeHover: hideNodeHover,
    updateNodeHoverPosition: updateNodeHoverPosition,
    showEdgeHover: showEdgeHover,
  };
})();

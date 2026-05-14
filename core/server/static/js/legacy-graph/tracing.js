// legacy-graph module: tracing.js
// Source tracing functions

// Search focus enhancement
function enhanceSearchFocus(matchedEntities, matchedRelations) {
    searchFocusState.matchedNodes.clear();
    searchFocusState.matchedEdges.clear();

    // Collect matched node IDs
    if (matchedEntities) {
        matchedEntities.forEach(function(entity) {
            searchFocusState.matchedNodes.add(entity.family_id);
        });
    }

    // Collect matched edge IDs
    if (matchedRelations) {
        matchedRelations.forEach(function(relation) {
            searchFocusState.matchedEdges.add(relation.family_id);
        });
    }

    // Apply visual effects
    if (nodesDataSet) {
        nodesDataSet.get().forEach(function(node) {
            var isMatched = searchFocusState.matchedNodes.has(node.id);
            var update = {
                size: isMatched ? 30 : 20,
                opacity: isMatched ? 1 : 0.4
            };
            nodesDataSet.update(node.id, update);
        });
    }

    if (edges) {
        edges.get().forEach(function(edge) {
            var isMatched = searchFocusState.matchedEdges.has(edge.family_id);
            var update = {
                width: isMatched ? 4 : 2,
                opacity: isMatched ? 1 : 0.4
            };
            edges.update(edge.id, update);
        });
    }
}

function clearSearchFocus() {
    searchFocusState.matchedNodes.clear();
    searchFocusState.matchedEdges.clear();

    if (nodesDataSet) {
        nodesDataSet.get().forEach(function(node) {
            nodesDataSet.update(node.id, {
                size: 20,
                opacity: 1
            });
        });
    }

    if (edges) {
        edges.get().forEach(function(edge) {
            edges.update(edge.id, {
                width: 2,
                opacity: 1
            });
        });
    }
}

// Source Tracing Functions
function toggleSourceTracing() {
    var toggle = document.getElementById('source-tracing-toggle');
    var info = document.getElementById('source-tracing-info');

    sourceTracingState.enabled = !sourceTracingState.enabled;
    toggle.classList.toggle('active', sourceTracingState.enabled);

    if (sourceTracingState.enabled) {
        info.textContent = '溯源模式已启用 - 点击节点查看其溯源路径';
        info.style.color = '#f39c12';
    } else {
        info.textContent = '点击节点查看其溯源路径（连接到源观察文本的所有路径）';
        info.style.color = '#888888';
        clearSourceTracing();
    }
}

function performSourceTracing(nodeId) {
    if (!sourceTracingState.enabled) return;

    console.log('开始溯源节点:', nodeId);

    // Clear previous tracing
    clearSourceTracing();

    // Use BFS to find all paths to source observation nodes
    var paths = findPathsToSources(nodeId);

    // Highlight the paths
    highlightSourcePaths(paths);

    // Update info
    var info = document.getElementById('source-tracing-info');
    info.textContent = '找到 ' + paths.length + ' 条溯源路径，显示 ' + sourceTracingState.pathNodes.size + ' 个节点';
}

function findPathsToSources(startNodeId) {
    var paths = [];
    var visited = new Set();
    var queue = [[startNodeId, [startNodeId]]];

    while (queue.length > 0) {
        var [currentId, path] = queue.shift();

        if (visited.has(currentId)) continue;
        visited.add(currentId);

        // Check if this is a source observation node
        var node = nodesDataSet.get(currentId);
        if (node && node.entity_type === 'observation') {
            paths.push(path);
            // Continue to find other paths
        }

        // Explore connected nodes
        var connectedEdges = edges.get().filter(function(edge) {
            return edge.from === currentId || edge.to === currentId;
        });

        connectedEdges.forEach(function(edge) {
            var nextId = edge.from === currentId ? edge.to : edge.from;
            if (!visited.has(nextId)) {
                queue.push([nextId, [...path, nextId]]);
            }
        });
    }

    return paths;
}

function highlightSourcePaths(paths) {
    // Collect all nodes and edges in paths
    sourceTracingState.pathNodes.clear();
    sourceTracingState.pathEdges.clear();

    paths.forEach(function(path) {
        for (var i = 0; i < path.length; i++) {
            sourceTracingState.pathNodes.add(path[i]);

            if (i < path.length - 1) {
                // Find the edge between consecutive nodes
                var edge = edges.get().find(function(e) {
                    return (e.from === path[i] && e.to === path[i + 1]) ||
                           (e.to === path[i] && e.from === path[i + 1]);
                });
                if (edge) {
                    sourceTracingState.pathEdges.add(edge.id);
                }
            }
        }
    });

    // Apply highlighting
    sourceTracingState.pathNodes.forEach(function(nodeId) {
        nodesDataSet.update(nodeId, {
            color: '#f39c12',
            borderWidth: 3,
            opacity: 1
        });
    });

    sourceTracingState.pathEdges.forEach(function(edgeId) {
        edges.update(edgeId, {
            color: '#f39c12',
            width: 4,
            dashes: true,
            opacity: 1
        });
    });

    // Dim non-path nodes and edges
    nodesDataSet.get().forEach(function(node) {
        if (!sourceTracingState.pathNodes.has(node.id)) {
            nodesDataSet.update(node.id, {
                opacity: 0.3
            });
        }
    });

    edges.get().forEach(function(edge) {
        if (!sourceTracingState.pathEdges.has(edge.id)) {
            edges.update(edge.id, {
                opacity: 0.3
            });
        }
    });
}

function clearSourceTracing() {
    // Reset all highlighted nodes
    sourceTracingState.pathNodes.forEach(function(nodeId) {
        var node = nodesDataSet.get(nodeId);
        if (node) {
            var update = {
                opacity: 1
            };
            // Restore original color if available
            if (node.processed_time) {
                update.color = getTimeBasedColor(new Date(node.processed_time));
            } else {
                update.color = '#4A90E2';
            }
            nodesDataSet.update(nodeId, update);
        }
    });

    // Reset all highlighted edges
    sourceTracingState.pathEdges.forEach(function(edgeId) {
        edges.update(edgeId, {
            color: '#888888',
            width: 2,
            dashes: false,
            opacity: 1
        });
    });

    // Reset all dimmed nodes
    nodesDataSet.get().forEach(function(node) {
        if (!sourceTracingState.pathNodes.has(node.id) && node.opacity === 0.3) {
            nodesDataSet.update(node.id, { opacity: 1 });
        }
    });

    // Reset all dimmed edges
    edges.get().forEach(function(edge) {
        if (!sourceTracingState.pathEdges.has(edge.id) && edge.opacity === 0.3) {
            edges.update(edge.id, { opacity: 1 });
        }
    });

    sourceTracingState.pathNodes.clear();
    sourceTracingState.pathEdges.clear();

    // Update info
    var info = document.getElementById('source-tracing-info');
    if (sourceTracingState.enabled) {
        info.textContent = '点击节点查看其溯源路径（连接到源观察文本的所有路径）';
    }
}

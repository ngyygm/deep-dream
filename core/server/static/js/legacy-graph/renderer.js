// legacy-graph module: renderer.js
// Graph rendering, UI functions, search, and sidebar display

async function loadGraph() {
    // 从输入框获取参数
    var limitEntities = parseInt(document.getElementById('limit-entities-input').value) || 100;
    var limitEdgesPerEntity = parseInt(document.getElementById('limit-edges-input').value) || 50;
    var hops = parseInt(document.getElementById('hops-input').value) || 1;
    var graphPath = document.getElementById('graph-path-input').value.trim();

    // 验证输入
    if (limitEntities < 1) {
        alert('实体数量必须大于等于1');
        return;
    }
    if (limitEdgesPerEntity < 1) {
        alert('每实体边数必须大于等于1');
        return;
    }
    if (hops < 1) {
        alert('跳数必须大于等于1');
        return;
    }

    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    console.log('[1/4] 开始加载图谱数据（默认模式）...');
    console.log('   实体数量限制:', limitEntities);
    console.log('   每实体边数限制:', limitEdgesPerEntity);
    console.log('   跳数:', hops);
    console.log('   图谱路径:', graphPath || '(使用默认路径)');

    // 构建请求URL
    var url = '/api/graphs/data?limit_entities=' + limitEntities + '&limit_edges_per_entity=' + limitEdgesPerEntity + '&hops=' + hops;
    if (graphPath) {
        url += '&storage_path=' + encodeURIComponent(graphPath);
    }
    console.log('   请求URL:', url);

    currentMode = 'default';
    currentSearchQuery = '';
    updateSearchStatus('');
    document.getElementById('clear-search-btn').style.display = 'none';
    document.getElementById('search-input').value = '';

    try {
        var response = await fetch(url);
        console.log('[2/4] 收到HTTP响应');
        console.log('   状态码:', response.status);
        console.log('   状态文本:', response.statusText);
        console.log('   Content-Type:', response.headers.get('content-type'));

        if (!response.ok) {
            throw new Error('HTTP错误: ' + response.status + ' ' + response.statusText);
        }

        var data = await response.json();
        console.log('[3/4] JSON数据解析完成');
        console.log('   数据键:', Object.keys(data));

        if (data.success) {
            console.log('API返回成功');
            console.log('   节点数量:', data.nodes ? data.nodes.length : 0);
            console.log('   边数量:', data.edges ? data.edges.length : 0);
            console.log('   统计信息:', data.stats);

            // 显示前几个节点的信息
            if (data.nodes && data.nodes.length > 0) {
                console.log('   前3个节点示例:');
                data.nodes.slice(0, 3).forEach(function(node, index) {
                    console.log('     [' + index + ']', {
                        id: node.id,
                        label: node.label,
                        hasEntityId: !!node.entity_id,
                        hasAbsoluteId: !!node.absolute_id
                    });
                });
            }

            // 显示前几条边的信息
            if (data.edges && data.edges.length > 0) {
                console.log('   前3条边示例:');
                data.edges.slice(0, 3).forEach(function(edge, index) {
                    console.log('     [' + index + ']', {
                        from: edge.from,
                        to: edge.to,
                        hasRelationId: !!edge.relation_id,
                        hasAbsoluteId: !!edge.absolute_id
                    });
                });
            }

            updateStats(data.stats);
            console.log('[4/4] 开始绘制图谱...');
            drawGraph(data.nodes, data.edges);
            updateTimelineData(data.nodes);
            updateLastUpdate();
            console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
        } else {
            console.error('API返回失败');
            console.error('   错误信息:', data.error);
            alert('加载图谱数据失败: ' + data.error);
        }
    } catch (error) {
        console.error('请求过程中发生错误');
        console.error('   错误类型:', error.name);
        console.error('   错误消息:', error.message);
        console.error('   错误堆栈:', error.stack);
        alert('请求错误: ' + error.message);
    }
}

async function searchGraph() {
    var query = document.getElementById('search-input').value.trim();
    var maxResultsInput = document.getElementById('search-count-input').value;
    var maxResults = parseInt(maxResultsInput);
    var graphPath = document.getElementById('graph-path-input').value.trim();

    if (!query) {
        alert('请输入搜索查询');
        return;
    }

    // 验证结果数量
    if (isNaN(maxResults) || maxResults < 1) {
        alert('结果数量必须是大于等于1的整数');
        document.getElementById('search-count-input').focus();
        return;
    }

    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    console.log('[1/4] 开始搜索图谱...');
    console.log('   查询文本:', query);
    console.log('   最大结果数:', maxResults);
    console.log('   图谱路径:', graphPath || '(使用默认路径)');
    console.log('   请求URL: /api/graphs/search');

    currentMode = 'search';
    currentSearchQuery = query;
    updateSearchStatus('正在搜索...');
    document.getElementById('clear-search-btn').style.display = 'inline-block';
    document.getElementById('search-btn').disabled = true;

    // 构建请求体
    var requestBody = {
        query: query,
        max_results: maxResults,
        limit_edges_per_entity: 50
    };
    if (graphPath) {
        requestBody.storage_path = graphPath;
    }

    try {
        var response = await fetch('/api/graphs/search', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestBody)
        });
        console.log('[2/4] 收到HTTP响应');
        console.log('   状态码:', response.status);

        if (!response.ok) {
            throw new Error('HTTP错误: ' + response.status + ' ' + response.statusText);
        }

        var data = await response.json();
        console.log('[3/4] JSON数据解析完成');
        console.log('   完整响应数据:', JSON.stringify(data, null, 2));
        console.log('   数据键:', Object.keys(data));

        document.getElementById('search-btn').disabled = false;

        if (data.success) {
            console.log('搜索成功');
            console.log('   匹配实体数:', data.stats ? data.stats.matched_entities : 0);
            console.log('   节点数量:', data.nodes ? data.nodes.length : 0);
            console.log('   边数量:', data.edges ? data.edges.length : 0);
            console.log('   统计信息:', data.stats);
            console.log('   查询文本:', data.query);

            // 检查是否有节点数据
            if (!data.nodes || data.nodes.length === 0) {
                console.warn('搜索结果中没有节点数据');
                updateSearchStatus('未找到匹配的实体，请尝试其他查询词');
                alert('未找到匹配的实体，请尝试其他查询词');
                return;
            }

            if (data.stats && data.stats.matched_entities > 0) {
                updateSearchStatus('找到 ' + data.stats.matched_entities + ' 个匹配实体，共显示 ' + data.stats.total_entities + ' 个实体（包含关联实体）');
            } else {
                updateSearchStatus('未找到匹配的实体');
            }

            updateStats(data.stats);
            console.log('[4/4] 开始绘制搜索结果图谱...');
            console.log('   准备绘制的节点数:', data.nodes.length);
            console.log('   准备绘制的边数:', data.edges.length);
            drawGraph(data.nodes, data.edges);

            // Apply search focus enhancement
            enhanceSearchFocus(data.matched_entities, data.matched_relations);

            updateLastUpdate();
            console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
        } else {
            console.error('搜索失败');
            console.error('   错误信息:', data.error);
            updateSearchStatus('搜索失败: ' + data.error);
            alert('搜索失败: ' + data.error);
        }
    } catch (error) {
        console.error('搜索过程中发生错误');
        console.error('   错误类型:', error.name);
        console.error('   错误消息:', error.message);
        console.error('   错误堆栈:', error.stack);
        document.getElementById('search-btn').disabled = false;
        updateSearchStatus('搜索错误: ' + error.message);
        alert('搜索错误: ' + error.message);
    }
}

function clearSearch() {
    console.log('清除搜索，返回默认视图');
    document.getElementById('search-input').value = '';
    document.getElementById('clear-search-btn').style.display = 'none';
    updateSearchStatus('');
    clearSearchFocus();
    loadGraph();
}

function handleSearchKeyPress(event) {
    if (event.key === 'Enter') {
        searchGraph();
    }
}

function drawGraph(nodesData, edgesData) {
    try {
        console.log('drawGraph 函数开始执行');
        console.log('输入数据检查:');
        console.log('   节点数据类型:', typeof nodesData, Array.isArray(nodesData) ? '(数组)' : '(非数组)');
        console.log('   节点数据长度:', nodesData ? nodesData.length : 'null/undefined');
        console.log('   边数据类型:', typeof edgesData, Array.isArray(edgesData) ? '(数组)' : '(非数组)');
        console.log('   边数据长度:', edgesData ? edgesData.length : 'null/undefined');

        if (!nodesData || nodesData.length === 0) {
            console.error('节点数据为空或无效');
            alert('没有节点数据，无法绘制图谱');
            return;
        }

        if (!container) {
            console.error('容器元素不存在');
            alert('找不到图谱容器元素');
            return;
        }
        console.log('容器元素检查通过');

        // 如果已存在网络，先清除旧数据
        if (network) {
            console.log('清除现有网络数据...');
            network.destroy();
            network = null;
            nodesDataSet = null;
        }

        // 创建节点数据集
        console.log('步骤1: 创建节点数据集...');
        try {
            // 去重：确保每个节点ID只出现一次
            var uniqueNodes = [];
            var seenNodeIds = new Set();
            var allTimes = [];

            for (var i = 0; i < nodesData.length; i++) {
                var node = nodesData[i];
                if (!seenNodeIds.has(node.id)) {
                    seenNodeIds.add(node.id);

                    // Apply time-based coloring
                    if (node.processed_time) {
                        var nodeTime = new Date(node.processed_time);
                        node.color = getTimeBasedColor(nodeTime);
                        allTimes.push(nodeTime);
                    }

                    // Apply multi-dimension encoding
                    // Size based on importance (degree)
                    node.size = node.size || 20;

                    // Border style based on type (if available)
                    if (node.entity_type) {
                        switch(node.entity_type) {
                            case 'observation':
                                node.shapeProperties = { borderDashes: false };
                                node.borderWidth = 3;
                                break;
                            case 'entity':
                                node.shapeProperties = { borderDashes: true };
                                node.borderWidth = 2;
                                break;
                            case 'relation':
                                node.shapeProperties = { borderDashes: [5, 5] };
                                node.borderWidth = 1;
                                break;
                        }
                    }

                    // Opacity based on confidence score (if available)
                    if (node.confidence !== undefined) {
                        // Map confidence (0-1) to opacity (0.3-1.0)
                        node.opacity = 0.3 + (node.confidence * 0.7);
                    }

                    // Glow effect for recently active nodes (modified within last 24 hours)
                    if (node.processed_time) {
                        var nodeTime = new Date(node.processed_time);
                        var now = new Date();
                        var hoursSinceModified = (now - nodeTime) / (1000 * 60 * 60);
                        if (hoursSinceModified < 24) {
                            node.title = (node.title || node.label) + ' [最近活跃]';
                            // Note: vis-network doesn't support CSS animations on nodes directly
                            // We use a lighter border color to indicate recent activity
                            node.borderColor = '#6ab04c';
                        }
                    }

                    uniqueNodes.push(node);
                } else {
                    console.warn('发现重复节点ID，跳过:', node.id, node.label);
                }
            }
            console.log('   去重前节点数:', nodesData.length, '去重后节点数:', uniqueNodes.length);

            nodesDataSet = new vis.DataSet(uniqueNodes);
            console.log('节点数据集创建成功');
            console.log('   数据集节点数:', nodesDataSet.length);
            if (nodesDataSet.length > 0) {
                console.log('   第一个节点示例:', nodesDataSet.get()[0]);
            }
        } catch (e) {
            console.error('创建节点数据集失败:', e);
            throw e;
        }

        // 存储边的完整数据，用于点击时显示
        console.log('步骤2: 处理边数据...');
        edgesDataMap = {};
        var edgeIndex = 0;
        if (edgesData && edgesData.length > 0) {
            console.log('   处理', edgesData.length, '条边');
            edgesData.forEach(function(edge, index) {
                var edgeId = 'edge_' + edgeIndex++;
                var edgeKey = edge.from + '_' + edge.to + '_' + edgeId;
                edge._visId = edgeId; // 临时存储vis-network的ID
                edgesDataMap[edgeKey] = edge;

                if (index < 3) {
                    console.log('   边[' + index + ']:', {
                        from: edge.from,
                        to: edge.to,
                        visId: edge._visId,
                        hasRelationId: !!edge.relation_id,
                        hasAbsoluteId: !!edge.absolute_id
                    });
                }
            });
            console.log('边数据映射完成，共', Object.keys(edgesDataMap).length, '条');
        } else {
            console.log('没有边数据');
        }

        // 创建边数据集（不显示label，避免糊在一起）
        console.log('步骤3: 创建边数据集...');
        var edgesForVis = [];
        if (edgesData && edgesData.length > 0) {
            edgesForVis = edgesData.map(function(edge, index) {
                if (!edge._visId) {
                    console.warn('边[' + index + ']缺少_visId，自动生成:', edge);
                    edge._visId = 'edge_' + Math.random().toString(36).substr(2, 9);
                }
                return {
                    from: edge.from,
                    to: edge.to,
                    id: edge._visId, // 使用唯一的ID
                    // 不设置label，避免文字重叠
                    arrows: '',
                    width: 2,
                    color: {
                        color: "#888888",
                        highlight: "#4A90E2"
                    }
                };
            });
        }

        try {
            edges = new vis.DataSet(edgesForVis);  // 使用全局变量
            console.log('边数据集创建成功');
            console.log('   数据集边数:', edges.length);
            if (edges.length > 0) {
                console.log('   第一条边示例:', edges.get()[0]);
            }
        } catch (e) {
            console.error('创建边数据集失败:', e);
            throw e;
        }

        // 创建数据对象
        console.log('步骤4: 组装数据对象...');
        var data = {
            nodes: nodesDataSet,
            edges: edges
        };
        console.log('数据对象创建完成');
        console.log('   节点数:', data.nodes.length);
        console.log('   边数:', data.edges.length);

        // 配置选项
        console.log('步骤5: 配置选项...');
        var options = {
            physics: {
                enabled: true,
                solver: "forceAtlas2Based",
                forceAtlas2Based: {
                    gravitationalConstant: -50,
                    centralGravity: 0.01,
                    springLength: 100,
                    springConstant: 0.08
                }
            },
            nodes: {
                font: {
                    color: "white",
                    size: 14
                }
            },
            edges: {
                arrows: {
                    to: {
                        enabled: true,
                        scaleFactor: 1.2
                    }
                },
                width: 2,
                scaling: {
                    min: 1,
                    max: 1,
                    label: {
                        enabled: false
                    }
                },
                color: {
                    color: "#888888",
                    highlight: "#4A90E2"
                },
                selectionWidth: 3
            },
            interaction: {
                hover: true,
                tooltipDelay: 100
            }
        };
        console.log('选项配置完成');

        // 创建网络
        console.log('步骤6: 创建vis.Network实例...');
        console.log('   容器:', container);
        console.log('   数据:', { nodes: data.nodes.length, edges: data.edges.length });
        console.log('   选项:', options);

        if (typeof vis === 'undefined' || !vis.Network) {
            throw new Error('vis.Network 不可用，请检查vis-network库是否正确加载');
        }

        try {
            network = new vis.Network(container, data, options);
            console.log('vis.Network 实例创建成功');
            console.log('   网络对象:', network);

            // 监听网络事件
            network.on("stabilizationEnd", function() {
                console.log('网络布局稳定完成');
            });

            network.on("stabilizationProgress", function(params) {
                if (params.iterations % 50 === 0) {
                    console.log('   布局进度:', params.iterations, '/', params.total);
                }
            });
        } catch (e) {
            console.error('创建vis.Network实例失败:', e);
            console.error('   错误详情:', e.message);
            console.error('   错误堆栈:', e.stack);
            throw e;
        }

        // 添加事件监听
        console.log('步骤7: 添加点击事件监听...');
        network.on("click", function(params) {
            console.log('图谱被点击:', params);
            // 优先判断节点（点击节点时优先显示节点信息）
            if (params.nodes.length > 0) {
                var nodeId = params.nodes[0];
                var node = nodesDataSet.get(nodeId);
                if (node) {
                    console.log('   点击了节点:', nodeId, node.label);

                    // Perform source tracing if enabled
                    if (sourceTracingState.enabled) {
                        performSourceTracing(nodeId);
                    }

                    showNodeDetail(node);
                }
            } else if (params.edges.length > 0) {
                // 如果没有点击节点，再判断是否点击了边
                var edgeId = params.edges[0];
                var edge = edges.get(edgeId);
                if (edge) {
                    console.log('   点击了边:', edgeId);
                    // 遍历edgesDataMap找到对应的edgeData
                    for (var key in edgesDataMap) {
                        if (edgesDataMap[key]._visId === edgeId) {
                            showEdgeDetail(edgesDataMap[key]);
                            break;
                        }
                    }
                }
            } else {
                // 点击空白处，重置侧边栏和溯源
                console.log('   点击了空白处');
                if (sourceTracingState.enabled) {
                    clearSourceTracing();
                }
                resetSidebar();
            }
        });
        console.log('事件监听添加完成');
        console.log('drawGraph 函数执行完成');
    } catch (error) {
        console.error('绘制图谱时发生错误');
        console.error('   错误类型:', error.name);
        console.error('   错误消息:', error.message);
        console.error('   错误堆栈:', error.stack);
        alert('绘制图谱时发生错误: ' + error.message);
    }
}

function resetSidebar() {
    document.getElementById('sidebar-title').textContent = '详细信息';
    document.getElementById('sidebar-subtitle').textContent = '点击节点或边查看详情';
    document.getElementById('sidebar-content').innerHTML =
        '<div class="empty-state">点击图谱中的节点或关系边查看详细信息</div>';
}

async function showNodeDetail(node) {
    document.getElementById('sidebar-title').textContent = '实体详情';
    document.getElementById('sidebar-subtitle').textContent = node.label;

    currentEntityId = node.entity_id || node.id;
    currentEntityAbsoluteId = node.absolute_id || node.id;
    currentEntityVersions = null;
    currentRelationId = null;
    currentRelationVersions = null;
    currentRelationAbsoluteId = null;

    // 通过API获取完整信息（包括embedding_preview和episode_text）
    if (node.absolute_id) {
        try {
            var response = await fetch('/api/entities/' + encodeURIComponent(currentEntityId) + '/versions/' + encodeURIComponent(node.absolute_id));
            var data = await response.json();
            if (data.success) {
                var entity = data.entity;
                // 更新显示
                var html = renderVersionSelector('entity', currentEntityId, currentEntityAbsoluteId, null);
                // Add version timeline button
                html += '<div style="margin-bottom: 15px;">';
                html += '<button class="timeline-btn" onclick="showVersionTimeline(\'' + currentEntityId + '\', \'' + currentEntityAbsoluteId + '\')" style="width: 100%;">';
                html += '<i class="fas fa-stream"></i> 查看版本历史时间轴';
                html += '</button>';
                html += '</div>';
                html += renderEntityDetail({
                    entity_id: entity.entity_id,
                    absolute_id: entity.absolute_id,
                    id: entity.entity_id,
                    name: entity.name,
                    label: entity.name,
                    content: entity.content,
                    event_time: entity.event_time,
                    processed_time: entity.processed_time,
                    episode_content: entity.episode_content,
                    episode_text: entity.episode_text,
                    source_document: entity.source_document,
                    doc_name: entity.source_document,
                    embedding_preview: entity.embedding_preview
                });
                document.getElementById('sidebar-content').innerHTML = html;
                loadEntityVersions(currentEntityId);
            } else {
                // 如果API失败，使用节点数据
                var html = renderVersionSelector('entity', currentEntityId, currentEntityAbsoluteId, null);
                html += renderEntityDetail(node);
                document.getElementById('sidebar-content').innerHTML = html;
                loadEntityVersions(currentEntityId);
            }
        } catch (error) {
            console.error('获取实体详情失败:', error);
            // 如果API失败，使用节点数据
            var html = renderVersionSelector('entity', currentEntityId, currentEntityAbsoluteId, null);
            html += renderEntityDetail(node);
            document.getElementById('sidebar-content').innerHTML = html;
            loadEntityVersions(currentEntityId);
        }
    } else {
        // 如果没有absolute_id，直接显示节点数据
        var html = renderVersionSelector('entity', currentEntityId, currentEntityAbsoluteId, null);
        html += renderEntityDetail(node);
        document.getElementById('sidebar-content').innerHTML = html;
        loadEntityVersions(currentEntityId);
    }
}

function renderVersionSelector(type, id, currentAbsoluteId, currentVersionIndex) {
    var html = '<div class="version-selector" id="version-selector-' + type + '">';
    html += '<div class="version-selector-header">';
    html += '<div class="version-selector-title">版本选择</div>';
    html += '<div class="version-selector-info" id="version-info-' + type + '">加载中...</div>';
    html += '</div>';
    html += '<div class="version-selector-controls">';
    html += '<select id="version-select-' + type + '" onchange="onVersionChange(\\'' + type + '\\', this.value)">';
    html += '<option value="">加载版本列表...</option>';
    html += '</select>';
    html += '<div class="version-nav-buttons">';
    html += '<button class="version-nav-btn" id="version-prev-' + type + '" onclick="navigateVersion(\\'' + type + '\\', -1)" disabled>◀</button>';
    html += '<button class="version-nav-btn" id="version-next-' + type + '" onclick="navigateVersion(\\'' + type + '\\', 1)" disabled>▶</button>';
    html += '</div>';
    html += '</div>';
    html += '</div>';
    return html;
}

function renderEntityDetail(entity) {
    var html = '<div class="detail-section">';
    html += '<div class="detail-section-title">实体 ID | 绝对 ID</div>';
    html += '<div class="detail-content">';
    html += '<div style="margin-bottom: 5px;">实体ID: ' + escapeHtml(entity.entity_id || entity.id) + '</div>';
    if (entity.absolute_id) {
        html += '<div>绝对ID: ' + escapeHtml(entity.absolute_id) + '</div>';
    }
    html += '</div>';
    html += '</div>';

    html += '<div class="detail-section">';
    html += '<div class="detail-section-title">实体名称</div>';
    html += '<div class="detail-content">' + escapeHtml(entity.name || entity.label) + '</div>';
    html += '</div>';

    html += '<div class="detail-section">';
    html += '<div class="detail-section-title">实体描述</div>';
    html += '<div class="detail-content">' + escapeHtml(entity.content || entity.title || entity.label) + '</div>';
    html += '</div>';

    // 时间信息
    if (entity.event_time) {
        html += '<div class="detail-section">';
        html += '<div class="detail-section-title">事件时间</div>';
        html += '<div class="detail-content">';
        try {
            var time = new Date(entity.event_time);
            html += '<div>' + time.toLocaleString('zh-CN') + '</div>';
            html += '<div style="color: #888888; font-size: 12px; margin-top: 5px;">' + time.toISOString() + '</div>';
        } catch (e) {
            html += '<div>' + escapeHtml(entity.event_time) + '</div>';
        }
        html += '</div>';
        html += '</div>';
    }
    if (entity.processed_time) {
        html += '<div class="detail-section">';
        html += '<div class="detail-section-title">处理时间</div>';
        html += '<div class="detail-content">';
        try {
            var time = new Date(entity.processed_time);
            html += '<div>' + time.toLocaleString('zh-CN') + '</div>';
            html += '<div style="color: #888888; font-size: 12px; margin-top: 5px;">' + time.toISOString() + '</div>';
        } catch (e) {
            html += '<div>' + escapeHtml(entity.processed_time) + '</div>';
        }
        html += '</div>';
        html += '</div>';
    }

    // 缓存记忆（episode_id对应的md文档内容）
    if (entity.episode_content) {
        html += '<div class="detail-section">';
        html += '<div class="detail-section-title">缓存记忆</div>';
        html += '<div class="detail-content">';
        html += '<div style="max-height: 200px; overflow-y: auto; font-size: 13px; line-height: 1.5; white-space: pre-wrap; word-wrap: break-word;">';
        html += escapeHtml(entity.episode_content);
        html += '</div>';
        html += '</div>';
        html += '</div>';
    }

    // 原文内容（episode_id对应json中的text内容）
    if (entity.episode_text) {
        html += '<div class="detail-section">';
        html += '<div class="detail-section-title">原文内容</div>';
        html += '<div class="detail-content">';
        html += '<div style="max-height: 200px; overflow-y: auto; font-size: 13px; line-height: 1.5; white-space: pre-wrap; word-wrap: break-word;">';
        html += escapeHtml(entity.episode_text);
        html += '</div>';
        html += '</div>';
        html += '</div>';
    }

    // 文档名称
    if (entity.source_document || entity.doc_name) {
        html += '<div class="detail-section">';
        html += '<div class="detail-section-title">文档名称</div>';
        html += '<div class="detail-content">' + escapeHtml(entity.source_document || entity.doc_name) + '</div>';
        html += '</div>';
    }

    // Embedding向量前4个值
    if (entity.embedding_preview && Array.isArray(entity.embedding_preview)) {
        html += '<div class="detail-section">';
        html += '<div class="detail-section-title">编码向量（前4个值）</div>';
        html += '<div class="detail-content">';
        html += '<div style="font-family: monospace; font-size: 12px;">';
        // 只显示前4个值
        var previewValues = entity.embedding_preview.slice(0, 4);
        html += '[' + previewValues.map(function(val) {
            return val.toFixed(6);
        }).join(', ') + ']';
        html += '</div>';
        html += '</div>';
        html += '</div>';
    }

    return html;
}

async function showEdgeDetail(edgeData) {
    var fromNode = nodesDataSet.get(edgeData.from);
    var toNode = nodesDataSet.get(edgeData.to);

    var fromName = fromNode ? fromNode.label : edgeData.from;
    var toName = toNode ? toNode.label : edgeData.to;
    var fromId = edgeData.from;
    var toId = edgeData.to;

    currentRelationId = edgeData.relation_id;
    currentRelationAbsoluteId = edgeData.absolute_id || edgeData.id;  // 兼容处理
    currentRelationVersions = null;
    currentEntityId = null;
    currentEntityVersions = null;
    currentEntityAbsoluteId = null;

    document.getElementById('sidebar-title').textContent = '关系详情';
    document.getElementById('sidebar-subtitle').textContent = fromName + ' → ' + toName;

    // 如果没有embedding_preview，通过API获取
    if (!edgeData.embedding_preview && edgeData.absolute_id && currentRelationId) {
        try {
            var response = await fetch('/api/relations/' + encodeURIComponent(currentRelationId) + '/versions/' + encodeURIComponent(edgeData.absolute_id));
            var data = await response.json();
            if (data.success && data.relation.embedding_preview) {
                edgeData.embedding_preview = data.relation.embedding_preview;
                // 更新显示
                var html = renderVersionSelector('relation', currentRelationId, currentRelationAbsoluteId, null);
                html += renderRelationDetail(edgeData, fromNode, toNode, fromId, toId, fromName, toName);
                document.getElementById('sidebar-content').innerHTML = html;
                if (currentRelationId) {
                    loadRelationVersions(currentRelationId);
                }
            }
        } catch (error) {
            console.error('获取embedding失败:', error);
        }
    }

    // 显示版本选择器和详细信息
    var html = renderVersionSelector('relation', currentRelationId, currentRelationAbsoluteId, null);
    html += renderRelationDetail(edgeData, fromNode, toNode, fromId, toId, fromName, toName);

    document.getElementById('sidebar-content').innerHTML = html;

    // 加载版本列表
    if (currentRelationId) {
        loadRelationVersions(currentRelationId);
    }
}

function renderRelationDetail(edgeData, fromNode, toNode, fromId, toId, fromName, toName) {
    // 使用完整的content，如果没有则使用title，最后使用label
    var fromContent = fromNode ? (fromNode.content || fromNode.title || fromNode.label) : '未知实体';
    var toContent = toNode ? (toNode.content || toNode.title || toNode.label) : '未知实体';

    var html = '<div class="detail-section">';
    html += '<div class="detail-section-title">关系描述</div>';
    // 优先使用完整的content，如果没有则使用title
    html += '<div class="detail-content">' + escapeHtml(edgeData.content || edgeData.title || '无描述') + '</div>';
    html += '</div>';

    // 起点实体信息
    html += '<div class="detail-section">';
    html += '<div class="detail-section-title">起点实体</div>';
    html += '<div class="detail-content">';
    html += '<strong>实体 ID:</strong> ' + escapeHtml(fromId) + '<br>';
    html += '<strong>实体名称:</strong> ' + escapeHtml(fromName) + '<br><br>';
    html += '<strong>实体描述:</strong><br>' + escapeHtml(fromContent);
    html += '</div>';
    html += '</div>';

    // 终点实体信息
    html += '<div class="detail-section">';
    html += '<div class="detail-section-title">终点实体</div>';
    html += '<div class="detail-content">';
    html += '<strong>实体 ID:</strong> ' + escapeHtml(toId) + '<br>';
    html += '<strong>实体名称:</strong> ' + escapeHtml(toName) + '<br><br>';
    html += '<strong>实体描述:</strong><br>' + escapeHtml(toContent);
    html += '</div>';
    html += '</div>';

    // 时间信息
    if (edgeData.event_time) {
        html += '<div class="detail-section">';
        html += '<div class="detail-section-title">事件时间</div>';
        html += '<div class="detail-content">';
        try {
            var time = new Date(edgeData.event_time);
            html += '<div>' + time.toLocaleString('zh-CN') + '</div>';
            html += '<div style="color: #888888; font-size: 12px; margin-top: 5px;">' + time.toISOString() + '</div>';
        } catch (e) {
            html += '<div>' + escapeHtml(edgeData.event_time) + '</div>';
        }
        html += '</div>';
        html += '</div>';
    }
    if (edgeData.processed_time) {
        html += '<div class="detail-section">';
        html += '<div class="detail-section-title">处理时间</div>';
        html += '<div class="detail-content">';
        try {
            var time = new Date(edgeData.processed_time);
            html += '<div>' + time.toLocaleString('zh-CN') + '</div>';
            html += '<div style="color: #888888; font-size: 12px; margin-top: 5px;">' + time.toISOString() + '</div>';
        } catch (e) {
            html += '<div>' + escapeHtml(edgeData.processed_time) + '</div>';
        }
        html += '</div>';
        html += '</div>';
    }

    // Embedding向量前5个值
    if (edgeData.embedding_preview && Array.isArray(edgeData.embedding_preview)) {
        html += '<div class="detail-section">';
        html += '<div class="detail-section-title">Embedding向量（前5个值）</div>';
        html += '<div class="detail-content">';
        html += '<div style="font-family: monospace; font-size: 12px;">';
        html += '[' + edgeData.embedding_preview.map(function(val) {
            return val.toFixed(6);
        }).join(', ') + ']';
        html += '</div>';
        html += '</div>';
        html += '</div>';
    }

    return html;
}

function updateSearchStatus(message) {
    document.getElementById('search-status').textContent = message;
}

function updateStats(stats) {
    document.getElementById('entity-count').textContent = stats.total_entities || 0;
    document.getElementById('relation-count').textContent = stats.total_relations || 0;
}

function updateLastUpdate() {
    const now = new Date();
    document.getElementById('last-update').textContent = now.toLocaleString('zh-CN');
}

// HTML 转义
function escapeHtml(text) {
    if (!text) return '';
    var div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

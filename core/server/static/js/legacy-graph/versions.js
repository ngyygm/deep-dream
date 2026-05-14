// legacy-graph module: versions.js
// Version management, version timeline modal, and version switching functions

// Version timeline modal functions
async function showVersionTimeline(entityId, absoluteId) {
    var modal = document.getElementById('version-timeline-modal');
    var track = document.getElementById('version-timeline-track');

    // Clear existing items (except the line)
    track.innerHTML = '<div class="version-timeline-line"></div>';

    try {
        var response = await fetch('/api/entities/' + encodeURIComponent(entityId) + '/versions');
        var data = await response.json();

        if (data.success && data.versions) {
            // Sort versions by time
            var sortedVersions = data.versions.slice().sort(function(a, b) {
                return new Date(a.processed_time) - new Date(b.processed_time);
            });

            sortedVersions.forEach(function(version, index) {
                var item = createVersionTimelineItem(version, index, sortedVersions.length, absoluteId);
                track.appendChild(item);
            });

            modal.classList.add('active');
        }
    } catch (error) {
        console.error('Failed to load versions:', error);
    }
}

function createVersionTimelineItem(version, index, total, currentAbsoluteId) {
    var item = document.createElement('div');
    item.className = 'version-timeline-item';

    var dot = document.createElement('div');
    dot.className = 'version-timeline-dot';
    if (version.absolute_id === currentAbsoluteId) {
        dot.classList.add('active');
    }
    dot.style.backgroundColor = getTimeBasedColor(new Date(version.processed_time));
    dot.onclick = function() {
        switchEntityVersion(version.family_id, version.absolute_id);
        closeModal();
    };

    var card = document.createElement('div');
    card.className = 'version-timeline-card';
    card.style.borderLeftColor = dot.style.backgroundColor;

    var header = document.createElement('div');
    header.className = 'version-timeline-card-header';

    var versionSpan = document.createElement('span');
    versionSpan.className = 'version-timeline-version';
    versionSpan.textContent = '版本 ' + (index + 1) + '/' + total;

    var timeSpan = document.createElement('span');
    timeSpan.className = 'version-timeline-time';
    timeSpan.textContent = formatDateTime(version.processed_time);

    header.appendChild(versionSpan);
    header.appendChild(timeSpan);

    var content = document.createElement('div');
    content.className = 'version-timeline-content';
    content.textContent = version.content || version.name || '';

    card.appendChild(header);
    card.appendChild(content);

    item.appendChild(dot);
    item.appendChild(card);

    return item;
}

function closeModal() {
    document.getElementById('version-timeline-modal').classList.remove('active');
}

document.getElementById('version-timeline-close').addEventListener('click', closeModal);

// Close modal on outside click
document.getElementById('version-timeline-modal').addEventListener('click', function(e) {
    if (e.target === this) {
        closeModal();
    }
});

async function loadEntityVersions(entityId) {
    try {
        var response = await fetch('/api/entities/' + encodeURIComponent(entityId) + '/versions');
        var data = await response.json();
        if (data.success) {
            currentEntityVersions = data.versions;
            updateVersionSelector('entity', data.versions, currentEntityAbsoluteId);
        } else {
            console.error('加载实体版本失败:', data.error);
            document.getElementById('version-info-entity').textContent = '加载失败';
        }
    } catch (error) {
        console.error('请求错误:', error);
        document.getElementById('version-info-entity').textContent = '加载失败';
    }
}

function updateVersionSelector(type, versions, currentAbsoluteId) {
    var select = document.getElementById('version-select-' + type);
    var info = document.getElementById('version-info-' + type);
    var prevBtn = document.getElementById('version-prev-' + type);
    var nextBtn = document.getElementById('version-next-' + type);

    // 清空选项
    select.innerHTML = '';

    if (!versions || versions.length === 0) {
        select.innerHTML = '<option value="">无版本数据</option>';
        info.textContent = '无版本';
        prevBtn.disabled = true;
        nextBtn.disabled = true;
        return;
    }

    // 找到当前版本的索引
    var currentIndex = versions.findIndex(v => v.absolute_id === currentAbsoluteId);
    if (currentIndex === -1) {
        currentIndex = 0; // 默认选择第一个（最新版本）
    }

    // 填充选项
    versions.forEach(function(version, index) {
        var option = document.createElement('option');
        option.value = version.absolute_id;
        option.textContent = '版本 ' + version.index + '/' + version.total + ' (' + formatDateTime(version.processed_time) + ')';
        if (index === currentIndex) {
            option.selected = true;
        }
        select.appendChild(option);
    });

    // 更新信息显示
    var currentVersion = versions[currentIndex];
    info.textContent = '版本 ' + currentVersion.index + '/' + currentVersion.total;

    // 更新导航按钮状态
    prevBtn.disabled = currentIndex === 0;
    nextBtn.disabled = currentIndex === versions.length - 1;
}

function onVersionChange(type, absoluteId) {
    if (!absoluteId) return;

    if (type === 'entity') {
        switchEntityVersion(currentEntityId, absoluteId);
    } else if (type === 'relation') {
        switchRelationVersion(currentRelationId, absoluteId);
    }
}

function navigateVersion(type, direction) {
    var versions = type === 'entity' ? currentEntityVersions : currentRelationVersions;
    if (!versions || versions.length === 0) return;

    var select = document.getElementById('version-select-' + type);
    var currentAbsoluteId = select.value;
    var currentIndex = versions.findIndex(v => v.absolute_id === currentAbsoluteId);

    if (currentIndex === -1) return;

    var newIndex = currentIndex + direction;
    if (newIndex < 0 || newIndex >= versions.length) return;

    var newAbsoluteId = versions[newIndex].absolute_id;
    select.value = newAbsoluteId;
    onVersionChange(type, newAbsoluteId);
}

async function switchEntityVersion(entityId, absoluteId) {
    try {
        var response = await fetch('/api/entities/' + encodeURIComponent(entityId) + '/versions/' + encodeURIComponent(absoluteId));
        var data = await response.json();
        if (data.success) {
            currentEntityAbsoluteId = absoluteId;
            var entity = data.entity;

            // 更新详细信息显示
            var detailHtml = renderEntityDetail({
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

            // 保留版本选择器，只更新详细信息部分
            var versionSelector = document.getElementById('version-selector-entity');
            var detailSection = versionSelector.nextElementSibling;
            if (detailSection && detailSection.classList.contains('detail-section')) {
                // 找到所有详细信息部分并替换
                var content = document.getElementById('sidebar-content');
                var versionSelectorHtml = versionSelector.outerHTML;
                content.innerHTML = versionSelectorHtml + detailHtml;

                // 重新绑定事件
                var newSelect = document.getElementById('version-select-entity');
                newSelect.onchange = function() { onVersionChange('entity', this.value); };
                document.getElementById('version-prev-entity').onclick = function() { navigateVersion('entity', -1); };
                document.getElementById('version-next-entity').onclick = function() { navigateVersion('entity', 1); };

                // 更新版本选择器
                updateVersionSelector('entity', currentEntityVersions, absoluteId);
            } else {
                // 如果结构不对，重新渲染整个内容
                var html = renderVersionSelector('entity', entityId, absoluteId, entity.version_index);
                html += detailHtml;
                document.getElementById('sidebar-content').innerHTML = html;
                loadEntityVersions(entityId);
            }

            // 更新图谱中的节点和相关的边、实体
            updateGraphForEntityVersion(entityId, absoluteId, entity.event_time);
        } else {
            console.error('切换实体版本失败:', data.error);
            alert('切换版本失败: ' + data.error);
        }
    } catch (error) {
        console.error('请求错误:', error);
        alert('请求错误: ' + error);
    }
}

async function updateGraphForEntityVersion(entityId, absoluteId, timePoint) {
    if (!network || !nodesDataSet) return;

    console.log('更新图谱以反映实体版本变化:', entityId, absoluteId);
    console.log('   版本ID:', absoluteId);

    // 获取当前的限制参数
    var limitEntities = parseInt(document.getElementById('limit-entities-input').value) || 100;
    var limitEdgesPerEntity = parseInt(document.getElementById('limit-edges-input').value) || 50;
    var hops = parseInt(document.getElementById('hops-input').value) || 1;

    // 重新加载图谱，以该实体版本为中心
    currentMode = 'version_snapshot';
    currentSearchQuery = '';
    updateSearchStatus('显示实体版本: ' + entityId + ' (到版本 ' + absoluteId.substring(0, 8) + '...) - ' + hops + '跳 - 每实体最多' + limitEdgesPerEntity + '条关系');

    var url = '/api/graphs/data?limit_entities=' + limitEntities +
              '&limit_edges_per_entity=' + limitEdgesPerEntity +
              '&hops=' + hops +
              '&focus_family_id=' + encodeURIComponent(entityId) +
              '&focus_absolute_id=' + encodeURIComponent(absoluteId);

    try {
        var response = await fetch(url);
        if (!response.ok) {
            throw new Error('HTTP错误: ' + response.status + ' ' + response.statusText);
        }
        var data = await response.json();
        if (data.success) {
            console.log('获取实体版本图谱成功');
            console.log('   节点数量:', data.nodes ? data.nodes.length : 0);
            console.log('   边数量:', data.edges ? data.edges.length : 0);

            updateStats(data.stats);
            drawGraph(data.nodes, data.edges);
            updateLastUpdate();
        } else {
            console.error('获取实体版本图谱失败:', data.error);
            alert('获取实体版本图谱失败: ' + data.error);
        }
    } catch (error) {
        console.error('请求错误:', error);
        alert('请求错误: ' + error.message);
    }
}

async function updateGraphForRelationVersion(relationId, absoluteId, timePoint, fromEntityId, toEntityId) {
    if (!network || !nodesDataSet || !edges) return;

    console.log('更新图谱以反映关系版本变化:', relationId, absoluteId, timePoint);
    console.log('   时间点:', timePoint);

    // 获取当前的限制参数
    var limitEntities = parseInt(document.getElementById('limit-entities-input').value) || 100;
    var limitEdgesPerEntity = parseInt(document.getElementById('limit-edges-input').value) || 50;

    // 重新加载图谱，但只显示该时间点之前的数据
    currentMode = 'version_snapshot';
    currentSearchQuery = '';
    updateSearchStatus('显示时间点: ' + new Date(timePoint).toLocaleString('zh-CN'));

    try {
        var response = await fetch('/api/graphs/data?limit_entities=' + limitEntities + '&limit_edges_per_entity=' + limitEdgesPerEntity + '&time_point=' + encodeURIComponent(timePoint));
        if (!response.ok) {
            throw new Error('HTTP错误: ' + response.status + ' ' + response.statusText);
        }
        var data = await response.json();
        if (data.success) {
            console.log('获取时间点快照成功');
            console.log('   节点数量:', data.nodes ? data.nodes.length : 0);
            console.log('   边数量:', data.edges ? data.edges.length : 0);

            updateStats(data.stats);
            drawGraph(data.nodes, data.edges);
            updateLastUpdate();
        } else {
            console.error('获取时间点快照失败:', data.error);
            alert('获取时间点快照失败: ' + data.error);
        }
    } catch (error) {
        console.error('请求错误:', error);
        alert('请求错误: ' + error.message);
    }
}

function formatDateTime(isoString) {
    try {
        var date = new Date(isoString);
        return date.toLocaleString('zh-CN', {
            year: 'numeric',
            month: '2-digit',
            day: '2-digit',
            hour: '2-digit',
            minute: '2-digit'
        });
    } catch (e) {
        return isoString;
    }
}

async function loadRelationVersions(relationId) {
    try {
        var response = await fetch('/api/relations/' + encodeURIComponent(relationId) + '/versions');
        var data = await response.json();
        if (data.success) {
            currentRelationVersions = data.versions;
            updateVersionSelector('relation', data.versions, currentRelationAbsoluteId);
        } else {
            console.error('加载关系版本失败:', data.error);
            document.getElementById('version-info-relation').textContent = '加载失败';
        }
    } catch (error) {
        console.error('请求错误:', error);
        document.getElementById('version-info-relation').textContent = '加载失败';
    }
}

async function switchRelationVersion(relationId, absoluteId) {
    try {
        var response = await fetch('/api/relations/' + encodeURIComponent(relationId) + '/versions/' + encodeURIComponent(absoluteId));
        var data = await response.json();
        if (data.success) {
            currentRelationAbsoluteId = absoluteId;
            var relation = data.relation;

            // 更新图谱中的关系边和连接的实体
            updateGraphForRelationVersion(relationId, absoluteId, relation.event_time, relation.from_entity_id, relation.to_entity_id);

            // 获取实体信息
            var fromNode = null;
            var toNode = null;
            if (relation.from_entity_id) {
                fromNode = nodesDataSet.get(relation.from_entity_id);
            }
            if (relation.to_entity_id) {
                toNode = nodesDataSet.get(relation.to_entity_id);
            }

            var fromName = relation.from_entity_name || (fromNode ? fromNode.label : '未知实体');
            var toName = relation.to_entity_name || (toNode ? toNode.label : '未知实体');
            var fromId = relation.from_entity_id || '未知';
            var toId = relation.to_entity_id || '未知';

            // 获取实体内容
            var fromContent = fromNode ? (fromNode.content || fromNode.title || fromNode.label) : '未知实体';
            var toContent = toNode ? (toNode.content || toNode.title || toNode.label) : '未知实体';

            // 如果节点不存在，尝试通过API获取实体信息
            var entityPromises = [];
            if (!fromNode && relation.from_entity_id) {
                entityPromises.push(
                    (async () => {
                        try {
                            var r = await fetch('/api/entities/' + encodeURIComponent(relation.from_entity_id) + '/versions');
                            var d = await r.json();
                            if (d.success && d.versions.length > 0) {
                                return { type: 'from', content: d.versions[0].content || fromName };
                            }
                            return null;
                        } catch (e) {
                            return null;
                        }
                    })()
                );
            } else {
                entityPromises.push(Promise.resolve(null));
            }

            if (!toNode && relation.to_entity_id) {
                entityPromises.push(
                    (async () => {
                        try {
                            var r = await fetch('/api/entities/' + encodeURIComponent(relation.to_entity_id) + '/versions');
                            var d = await r.json();
                            if (d.success && d.versions.length > 0) {
                                return { type: 'to', content: d.versions[0].content || toName };
                            }
                            return null;
                        } catch (e) {
                            return null;
                        }
                    })()
                );
            } else {
                entityPromises.push(Promise.resolve(null));
            }

            // 等待所有实体信息加载完成后再更新显示
            var results = await Promise.all(entityPromises);
            // 更新实体内容
            results.forEach(function(result) {
                if (result) {
                    if (result.type === 'from') {
                        fromContent = result.content;
                    } else if (result.type === 'to') {
                        toContent = result.content;
                    }
                }
            });

            // 更新标题
            document.getElementById('sidebar-subtitle').textContent = fromName + ' → ' + toName;

            // 创建虚拟节点对象用于renderRelationDetail
            var virtualFromNode = fromNode || { label: fromName, content: fromContent, title: fromName };
            var virtualToNode = toNode || { label: toName, content: toContent, title: toName };

            // 更新详细信息显示
            var detailHtml = renderRelationDetail({
                content: relation.content,
                event_time: relation.event_time,
                processed_time: relation.processed_time,
                embedding_preview: relation.embedding_preview
            }, virtualFromNode, virtualToNode, fromId, toId, fromName, toName);

            // 保留版本选择器，只更新详细信息部分
            var versionSelector = document.getElementById('version-selector-relation');
            var detailSection = versionSelector.nextElementSibling;
            if (detailSection && detailSection.classList.contains('detail-section')) {
                // 找到所有详细信息部分并替换
                var content = document.getElementById('sidebar-content');
                var versionSelectorHtml = versionSelector.outerHTML;
                content.innerHTML = versionSelectorHtml + detailHtml;

                // 重新绑定事件
                var newSelect = document.getElementById('version-select-relation');
                newSelect.onchange = function() { onVersionChange('relation', this.value); };
                document.getElementById('version-prev-relation').onclick = function() { navigateVersion('relation', -1); };
                document.getElementById('version-next-relation').onclick = function() { navigateVersion('relation', 1); };

                // 更新版本选择器
                updateVersionSelector('relation', currentRelationVersions, absoluteId);
            } else {
                // 如果结构不对，重新渲染整个内容
                var html = renderVersionSelector('relation', relationId, absoluteId, relation.version_index);
                html += detailHtml;
                document.getElementById('sidebar-content').innerHTML = html;
                loadRelationVersions(relationId);
            }
        } else {
            console.error('切换关系版本失败:', data.error);
            alert('切换版本失败: ' + data.error);
        }
    } catch (error) {
        console.error('请求错误:', error);
        alert('请求错误: ' + error);
    }
}

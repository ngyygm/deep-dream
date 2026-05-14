// legacy-graph module: state.js
// Shared global state variables

console.log('=== 图谱可视化脚本开始加载 ===');

// 检查vis-network库是否加载
if (typeof vis === 'undefined') {
    console.error('vis-network库未加载！请检查CDN链接');
} else {
    console.log('vis-network库已加载，版本:', vis.Network ? '可用' : '不可用');
}

var network = null;
var container = document.getElementById('mynetwork');
var nodesDataSet = null;
var edges = null;  // 边数据集，用于版本切换时更新

// 检查容器元素
if (!container) {
    console.error('找不到容器元素 #mynetwork');
} else {
    console.log('容器元素找到:', container);
    console.log('   容器尺寸:', container.offsetWidth, 'x', container.offsetHeight);
}

// 跟踪当前模式：'default' 或 'search'
var currentMode = 'default';
var currentSearchQuery = '';

// Timeline state
var timelineState = {
    enabled: false,
    earliestTime: null,
    latestTime: null,
    currentTime: null,
    timePoints: [],
    isPlaying: false,
    playInterval: null,
    playSpeed: 2000 // ms per step
};

// Search focus state
var searchFocusState = {
    matchedNodes: new Set(),
    matchedEdges: new Set()
};

// Source tracing state
var sourceTracingState = {
    enabled: false,
    sourceNodes: new Set(),
    pathNodes: new Set(),
    pathEdges: new Set()
};

// 存储边的完整数据
var edgesDataMap = {};

// 存储当前显示的实体和关系信息
var currentEntityId = null;
var currentEntityVersions = null;
var currentEntityAbsoluteId = null;
var currentRelationId = null;
var currentRelationVersions = null;
var currentRelationAbsoluteId = null;

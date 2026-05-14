// legacy-graph module: init.js
// Page initialization and window.onload handler

// 页面加载时自动加载图谱
window.onload = async function() {
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    console.log('页面加载完成');
    console.log('   当前URL:', window.location.href);
    console.log('   页面标题:', document.title);
    console.log('   容器元素:', container ? '找到' : '未找到');
    console.log('   vis库状态:', typeof vis !== 'undefined' ? '已加载' : '未加载');
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');

    // Initialize timeline
    initTimeline();

    // 获取当前配置并设置默认路径
    try {
        var response = await fetch('/api/graphs/config');
        var data = await response.json();
        if (data.success && data.storage_path) {
            var pathInput = document.getElementById('graph-path-input');
            if (pathInput && !pathInput.value.trim()) {
                pathInput.value = data.storage_path;
                console.log('已设置默认图谱路径:', data.storage_path);
            }
        }
        console.log('开始自动加载图谱...');
        loadGraph();
    } catch (error) {
        console.warn('获取配置失败，使用默认设置:', error);
        console.log('开始自动加载图谱...');
        loadGraph();
    }
};

console.log('图谱可视化脚本加载完成');

"""
图谱可视化 Web 服务
提供实时查看图谱可视化的 Web 界面（含图谱可视化器）
"""
import sys
import json
from pathlib import Path
from typing import List, Optional
from flask import Flask, render_template_string, jsonify

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from processor import StorageManager, EmbeddingClient, Entity, Relation
from processor.llm_client import LLMClient


# ---------------------------------------------------------------------------
# 图谱可视化器（原 visualizer 模块）
# ---------------------------------------------------------------------------

class GraphVisualizer:
    """图谱可视化器"""
    
    def __init__(self, storage_manager: StorageManager):
        """
        初始化可视化器
        
        Args:
            storage_manager: 存储管理器实例
        """
        self.storage = storage_manager
    
    def visualize_interactive(self, output_path: str = "graph_visualization.html", 
                               layout: str = "spring", height: str = "800px", 
                               width: str = "100%", show_labels: bool = True):
        """
        生成交互式 HTML 可视化（使用 pyvis）
        
        Args:
            output_path: 输出 HTML 文件路径
            layout: 布局算法 ("spring", "hierarchical", "circular" 等)
            height: 画布高度
            width: 画布宽度
            show_labels: 是否显示标签
        """
        try:
            from pyvis.network import Network
        except ImportError:
            print("错误：未安装 pyvis 库")
            print("请运行: pip install pyvis")
            return False
        
        # 获取所有实体和关系
        entities = self.storage.get_all_entities()
        relations = self.storage.get_all_relations()
        
        if not entities:
            print("警告：没有找到实体，无法生成可视化")
            return False
        
        # 创建网络图（无向图）
        net = Network(height=height, width=width, directed=False, bgcolor="#222222", font_color="white")
        net.set_options("""
        {
          "physics": {
            "enabled": true,
            "solver": "forceAtlas2Based",
            "forceAtlas2Based": {
              "gravitationalConstant": -50,
              "centralGravity": 0.01,
              "springLength": 100,
              "springConstant": 0.08
            }
          }
        }
        """)
        
        # 添加实体节点
        entity_id_to_name = {}
        for entity in entities:
            entity_id_to_name[entity.entity_id] = entity.name
            # 创建节点标签（显示名称和简短描述）
            label = entity.name
            if show_labels and entity.content:
                # 截取前50个字符作为描述
                short_content = entity.content[:50] + "..." if len(entity.content) > 50 else entity.content
                title = f"{entity.name}\n\n{short_content}"
            else:
                title = entity.name
            
            net.add_node(
                entity.entity_id,
                label=label,
                title=title,
                color="#4A90E2",
                size=20,
                shape="dot"
            )
        
        # 添加关系边（通过绝对ID获取实体，无向关系）
        for relation in relations:
            # 通过绝对ID获取实体
            entity1 = self.storage.get_entity_by_absolute_id(relation.entity1_absolute_id)
            entity2 = self.storage.get_entity_by_absolute_id(relation.entity2_absolute_id)
            
            if entity1 and entity2:
                entity1_id = entity1.entity_id
                entity2_id = entity2.entity_id
                
                # 确保两个实体都存在
                if entity1_id in entity_id_to_name and entity2_id in entity_id_to_name:
                    # 创建边的标签（显示关系内容的前30个字符）
                    edge_label = ""
                    if show_labels and relation.content:
                        edge_label = relation.content[:30] + "..." if len(relation.content) > 30 else relation.content
                    
                    net.add_edge(
                        entity1_id,
                        entity2_id,
                        label=edge_label,
                        title=relation.content,
                        color="#888888",
                        width=2
                    )
        
        # 保存为 HTML
        net.save_graph(output_path)
        print(f"交互式可视化已保存到: {output_path}")
        print(f"实体数量: {len(entities)}, 关系数量: {len(relations)}")
        return True
    
    def visualize_static(self, output_path: str = "graph_visualization.png",
                        figsize: tuple = (16, 12), node_size: int = 1000,
                        font_size: int = 10, layout: str = "spring"):
        """
        生成静态图片可视化（使用 networkx + matplotlib）
        
        Args:
            output_path: 输出图片文件路径
            figsize: 图片尺寸 (width, height)
            node_size: 节点大小
            font_size: 字体大小
            layout: 布局算法 ("spring", "circular", "kamada_kawai" 等)
        """
        try:
            import networkx as nx
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')  # 使用非交互式后端
            
            # 配置中文字体支持
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
            plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        except ImportError:
            print("错误：未安装 networkx 或 matplotlib 库")
            print("请运行: pip install networkx matplotlib")
            return False
        
        # 获取所有实体和关系
        entities = self.storage.get_all_entities()
        relations = self.storage.get_all_relations()
        
        if not entities:
            print("警告：没有找到实体，无法生成可视化")
            return False
        
        # 创建无向图
        G = nx.Graph()
        
        # 添加节点
        entity_id_to_name = {}
        for entity in entities:
            entity_id_to_name[entity.entity_id] = entity.name
            G.add_node(entity.entity_id, name=entity.name, content=entity.content)
        
        # 添加边（通过绝对ID获取实体，无向关系）
        for relation in relations:
            # 通过绝对ID获取实体
            entity1 = self.storage.get_entity_by_absolute_id(relation.entity1_absolute_id)
            entity2 = self.storage.get_entity_by_absolute_id(relation.entity2_absolute_id)
            
            if entity1 and entity2:
                entity1_id = entity1.entity_id
                entity2_id = entity2.entity_id
                
                if entity1_id in entity_id_to_name and entity2_id in entity_id_to_name:
                    G.add_edge(entity1_id, entity2_id, content=relation.content)
        
        # 选择布局算法
        if layout == "spring":
            pos = nx.spring_layout(G, k=1, iterations=50)
        elif layout == "circular":
            pos = nx.circular_layout(G)
        elif layout == "kamada_kawai":
            try:
                pos = nx.kamada_kawai_layout(G)
            except:
                pos = nx.spring_layout(G)
        else:
            pos = nx.spring_layout(G)
        
        # 绘制图形
        plt.figure(figsize=figsize)
        plt.axis('off')
        
        # 绘制节点
        nx.draw_networkx_nodes(
            G, pos,
            node_color='#4A90E2',
            node_size=node_size,
            alpha=0.9
        )
        
        # 绘制边（无向图，无箭头）
        nx.draw_networkx_edges(
            G, pos,
            edge_color='#888888',
            arrows=False,
            alpha=0.6,
            width=2
        )
        
        # 绘制标签
        labels = {node_id: entity_id_to_name[node_id] for node_id in G.nodes()}
        nx.draw_networkx_labels(
            G, pos,
            labels=labels,
            font_size=font_size,
            font_color='white',
            font_weight='bold'
        )
        
        # 保存图片
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='#222222')
        plt.close()
        
        print(f"静态可视化已保存到: {output_path}")
        print(f"实体数量: {len(entities)}, 关系数量: {len(relations)}")
        return True
    
    def export_json(self, output_path: str = "graph_data.json"):
        """
        导出图谱数据为 JSON 格式
        
        Args:
            output_path: 输出 JSON 文件路径
        """
        entities = self.storage.get_all_entities()
        relations = self.storage.get_all_relations()
        
        data = {
            "entities": [
                {
                    "entity_id": e.entity_id,
                    "name": e.name,
                    "content": e.content,
                    "physical_time": e.physical_time.isoformat(),
                    "memory_cache_id": e.memory_cache_id
                }
                for e in entities
            ],
            "relations": [
                {
                    "relation_id": r.relation_id,
                    "entity1_absolute_id": r.entity1_absolute_id,
                    "entity2_absolute_id": r.entity2_absolute_id,
                    "entity1_id": self.storage.get_entity_by_absolute_id(r.entity1_absolute_id).entity_id if self.storage.get_entity_by_absolute_id(r.entity1_absolute_id) else None,
                    "entity2_id": self.storage.get_entity_by_absolute_id(r.entity2_absolute_id).entity_id if self.storage.get_entity_by_absolute_id(r.entity2_absolute_id) else None,
                    "content": r.content,
                    "physical_time": r.physical_time.isoformat(),
                    "memory_cache_id": r.memory_cache_id
                }
                for r in relations
            ],
            "statistics": {
                "total_entities": len(entities),
                "total_relations": len(relations)
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"图谱数据已导出到: {output_path}")
        return True
    
    def print_statistics(self):
        """打印图谱统计信息"""
        entities = self.storage.get_all_entities()
        relations = self.storage.get_all_relations()
        
        print("\n" + "="*50)
        print("图谱统计信息")
        print("="*50)
        print(f"实体总数: {len(entities)}")
        print(f"关系总数: {len(relations)}")
        
        if entities:
            print("\n实体列表:")
            for i, entity in enumerate(entities[:10], 1):  # 只显示前10个
                print(f"  {i}. {entity.name} (ID: {entity.entity_id})")
            if len(entities) > 10:
                print(f"  ... 还有 {len(entities) - 10} 个实体")
        
        if relations:
            print("\n关系列表:")
            entity_id_to_name = {e.entity_id: e.name for e in entities}
            for i, relation in enumerate(relations[:10], 1):  # 只显示前10个
                # 通过绝对ID获取实体
                entity1 = self.storage.get_entity_by_absolute_id(relation.entity1_absolute_id)
                entity2 = self.storage.get_entity_by_absolute_id(relation.entity2_absolute_id)
                
                if entity1 and entity2:
                    entity1_name = entity1.name
                    entity2_name = entity2.name
                else:
                    entity1_name = relation.entity1_absolute_id
                    entity2_name = relation.entity2_absolute_id
                
                print(f"  {i}. {entity1_name} -- {entity2_name}")
                print(f"     内容: {relation.content[:50]}...")
            if len(relations) > 10:
                print(f"  ... 还有 {len(relations) - 10} 个关系")
        
        print("="*50 + "\n")


# ---------------------------------------------------------------------------
# Web 服务
# ---------------------------------------------------------------------------

# HTML 模板
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>时序记忆图谱可视化</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/dist/dist/vis-network.min.css" 
          integrity="sha512-WgxfT5LWjfszlPHXRmBWHkV2eceiWTOBvrKCNbdgDYTHrT2AeLCGbF4sZlZw3UMN3WtL0tGUoIAKsu8mllg/XA==" 
          crossorigin="anonymous" referrerpolicy="no-referrer" />
    <script src="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/dist/vis-network.min.js" 
            integrity="sha512-LnvoEWDFrqGHlHmDD2101OrLcbsfkrzoSpvtSQtxK3RMnRV0eOkhhBN2dXHKRrUU8p2DGRTk35n4O8nWSVe1mQ==" 
            crossorigin="anonymous" referrerpolicy="no-referrer"></script>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.0.0-beta3/dist/css/bootstrap.min.css"
          rel="stylesheet"
          integrity="sha384-eOJMYsd53ii+scO/bJGFsiCZc+5NDVN2yr8+0RDqr0Ql0h+rP48ckxlpbzKgwra6"
          crossorigin="anonymous" />
      <style>
          body {
              background-color: #1a1a1a;
              color: #ffffff;
              font-family: 'Microsoft YaHei', 'SimHei', Arial, sans-serif;
              margin: 0;
              padding: 0;
              display: flex;
              height: 100vh;
              overflow: hidden;
          }
          .main-content {
              flex: 1;
              display: flex;
              flex-direction: column;
              padding: 20px;
              overflow: hidden;
          }
          .container {
              padding: 0;
          }
          h1 {
              margin: 0 0 15px 0;
              font-size: 24px;
          }
          .stats-panel {
              background-color: #2a2a2a;
              padding: 15px;
              border-radius: 8px;
              margin-bottom: 15px;
              flex-shrink: 0;
          }
          .btn-refresh {
              margin-bottom: 0;
          }
          .info-text {
              color: #aaaaaa;
              font-size: 14px;
              margin: 0;
          }
          #mynetwork {
              flex: 1;
              min-height: 0;
              background-color: #222222;
              border: 1px solid #444444;
              border-radius: 8px;
              position: relative;
          }
          .sidebar {
              width: 500px;
              background-color: #2a2a2a;
              border-left: 2px solid #444444;
              display: flex;
              flex-direction: column;
              overflow: hidden;
              flex-shrink: 0;
          }
          .sidebar-header {
              padding: 20px;
              border-bottom: 2px solid #444444;
              background-color: #1a1a1a;
              flex-shrink: 0;
          }
          .sidebar-title {
              font-size: 18px;
              font-weight: bold;
              color: #4A90E2;
              margin: 0 0 5px 0;
          }
          .sidebar-subtitle {
              font-size: 14px;
              color: #888888;
              margin: 0;
          }
          .sidebar-content {
              flex: 1;
              overflow-y: auto;
              padding: 20px;
          }
          .detail-section {
              margin-bottom: 20px;
          }
          .detail-section-title {
              font-size: 16px;
              font-weight: bold;
              color: #4A90E2;
              margin-bottom: 10px;
              padding-bottom: 8px;
              border-bottom: 1px solid #444444;
          }
          .detail-content {
              color: #cccccc;
              line-height: 1.6;
              white-space: pre-wrap;
              word-wrap: break-word;
              background-color: #1a1a1a;
              padding: 15px;
              border-radius: 6px;
          }
          .empty-state {
              text-align: center;
              color: #888888;
              padding: 40px 20px;
          }
          .version-selector {
              background-color: #1a1a1a;
              padding: 15px;
              border-radius: 6px;
              margin-bottom: 20px;
              border: 1px solid #444444;
          }
          .version-selector-header {
              display: flex;
              justify-content: space-between;
              align-items: center;
              margin-bottom: 10px;
          }
          .version-selector-title {
              font-size: 14px;
              font-weight: bold;
              color: #4A90E2;
          }
          .version-selector-info {
              font-size: 12px;
              color: #888888;
          }
          .version-selector-controls {
              display: flex;
              gap: 10px;
              align-items: center;
          }
          .version-selector select {
              flex: 1;
              background-color: #2a2a2a;
              color: #ffffff;
              border: 1px solid #444444;
              border-radius: 4px;
              padding: 8px 12px;
              font-size: 14px;
              cursor: pointer;
          }
          .version-selector select:hover {
              border-color: #4A90E2;
          }
          .version-selector select:focus {
              outline: none;
              border-color: #4A90E2;
          }
          .version-nav-buttons {
              display: flex;
              gap: 5px;
          }
          .version-nav-btn {
              background-color: #2a2a2a;
              color: #ffffff;
              border: 1px solid #444444;
              border-radius: 4px;
              padding: 6px 10px;
              font-size: 14px;
              cursor: pointer;
              transition: all 0.2s;
              min-width: 32px;
              text-align: center;
          }
          .version-nav-btn:hover:not(:disabled) {
              background-color: #4A90E2;
              border-color: #4A90E2;
          }
          .version-nav-btn:disabled {
              opacity: 0.5;
              cursor: not-allowed;
          }
          .version-loading {
              text-align: center;
              color: #888888;
              padding: 10px;
              font-size: 12px;
          }
          .search-panel {
              background-color: #2a2a2a;
              padding: 15px;
              border-radius: 8px;
              margin-bottom: 15px;
              flex-shrink: 0;
          }
          .search-controls {
              display: flex;
              gap: 10px;
              align-items: center;
              flex-wrap: wrap;
          }
          .search-input {
              flex: 1;
              min-width: 200px;
              background-color: #1a1a1a;
              color: #ffffff;
              border: 1px solid #444444;
              border-radius: 4px;
              padding: 8px 12px;
              font-size: 14px;
          }
          .search-input:focus {
              outline: none;
              border-color: #4A90E2;
          }
          .search-count-select {
              background-color: #1a1a1a;
              color: #ffffff;
              border: 1px solid #444444;
              border-radius: 4px;
              padding: 8px 12px;
              font-size: 14px;
              cursor: pointer;
          }
          .search-count-select:focus {
              outline: none;
              border-color: #4A90E2;
          }
          .search-count-input {
              width: 80px;
              background-color: #1a1a1a;
              color: #ffffff;
              border: 1px solid #444444;
              border-radius: 4px;
              padding: 8px 12px;
              font-size: 14px;
              text-align: center;
          }
          .search-count-input:focus {
              outline: none;
              border-color: #4A90E2;
          }
          .search-count-label {
              color: #aaaaaa;
              font-size: 14px;
              margin: 0 5px;
          }
          .search-btn {
              background-color: #4A90E2;
              color: #ffffff;
              border: none;
              border-radius: 4px;
              padding: 8px 16px;
              font-size: 14px;
              cursor: pointer;
              transition: background-color 0.2s;
          }
          .search-btn:hover {
              background-color: #357ABD;
          }
          .search-btn:disabled {
              background-color: #555555;
              cursor: not-allowed;
          }
          .clear-search-btn {
              background-color: #666666;
              color: #ffffff;
              border: none;
              border-radius: 4px;
              padding: 8px 16px;
              font-size: 14px;
              cursor: pointer;
              transition: background-color 0.2s;
          }
          .clear-search-btn:hover {
              background-color: #777777;
          }
          .search-status {
              margin-top: 10px;
              font-size: 12px;
              color: #888888;
          }
      </style>
  </head>
  <body>
      <div class="main-content">
          <h1>时序记忆图谱可视化</h1>
          
          <div class="stats-panel">
              <div style="display: flex; flex-wrap: wrap; gap: 15px; align-items: center;">
                  <!-- 统计信息 -->
                  <div style="display: flex; gap: 20px; align-items: center;">
                      <span style="color: #aaaaaa; font-size: 14px;"><strong style="color: #ffffff;">实体数量:</strong> <span id="entity-count" style="color: #4A90E2;">-</span></span>
                      <span style="color: #aaaaaa; font-size: 14px;"><strong style="color: #ffffff;">关系数量:</strong> <span id="relation-count" style="color: #4A90E2;">-</span></span>
                  </div>
                  
                  <!-- 分隔线 -->
                  <div style="width: 1px; height: 24px; background-color: #444444;"></div>
                  
                  <!-- 控制参数 -->
                  <div style="display: flex; gap: 15px; align-items: center; flex-wrap: wrap;">
                      <div style="display: flex; gap: 8px; align-items: center;">
                          <label for="limit-entities-input" style="color: #aaaaaa; font-size: 14px; margin: 0; white-space: nowrap;">显示实体数:</label>
                          <input type="number" 
                                 id="limit-entities-input" 
                                 class="search-count-input" 
                                 min="1" 
                                 value="50"
                                 placeholder="100"
                                 style="width: 70px;">
                      </div>
                      <div style="display: flex; gap: 8px; align-items: center;">
                          <label for="limit-edges-input" style="color: #aaaaaa; font-size: 14px; margin: 0; white-space: nowrap;">每实体边数:</label>
                          <input type="number" 
                                 id="limit-edges-input"
                                 class="search-count-input" 
                                 min="1" 
                                 value="50"
                                 placeholder="50"
                                 style="width: 70px;">
                      </div>
                      <div style="display: flex; gap: 8px; align-items: center;">
                          <label for="hops-input" style="color: #aaaaaa; font-size: 14px; margin: 0; white-space: nowrap;">跳数:</label>
                          <input type="number" 
                                 id="hops-input"
                                 class="search-count-input" 
                                 min="1" 
                                 value="1"
                                 placeholder="1"
                                 style="width: 70px;">
                      </div>
                  </div>
                  
                  <!-- 分隔线 -->
                  <div style="width: 1px; height: 24px; background-color: #444444;"></div>
                  
                  <!-- 图谱路径 -->
                  <div style="display: flex; gap: 8px; align-items: center; flex: 1; min-width: 200px; max-width: 400px;">
                      <label for="graph-path-input" style="color: #aaaaaa; font-size: 14px; margin: 0; white-space: nowrap;">图谱路径:</label>
                      <input type="text" 
                             id="graph-path-input" 
                             class="search-input" 
                             placeholder="./graph/tmg_storage"
                             style="flex: 1; min-width: 150px;">
                  </div>
                  
                  <!-- 刷新按钮 -->
                  <button class="btn btn-primary btn-refresh" onclick="loadGraph()" style="white-space: nowrap;">🔄 刷新图谱</button>
                  
                  <!-- 最后更新时间 -->
                  <div style="color: #888888; font-size: 12px; white-space: nowrap;">
                      最后更新: <span id="last-update">-</span>
                  </div>
              </div>
          </div>
          
          <div class="search-panel">
              <div class="search-controls">
                  <input type="text" 
                         id="search-input" 
                         class="search-input" 
                         placeholder="输入自然语言查询，例如：人物、事件、概念等..."
                         onkeypress="handleSearchKeyPress(event)">
                  <span class="search-count-label">结果数量:</span>
                  <input type="number" 
                         id="search-count-input" 
                         class="search-count-input" 
                         min="1" 
                         value="1"
                         placeholder="10">
                  <button class="search-btn" id="search-btn" onclick="searchGraph()">🔍 搜索</button>
                  <button class="clear-search-btn" id="clear-search-btn" onclick="clearSearch()" style="display: none;">清除搜索</button>
              </div>
              <div class="search-status" id="search-status"></div>
          </div>
          
          <div id="mynetwork"></div>
      </div>
      
      <!-- 右侧边栏 -->
      <div class="sidebar">
          <div class="sidebar-header">
              <div class="sidebar-title" id="sidebar-title">详细信息</div>
              <div class="sidebar-subtitle" id="sidebar-subtitle">点击节点或边查看详情</div>
          </div>
          <div class="sidebar-content" id="sidebar-content">
              <div class="empty-state">点击图谱中的节点或关系边查看详细信息</div>
          </div>
      </div>

    <script type="text/javascript">
        console.log('=== 图谱可视化脚本开始加载 ===');
        
        // 检查vis-network库是否加载
        if (typeof vis === 'undefined') {
            console.error('❌ vis-network库未加载！请检查CDN链接');
        } else {
            console.log('✅ vis-network库已加载，版本:', vis.Network ? '可用' : '不可用');
        }
        
        var network;
        var container = document.getElementById('mynetwork');
        var nodesDataSet;
        var edges;  // 边数据集，用于版本切换时更新
        
        // 检查容器元素
        if (!container) {
            console.error('❌ 找不到容器元素 #mynetwork');
        } else {
            console.log('✅ 容器元素找到:', container);
            console.log('   容器尺寸:', container.offsetWidth, 'x', container.offsetHeight);
        }
        
        // 跟踪当前模式：'default' 或 'search'
        var currentMode = 'default';
        var currentSearchQuery = '';
        
        function loadGraph() {
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
            console.log('📡 [1/4] 开始加载图谱数据（默认模式）...');
            console.log('   实体数量限制:', limitEntities);
            console.log('   每实体边数限制:', limitEdgesPerEntity);
            console.log('   跳数:', hops);
            console.log('   图谱路径:', graphPath || '(使用默认路径)');
            
            // 构建请求URL
            var url = '/api/graph/data?limit_entities=' + limitEntities + '&limit_edges_per_entity=' + limitEdgesPerEntity + '&hops=' + hops;
            if (graphPath) {
                url += '&storage_path=' + encodeURIComponent(graphPath);
            }
            console.log('   请求URL:', url);
            
            currentMode = 'default';
            currentSearchQuery = '';
            updateSearchStatus('');
            document.getElementById('clear-search-btn').style.display = 'none';
            document.getElementById('search-input').value = '';
            
            fetch(url)
                .then(response => {
                    console.log('📥 [2/4] 收到HTTP响应');
                    console.log('   状态码:', response.status);
                    console.log('   状态文本:', response.statusText);
                    console.log('   Content-Type:', response.headers.get('content-type'));
                    
                    if (!response.ok) {
                        throw new Error('HTTP错误: ' + response.status + ' ' + response.statusText);
                    }
                    
                    return response.json();
                })
                .then(data => {
                    console.log('📦 [3/4] JSON数据解析完成');
                    console.log('   数据键:', Object.keys(data));
                    
                    if (data.success) {
                        console.log('✅ API返回成功');
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
                        console.log('🎨 [4/4] 开始绘制图谱...');
                        drawGraph(data.nodes, data.edges);
                        updateLastUpdate();
                        console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
                    } else {
                        console.error('❌ API返回失败');
                        console.error('   错误信息:', data.error);
                        alert('加载图谱数据失败: ' + data.error);
                    }
                })
                .catch(error => {
                    console.error('❌ 请求过程中发生错误');
                    console.error('   错误类型:', error.name);
                    console.error('   错误消息:', error.message);
                    console.error('   错误堆栈:', error.stack);
                    alert('请求错误: ' + error.message);
                });
        }
        
        function searchGraph() {
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
            console.log('🔍 [1/4] 开始搜索图谱...');
            console.log('   查询文本:', query);
            console.log('   最大结果数:', maxResults);
            console.log('   图谱路径:', graphPath || '(使用默认路径)');
            console.log('   请求URL: /api/graph/search');
            
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
            
            fetch('/api/graph/search', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(requestBody)
            })
                .then(response => {
                    console.log('📥 [2/4] 收到HTTP响应');
                    console.log('   状态码:', response.status);
                    
                    if (!response.ok) {
                        throw new Error('HTTP错误: ' + response.status + ' ' + response.statusText);
                    }
                    
                    return response.json();
                })
                .then(data => {
                    console.log('📦 [3/4] JSON数据解析完成');
                    console.log('   完整响应数据:', JSON.stringify(data, null, 2));
                    console.log('   数据键:', Object.keys(data));
                    
                    document.getElementById('search-btn').disabled = false;
                    
                    if (data.success) {
                        console.log('✅ 搜索成功');
                        console.log('   匹配实体数:', data.stats ? data.stats.matched_entities : 0);
                        console.log('   节点数量:', data.nodes ? data.nodes.length : 0);
                        console.log('   边数量:', data.edges ? data.edges.length : 0);
                        console.log('   统计信息:', data.stats);
                        console.log('   查询文本:', data.query);
                        
                        // 检查是否有节点数据
                        if (!data.nodes || data.nodes.length === 0) {
                            console.warn('⚠️  搜索结果中没有节点数据');
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
                        console.log('🎨 [4/4] 开始绘制搜索结果图谱...');
                        console.log('   准备绘制的节点数:', data.nodes.length);
                        console.log('   准备绘制的边数:', data.edges.length);
                        drawGraph(data.nodes, data.edges);
                        updateLastUpdate();
                        console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
                    } else {
                        console.error('❌ 搜索失败');
                        console.error('   错误信息:', data.error);
                        updateSearchStatus('搜索失败: ' + data.error);
                        alert('搜索失败: ' + data.error);
                    }
                })
                .catch(error => {
                    console.error('❌ 搜索过程中发生错误');
                    console.error('   错误类型:', error.name);
                    console.error('   错误消息:', error.message);
                    console.error('   错误堆栈:', error.stack);
                    document.getElementById('search-btn').disabled = false;
                    updateSearchStatus('搜索错误: ' + error.message);
                    alert('搜索错误: ' + error.message);
                });
        }
        
        function clearSearch() {
            console.log('清除搜索，返回默认视图');
            document.getElementById('search-input').value = '';
            document.getElementById('clear-search-btn').style.display = 'none';
            updateSearchStatus('');
            loadGraph();
        }
        
        function handleSearchKeyPress(event) {
            if (event.key === 'Enter') {
                searchGraph();
            }
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
        
        // 存储边的完整数据
        var edgesDataMap = {};
        
        function drawGraph(nodesData, edgesData) {
            try {
                console.log('┌─ drawGraph 函数开始执行 ──────────────────────────');
                console.log('📊 输入数据检查:');
                console.log('   节点数据类型:', typeof nodesData, Array.isArray(nodesData) ? '(数组)' : '(非数组)');
                console.log('   节点数据长度:', nodesData ? nodesData.length : 'null/undefined');
                console.log('   边数据类型:', typeof edgesData, Array.isArray(edgesData) ? '(数组)' : '(非数组)');
                console.log('   边数据长度:', edgesData ? edgesData.length : 'null/undefined');
                
                if (!nodesData || nodesData.length === 0) {
                    console.error('❌ 节点数据为空或无效');
                    alert('没有节点数据，无法绘制图谱');
                    return;
                }
                
                if (!container) {
                    console.error('❌ 容器元素不存在');
                    alert('找不到图谱容器元素');
                    return;
                }
                console.log('✅ 容器元素检查通过');
                
                // 如果已存在网络，先清除旧数据
                if (network) {
                    console.log('🗑️  清除现有网络数据...');
                    network.destroy();
                    network = null;
                    nodesDataSet = null;
                }
                
                // 创建节点数据集
                console.log('📦 步骤1: 创建节点数据集...');
                try {
                    // 去重：确保每个节点ID只出现一次
                    var uniqueNodes = [];
                    var seenNodeIds = new Set();
                    for (var i = 0; i < nodesData.length; i++) {
                        var node = nodesData[i];
                        if (!seenNodeIds.has(node.id)) {
                            seenNodeIds.add(node.id);
                            uniqueNodes.push(node);
                        } else {
                            console.warn('⚠️  发现重复节点ID，跳过:', node.id, node.label);
                        }
                    }
                    console.log('   去重前节点数:', nodesData.length, '去重后节点数:', uniqueNodes.length);
                    
                    nodesDataSet = new vis.DataSet(uniqueNodes);
                    console.log('✅ 节点数据集创建成功');
                    console.log('   数据集节点数:', nodesDataSet.length);
                    if (nodesDataSet.length > 0) {
                        console.log('   第一个节点示例:', nodesDataSet.get()[0]);
                    }
                } catch (e) {
                    console.error('❌ 创建节点数据集失败:', e);
                    throw e;
                }
                
                // 存储边的完整数据，用于点击时显示
                console.log('🔗 步骤2: 处理边数据...');
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
                    console.log('✅ 边数据映射完成，共', Object.keys(edgesDataMap).length, '条');
                } else {
                    console.log('⚠️  没有边数据');
                }
                
                // 创建边数据集（不显示label，避免糊在一起）
                console.log('🔗 步骤3: 创建边数据集...');
                var edgesForVis = [];
                if (edgesData && edgesData.length > 0) {
                    edgesForVis = edgesData.map(function(edge, index) {
                        if (!edge._visId) {
                            console.warn('⚠️  边[' + index + ']缺少_visId，自动生成:', edge);
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
                    console.log('✅ 边数据集创建成功');
                    console.log('   数据集边数:', edges.length);
                    if (edges.length > 0) {
                        console.log('   第一条边示例:', edges.get()[0]);
                    }
                } catch (e) {
                    console.error('❌ 创建边数据集失败:', e);
                    throw e;
                }
                
                // 创建数据对象
                console.log('📋 步骤4: 组装数据对象...');
                var data = {
                    nodes: nodesDataSet,
                    edges: edges
                };
                console.log('✅ 数据对象创建完成');
                console.log('   节点数:', data.nodes.length);
                console.log('   边数:', data.edges.length);
                
                // 配置选项
                console.log('⚙️  步骤5: 配置选项...');
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
                console.log('✅ 选项配置完成');
                
                // 创建网络
                console.log('🎨 步骤6: 创建vis.Network实例...');
                console.log('   容器:', container);
                console.log('   数据:', { nodes: data.nodes.length, edges: data.edges.length });
                console.log('   选项:', options);
                
                if (typeof vis === 'undefined' || !vis.Network) {
                    throw new Error('vis.Network 不可用，请检查vis-network库是否正确加载');
                }
                
                try {
                    network = new vis.Network(container, data, options);
                    console.log('✅ vis.Network 实例创建成功');
                    console.log('   网络对象:', network);
                    
                    // 监听网络事件
                    network.on("stabilizationEnd", function() {
                        console.log('✅ 网络布局稳定完成');
                    });
                    
                    network.on("stabilizationProgress", function(params) {
                        if (params.iterations % 50 === 0) {
                            console.log('   布局进度:', params.iterations, '/', params.total);
                        }
                    });
                } catch (e) {
                    console.error('❌ 创建vis.Network实例失败:', e);
                    console.error('   错误详情:', e.message);
                    console.error('   错误堆栈:', e.stack);
                    throw e;
                }
                
                // 添加事件监听
                console.log('👂 步骤7: 添加点击事件监听...');
                network.on("click", function(params) {
                    console.log('🖱️  图谱被点击:', params);
                    // 优先判断节点（点击节点时优先显示节点信息）
                    if (params.nodes.length > 0) {
                        var nodeId = params.nodes[0];
                        var node = nodesDataSet.get(nodeId);
                        if (node) {
                            console.log('   点击了节点:', nodeId, node.label);
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
                        // 点击空白处，重置侧边栏
                        console.log('   点击了空白处');
                        resetSidebar();
                    }
                });
                console.log('✅ 事件监听添加完成');
                console.log('└─ drawGraph 函数执行完成 ──────────────────────────');
            } catch (error) {
                console.error('❌ 绘制图谱时发生错误');
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
        
        // 存储当前显示的实体和关系信息
        var currentEntityId = null;
        var currentEntityVersions = null;
        var currentEntityAbsoluteId = null;
        var currentRelationId = null;
        var currentRelationVersions = null;
        var currentRelationAbsoluteId = null;
        
        function showNodeDetail(node) {
            document.getElementById('sidebar-title').textContent = '实体详情';
            document.getElementById('sidebar-subtitle').textContent = node.label;
            
            currentEntityId = node.entity_id || node.id;
            currentEntityAbsoluteId = node.absolute_id || node.id;
            currentEntityVersions = null;
            currentRelationId = null;
            currentRelationVersions = null;
            currentRelationAbsoluteId = null;
            
            // 通过API获取完整信息（包括embedding_preview和memory_cache_text）
            if (node.absolute_id) {
                fetch('/api/entity/' + encodeURIComponent(currentEntityId) + '/version/' + encodeURIComponent(node.absolute_id))
                    .then(response => response.json())
                    .then(data => {
                        if (data.success) {
                            var entity = data.entity;
                            // 更新显示
                            var html = renderVersionSelector('entity', currentEntityId, currentEntityAbsoluteId, null);
                            html += renderEntityDetail({
                                entity_id: entity.entity_id,
                                absolute_id: entity.absolute_id,
                                id: entity.entity_id,
                                name: entity.name,
                                label: entity.name,
                                content: entity.content,
                                physical_time: entity.physical_time,
                                memory_cache_content: entity.memory_cache_content,
                                memory_cache_text: entity.memory_cache_text,
                                doc_name: entity.doc_name,
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
                    })
                    .catch(error => {
                        console.error('获取实体详情失败:', error);
                        // 如果API失败，使用节点数据
                        var html = renderVersionSelector('entity', currentEntityId, currentEntityAbsoluteId, null);
                        html += renderEntityDetail(node);
                        document.getElementById('sidebar-content').innerHTML = html;
                        loadEntityVersions(currentEntityId);
                    });
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
            html += '<div class="version-selector-title">📋 版本选择</div>';
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
            html += '<div class="detail-section-title">📌 实体 ID | 绝对 ID</div>';
            html += '<div class="detail-content">';
            html += '<div style="margin-bottom: 5px;">实体ID: ' + escapeHtml(entity.entity_id || entity.id) + '</div>';
            if (entity.absolute_id) {
                html += '<div>绝对ID: ' + escapeHtml(entity.absolute_id) + '</div>';
            }
            html += '</div>';
            html += '</div>';
            
            html += '<div class="detail-section">';
            html += '<div class="detail-section-title">📝 实体名称</div>';
            html += '<div class="detail-content">' + escapeHtml(entity.name || entity.label) + '</div>';
            html += '</div>';
            
            html += '<div class="detail-section">';
            html += '<div class="detail-section-title">📄 实体描述</div>';
            html += '<div class="detail-content">' + escapeHtml(entity.content || entity.title || entity.label) + '</div>';
            html += '</div>';
            
            // 时间信息
            if (entity.physical_time) {
                html += '<div class="detail-section">';
                html += '<div class="detail-section-title">🕐 创建时间</div>';
                html += '<div class="detail-content">';
                try {
                    var time = new Date(entity.physical_time);
                    html += '<div>' + time.toLocaleString('zh-CN') + '</div>';
                    html += '<div style="color: #888888; font-size: 12px; margin-top: 5px;">' + time.toISOString() + '</div>';
                } catch (e) {
                    html += '<div>' + escapeHtml(entity.physical_time) + '</div>';
                }
                html += '</div>';
                html += '</div>';
            }
            
            // 缓存记忆（memory_cache_id对应的md文档内容）
            if (entity.memory_cache_content) {
                html += '<div class="detail-section">';
                html += '<div class="detail-section-title">💾 缓存记忆</div>';
                html += '<div class="detail-content">';
                html += '<div style="max-height: 200px; overflow-y: auto; font-size: 13px; line-height: 1.5; white-space: pre-wrap; word-wrap: break-word;">';
                html += escapeHtml(entity.memory_cache_content);
                html += '</div>';
                html += '</div>';
                html += '</div>';
            }
            
            // 原文内容（memory_cache_id对应json中的text内容）
            if (entity.memory_cache_text) {
                html += '<div class="detail-section">';
                html += '<div class="detail-section-title">📄 原文内容</div>';
                html += '<div class="detail-content">';
                html += '<div style="max-height: 200px; overflow-y: auto; font-size: 13px; line-height: 1.5; white-space: pre-wrap; word-wrap: break-word;">';
                html += escapeHtml(entity.memory_cache_text);
                html += '</div>';
                html += '</div>';
                html += '</div>';
            }
            
            // 文档名称
            if (entity.doc_name) {
                html += '<div class="detail-section">';
                html += '<div class="detail-section-title">📁 文档名称</div>';
                html += '<div class="detail-content">' + escapeHtml(entity.doc_name) + '</div>';
                html += '</div>';
            }
            
            // Embedding向量前4个值
            if (entity.embedding_preview && Array.isArray(entity.embedding_preview)) {
                html += '<div class="detail-section">';
                html += '<div class="detail-section-title">🔢 编码向量（前4个值）</div>';
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
        
        function loadEntityVersions(entityId) {
            fetch('/api/entity/' + encodeURIComponent(entityId) + '/versions')
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        currentEntityVersions = data.versions;
                        updateVersionSelector('entity', data.versions, currentEntityAbsoluteId);
                    } else {
                        console.error('加载实体版本失败:', data.error);
                        document.getElementById('version-info-entity').textContent = '加载失败';
                    }
                })
                .catch(error => {
                    console.error('请求错误:', error);
                    document.getElementById('version-info-entity').textContent = '加载失败';
                });
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
                option.textContent = '版本 ' + version.index + '/' + version.total + ' (' + formatDateTime(version.physical_time) + ')';
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
        
        function switchEntityVersion(entityId, absoluteId) {
            fetch('/api/entity/' + encodeURIComponent(entityId) + '/version/' + encodeURIComponent(absoluteId))
                .then(response => response.json())
                .then(data => {
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
                            physical_time: entity.physical_time,
                            memory_cache_content: entity.memory_cache_content,
                            memory_cache_text: entity.memory_cache_text,
                            doc_name: entity.doc_name,
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
                        updateGraphForEntityVersion(entityId, absoluteId, entity.physical_time);
                    } else {
                        console.error('切换实体版本失败:', data.error);
                        alert('切换版本失败: ' + data.error);
                    }
                })
                .catch(error => {
                    console.error('请求错误:', error);
                    alert('请求错误: ' + error);
                });
        }
        
        function updateGraphForEntityVersion(entityId, absoluteId, timePoint) {
            if (!network || !nodesDataSet) return;
            
            console.log('🔄 更新图谱以反映实体版本变化:', entityId, absoluteId);
            console.log('   版本ID:', absoluteId);
            
            // 获取当前的限制参数
            var limitEntities = parseInt(document.getElementById('limit-entities-input').value) || 100;
            var limitEdgesPerEntity = parseInt(document.getElementById('limit-edges-input').value) || 50;
            var hops = parseInt(document.getElementById('hops-input').value) || 1;
            
            // 重新加载图谱，以该实体版本为中心，显示从最早版本到该版本的所有关系
            // 只需要传递focus_entity_id和focus_absolute_id，后端会自动根据版本确定时间点
            // 使用limit_edges_per_entity参数控制每个实体显示的关系边数量
            // 使用hops参数控制跳数
            currentMode = 'version_snapshot';
            currentSearchQuery = '';
            updateSearchStatus('显示实体版本: ' + entityId + ' (到版本 ' + absoluteId.substring(0, 8) + '...) - ' + hops + '跳 - 每实体最多' + limitEdgesPerEntity + '条关系');
            
            // 使用focus_entity_id和focus_absolute_id参数，以该实体版本为中心显示图谱
            // 使用limit_edges_per_entity参数控制关系边数量
            // 使用hops参数控制跳数
            var url = '/api/graph/data?limit_entities=' + limitEntities + 
                      '&limit_edges_per_entity=' + limitEdgesPerEntity + 
                      '&hops=' + hops +
                      '&focus_entity_id=' + encodeURIComponent(entityId) + 
                      '&focus_absolute_id=' + encodeURIComponent(absoluteId);
            
            fetch(url)
                .then(response => {
                    if (!response.ok) {
                        throw new Error('HTTP错误: ' + response.status + ' ' + response.statusText);
                    }
                    return response.json();
                })
                .then(data => {
                    if (data.success) {
                        console.log('✅ 获取实体版本图谱成功');
                        console.log('   节点数量:', data.nodes ? data.nodes.length : 0);
                        console.log('   边数量:', data.edges ? data.edges.length : 0);
                        
                        updateStats(data.stats);
                        drawGraph(data.nodes, data.edges);
                        updateLastUpdate();
                    } else {
                        console.error('❌ 获取实体版本图谱失败:', data.error);
                        alert('获取实体版本图谱失败: ' + data.error);
                    }
                })
                .catch(error => {
                    console.error('❌ 请求错误:', error);
                    alert('请求错误: ' + error.message);
                });
        }
        
        function updateGraphForRelationVersion(relationId, absoluteId, timePoint, fromEntityId, toEntityId) {
            if (!network || !nodesDataSet || !edges) return;
            
            console.log('🔄 更新图谱以反映关系版本变化:', relationId, absoluteId, timePoint);
            console.log('   时间点:', timePoint);
            
            // 获取当前的限制参数
            var limitEntities = parseInt(document.getElementById('limit-entities-input').value) || 100;
            var limitEdgesPerEntity = parseInt(document.getElementById('limit-edges-input').value) || 50;
            
            // 重新加载图谱，但只显示该时间点之前的数据
            currentMode = 'version_snapshot';
            currentSearchQuery = '';
            updateSearchStatus('显示时间点: ' + new Date(timePoint).toLocaleString('zh-CN'));
            
            fetch('/api/graph/data?limit_entities=' + limitEntities + '&limit_edges_per_entity=' + limitEdgesPerEntity + '&time_point=' + encodeURIComponent(timePoint))
                .then(response => {
                    if (!response.ok) {
                        throw new Error('HTTP错误: ' + response.status + ' ' + response.statusText);
                    }
                    return response.json();
                })
                .then(data => {
                    if (data.success) {
                        console.log('✅ 获取时间点快照成功');
                        console.log('   节点数量:', data.nodes ? data.nodes.length : 0);
                        console.log('   边数量:', data.edges ? data.edges.length : 0);
                        
                        updateStats(data.stats);
                        drawGraph(data.nodes, data.edges);
                        updateLastUpdate();
                    } else {
                        console.error('❌ 获取时间点快照失败:', data.error);
                        alert('获取时间点快照失败: ' + data.error);
                    }
                })
                .catch(error => {
                    console.error('❌ 请求错误:', error);
                    alert('请求错误: ' + error.message);
                });
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
        
        function showEdgeDetail(edgeData) {
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
                fetch('/api/relation/' + encodeURIComponent(currentRelationId) + '/version/' + encodeURIComponent(edgeData.absolute_id))
                    .then(response => response.json())
                    .then(data => {
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
                    })
                    .catch(error => {
                        console.error('获取embedding失败:', error);
                    });
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
            html += '<div class="detail-section-title">🔗 关系描述</div>';
            // 优先使用完整的content，如果没有则使用title
            html += '<div class="detail-content">' + escapeHtml(edgeData.content || edgeData.title || '无描述') + '</div>';
            html += '</div>';
            
            // 起点实体信息
            html += '<div class="detail-section">';
            html += '<div class="detail-section-title">🎯 起点实体</div>';
            html += '<div class="detail-content">';
            html += '<strong>实体 ID:</strong> ' + escapeHtml(fromId) + '<br>';
            html += '<strong>实体名称:</strong> ' + escapeHtml(fromName) + '<br><br>';
            html += '<strong>实体描述:</strong><br>' + escapeHtml(fromContent);
            html += '</div>';
            html += '</div>';
            
            // 终点实体信息
            html += '<div class="detail-section">';
            html += '<div class="detail-section-title">🎯 终点实体</div>';
            html += '<div class="detail-content">';
            html += '<strong>实体 ID:</strong> ' + escapeHtml(toId) + '<br>';
            html += '<strong>实体名称:</strong> ' + escapeHtml(toName) + '<br><br>';
            html += '<strong>实体描述:</strong><br>' + escapeHtml(toContent);
            html += '</div>';
            html += '</div>';
            
            // 时间信息
            if (edgeData.physical_time) {
                html += '<div class="detail-section">';
                html += '<div class="detail-section-title">🕐 创建时间</div>';
                html += '<div class="detail-content">';
                try {
                    var time = new Date(edgeData.physical_time);
                    html += '<div>' + time.toLocaleString('zh-CN') + '</div>';
                    html += '<div style="color: #888888; font-size: 12px; margin-top: 5px;">' + time.toISOString() + '</div>';
                } catch (e) {
                    html += '<div>' + escapeHtml(edgeData.physical_time) + '</div>';
                }
                html += '</div>';
                html += '</div>';
            }
            
            // Embedding向量前5个值
            if (edgeData.embedding_preview && Array.isArray(edgeData.embedding_preview)) {
                html += '<div class="detail-section">';
                html += '<div class="detail-section-title">🔢 Embedding向量（前5个值）</div>';
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
        
        function loadRelationVersions(relationId) {
            fetch('/api/relation/' + encodeURIComponent(relationId) + '/versions')
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        currentRelationVersions = data.versions;
                        updateVersionSelector('relation', data.versions, currentRelationAbsoluteId);
                    } else {
                        console.error('加载关系版本失败:', data.error);
                        document.getElementById('version-info-relation').textContent = '加载失败';
                    }
                })
                .catch(error => {
                    console.error('请求错误:', error);
                    document.getElementById('version-info-relation').textContent = '加载失败';
                });
        }
        
        function switchRelationVersion(relationId, absoluteId) {
            fetch('/api/relation/' + encodeURIComponent(relationId) + '/version/' + encodeURIComponent(absoluteId))
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        currentRelationAbsoluteId = absoluteId;
                        var relation = data.relation;
                        
                        // 更新图谱中的关系边和连接的实体
                        updateGraphForRelationVersion(relationId, absoluteId, relation.physical_time, relation.from_entity_id, relation.to_entity_id);
                        
                        // 获取实体信息 - 优先使用API返回的实体名称，如果没有则从当前图谱中获取
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
                        
                        // 获取实体内容 - 如果节点不存在，尝试通过API获取
                        var fromContent = fromNode ? (fromNode.content || fromNode.title || fromNode.label) : '未知实体';
                        var toContent = toNode ? (toNode.content || toNode.title || toNode.label) : '未知实体';
                        
                        // 如果节点不存在，尝试通过API获取实体信息
                        var entityPromises = [];
                        if (!fromNode && relation.from_entity_id) {
                            entityPromises.push(
                                fetch('/api/entity/' + encodeURIComponent(relation.from_entity_id) + '/versions')
                                    .then(response => response.json())
                                    .then(data => {
                                        if (data.success && data.versions.length > 0) {
                                            return { type: 'from', content: data.versions[0].content || fromName };
                                        }
                                        return null;
                                    })
                                    .catch(() => null)
                            );
                        } else {
                            entityPromises.push(Promise.resolve(null));
                        }
                        
                        if (!toNode && relation.to_entity_id) {
                            entityPromises.push(
                                fetch('/api/entity/' + encodeURIComponent(relation.to_entity_id) + '/versions')
                                    .then(response => response.json())
                                    .then(data => {
                                        if (data.success && data.versions.length > 0) {
                                            return { type: 'to', content: data.versions[0].content || toName };
                                        }
                                        return null;
                                    })
                                    .catch(() => null)
                            );
                        } else {
                            entityPromises.push(Promise.resolve(null));
                        }
                        
                        // 等待所有实体信息加载完成后再更新显示
                        Promise.all(entityPromises).then(function(results) {
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
                                physical_time: relation.physical_time,
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
                        });
                    } else {
                        console.error('切换关系版本失败:', data.error);
                        alert('切换版本失败: ' + data.error);
                    }
                })
                .catch(error => {
                    console.error('请求错误:', error);
                    alert('请求错误: ' + error);
                });
        }
        
        // HTML 转义
        function escapeHtml(text) {
            if (!text) return '';
            var div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }
        
        // 页面加载时自动加载图谱
        window.onload = function() {
            console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
            console.log('🌐 页面加载完成');
            console.log('   当前URL:', window.location.href);
            console.log('   页面标题:', document.title);
            console.log('   容器元素:', container ? '找到' : '未找到');
            console.log('   vis库状态:', typeof vis !== 'undefined' ? '已加载' : '未加载');
            console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
            
            // 获取当前配置并设置默认路径
            fetch('/api/graph/config')
                .then(response => response.json())
                .then(data => {
                    if (data.success && data.storage_path) {
                        var pathInput = document.getElementById('graph-path-input');
                        if (pathInput && !pathInput.value.trim()) {
                            pathInput.value = data.storage_path;
                            console.log('✅ 已设置默认图谱路径:', data.storage_path);
                        }
                    }
                    console.log('🚀 开始自动加载图谱...');
                    loadGraph();
                })
                .catch(error => {
                    console.warn('⚠️  获取配置失败，使用默认设置:', error);
                    console.log('🚀 开始自动加载图谱...');
                    loadGraph();
                });
        };
        
        console.log('✅ 图谱可视化脚本加载完成');
    </script>
</body>
</html>
"""


class GraphWebServer:
    """图谱可视化 Web 服务器"""
    
    def __init__(self, storage_path: str = "./graph/tmg_storage", port: int = 5000,
                 embedding_model_path: Optional[str] = None,
                 embedding_model_name: Optional[str] = None,
                 embedding_device: str = "cpu",
                 embedding_use_local: bool = True):
        """
        初始化 Web 服务器
        
        Args:
            storage_path: 存储路径
            port: 服务器端口
            embedding_model_path: 本地embedding模型路径（优先使用）
            embedding_model_name: HuggingFace embedding模型名称
            embedding_device: 计算设备 ("cpu" 或 "cuda")
            embedding_use_local: 是否优先使用本地模型
        """
        self.storage_path = storage_path
        self.port = port
        self.app = Flask(__name__)
        
        # 初始化embedding客户端
        self.embedding_client = EmbeddingClient(
            model_path=embedding_model_path,
            model_name=embedding_model_name,
            device=embedding_device,
            use_local=embedding_use_local
        )
        
        # 初始化存储和可视化器
        self.storage = StorageManager(storage_path, embedding_client=self.embedding_client)
        self.visualizer = GraphVisualizer(self.storage)
        
        # 缓存当前使用的存储路径（用于路径切换检测）
        self._current_storage_path = storage_path
        
        # 设置路由
        self._setup_routes()
    
    def _switch_storage_path(self, new_path: str):
        """
        切换存储路径
        
        Args:
            new_path: 新的存储路径
        """
        if new_path != self._current_storage_path:
            try:
                # 重新初始化存储和可视化器
                self.storage = StorageManager(new_path, embedding_client=self.embedding_client)
                self.visualizer = GraphVisualizer(self.storage)
                self._current_storage_path = new_path
                print(f"✅ 已切换到新的存储路径: {new_path}")
            except Exception as e:
                print(f"❌ 切换存储路径失败: {str(e)}")
                raise
    
    def _setup_routes(self):
        """设置 Flask 路由"""
        
        @self.app.route('/')
        def index():
            """主页"""
            return render_template_string(HTML_TEMPLATE)
        
        @self.app.route('/api/graph/data')
        def get_graph_data():
            """获取图谱数据 API
            
            支持参数:
            - limit_entities: 限制返回的实体数量（默认100）
            - limit_edges_per_entity: 每个实体最多返回的关系边数量（默认50）
            - time_point: ISO格式的时间点（可选），如果提供，只返回该时间点之前或等于该时间点的数据
            - storage_path: 图谱存储路径（可选），如果提供且与当前路径不同，会切换存储路径
            - focus_entity_id: 聚焦的实体ID（可选），如果提供，只显示该实体从最早版本到指定版本的所有关系
            - focus_absolute_id: 聚焦的实体版本absolute_id（可选），需要与focus_entity_id一起使用
            - hops: 跳数（默认1），在focus模式下，表示要显示多少层关联实体和关系
            """
            try:
                from flask import request
                from datetime import datetime
                
                # 获取参数
                limit_entities = request.args.get('limit_entities', type=int, default=100)
                limit_edges_per_entity = request.args.get('limit_edges_per_entity', type=int, default=50)
                time_point_str = request.args.get('time_point')
                storage_path_param = request.args.get('storage_path')
                focus_entity_id = request.args.get('focus_entity_id')
                focus_absolute_id = request.args.get('focus_absolute_id')
                hops = request.args.get('hops', type=int, default=1)
                
                # 如果提供了存储路径参数，且与当前路径不同，则切换存储路径
                if storage_path_param and storage_path_param.strip():
                    storage_path_param = storage_path_param.strip()
                    try:
                        self._switch_storage_path(storage_path_param)
                    except Exception as e:
                        return jsonify({
                            'success': False,
                            'error': f'切换存储路径失败: {str(e)}'
                        }), 400
                
                time_point = None
                if time_point_str:
                    try:
                        time_point = datetime.fromisoformat(time_point_str)
                    except:
                        pass
                
                # 如果指定了focus_entity_id和focus_absolute_id，以该实体为中心显示图谱
                if focus_entity_id and focus_absolute_id:
                    # 获取聚焦实体的指定版本
                    focus_entity = self.storage.get_entity_by_absolute_id(focus_absolute_id)
                    if not focus_entity or focus_entity.entity_id != focus_entity_id:
                        return jsonify({
                            'success': False,
                            'error': f'未找到指定的实体版本: {focus_entity_id}/{focus_absolute_id}'
                        }), 404
                    
                    # 只显示该实体从最早版本到指定版本的所有关系
                    # 时间点自动从该版本获取，不需要单独传递time_point参数
                    entities = [focus_entity]
                    focus_time_point = focus_entity.physical_time
                else:
                    # 获取最近更新的实体（限制数量）
                    if time_point:
                        # 根据时间点获取实体
                        entities = self.storage.get_all_entities_before_time(time_point, limit=limit_entities)
                    else:
                        entities = self.storage.get_all_entities(limit=limit_entities)
                    
                    if not entities:
                        return jsonify({
                            'success': False,
                            'error': '没有找到实体数据'
                        })
                
                # 收集所有需要显示的实体ID（初始实体 + 关联实体）
                entity_absolute_ids = {entity.id for entity in entities}
                entity_id_to_name = {}
                entity_id_to_absolute_id = {}
                
                # 定义跳数颜色映射函数
                def get_hop_color(hop_level):
                    """根据跳数层级返回对应的颜色"""
                    colors = [
                        '#4A90E2',  # 第0跳（focus实体）：蓝色
                        '#E67E22',  # 第1跳：橙色
                        '#27AE60',  # 第2跳：绿色
                        '#9B59B6',  # 第3跳：紫色
                        '#E74C3C',  # 第4跳：红色
                        '#F39C12',  # 第5跳：黄色
                        '#1ABC9C',  # 第6跳：青色
                        '#34495E',  # 第7跳：深灰色
                    ]
                    # 如果跳数超过预定义颜色数量，循环使用
                    return colors[hop_level % len(colors)]
                
                # 记录每个实体所在的跳数层级（用于颜色区分）
                entity_id_to_hop_level = {}
                
                # 构建初始节点数据
                nodes = []
                for entity in entities:
                    try:
                        # 防御性检查：确保实体有必要的属性
                        if not entity or not hasattr(entity, 'entity_id') or not hasattr(entity, 'id'):
                            print(f"⚠️  跳过无效实体: {entity}")
                            continue
                        
                        entity_id_to_name[entity.entity_id] = entity.name
                        entity_id_to_absolute_id[entity.entity_id] = entity.id
                        # focus实体是第0跳
                        if focus_entity_id and focus_entity_id == entity.entity_id:
                            entity_id_to_hop_level[entity.entity_id] = 0
                        
                        # 获取版本数量
                        try:
                            versions = self.storage.get_entity_versions(entity.entity_id)
                            version_count = len(versions) if versions else 0
                        except Exception as e:
                            print(f"⚠️  获取实体版本失败 (entity_id={entity.entity_id}): {str(e)}")
                            versions = []
                            version_count = 1  # 默认值
                        
                        # 在focus模式下，显示当前版本索引（如 "实体名 (3/5版本)"）
                        # 否则显示总版本数（如 "实体名 (5版本)"）
                        if focus_entity_id and focus_entity_id == entity.entity_id and focus_absolute_id:
                            # 找到当前版本在版本列表中的索引（从1开始）
                            # 版本列表按时间倒序排列（最新版本在前），需要反转后计算索引
                            try:
                                versions_sorted = sorted(versions, key=lambda v: v.physical_time)
                                current_version_index = None
                                for idx, v in enumerate(versions_sorted, 1):
                                    if v.id == focus_absolute_id:
                                        current_version_index = idx
                                        break
                                
                                if current_version_index:
                                    label = f"{entity.name} ({current_version_index}/{version_count}版本)" if version_count > 1 else entity.name
                                else:
                                    label = f"{entity.name} ({version_count}版本)" if version_count > 1 else entity.name
                            except Exception as e:
                                print(f"⚠️  处理版本索引时出错 (entity_id={entity.entity_id}): {str(e)}")
                                label = f"{entity.name} ({version_count}版本)" if version_count > 1 else entity.name
                        else:
                            # 在标签中显示版本数量
                            label = f"{entity.name} ({version_count}版本)" if version_count > 1 else entity.name
                        
                        # 根据跳数设置颜色
                        hop_level = entity_id_to_hop_level.get(entity.entity_id, 0)
                        node_color = get_hop_color(hop_level)
                        
                        # 安全地处理physical_time
                        try:
                            physical_time_str = entity.physical_time.isoformat() if entity.physical_time else None
                        except Exception as e:
                            print(f"⚠️  实体时间格式错误 (entity_id={entity.entity_id}): {str(e)}")
                            physical_time_str = None
                        
                        # 安全地处理content
                        content = entity.content if hasattr(entity, 'content') and entity.content else ''
                        name = entity.name if hasattr(entity, 'name') and entity.name else '未知实体'
                        
                        nodes.append({
                            'id': entity.entity_id,
                            'entity_id': entity.entity_id,
                            'absolute_id': entity.id,
                            'label': label,
                            'title': f"{name}\n\n{content[:100]}..." if len(content) > 100 else f"{name}\n\n{content}",
                            'content': content,
                            'physical_time': physical_time_str,
                            'version_count': version_count,
                            'color': node_color,
                            'shape': 'dot',
                            'size': 20,
                            'font': {'color': 'white'}
                        })
                    except Exception as e:
                        import traceback
                        print(f"⚠️  处理实体时发生错误 (entity={entity}): {str(e)}")
                        print(traceback.format_exc())
                        continue  # 跳过有问题的实体，继续处理其他实体
                
                # 为每个实体获取关系边（限制数量）
                edges = []
                edges_seen = set()  # 用于去重，使用 (from_id, to_id, relation_id) 作为唯一标识
                all_related_entity_ids = set()
                
                # 辅助函数：统计实体拥有的关系边数量（去重后）
                def count_entity_relations(entity_id, max_abs_id=None):
                    """统计实体拥有的关系边数量（去重后）"""
                    if max_abs_id:
                        entity_abs_ids = self.storage.get_entity_absolute_ids_up_to_version(entity_id, max_abs_id)
                    else:
                        versions = self.storage.get_entity_versions(entity_id)
                        entity_abs_ids = [v.id for v in versions]
                    
                    if not entity_abs_ids:
                        return 0
                    
                    relations = self.storage.get_relations_by_entity_absolute_ids(entity_abs_ids, limit=None)
                    # 按 relation_id 去重
                    unique_relation_ids = set(r.relation_id for r in relations)
                    return len(unique_relation_ids)
                
                # 在focus模式下，实现多跳逻辑
                # 关键：从最早版本到当前版本的所有 absolute_id 关联的关系边
                if focus_entity_id and focus_absolute_id and hops > 0:
                    # ========== 第一步：收集所有边和节点，确定最终要显示的图 ==========
                    # 存储最终确定要显示的边信息
                    final_edge_candidates = []  # 存储边的完整信息
                    graph_edges_for_bfs = []  # 仅用于 BFS 的边列表 (entity1_id, entity2_id)
                    graph_nodes = set()  # 存储所有节点
                    graph_nodes.add(focus_entity_id)
                    
                    # 递归获取多跳的实体和关系
                    current_level_entities = {focus_entity_id: focus_absolute_id}
                    processed_entity_ids = set()
                    
                    for current_hop in range(1, hops + 1):
                        next_level_entities = {}
                        
                        for entity_id, max_abs_id in current_level_entities.items():
                            if entity_id in processed_entity_ids:
                                continue
                            processed_entity_ids.add(entity_id)
                            
                            # 获取该实体从最早版本到当前版本的所有 absolute_id
                            entity_abs_ids = self.storage.get_entity_absolute_ids_up_to_version(
                                entity_id, max_abs_id
                            )
                            
                            if not entity_abs_ids:
                                continue
                            
                            # 获取这些 absolute_id 关联的所有关系边
                            entity_relations = self.storage.get_relations_by_entity_absolute_ids(
                                entity_abs_ids, 
                                limit=None
                            )
                            
                            # 收集关系边和对应的另一端实体信息，用于排序
                            relation_candidates = []
                            
                            for relation in entity_relations:
                                entity1 = self.storage.get_entity_by_absolute_id(relation.entity1_absolute_id)
                                entity2 = self.storage.get_entity_by_absolute_id(relation.entity2_absolute_id)
                                
                                if entity1 and entity2:
                                    entity1_id = entity1.entity_id
                                    entity2_id = entity2.entity_id
                                    
                                    normalized_pair = LLMClient._normalize_entity_pair(entity1_id, entity2_id)
                                    normalized_entity1_id = normalized_pair[0]
                                    normalized_entity2_id = normalized_pair[1]
                                    
                                    edge_key = (normalized_entity1_id, normalized_entity2_id, relation.id)
                                    if edge_key not in edges_seen:
                                        other_entity = entity2 if relation.entity1_absolute_id in entity_abs_ids else entity1
                                        other_entity_id = other_entity.entity_id
                                        other_entity_abs_id = other_entity.id
                                        
                                        other_entity_edge_count = count_entity_relations(other_entity_id, other_entity_abs_id)
                                        
                                        relation_candidates.append({
                                            'relation': relation,
                                            'entity1': entity1,
                                            'entity2': entity2,
                                            'normalized_entity1_id': normalized_entity1_id,
                                            'normalized_entity2_id': normalized_entity2_id,
                                            'edge_key': edge_key,
                                            'other_entity': other_entity,
                                            'other_entity_id': other_entity_id,
                                            'other_entity_abs_id': other_entity_abs_id,
                                            'other_entity_edge_count': other_entity_edge_count
                                        })
                            
                            # 按照另一端实体的边数从多到少排序
                            relation_candidates.sort(key=lambda x: x['other_entity_edge_count'], reverse=True)
                            
                            # 应用 limit_edges_per_entity 限制
                            if limit_edges_per_entity:
                                relation_candidates = relation_candidates[:limit_edges_per_entity]
                            
                            # 将选中的边加入最终结果
                            for candidate in relation_candidates:
                                edges_seen.add(candidate['edge_key'])
                                final_edge_candidates.append(candidate)
                                
                                # 记录图结构用于 BFS
                                graph_edges_for_bfs.append((
                                    candidate['normalized_entity1_id'], 
                                    candidate['normalized_entity2_id']
                                ))
                                graph_nodes.add(candidate['normalized_entity1_id'])
                                graph_nodes.add(candidate['normalized_entity2_id'])
                                
                                # 更新实体名称映射
                                entity1 = candidate['entity1']
                                entity2 = candidate['entity2']
                                normalized_entity1_id = candidate['normalized_entity1_id']
                                normalized_entity2_id = candidate['normalized_entity2_id']
                                
                                if normalized_entity1_id not in entity_id_to_name:
                                    entity_id_to_name[normalized_entity1_id] = entity1.name
                                    entity_id_to_absolute_id[normalized_entity1_id] = entity1.id
                                    all_related_entity_ids.add(normalized_entity1_id)
                                
                                if normalized_entity2_id not in entity_id_to_name:
                                    entity_id_to_name[normalized_entity2_id] = entity2.name
                                    entity_id_to_absolute_id[normalized_entity2_id] = entity2.id
                                    all_related_entity_ids.add(normalized_entity2_id)
                                
                                # 下一跳
                                other_entity = candidate['other_entity']
                                other_entity_id = candidate['other_entity_id']
                                other_entity_abs_id = candidate['other_entity_abs_id']
                                
                                if current_hop < hops and other_entity_id not in processed_entity_ids:
                                    if other_entity_id in next_level_entities:
                                        existing_abs_id = next_level_entities[other_entity_id]
                                        existing_entity = self.storage.get_entity_by_absolute_id(existing_abs_id)
                                        if existing_entity and other_entity.physical_time > existing_entity.physical_time:
                                            next_level_entities[other_entity_id] = other_entity_abs_id
                                    else:
                                        next_level_entities[other_entity_id] = other_entity_abs_id
                        
                        current_level_entities = next_level_entities
                        if not current_level_entities:
                            break
                    
                    # ========== 第二步：基于最终确定的图计算最短路径 ==========
                    def bfs_shortest_paths(start_node, edges_list):
                        """使用BFS计算从起始节点到所有节点的最短路径长度"""
                        # 构建邻接表
                        graph = {}
                        for u, v in edges_list:
                            if u not in graph:
                                graph[u] = []
                            if v not in graph:
                                graph[v] = []
                            graph[u].append(v)
                            graph[v].append(u)
                        
                        # BFS
                        distances = {start_node: 0}
                        queue = [start_node]
                        
                        while queue:
                            current = queue.pop(0)
                            if current not in graph:
                                continue
                            
                            for neighbor in graph[current]:
                                if neighbor not in distances:
                                    distances[neighbor] = distances[current] + 1
                                    queue.append(neighbor)
                        
                        return distances
                    
                    # 计算最短路径
                    shortest_paths = bfs_shortest_paths(focus_entity_id, graph_edges_for_bfs)
                    
                    # 根据最短路径长度设置跳数层级
                    for entity_id in graph_nodes:
                        if entity_id in shortest_paths:
                            entity_id_to_hop_level[entity_id] = shortest_paths[entity_id]
                        else:
                            entity_id_to_hop_level[entity_id] = 999
                    
                    # ========== 第三步：生成最终的边列表 ==========
                    for candidate in final_edge_candidates:
                        relation = candidate['relation']
                        normalized_entity1_id = candidate['normalized_entity1_id']
                        normalized_entity2_id = candidate['normalized_entity2_id']
                        
                        edge_label = ""
                        if relation.content:
                            edge_label = relation.content[:30] + "..." if len(relation.content) > 30 else relation.content
                        
                        edges.append({
                            'from': normalized_entity1_id,
                            'to': normalized_entity2_id,
                            'label': edge_label,
                            'title': relation.content,
                            'content': relation.content,
                            'physical_time': relation.physical_time.isoformat(),
                            'relation_id': relation.relation_id,
                            'absolute_id': relation.id,
                            'color': '#888888',
                            'width': 2,
                            'arrows': ''
                        })
                else:
                    # 非focus模式或hops=0，使用原来的单层逻辑，但也要按另一端实体的边数排序
                    for entity in entities:
                        max_version_absolute_id = focus_absolute_id if (focus_entity_id and focus_entity_id == entity.entity_id) else None
                        effective_time_point = None if max_version_absolute_id else time_point
                        
                        # 先获取所有关系边，不限制数量，用于排序
                        entity_relations = self.storage.get_entity_relations_by_entity_id(
                            entity.entity_id, 
                            limit=None, 
                            time_point=effective_time_point,
                            max_version_absolute_id=max_version_absolute_id
                        )
                        
                        # 收集关系边和对应的另一端实体信息，用于排序
                        relation_candidates = []
                        
                        for relation in entity_relations:
                            entity1_temp = self.storage.get_entity_by_absolute_id(relation.entity1_absolute_id)
                            entity2_temp = self.storage.get_entity_by_absolute_id(relation.entity2_absolute_id)
                            
                            if entity1_temp and entity2_temp:
                                effective_time_point = focus_time_point if focus_entity_id else time_point
                                if effective_time_point:
                                    entity1 = self.storage.get_entity_version_at_time(entity1_temp.entity_id, effective_time_point)
                                    entity2 = self.storage.get_entity_version_at_time(entity2_temp.entity_id, effective_time_point)
                                else:
                                    entity1 = entity1_temp
                                    entity2 = entity2_temp
                                
                                if entity1 and entity2:
                                    entity1_id = entity1.entity_id
                                    entity2_id = entity2.entity_id
                                    
                                    # 标准化实体对（按字母顺序排序，使关系无向化）
                                    normalized_pair = LLMClient._normalize_entity_pair(entity1_id, entity2_id)
                                    normalized_entity1_id = normalized_pair[0]
                                    normalized_entity2_id = normalized_pair[1]
                                    
                                    edge_key = (normalized_entity1_id, normalized_entity2_id, relation.relation_id)
                                    if edge_key not in edges_seen:
                                        # 判断哪个是"另一端"的实体（相对于当前entity）
                                        other_entity = entity2 if entity1_id == entity.entity_id else entity1
                                        other_entity_id = other_entity.entity_id
                                        
                                        # 统计另一端实体拥有的关系边数量（去重后）
                                        other_entity_edge_count = count_entity_relations(other_entity_id)
                                        
                                        relation_candidates.append({
                                            'relation': relation,
                                            'entity1': entity1,
                                            'entity2': entity2,
                                            'normalized_entity1_id': normalized_entity1_id,
                                            'normalized_entity2_id': normalized_entity2_id,
                                            'edge_key': edge_key,
                                            'other_entity_edge_count': other_entity_edge_count
                                        })
                        
                        # 按照另一端实体的边数从多到少排序
                        relation_candidates.sort(key=lambda x: x['other_entity_edge_count'], reverse=True)
                        
                        # 应用 limit_edges_per_entity 限制
                        if limit_edges_per_entity:
                            relation_candidates = relation_candidates[:limit_edges_per_entity]
                        
                        # 按排序后的顺序添加关系边
                        for candidate in relation_candidates:
                            relation = candidate['relation']
                            entity1 = candidate['entity1']
                            entity2 = candidate['entity2']
                            normalized_entity1_id = candidate['normalized_entity1_id']
                            normalized_entity2_id = candidate['normalized_entity2_id']
                            edge_key = candidate['edge_key']
                            
                            edges_seen.add(edge_key)
                            
                            if entity1.id not in entity_absolute_ids:
                                entity_absolute_ids.add(entity1.id)
                                all_related_entity_ids.add(normalized_entity1_id)
                                entity_id_to_name[normalized_entity1_id] = entity1.name
                                entity_id_to_absolute_id[normalized_entity1_id] = entity1.id
                            
                            if entity2.id not in entity_absolute_ids:
                                entity_absolute_ids.add(entity2.id)
                                all_related_entity_ids.add(normalized_entity2_id)
                                entity_id_to_name[normalized_entity2_id] = entity2.name
                                entity_id_to_absolute_id[normalized_entity2_id] = entity2.id
                            
                            # 添加边
                            if entity1.id in entity_absolute_ids or entity2.id in entity_absolute_ids:
                                edge_label = ""
                                if relation.content:
                                    edge_label = relation.content[:30] + "..." if len(relation.content) > 30 else relation.content
                                
                                edges.append({
                                    'from': normalized_entity1_id,
                                    'to': normalized_entity2_id,
                                    'label': edge_label,
                                    'title': relation.content,
                                    'content': relation.content,
                                    'physical_time': relation.physical_time.isoformat(),
                                    'relation_id': relation.relation_id,
                                    'absolute_id': relation.id,
                                    'color': '#888888',
                                    'width': 2,
                                    'arrows': ''
                                })
                
                # 添加关联实体节点（如果还没有添加）
                for entity_id in all_related_entity_ids:
                    if entity_id not in [node['id'] for node in nodes]:
                        # 在 focus 模式下，直接使用记录的 absolute_id 获取实体版本
                        # 这确保我们显示的是关系边直接引用的实体版本
                        absolute_id = entity_id_to_absolute_id.get(entity_id)
                        if absolute_id:
                            related_entity = self.storage.get_entity_by_absolute_id(absolute_id)
                        else:
                            # 回退：如果没有记录 absolute_id，使用时间点
                            effective_time_point = focus_time_point if focus_entity_id else time_point
                            if effective_time_point:
                                related_entity = self.storage.get_entity_version_at_time(entity_id, effective_time_point)
                            else:
                                related_entity = None
                        
                        if related_entity:
                            versions = self.storage.get_entity_versions(related_entity.entity_id)
                            version_count = len(versions)
                            
                            # 在focus模式下，显示该实体版本的索引
                            if focus_entity_id and absolute_id:
                                versions_sorted = sorted(versions, key=lambda v: v.physical_time)
                                current_version_index = None
                                for idx, v in enumerate(versions_sorted, 1):
                                    if v.id == related_entity.id:
                                        current_version_index = idx
                                        break
                                
                                if current_version_index:
                                    label = f"{related_entity.name} ({current_version_index}/{version_count}版本)" if version_count > 1 else related_entity.name
                                else:
                                    label = f"{related_entity.name} ({version_count}版本)" if version_count > 1 else related_entity.name
                            else:
                                label = f"{related_entity.name} ({version_count}版本)" if version_count > 1 else related_entity.name
                            
                            # 根据跳数层级设置颜色
                            hop_level = entity_id_to_hop_level.get(entity_id, 0)
                            node_color = get_hop_color(hop_level)
                            
                            nodes.append({
                                'id': related_entity.entity_id,
                                'entity_id': related_entity.entity_id,
                                'absolute_id': related_entity.id,
                                'label': label,
                                'title': f"{related_entity.name}\n\n{related_entity.content[:100]}..." if len(related_entity.content) > 100 else f"{related_entity.name}\n\n{related_entity.content}",
                                'content': related_entity.content,
                                'physical_time': related_entity.physical_time.isoformat(),
                                'version_count': version_count,
                                'color': node_color,  # 根据跳数层级设置颜色
                                'shape': 'dot',
                                'size': 20,
                                'font': {'color': 'white'}
                            })
                
                return jsonify({
                    'success': True,
                    'nodes': nodes,
                    'edges': edges,
                    'stats': {
                        'total_entities': len(nodes),
                        'total_relations': len(edges),
                        'initial_entities': len(entities),
                        'related_entities': len(all_related_entity_ids)
                    }
                })
            except Exception as e:
                import traceback
                error_traceback = traceback.format_exc()
                print(f"❌ [API错误] /api/graph/data 发生异常:")
                print(f"   错误类型: {type(e).__name__}")
                print(f"   错误消息: {str(e)}")
                print(f"   错误堆栈:\n{error_traceback}")
                return jsonify({
                    'success': False,
                    'error': str(e),
                    'error_type': type(e).__name__
                }), 500
        
        @self.app.route('/api/graph/config')
        def get_config():
            """获取当前配置信息 API"""
            try:
                return jsonify({
                    'success': True,
                    'storage_path': self._current_storage_path
                })
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @self.app.route('/api/graph/stats')
        def get_stats():
            """获取统计信息 API"""
            try:
                entities = self.storage.get_all_entities()
                relations = self.storage.get_all_relations()
                
                return jsonify({
                    'success': True,
                    'stats': {
                        'total_entities': len(entities),
                        'total_relations': len(relations)
                    }
                })
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @self.app.route('/api/graph/search', methods=['POST'])
        def search_graph():
            """搜索图谱 API
            
            接收JSON数据:
            - query: 自然语言查询文本
            - max_results: 返回的最大实体+关系数量（默认10）
            - storage_path: 图谱存储路径（可选），如果提供且与当前路径不同，会切换存储路径
            
            搜索逻辑:
            1. 将查询转成embedding向量
            2. 与所有实体embedding和关系边embedding进行相似度匹配
            3. 选取前n条实体+关系的集合（同ID去重，选取最新版本）
            4. 以这些实体和关系为起点，只显示1跳距离的数据：
               - 实体：找相关联的边以及连接的另一个实体
               - 关系边：找对应的两个实体
            """
            try:
                from flask import request
                
                data = request.get_json()
                if not data:
                    return jsonify({
                        'success': False,
                        'error': '请求数据格式错误，需要JSON格式'
                    }), 400
                
                query = data.get('query', '').strip()
                if not query:
                    return jsonify({
                        'success': False,
                        'error': '查询文本不能为空'
                    }), 400
                
                max_results = data.get('max_results', 10)
                storage_path_param = data.get('storage_path', '').strip() if data.get('storage_path') else None
                
                # 如果提供了存储路径参数，且与当前路径不同，则切换存储路径
                if storage_path_param:
                    try:
                        self._switch_storage_path(storage_path_param)
                    except Exception as e:
                        return jsonify({
                            'success': False,
                            'error': f'切换存储路径失败: {str(e)}'
                        }), 400
                
                # 使用embedding进行语义搜索
                # 设置较低的阈值以支持更灵活的匹配
                # 如果embedding不可用，会自动回退到文本相似度搜索
                try:
                    print(f"[搜索API] 开始搜索，查询: {query}, 最大结果数: {max_results}")
                    print(f"[搜索API] Embedding客户端可用: {self.embedding_client.is_available() if self.embedding_client else False}")
                    
                    # 同时搜索实体和关系
                    matched_entities = self.storage.search_entities_by_similarity(
                        query_name=query,
                        query_content=query,
                        threshold=0.3,
                        max_results=max_results,  # 先搜索max_results个实体
                        content_snippet_length=100
                    )
                    
                    matched_relations = self.storage.search_relations_by_similarity(
                        query_text=query,
                        threshold=0.3,
                        max_results=max_results  # 先搜索max_results个关系
                    )
                    
                    print(f"[搜索API] 搜索完成，找到 {len(matched_entities)} 个匹配实体，{len(matched_relations)} 个匹配关系")
                    
                    # 合并实体和关系，去重（同ID只保留最新版本）
                    # 收集匹配的实体ID和关系ID
                    matched_entity_absolute_ids = {entity.id for entity in matched_entities}
                    matched_relation_absolute_ids = {relation.id for relation in matched_relations}
                    
                    # 收集匹配实体的entity_id（用于节点显示）
                    matched_entity_ids = {entity.entity_id for entity in matched_entities}
                    matched_relation_ids = {relation.relation_id for relation in matched_relations}
                    
                except Exception as e:
                    import traceback
                    print(f"[搜索API] 搜索错误: {str(e)}")
                    print(traceback.format_exc())
                    return jsonify({
                        'success': False,
                        'error': f'搜索过程中发生错误: {str(e)}'
                    }), 500
                
                if not matched_entities and not matched_relations:
                    print("[搜索API] 未找到匹配的实体或关系")
                    return jsonify({
                        'success': True,
                        'nodes': [],
                        'edges': [],
                        'stats': {
                            'total_entities': 0,
                            'total_relations': 0,
                            'matched_entities': 0,
                            'matched_relations': 0
                        },
                        'query': query
                    })
                
                # 收集所有需要显示的实体（1跳距离）
                # 1. 匹配的实体本身
                entity_absolute_ids = set(matched_entity_absolute_ids)
                entity_id_to_name = {}
                entity_id_to_absolute_id = {}
                
                # 2. 匹配关系对应的两个实体
                for relation in matched_relations:
                    entity1 = self.storage.get_entity_by_absolute_id(relation.entity1_absolute_id)
                    entity2 = self.storage.get_entity_by_absolute_id(relation.entity2_absolute_id)
                    if entity1:
                        entity_absolute_ids.add(entity1.id)
                        entity_id_to_name[entity1.entity_id] = entity1.name
                        entity_id_to_absolute_id[entity1.entity_id] = entity1.id
                    if entity2:
                        entity_absolute_ids.add(entity2.id)
                        entity_id_to_name[entity2.entity_id] = entity2.name
                        entity_id_to_absolute_id[entity2.entity_id] = entity2.id
                
                # 3. 匹配实体相关联的边以及连接的另一个实体（1跳距离）
                relation_absolute_ids = set(matched_relation_absolute_ids)
                edges_seen = set()  # 用于去重，使用 (from_id, to_id, relation_id) 作为唯一标识
                
                for entity in matched_entities:
                    # 获取该实体的所有关系边（1跳距离，不限制数量）
                    entity_relations = self.storage.get_entity_relations(entity.id, limit=None)
                    
                    for relation in entity_relations:
                        relation_absolute_ids.add(relation.id)
                        
                        # 通过绝对ID获取实体
                        entity1 = self.storage.get_entity_by_absolute_id(relation.entity1_absolute_id)
                        entity2 = self.storage.get_entity_by_absolute_id(relation.entity2_absolute_id)
                        
                        if entity1 and entity2:
                            entity1_id = entity1.entity_id
                            entity2_id = entity2.entity_id
                            
                            # 标准化实体对（按字母顺序排序，使关系无向化）
                            normalized_pair = LLMClient._normalize_entity_pair(entity1_id, entity2_id)
                            normalized_entity1_id = normalized_pair[0]
                            normalized_entity2_id = normalized_pair[1]
                            
                            # 创建唯一标识符，避免重复添加同一条边（使用标准化后的实体对）
                            edge_key = (normalized_entity1_id, normalized_entity2_id, relation.relation_id)
                            if edge_key in edges_seen:
                                continue
                            edges_seen.add(edge_key)
                            
                            # 添加连接的实体（1跳距离）
                            if entity1.id not in entity_absolute_ids:
                                entity_absolute_ids.add(entity1.id)
                                entity_id_to_name[normalized_entity1_id] = entity1.name
                                entity_id_to_absolute_id[normalized_entity1_id] = entity1.id
                            
                            if entity2.id not in entity_absolute_ids:
                                entity_absolute_ids.add(entity2.id)
                                entity_id_to_name[normalized_entity2_id] = entity2.name
                                entity_id_to_absolute_id[normalized_entity2_id] = entity2.id
                
                # 构建节点数据（使用entity_id去重，每个entity_id只保留一个节点）
                nodes = []
                seen_entity_ids = set()  # 用于去重，确保每个entity_id只添加一次
                entity_id_to_latest_absolute_id = {}  # 记录每个entity_id对应的最新absolute_id
                
                # 首先收集所有entity_id及其对应的最新absolute_id
                for entity_abs_id in entity_absolute_ids:
                    entity = self.storage.get_entity_by_absolute_id(entity_abs_id)
                    if entity:
                        entity_id = entity.entity_id
                        # 如果这个entity_id还没有记录，或者当前版本更新，则更新记录
                        if entity_id not in entity_id_to_latest_absolute_id:
                            entity_id_to_latest_absolute_id[entity_id] = entity_abs_id
                        else:
                            # 比较时间，保留更新的版本
                            existing_entity = self.storage.get_entity_by_absolute_id(entity_id_to_latest_absolute_id[entity_id])
                            if existing_entity and entity.physical_time > existing_entity.physical_time:
                                entity_id_to_latest_absolute_id[entity_id] = entity_abs_id
                
                # 然后为每个唯一的entity_id创建一个节点
                for entity_id, entity_abs_id in entity_id_to_latest_absolute_id.items():
                    entity = self.storage.get_entity_by_absolute_id(entity_abs_id)
                    if entity:
                        # 判断是否为匹配的实体
                        is_matched = entity.entity_id in matched_entity_ids
                        
                        # 获取版本数量
                        versions = self.storage.get_entity_versions(entity.entity_id)
                        version_count = len(versions)
                        
                        # 在标签中显示版本数量
                        label = f"{entity.name} ({version_count}版本)" if version_count > 1 else entity.name
                        
                        nodes.append({
                            'id': entity.entity_id,
                            'entity_id': entity.entity_id,
                            'absolute_id': entity.id,
                            'label': label,
                            'title': f"{entity.name}\n\n{entity.content[:100]}..." if len(entity.content) > 100 else f"{entity.name}\n\n{entity.content}",
                            'content': entity.content,
                            'physical_time': entity.physical_time.isoformat(),
                            'version_count': version_count,
                            'color': '#FF6B6B' if is_matched else '#97C2FC',  # 匹配的实体用红色，其他用蓝色
                            'shape': 'dot',
                            'size': 25 if is_matched else 20,
                            'font': {'color': 'white'}  # 所有搜索结果中的节点字体都用白色
                        })
                
                # 构建边数据
                edges = []
                edges_seen = set()  # 重新初始化，用于最终去重
                
                # 1. 添加匹配关系对应的边
                for relation in matched_relations:
                    entity1 = self.storage.get_entity_by_absolute_id(relation.entity1_absolute_id)
                    entity2 = self.storage.get_entity_by_absolute_id(relation.entity2_absolute_id)
                    
                    if entity1 and entity2:
                        entity1_id = entity1.entity_id
                        entity2_id = entity2.entity_id
                        
                        edge_key = (entity1_id, entity2_id, relation.relation_id)
                        if edge_key not in edges_seen:
                            edges_seen.add(edge_key)
                            
                            edge_label = ""
                            if relation.content:
                                edge_label = relation.content[:30] + "..." if len(relation.content) > 30 else relation.content
                            
                            edges.append({
                                'from': entity1_id,
                                'to': entity2_id,
                                'label': edge_label,
                                'title': relation.content,
                                'content': relation.content,
                                'physical_time': relation.physical_time.isoformat(),
                                'relation_id': relation.relation_id,
                                'absolute_id': relation.id,
                                'color': '#FF6B6B',  # 匹配的关系边用红色，和匹配实体颜色一致
                                'width': 3,
                                'arrows': ''
                            })
                
                # 2. 添加匹配实体相关联的边（1跳距离）
                for entity in matched_entities:
                    entity_relations = self.storage.get_entity_relations(entity.id, limit=None)
                    
                    for relation in entity_relations:
                        entity1 = self.storage.get_entity_by_absolute_id(relation.entity1_absolute_id)
                        entity2 = self.storage.get_entity_by_absolute_id(relation.entity2_absolute_id)
                        
                        if entity1 and entity2:
                            entity1_id = entity1.entity_id
                            entity2_id = entity2.entity_id
                            
                            edge_key = (entity1_id, entity2_id, relation.relation_id)
                            if edge_key not in edges_seen:
                                edges_seen.add(edge_key)
                                
                                # 判断是否为匹配的关系
                                is_matched = relation.relation_id in matched_relation_ids
                                
                                edge_label = ""
                                if relation.content:
                                    edge_label = relation.content[:30] + "..." if len(relation.content) > 30 else relation.content
                                
                                edges.append({
                                    'from': entity1_id,
                                    'to': entity2_id,
                                    'label': edge_label,
                                    'title': relation.content,
                                    'content': relation.content,
                                    'physical_time': relation.physical_time.isoformat(),
                                    'relation_id': relation.relation_id,
                                    'absolute_id': relation.id,
                                    'color': '#FF6B6B' if is_matched else '#97C2FC',  # 匹配的关系边用红色（和匹配实体颜色一致），其他用蓝色
                                    'width': 3 if is_matched else 2,
                                    'arrows': ''
                                })
                
                return jsonify({
                    'success': True,
                    'nodes': nodes,
                    'edges': edges,
                    'stats': {
                        'total_entities': len(nodes),
                        'total_relations': len(edges),
                        'matched_entities': len(matched_entities),
                        'matched_relations': len(matched_relations)
                    },
                    'query': query
                })
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @self.app.route('/api/entity/<entity_id>/versions')
        def get_entity_versions(entity_id):
            """获取实体的所有版本列表"""
            try:
                versions = self.storage.get_entity_versions(entity_id)
                
                if not versions:
                    return jsonify({
                        'success': False,
                        'error': f'未找到实体 {entity_id} 的版本'
                    }), 404
                
                versions_data = []
                for i, entity in enumerate(versions, 1):
                    versions_data.append({
                        'index': i,
                        'total': len(versions),
                        'absolute_id': entity.id,
                        'entity_id': entity.entity_id,
                        'name': entity.name,
                        'content': entity.content,
                        'physical_time': entity.physical_time.isoformat(),
                        'memory_cache_id': entity.memory_cache_id
                    })
                
                return jsonify({
                    'success': True,
                    'entity_id': entity_id,
                    'versions': versions_data
                })
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @self.app.route('/api/entity/<entity_id>/version/<absolute_id>')
        def get_entity_version(entity_id, absolute_id):
            """获取实体的特定版本"""
            try:
                entity = self.storage.get_entity_by_absolute_id(absolute_id)
                
                if not entity:
                    return jsonify({
                        'success': False,
                        'error': f'未找到实体版本 {absolute_id}'
                    }), 404
                
                if entity.entity_id != entity_id:
                    return jsonify({
                        'success': False,
                        'error': f'实体ID不匹配'
                    }), 400
                
                # 获取版本索引
                versions = self.storage.get_entity_versions(entity_id)
                version_index = next((i for i, e in enumerate(versions, 1) if e.id == absolute_id), None)
                
                # 获取embedding前4个值
                embedding_preview = self.storage.get_entity_embedding_preview(absolute_id, 4)
                
                # 获取memory_cache对应的md文档内容和json中的原文内容
                memory_cache_content = None  # md文档内容
                memory_cache_text = None  # json中的原文内容
                doc_name = None  # 文档名称
                if entity.memory_cache_id:
                    # 获取md文档内容（MemoryCache的content字段）
                    memory_cache = self.storage.load_memory_cache(entity.memory_cache_id)
                    if memory_cache:
                        memory_cache_content = memory_cache.content
                        doc_name = memory_cache.doc_name  # 从MemoryCache对象获取文档名称
                    # 获取json中的原文内容
                    memory_cache_text = self.storage.get_memory_cache_text(entity.memory_cache_id)
                
                return jsonify({
                    'success': True,
                    'entity': {
                        'absolute_id': entity.id,
                        'entity_id': entity.entity_id,
                        'name': entity.name,
                        'content': entity.content,
                        'physical_time': entity.physical_time.isoformat(),
                        'memory_cache_id': entity.memory_cache_id,
                        'memory_cache_content': memory_cache_content,  # md文档内容
                        'memory_cache_text': memory_cache_text,  # json中的原文内容
                        'doc_name': doc_name,  # 文档名称
                        'version_index': version_index,
                        'total_versions': len(versions),
                        'embedding_preview': embedding_preview  # embedding前4个值
                    }
                })
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @self.app.route('/api/relation/<relation_id>/versions')
        def get_relation_versions(relation_id):
            """获取关系的所有版本列表"""
            try:
                versions = self.storage.get_relation_versions(relation_id)
                
                if not versions:
                    return jsonify({
                        'success': False,
                        'error': f'未找到关系 {relation_id} 的版本'
                    }), 404
                
                versions_data = []
                for i, relation in enumerate(versions, 1):
                    # 获取实体信息
                    entity1 = self.storage.get_entity_by_absolute_id(relation.entity1_absolute_id)
                    entity2 = self.storage.get_entity_by_absolute_id(relation.entity2_absolute_id)
                    
                    versions_data.append({
                        'index': i,
                        'total': len(versions),
                        'absolute_id': relation.id,
                        'relation_id': relation.relation_id,
                        'content': relation.content,
                        'physical_time': relation.physical_time.isoformat(),
                        'memory_cache_id': relation.memory_cache_id,
                        'entity1_absolute_id': relation.entity1_absolute_id,
                        'entity2_absolute_id': relation.entity2_absolute_id,
                        'entity1_id': entity1.entity_id if entity1 else None,
                        'entity2_id': entity2.entity_id if entity2 else None
                    })
                
                return jsonify({
                    'success': True,
                    'relation_id': relation_id,
                    'versions': versions_data
                })
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @self.app.route('/api/graph/snapshot', methods=['POST'])
        def get_graph_snapshot():
            """根据时间点获取图谱快照
            
            接收JSON数据:
            - entity_versions: {entity_id: absolute_id} 字典，指定要显示的实体版本
            - relation_versions: {relation_id: absolute_id} 字典，指定要显示的关系版本
            - time_point: ISO格式的时间点（可选，用于筛选）
            """
            try:
                from flask import request
                from datetime import datetime
                
                data = request.get_json()
                if not data:
                    return jsonify({
                        'success': False,
                        'error': '请求数据格式错误，需要JSON格式'
                    }), 400
                
                entity_versions = data.get('entity_versions', {})  # {entity_id: absolute_id}
                relation_versions = data.get('relation_versions', {})  # {relation_id: absolute_id}
                time_point_str = data.get('time_point')
                
                time_point = None
                if time_point_str:
                    try:
                        time_point = datetime.fromisoformat(time_point_str)
                    except:
                        pass
                
                # 获取指定版本的实体信息
                nodes_data = []
                for entity_id, absolute_id in entity_versions.items():
                    entity = self.storage.get_entity_by_absolute_id(absolute_id)
                    if entity:
                        versions = self.storage.get_entity_versions(entity_id)
                        version_count = len(versions)
                        label = f"{entity.name} ({version_count}版本)" if version_count > 1 else entity.name
                        
                        nodes_data.append({
                            'id': entity_id,
                            'entity_id': entity_id,
                            'absolute_id': absolute_id,
                            'label': label,
                            'name': entity.name,
                            'content': entity.content,
                            'physical_time': entity.physical_time.isoformat(),
                            'version_count': version_count
                        })
                
                # 获取指定版本的关系信息
                edges_data = []
                for relation_id, absolute_id in relation_versions.items():
                    versions = self.storage.get_relation_versions(relation_id)
                    relation = next((r for r in versions if r.id == absolute_id), None)
                    if relation:
                        entity1 = self.storage.get_entity_by_absolute_id(relation.entity1_absolute_id)
                        entity2 = self.storage.get_entity_by_absolute_id(relation.entity2_absolute_id)
                        if entity1 and entity2:
                            edges_data.append({
                                'relation_id': relation_id,
                                'absolute_id': absolute_id,
                                'from': entity1.entity_id,
                                'to': entity2.entity_id,
                                'content': relation.content,
                                'physical_time': relation.physical_time.isoformat()
                            })
                
                return jsonify({
                    'success': True,
                    'nodes': nodes_data,
                    'edges': edges_data
                })
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @self.app.route('/api/relation/<relation_id>/version/<absolute_id>')
        def get_relation_version(relation_id, absolute_id):
            """获取关系的特定版本"""
            try:
                # 获取所有版本来找到特定版本
                versions = self.storage.get_relation_versions(relation_id)
                relation = next((r for r in versions if r.id == absolute_id), None)
                
                if not relation:
                    return jsonify({
                        'success': False,
                        'error': f'未找到关系版本 {absolute_id}'
                    }), 404
                
                if relation.relation_id != relation_id:
                    return jsonify({
                        'success': False,
                        'error': f'关系ID不匹配'
                    }), 400
                
                # 获取实体信息
                entity1 = self.storage.get_entity_by_absolute_id(relation.entity1_absolute_id)
                entity2 = self.storage.get_entity_by_absolute_id(relation.entity2_absolute_id)
                
                # 获取版本索引
                version_index = next((i for i, r in enumerate(versions, 1) if r.id == absolute_id), None)
                
                # 获取embedding前5个值
                embedding_preview = self.storage.get_relation_embedding_preview(absolute_id, 4)
                
                return jsonify({
                    'success': True,
                    'relation': {
                        'absolute_id': relation.id,
                        'relation_id': relation.relation_id,
                        'content': relation.content,
                        'physical_time': relation.physical_time.isoformat(),
                        'memory_cache_id': relation.memory_cache_id,
                        'entity1_absolute_id': relation.entity1_absolute_id,
                        'entity2_absolute_id': relation.entity2_absolute_id,
                        'entity1_id': entity1.entity_id if entity1 else None,
                        'entity2_id': entity2.entity_id if entity2 else None,
                        'entity1_name': entity1.name if entity1 else None,
                        'entity2_name': entity2.name if entity2 else None,
                        'version_index': version_index,
                        'total_versions': len(versions),
                        'embedding_preview': embedding_preview  # 添加embedding前5个值
                    }
                })
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
    
    def run(self, debug: bool = False, host: str = '0.0.0.0'):
        """
        启动 Web 服务器
        
        Args:
            debug: 是否开启调试模式
            host: 监听地址
        """
        # 获取embedding模型信息
        embedding_info = "未配置"
        if self.embedding_client.model:
            if self.embedding_client.model_path:
                embedding_info = f"本地模型: {self.embedding_client.model_path}"
            elif self.embedding_client.model_name:
                embedding_info = f"HuggingFace: {self.embedding_client.model_name}"
            else:
                embedding_info = "默认模型: all-MiniLM-L6-v2"
        else:
            embedding_info = "未安装sentence-transformers（将使用文本相似度搜索）"
        
        print(f"""
╔════════════════════════════════════════╗
║   时序记忆图谱可视化 Web 服务            ║
╚════════════════════════════════════════╝

🌐 Web服务器已启动
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
访问地址: http://localhost:{self.port}
API地址:  http://localhost:{self.port}/api/graph/data

📦 Embedding模型: {embedding_info}
📁 存储路径: {self.storage_path}

提示:
  1. 在浏览器中打开 http://localhost:{self.port}
  2. 图谱会自动加载并显示
  3. 点击"刷新图谱"按钮手动更新
  4. 点击节点或边查看详细信息
  5. 使用搜索功能进行语义搜索

按 Ctrl+C 停止服务器
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        """)
        
        try:
            self.app.run(host=host, port=self.port, debug=debug)
        except KeyboardInterrupt:
            print("\n\n👋 服务器已停止")
        except Exception as e:
            print(f"\n❌ 错误: {e}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='时序记忆图谱可视化 Web 服务')
    parser.add_argument('--storage', type=str, default='./graph/santi',
                       help='存储路径 (默认: ./graph/santi)')
    parser.add_argument('--port', type=int, default=5000,
                       help='服务器端口 (默认: 5000)')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                       help='监听地址 (默认: 0.0.0.0)')
    parser.add_argument('--debug', action='store_true',
                       help='开启调试模式')
    parser.add_argument('--embedding-model-path', type=str, default="/home/linkco/exa/models/Qwen3-Embedding-0.6B",
                       help='本地embedding模型路径（优先使用）')
    parser.add_argument('--embedding-model-name', type=str, default=None,
                       help='HuggingFace embedding模型名称（例如: all-MiniLM-L6-v2）')
    parser.add_argument('--embedding-device', type=str, default='cuda:1',
                       choices=['cpu', 'cuda'],
                       help='计算设备 (默认: cpu)')
    parser.add_argument('--embedding-use-local', action='store_true', default=True,
                       help='优先使用本地模型（默认: True）')
    parser.add_argument('--embedding-use-hf', action='store_true', default=False,
                       help='优先使用HuggingFace模型（与--embedding-use-local互斥）')
    
    args = parser.parse_args()
    
    # 检查存储路径
    if not Path(args.storage).exists():
        print(f"错误：存储路径不存在: {args.storage}")
        return 1
    
    # 处理embedding模型参数
    embedding_use_local = args.embedding_use_local and not args.embedding_use_hf
    
    # 创建并启动服务器
    server = GraphWebServer(
        storage_path=args.storage,
        port=args.port,
        embedding_model_path=args.embedding_model_path,
        embedding_model_name=args.embedding_model_name,
        embedding_device=args.embedding_device,
        embedding_use_local=embedding_use_local
    )
    server.run(debug=args.debug, host=args.host)
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())

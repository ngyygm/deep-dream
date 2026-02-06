"""
图谱可视化模块：支持交互式和静态可视化
"""
from typing import List, Optional
from pathlib import Path
import json

from processor import StorageManager, Entity, Relation


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

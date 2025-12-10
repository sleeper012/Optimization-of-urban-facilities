import geopandas as gpd
import osmnx as ox
import networkx as nx
import numpy as np
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

print("="*60)
print("15分钟生活圈可达性分析")
print("="*60)

# 配置参数
WALK_SPEED = 4.5  # 步行速度 km/h
TIME_LIMIT = 15   # 时间限制 分钟
DISTANCE_LIMIT = (WALK_SPEED * 1000 / 60) * TIME_LIMIT  # 转换为米

print(f"\n配置: 步行速度={WALK_SPEED}km/h, 时间限制={TIME_LIMIT}分钟")
print(f"等效距离: {DISTANCE_LIMIT:.0f}米")

# 1. 加载数据
print("\n正在加载数据...")
facilities = gpd.read_file("munich_public_facilities.geojson")
graph = ox.load_graphml("munich_street_network.graphml")
print(f"设施数量: {len(facilities)}")
print(f"路网节点: {len(graph.nodes)}")

# 2. 提取超市作为示例(最重要的日常设施)
supermarkets = facilities[facilities['amenity'].isin(['community_centre', 'townhall'])].copy()
print(f"\n社区综合服务站数量: {len(supermarkets)}")

# 3. 将超市点投影到最近的路网节点
print("\n正在匹配社区综合服务站到路网节点...")
supermarkets = supermarkets.to_crs(epsg=4326)  # 确保是WGS84坐标系

# 获取最近节点
nearest_nodes = []
for idx, row in supermarkets.iterrows():
    try:
        point = row.geometry.centroid if row.geometry.geom_type == 'Polygon' else row.geometry
        nearest_node = ox.distance.nearest_nodes(graph, point.x, point.y)
        nearest_nodes.append(nearest_node)
    except:
        nearest_nodes.append(None)

supermarkets['nearest_node'] = nearest_nodes
supermarkets = supermarkets[supermarkets['nearest_node'].notna()]
print(f"成功匹配 {len(supermarkets)} 个社区综合服务站到路网")

# 4. 计算等时圈(选取前3个超市作为示例)
print("\n正在计算15分钟步行可达范围...")
sample_size = min(3, len(supermarkets))

fig, axes = plt.subplots(1, sample_size, figsize=(18, 6))
if sample_size == 1:
    axes = [axes]

for i in range(sample_size):
    center_node = supermarkets.iloc[i]['nearest_node']
    
    # 计算从该超市可达的所有节点
    subgraph = nx.ego_graph(graph, center_node, radius=DISTANCE_LIMIT, distance='length')
    
    # 可视化
    ax = axes[i]
    ox.plot_graph(subgraph, ax=ax, node_size=0, edge_color='lightblue', 
                  edge_linewidth=0.5, show=False, close=False)
    
    # 标注中心点
    center_y = graph.nodes[center_node]['y']
    center_x = graph.nodes[center_node]['x']
    ax.scatter(center_x, center_y, c='red', s=200, marker='*', 
               zorder=5, edgecolors='black', linewidth=2)
    
    ax.set_title(f'社区综合服务站 #{i+1} 的15分钟步行圈\n(覆盖{len(subgraph.nodes)}个路口)', 
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('经度')
    ax.set_ylabel('纬度')

plt.tight_layout()
plt.savefig('walkability_15min_社区综合服务站.png', dpi=600, bbox_inches='tight')
print("\n可视化完成! 保存为: walkability_15min_社区综合服务站.png")
plt.show()

# 5. 覆盖率统计
print("\n" + "="*60)
print("可达性统计:")
print("-"*60)
total_nodes = len(graph.nodes)
covered_nodes = set()

for idx, row in supermarkets.iterrows():
    center_node = row['nearest_node']
    subgraph = nx.ego_graph(graph, center_node, radius=DISTANCE_LIMIT, distance='length')
    covered_nodes.update(subgraph.nodes)

coverage_rate = len(covered_nodes) / total_nodes * 100
print(f"总路网节点数: {total_nodes}")
print(f"社区综合服务站覆盖节点数: {len(covered_nodes)}")
print(f"覆盖率: {coverage_rate:.2f}%")

print("\n" + "="*60)
print("下一步建议:")
print("1. 识别服务空白区域(未覆盖的节点)")
print("2. 运行设施选址优化模型,找出最佳新增位置")
print("3. 对其他设施类型(幼儿园、医疗等)重复此分析")


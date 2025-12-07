"""
设施选址优化 - 最大覆盖选址模型 (MCLP: Maximal Covering Location Problem)
目标: 在有限预算下,选择P个新设施位置,最大化被覆盖的需求点数量

适用场景: 老旧小区改造、15分钟生活圈建设
优势: 
  1. 不强制100%覆盖(更现实)
  2. 关注"边际改善最大化"
  3. 可引入人口权重
  4. 符合有限预算约束
"""
import geopandas as gpd
import osmnx as ox
import networkx as nx
import numpy as np
from pulp import *
import matplotlib.pyplot as plt
import matplotlib
from sklearn.cluster import KMeans
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

print("="*70)
print("最大覆盖选址优化模型 (MCLP)")
print("="*70)

# ========== 配置参数 ==========
WALK_SPEED = 4.5  # km/h
TIME_LIMIT = 15   # 分钟
DISTANCE_LIMIT = (WALK_SPEED * 1000 / 60) * TIME_LIMIT
NUM_NEW_FACILITIES = 5  # 新增设施数量(预算约束)

# 采样参数
NUM_DEMAND_POINTS = 1000  # 需求点采样数(可根据计算能力调整)
NUM_CANDIDATE_POINTS = 200  # 候选点数量(通过聚类生成)

print(f"\n配置:")
print(f"  新增设施数: {NUM_NEW_FACILITIES}")
print(f"  步行距离上限: {DISTANCE_LIMIT:.0f}米 ({TIME_LIMIT}分钟)")
print(f"  需求点采样: {NUM_DEMAND_POINTS}")
print(f"  候选位置数: {NUM_CANDIDATE_POINTS}")

# ========== 1. 加载数据 ==========
print("\n" + "="*70)
print("步骤1: 加载数据")
print("="*70)
facilities = gpd.read_file("munich_public_facilities.geojson")
graph = ox.load_graphml("munich_street_network.graphml")
print(f"✓ 设施数: {len(facilities)}")
print(f"✓ 路网节点数: {len(graph.nodes)}")

# ========== 2. 提取现有超市 ==========
print("\n" + "="*70)
print("步骤2: 提取现有设施")
print("="*70)
supermarkets = facilities[facilities['shop'] == 'supermarket'].copy()
supermarkets = supermarkets.to_crs(epsg=4326)
print(f"✓ 现有超市: {len(supermarkets)} 个")

# 匹配到路网
existing_facilities = []
for idx, row in supermarkets.iterrows():
    try:
        point = row.geometry.centroid if row.geometry.geom_type == 'Polygon' else row.geometry
        nearest_node = ox.distance.nearest_nodes(graph, point.x, point.y)
        existing_facilities.append(nearest_node)
    except:
        pass
print(f"✓ 成功匹配到路网: {len(existing_facilities)} 个")

# ========== 3. 智能采样需求点 ==========
print("\n" + "="*70)
print("步骤3: 生成需求点")
print("="*70)

all_nodes = list(graph.nodes())
np.random.seed(42)

# 优先在现有设施覆盖薄弱区域采样
print("正在识别服务薄弱区域...")
covered_nodes = set()
for facility_node in existing_facilities:
    try:
        lengths = nx.single_source_dijkstra_path_length(
            graph, facility_node, cutoff=DISTANCE_LIMIT, weight='length'
        )
        covered_nodes.update(lengths.keys())
    except:
        pass

uncovered_nodes = [n for n in all_nodes if n not in covered_nodes]
print(f"✓ 现有覆盖节点: {len(covered_nodes)} ({len(covered_nodes)/len(all_nodes)*100:.1f}%)")
print(f"✓ 未覆盖节点: {len(uncovered_nodes)} ({len(uncovered_nodes)/len(all_nodes)*100:.1f}%)")

# 混合采样: 70%来自未覆盖区域, 30%来自全域(考虑重叠覆盖的价值)
num_from_uncovered = int(NUM_DEMAND_POINTS * 0.7)
num_from_all = NUM_DEMAND_POINTS - num_from_uncovered

if len(uncovered_nodes) > 0:
    sampled_uncovered = np.random.choice(
        uncovered_nodes, 
        size=min(num_from_uncovered, len(uncovered_nodes)), 
        replace=False
    )
else:
    sampled_uncovered = []

sampled_all = np.random.choice(all_nodes, size=num_from_all, replace=False)
demand_nodes = list(set(list(sampled_uncovered) + list(sampled_all)))
print(f"✓ 需求点总数: {len(demand_nodes)}")

# ========== 4. 生成候选位置(使用K-Means聚类) ==========
print("\n" + "="*70)
print("步骤4: 生成候选设施位置")
print("="*70)

# 只在未覆盖区域聚类生成候选点
if len(uncovered_nodes) > NUM_CANDIDATE_POINTS:
    uncovered_coords = np.array([
        [graph.nodes[n]['x'], graph.nodes[n]['y']] 
        for n in uncovered_nodes
    ])
    
    print(f"正在对 {len(uncovered_nodes)} 个未覆盖节点进行聚类...")
    kmeans = KMeans(n_clusters=NUM_CANDIDATE_POINTS, random_state=42, n_init=10)
    kmeans.fit(uncovered_coords)
    
    # 找到最接近聚类中心的路网节点作为候选点
    candidate_nodes = []
    for center in kmeans.cluster_centers_:
        nearest = ox.distance.nearest_nodes(graph, center[0], center[1])
        if nearest not in existing_facilities:  # 避免与现有设施重复
            candidate_nodes.append(nearest)
    
    print(f"✓ 候选位置: {len(candidate_nodes)} 个 (聚类生成)")
else:
    candidate_nodes = [n for n in uncovered_nodes if n not in existing_facilities]
    print(f"✓ 候选位置: {len(candidate_nodes)} 个 (直接使用未覆盖节点)")

# ========== 5. 计算距离矩阵 ==========
print("\n" + "="*70)
print("步骤5: 计算距离矩阵")
print("="*70)

all_facility_nodes = existing_facilities + candidate_nodes
print(f"距离矩阵规模: {len(demand_nodes)} × {len(all_facility_nodes)}")

distance_matrix = np.full((len(demand_nodes), len(all_facility_nodes)), np.inf)

for i, demand in enumerate(demand_nodes):
    if i % 100 == 0:
        print(f"  进度: {i}/{len(demand_nodes)}")
    
    try:
        lengths = nx.single_source_dijkstra_path_length(
            graph, demand, cutoff=DISTANCE_LIMIT*1.5, weight='length'
        )
        for j, facility in enumerate(all_facility_nodes):
            if facility in lengths:
                distance_matrix[i, j] = lengths[facility]
    except:
        pass

print("✓ 距离矩阵计算完成")

# 统计覆盖情况
before_coverage = sum(
    1 for i in range(len(demand_nodes))
    if any(distance_matrix[i][j] <= DISTANCE_LIMIT 
           for j in range(len(existing_facilities)))
)
print(f"\n现状分析:")
print(f"  被现有设施覆盖: {before_coverage}/{len(demand_nodes)} ({before_coverage/len(demand_nodes)*100:.1f}%)")

# ========== 6. 构建MCLP优化模型 ==========
print("\n" + "="*70)
print("步骤6: 构建最大覆盖模型")
print("="*70)

model = LpProblem("Maximum_Covering_Location_Problem", LpMaximize)

# 决策变量
# y[j] = 1 表示在候选点j建设施
y = LpVariable.dicts("build", range(len(candidate_nodes)), cat='Binary')

# z[i] = 1 表示需求点i被覆盖(现有设施或新建设施)
z = LpVariable.dicts("covered", range(len(demand_nodes)), cat='Binary')

# 目标函数: 最大化被覆盖的需求点数量
model += lpSum([z[i] for i in range(len(demand_nodes))])

# 约束1: 需求点i只有在某个设施能覆盖它时才能标记为"被覆盖"
for i in range(len(demand_nodes)):
    # 找出所有能覆盖需求点i的设施(现有+候选)
    covering_facilities = []
    
    # 现有设施
    for j in range(len(existing_facilities)):
        if distance_matrix[i][j] <= DISTANCE_LIMIT:
            covering_facilities.append(('existing', j))
    
    # 候选设施
    for j in range(len(candidate_nodes)):
        facility_idx = j + len(existing_facilities)
        if distance_matrix[i][facility_idx] <= DISTANCE_LIMIT:
            covering_facilities.append(('candidate', j))
    
    if covering_facilities:
        # 需求点i被覆盖 <= (现有设施覆盖 + 新建设施覆盖)
        model += z[i] <= lpSum([
            1 if ftype == 'existing' else y[fid]
            for ftype, fid in covering_facilities
        ])

# 约束2: 新建设施数量限制(预算约束)
model += lpSum([y[j] for j in range(len(candidate_nodes))]) == NUM_NEW_FACILITIES

print("✓ 模型构建完成")
print(f"  决策变量: {len(candidate_nodes)} 个候选位置")
print(f"  约束条件: 新建设施数 = {NUM_NEW_FACILITIES}")

# ========== 7. 求解 ==========
print("\n" + "="*70)
print("步骤7: 求解优化问题")
print("="*70)
print("正在求解(可能需要1-3分钟)...")

model.solve(PULP_CBC_CMD(msg=1, timeLimit=300))

# ========== 8. 结果分析 ==========
print("\n" + "="*70)
print("优化结果")
print("="*70)

if model.status == 1:  # Optimal
    # 提取选中位置
    selected_indices = [j for j in range(len(candidate_nodes)) if value(y[j]) == 1]
    selected_nodes = [candidate_nodes[j] for j in selected_indices]
    
    # 计算覆盖改善
    after_coverage = sum(1 for i in range(len(demand_nodes)) if value(z[i]) == 1)
    improvement = after_coverage - before_coverage
    
    print(f"\n✓ 求解成功!")
    print(f"\n覆盖效果:")
    print(f"  优化前: {before_coverage}/{len(demand_nodes)} 个需求点被覆盖 ({before_coverage/len(demand_nodes)*100:.1f}%)")
    print(f"  优化后: {after_coverage}/{len(demand_nodes)} 个需求点被覆盖 ({after_coverage/len(demand_nodes)*100:.1f}%)")
    print(f"  改善量: +{improvement} 个需求点 (+{improvement/len(demand_nodes)*100:.1f}%)")
    
    print(f"\n建议新增设施位置:")
    for idx, node in enumerate(selected_nodes):
        node_data = graph.nodes[node]
        print(f"  位置 {idx+1}: 节点ID={node}, 坐标=({node_data['y']:.6f}, {node_data['x']:.6f})")
    
    # ========== 9. 可视化 ==========
    print("\n" + "="*70)
    print("步骤8: 生成可视化")
    print("="*70)
    
    fig, axes = plt.subplots(1, 2, figsize=(24, 10))
    
    # ===== 左图: 优化前 =====
    ax1 = axes[0]
    ox.plot_graph(graph, ax=ax1, node_size=0, edge_color='#E8E8E8', 
                  edge_linewidth=0.3, show=False, close=False)
    
    # 现有设施覆盖范围(浅蓝色)
    for facility_node in existing_facilities[:50]:  # 限制数量避免过密
        try:
            subgraph = nx.ego_graph(graph, facility_node, radius=DISTANCE_LIMIT, distance='length')
            node_coords = [(graph.nodes[n]['x'], graph.nodes[n]['y']) for n in subgraph.nodes()]
            ax1.scatter([c[0] for c in node_coords], [c[1] for c in node_coords],
                       c='lightblue', s=2, alpha=0.3, zorder=1)
        except:
            pass
    
    # 现有设施
    existing_coords = [(graph.nodes[n]['x'], graph.nodes[n]['y']) for n in existing_facilities]
    ax1.scatter([c[0] for c in existing_coords], [c[1] for c in existing_coords],
               c='blue', s=80, marker='o', label=f'现有超市({len(existing_facilities)}个)', 
               zorder=4, edgecolors='white', linewidth=1)
    
    # 未覆盖需求点
    uncovered_demand_nodes = [
        demand_nodes[i] for i in range(len(demand_nodes))
        if not any(distance_matrix[i][j] <= DISTANCE_LIMIT 
                  for j in range(len(existing_facilities)))
    ]
    if uncovered_demand_nodes:
        uncovered_coords = [(graph.nodes[n]['x'], graph.nodes[n]['y']) 
                           for n in uncovered_demand_nodes[:500]]  # 限制显示数量
        ax1.scatter([c[0] for c in uncovered_coords], [c[1] for c in uncovered_coords],
                   c='red', s=15, alpha=0.5, marker='x', 
                   label=f'服务不足区域({len(uncovered_demand_nodes)}点)', zorder=3)
    
    ax1.set_title(f'优化前: 覆盖率 {before_coverage/len(demand_nodes)*100:.1f}%', 
                 fontsize=16, fontweight='bold', pad=20)
    ax1.set_xlabel('经度', fontsize=12)
    ax1.set_ylabel('纬度', fontsize=12)
    ax1.legend(loc='upper right', fontsize=11, framealpha=0.9)
    
    # ===== 右图: 优化后 =====
    ax2 = axes[1]
    ox.plot_graph(graph, ax=ax2, node_size=0, edge_color='#E8E8E8', 
                  edge_linewidth=0.3, show=False, close=False)
    
    # 现有设施
    ax2.scatter([c[0] for c in existing_coords], [c[1] for c in existing_coords],
               c='blue', s=80, marker='o', label=f'现有超市({len(existing_facilities)}个)', 
               zorder=4, edgecolors='white', linewidth=1)
    
    # 新增设施及其覆盖范围
    new_coords = [(graph.nodes[n]['x'], graph.nodes[n]['y']) for n in selected_nodes]
    
    for facility_node in selected_nodes:
        try:
            subgraph = nx.ego_graph(graph, facility_node, radius=DISTANCE_LIMIT, distance='length')
            node_coords = [(graph.nodes[n]['x'], graph.nodes[n]['y']) for n in subgraph.nodes()]
            ax2.scatter([c[0] for c in node_coords], [c[1] for c in node_coords],
                       c='lightcoral', s=3, alpha=0.4, zorder=1)
        except:
            pass
    
    ax2.scatter([c[0] for c in new_coords], [c[1] for c in new_coords],
               c='#FF4500', s=350, marker='*', 
               label=f'新增设施({NUM_NEW_FACILITIES}个)', 
               zorder=5, edgecolors='darkred', linewidth=2)
    
    # 标注编号
    for idx, coord in enumerate(new_coords):
        ax2.annotate(f'{idx+1}', xy=coord, xytext=(8, 8), 
                    textcoords='offset points', fontsize=11, 
                    fontweight='bold', color='darkred',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    ax2.set_title(f'优化后: 覆盖率 {after_coverage/len(demand_nodes)*100:.1f}% (+{improvement/len(demand_nodes)*100:.1f}%)', 
                 fontsize=16, fontweight='bold', pad=20, color='darkgreen')
    ax2.set_xlabel('经度', fontsize=12)
    ax2.set_ylabel('纬度', fontsize=12)
    ax2.legend(loc='upper right', fontsize=11, framealpha=0.9)
    
    plt.suptitle('最大覆盖选址优化 - 超市设施布局改善方案', 
                fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('facility_optimization_MCLP.png', dpi=300, bbox_inches='tight')
    print("✓ 可视化完成: facility_optimization_MCLP.png")
    plt.show()
    
else:
    print(f"❌ 求解失败: {LpStatus[model.status]}")

print("\n" + "="*70)
print("优化完成!")
print("="*70)


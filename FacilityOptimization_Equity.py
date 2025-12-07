"""
设施选址优化 - 公平性导向的加权覆盖模型
核心思想:
  1. 考虑不同区域的脆弱性(老年人口比例、低收入等)
  2. 最小化"最差服务水平"(Minimax目标)
  3. 确保服务公平分配

适用场景: 老旧小区改造、关注"一老一小"的公平性规划
优势:
  1. 关注弱势群体
  2. 避免"富者越富"
  3. 符合社会公平原则
  4. 可纳入人口、年龄等权重
"""
import geopandas as gpd
import osmnx as ox
import networkx as nx
import numpy as np
from pulp import *
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

print("="*70)
print("公平性导向的设施选址优化")
print("="*70)

# ========== 配置参数 ==========
WALK_SPEED = 4.5
TIME_LIMIT = 15
DISTANCE_LIMIT = (WALK_SPEED * 1000 / 60) * TIME_LIMIT
NUM_NEW_FACILITIES = 5

NUM_DEMAND_POINTS = 800
NUM_CANDIDATE_POINTS = 150

print(f"\n配置:")
print(f"  新增设施数: {NUM_NEW_FACILITIES}")
print(f"  步行距离上限: {DISTANCE_LIMIT:.0f}米")

# ========== 1. 加载数据 ==========
print("\n" + "="*70)
print("步骤1: 加载数据")
print("="*70)

facilities = gpd.read_file("munich_public_facilities.geojson")
graph = ox.load_graphml("munich_street_network.graphml")

supermarkets = facilities[facilities['shop'] == 'supermarket'].copy()
supermarkets = supermarkets.to_crs(epsg=4326)
print(f"✓ 现有超市: {len(supermarkets)} 个")

# ========== 2. 匹配现有设施 ==========
existing_facilities = []
for idx, row in supermarkets.iterrows():
    try:
        point = row.geometry.centroid if row.geometry.geom_type == 'Polygon' else row.geometry
        nearest_node = ox.distance.nearest_nodes(graph, point.x, point.y)
        existing_facilities.append(nearest_node)
    except:
        pass
print(f"✓ 匹配到路网: {len(existing_facilities)} 个")

# ========== 3. 采样需求点并分配权重 ==========
print("\n" + "="*70)
print("步骤3: 生成需求点并分配公平性权重")
print("="*70)

all_nodes = list(graph.nodes())
np.random.seed(42)

# 识别服务薄弱区域
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
print(f"✓ 未覆盖节点: {len(uncovered_nodes)} ({len(uncovered_nodes)/len(all_nodes)*100:.1f}%)")

# 采样(60%未覆盖 + 40%全域)
num_from_uncovered = int(NUM_DEMAND_POINTS * 0.6)
num_from_all = NUM_DEMAND_POINTS - num_from_uncovered

sampled_uncovered = np.random.choice(
    uncovered_nodes, 
    size=min(num_from_uncovered, len(uncovered_nodes)), 
    replace=False
) if len(uncovered_nodes) > 0 else []

sampled_all = np.random.choice(all_nodes, size=num_from_all, replace=False)
demand_nodes = list(set(list(sampled_uncovered) + list(sampled_all)))

# ===== 分配公平性权重 =====
# 模拟权重: 假设未覆盖区域 + 边缘区域有更高的脆弱性
# 实际应用中可以用真实的人口数据(老年人口比例、收入水平等)

demand_weights = {}
for node in demand_nodes:
    # 基础权重
    weight = 1.0
    
    # 如果当前未被覆盖,权重+2
    if node in uncovered_nodes:
        weight += 2.0
    
    # 如果远离市中心(简化为距离最近设施的距离),权重增加
    min_dist_to_facility = np.inf
    for facility_node in existing_facilities:
        try:
            dist = nx.shortest_path_length(graph, node, facility_node, weight='length')
            min_dist_to_facility = min(min_dist_to_facility, dist)
        except:
            pass
    
    # 距离越远,权重越高(模拟"老旧小区"的脆弱性)
    if min_dist_to_facility > DISTANCE_LIMIT * 1.5:
        weight += 1.5
    elif min_dist_to_facility > DISTANCE_LIMIT:
        weight += 1.0
    
    demand_weights[node] = weight

print(f"✓ 需求点数: {len(demand_nodes)}")
print(f"  权重分布:")
print(f"    最小权重: {min(demand_weights.values()):.1f}")
print(f"    最大权重: {max(demand_weights.values()):.1f}")
print(f"    平均权重: {np.mean(list(demand_weights.values())):.1f}")

# ========== 4. 生成候选位置 ==========
print("\n" + "="*70)
print("步骤4: 生成候选位置")
print("="*70)

# 优先在高权重区域附近采样
high_weight_nodes = [n for n in demand_nodes if demand_weights[n] >= 2.5]
print(f"高权重节点数: {len(high_weight_nodes)}")

if len(high_weight_nodes) > NUM_CANDIDATE_POINTS:
    candidate_nodes = list(np.random.choice(
        high_weight_nodes, 
        size=NUM_CANDIDATE_POINTS, 
        replace=False
    ))
else:
    candidate_nodes = high_weight_nodes + list(np.random.choice(
        [n for n in demand_nodes if n not in high_weight_nodes],
        size=NUM_CANDIDATE_POINTS - len(high_weight_nodes),
        replace=False
    ))

# 排除与现有设施重复的点
candidate_nodes = [n for n in candidate_nodes if n not in existing_facilities]
print(f"✓ 候选位置: {len(candidate_nodes)} 个")

# ========== 5. 计算距离矩阵 ==========
print("\n" + "="*70)
print("步骤5: 计算距离矩阵")
print("="*70)

all_facility_nodes = existing_facilities + candidate_nodes
distance_matrix = np.full((len(demand_nodes), len(all_facility_nodes)), np.inf)

for i, demand in enumerate(demand_nodes):
    if i % 100 == 0:
        print(f"  进度: {i}/{len(demand_nodes)}")
    
    try:
        lengths = nx.single_source_dijkstra_path_length(
            graph, demand, cutoff=DISTANCE_LIMIT*2, weight='length'
        )
        for j, facility in enumerate(all_facility_nodes):
            if facility in lengths:
                distance_matrix[i, j] = lengths[facility]
    except:
        pass

print("✓ 距离矩阵计算完成")

# ========== 6. 构建公平性优化模型 ==========
print("\n" + "="*70)
print("步骤6: 构建公平性优化模型")
print("="*70)

# 模型选择: Minimax + 加权覆盖混合目标
# 目标1: 最小化"最差服务距离"(公平性)
# 目标2: 最大化加权覆盖人口(效率)

model = LpProblem("Equity_Oriented_Facility_Location", LpMinimize)

# 决策变量
y = LpVariable.dicts("build", range(len(candidate_nodes)), cat='Binary')

# d[i] = 需求点i到最近设施的距离
d = LpVariable.dicts("distance", range(len(demand_nodes)), lowBound=0)

# d_max = 所有需求点中的最大距离(Minimax目标)
d_max = LpVariable("max_distance", lowBound=0)

# 目标函数: 最小化(最大距离 + 加权平均距离)
# 权衡公平性和效率
alpha = 0.6  # 公平性权重
beta = 0.4   # 效率权重

total_weight = sum(demand_weights.values())

model += alpha * d_max + beta * lpSum([
    demand_weights[demand_nodes[i]] * d[i] / total_weight
    for i in range(len(demand_nodes))
])

# 约束1: d[i] 是需求点i到最近设施的距离
for i in range(len(demand_nodes)):
    for j in range(len(all_facility_nodes)):
        if distance_matrix[i, j] < np.inf:
            if j < len(existing_facilities):
                # 现有设施总是可用
                model += d[i] <= distance_matrix[i, j]
            else:
                # 新建设施只有被选中时才可用
                candidate_idx = j - len(existing_facilities)
                model += d[i] <= distance_matrix[i, j] + (1 - y[candidate_idx]) * 100000

# 约束2: d_max 是所有d[i]中的最大值
for i in range(len(demand_nodes)):
    model += d_max >= d[i]

# 约束3: 新建设施数量限制
model += lpSum([y[j] for j in range(len(candidate_nodes))]) == NUM_NEW_FACILITIES

print("✓ 模型构建完成")
print(f"  目标函数: {alpha*100:.0f}% 公平性 + {beta*100:.0f}% 效率")

# ========== 7. 求解 ==========
print("\n" + "="*70)
print("步骤7: 求解")
print("="*70)

model.solve(PULP_CBC_CMD(msg=1, timeLimit=300))

# ========== 8. 结果分析 ==========
print("\n" + "="*70)
print("优化结果")
print("="*70)

if model.status == 1:
    selected_indices = [j for j in range(len(candidate_nodes)) if value(y[j]) == 1]
    selected_nodes = [candidate_nodes[j] for j in selected_indices]
    
    print(f"✓ 求解成功!")
    
    # 计算优化前后的公平性指标
    distances_before = []
    distances_after = []
    
    for i in range(len(demand_nodes)):
        # 优化前
        dist_before = min(
            distance_matrix[i, j] 
            for j in range(len(existing_facilities))
            if distance_matrix[i, j] < np.inf
        )
        distances_before.append(dist_before)
        
        # 优化后
        dist_after = value(d[i])
        distances_after.append(dist_after)
    
    print(f"\n公平性指标对比:")
    print(f"  最大距离(最差服务水平):")
    print(f"    优化前: {max(distances_before):.0f} 米")
    print(f"    优化后: {max(distances_after):.0f} 米")
    print(f"    改善: {max(distances_before) - max(distances_after):.0f} 米")
    
    print(f"\n  平均距离:")
    print(f"    优化前: {np.mean(distances_before):.0f} 米")
    print(f"    优化后: {np.mean(distances_after):.0f} 米")
    
    print(f"\n  标准差(公平性指标,越小越公平):")
    print(f"    优化前: {np.std(distances_before):.0f}")
    print(f"    优化后: {np.std(distances_after):.0f}")
    
    # 覆盖率
    before_covered = sum(1 for d in distances_before if d <= DISTANCE_LIMIT)
    after_covered = sum(1 for d in distances_after if d <= DISTANCE_LIMIT)
    
    print(f"\n  {TIME_LIMIT}分钟覆盖率:")
    print(f"    优化前: {before_covered}/{len(demand_nodes)} ({before_covered/len(demand_nodes)*100:.1f}%)")
    print(f"    优化后: {after_covered}/{len(demand_nodes)} ({after_covered/len(demand_nodes)*100:.1f}%)")
    
    print(f"\n建议新增设施位置:")
    for idx, node in enumerate(selected_nodes):
        node_data = graph.nodes[node]
        print(f"  位置 {idx+1}: 节点ID={node}, 坐标=({node_data['y']:.6f}, {node_data['x']:.6f})")
    
    # ========== 9. 可视化 ==========
    print("\n" + "="*70)
    print("步骤8: 生成可视化")
    print("="*70)
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    # ===== 子图1: 需求点权重分布 =====
    ax1 = axes[0, 0]
    ox.plot_graph(graph, ax=ax1, node_size=0, edge_color='lightgray',
                  edge_linewidth=0.2, show=False, close=False)
    
    demand_coords = [(graph.nodes[n]['x'], graph.nodes[n]['y']) for n in demand_nodes]
    weights = [demand_weights[n] for n in demand_nodes]
    
    scatter1 = ax1.scatter([c[0] for c in demand_coords], [c[1] for c in demand_coords],
                          c=weights, s=30, cmap='YlOrRd', alpha=0.7, zorder=3)
    plt.colorbar(scatter1, ax=ax1, label='权重(脆弱性)')
    
    ax1.set_title('需求点权重分布(红色=高脆弱性)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('经度')
    ax1.set_ylabel('纬度')
    
    # ===== 子图2: 优化前服务距离分布 =====
    ax2 = axes[0, 1]
    ox.plot_graph(graph, ax=ax2, node_size=0, edge_color='lightgray',
                  edge_linewidth=0.2, show=False, close=False)
    
    scatter2 = ax2.scatter([c[0] for c in demand_coords], [c[1] for c in demand_coords],
                          c=distances_before, s=30, cmap='RdYlGn_r', 
                          vmin=0, vmax=DISTANCE_LIMIT*2, alpha=0.7, zorder=3)
    plt.colorbar(scatter2, ax=ax2, label='距离(米)')
    
    existing_coords = [(graph.nodes[n]['x'], graph.nodes[n]['y']) for n in existing_facilities]
    ax2.scatter([c[0] for c in existing_coords], [c[1] for c in existing_coords],
               c='blue', s=100, marker='o', label='现有超市', zorder=4,
               edgecolors='white', linewidth=1.5)
    
    ax2.set_title(f'优化前: 服务距离分布 (最大={max(distances_before):.0f}m)', 
                 fontsize=14, fontweight='bold')
    ax2.set_xlabel('经度')
    ax2.set_ylabel('纬度')
    ax2.legend()
    
    # ===== 子图3: 优化后服务距离分布 =====
    ax3 = axes[1, 0]
    ox.plot_graph(graph, ax=ax3, node_size=0, edge_color='lightgray',
                  edge_linewidth=0.2, show=False, close=False)
    
    scatter3 = ax3.scatter([c[0] for c in demand_coords], [c[1] for c in demand_coords],
                          c=distances_after, s=30, cmap='RdYlGn_r',
                          vmin=0, vmax=DISTANCE_LIMIT*2, alpha=0.7, zorder=3)
    plt.colorbar(scatter3, ax=ax3, label='距离(米)')
    
    ax3.scatter([c[0] for c in existing_coords], [c[1] for c in existing_coords],
               c='blue', s=100, marker='o', label='现有超市', zorder=4,
               edgecolors='white', linewidth=1.5)
    
    new_coords = [(graph.nodes[n]['x'], graph.nodes[n]['y']) for n in selected_nodes]
    ax3.scatter([c[0] for c in new_coords], [c[1] for c in new_coords],
               c='red', s=350, marker='*', label=f'新增设施({NUM_NEW_FACILITIES}个)',
               zorder=5, edgecolors='darkred', linewidth=2)
    
    for idx, coord in enumerate(new_coords):
        ax3.annotate(f'{idx+1}', xy=coord, xytext=(8, 8),
                    textcoords='offset points', fontsize=11,
                    fontweight='bold', color='darkred',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    ax3.set_title(f'优化后: 服务距离分布 (最大={max(distances_after):.0f}m)', 
                 fontsize=14, fontweight='bold', color='darkgreen')
    ax3.set_xlabel('经度')
    ax3.set_ylabel('纬度')
    ax3.legend()
    
    # ===== 子图4: 距离分布直方图对比 =====
    ax4 = axes[1, 1]
    ax4.hist(distances_before, bins=30, alpha=0.6, label='优化前', color='red', edgecolor='black')
    ax4.hist(distances_after, bins=30, alpha=0.6, label='优化后', color='green', edgecolor='black')
    ax4.axvline(DISTANCE_LIMIT, color='blue', linestyle='--', linewidth=2, label=f'{TIME_LIMIT}分钟标准线')
    
    ax4.set_xlabel('到最近设施的距离 (米)', fontsize=12)
    ax4.set_ylabel('需求点数量', fontsize=12)
    ax4.set_title('服务距离分布对比', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=12)
    ax4.grid(alpha=0.3)
    
    # 添加统计信息
    textstr = f'优化前:\n  最大距离: {max(distances_before):.0f}m\n  平均距离: {np.mean(distances_before):.0f}m\n  标准差: {np.std(distances_before):.0f}\n\n'
    textstr += f'优化后:\n  最大距离: {max(distances_after):.0f}m\n  平均距离: {np.mean(distances_after):.0f}m\n  标准差: {np.std(distances_after):.0f}'
    ax4.text(0.98, 0.97, textstr, transform=ax4.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('公平性导向的设施选址优化 - 关注服务薄弱区域', 
                fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig('facility_optimization_equity.png', dpi=300, bbox_inches='tight')
    print("✓ 可视化完成: facility_optimization_equity.png")
    plt.show()
    
else:
    print(f"❌ 求解失败: {LpStatus[model.status]}")

print("\n" + "="*70)
print("优化完成!")
print("="*70)


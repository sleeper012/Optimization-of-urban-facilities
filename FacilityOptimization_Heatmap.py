"""
设施选址优化 - 基于热力图的需求驱动方法
核心思想: 
  1. 计算"服务缺口热力图"(需求密度 - 供给密度)
  2. 在热力最高的区域优先布局新设施
  3. 迭代优化直到覆盖率达标

适用场景: 老旧小区改造、需要可视化决策依据的规划项目
优势:
  1. 直观易懂,便于向决策者展示
  2. 不需要复杂求解器
  3. 可以灵活调整策略
  4. 计算速度快
"""
import geopandas as gpd
import osmnx as ox
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.spatial import distance_matrix as scipy_distance_matrix
from sklearn.neighbors import KernelDensity
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

print("="*70)
print("基于热力图的需求驱动选址优化")
print("="*70)

# ========== 配置参数 ==========
WALK_SPEED = 4.5
TIME_LIMIT = 15
DISTANCE_LIMIT = (WALK_SPEED * 1000 / 60) * TIME_LIMIT
NUM_NEW_FACILITIES = 5

GRID_SIZE = 100  # 热力图网格数量
BANDWIDTH = 0.005  # 核密度估计带宽(经纬度单位)

print(f"\n配置:")
print(f"  新增设施数: {NUM_NEW_FACILITIES}")
print(f"  步行距离上限: {DISTANCE_LIMIT:.0f}米")
print(f"  热力图分辨率: {GRID_SIZE}×{GRID_SIZE}")

# ========== 1. 加载数据 ==========
print("\n" + "="*70)
print("步骤1: 加载数据")
print("="*70)

facilities = gpd.read_file("munich_public_facilities.geojson")
graph = ox.load_graphml("munich_street_network.graphml")

supermarkets = facilities[facilities['shop'] == 'supermarket'].copy()
supermarkets = supermarkets.to_crs(epsg=4326)
print(f"✓ 现有超市: {len(supermarkets)} 个")

# ========== 2. 匹配现有设施到路网 ==========
print("\n" + "="*70)
print("步骤2: 匹配设施到路网")
print("="*70)

existing_facilities = []
existing_coords = []

for idx, row in supermarkets.iterrows():
    try:
        point = row.geometry.centroid if row.geometry.geom_type == 'Polygon' else row.geometry
        nearest_node = ox.distance.nearest_nodes(graph, point.x, point.y)
        existing_facilities.append(nearest_node)
        existing_coords.append([graph.nodes[nearest_node]['x'], graph.nodes[nearest_node]['y']])
    except:
        pass

existing_coords = np.array(existing_coords)
print(f"✓ 成功匹配: {len(existing_facilities)} 个")

# ========== 3. 计算需求密度热力图 ==========
print("\n" + "="*70)
print("步骤3: 生成需求密度热力图")
print("="*70)

# 使用路网节点作为需求点
all_nodes = list(graph.nodes())
demand_coords = np.array([[graph.nodes[n]['x'], graph.nodes[n]['y']] for n in all_nodes])

print(f"需求点数量(路网节点): {len(demand_coords)}")

# 获取区域边界
min_x, max_x = demand_coords[:, 0].min(), demand_coords[:, 0].max()
min_y, max_y = demand_coords[:, 1].min(), demand_coords[:, 1].max()

# 创建网格
x_grid = np.linspace(min_x, max_x, GRID_SIZE)
y_grid = np.linspace(min_y, max_y, GRID_SIZE)
xx, yy = np.meshgrid(x_grid, y_grid)
grid_points = np.c_[xx.ravel(), yy.ravel()]

print("正在计算需求密度(核密度估计)...")
kde_demand = KernelDensity(bandwidth=BANDWIDTH, kernel='gaussian')
kde_demand.fit(demand_coords)
log_density_demand = kde_demand.score_samples(grid_points)
density_demand = np.exp(log_density_demand).reshape(GRID_SIZE, GRID_SIZE)

print("✓ 需求密度计算完成")

# ========== 4. 计算供给密度热力图 ==========
print("\n" + "="*70)
print("步骤4: 生成供给密度热力图")
print("="*70)

if len(existing_coords) > 0:
    print("正在计算供给密度...")
    kde_supply = KernelDensity(bandwidth=BANDWIDTH, kernel='gaussian')
    kde_supply.fit(existing_coords)
    log_density_supply = kde_supply.score_samples(grid_points)
    density_supply = np.exp(log_density_supply).reshape(GRID_SIZE, GRID_SIZE)
    print("✓ 供给密度计算完成")
else:
    density_supply = np.zeros((GRID_SIZE, GRID_SIZE))

# ========== 5. 计算服务缺口(需求-供给) ==========
print("\n" + "="*70)
print("步骤5: 计算服务缺口热力图")
print("="*70)

# 归一化
density_demand_norm = (density_demand - density_demand.min()) / (density_demand.max() - density_demand.min() + 1e-10)
density_supply_norm = (density_supply - density_supply.min()) / (density_supply.max() - density_supply.min() + 1e-10)

# 服务缺口 = 需求 - 供给(供给越低,缺口越大)
service_gap = density_demand_norm - density_supply_norm * 0.8  # 0.8是权重系数

print("✓ 服务缺口计算完成")
print(f"  最大缺口: {service_gap.max():.3f}")
print(f"  最小缺口: {service_gap.min():.3f}")

# ========== 6. 选择新设施位置(迭代贪心策略) ==========
print("\n" + "="*70)
print("步骤6: 选择最佳新增位置")
print("="*70)

selected_nodes = []
selected_coords = []
current_gap = service_gap.copy()

# 创建候选位置池(从路网节点中选择)
np.random.seed(42)
candidate_pool_size = min(5000, len(all_nodes))
candidate_nodes = np.random.choice(all_nodes, size=candidate_pool_size, replace=False)
candidate_coords = np.array([[graph.nodes[n]['x'], graph.nodes[n]['y']] for n in candidate_nodes])

print(f"候选位置池: {len(candidate_nodes)} 个路网节点")

for i in range(NUM_NEW_FACILITIES):
    print(f"\n  选择第 {i+1} 个设施...")
    
    best_score = -np.inf
    best_node = None
    best_coord = None
    
    # 遍历候选位置,找到能最大化覆盖服务缺口的点
    for j, node in enumerate(candidate_nodes):
        coord = candidate_coords[j]
        
        # 跳过已选中的位置附近
        if len(selected_coords) > 0:
            min_dist = np.min([
                np.sqrt((coord[0] - sc[0])**2 + (coord[1] - sc[1])**2)
                for sc in selected_coords
            ])
            if min_dist < 0.01:  # 约1km(经纬度单位)
                continue
        
        # 计算该位置的影响范围内的服务缺口总和
        # 在网格上找到影响范围
        x_idx = np.argmin(np.abs(x_grid - coord[0]))
        y_idx = np.argmin(np.abs(y_grid - coord[1]))
        
        # 计算影响范围(简化为圆形区域,半径对应15分钟步行)
        influence_radius = 5  # 网格单元数
        y_min = max(0, y_idx - influence_radius)
        y_max = min(GRID_SIZE, y_idx + influence_radius + 1)
        x_min = max(0, x_idx - influence_radius)
        x_max = min(GRID_SIZE, x_idx + influence_radius + 1)
        
        # 计算影响区域内的服务缺口总和
        score = np.sum(current_gap[y_min:y_max, x_min:x_max])
        
        if score > best_score:
            best_score = score
            best_node = node
            best_coord = coord
    
    if best_node is not None:
        selected_nodes.append(best_node)
        selected_coords.append(best_coord)
        
        # 更新服务缺口(减去新设施的贡献)
        x_idx = np.argmin(np.abs(x_grid - best_coord[0]))
        y_idx = np.argmin(np.abs(y_grid - best_coord[1]))
        
        influence_radius = 5
        y_min = max(0, y_idx - influence_radius)
        y_max = min(GRID_SIZE, y_idx + influence_radius + 1)
        x_min = max(0, x_idx - influence_radius)
        x_max = min(GRID_SIZE, x_idx + influence_radius + 1)
        
        # 使用高斯衰减
        for yi in range(y_min, y_max):
            for xi in range(x_min, x_max):
                dist = np.sqrt((yi - y_idx)**2 + (xi - x_idx)**2)
                reduction = 0.5 * np.exp(-dist / influence_radius)
                current_gap[yi, xi] -= reduction
        
        node_data = graph.nodes[best_node]
        print(f"    位置 {i+1}: 节点ID={best_node}, 坐标=({node_data['y']:.6f}, {node_data['x']:.6f})")
        print(f"    服务缺口改善: {best_score:.3f}")

print(f"\n✓ 完成选址,共选择 {len(selected_nodes)} 个位置")

# ========== 7. 评估覆盖改善(优化版:批量计算) ==========
print("\n" + "="*70)
print("步骤7: 评估覆盖效果(快速批量计算)")
print("="*70)

# 采样评估(避免计算全部节点)
eval_sample_size = min(1000, len(all_nodes))  # 减少到1000个采样点
eval_nodes = np.random.choice(all_nodes, size=eval_sample_size, replace=False)

print(f"正在批量计算覆盖率...")

# 优化策略:从设施出发计算可达范围(比从需求点出发快得多!)
covered_before = set()
covered_after = set()

# 计算现有设施的覆盖范围
print(f"  计算现有设施覆盖范围...")
for i, facility_node in enumerate(existing_facilities):
    if i % 100 == 0:
        print(f"    进度: {i}/{len(existing_facilities)}")
    try:
        # 从设施出发,找到所有可达节点
        lengths = nx.single_source_dijkstra_path_length(
            graph, facility_node, cutoff=DISTANCE_LIMIT, weight='length'
        )
        # 只记录在评估样本中的节点
        covered_before.update(n for n in lengths.keys() if n in eval_nodes)
    except:
        pass

# 计算新增设施的额外覆盖
print(f"  计算新增设施覆盖范围...")
covered_after = covered_before.copy()
for i, facility_node in enumerate(selected_nodes):
    print(f"    新设施 {i+1}/{len(selected_nodes)}")
    try:
        lengths = nx.single_source_dijkstra_path_length(
            graph, facility_node, cutoff=DISTANCE_LIMIT, weight='length'
        )
        covered_after.update(n for n in lengths.keys() if n in eval_nodes)
    except:
        pass

before_coverage = len(covered_before)
after_coverage = len(covered_after)

before_rate = before_coverage / eval_sample_size * 100
after_rate = after_coverage / eval_sample_size * 100

print(f"\n覆盖率评估(基于 {eval_sample_size} 个采样点):")
print(f"  优化前: {before_coverage}/{eval_sample_size} ({before_rate:.1f}%)")
print(f"  优化后: {after_coverage}/{eval_sample_size} ({after_rate:.1f}%)")
print(f"  改善量: +{after_coverage - before_coverage} (+{after_rate - before_rate:.1f}%)")

# ========== 8. 可视化 ==========
print("\n" + "="*70)
print("步骤8: 生成可视化")
print("="*70)

fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

# ===== 子图1: 需求密度 =====
ax1 = fig.add_subplot(gs[0, 0])
im1 = ax1.contourf(xx, yy, density_demand_norm, levels=20, cmap='YlOrRd', alpha=0.7)
ax1.scatter(existing_coords[:, 0], existing_coords[:, 1], 
           c='blue', s=30, marker='o', label='现有超市', zorder=3)
ax1.set_title('需求密度分布', fontsize=14, fontweight='bold')
ax1.set_xlabel('经度')
ax1.set_ylabel('纬度')
plt.colorbar(im1, ax=ax1, label='密度')
ax1.legend()

# ===== 子图2: 供给密度 =====
ax2 = fig.add_subplot(gs[0, 1])
im2 = ax2.contourf(xx, yy, density_supply_norm, levels=20, cmap='Blues', alpha=0.7)
ax2.scatter(existing_coords[:, 0], existing_coords[:, 1], 
           c='darkblue', s=30, marker='o', label='现有超市', zorder=3)
ax2.set_title('供给密度分布', fontsize=14, fontweight='bold')
ax2.set_xlabel('经度')
ax2.set_ylabel('纬度')
plt.colorbar(im2, ax=ax2, label='密度')
ax2.legend()

# ===== 子图3: 服务缺口 =====
ax3 = fig.add_subplot(gs[0, 2])
im3 = ax3.contourf(xx, yy, service_gap, levels=20, cmap='RdYlGn_r', alpha=0.7)
ax3.scatter(existing_coords[:, 0], existing_coords[:, 1], 
           c='blue', s=30, marker='o', label='现有超市', zorder=3)
if selected_coords:
    selected_coords_array = np.array(selected_coords)
    ax3.scatter(selected_coords_array[:, 0], selected_coords_array[:, 1],
               c='red', s=200, marker='*', label='建议新增', zorder=4,
               edgecolors='darkred', linewidth=2)
ax3.set_title('服务缺口热力图', fontsize=14, fontweight='bold')
ax3.set_xlabel('经度')
ax3.set_ylabel('纬度')
plt.colorbar(im3, ax=ax3, label='缺口强度')
ax3.legend()

# ===== 子图4: 优化前 =====
ax4 = fig.add_subplot(gs[1, :2])
ox.plot_graph(graph, ax=ax4, node_size=0, edge_color='lightgray', 
              edge_linewidth=0.2, show=False, close=False)

ax4.scatter(existing_coords[:, 0], existing_coords[:, 1],
           c='blue', s=100, marker='o', label=f'现有超市({len(existing_facilities)}个)',
           zorder=4, edgecolors='white', linewidth=1.5)

ax4.set_title(f'优化前: 覆盖率 {before_rate:.1f}%', 
             fontsize=15, fontweight='bold')
ax4.set_xlabel('经度')
ax4.set_ylabel('纬度')
ax4.legend(fontsize=12)

# ===== 子图5: 优化后 =====
ax5 = fig.add_subplot(gs[1, 2])
ox.plot_graph(graph, ax=ax5, node_size=0, edge_color='lightgray', 
              edge_linewidth=0.2, show=False, close=False)

ax5.scatter(existing_coords[:, 0], existing_coords[:, 1],
           c='blue', s=100, marker='o', label=f'现有超市({len(existing_facilities)}个)',
           zorder=4, edgecolors='white', linewidth=1.5)

if selected_coords:
    selected_coords_array = np.array(selected_coords)
    ax5.scatter(selected_coords_array[:, 0], selected_coords_array[:, 1],
               c='#FF4500', s=350, marker='*', label=f'新增设施({NUM_NEW_FACILITIES}个)',
               zorder=5, edgecolors='darkred', linewidth=2)
    
    for idx, coord in enumerate(selected_coords):
        ax5.annotate(f'{idx+1}', xy=coord, xytext=(8, 8),
                    textcoords='offset points', fontsize=10,
                    fontweight='bold', color='darkred',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

ax5.set_title(f'优化后: 覆盖率 {after_rate:.1f}% (+{after_rate-before_rate:.1f}%)', 
             fontsize=15, fontweight='bold', color='darkgreen')
ax5.set_xlabel('经度')
ax5.set_ylabel('纬度')
ax5.legend(fontsize=12)

plt.suptitle('基于热力图的需求驱动选址优化', fontsize=18, fontweight='bold')
plt.savefig('facility_optimization_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ 可视化完成: facility_optimization_heatmap.png")
plt.show()

print("\n" + "="*70)
print("优化完成!")
print("="*70)


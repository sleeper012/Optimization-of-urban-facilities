"""
批量设施分析脚本 - 一键运行所有设施类型的可达性分析和选址优化
适用于15分钟生活圈研究

使用方法:
    python batch_analysis.py --all                    # 运行所有分析
    python batch_analysis.py --accessibility          # 仅运行可达性分析
    python batch_analysis.py --optimization mclp      # 仅运行MCLP优化
    python batch_analysis.py --facility supermarket   # 仅分析超市
"""

import geopandas as gpd
import osmnx as ox
import networkx as nx
import numpy as np
from pulp import *
import matplotlib.pyplot as plt
import matplotlib
from sklearn.cluster import KMeans
import argparse
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# ==================== 配置参数 ====================
WALK_SPEED = 4.5  # km/h
TIME_LIMIT = 15   # 分钟
DISTANCE_LIMIT = (WALK_SPEED * 1000 / 60) * TIME_LIMIT

NUM_NEW_FACILITIES = 5  # 新增设施数量
NUM_DEMAND_POINTS = 1000
NUM_CANDIDATE_POINTS = 200

# 设施配置字典
FACILITY_CONFIG = {
    '社区综合服务站': {
        'filter': lambda df: df[df['amenity'].isin(['community_centre', 'townhall'])],
        'color': '#FF6B6B',
        'marker': '★',
        'priority': 1
    },
    '幼儿园': {
        'filter': lambda df: df[df['amenity'] == 'kindergarten'],
        'color': '#4ECDC4',
        'marker': '●',
        'priority': 2
    },
    '托儿所': {
        'filter': lambda df: df[df['amenity'] == 'childcare'],
        'color': '#45B7D1',
        'marker': '▲',
        'priority': 3
    },
    '老年服务站': {
        'filter': lambda df: df[df['amenity'] == 'social_facility'],
        'color': '#96CEB4',
        'marker': '■',
        'priority': 4
    },
    '医疗服务': {
        'filter': lambda df: df[df['amenity'].isin(['clinic', 'doctors', 'dentist'])],
        'color': '#E74C3C',
        'marker': '+',
        'priority': 5
    },
    '药店': {
        'filter': lambda df: df[df['amenity'] == 'pharmacy'],
        'color': '#3498DB',
        'marker': '⊕',
        'priority': 6
    },
    '超市': {
        'filter': lambda df: df[df['shop'] == 'supermarket'],
        'color': '#F39C12',
        'marker': '♦',
        'priority': 7
    },
    '公共绿地': {
        'filter': lambda df: df[df['leisure'].isin(['park', 'garden'])],
        'color': '#27AE60',
        'marker': '◆',
        'priority': 8
    }
}

# ==================== 工具函数 ====================
def load_data():
    """加载数据"""
    print("\n[加载数据]")
    facilities = gpd.read_file("munich_public_facilities.geojson")
    graph = ox.load_graphml("munich_street_network.graphml")
    print(f"✓ 设施数: {len(facilities)} | 路网节点: {len(graph.nodes)}")
    return facilities, graph

def match_facilities_to_network(facilities_gdf, graph):
    """将设施匹配到路网节点"""
    # 确保是GeoDataFrame且已经是正确的坐标系
    if not isinstance(facilities_gdf, gpd.GeoDataFrame):
        print("⚠️  输入不是GeoDataFrame")
        return facilities_gdf
    
    # 检查并转换坐标系
    if facilities_gdf.crs is None:
        facilities_gdf = facilities_gdf.set_crs(epsg=4326)
    elif facilities_gdf.crs.to_epsg() != 4326:
        facilities_gdf = facilities_gdf.to_crs(epsg=4326)
    
    nearest_nodes = []
    
    for idx, row in facilities_gdf.iterrows():
        try:
            point = row.geometry.centroid if row.geometry.geom_type == 'Polygon' else row.geometry
            nearest_node = ox.distance.nearest_nodes(graph, point.x, point.y)
            nearest_nodes.append(nearest_node)
        except:
            nearest_nodes.append(None)
    
    facilities_gdf['nearest_node'] = nearest_nodes
    facilities_gdf = facilities_gdf[facilities_gdf['nearest_node'].notna()]
    
    return facilities_gdf

# ==================== 可达性分析 ====================
def accessibility_analysis(facility_name, facilities_gdf, graph, output_dir):
    """执行15分钟步行圈可达性分析"""
    print(f"\n{'='*70}")
    print(f"[可达性分析] {facility_name}")
    print(f"{'='*70}")
    
    if len(facilities_gdf) == 0:
        print(f"⚠️  {facility_name} 数量为0，跳过分析")
        return None
    
    print(f"设施数量: {len(facilities_gdf)}")
    
    # 计算覆盖率
    total_nodes = len(graph.nodes)
    covered_nodes = set()
    
    print("正在计算覆盖范围...")
    for idx, row in facilities_gdf.iterrows():
        center_node = row['nearest_node']
        try:
            subgraph = nx.ego_graph(graph, center_node, radius=DISTANCE_LIMIT, distance='length')
            covered_nodes.update(subgraph.nodes)
        except:
            pass
    
    coverage_rate = len(covered_nodes) / total_nodes * 100
    
    print(f"\n覆盖统计:")
    print(f"  总节点数: {total_nodes}")
    print(f"  覆盖节点数: {len(covered_nodes)}")
    print(f"  覆盖率: {coverage_rate:.2f}%")
    
    # 可视化（选取前3个示例）
    sample_size = min(3, len(facilities_gdf))
    fig, axes = plt.subplots(1, sample_size, figsize=(18, 6))
    if sample_size == 1:
        axes = [axes]
    
    for i in range(sample_size):
        center_node = facilities_gdf.iloc[i]['nearest_node']
        
        try:
            subgraph = nx.ego_graph(graph, center_node, radius=DISTANCE_LIMIT, distance='length')
            
            ax = axes[i]
            ox.plot_graph(subgraph, ax=ax, node_size=0, edge_color='lightblue', 
                          edge_linewidth=0.5, show=False, close=False)
            
            center_y = graph.nodes[center_node]['y']
            center_x = graph.nodes[center_node]['x']
            ax.scatter(center_x, center_y, c='red', s=200, marker='*', 
                       zorder=5, edgecolors='black', linewidth=2)
            
            ax.set_title(f'{facility_name} #{i+1}\n15分钟步行圈\n(覆盖{len(subgraph.nodes)}个路口)', 
                         fontsize=12, fontweight='bold')
            ax.set_xlabel('经度')
            ax.set_ylabel('纬度')
        except:
            pass
    
    plt.tight_layout()
    filename = os.path.join(output_dir, f'accessibility_{facility_name}.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 可视化完成: {filename}")
    
    return {
        'facility_name': facility_name,
        'total_facilities': len(facilities_gdf),
        'coverage_rate': coverage_rate,
        'covered_nodes': len(covered_nodes),
        'uncovered_nodes': total_nodes - len(covered_nodes)
    }

# ==================== MCLP优化 ====================
def mclp_optimization(facility_name, facilities_gdf, graph, output_dir):
    """最大覆盖选址优化"""
    print(f"\n{'='*70}")
    print(f"[MCLP优化] {facility_name}")
    print(f"{'='*70}")
    
    if len(facilities_gdf) < 3:
        print(f"⚠️  {facility_name} 数量过少({len(facilities_gdf)})，跳过优化")
        return None
    
    # 1. 提取现有设施
    existing_facilities = list(facilities_gdf['nearest_node'].values)
    print(f"现有设施: {len(existing_facilities)} 个")
    
    # 2. 生成需求点
    all_nodes = list(graph.nodes())
    np.random.seed(42)
    
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
    print(f"未覆盖节点: {len(uncovered_nodes)} ({len(uncovered_nodes)/len(all_nodes)*100:.1f}%)")
    
    # 混合采样
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
    print(f"需求点: {len(demand_nodes)}")
    
    # 3. 生成候选位置
    if len(uncovered_nodes) > NUM_CANDIDATE_POINTS:
        uncovered_coords = np.array([
            [graph.nodes[n]['x'], graph.nodes[n]['y']] 
            for n in uncovered_nodes
        ])
        
        kmeans = KMeans(n_clusters=NUM_CANDIDATE_POINTS, random_state=42, n_init=10)
        kmeans.fit(uncovered_coords)
        
        candidate_nodes = []
        for center in kmeans.cluster_centers_:
            nearest = ox.distance.nearest_nodes(graph, center[0], center[1])
            if nearest not in existing_facilities:
                candidate_nodes.append(nearest)
    else:
        candidate_nodes = [n for n in uncovered_nodes if n not in existing_facilities]
    
    print(f"候选位置: {len(candidate_nodes)}")
    
    # 4. 计算距离矩阵
    all_facility_nodes = existing_facilities + candidate_nodes
    distance_matrix = np.full((len(demand_nodes), len(all_facility_nodes)), np.inf)
    
    print("正在计算距离矩阵...")
    for i, demand in enumerate(demand_nodes):
        try:
            lengths = nx.single_source_dijkstra_path_length(
                graph, demand, cutoff=DISTANCE_LIMIT*1.5, weight='length'
            )
            for j, facility in enumerate(all_facility_nodes):
                if facility in lengths:
                    distance_matrix[i, j] = lengths[facility]
        except:
            pass
    
    before_coverage = sum(
        1 for i in range(len(demand_nodes))
        if any(distance_matrix[i][j] <= DISTANCE_LIMIT 
               for j in range(len(existing_facilities)))
    )
    
    # 5. 构建模型
    print("正在构建优化模型...")
    model = LpProblem("MCLP", LpMaximize)
    
    y = LpVariable.dicts("build", range(len(candidate_nodes)), cat='Binary')
    z = LpVariable.dicts("covered", range(len(demand_nodes)), cat='Binary')
    
    model += lpSum([z[i] for i in range(len(demand_nodes))])
    
    for i in range(len(demand_nodes)):
        covering_facilities = []
        
        for j in range(len(existing_facilities)):
            if distance_matrix[i][j] <= DISTANCE_LIMIT:
                covering_facilities.append(('existing', j))
        
        for j in range(len(candidate_nodes)):
            facility_idx = j + len(existing_facilities)
            if distance_matrix[i][facility_idx] <= DISTANCE_LIMIT:
                covering_facilities.append(('candidate', j))
        
        if covering_facilities:
            model += z[i] <= lpSum([
                1 if ftype == 'existing' else y[fid]
                for ftype, fid in covering_facilities
            ])
    
    model += lpSum([y[j] for j in range(len(candidate_nodes))]) == NUM_NEW_FACILITIES
    
    # 6. 求解
    print("正在求解...")
    model.solve(PULP_CBC_CMD(msg=0, timeLimit=300))
    
    if model.status != 1:
        print(f"❌ 求解失败")
        return None
    
    # 7. 提取结果
    selected_indices = [j for j in range(len(candidate_nodes)) if value(y[j]) == 1]
    selected_nodes = [candidate_nodes[j] for j in selected_indices]
    after_coverage = sum(1 for i in range(len(demand_nodes)) if value(z[i]) == 1)
    improvement = after_coverage - before_coverage
    
    print(f"\n优化结果:")
    print(f"  优化前覆盖: {before_coverage}/{len(demand_nodes)} ({before_coverage/len(demand_nodes)*100:.1f}%)")
    print(f"  优化后覆盖: {after_coverage}/{len(demand_nodes)} ({after_coverage/len(demand_nodes)*100:.1f}%)")
    print(f"  改善量: +{improvement} (+{improvement/len(demand_nodes)*100:.1f}%)")
    
    # 8. 可视化（保留原始样式）
    print("正在生成可视化...")
    fig, axes = plt.subplots(1, 2, figsize=(24, 10))
    
    # ===== 左图: 优化前 =====
    ax1 = axes[0]
    ox.plot_graph(graph, ax=ax1, node_size=0, edge_color='#E8E8E8', 
                  edge_linewidth=0.3, show=False, close=False)
    
    # 现有设施覆盖范围(浅蓝色圆圈)
    print("  绘制优化前覆盖范围...")
    for facility_node in existing_facilities[:50]:  # 限制数量避免过密
        try:
            subgraph = nx.ego_graph(graph, facility_node, radius=DISTANCE_LIMIT, distance='length')
            node_coords = [(graph.nodes[n]['x'], graph.nodes[n]['y']) for n in subgraph.nodes()]
            ax1.scatter([c[0] for c in node_coords], [c[1] for c in node_coords],
                       c='lightblue', s=2, alpha=0.3, zorder=1)
        except:
            pass
    
    # 现有设施点
    existing_coords = [(graph.nodes[n]['x'], graph.nodes[n]['y']) for n in existing_facilities]
    ax1.scatter([c[0] for c in existing_coords], [c[1] for c in existing_coords],
               c='blue', s=80, marker='o', label=f'现有设施({len(existing_facilities)}个)', 
               zorder=4, edgecolors='white', linewidth=1)
    
    # 未覆盖需求点（红色×）
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
               c='blue', s=80, marker='o', label=f'现有{facility_name}({len(existing_facilities)}个)', 
               zorder=4, edgecolors='white', linewidth=1)
    
    # 新增设施及其覆盖范围（浅红色圆圈）
    new_coords = [(graph.nodes[n]['x'], graph.nodes[n]['y']) for n in selected_nodes]
    
    print("  绘制新增设施覆盖范围...")
    for facility_node in selected_nodes:
        try:
            subgraph = nx.ego_graph(graph, facility_node, radius=DISTANCE_LIMIT, distance='length')
            node_coords = [(graph.nodes[n]['x'], graph.nodes[n]['y']) for n in subgraph.nodes()]
            ax2.scatter([c[0] for c in node_coords], [c[1] for c in node_coords],
                       c='lightcoral', s=3, alpha=0.4, zorder=1)
        except:
            pass
    
    # 新增设施点（红色星星）
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
    
    plt.suptitle(f'最大覆盖选址优化 - {facility_name}设施布局改善方案', 
                fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    filename = os.path.join(output_dir, f'optimization_mclp_{facility_name}.png')
    plt.savefig(filename, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"✓ 可视化完成: {filename}")
    
    return {
        'facility_name': facility_name,
        'before_coverage': before_coverage / len(demand_nodes) * 100,
        'after_coverage': after_coverage / len(demand_nodes) * 100,
        'improvement': improvement / len(demand_nodes) * 100,
        'selected_locations': [(graph.nodes[n]['y'], graph.nodes[n]['x']) for n in selected_nodes]
    }

# ==================== 主函数 ====================
def main():
    parser = argparse.ArgumentParser(description='批量设施分析脚本')
    parser.add_argument('--all', action='store_true', help='运行所有分析')
    parser.add_argument('--accessibility', action='store_true', help='仅运行可达性分析')
    parser.add_argument('--optimization', type=str, choices=['mclp'], help='运行优化分析')
    parser.add_argument('--facility', type=str, help='指定设施类型')
    parser.add_argument('--output', type=str, default='results', help='输出目录')
    
    args = parser.parse_args()
    
    # 创建输出目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"{'='*70}")
    print(f"批量设施分析脚本")
    print(f"输出目录: {output_dir}")
    print(f"{'='*70}")
    
    # 加载数据
    facilities, graph = load_data()
    
    # 确定要分析的设施类型
    if args.facility:
        if args.facility not in FACILITY_CONFIG:
            print(f"❌ 未知设施类型: {args.facility}")
            print(f"可选类型: {', '.join(FACILITY_CONFIG.keys())}")
            return
        facility_list = [args.facility]
    else:
        facility_list = sorted(FACILITY_CONFIG.keys(), 
                              key=lambda x: FACILITY_CONFIG[x]['priority'])
    
    # 运行分析
    results = {
        'accessibility': [],
        'optimization': []
    }
    
    for facility_name in facility_list:
        print(f"\n{'#'*70}")
        print(f"# 处理: {facility_name}")
        print(f"{'#'*70}")
        
        # 提取设施
        config = FACILITY_CONFIG[facility_name]
        facility_gdf = config['filter'](facilities)
        
        if len(facility_gdf) == 0:
            print(f"⚠️  {facility_name} 无数据，跳过")
            continue
        
        # 匹配到路网
        facility_gdf = match_facilities_to_network(facility_gdf, graph)
        print(f"✓ 匹配到路网: {len(facility_gdf)} 个")
        
        # 可达性分析
        if args.all or args.accessibility:
            result = accessibility_analysis(facility_name, facility_gdf, graph, output_dir)
            if result:
                results['accessibility'].append(result)
        
        # 优化分析
        if args.all or args.optimization == 'mclp':
            result = mclp_optimization(facility_name, facility_gdf, graph, output_dir)
            if result:
                results['optimization'].append(result)
    
    # 生成汇总报告
    print(f"\n{'='*70}")
    print("生成汇总报告")
    print(f"{'='*70}")
    
    report_lines = [
        "=" * 70,
        "批量设施分析报告",
        "=" * 70,
        f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"输出目录: {output_dir}\n",
    ]
    
    if results['accessibility']:
        report_lines.append("\n" + "="*70)
        report_lines.append("可达性分析结果")
        report_lines.append("="*70)
        for r in results['accessibility']:
            report_lines.append(f"\n【{r['facility_name']}】")
            report_lines.append(f"  设施总数: {r['total_facilities']}")
            report_lines.append(f"  覆盖率: {r['coverage_rate']:.2f}%")
            report_lines.append(f"  覆盖节点: {r['covered_nodes']}")
            report_lines.append(f"  未覆盖节点: {r['uncovered_nodes']}")
    
    if results['optimization']:
        report_lines.append("\n" + "="*70)
        report_lines.append("MCLP优化结果")
        report_lines.append("="*70)
        for r in results['optimization']:
            report_lines.append(f"\n【{r['facility_name']}】")
            report_lines.append(f"  优化前覆盖: {r['before_coverage']:.1f}%")
            report_lines.append(f"  优化后覆盖: {r['after_coverage']:.1f}%")
            report_lines.append(f"  改善幅度: +{r['improvement']:.1f}%")
            report_lines.append(f"  新增位置数: {len(r['selected_locations'])}")
    
    report_lines.append("\n" + "="*70)
    report_lines.append("分析完成!")
    report_lines.append("="*70)
    
    report_text = "\n".join(report_lines)
    print(report_text)
    
    # 保存报告
    report_file = os.path.join(output_dir, 'analysis_report.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"\n✓ 报告已保存: {report_file}")

if __name__ == '__main__':
    main()
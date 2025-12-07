import geopandas as gpd
import matplotlib.pyplot as plt
import osmnx as ox
import matplotlib.lines as mlines
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体
matplotlib.rcParams['axes.unicode_minus'] = False

print("正在加载数据 (这可能需要几秒钟)...")
facilities = gpd.read_file("munich_public_facilities.geojson")
graph = ox.load_graphml("munich_street_network.graphml")

# 转换为投影坐标系 (UTM 32N)，这样点的大小单位就是米，更准确
# 慕尼黑位于 UTM zone 32N (EPSG:32632)
facilities = facilities.to_crs(epsg=32632)
# Graph 也需要投影才能配合绘图(ox.plot_graph默认使用原坐标，我们需要统一)
graph_proj = ox.project_graph(graph, to_crs='EPSG:32632')

print("数据加载完成，开始绘图...")

# 创建设施分类
categories = {
    '幼儿园': {'data': facilities[facilities['amenity'] == 'kindergarten'], 'color': '#00FF00', 'marker': 'o'}, # 荧光绿
    '医疗': {'data': facilities[facilities['amenity'].isin(['clinic', 'doctors', 'dentist'])], 'color': '#FF00FF', 'marker': '+'}, # 品红
    '超市': {'data': facilities[facilities['shop'] == 'supermarket'], 'color': '#00FFFF', 'marker': 's'}, # 青色
    '公园': {'data': facilities[facilities['leisure'].isin(['park', 'garden'])], 'color': '#FFFF00', 'marker': '*'}, # 亮黄
}

# 设置暗色背景风格
plt.style.use('dark_background')
fig, axes = plt.subplots(2, 2, figsize=(20, 20), facecolor='black')
axes = axes.flatten()

# 统一获取路网的边界，确保所有子图范围一致（美观）
# 我们可以直接用 ox.plot_graph 的特性
node_points = ox.graph_to_gdfs(graph_proj, nodes=True, edges=False)
total_bounds = node_points.total_bounds # [minx, miny, maxx, maxy]

for idx, (name, info) in enumerate(categories.items()):
    ax = axes[idx]
    data = info['data']
    color = info['color']
    
    print(f"正在绘制: {name}...")
    
    # 1. 绘制路网底图 (深灰色，极细)
    # 使用 ox.plot_graph 绘制到底层
    ox.plot_graph(graph_proj, ax=ax, node_size=0, edge_color='#333333', 
                  edge_linewidth=0.3, show=False, close=False, bgcolor='black')
    
    # 2. 绘制设施点 (高亮)
    # 根据数据类型决定绘制方式
    if len(data) > 0:
        # 区分点和多边形
        points = data[data.geometry.type == 'Point']
        polys = data[data.geometry.type.isin(['Polygon', 'MultiPolygon'])]
        
        if not points.empty:
            points.plot(ax=ax, color=color, markersize=15, alpha=0.8, zorder=5)
        
        if not polys.empty:
            polys.plot(ax=ax, color=color, alpha=0.6, zorder=5)
            # 同时也画出多边形的中心点，增加视觉显著性
            polys.centroid.plot(ax=ax, color=color, markersize=15, alpha=0.8, zorder=5)

    # 3. 美化图表
    ax.set_title(f'{name}分布 ({len(data)}个)', fontsize=20, color='white', pad=20, fontweight='bold')
    
    # 移除坐标轴刻度，只保留纯净地图
    ax.axis('off')
    
    # 手动添加图例
    legend_handle = mlines.Line2D([], [], color=color, marker=info['marker'], linestyle='None',
                          markersize=10, label=f'{name}位置')
    ax.legend(handles=[legend_handle], loc='upper right', fontsize=12, facecolor='#222222', edgecolor='white')

    # 强制设置相同的视野范围
    ax.set_xlim(total_bounds[0], total_bounds[2])
    ax.set_ylim(total_bounds[1], total_bounds[3])

plt.suptitle("慕尼黑核心公共服务设施空间分布图", fontsize=28, color='white', y=0.95)
plt.tight_layout(rect=[0, 0, 1, 0.93]) # 留出标题空间

output_file = 'munich_facilities_dark_mode.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='black')
print(f"\n可视化完成! 高清美图已保存为: {output_file}")
plt.show()

# 简单的文本统计
print("\n" + "="*50)
print("可视化统计信息:")
for name, info in categories.items():
    print(f"  - {name}: {len(info['data'])} 个点位")
print("="*50)

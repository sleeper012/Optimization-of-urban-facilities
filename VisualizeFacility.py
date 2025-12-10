import geopandas as gpd
import matplotlib.pyplot as plt
import osmnx as ox
import matplotlib.lines as mlines
import matplotlib
import os # 引入 os 模块用于创建目录

# --- 1. 初始化设置 ---
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体
matplotlib.rcParams['axes.unicode_minus'] = False

# 定义输出目录
output_dir = './visualresults/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"已创建输出目录: {output_dir}")

print("正在加载数据 (这可能需要几秒钟)...")
# 假设文件在运行脚本的当前目录
try:
    facilities = gpd.read_file("munich_public_facilities.geojson")
    graph = ox.load_graphml("munich_street_network.graphml")
except FileNotFoundError as e:
    print(f"错误: 找不到数据文件。请确保 .geojson 和 .graphml 文件在当前目录下。\n详情: {e}")
    exit()

# 转换为投影坐标系 (UTM 32N)，慕尼黑位于 UTM zone 32N (EPSG:32632)
facilities = facilities.to_crs(epsg=32632)
# Graph 也需要投影才能配合绘图
graph_proj = ox.project_graph(graph, to_crs='EPSG:32632')

print("数据加载完成，开始逐一绘图...")

# --- 2. 设施分类定义 (应用新配色方案) ---
# 颜色代码源自用户提供的 RGB 图片转换
categories = {
    '社区综合服务': {'data': facilities[facilities['amenity'].isin(['community_centre', 'townhall'])], 'color': '#3951A2', 'marker': 'o'}, # 深蓝
    '医疗': {'data': facilities[facilities['amenity'].isin(['clinic', 'doctors', 'dentist'])], 'color': '#72AACF', 'marker': '+'}, # 中蓝
    '超市': {'data': facilities[facilities['shop'] == 'supermarket'], 'color': '#CAE8F2', 'marker': 's'}, # 浅蓝
    '公园': {'data': facilities[facilities['leisure'].isin(['park', 'garden'])], 'color': '#FEFBBA', 'marker': '*'}, # 米黄
    '停车充电': {'data': facilities[facilities['amenity'].isin(['parking', 'charging_station'])], 'color': '#FDB96B', 'marker': 's'}, # 浅橙
    '公共绿地': {'data': facilities[facilities['leisure'].isin(['park', 'garden'])], 'color': '#EC5D3B', 'marker': '*'}, # 橙红
    '运动场地': {'data': facilities[facilities['leisure'].isin(['playground', 'pitch', 'sports_centre', 'fitness_station'])], 'color': '#A80326', 'marker': 's'}, # 深红
}

# 统一获取路网的边界，确保所有子图范围一致（美观）
node_points = ox.graph_to_gdfs(graph_proj, nodes=True, edges=False)
total_bounds = node_points.total_bounds # [minx, miny, maxx, maxy]

# 设置全局点标记大小 (较小以保持美观)
NEW_MARKERSIZE = 1

# --- 3. 循环绘图并保存 ---
plt.style.use('dark_background') # 设置暗色背景风格

for name, info in categories.items():
    data = info['data']
    color = info['color']
    marker = info['marker']
    
    print(f"正在绘制并保存: {name}...")
    
    # 为每个类别创建新的 Figure 和 Axes
    fig, ax = plt.subplots(1, 1, figsize=(10, 10), facecolor='black')
    
    # 1. 绘制路网底图 (深灰色，极细)
    ox.plot_graph(graph_proj, ax=ax, node_size=0, edge_color='#333333', 
                  edge_linewidth=0.3, show=False, close=False, bgcolor='black')
    
    # 2. 绘制设施点 (应用新颜色和较小的尺寸)
    if len(data) > 0:
        # 区分点和多边形
        points = data[data.geometry.type == 'Point']
        polys = data[data.geometry.type.isin(['Polygon', 'MultiPolygon'])]
        
        # 使用新的点大小
        if not points.empty:
            points.plot(ax=ax, color=color, markersize=NEW_MARKERSIZE, alpha=0.9, zorder=5, marker=marker)
        
        if not polys.empty:
            # 多边形填充透明度稍高一点，让底图显露
            polys.plot(ax=ax, color=color, alpha=0.5, zorder=4)
            # 同时也画出多边形的中心点，增加视觉显著性
            polys.centroid.plot(ax=ax, color=color, markersize=NEW_MARKERSIZE, alpha=0.9, zorder=5, marker=marker)

    # 3. 美化图表
    ax.set_title(f'慕尼黑 {name}分布 ({len(data)}个)', fontsize=20, color='white', pad=20, fontweight='bold')
    
    # 移除坐标轴刻度，只保留纯净地图
    ax.axis('off')
    
    # 手动添加图例 (图例标记稍微大一点以便看清形状)
    legend_handle = mlines.Line2D([], [], color=color, marker=marker, linestyle='None',
                          markersize=12, label=f'{name}位置')
    ax.legend(handles=[legend_handle], loc='upper right', fontsize=12, facecolor='#222222', edgecolor='white')

    # 强制设置相同的视野范围
    ax.set_xlim(total_bounds[0], total_bounds[2])
    ax.set_ylim(total_bounds[1], total_bounds[3])

    # 4. 保存图表
    # 构造文件名，将中文名称转换为拼音或英文以避免文件系统问题，并添加颜色提示
    color_name = color.replace('#', '')
    filename = f"munich_{name}_{color_name}.png".replace(' ', '_').replace('/', '_') 
    output_path = os.path.join(output_dir, filename)
    
    plt.savefig(output_path, dpi=800, bbox_inches='tight', facecolor='black')
    plt.close(fig) # 关闭当前图表，释放内存
    
    print(f"  -> 已保存至: {output_path}")


print("\n" + "="*50)
print("所有设施的可视化已完成并保存到 ./visualresults/ 目录中。")
print("="*50)

# 简单的文本统计
print("\n可视化统计信息:")
for name, info in categories.items():
    print(f"  - {name}: {len(info['data'])} 个点位 (颜色: {info['color']})")
print("="*50)
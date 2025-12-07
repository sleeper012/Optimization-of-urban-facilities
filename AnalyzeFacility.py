import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import osmnx as ox
from shapely.geometry import Point
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体支持
matplotlib.rcParams['axes.unicode_minus'] = False

print("正在加载数据...")

# 1. 加载设施数据
facilities = gpd.read_file("munich_public_facilities.geojson")
print(f"设施数据加载完成: {len(facilities)} 个设施")

# 2. 加载路网数据
graph = ox.load_graphml("munich_street_network.graphml")
print(f"路网数据加载完成: {len(graph.nodes)} 个节点, {len(graph.edges)} 条边")

# 3. 数据清洗与分类
print("\n" + "="*50)
print("开始数据分析...")

# 创建设施分类字典
facility_categories = {
    '社区综合服务': facilities[facilities['amenity'].isin(['community_centre', 'townhall'])],
    '幼儿园': facilities[facilities['amenity'] == 'kindergarten'],
    '托儿所': facilities[facilities['amenity'] == 'childcare'],
    '老年服务站': facilities[facilities['amenity'] == 'social_facility'],
    '医疗服务': facilities[facilities['amenity'].isin(['clinic', 'doctors', 'dentist'])],
    '药店': facilities[facilities['amenity'] == 'pharmacy'],
    '超市': facilities[facilities['shop'] == 'supermarket'],
    '邮件快递': facilities[facilities['amenity'].isin(['post_office', 'post_box', 'parcel_locker'])],
    '餐饮': facilities[facilities['amenity'].isin(['restaurant', 'cafe', 'fast_food'])],
    '理发店': facilities[facilities['shop'] == 'hairdresser'],
    '停车充电': facilities[facilities['amenity'].isin(['parking', 'charging_station'])],
    '公共绿地': facilities[facilities['leisure'].isin(['park', 'garden'])],
    '运动场地': facilities[facilities['leisure'].isin(['playground', 'pitch', 'sports_centre', 'fitness_station'])],
}

# 4. 统计各类设施数量
print("\n各类设施统计:")
print("-"*50)
for category, data in facility_categories.items():
    print(f"{category:12s}: {len(data):5d} 个")

# 5. 筛选核心设施(用于后续可达性分析)
core_facilities = {
    '幼儿园': facility_categories['幼儿园'],
    '医疗服务': facility_categories['医疗服务'],
    '超市': facility_categories['超市'],
    '公共绿地': facility_categories['公共绿地'],
}

print("\n" + "="*50)
print("核心设施分类完成,可用于下一步可达性分析!")
print("\n建议下一步:")
print("1. 可视化设施分布 (使用 VisualizeFacility.py)")
print("2. 计算15分钟步行圈覆盖范围")
print("3. 识别服务不足区域")
print("4. 运行设施选址优化模型")


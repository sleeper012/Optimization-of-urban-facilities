import osmnx as ox
import pandas as pd

# 配置: 增加超时时间，因为慕尼黑数据量较大，且我们需要下载很多类型的设施
ox.settings.timeout = 10000 

# Step 1: 定义地点
place_name = "Munich, Germany"
print(f"正在下载 {place_name} 的数据，请耐心等待...")

# Step 2: 下载街道网络 (对应：慢行系统、路网)
# network_type='all' 包含了车道、步行道和自行车道（慢行系统）
graph = ox.graph_from_place(place_name, network_type='all')
print("街道网络下载完成。")

# Step 3: 定义公共设施的标签 (Tags)
# 根据你提供的15项需求，映射到 OpenStreetMap 的标签体系
tags = {
    # --- 基本公共服务 ---
    'amenity': [
        'community_centre',      # 1. 社区综合服务站
        'kindergarten',          # 2. 幼儿园
        'childcare',             # 3. 托儿所
        'social_facility',       # 4. 老年服务站 (需后续筛选 social_facility:for=senior)
        'clinic', 'doctors', 'dentist', 'pharmacy', # 5. 社区卫生服务站 & 8. 药店
        'post_office', 'post_box', 'parcel_locker', # 7. 邮件和快件寄递
        'restaurant', 'cafe', 'fast_food', # 8. 餐饮店
        'charging_station', 'parking',     # 10. 停车寄充电设施
        'bicycle_parking', 'bicycle_rental', # 11. 慢行系统配套(停车/租赁)
        'waste_basket', 'recycling', 'toilets', 'waste_disposal', # 13. 环境卫生设施
    ],
    
    # --- 便民商业 ---
    'shop': [
        'supermarket',           # 6. 综合超市
        'hairdresser',           # 8. 理发店
        'car_repair', 'bicycle_repair', 'electronics', # 8. 维修点
        'convenience', 'bakery', 'butcher', 'greengrocer' # 8. 其他便民店
    ],
    
    # --- 公共活动与绿地 ---
    'leisure': [
        'park',                  # 15. 公共绿地
        'garden',                # 15. 公共绿地
        'playground',            # 14. 公共活动场地
        'pitch',                 # 14. 球场
        'fitness_station',       # 14. 健身设施
        'sports_centre'          # 14. 运动中心
    ],
    
    # --- 土地利用 (补充绿地) ---
    'landuse': [
        'grass', 'recreation_ground', 'village_green'
    ],
    
    # --- 办公 (补充社区服务) ---
    'office': [
        'government', 'community_centre'
    ],
    
    # --- 基础设施 (部分) ---
    # 注意：地下的水电气热管网在OSM中通常不公开，只能获取变电站等地表设施
    'power': ['substation', 'plant'] # 9. 电力设施
}

# Step 4: 下载公共设施数据
print("正在下载公共设施数据 (这可能需要几分钟)...")
facilities = ox.features.features_from_place(place_name, tags=tags)

# Step 5: 保存数据
# 保存街道网络
ox.save_graphml(graph, "munich_street_network.graphml")

# 保存公共设施数据
# 注意：GeoJSON不支持复杂列类型，我们需要保留geometry列，其他列转为字符串
facilities_copy = facilities.copy()
for col in facilities_copy.columns:
    if col != 'geometry':
        facilities_copy[col] = facilities_copy[col].astype(str)
facilities_copy.to_file("munich_public_facilities.geojson", driver='GeoJSON')

print("数据保存成功！")

# Step 6: 详细的统计与查看
print(f"共下载了 {len(facilities)} 个设施点/区域。")
print("=" * 50)

# 遍历我们定义的所有标签类别，统计每个类别的数据
tag_categories = ['amenity', 'shop', 'leisure', 'landuse', 'office', 'power']

for category in tag_categories:
    if category in facilities.columns:
        counts = facilities[category].value_counts()
        if len(counts) > 0:
            print(f"\n【{category.upper()} 类别】- 共 {counts.sum()} 个")
            print("-" * 50)
            print(counts.to_string())
        else:
            print(f"\n【{category.upper()} 类别】- 无数据")
    else:
        print(f"\n【{category.upper()} 类别】- 该列不存在")

print("\n" + "=" * 50)
print("数据保存位置:")
print("  - 路网: munich_street_network.graphml")
print("  - 设施: munich_public_facilities.geojson")
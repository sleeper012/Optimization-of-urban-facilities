import osmnx as ox

# Step 1: 下载柏林的街道网络
place_name = "Berlin, Germany"

# 下载柏林的街道网络（可以选择“drive”, “walk”, “bike”等）
graph = ox.graph_from_place(place_name, network_type='all')  # 使用'所有'类型的网络（包括步行、开车等）

# Step 2: 下载柏林的公共设施数据
# 我们可以查询柏林的公共设施，如医院、学校、超市等
tags = {
    'amenity': True  # 返回所有的公共设施
}
facilities = ox.features.features_from_place(place_name, tags=tags)

# Step 3: 保存街道网络和公共设施数据
# 保存街道网络为GraphML格式
ox.save_graphml(graph, "berlin_street_network.graphml")

# 保存公共设施数据为GeoJSON格式
facilities.to_file("berlin_public_facilities.geojson")

# 打印公共设施的前几行查看数据
print(facilities.head())

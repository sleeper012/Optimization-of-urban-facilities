# Optimization-of-urban-facilities

##  ✅ 一、从“宜居城市”角度细化的城市计算选题（以服务可达性为核心）

基于文章关键词：**老旧小区、15 分钟生活圈、完整社区、公共服务可获得性、一老一小、绿地系统**。

---

## **课题 1：基于 15 分钟生活圈的老旧小区公共服务可达性测度与优化研究**

**关键词对应：** “老旧小区改造”“服务可获得性”“完整社区”“一老一小”

**可做内容：**

* 使用 POI 数据提取基本公共服务（医疗、菜店、养老、托幼等）
* 利用路网数据计算步行 15 分钟等时圈
* 评估每个小区的实际服务覆盖程度
* 使用优化算法（p-median/设施选址模型）给出“新增设施最优点位”

**城市计算味道：**
* 时空数据分析 + 优化模型

---


### 社区基本公共服务设施

##### 基本公共服务设施

1.一个社区综合服务站

2.一个幼儿园

3.一个托儿所

4.一个老年服务站

5.一个社区卫生服务站

##### 便民商业服务

6.一个综合超市

7.多个邮件和快件寄递服务设施

8.其他便民商业网点（理发店、药店、维修点、餐饮店等）

##### 市政配套基础设施

9.水、电、路、气、热、信等设施

10.停车寄充电设施

11.慢行系统

12.无障碍设施

13.环境卫生设施

##### 公共活动空间

14、公共活动场地（面积不小于150平方米）

15.公共绿地





社区综合服务站

幼儿园

托儿所

老年服务站

社区卫生服务

综合超市

邮件和快件寄递服务设施

其他便民商业网点（理发店、药店、维修点、餐饮店等）

停车充电设施

环境卫生设施





### 需要三个层面的数据：

1.公共服务设施数据：高德地图API、OpenStreetMap（缺少小微设施）

2.老旧小区数据：直到小区位置和范围、人口估算

3.路网数据：OpenStreetMap



> 函数说明

#### `MapFacility.py`

将地点改为了 **慕尼黑 (Munich, Germany)**，并且针对你列出的 **“社区基本公共服务设施”** 详细清单，我构建了一个涵盖面非常广的 `tags` 字典。

```python
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

# 保存街道网络
ox.save_graphml(graph, "munich_street_network.graphml")

# 保存公共设施数据
facilities_str.to_file("munich_public_facilities.geojson", driver='GeoJSON')

```

-----

### 代码逻辑与标签映射详解

为了确保你获取的数据准确覆盖了“社区基本公共服务设施”，我做了以下处理：

#### 1\. 基本公共服务设施

  * **社区服务/老年服务:** 使用了 `amenity='social_facility'` 和 `amenity='community_centre'`。
      * *注意：* OSM数据中，老年服务站通常标记在 `social_facility` 里，下载后你可能需要查看 `social_facility:for` 这一列来筛选 `senior`（老年人）。
  * **幼托:** `kindergarten` (幼儿园) 和 `childcare` (托儿所) 分得比较细，都包含在内了。
  * **医疗:** 包含了 `clinic` (诊所), `doctors` (医生), `dentist` (牙医)。

#### 2\. 便民商业服务

  * **超市:** `shop='supermarket'`。
  * **快递:** `amenity` 中的 `post_office`, `post_box` (邮筒), `parcel_locker` (快递柜)。
  * **便民网点:** 在 `shop` 标签中加入了理发 (`hairdresser`)、维修 (`repair`) 以及面包房、便利店等。

#### 3\. 市政配套基础设施

  * **水/电/气/热 (难点):** OpenStreetMap 是基于视觉观测的地图，**地下的**水管、燃气管、供热管通常没有数据。我添加了 `power='substation'` (变电站) 作为参考，但这一项数据通常是不完整的。
  * **停车/充电:** `amenity` 中的 `parking` 和 `charging_station`。
  * **慢行系统:** 实际上由 `graph` (街道网络) 覆盖，但我额外添加了 `bicycle_parking` 和 `bicycle_rental` 作为设施点。
  * **无障碍设施:** 通常作为道路的属性（如 `tactile_paving=yes`），很难作为单独的“设施点”下载，因此未单独列出，建议通过分析路网属性获取。

#### 4\. 公共活动空间

  * **场地与绿地:** 使用了 `leisure` (休闲) 和 `landuse` (土地利用) 标签。
  * **面积筛选:** 你提到的 *“面积不小于150平方米”* 无法在下载时直接筛选。你需要下载完成后（代码已完成此步），将数据投影到投影坐标系（如 UTM），计算 `geometry.area`，然后过滤掉面积小于 150 的多边形。

### 下一步建议

数据下载下来只是第一步，慕尼黑的数据量很大，包含了几万个设施点。

**Would you like me to create a follow-up script to analyze this data?**
For example, I can help you **filter specific facilities** (e.g., "Find all parks \> 150sqm") or **visualize** the coverage of supermarkets (e.g., "Show 15-minute walking circles around supermarkets").



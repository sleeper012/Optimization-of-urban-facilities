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


"""
正在下载 Munich, Germany 的数据，请耐心等待...
街道网络下载完成。
正在下载公共设施数据 (这可能需要几分钟)...
数据保存成功！
共下载了 47547 个设施点/区域。
==================================================

【AMENITY 类别】- 共 28070 个
--------------------------------------------------
amenity
waste_basket        6649
parking             5849
bicycle_parking     5055
restaurant          2082
recycling           1186
kindergarten         988
doctors              855
fast_food            822
cafe                 808
post_box             756
charging_station     717
toilets              435
dentist              318
pharmacy             316
social_facility      300
parcel_locker        222
community_centre     184
waste_disposal       182
post_office          122
childcare            100
clinic                78
bicycle_rental        35
townhall               4
fuel                   1
biergarten             1
public_building        1
animal_training        1
school                 1
smoking_area           1
driver_training        1

【SHOP 类别】- 共 2990 个
--------------------------------------------------
shop
hairdresser             952
bakery                  693
supermarket             587
car_repair              208
butcher                 172
convenience             155
greengrocer             121
electronics              47
kiosk                     9
pastry                    6
coffee                    5
deli                      3
confectionery             3
rental                    3
bicycle                   2
yes                       2
tea                       2
stationery                1
fabric                    1
chemist                   1
clothes                   1
lottery                   1
vacant                    1
newsagent;stationery      1
beauty                    1
bakery;pastry             1
bookmaker                 1
scuba_diving              1
alcohol                   1
hifi                      1
chocolate                 1
ice_cream                 1
video                     1
medical_supply            1
beverages                 1
massage                   1
collector                 1

【LEISURE 类别】- 共 7316 个
--------------------------------------------------
leisure
playground         2921
pitch              2299
park                951
garden              758
sports_centre       267
fitness_station     119
bowling_alley         1

【LANDUSE 类别】- 共 7918 个
--------------------------------------------------
landuse
grass                7835
recreation_ground      49
industrial             15
meadow                  4
forest                  3
brownfield              2
residential             2
garages                 2
farmland                2
commercial              2
landfill                1
construction            1

【OFFICE 类别】- 共 146 个
--------------------------------------------------
office
government     130
physician        7
association      5
company          2
ngo              1
foundation       1

【POWER 类别】- 共 1397 个
--------------------------------------------------
power
substation    1367
plant           26
22000            3
80333            1

==================================================
数据保存位置:
  - 路网: munich_street_network.graphml
  - 设施: munich_public_facilities.geojson
"""

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

---

## 📊 数据分析工作流程

### **阶段一: 数据探索与可视化**

#### 1. `AnalyzeFacility.py` - 数据预处理与分类
**功能:**
- 加载并清洗设施数据
- 按照15项社区服务需求进行分类
- 统计各类设施数量
- 为后续分析准备核心设施数据

**运行:**
```bash
python AnalyzeFacility.py
```

#### 2. `VisualizeFacility.py` - 设施空间分布可视化
**功能:**
- 可视化幼儿园、医疗、超市、公园四类核心设施的空间分布
- 使用OpenStreetMap底图
- 生成高分辨率分布图

**运行:**
```bash
python VisualizeFacility.py
```

**输出:** `munich_facilities_distribution.png`

---

### **阶段二: 可达性测度**

#### 3. `AccessibilityAnalysis.py` - 15分钟生活圈可达性分析
**功能:**
- 基于路网计算步行15分钟等时圈
- 以超市为例,计算服务覆盖范围
- 识别服务空白区域
- 计算整体覆盖率

**关键参数:**
- 步行速度: 4.5 km/h
- 时间限制: 15分钟
- 等效距离: 1125米

**运行:**
```bash
python AccessibilityAnalysis.py
```

**输出:** `walkability_15min_supermarkets.png`

---

### **阶段三: 优化模型** ⭐改进版

> **重要说明:** 原P-Median模型过于严苛(强制100%覆盖),现提供**三种改进方法**,更符合实际规划需求!

---

#### 4.1 `FacilityOptimization_MCLP.py` - 最大覆盖选址模型 ⭐⭐⭐⭐⭐
**核心思想:** 在有限预算下,最大化被覆盖的人口数量

**优势:**
- ✅ 不强制100%覆盖(更现实)
- ✅ 关注边际改善最大化
- ✅ 计算速度快,易求解
- ✅ 适合实际规划项目

**模型说明:**
- **目标函数:** 最大化被覆盖的需求点数量
- **约束条件:** 
  - 新建设施数 = P (预算约束)
  - 需求点只有在距离≤1125米时才算被覆盖

**运行:**
```bash
python FacilityOptimization_MCLP.py
```

**输出:** `facility_optimization_MCLP.png`

**适用场景:** 有限预算的政府项目、老旧小区分阶段改造

---

#### 4.2 `FacilityOptimization_Heatmap.py` - 热力图驱动选址 ⭐⭐⭐⭐
**核心思想:** 识别"服务缺口热力图",在热点区域优先布局

**优势:**
- ✅ 直观易懂,便于决策汇报
- ✅ 多层次可视化(需求、供给、缺口)
- ✅ 不需要复杂求解器
- ✅ 灵活可调整

**方法流程:**
1. 核密度估计 → 需求密度热力图
2. 核密度估计 → 供给密度热力图
3. 服务缺口 = 需求 - 供给
4. 迭代贪心选择缺口最大区域

**运行:**
```bash
python FacilityOptimization_Heatmap.py
```

**输出:** `facility_optimization_heatmap.png` (6子图组合)

**适用场景:** 需要可视化决策支持、空间规划项目

---

#### 4.3 `FacilityOptimization_Equity.py` - 公平性导向优化 ⭐⭐⭐⭐⭐
**核心思想:** 最小化"最差服务水平",关注弱势群体

**优势:**
- ✅ 符合"一老一小"政策导向
- ✅ 引入脆弱性权重(老旧小区优先)
- ✅ Minimax目标确保公平
- ✅ 社会价值高,适合发表

**模型说明:**
- **目标函数:** 0.6×最大服务距离 + 0.4×加权平均距离
- **权重设计:** 未覆盖区域、远离市中心 → 高权重
- **公平性指标:** 最大距离、标准差

**运行:**
```bash
python FacilityOptimization_Equity.py
```

**输出:** `facility_optimization_equity.png` (4子图:权重、优化前后、直方图)

**适用场景:** 老旧小区改造、社会公平导向的规划研究

---

#### 4.4 `FacilityOptimization.py` - P-Median模型(原方法,保留)
**核心思想:** 强制100%覆盖,最小化总距离

**局限性:**
- ❌ 过于严苛,不现实
- ❌ 求解困难
- ❌ 缺乏灵活性

**建议:** 仅作为理论基准对比,实际应用请使用上述三种改进方法

---

### **📊 方法对比与选择**

| 方法 | 适用场景 | 计算速度 | 可视化 | 推荐度 |
|------|---------|---------|--------|--------|
| **MCLP** | 实际项目,预算约束 | ⚡⚡⚡⚡ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **热力图** | 决策汇报,空间规划 | ⚡⚡⚡⚡⚡ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **公平性** | 学术研究,社会公平 | ⚡⚡⚡ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| P-Median | 理论对比 | ⚡⚡ | ⭐⭐ | ⭐⭐ |

**详细对比请查看:** `优化方法对比说明.md`

---

### **依赖包安装**

```bash
# 基础依赖(原有)
pip install osmnx geopandas matplotlib networkx contextily pulp

# 新增依赖(用于改进版优化模型)
pip install scikit-learn scipy
```

---

### **完整工作流示例**

> 以**超市**为例：

```bash
# Step 1: 下载数据(已完成)
python MapFacility.py
 
# step 2: 分析数据，统计各类设施数量,帮你了解数据结构
python AnalyzeFacility.py

"""
正在加载数据...
设施数据加载完成: 47547 个设施
路网数据加载完成: 164277 个节点, 433026 条边

==================================================
开始数据分析...

各类设施统计:
--------------------------------------------------
社区综合服务      :   188 个
幼儿园         :   988 个
托儿所         :   100 个
老年服务站       :   300 个
医疗服务        :  1251 个
药店          :   316 个
超市          :   587 个
邮件快递        :  1100 个
餐饮          :  3712 个
理发店         :   952 个
停车充电        :  6566 个
公共绿地        :  1709 个
运动场地        :  5606 个


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
==================================================
核心设施分类完成,可用于下一步可达性分析!

建议下一步:
1. 可视化设施分布 (使用 VisualizeFacility.py)
2. 计算15分钟步行圈覆盖范围
3. 识别服务不足区域
4. 运行设施选址优化模型
"""

# Step 3: 可视化分布，生成4类核心设施的空间分布图,直观看到设施在哪里
python VisualizeFacility.py

"""
可视化完成! 高清美图已保存为: munich_facilities_dark_mode.png

==================================================
可视化统计信息:
  - 幼儿园: 988 个点位
  - 医疗: 1251 个点位
  - 超市: 587 个点位
  - 公园: 1709 个点位
==================================================
"""

# Step 4: 可达性分析 计算15分钟步行圈,看看超市的覆盖情况 画图列举几个超市的周围分布情况
python AccessibilityAnalysis.py

"""
============================================================
15分钟生活圈可达性分析
============================================================

配置: 步行速度=4.5km/h, 时间限制=15分钟
等效距离: 1125米

正在加载数据...
设施数量: 47547
路网节点: 164277

超市数量: 587

正在匹配超市到路网节点...
成功匹配 587 个超市到路网

正在计算15分钟步行可达范围...

可视化完成! 保存为: walkability_15min_supermarkets.png

============================================================
可达性统计:
------------------------------------------------------------
总路网节点数: 164277
超市覆盖节点数: 151507
覆盖率: 92.23%

============================================================
下一步建议:
1. 识别服务空白区域(未覆盖的节点)
2. 运行设施选址优化模型,找出最佳新增位置
3. 对其他设施类型(幼儿园、医疗等)重复此分析
"""

# Step 5: 选址优化 (三选一,推荐方法2或3)

# 方法1: 最大覆盖模型(MCLP) - 实用性强------代码中针对为对超市的选址  可以设定有几个超市建址
python FacilityOptimization_MCLP.py

"""
======================================================================
最大覆盖选址优化模型 (MCLP)
======================================================================
配置:
  新增设施数: 5
  步行距离上限: 1125米 (15分钟)
  需求点采样: 1000
  候选位置数: 200

======================================================================
步骤1: 加载数据
======================================================================
✓ 设施数: 47547
✓ 路网节点数: 164277

======================================================================
步骤2: 提取现有设施
======================================================================
✓ 现有超市: 587 个
✓ 成功匹配到路网: 587 个

======================================================================
步骤3: 生成需求点
======================================================================
正在识别服务薄弱区域...
✓ 现有覆盖节点: 151507 (92.2%)
✓ 未覆盖节点: 12770 (7.8%)
✓ 需求点总数: 999

======================================================================
步骤4: 生成候选设施位置
======================================================================
正在对 12770 个未覆盖节点进行聚类...
✓ 候选位置: 200 个 (聚类生成)

======================================================================
步骤5: 计算距离矩阵
======================================================================
距离矩阵规模: 999 × 787
  进度: 0/999
  进度: 100/999
  进度: 200/999
  进度: 300/999
  进度: 400/999
  进度: 500/999
  进度: 600/999
  进度: 700/999
  进度: 800/999
  进度: 900/999
✓ 距离矩阵计算完成

现状分析:
  被现有设施覆盖: 341/999 (34.1%)

======================================================================
步骤6: 构建最大覆盖模型
======================================================================
✓ 模型构建完成
  决策变量: 200 个候选位置
  约束条件: 新建设施数 = 5

======================================================================
步骤7: 求解优化问题
======================================================================
正在求解(可能需要1-3分钟)...
Welcome to the CBC MILP Solver 
Version: 2.10.3
Build Date: Dec 15 2019
(default strategy 1)
At line 2 NAME          MODEL
At line 3 ROWS
At line 984 COLUMNS
At line 7591 RHS
At line 8571 BOUNDS
At line 9771 ENDATA
Problem MODEL has 979 rows, 1199 columns and 3209 elements
Coin0008I MODEL read with 0 errors
seconds was changed from 1e+100 to 300
Option for timeMode changed from cpu to elapsed
Continuous objective value is 514 - 0.01 seconds
Cgl0004I processed model has 522 rows, 680 columns (680 integer (668 of which binary)) and 2251 elements
Cutoff increment increased from 1e-05 to 0.9999
Cbc0038I Initial state - 0 integers unsatisfied sum - 0
Cbc0038I Solution found of -514
Cbc0038I Cleaned solution of -514
Cbc0038I Before mini branch and bound, 680 integers at bound fixed and 0 continuous
Cbc0038I Mini branch and bound did not improve solution (0.02 seconds)
Cbc0038I After 0.02 seconds - Feasibility pump exiting with objective of -514 - took 0.00 seconds
Cbc0012I Integer solution of -514 found by feasibility pump after 0 iterations and 0 nodes (0.02 seconds)
Cbc0001I Search completed - best objective -514, took 0 iterations and 0 nodes (0.02 seconds)
Cbc0035I Maximum depth 0, 0 variables fixed on reduced cost
Cuts at root node changed objective from -514 to -514
Probing was tried 0 times and created 0 cuts of which 0 were active after adding rounds of cuts (0.000 seconds)
Gomory was tried 0 times and created 0 cuts of which 0 were active after adding rounds of cuts (0.000 seconds)
Knapsack was tried 0 times and created 0 cuts of which 0 were active after adding rounds of cuts (0.000 seconds)
Clique was tried 0 times and created 0 cuts of which 0 were active after adding rounds of cuts (0.000 seconds)
MixedIntegerRounding2 was tried 0 times and created 0 cuts of which 0 were active after adding rounds of cuts (0.000 seconds)
FlowCover was tried 0 times and created 0 cuts of which 0 were active after adding rounds of cuts (0.000 seconds)
TwoMirCuts was tried 0 times and created 0 cuts of which 0 were active after adding rounds of cuts (0.000 seconds)
ZeroHalf was tried 0 times and created 0 cuts of which 0 were active after adding rounds of cuts (0.000 seconds)

Result - Optimal solution found

Objective value:                514.00000000
Enumerated nodes:               0
Total iterations:               0
Time (CPU seconds):             0.03
Time (Wallclock seconds):       0.03

Option for printingOptions changed from normal to all
Total time (CPU seconds):       0.04   (Wallclock seconds):       0.04


覆盖效果:
  优化前: 341/999 个需求点被覆盖 (34.1%)
  优化后: 514/999 个需求点被覆盖 (51.5%)
  改善量: +173 个需求点 (+17.3%)

建议新增设施位置:
  位置 1: 节点ID=222018639, 坐标=(48.210389, 11.620877)
  位置 2: 节点ID=310653583, 坐标=(48.218230, 11.482071)
  位置 1: 节点ID=222018639, 坐标=(48.210389, 11.620877)
  位置 2: 节点ID=310653583, 坐标=(48.218230, 11.482071)
  位置 2: 节点ID=310653583, 坐标=(48.218230, 11.482071)
  位置 3: 节点ID=2010460896, 坐标=(48.212426, 11.586407)
  位置 3: 节点ID=2010460896, 坐标=(48.212426, 11.586407)
  位置 4: 节点ID=11792955485, 坐标=(48.100477, 11.696966)
  位置 5: 节点ID=8232356993, 坐标=(48.082089, 11.551053)

======================================================================
步骤8: 生成可视化
======================================================================
✓ 可视化完成: facility_optimization_MCLP.png

======================================================================
优化完成!
======================================================================
"""

# 方法2: 热力图驱动(推荐!) - 可视化最强 已优化性能!
python FacilityOptimization_Heatmap.py

"""

(paimai) PS D:\AppData\pycharm\code\city\Optimization-of-urban-facilities> python FacilityOptimization_Heatmap.py
======================================================================
基于热力图的需求驱动选址优化
======================================================================

配置:
  新增设施数: 5
  步行距离上限: 1125米
  热力图分辨率: 100×100

======================================================================
步骤1: 加载数据
======================================================================
✓ 现有超市: 587 个

======================================================================
步骤2: 匹配设施到路网
======================================================================
✓ 成功匹配: 587 个

======================================================================
步骤3: 生成需求密度热力图
======================================================================
需求点数量(路网节点): 164277
正在计算需求密度(核密度估计)...
✓ 需求密度计算完成

======================================================================
步骤4: 生成供给密度热力图
======================================================================
正在计算供给密度...
✓ 供给密度计算完成

======================================================================
步骤5: 计算服务缺口热力图
======================================================================
✓ 服务缺口计算完成
  最大缺口: 0.819
  最小缺口: 0.000

======================================================================
步骤6: 选择最佳新增位置
======================================================================
候选位置池: 5000 个路网节点

  选择第 1 个设施...
    位置 1: 节点ID=156686821, 坐标=(48.170111, 11.522312)
    服务缺口改善: 68.353

  选择第 2 个设施...
    位置 2: 节点ID=36246326, 坐标=(48.180878, 11.573147)
    服务缺口改善: 68.147

  选择第 3 个设施...
    位置 3: 节点ID=715492867, 坐标=(48.124813, 11.609923)
    服务缺口改善: 62.746

  选择第 4 个设施...
    位置 4: 节点ID=442310367, 坐标=(48.098079, 11.639360)
    服务缺口改善: 61.297

  选择第 5 个设施...
    位置 5: 节点ID=248792120, 坐标=(48.144717, 11.598059)
    服务缺口改善: 52.569

✓ 完成选址,共选择 5 个位置

======================================================================
步骤7: 评估覆盖效果(快速批量计算)
======================================================================
正在批量计算覆盖率...
  计算现有设施覆盖范围...
    进度: 0/587
    进度: 100/587
    进度: 200/587
    进度: 300/587
    进度: 400/587
    进度: 500/587
  计算新增设施覆盖范围...
    新设施 1/5
    新设施 2/5
    新设施 3/5
    新设施 4/5
    新设施 5/5

覆盖率评估(基于 1000 个采样点):
  优化前: 919/1000 (91.9%)
  优化后: 919/1000 (91.9%)
  改善量: +0 (+0.0%)

======================================================================
步骤8: 生成可视化
======================================================================
✓ 可视化完成: facility_optimization_heatmap.png

======================================================================
优化完成!
======================================================================
"""
# 方法3: 公平性优化(推荐!) - 学术价值高
python FacilityOptimization_Equity.py

```

### **结果展示**
 你会得到什么结果?
✅ **6-8张高清分析图**(根据选择的方法)
✅ **详细的统计数据**(覆盖率、公平性指标、距离分布)
✅ **具体的选址建议**(经纬度坐标)
✅ **优化效果对比**(优化前后可视化)
✅ **热力图分析**(需求密度、供给密度、服务缺口)
✅ **公平性分析**(最大距离、标准差、直方图)

**详细方法对比请查看:** `优化方法对比说明.md`

---

### **研究成果输出建议**

### **论文写作建议**

#### 📝 推荐方案1: 公平性导向研究(社会价值高)

**章节结构:**
- 第1章: 引言(老旧小区改造背景 + "一老一小"政策)
- 第2章: 文献综述(设施选址模型 + 空间公平理论)
- 第3章: 研究方法
  - 3.1 数据来源(OSM + 路网)
  - 3.2 脆弱性指标构建(服务缺口、距离惩罚)
  - 3.3 公平性优化模型(Minimax + 加权覆盖)
- 第4章: 慕尼黑实证分析
  - 4.1 设施现状(可达性测度)
  - 4.2 优化结果(覆盖率、公平性指标)
  - 4.3 敏感性分析(不同权重组合)
- 第5章: 讨论与政策建议
  - 空间正义、弱势群体关怀

**关键词:** 老旧小区、15分钟生活圈、空间公平、设施选址、公共服务可达性

---

#### 📝 推荐方案2: 对比研究(工作量充足)

**章节结构:**
- 第3章: 研究方法
  - 3.1 最大覆盖模型(效率导向)
  - 3.2 公平性优化模型(公平导向)
  - 3.3 热力图驱动方法(可视化决策)
- 第4章: 实证分析
  - 4.1 三种方法的优化结果对比
  - 4.2 效率与公平的权衡(Trade-off分析)
  - 4.3 不同政策目标下的模型选择建议

**优势:** 有对比有深度,容易发顶刊

---

#### 🚀 可扩展方向

1. **多设施类型对比**
   - 超市、幼儿园、医疗、养老设施的差异化选址策略
   
2. **真实数据融合**
   - 引入人口普查数据(老年人口比例)
   - 加入收入水平、住房年代等脆弱性指标
   
3. **多目标优化**
   - 覆盖率 + 公平性 + 成本最小化
   - 帕累托前沿分析
   
4. **动态分析**
   - 分阶段选址策略(第1年建3个、第2年建5个)
   - 考虑人口变化趋势
   
5. **国际对比**
   - 慕尼黑 vs 中国城市(如上海、北京老旧小区)
   - 政策环境差异分析



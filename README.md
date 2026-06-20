# GeoDetector · 地理探测器

[English](#english) | [中文](#中文)

A Python toolkit for detecting **spatial stratified heterogeneity** based on the
Geographical Detector method proposed by Wang et al. (2010).

基于王劲峰等 (2010) 提出的地理探测器方法，用于探测**空间分层异质性**的 Python 工具包。

---

## English

### Installation

```bash
pip install geodetector
```

### Quick Start

```python
from geodetector import GeoDetector
from geodetector.dataset import load_disease

df = load_disease()
gd = GeoDetector(factors=["type", "region", "level"], target="incidence")
gd.fit(df)

# Factor detector
print(gd.q_values_)
#   variable  q_value   p_value  significant
# 0     type   0.3857  0.372145        False
# 1   region   0.6378  0.000129         True
# 2    level   0.6067  0.043382         True

# Summary and plots
print(gd.summary())
gd.plot()               # horizontal bar chart of q-values
gd.plot_interaction()   # interaction q-value heatmap
gd.plot_dashboard()     # all four detectors in one figure
```

### Mathematical Foundation

The **q-statistic** measures the explanatory power of a stratification X on Y:

$$q = 1 - \frac{SSW}{SST} = 1 - \frac{\sum_{h=1}^L N_h \cdot \mathrm{Var}(Y_h)}{\sum_{i=1}^N (Y_i - \bar{Y})^2}$$

where $L$ is the number of strata and $\mathrm{Var}(Y_h)$ is the within-stratum variance.

The q-statistic is algebraically identical to the ANOVA effect size η² and the R² of a stratum-mean predictor: **q ≡ R² ≡ η²**.

#### Significance Test (Non-central F-test)

$$F = \frac{N-L}{L-1} \cdot \frac{q}{1-q} \sim F(L-1,\ N-L,\ \lambda)$$

The non-centrality parameter λ follows the R GD / gdverse formula.

### Four Core Detectors

| Detector | Question | Output |
|----------|----------|--------|
| **Factor** | Does stratification X explain Y? | q-value, p-value |
| **Interaction** | Do X₁ and X₂ have synergistic effects? | Interaction type (0–4) |
| **Risk** | Are Y means significantly different between strata? | t-test results |
| **Ecological** | Do X₁ and X₂ differ significantly in explanatory power? | F-test results |

#### Interaction Types

| Type | Condition | Label |
|------|-----------|-------|
| 0 | q(X₁∩X₂) < min(q₁, q₂) | Weaken, nonlinear |
| 1 | min ≤ q(X₁∩X₂) ≤ max | Weaken, uni-variable |
| 2 | max < q(X₁∩X₂) < q₁+q₂ | Enhance, bi-variable |
| 3 | q(X₁∩X₂) ≈ q₁+q₂ | Independent |
| 4 | q(X₁∩X₂) > q₁+q₂ | Enhance, nonlinear |

### Core API

```python
from geodetector import (
    GeoDetector,
    FactorDetector,
    InteractionDetector,
    RiskDetector,
    EcologicalDetector,
    discretize,
    Discretizer,
    OptimalDiscretizer,
)

# Discretize continuous variables
strata = discretize(data, method="quantile", n_strata=5)

# Individual detectors
fd = FactorDetector(discretize="quantile", n_strata=5)
fd.fit(X[["factor1"]], y)
print(fd.q_value_, fd.p_value_)

id_ = InteractionDetector()
id_.fit(X, y)
print(id_.interaction_q_)
print(id_.interaction_type_)
```

### Advanced Extensions

```python
from geodetector.extensions import (
    OPGDDetector,      # Optimal Parameter GD
    GOZHDetector,      # Decision-tree-based GD
    RGDDetector,       # Robust GD
    shapley_decompose, # Shapley decomposition (LESH)
    lesh,              # Full LESH analysis
)
```

#### OPGD — Optimal Parameter Geographical Detector

Searches over discretization methods and stratum counts to maximize q-value.

```python
opgd = OPGDDetector(methods=["sd", "equal", "geometric", "quantile", "natural"],
                    k_range=(3, 8))
opgd.fit(data, factors=["xa", "xb", "xc"], target="y")
print(opgd.opt_params_)  # optimal method & k per variable
print(opgd.q_values_)
```

#### GOZH — Geographically Optimal Zones-based Heterogeneity

Uses decision trees to automatically find optimal strata. Interaction detection uses joint decision trees per factor pair (matching gdverse).

```python
gozh = GOZHDetector(max_depth=3, min_samples_leaf=5)
gozh.fit(data, factors=["xa", "xb"], target="y")
print(gozh.n_zones_)          # number of zones per factor
print(gozh.interaction_pairs_) # joint-tree interaction results
```

#### LESH — Locally Explained Stratified Heterogeneity

Shapley-value decomposition of q-statistics. Supports both traditional discretization (`method="quantile"`) and GOZH-style decision-tree discretization (`method="gozh"`).

```python
# Traditional discretization
result = shapley_decompose(data, ["xa", "xb", "xc"], "y", method="quantile")
print(result[["variable", "shapley_value", "shapley_pct"]])

# GOZH-style (matching gdverse LESH)
result = lesh(data, ["xa", "xb", "xc"], "y", method="gozh", max_depth=3)
print(result["shapley"])
print(result["interaction"])  # SPD-attributed interaction
```

#### RGD — Robust Geographical Detector

Variance-based change-point detection for discretization, robust to outliers. Supports multi-k search with LOESS elbow detection for optimal stratum count selection.

```python
rgd = RGDDetector(discnum=range(3, 8), strategy=2, increase_rate=0.05)
rgd.fit(data, factors=["xa", "xb"], target="y")
print(rgd.opt_discnum_)   # optimal discnum per factor
print(rgd.all_q_values_)  # q-values across all discnums
```

### Comparison with R Implementations

This package is aligned with two R reference implementations:

| Feature | GD-main (`GD`) | gdverse | This package |
|---------|---------------|---------|--------------|
| Factor detector q | ✓ same formula | ✓ same formula | ✓ matches both |
| Non-central F λ | ✓ | ✓ | ✓ matches both |
| Interaction q12 | per-pair `gd()` | per-pair `gd()` | per-pair consistent subset |
| Ecological F | `q₂/q₁` (α≈0.2) | `(1−q₁)/(1−q₂)` 1-tailed | `(1−q₁)/(1−q₂)` 2-tailed |
| OPGD defaults | user-specified | 5 methods | 5 methods (matches gdverse) |
| GOZH interaction | — | joint tree per pair | joint tree per pair |
| LESH discretization | — | `rpart_disc` (GOZH) | supports both |
| RGD discnum search | — | 3:8 + LOESS | 3:8 + LOESS |

**Ecological detector note**: This package uses a two-tailed F-test with `F = (1−q₁)/(1−q₂)`. GD-main uses `F = q₂/q₁` with `qf(0.9, n−1, n−1)`. gdverse uses one-tailed `pf(F, n−1, n−1, lower.tail=FALSE)`. The two-tailed test is more conservative and symmetric (order-independent).

### References

- Wang JF, Li XH, Christakos G, Liao YL, Zhang T, Gu X & Zheng XY. 2010. Geographical detectors-based health risk assessment. *IJGIS* 24(1): 107-127.
- Wang JF, Zhang TL, Fu BJ. 2016. A measure of spatial stratified heterogeneity. *Ecological Indicators* 67: 250-256.
- Song Y, Wang J, Ge Y, Xu C. 2020. An optimal parameters-based geographical detector model. *GIScience & Remote Sensing* 57(5): 593-610.
- Luo P, Song Y, et al. 2022. GOZH model. *ISPRS Journal of Photogrammetry and Remote Sensing* 185: 111-128.
- Li Y, Luo P, Song Y, et al. 2023. LESH model. *International Journal of Digital Earth* 16(2): 4533-4552.
- Zhang Z, Song Y, Wu P. 2022. Robust geographical detector. *IJAEOG* 109: 102782.
- Lv W, Lei Y, et al. 2025. gdverse: An R Package for Spatial Stratified Heterogeneity Family. *Transactions in GIS* 29.

### License

MIT

---

## 中文

### 安装

```bash
pip install geodetector
```

### 快速开始

```python
from geodetector import GeoDetector
from geodetector.dataset import load_disease

df = load_disease()
gd = GeoDetector(factors=["type", "region", "level"], target="incidence")
gd.fit(df)

# 因子探测器
print(gd.q_values_)
#   variable  q_value   p_value  significant
# 0     type   0.3857  0.372145        False
# 1   region   0.6378  0.000129         True
# 2    level   0.6067  0.043382         True

# 摘要与可视化
print(gd.summary())
gd.plot()               # q值水平柱状图
gd.plot_interaction()   # 交互作用热力图
gd.plot_dashboard()     # 四合一仪表盘
```

### 数学基础

**q 统计量**衡量分层 X 对结果 Y 的解释力：

$$q = 1 - \frac{SSW}{SST} = 1 - \frac{\sum_{h=1}^L N_h \cdot \mathrm{Var}(Y_h)}{\sum_{i=1}^N (Y_i - \bar{Y})^2}$$

其中 $L$ 为分层数，$\mathrm{Var}(Y_h)$ 为层内方差。

q 统计量与 ANOVA 效应量 η² 及层均值预测器的 R² 代数等价：**q ≡ R² ≡ η²**。

#### 显著性检验（非中心 F 检验）

$$F = \frac{N-L}{L-1} \cdot \frac{q}{1-q} \sim F(L-1,\ N-L,\ \lambda)$$

非中心参数 λ 采用 R GD / gdverse 公式。

### 四种核心探测器

| 探测器 | 回答的问题 | 输出 |
|--------|-----------|------|
| **因子** | 分层 X 能否解释 Y？ | q 值、p 值 |
| **交互** | X₁ 与 X₂ 是否有协同/拮抗效应？ | 交互类型 (0–4) |
| **风险** | 不同分层之间 Y 的均值是否有显著差异？ | t 检验结果 |
| **生态** | X₁ 与 X₂ 的解释力是否有显著差异？ | F 检验结果 |

#### 交互作用类型

| 类型 | 条件 | 含义 |
|------|------|------|
| 0 | q(X₁∩X₂) < min(q₁, q₂) | 非线性减弱 |
| 1 | min ≤ q(X₁∩X₂) ≤ max | 单因子非线性减弱 |
| 2 | max < q(X₁∩X₂) < q₁+q₂ | 双因子增强 |
| 3 | q(X₁∩X₂) ≈ q₁+q₂ | 独立 |
| 4 | q(X₁∩X₂) > q₁+q₂ | 非线性增强 |

### 核心 API

```python
from geodetector import (
    GeoDetector,          # 主控类
    FactorDetector,       # 因子探测器
    InteractionDetector,  # 交互探测器
    RiskDetector,         # 风险探测器
    EcologicalDetector,   # 生态探测器
    discretize,           # 离散化函数
    Discretizer,          # sklearn 兼容转换器
    OptimalDiscretizer,   # 最优离散化
)

# 连续变量离散化
strata = discretize(data, method="quantile", n_strata=5)

# 单独使用探测器
fd = FactorDetector(discretize="quantile", n_strata=5)
fd.fit(X[["factor1"]], y)
print(fd.q_value_, fd.p_value_)
```

### 高级扩展

```python
from geodetector.extensions import (
    OPGDDetector,      # 最优参数地理探测器
    GOZHDetector,      # 决策树最优分区
    RGDDetector,       # 鲁棒地理探测器
    shapley_decompose, # Shapley 分解 (LESH)
    lesh,              # 完整 LESH 分析
)
```

#### OPGD — 最优参数地理探测器

遍历离散化方法和分层数，选择使 q 值最大的组合。

```python
opgd = OPGDDetector(
    methods=["sd", "equal", "geometric", "quantile", "natural"],
    k_range=(3, 8)
)
opgd.fit(data, factors=["xa", "xb", "xc"], target="y")
print(opgd.opt_params_)  # 每个变量的最优方法 & k
```

#### GOZH — 决策树最优分区

使用决策树回归器自动寻找最优分层。交互检测使用联合决策树（与 gdverse 一致）。

```python
gozh = GOZHDetector(max_depth=3, min_samples_leaf=5)
gozh.fit(data, factors=["xa", "xb"], target="y")
print(gozh.n_zones_)           # 每个因子的分区数
print(gozh.interaction_pairs_)  # 联合决策树交互结果
```

#### LESH — 局部解释的分层异质性

基于 Shapley 值的 q 统计量贡献分解。支持传统离散化 (`method="quantile"`) 和 GOZH 决策树离散化 (`method="gozh"`)。

```python
# 传统离散化
result = shapley_decompose(data, ["xa", "xb", "xc"], "y", method="quantile")
print(result[["variable", "shapley_value", "shapley_pct"]])

# GOZH 模式 (匹配 gdverse LESH)
result = lesh(data, ["xa", "xb", "xc"], "y", method="gozh", max_depth=3)
print(result["shapley"])       # Shapley 分解结果
print(result["interaction"])   # SPD 归因的交互作用
```

#### RGD — 鲁棒地理探测器

基于方差的变点检测离散化方法，对异常值鲁棒。支持多 k 搜索和 LOESS 曲率检测自动选择最优分层数。

```python
rgd = RGDDetector(discnum=range(3, 8), strategy=2, increase_rate=0.05)
rgd.fit(data, factors=["xa", "xb"], target="y")
print(rgd.opt_discnum_)   # 每个因子的最优分层数
print(rgd.all_q_values_)  # 所有分层数下的 q 值
```

### 与 R 实现的对比

本工具包与两个 R 参考实现对齐：

| 功能 | GD-main (`GD`) | gdverse | 本工具包 |
|------|---------------|---------|----------|
| 因子探测器 q | ✓ 相同公式 | ✓ 相同公式 | ✓ 匹配两者 |
| 非中心 F λ | ✓ | ✓ | ✓ 匹配两者 |
| 交互 q12 | 每对调用 `gd()` | 每对调用 `gd()` | 每对使用一致子集 |
| 生态检测器 F | `q₂/q₁` (α≈0.2) | `(1−q₁)/(1−q₂)` 单侧 | `(1−q₁)/(1−q₂)` 双侧 |
| OPGD 默认方法 | 用户指定 | 5 种方法 | 5 种方法 (匹配 gdverse) |
| GOZH 交互 | — | 每对联合决策树 | 每对联合决策树 |
| LESH 离散化 | — | `rpart_disc` (GOZH) | 支持两种方式 |
| RGD 分层数搜索 | — | 3:8 + LOESS | 3:8 + LOESS |

**生态检测器说明**：本工具包使用双侧 F 检验，`F = (1−q₁)/(1−q₂)`。GD-main 使用 `F = q₂/q₁` 配合 `qf(0.9, n−1, n−1)` 临界值。gdverse 使用单侧检验 `pf(F, n−1, n−1, lower.tail=FALSE)`。双侧检验更保守且对称（与因子顺序无关）。

### 参考文献

- 王劲峰, 李新虎, Christakos G, 等. 2010. 地理探测器-based 健康风险评估. *IJGIS* 24(1): 107-127.
- Wang JF, Zhang TL, Fu BJ. 2016. A measure of spatial stratified heterogeneity. *Ecological Indicators* 67: 250-256.
- Song Y, Wang J, Ge Y, Xu C. 2020. 最优参数地理探测器模型. *GIScience & Remote Sensing* 57(5): 593-610.
- Luo P, Song Y, et al. 2022. GOZH 模型. *ISPRS JPRS* 185: 111-128.
- Li Y, Luo P, Song Y, et al. 2023. LESH 模型. *Int. J. Digital Earth* 16(2): 4533-4552.
- Zhang Z, Song Y, Wu P. 2022. 鲁棒地理探测器. *IJAEOG* 109: 102782.
- Lv W, Lei Y, et al. 2025. gdverse: 空间分层异质性家族的 R 包. *Transactions in GIS* 29.

### 许可证

MIT

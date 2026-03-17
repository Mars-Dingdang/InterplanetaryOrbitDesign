这份Project Proposal（项目计划书）的核心是**行星际轨道设计**，特别是针对前往水星的高难度任务。以下为你概括该项目需要解决的问题、用到的算法与知识点，并按步骤带你开始解决文档中的具体问题。

### 一、 项目需要解决的问题概括

**核心目标：** 为2026年至2040年间前往水星的探测任务，设计一条满足特定工程限制（运载火箭逃逸速度限制、航天器变轨燃料限制、最长任务时间）的行星际轨道，并找到可行的发射窗口。

**具体拆解：**
1.  **理解基础轨道力学：** 计算和模拟最基础的地球-火星霍曼转移（Hohmann Transfer）轨道，理解燃料消耗（$\Delta v$）与发射/到达时间的关系。
2.  **揭示水星探测的痛点：** 计算从地球直接霍曼转移到水星的 $\Delta v$。由于水星深陷太阳重力井，航天器直接抵达时相对水星的速度极高，揭示“直接飞往水星燃料根本不够”的残酷现实。
3.  **设计引力弹弓（Gravity Assist）序列：** 为了解决燃料不足的问题，需要设计利用金星、地球或水星自身的引力弹弓效应来减速的飞行序列。
4.  **最终项目综合寻优：** 在真实的星历表（2026-2040）中，利用算法找出能够结合引力弹弓效应、且满足火箭运力（$<3.5 \text{ km/s}$ 逃逸速度）和探测器燃料（$<1.5 \text{ km/s}$ 变轨能力）的发射窗口期。

---

### 二、 用到的算法与知识储备

#### 1. 物理与轨道力学知识 (Astrodynamics)
*   **霍曼转移轨道 (Hohmann Transfer)**：用于计算共面圆轨道间最省燃料的转移路径。涉及半长轴 $a$、开普勒第三定律（计算转移时间 $T$）。
*   **活力公式 (Vis-viva Equation)**：计算航天器在轨道任意一点的瞬时速度。
*   **引力弹弓/飞掠 (Gravity Assists / Flyby)**：在行星参考系中航天器速度大小不变（双曲线轨道），但在太阳参考系中借用行星的动量实现加速或减速。
*   **猪排图 (Pork-chop Plot)**：等高线图，横轴和纵轴通常是到达时间和发射时间，等高线表示所需的特征能量 ($C_3$) 或 $\Delta v$，用于寻找最佳发射窗口。
*   **星历表 (Ephemeris)**：特定时期内太阳系行星真实位置的数据集。

#### 2. 数学与计算机算法 (Algorithms & Tools)
*   **兰伯特问题求解器 (Lambert's Problem Solver)**：已知两个位置向量（出发点和到达点）以及飞行时间，求解连接这两点的轨道（计算 $\Delta v$ 的核心算法）。
*   **圆锥曲线拼接法 (Patched Conics Method)**：将复杂的星际飞行近似分解为多个单体引力（日心或行星中心）轨道的拼接。
*   **数值积分与优化算法**：遗传算法 (Genetic Algorithms) 或 粒子群算法 (Particle Swarm Optimization)，用于在广阔的时间域里寻找最优的引力弹弓组合。
*   **开源库**：强烈推荐使用 Python 的 **`poliastro`** 库（专门用于交互式轨道力学计算）或 NASA 的 GMAT 软件。

---

### 三、 着手解决问题 (Step-by-Step 求解)

我将带你逐一解决 PDF 中的 Question 2.1、3.1、3.2，并为你铺垫最终项目。

#### **Question 2.1: 绘制地球-火星的猪排图 (Pork-chop plot)**
你可以使用 Python 的 `poliastro` 库直接计算兰伯特问题并绘制。以下是起步代码示例：

```python
import numpy as np
from astropy import time
from poliastro.plotting.porkchop import porkchop
from poliastro.bodies import Earth, Mars
from poliastro.util import time_range

# 设定发射和到达的时间窗口（例如2026年）
launch_span = time_range("2026-01-01", end="2026-12-31", periods=100)
arrival_span = time_range("2026-08-01", end="2027-12-31", periods=100)

# 绘制猪排图计算 dv (利用 poliastro 内部的 Lambert 求解器)
# dv_dpt 是出发所需 delta-v，dv_arr 是到达所需 delta-v
dv_dpt, dv_arr, c3dpt, c3arr, tof = porkchop(Earth, Mars, launch_span, arrival_span)
```
*这会生成一张等高线图，蓝色的“盆地”中心就是最优发射窗口。*

#### **Question 3.1: 计算直接飞往水星的代价 (Cost of going inward)**
我们来手动计算这道题的数学论证。
已知：$r_E = 1.496 \times 10^8 \text{ km}$, $r_M = 5.791 \times 10^7 \text{ km}$, $\mu_S = 1.327 \times 10^{11} \text{ km}^3/\text{s}^2$。

1. **计算地球和水星的公转速度 (假设为圆轨道):**
   * 地球：$v_E = \sqrt{\mu_S / r_E} \approx 29.78 \text{ km/s}$
   * 水星：$v_M = \sqrt{\mu_S / r_M} \approx 47.87 \text{ km/s}$
2. **计算霍曼转移轨道参数:**
   * 半长轴 $a = (r_E + r_M) / 2 = 1.03755 \times 10^8 \text{ km}$
3. **计算出发阶段需要的减速 (地球处):**
   * $\Delta v_1 = v_E \left| \sqrt{\frac{2r_M}{r_E + r_M}} - 1 \right| = 29.78 \times \left| 0.747 - 1 \right| \approx \mathbf{7.53 \text{ km/s}}$
4. **计算到达水星时的速度及制动所需的 $\Delta v$ (水星处):**
   * 航天器到达水星轨道（近日点）时的速度为：$v_{arr} = \sqrt{\mu_S \left(\frac{2}{r_M} - \frac{1}{a}\right)} \approx 57.48 \text{ km/s}$
   * 相对水星的到达速度：$\Delta v_2 = v_{arr} - v_M = 57.48 - 47.87 = \mathbf{9.61 \text{ km/s}}$
   * **总 $\Delta v$** = $7.53 + 9.61 = \mathbf{17.14 \text{ km/s}}$

**结论与启示：** 相对水星的到达速度高达 $9.61 \text{ km/s}$（约为水星轨道速度的 20%）。根据齐奥尔科夫斯基火箭方程，要让探测器减速 $9.61 \text{ km/s}$ 以进入水星轨道，需要消耗极其海量的燃料（甚至超过航天器总质量的90%），这在工程上是**不可能实现**的。这就是为什么必须利用引力弹弓！

#### **Question 3.2: 设计引力弹弓序列 (Design a flyby sequence)**
*   **序列建议 (BepiColombo同款)**：Earth $\rightarrow$ Venus $\rightarrow$ Venus $\rightarrow$ Mercury $\times$ 1~6 $\rightarrow$ Orbit。
    *   *原理*：地球出发，先在金星进行1-2次飞掠大幅度降低近日点；然后再飞掠水星数次（每次微调速度），渐渐把相对速度降下来，最后再消耗少量燃料进入轨道。
*   **猪排图特征 (Qualitative Sketch)**：
    *   常规的霍曼转移猪排图是巨大的、平缓的“同心圆盆地”。
    *   **多重飞掠的猪排图**会是非常破碎的、极其狭窄的“孤岛”或“细线”。因为你必须同时满足地球在特定位置、X个月后金星刚好在特定位置、再过Y个月水星也刚好在特定位置。这种多天体几何对齐极其苛刻，窗口极窄。

#### **Final Project 入手建议**
你现在可以准备进入最终项目了。建议步骤：
1. **明确约束：** 你的出发 $\Delta v$ 不能超过 $3.5 \text{ km/s}$（这是Ariane-5火箭给你的初始推力），而到达时的轨道插入 $\Delta v$ 不能超过 $1.5 \text{ km/s}$（这是航天器自带燃料的极限）。
2. **构建搜索空间：** 依靠上述发现，直接去是不可能的。你必须写一个 Python 脚本（利用 `poliastro`），搜索 **EVVM** (Earth-Venus-Venus-Mercury) 的飞行窗口。
3. **分段求解：** 
   * 第一段：扫描 2026-2035 之间地球到金星的兰伯特问题（要求出发 $C_3$ 满足火箭限制）。
   * 第二段：计算在金星的引力弹弓，看获取的 $\Delta v$ 能否把航天器送到下一次金星相遇或水星相遇。
   * 最终对比：与 BepiColombo 任务（2018年发射，历时7年）做对比，证明你找出的 2026-2040 期间的“某几个特定日期”是具有类似可行性的。

如果你需要，我可以帮你写一段利用 `poliastro` 计算基础 **地球-金星** 引力弹弓效应的代码，作为你最终项目的脚手架！# InterplanetaryOrbitDesign

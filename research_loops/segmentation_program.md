# 分割算法自动化迭代程序

## 目标

建立与正式系统解耦的分割研发线，用多个分割模型对照，逐步把弱标签训练推进为稳定的 mask 预测与后续 `mask->bbox` 可比评测。

## 主指标

- Primary: `L1 + L2` 私有样本上的 `scratch IoU / mIoU`
- Secondary: `L1 + L2` 私有样本上的 `mask->bbox AP@0.5`
- Reference only: `L3` 复杂样本上的同类指标，仅作退化观察与人工审核成本参考
- Analysis: 边界质量、连通性、骨架长度稳定性

## 详细量化指标

为兼顾模型选型、物理意义和后续标注更新, 分割分支统一增加三层指标:

### A. 结构学习指标

- `val_mIoU`：源域结构对照的主排序指标
- `scratch IoU`：细长划痕的首要指标
- `spot IoU` / `damage IoU`：辅助观察模型对块状和大面积区域的适应性
- `scratch Dice`：补足细长区域在小偏移下 IoU 过于苛刻的问题

### B. 标注更新指标

- `consensus ratio`：多模型一致像素比例，用于识别高置信区域
- `pairwise disagreement ratio`：模型间两两差异比例，用于挑选人工复核优先级样本
- `pixel accuracy vs weak mask`：弱标签一致性参考值
- `scratch_length_error` / `scratch_area_error`：预测与弱标签在长度/面积上的相对误差

### C. 物理量化指标

- `scratch_length_mm`：划痕骨架总长度，对应实际磨损轨迹长度
- `scratch_avg_width_mm`：划痕平均宽度，辅助判断损伤程度
- `scratch_area_mm2`：划痕总面积
- `spot_area_mm2`：斑点污染总面积
- `damage_area_mm2`：大面积损伤总面积
- `scratch_components`：划痕连通域数量，辅助判断断裂/过分碎片化问题

当前系统统一采用 `6.8 um/pixel = 0.0068 mm/pixel` 进行物理量换算, 研发记录默认优先展示 `mm / mm²`.

## 数据边界

- `CNAS` 测试子系统独立后，20 张留出图像不参与分割分支日常训练、模型选择和超参数调整
- `MSD` 只作为源域预训练 / 结构对照基线
- 当前私有数据集只作为目标域训练验证数据，不作为“测试集”表述
- 私有数据评估按复杂度分层执行：
  - `L1 + L2` 作为正式主指标
  - `L3` 作为复杂样本参考集，在有效分割模型完全成熟前不计入主排名
- 若需要最终与正式系统对照，只在研发后期追加一次 `mask->bbox` 形式的留出集核验

## 下一步实验基本框架

分割研发线采用“三层系统”组织下一步实验。该框架用于初步研究阶段的方法归类、变量控制和横评比较，不预设最终最优方法。

### 第一层：高召回缺陷前景层

目标是在尽量少依赖检测框的条件下，从 ROI 内稳定提取疑似缺陷区域。该层输出候选概率图、二值 mask、连通域和置信度，优先保证召回率。

候选方法:

- 背景校正、局部对比增强、亮度归一化
- Otsu / adaptive threshold / Sauvola / Niblack
- Frangi / Sato / Gabor 多尺度线状结构增强
- 形态学开闭运算、细小噪声过滤、连通域规则筛选
- 正常样本建模或工业异常检测热图

### 第二层：学习型分割层

目标是训练可泛化的缺陷 mask 预测模型，使输出结果比传统阈值更稳定、更适合物理量计算。

重要约束:

- 前期识别算法产生的弱标签可靠性较差，后续分割训练中应尽量少利用。
- 检测结果只作为候选生成、样本筛选、召回补偿或人工复核线索。
- 不把检测框内的弱 mask 直接视为真值。
- 不把检测框裁切结果作为分割模型唯一输入。
- 训练样本应优先来自人工核查 mask、传统方法高置信区域、多模型一致区域和高分歧样本复核结果。

### 第三层：结构解析与物理量评估层

目标是把 mask 转换为可解释的结构和物理量，而不是只输出像素类别。

核心输出:

- 连通域、骨架、端点、分叉点、交叉点
- 划痕图结构：节点为端点/交叉点/分叉点，边为划痕段
- 长度、平均宽度、面积、方向、曲率、亮度峰值、边缘梯度、散射强度
- 基础形态类别：`spot / scratch / damage`
- 区域级复合风险：`critical / crash`
- 与 WearScore 关联的中心区、微结构区、边缘区加权物理量

## Baseline 分类清单

研究初期按方法类型做横评，不直接追求单一 SOTA。所有 baseline 尽量使用相同数据划分、相同 ROI、相同像素尺寸换算和相同可视化协议。

| 类别 | 代表方法 | 主要比较问题 |
|---|---|---|
| 传统阈值 baseline | Otsu / adaptive threshold / Sauvola / Niblack | 无学习方法能否稳定提供高召回前景 |
| 线状增强 baseline | Frangi / Sato / Gabor / top-hat | 细长划痕召回、断裂率和噪声敏感性 |
| 形态规则 baseline | connected components / skeleton / graph rules | 后分割结构解析和类别解释的下限 |
| 轻量 CNN baseline | LightUNet / U-Net / U-Net++ | 少量弱标签下的基本可学习性 |
| 多尺度 CNN baseline | FPN / DeepLabV3+ / PSPNet | 多尺度上下文对 scratch / spot / damage 的收益 |
| 高分辨率 baseline | HRNet / HRNet-OCR | 细线边界、骨架连通性和小结构保真度 |
| Transformer baseline | SegFormer / Swin-UPerNet / Mask2Former | 全局上下文、复杂重叠区域和统一分割建模能力 |
| 工业异常分割 baseline | PatchCore / DRAEM / EfficientAD 类方法 | 少标注条件下异常热图是否能作为候选前景 |
| 交互式标注辅助 | SAM / SAM2 | 人工精修、弱标签升级和高分歧样本复核效率 |

## 弱标签使用原则

弱标签必须显式记录来源和可信度。每张 mask 至少标注:

- `source`: `manual / threshold / frangi / anomaly / model_consensus / detector_box_constrained`
- `confidence`: `high / medium / low`
- `reviewed`: 是否人工复核
- `roi_valid`: 是否在有效 ROI 内
- `detector_constrained`: 是否由检测框约束生成
- `used_for_training`: 是否进入训练集

默认训练优先级:

1. 人工复核 mask
2. 多方法一致的高置信前景
3. 传统方法高置信前景
4. 分割模型 teacher 高置信预测
5. 检测框约束弱标签，仅作为低优先级辅助或复核线索

## 后续实验阶段安排

后续实验按 P0-P5 推进。每一阶段完成后都要输出独立结果目录、指标汇总和 20-30 张代表性可视化样本。

### P0：统一实验协议

目标是先固定横评口径，避免后续模型、标签和后处理混在一起比较。

必须完成:

- 固定 `L1 / L2 / L3` 数据分层
- 固定 ROI 与 `6.8 um/pixel = 0.0068 mm/pixel` 换算
- 固定输出格式：`mask / probability map / components / skeleton / metrics json / overlay`
- 固定指标：`mIoU / Dice / scratch Dice / skeleton length error / area error / components count / break rate / review cost`
- 固定可视化模板：原图、候选图、mask、骨架、结构属性和错误案例并列显示

### P1：传统高召回前景 baseline

目标是建立无学习方法的前景提取下限，并筛选可进入弱标签生成的高置信候选。

优先比较:

- Otsu
- adaptive threshold
- Sauvola / Niblack
- Frangi
- Sato
- Gabor
- top-hat + morphology
- Frangi/Sato + adaptive threshold 组合

该阶段只评价前景召回、噪声、断裂和高光环误检，不承担最终类别解释。

### P2：小规模可靠 mask 样本

目标是建立后续模型横评的可信锚点集。

建议规模:

- `L1`: 20-30 张
- `L2`: 20-30 张
- `L3`: 10-15 张，仅用于观察复杂场景退化

每张样本都要保留弱标签元数据，至少记录 `source / confidence / reviewed / roi_valid / detector_constrained / used_for_training`。

### P3：分割模型横评

目标是在统一协议和可靠锚点集上比较主流 baseline。

第一批:

- LightUNet
- U-Net
- U-Net++
- FPN
- DeepLabV3+

第二批:

- HRNet / HRNet-OCR
- SegFormer-B2
- SegNeXt
- Swin-UPerNet 或 Mask2Former，作为高成本上限对照

训练策略从 `CE/BCE + Dice` 开始，再按单变量原则尝试 `Lovasz / Boundary loss / clDice`。检测框弱标签只作为低优先级辅助，不直接作为真值。

### P4：结构解析实验

目标是把可用 mask 转换为物理可解释结构。

重点比较:

- 连通域分析
- 骨架化
- 端点、分叉点、交叉点检测
- 划痕段图结构
- 断裂划痕连接
- 粘连/交叉区域拆解
- `spot / scratch / damage` 基础形态分类
- `critical / crash` 区域级风险判断

该阶段主指标从像素 `IoU` 转向长度误差、面积误差、宽度误差、交叉点稳定性和断裂率。

### P5：WearScore 关联验证

目标是评估分割和结构解析结果对磨损评分的实际贡献。

优先分析:

- `scratch_length_mm`
- `scratch_avg_width_mm`
- `scratch_area_mm2`
- `scatter_intensity`
- `edge_sharpness`
- `center / microstructure / edge` 区域权重
- `critical / crash` 区域密度

在完成稳定性分析前，不直接修改最终 A/B/C/D 阈值。先比较不同分割方法对 WearScore 的波动影响，再决定是否调整评分公式或接入 GUI 主线。

## 第一批模型

- `light_unet`
- `unetplusplus + resnet34`
- `deeplabv3plus + resnet34`
- `fpn + resnet34`

## 第二批模型与框架策略

- `FPN + resnet34`  
  - 继续作为 `SMP` 下的 CNN 主线
- `SegFormer-B2`  
  - 切换到 `HF Transformers` 实现
- `HRNet-OCR-W18`
  - 基于官方 `HRNet-OCR` 结构在本仓实现本地训练版本
  - 同时保留官方实验包生成器，用于交叉核验

当前阶段不再把 `mmseg/mmcv` 作为第二批实验主线依赖。

## 允许改动的变量

一次实验只改一类变量：

1. 模型结构
2. loss 组合
3. 采样与增强
4. 训练预算
5. 弱标签生成策略

## 固定预算原则

- 预训练阶段优先使用固定分钟预算
- 微调阶段优先使用固定 epoch 或固定分钟预算
- 所有实验必须写入独立输出目录

## 当前推荐实验逻辑

1. 在 `MSD` 上比较结构基线，得到源域排名
2. 选择前两名结构进入私有弱标签微调
3. 在私有训练验证数据上先比较 `L1 + L2` 主指标，再单独查看 `L3` 复杂样本参考结果
4. 仅在需要与正式检测主线对照时，补做一次留出 20 图像的 `mask->bbox` 核验

## 检测-分割协同原则

“标注增强实验”统一遵循以下原则：

- 分割主线负责像素级缺陷区域、连通性、长度和面积等形态学习
- `B2_nwd_only_phase3e` 等检测模型作为辅助候选分支存在
- 检测用于候选生成、样本筛选和部署时的召回补偿
- 检测不作为分割训练的硬前置，不替代分割主干

当前推荐流程为：

1. 检测先提出疑似缺陷候选
2. 分割模型在候选及其上下文上输出主 mask
3. 形态规则与物理量分析层做类别解释
4. 仅把高分歧、高风险样本送人工复核

明确禁止的默认做法：

- 不把检测框裁切结果当成分割训练的唯一输入
- 不把分割模型降级为“框内修补器”
- 不让检测候选决定分割模型可见的全部结构上下文

## 结果记录

每轮实验都要输出：

- 模型名
- 编码器
- 数据集
- 训练预算
- 最佳 loss
- `mIoU`
- 后续是否进入私有弱标签微调
- 若是私有核查轮次，还要补充:
  - `consensus ratio`
  - `pairwise disagreement ratio`
  - `scratch_length_mm / scratch_area_mm2 / scratch_components`
  - `spot_area_mm2 / damage_area_mm2`
  - 20~30 张私有图像并列可视化结果

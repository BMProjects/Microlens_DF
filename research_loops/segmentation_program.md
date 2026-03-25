# 分割算法自动化迭代程序

## 目标

建立与正式系统解耦的分割研发线，用多个分割模型对照，逐步把弱标签训练推进为稳定的 mask 预测与后续 `mask->bbox` 可比评测。

## 主指标

- Primary: `scratch IoU` / `mIoU`
- Secondary: `mask->bbox AP@0.5`
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
- 若需要最终与正式系统对照，只在研发后期追加一次 `mask->bbox` 形式的留出集核验

## 第一批模型

- `light_unet`
- `unetplusplus + resnet34`
- `deeplabv3plus + resnet34`
- `fpn + resnet34`

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
3. 在私有训练验证数据上比较 `mIoU / scratch IoU`
4. 仅在需要与正式检测主线对照时，补做一次留出 20 图像的 `mask->bbox` 核验

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

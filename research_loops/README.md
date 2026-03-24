# 自动化迭代准备

本目录吸收 `karpathy/autoresearch` 的几个核心方法，并将其改写为适合本项目的双分支研发规则。

## 核心迁移原则

- 单分支单目标：识别算法分支与分割算法分支分开推进
- 单次只改一类变量：结构、损失、数据、推理后处理四类变量不得混改
- 固定预算：每次实验使用固定训练时长或固定 epoch，保证横向可比
- 固定主指标：识别分支看固定 `CNAS` 集 `mAP@0.5 / scratch AP@0.5`；分割分支看 `mIoU / scratch IoU / mask->bbox AP@0.5`
- 只保留增益：新实验若不能在固定协议下优于父实验，不进入正式主线
- 正式系统隔离：GUI、CNAS、评分主线不直接依赖研究分支实验代码

## 文件

- `detection_program.md`
- `segmentation_program.md`


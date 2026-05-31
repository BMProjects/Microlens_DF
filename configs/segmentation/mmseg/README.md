# Batch-2 MMSegmentation 配置模板

本目录用于第二批分割实验的配置模板占位，当前主要对应：

- `SegFormer-B2`
- `HRNet-OCR-W18`
- `HRNetV2-W18-Seg`

这些模板当前作为实验入口与命令清单的配置锚点，后续接入 `MMSegmentation` 后可逐步补成正式训练配置。

建议统一输出目录：

- `output/experiments/phase3_segmentation/batch2_*`

建议统一先做两阶段：

1. `MSD` 源域预训练
2. 私有弱标签微调

建议统一评测：

- `val_mIoU`
- `scratch IoU`
- `scratch Dice`
- `scratch_length_error`
- `scratch_area_error`
- 桥接验证

# output 目录说明

`output/` 用于存放本地运行、训练、评测、导出和数据构建过程中生成的大体积产物。

为保证 GitHub 协作同步效率，本目录默认不纳入版本管理，仅保留：

- 本说明文件
- `.gitkeep`

## 典型内容

- `output/dataset_v2/`
  原始/预处理后的整图数据、标定图和质量报告
- `output/tile_dataset/`
  切片图像、标签、索引、覆盖图、备份标签
- `output/training/`
  正式训练结果、权重、训练曲线、验证图
- `output/experiments/`
  各类研究实验产物、日志、对比报告、临时权重
- `output/cnas_eval/`
  独立评测执行后的结果文件
- `output/audit/`
  审计队列、候选图、覆层图和排查结果
- `output/release/`
  打包后的数据集或交付压缩包

## GitHub 协作建议

需要随仓库同步的小体积文档素材，请放入：

- `doc/assets/generated/`
- `doc/assets/references/`

不要继续把这类文档插图放在 `output/` 下，否则云端协作时文档引用容易失效。

# 智能磨损识别技术测试大纲

## 1. 基本信息

- 预期成果名称：智能磨损识别技术
- 指标名称：磨损识别准确率
- 中期指标值：60%
- 当前量化判据：`mAP@0.5`
- 推荐执行命令：`uv run python -m cnas_test.runner.run_eval`
- 兼容命令：`python scripts/run_cnas_eval.py`

## 2. 数据集口径

项目数据集采用两种数量表述方式并行说明：一是按实际拍摄得到的原图数量统计，二是按算法实际输入的切片数量统计。

| 数据集层级 | 原图数量 | 切片数量 | 标注缺陷框 |
| --- | ---: | ---: | ---: |
| 整体数据集 | 247 | 10621 | 112929 |
| 训练验证数据集 | 227 | 9761 | 104608 |
| 留出测试数据集 | 20 | 860 | 8321 |

| 数据集层级 | scratch | spot | critical |
| --- | ---: | ---: | ---: |
| 整体数据集 | 75995 | 19997 | 16937 |
| 训练验证数据集 | 70424 | 18506 | 15678 |
| 留出测试数据集 | 5571 | 1491 | 1259 |

## 3. 测试目标

验证智能磨损识别模型在留出测试数据集上的磨损识别准确率是否达到中期指标值 `60%`，并确保测试流程、输出结构和归档资料可直接用于第三方机构核验。

## 4. 指标定义与计算方法

1. 留出测试数据集采用固定清单 [`test_set_v1.json`](/home/bm/Dev/Microlens_DF/cnas_test/manifests/test_set_v1.json)。
2. 根据图像 stem 从 `output/tile_dataset/images/val/` 收集对应 `860` 个切片，并同步读取标签目录中的真实标注框。
3. 对全部测试切片执行批量推理，输出预测类别、预测框与置信度。
4. 当预测框与真实框类别一致且 `IoU >= 0.5` 时记为 `TP`；未匹配预测记为 `FP`；未匹配真实记为 `FN`。
5. `Precision = TP / (TP + FP)`。
6. `Recall = TP / (TP + FN)`。
7. 单类 `AP@0.5` 为对应 `Precision-Recall` 曲线下面积。
8. `mAP@0.5 = (AP50_scratch + AP50_spot + AP50_critical) / 3`。
9. 当 `mAP@0.5 >= 0.60` 时，判定“磨损识别准确率”达到中期目标。

## 5. 测试执行与输出

推荐在项目根目录执行：

```bash
uv run python -m cnas_test.runner.run_eval
```

执行后默认产物统一保存到 [`cnas_test/outputs/latest`](/home/bm/Dev/Microlens_DF/cnas_test/outputs/latest)：

- `dataset/`：测试切片清单与临时评测 YAML
- `metrics/`：`cnas_eval_results.json`
- `plots/`：评测曲线、混淆矩阵与可视化图
- `reports/`：自动生成的 Markdown 测试报告
- `delivery_manifest.json`：交付物总清单

## 6. 交付要求

- 固定测试集清单不随实验调整
- 测试参数固定为 `conf=0.001`、`iou=0.6`
- 输出目录结构固定，便于第三方测试和归档
- 正式对外提交时，`doc/` 中的 Word 版本应与本文件保持一致

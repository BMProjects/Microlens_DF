# CNAS 测试报告

## 1. 测试概况

- 测试时间：{{TEST_TIMESTAMP}}
- 标准命令：`{{COMMAND}}`
- 兼容命令：`{{COMPAT_COMMAND}}`
- 测试集清单：`{{TEST_SET_PATH}}`
- 模型权重：`{{MODEL_WEIGHTS}}`

## 2. 数据集说明

- 留出测试数据集：`{{HOLDOUT_IMAGES}}` 张原图，`{{HOLDOUT_TILES}}` 个切片，`{{HOLDOUT_BOXES}}` 个标注缺陷框
- 类别分布：`scratch={{SCRATCH_BOXES}}`，`spot={{SPOT_BOXES}}`，`critical={{CRITICAL_BOXES}}`
- 本次实际参与评测切片数：`{{N_TILES}}`

## 3. 评测参数

- 置信度阈值：`{{EVAL_CONF}}`
- IoU 阈值：`{{EVAL_IOU}}`
- 通过标准：`mAP@0.5 >= {{PASS_THRESHOLD}}`

## 4. 测试结果

| 指标 | 数值 |
| --- | --- |
| scratch AP@0.5 | {{SCRATCH_AP50}} |
| spot AP@0.5 | {{SPOT_AP50}} |
| critical AP@0.5 | {{CRITICAL_AP50}} |
| mAP@0.5 | {{MAP50}} |
| mAP@0.5:0.95 | {{MAP50_95}} |
| Precision | {{PRECISION}} |
| Recall | {{RECALL}} |
| 耗时（秒） | {{ELAPSED_SECONDS}} |

## 5. 结论

- 测试结论：`{{VERDICT}}`
- 判定依据：`mAP@0.5 = {{MAP50}}`，对照标准 `{{PASS_THRESHOLD}}`


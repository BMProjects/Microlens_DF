# CNAS 测试大纲模板

## 1. 基本信息

- 预期成果名称：智能磨损识别技术
- 指标名称：磨损识别准确率
- 中期指标值：60%
- 标准命令：`uv run python -m cnas_test.runner.run_eval`

## 2. 数据集口径

- 整体数据集：247 张原图，10621 个切片，112929 个标注缺陷框
- 训练验证数据集：227 张原图，9761 个切片，104608 个标注缺陷框
- 留出测试数据集：20 张原图，860 个切片，8321 个标注缺陷框

## 3. 指标定义

- 当预测框与真实框类别一致且 `IoU >= 0.5` 时记为 `TP`
- 未匹配预测框记为 `FP`
- 未匹配真实框记为 `FN`
- `Precision = TP / (TP + FP)`
- `Recall = TP / (TP + FN)`
- 单类 `AP@0.5` 为对应 `Precision-Recall` 曲线下面积
- `mAP@0.5 = (AP50_scratch + AP50_spot + AP50_critical) / 3`
- 判定规则：`mAP@0.5 >= 0.60`

## 4. 输出物要求

- `dataset/`：测试集 YAML 与切片清单
- `metrics/`：评测 JSON 结果
- `plots/`：曲线图与混淆矩阵
- `reports/`：Markdown 测试报告
- `delivery_manifest.json`：交付物总清单

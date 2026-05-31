# CNAS 测试报告

本报告为第三方测试结果记录，用于客观呈现测试环境、测试数据、测试过程、复现条件和实测指标。测试结论或符合性判定依据以委托测试文件、测试机构正式判定规则为准。

## 1. 测试概况

- 被测软件名称：{{SOFTWARE_NAME}}
- 测试版本：`{{SOFTWARE_VERSION}}`
- 版本备注：{{VERSION_NOTE}}
- 模型描述：{{MODEL_DESCRIPTION}}
- 测试时间：{{TEST_TIMESTAMP}}
- 标准命令：`{{COMMAND}}`
- 兼容命令：`{{COMPAT_COMMAND}}`
- 测试集清单：`{{TEST_SET_PATH}}`
- 模型权重：`{{MODEL_WEIGHTS}}`

## 2. 数据集说明

- 数据集名称：{{DATASET_NAME}}
- 数据集类型：{{STANDARD_SAMPLE_UNIT}}数据集
- 样本构建方法：{{DATASET_CONSTRUCTION_METHOD}}
- 样本图像尺寸：`640 × 640` 像素
- 样本图像数量：`{{DATASET_TILES}}`
- 标注缺陷框数量：`{{DATASET_BOXES}}`
- 类别分布：`scratch={{SCRATCH_BOXES}}`，`spot={{SPOT_BOXES}}`，`critical={{CRITICAL_BOXES}}`
- 本次实际参与评测样本数：`{{N_TILES}}`
- 空背景样本数：`{{BACKGROUND_TILES}}`

## 3. 数据划分与执行过程

### 3.1 单次测试

单次测试使用固定模型权重和经确认的测试集清单 `{{TEST_SET_PATH}}`，纳入全部 `{{N_TILES}}` 个标准化图像样本进行评测。测试程序自动生成数据清单、评测日志、过程截图、指标结果、溯源文件和本报告。

### 3.2 训练后测试

训练后测试先使用既定训练配置重新训练模型，再使用训练输出的 `weights/best.pt` 按单次测试流程评测。训练阶段使用项目内固定数据划分：

| 子集 | 图像编号数 | 标准化图像样本数 | 标注缺陷框数 | 用途 |
| --- | ---: | ---: | ---: | --- |
| 训练子集 | {{TRAIN_IMAGES}} | {{TRAIN_TILES}} | {{TRAIN_BOXES}} | 参数学习 |
| 验证子集 | {{VAL_IMAGES}} | {{VAL_TILES}} | {{VAL_BOXES}} | 训练过程监控与模型选择 |

第三方测试报告中的最终指标以测试命令对全量确认样本重新计算得到的结果为准。

## 4. 评测参数

- 置信度阈值：`{{EVAL_CONF}}`
- IoU 阈值：`{{EVAL_IOU}}`
- AP 匹配 IoU：`0.50`

## 5. 结果计算方法

对每个类别 \(c\)，预测框按置信度从高到低排序，并在 \(IoU = 0.5\) 条件下与同类别标注框进行一对一匹配。第 \(k\) 个阈值位置的精确率和召回率为：

\[
Precision_c(k)=\frac{TP_c(k)}{TP_c(k)+FP_c(k)}
\]

\[
Recall_c(k)=\frac{TP_c(k)}{N_{gt,c}}
\]

类别平均精度 \(AP50_c\) 为该类别精确率-召回率曲线下面积：

\[
AP50_c=\int_0^1 P_c(R)\,dR,\quad IoU=0.5
\]

本次共有 \(C=3\) 个评价类别，主指标 \(mAP@0.5\) 为三类 \(AP50\) 的算术平均：

\[
mAP@0.5=\frac{1}{C}\sum_{c=1}^{C}AP50_c
\]

## 6. 测试结果

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

## 7. 结论

- 本次测试完成，主指标实测值：`mAP@0.5 = {{MAP50}}`
- 本报告仅记录实测结果；符合性判定以委托测试文件或测试机构正式规则为准。

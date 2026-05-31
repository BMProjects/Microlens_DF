# 智能识别准确率测试报告

## 1 主题内容和适用范围

### 1.1 主题内容

本报告依据《智能识别准确率测试大纲》编制，用于记录“离焦微结构镜片磨损智能识别任务”中“磨损识别准确率”测试的实际执行情况、测试条件、测试方法、测试结果和测试结论。

本报告仅围绕课题任务书中“2.1 磨损识别准确率”相关要求形成，不扩展至其他功能指标、性能指标或研究性分析内容。

### 1.2 适用范围

本报告适用于本任务当前送测软件产品的第三方测试记录与结果说明。当前被测对象为采用目标级检测输出方式的镜片磨损智能识别算法。

本报告记录的是测试图像中的缺陷识别准确率测试结果。依据测试大纲，当前送测版本采用目标级评价方式，因此本报告中的“磨损识别准确率”以 `mAP50` 表示。

## 2 测试目的

本次测试的目的是验证被测软件产品对测试图中缺陷的识别结果是否达到规定的准确率要求，并为委托方、测试机构和验收使用方提供客观、可追溯的测试结果。

本次测试仅针对“磨损识别准确率”开展。依据测试大纲中“统一指标名称、按输出形式选择评价方式”的原则，当前被测软件因采用目标级检测输出，测试主指标确定为 `mAP50`。辅助指标包括 `Precision`、`Recall` 和 `mAP@0.5:0.95`，用于补充说明被测软件的识别性能。

## 3 测试依据和引用文件

本次测试主要依据以下文件执行：

- `/home/bm/Dev/Microlens_DF/cnas_test/docs/CNAS测试大纲_当前版.md`
- `/home/bm/Dev/Microlens_DF/doc/项目任务书2024YFC2419500.pdf`
- `/home/bm/Dev/Microlens_DF/doc/课题任务书2024YFC2419504.pdf`
- `GB/T 25000.51-2016《系统与软件工程 系统与软件质量要求与评价（SQuaRE） 第 51 部分：就绪可用软件产品（RUSP）的质量要求和测试细则》`
- `/home/bm/Dev/Microlens_DF/cnas_test/outputs/latest/metrics/cnas_eval_results.json`

## 4 测试组织管理和流程

### 4.1 测试组织管理

本次测试按照第三方软件测试的一般管理要求执行。测试中涉及的主要对象和记录内容如下：

- 被测软件名称：镜片磨损智能识别算法
- 测试版本：`LWIA-Det v1.0.0`
- 版本备注：内部训练注释 `B2_nwd_only_phase3e`
- 实现方式：目标级检测输出
- 模型描述：多尺度深度目标检测网络，结合小目标定位优化策略
- 模型权重：`/home/bm/Dev/Microlens_DF/output/experiments/phase3e/detection_training/b2_nwd_only_phase3e/weights/best.pt`

本次测试中，测试机构需核对以下信息并予以记录：

- 被测软件版本和模型版本
- 测试命令和运行环境
- 测试图像集清单
- 测试结果文件
- 异常信息和日志信息

### 4.2 测试流程

本次测试按以下流程完成：

1. 核对被测软件版本、模型权重和运行说明。
2. 核对测试图像集清单和基准结果。
3. 按规定命令在测试环境中执行识别评测。
4. 获取识别结果并进行统计。
5. 根据测试大纲规定的指标定义和计算方法形成测试结果。
6. 编制测试报告。

本次测试所采用的标准执行命令为：

```bash
cd /home/bm/Dev/Microlens_DF
uv run python -m cnas_test.runner.run_eval
```

本次测试的实际输出包括：

- 指标结果文件：`/home/bm/Dev/Microlens_DF/cnas_test/outputs/latest/metrics/cnas_eval_results.json`
- 交付清单：`/home/bm/Dev/Microlens_DF/cnas_test/outputs/latest/delivery_manifest.json`
- 混淆矩阵图：`/home/bm/Dev/Microlens_DF/cnas_test/outputs/latest/plots/ultralytics_output/confusion_matrix_normalized.png`
- 精确率-召回率曲线图：`/home/bm/Dev/Microlens_DF/cnas_test/outputs/latest/plots/ultralytics_output/BoxPR_curve.png`
- 典型预测结果图：`val_batch0_pred.jpg`、`val_batch1_pred.jpg`、`val_batch2_pred.jpg`
- 测试过程截图：`/home/bm/Dev/Microlens_DF/cnas_test/outputs/latest/screenshots/`
- 测试溯源文件：`/home/bm/Dev/Microlens_DF/cnas_test/outputs/latest/provenance/provenance.json`
- Web / Word 报告：`/home/bm/Dev/Microlens_DF/cnas_test/outputs/latest/reports/cnas_test_report.html`、`cnas_test_report.docx`

## 5 性能指标测试

### 5.1 测试说明和要求

本次测试仅设置一个测试项目，即“磨损识别准确率测试”。

测试内容为：在固定测试图像集上，对图像中的缺陷进行识别，并将被测软件的识别结果与人工确认的基准结果进行比较，计算磨损识别准确率。

本次测试中，“磨损识别准确率”采用目标级评价方式，正式评价指标为 `mAP50`。其原因是当前送测软件采用目标级检测输出，且在多个识别类别上统一进行评测，因此使用 `mAP50` 作为总体识别准确率指标，既符合当前项目实现方式，也符合通用目标检测评价标准。

本次测试所使用的数据如下：

- 测试图像集：完整测试数据集
- 原图数量：`247`
- 切片数量：`10621`
- 标注缺陷总数：`90325`
- 测试集清单：`/home/bm/Dev/Microlens_DF/cnas_test/manifests/full_dataset_v1.json`

### 5.2 测试环境及条件

本次测试环境和条件如下：

- 操作系统环境：Linux
- 执行目录：`/home/bm/Dev/Microlens_DF`
- 评测命令：`uv run python -m cnas_test.runner.run_eval`
- 置信度阈值：`0.001`
- 评测结果文件：`/home/bm/Dev/Microlens_DF/cnas_test/outputs/latest/metrics/cnas_eval_results.json`

测试环境满足以下条件：

- 能正常加载被测软件和模型权重
- 能正常读取测试图像集与基准结果
- 能正常输出识别结果、统计结果和评测日志
- 能完整保存测试记录与测试结果文件

测试过程中实际核验的关键输入和输出如下：

| 类别 | 文件或内容 | 作用 |
| --- | --- | --- |
| 输入 | `cnas_test/manifests/full_dataset_v1.json` | 完整测试图像集清单 |
| 输入 | `cnas_test/outputs/latest/dataset/cnas_val_list.txt` | 测试图像列表 |
| 输入 | `output/experiments/phase3e/detection_training/b2_nwd_only_phase3e/weights/best.pt` | 被测模型权重 |
| 输出 | `metrics/cnas_eval_results.json` | 正式统计指标结果 |
| 输出 | `plots/ultralytics_output/confusion_matrix_normalized.png` | 类别混淆情况 |
| 输出 | `plots/ultralytics_output/BoxPR_curve.png` | 精确率-召回率曲线 |
| 输出 | `plots/ultralytics_output/val_batch*_pred.jpg` | 典型预测可视化结果 |
| 输出 | `screenshots/01_startup_banner.png` 等 | 测试过程关键节点截图 |
| 输出 | `provenance/provenance.json` | git、权重、测试集、依赖与硬件溯源 |
| 输出 | `reports/cnas_test_report.html` / `.docx` | Web 与 Word 版正式测试报告 |

### 5.3 测试方法和步骤

本次测试按测试大纲规定的方法执行。依据测试大纲的统一口径，磨损识别准确率作为统一指标名称，可采用目标级评价方式或区域级评价方式；当前送测版本采用目标级检测输出，因此本次测试采用目标级评价方式，以 `mAP50` 计算。

术语定义统一表述如下：本报告中的“磨损识别准确率”系指被测软件产品在规定测试图像集上，对磨损目标识别结果与基准结果一致程度的评价指标。根据被测软件输出形式不同，磨损识别准确率可采用目标级评价方式或区域级评价方式进行计算。当被测软件输出为目标级识别结果时，磨损识别准确率可采用 `mAP50` 表示；当被测软件输出为区域级识别结果时，磨损识别准确率可采用区域级重合一致性指标表示。具体评价方式及其计算规则应由委托测试文件或测试实施细则明确，并在同一次正式测试中保持一致。`AP50` 指单一识别类别在 `IoU = 0.5` 条件下的平均准确率，`mAP50` 指全部参与评价类别的 `AP50` 的算术平均值；`Precision` 用于表征预测结果中正确识别结果所占比例，`Recall` 用于表征基准结果中被成功识别出的比例。当前报告仅对应目标级评价方式的测试结果。

`mAP50` 的严格定义如下：

对每一个识别类别，在 `IoU = 0.5` 的条件下，将预测结果按置信度从高到低排序，并与对应类别的基准结果进行一对一匹配。对于排序后的前 `k` 个预测结果，累计精确率和累计召回率分别定义为：

\[
Precision(k) = \frac{TP(k)}{TP(k) + FP(k)}
\]

\[
Recall(k) = \frac{TP(k)}{N_{gt}}
\]

由不同 `k` 形成精确率-召回率曲线。该类别的平均准确率 `AP50` 定义为在 `IoU = 0.5` 条件下精确率-召回率曲线下面积：

\[
AP50 = \int_{0}^{1} P(R)\,dR \quad \text{(IoU = 0.5)}
\]

对全部参与评价的类别分别计算 `AP50` 后，取其算术平均值，得到本次测试的主指标 `mAP50`：

\[
mAP50 = \frac{1}{C} \sum_{i=1}^{C} AP50_i
\]

其中：

- `TP` 为正确匹配的预测数量
- `FP` 为错误预测或多余预测数量
- `N_gt` 为该类别基准目标总数
- `C` 为参与评价的识别类别总数

辅助指标定义如下：

\[
Precision = \frac{TP}{TP + FP}
\]

\[
Recall = \frac{TP}{TP + FN}
\]

本次测试按以下步骤执行：

#### 步骤一：测试准备

1. 准备固定留出测试集及人工确认基准结果。
2. 核对被测模型权重、推理脚本、配置文件和测试图像清单。
3. 确认测试环境可正常加载模型并读取全部测试样本。

#### 步骤二：执行识别评测

1. 在规定目录下执行：

```bash
cd /home/bm/Dev/Microlens_DF
uv run python -m cnas_test.runner.run_eval
```

2. 被测软件对测试图像进行目标级识别，输出预测框、类别和置信度。
3. 评测程序按置信度排序全部预测结果，并在 `IoU = 0.5` 条件下完成逐类别匹配。

#### 步骤三：统计和计算

1. 统计各类别 `TP`、`FP`、`FN`。
2. 计算各类别的 `AP50`。
3. 对各类别 `AP50` 取平均，得到 `mAP50`。
4. 同时输出 `Precision`、`Recall` 和 `mAP@0.5:0.95` 作为辅助统计结果。

#### 步骤四：结果输出与核验

1. 输出正式指标文件 `cnas_eval_results.json`。
2. 输出混淆矩阵、PR 曲线和典型预测样例图。
3. 核对测试图像数量、切片数量、统计值和结果文件是否完整。
4. 依据测试大纲和委托测试要求形成测试报告。

本次测试结果如下：

| 指标 | 数值 |
| --- | ---: |
| 磨损识别准确率（mAP50） | 0.6844 |
| 精确率（Precision） | 0.6030 |
| 召回率（Recall） | 0.6725 |
| mAP@0.5:0.95 | 0.4606 |
| 测试耗时（秒） | 8.5 |

各识别类别的 `AP50` 结果如下：

| 类别 | AP50 |
| --- | ---: |
| scratch | 0.4671 |
| spot | 0.8178 |
| critical | 0.7685 |

结果说明如下：

- 本报告中的“磨损识别准确率”采用目标级评价方式，以 `mAP50` 表示
- 当前送测软件输出为目标级检测结果，因此 `mAP50` 与当前评测文件中的 `mAP@0.5` 对应
- `Precision` 和 `Recall` 反映预测结果的准确程度与检出完整程度
- `mAP@0.5:0.95` 用于补充反映在更严格重叠阈值范围下的整体表现

主要输出文件核验结果如下：

| 输出项 | 文件路径 | 核验情况 |
| --- | --- | --- |
| 正式指标结果 | `cnas_test/outputs/latest/metrics/cnas_eval_results.json` | 已生成 |
| 混淆矩阵 | `cnas_test/outputs/latest/plots/ultralytics_output/confusion_matrix_normalized.png` | 已生成 |
| PR 曲线 | `cnas_test/outputs/latest/plots/ultralytics_output/BoxPR_curve.png` | 已生成 |
| 预测样例图 | `cnas_test/outputs/latest/plots/ultralytics_output/val_batch0_pred.jpg` | 已生成 |
| 测试过程截图 | `cnas_test/outputs/latest/screenshots/01_startup_banner.png` 等 | 已生成 |
| 测试溯源文件 | `cnas_test/outputs/latest/provenance/provenance.json` | 已生成 |
| Web 测试报告 | `cnas_test/outputs/latest/reports/cnas_test_report.html` | 已生成 |
| Word 测试报告 | `cnas_test/outputs/latest/reports/cnas_test_report.docx` | 已生成 |

典型结果截图如下。

1. 精确率-召回率曲线  
![PR Curve](/home/bm/Dev/Microlens_DF/cnas_test/outputs/latest/plots/ultralytics_output/BoxPR_curve.png)

2. 归一化混淆矩阵  
![Confusion Matrix](/home/bm/Dev/Microlens_DF/cnas_test/outputs/latest/plots/ultralytics_output/confusion_matrix_normalized.png)

3. 典型预测样例一  
![Prediction Batch0](/home/bm/Dev/Microlens_DF/cnas_test/outputs/latest/plots/ultralytics_output/val_batch0_pred.jpg)

4. 典型预测样例二  
![Prediction Batch1](/home/bm/Dev/Microlens_DF/cnas_test/outputs/latest/plots/ultralytics_output/val_batch1_pred.jpg)

根据本次测试结果，被测软件产品在规定测试图像集上的“磨损识别准确率”为：

\[
mAP50 = 0.6844
\]

测试机构应将该结果与委托测试文件、任务书或测试实施细则中规定的阈值进行比较后，形成最终判定结论。若后续送测版本采用区域级识别输出，则仍使用同一“磨损识别准确率”指标名称，但按测试大纲规定切换为区域级评价方式并重新形成相应测试报告。

# Microlens_DF

> **暗场离焦微结构镜片缺陷检测、分析与磨损评估系统**
> Dark-field defect detection, analysis & wear-grading for defocused micro-structured lenses.

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](pyproject.toml)
[![Status](https://img.shields.io/badge/status-阶段性成果%20·%20架构定型-success.svg)](#)

从暗场显微图像中**检测三类缺陷（划痕 / 斑点 / 缺损）→ 像素级分析 → 量化磨损评分与 A/B/C/D 评级**，并提供面向非专业用户的图形化工作台。

![GUI 演示：上传 → 检测 → 结果](doc/assets/generated/gui_demo.webp)

*上传单张镜片图像，一键得到检测叠加图、缺陷地形图、WearScore 计算过程与磨损评级卡。*

---

## ✨ 核心能力

| 能力 | 说明 |
|---|---|
| 暗场预处理流水线 | 背景模板融合、高光环配准、亮度修正、ROI 提取 |
| 缺陷检测 | YOLOv12m + SAHI 切片推理 + IOS 跨片合并；mAP@0.5 = **0.6765**（CNAS 测试集） |
| 缺陷分割 | HRNet-OCR / SegFormer 像素级形态刻画（研发分支） |
| 磨损评分 | WearScore 五因子加权评分 → A/B/C/D 评级与结论 |
| 图形工作台 | Gradio 四步流程，面向非专业操作员 |
| 第三方测试 | 独立、可复现的 CNAS 准确率测试子系统 |

---

## 🏗️ 模块化架构

系统按 8 个接口清晰的模块组织（详见 [系统架构_模块化梳理](doc/系统架构_模块化梳理_20260531.md)）：

```text
原始暗场图像 ─①数据集─▶ ②预处理 ─▶ ④识别(YOLO+SFE+NWD) ─┐
                              └─▶ ⑤分割(HRNet-OCR/SegFormer)─┤
                                                              ▼
                              ⑥评估(mAP / WearScore 评级) ─▶ ⑦前端
                                                              ▲
              ③半监督标注(伪/弱标签)──回流──▶ ①数据集         │
                                                  ⑧CNAS 第三方测试
```

| # | 模块 | 主要代码 |
|---|---|---|
| ① | 数据集 | `output/dataset_v2/`（数据，不入库）、`src/.../data/`、`scripts/build_tile_dataset.py` |
| ② | 预处理 | `src/darkfield_defects/preprocessing/` |
| ③ | 半监督标注 | `scripts/step1_label_cleanup.py` … `step5_retrain.py`、`generate_weak_masks.py` |
| ④ | 识别算法 | `src/darkfield_defects/{detection,ml}/`、`scripts/infer_full_pipeline.py` |
| ⑤ | 分割算法 | `src/.../ml/{segmentation_factory,hrnet_ocr}.py`、`configs/segmentation/mmseg/` |
| ⑥ | 评估算法 | `src/darkfield_defects/{eval,scoring}/` |
| ⑦ | 前端系统 | `src/darkfield_defects/viz/app.py`、`app_services/inference_service.py` |
| ⑧ | 第三方测试 | `cnas_test/` |

![系统总架构](doc/assets/generated/project_system_architecture.png)

---

## 🚀 快速开始

项目使用 [`uv`](https://github.com/astral-sh/uv) 管理 Python 环境。

```bash
# 1. 安装依赖（GUI 主线通常只需这一条）
uv sync --extra dev --extra ml --extra viz

# 2. 启动图形工作台 → http://127.0.0.1:7860
uv run python -m darkfield_defects.viz.app

# 3. 运行测试
uv run pytest -q
```

> **注意**：仓库不含数据集与训练权重。GUI 默认权重路径为
> `output/experiments/phase3e/.../weights/best.pt`，需在本地提供权重后方可推理。

### 其他入口

```bash
uv run python -m cnas_test.runner.run_eval --help       # CNAS 第三方测试
uv run python scripts/train_msd_segmentation.py --help   # 分割实验
uv run python scripts/infer_full_pipeline.py --help      # 全图推理流水线
```

---

## 📂 目录结构

```text
src/            系统源码（8 模块）
scripts/        研究与批处理脚本
configs/        检测/分割配置
tests/          自动化测试
cnas_test/      独立第三方测试子系统
research_loops/ 自动化研发规则与实验程序
doc/            成果型图文文档（web 版） + doc/assets/ 长期引用图片
output/         数据集 / 权重 / 实验与运行产物（默认不入 Git）
```

---

## 🔒 数据与代码分离

仓库仅同步**源码、配置、测试、轻量清单/模板与成果型文档**。以下默认**不纳入版本控制**（见 [.gitignore](.gitignore)）：

- 数据集与切片（`output/dataset_v2/`、`output/tile_dataset/` …）
- 训练输出与实验产物（`output/training/`、`output/experiments/`、`runs/`）
- 模型权重（`*.pt` / `*.pth` / `*.onnx`）
- 评测运行产物（`cnas_test/outputs/`）
- 二进制/敏感交付件（`*.pdf` / `*.docx` / `*.odt`）与归档文档（`output/_doc_archive/`）

---

## 📑 文档索引

打开 [`doc/index.html`](doc/index.html) 浏览全部图文文档。关键成果文档：

- [系统架构_模块化梳理](doc/系统架构_模块化梳理_20260531.md) — 本项目模块化总览
- [软件系统说明文档 SDD v1](doc/软件系统说明文档_SDD_v1.md)
- [研究技术文档_预发布与算法演进](doc/研究技术文档_20260324_预发布与算法演进.md)
- [背景技术综合研究报告 v1](doc/背景技术综合研究报告_v1.md)
- [GitHub 协作同步与大文件管理说明](doc/GitHub协作同步与大文件管理说明_20260324.md)

---

## 📜 许可证与知识产权

代码以 [Apache License 2.0](LICENSE) 授权。本项目为国家重点研发计划相关课题
（2024YFC2419504 / 2024YFC2419500）研究成果的一部分；原始数据、官方任务书与
科技报告等不随本仓库公开。详见 [NOTICE](NOTICE)。

> 第三方组件 Ultralytics YOLO 采用 AGPL-3.0，若用于分发或网络服务请遵守其义务。

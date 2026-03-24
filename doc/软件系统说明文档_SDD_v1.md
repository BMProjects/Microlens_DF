# 暗场显微镜镜片缺陷智能检测系统
# 软件系统说明文档（SDD）v1.0

**文档编号**：Microlens_DF-SDD-v1.0
**版本**：1.0
**日期**：2026-03-22
**项目代号**：darkfield-defects
**密级**：内部文件

---

## 文档修订历史

| 版本 | 日期 | 修订人 | 修订说明 |
|------|------|--------|---------|
| 0.1 | 2026-02-18 | 开发团队 | 初稿，架构框架 |
| 0.2 | 2026-03-20 | 开发团队 | Phase 2B 完成后补充训练流程 |
| 1.0 | 2026-03-22 | 开发团队 | Phase 2C 完成，整合 CNAS 测试结果，正式发布 |

---

## 目录

1. [引言](#1-引言)
2. [系统概述](#2-系统概述)
3. [软件架构设计](#3-软件架构设计)
4. [核心算法说明](#4-核心算法说明)
5. [数据设计](#5-数据设计)
6. [接口设计](#6-接口设计)
7. [部署与运行](#7-部署与运行)
8. [性能指标与测试结果](#8-性能指标与测试结果)
9. [已知限制与待改进项](#9-已知限制与待改进项)
- [附录 A：训练配置参数完整表](#附录-a训练配置参数完整表)
- [附录 B：关键文件路径速查表](#附录-b关键文件路径速查表)
- [附录 C：常用命令速查表](#附录-c常用命令速查表)

---

## 1. 引言

### 1.1 文档目的与适用范围

本文档为"暗场显微镜镜片缺陷智能检测系统"的软件系统说明文档，综合覆盖软件需求规格（SRS）与软件设计描述（SDD）两类工程文档的内容。

**文档目的**：

1. 完整描述系统的功能需求、约束条件、运行环境；
2. 详细说明软件各模块的内部设计、算法原理与接口规范；
3. 为后续开发迭代、第三方测试（CNAS）、系统维护提供权威参考；
4. 作为项目交付物的技术附件。

**适用范围**：

本文档适用于以下读者：
- **系统工程师**：了解整体架构和模块划分；
- **算法工程师**：理解各检测算法的设计思路与参数含义；
- **测试工程师**（含 CNAS 第三方）：按接口规范复现测试结果；
- **运维人员**：按部署说明完成环境搭建与系统维护。

本文档描述的软件版本为 `v0.1.0`（pyproject.toml 定义），对应代码路径 `/home/bm/Dev/Microlens_DF/`。

---

### 1.2 术语定义与缩略语

| 编号 | 术语/缩略语 | 全称/定义 |
|------|------------|---------|
| 1 | **暗场显微镜** | Dark-field microscope。利用斜向照明使散射光成像、背景为黑色的光学显微系统，缺陷区域因散射而呈亮点/亮线 |
| 2 | **划痕（Scratch）** | 镜片表面线状机械损伤，在暗场像中呈细长高亮线条，YOLO 类别 ID = 0 |
| 3 | **斑点（Spot）** | 镜片表面点状缺陷（凹坑、气泡、夹杂物），暗场像中呈近圆形高亮点，YOLO 类别 ID = 1 |
| 4 | **临界缺陷（Critical）** | 大面积不规则高亮区域（dense damage 或 crash zone），YOLO 类别 ID = 2，由内部 DAMAGE 和 CRASH 两类合并而来 |
| 5 | **ROI** | Region of Interest，感兴趣区域，本系统中特指镜片圆形有效区域 |
| 6 | **Ring（高光环）** | 暗场显微镜照明在镜片边缘形成的高亮环形区域，作为配准模板的特征结构 |
| 7 | **ECC 配准** | Enhanced Correlation Coefficient，增强相关系数算法，OpenCV 实现的亚像素精度图像配准方法 |
| 8 | **仿射变换** | Affine Transform，保持平行线关系的线性变换，含平移、旋转、缩放（各向同性/异性） |
| 9 | **背景减除** | Background Subtraction，用参考背景图对目标图像进行除法校正，消除光照不均 |
| 10 | **Frangi 滤波器** | Frangi Vesselness Filter，基于 Hessian 矩阵特征值检测管状/线状结构的图像滤波器 |
| 11 | **TopHat 变换** | Top-hat Transform，形态学白顶帽变换，用于提取比结构元素更小的亮区特征 |
| 12 | **Otsu 阈值** | Otsu's Method，最大类间方差自动阈值分割算法 |
| 13 | **YOLO** | You Only Look Once，单阶段实时目标检测框架，本系统使用 YOLOv12m 变体 |
| 14 | **YOLOv12m** | YOLO 第 12 代 medium 规模模型，引入 Area Attention 机制，参数量约 20M |
| 15 | **SAHI** | Slicing Aided Hyper Inference，切片辅助超推理框架，通过滑窗切片提升小目标检测召回率 |
| 16 | **IOS NMS** | Intersection over Smaller，以较小框面积为分母的非极大值抑制，适用于长宽比差异大的划痕检测 |
| 17 | **mAP** | mean Average Precision，目标检测标准评估指标，mAP@0.5 表示 IoU 阈值为 0.5 时的平均精度均值 |
| 18 | **IoU** | Intersection over Union，两个边框（或掩码区域）的交集与并集之比 |
| 19 | **Tile（切片）** | 将大尺寸全图按固定尺寸（640×640）滑窗裁切得到的子图像 |
| 20 | **WearScore** | 本系统定义的镜片磨损综合评分，0-100 分，越高越严重 |
| 21 | **WearMetrics** | 磨损量化指标集合，包含划痕总长度、中心区长度、条数、面积、密度、散射强度等物理量 |
| 22 | **WearAssessment** | 磨损评估结果，包含 WearScore、等级（A/B/C/D）、眩光指数、雾化指数及可解释结论 |
| 23 | **BboxDefect** | 基于边框的缺陷代理适配器，将 YOLO 输出的 bbox 转换为 WearMetrics 所需的物理量（骨架长度以短边估算）|
| 24 | **伪标签** | Pseudo Labels，使用当前模型对未标注数据推理得到的标注，用于半监督学习迭代 |
| 25 | **connect_scratches** | 将共线、端点距离小于阈值的划痕碎片合并为完整划痕的后处理算法 |
| 26 | **CLAHE** | Contrast Limited Adaptive Histogram Equalization，限制对比度自适应直方图均衡化 |
| 27 | **CNAS** | China National Accreditation Service for Conformity Assessment，中国合格评定国家认可委员会，本项目指第三方测试机构 |
| 28 | **相位相关** | Phase Correlation，基于傅里叶频域互功率谱的快速平移估计算法 |
| 29 | **DFL** | Distribution Focal Loss，分布焦点损失，用于 YOLO 边框回归的连续分布预测 |
| 30 | **Varifocal Loss** | 可变焦点损失，非对称加权的分类损失函数，对正样本用 IoU 质量加权 |

---

### 1.3 参考文献与依赖项

**标准与规范**：
- GB/T 8567-2006《计算机软件文档编制规范》
- IEEE Std 1016-2009《软件设计描述》
- YOLO 目标检测评测：COCO mAP 标准（IoU 阈值 0.5:0.05:0.95）

**核心算法参考**：
- Frangi, A.F. et al. (1998). "Multiscale vessel enhancement filtering." *MICCAI*.
- Evangelidis, G.D. & Psarakis, E.Z. (2008). "Parametric image alignment using enhanced correlation coefficient." *IEEE TPAMI*.
- Wang, J. et al. (2024). "YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information."
- Akyon, F.C. et al. (2022). "Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection." *ICIP*.

**软件依赖**（详见 `pyproject.toml`）：

| 依赖包 | 版本要求 | 用途 |
|--------|---------|------|
| Python | 3.13.5（实测） | 运行环境 |
| numpy | ≥1.24 | 数值计算基础 |
| opencv-contrib-python | ≥4.8 | 图像处理（ECC 配准、形态学等）|
| scipy | ≥1.10 | 科学计算（连通域、卷积等）|
| scikit-image | ≥0.21 | Frangi 滤波器、骨架化 |
| matplotlib | ≥3.7 | 可视化辅助 |
| pillow | ≥10.0 | 图像 I/O |
| typer | ≥0.9 | CLI 框架 |
| pyyaml | ≥6.0 | 配置文件解析 |
| rich | ≥13.0 | 终端美化输出 |
| torch | ≥2.0（实测 2.10.0+cu128）| 深度学习推理/训练 |
| ultralytics | 8.4.24（固定）| YOLOv12 训练推理框架 |
| gradio | ≥4.0 | Web GUI 界面 |

---

## 2. 系统概述

### 2.1 系统背景

#### 2.1.1 暗场显微镜物理原理

暗场显微镜（Dark-field Microscopy）采用斜向环形照明设计：照明光以大角度斜射到样品表面，正常区域的镜面反射光不进入物镜（背景为黑暗），而表面缺陷（划痕、凹坑、异物）因散射效应将光线折入物镜视场，呈现为黑暗背景上的明亮点线。

本系统使用的暗场显微镜具有以下成像特性：
- **成像分辨率**：约 4096×3000 像素，覆盖镜片有效光学区域（约 30mm 直径）
- **照明模式**：同轴环形暗场照明，在镜片边缘形成特征性"高光环（Ring）"
- **图像格式**：原始采集为 BMP 格式，压缩存储为 PNG
- **多帧采集**：每个镜片采集 5 帧背景图（移除样品）+ N 帧目标图（含样品），背景帧用于平场标定

#### 2.1.2 镜片磨损评估需求

眼镜镜片在日常使用中不可避免地产生表面磨损，主要损伤形态包括：

| 损伤类型 | 物理特征 | 对视觉的影响 |
|---------|---------|------------|
| 划痕（Scratch）| 线状机械划伤，宽度 2-20px，长度可贯穿整个镜片 | 夜间行车产生眩光，严重时降低对比度 |
| 斑点（Spot）| 点状凹坑或异物，直径 5-50px | 轻微时几乎不感知，严重时产生光晕 |
| 临界缺陷（Critical）| 密集缺陷聚集或大面积不规则损伤区 | 显著降低透光率，影响视觉清晰度 |

传统人工检测依赖光学师经验，主观性强、效率低、无法量化。智能检测系统需要：
- 自动定位和分类镜片表面缺陷；
- 量化缺陷的物理尺寸、分布、严重程度；
- 给出标准化磨损等级（A/B/C/D）和可解释结论；
- 满足 CNAS 第三方检测认证标准（mAP@0.5 ≥ 60%）。

---

### 2.2 系统目标与约束

#### 功能目标

1. **预处理**：对多帧背景图进行融合与模板标定，对目标图像完成配准、背景减除、ROI 提取；
2. **缺陷检测**：基于 YOLOv12m 深度学习模型，在 640×640 切片上检测三类缺陷（scratch / spot / critical）；
3. **全图推理**：通过 SAHI 滑窗策略对 4096×3000 全分辨率图像完成无遗漏推理，并用 IOS NMS 消除跨切片重复检测；
4. **磨损评分**：计算 WearMetrics 物理量化指标，输出 WearScore（0-100）与等级（A/B/C/D）；
5. **报告输出**：生成 JSON 格式结构化报告和 HTML 格式可视化报告；
6. **交互界面**：提供 Gradio Web GUI 和 Typer CLI 两种操作方式。

#### 性能约束

- 推理速度：≤ 10ms/tile（640×640，GPU 模式）
- 检测精度：mAP@0.5 ≥ 60%（测试集评估）
- GPU 显存：推理时 ≤ 8GB VRAM（batch=1），训练时 ≤ 24GB VRAM（batch=32）

#### 设计约束

- 深度学习推理依赖 NVIDIA GPU（CUDA 12.8+）
- 模型权重文件固定为 `output/training/stage2_cleaned/weights/best.pt`（CNAS 认证版本）
- 训练框架锁定 Ultralytics 8.4.24，保证实验可复现性
- 代码包结构遵循 `src/` 布局，不污染系统 Python 环境

---

### 2.3 系统边界

#### 系统输入

| 输入类型 | 格式 | 说明 |
|---------|------|------|
| 背景图像组 | PNG / BMP，4096×3000 | 无样品的暗场图像，命名 `bg-*.png`，至少 1 帧，推荐 5 帧 |
| 目标图像 | PNG / BMP，4096×3000 | 含镜片样品的暗场图像 |
| YAML 配置文件 | `.yaml` | 检测参数配置（可选，默认值见 `configs/detect_default.yaml`）|
| 标定结果目录 | 目录 | 预先运行 `calibrate()` 保存的 `.npy` + `calibration.json` |
| 模型权重 | `.pt` | YOLOv12m Ultralytics 格式权重文件 |

#### 系统输出

| 输出类型 | 格式 | 说明 |
|---------|------|------|
| 预处理图像 | PNG | 背景减除后的校正图像 |
| 检测叠加图 | PNG | 原图上绘制缺陷边框和类别标签 |
| JSON 报告 | `.json` | WearMetrics + WearAssessment 结构化数据 |
| HTML 报告 | `.html` | 可视化磨损评估报告（含贡献因素柱状图）|
| COCO 标注 | `.json` | COCO 格式实例标注（用于评估）|
| 质量报告 | `.json` | 批处理时每帧的 ECC 配准分数、亮度修正参数 |

---

### 2.4 运行环境要求

#### 硬件要求

| 组件 | 最低配置 | 推荐配置（实测环境）|
|------|---------|-----------------|
| CPU | 8 核 x86_64 | AMD EPYC / Intel Xeon |
| 内存 | 16 GB | 64 GB |
| GPU | NVIDIA 8GB VRAM（推理）| NVIDIA RTX 4090D 24GB VRAM |
| 存储 | 50 GB 可用空间 | 500 GB NVMe SSD |
| 网络 | 局域网 | — |

#### 软件要求

| 软件 | 版本 | 说明 |
|------|------|------|
| Linux | Ubuntu 20.04+ / Debian 13+ | 开发与部署环境 |
| CUDA | 12.8+ | GPU 加速必需 |
| cuDNN | 9.x | PyTorch CUDA 后端 |
| Python | 3.13.5 | 实测版本（≥3.10 兼容）|
| uv | 最新版 | 推荐的 Python 包管理工具 |
| RustDesk | 最新版 | 第三方测试远程访问工具 |

---

## 3. 软件架构设计

### 3.1 整体架构图

系统采用分层架构设计，从底层到顶层共五层：

```
┌─────────────────────────────────────────────────────────────────┐
│                    用户交互层（Interface Layer）                   │
│                                                                 │
│   Gradio Web GUI (viz/app.py)    Typer CLI (cli/app.py)        │
│   5个标签页交互界面              命令行工具（detect/preprocess/     │
│                                  info/eval）                    │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│                    业务流水线层（Pipeline Layer）                   │
│                                                                 │
│   infer_full_pipeline.py         run_cnas_eval.py               │
│   完整推理流水线                   CNAS 标准评测                    │
│   SAHI滑窗→全图NMS→              review_fullimage.py            │
│   connect_scratches→评分          交互式标注审核                   │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│                    核心算法层（Algorithm Layer）                   │
│                                                                 │
│  preprocessing/                detection/          scoring/    │
│  ├── pipeline.py               ├── base.py          ├── quantify │
│  ├── background_fusion.py      ├── classical.py     ├── wear_   │
│  ├── registration.py           ├── features.py      │   score   │
│  └── roi_builder.py            ├── params.py        └── report  │
│                                └── rendering.py                 │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│                   机器学习层（ML Layer）                           │
│                                                                 │
│  ml/models.py   ml/train.py    ml/predict.py                   │
│  YOLOv12m 封装  训练脚本接口    推理接口                           │
│                                                                 │
│  scripts/ 训练流水线脚本                                          │
│  train_stage1_baseline.py → step1 → step2 → step3_retrain.py  │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│                    数据层（Data Layer）                            │
│                                                                 │
│  data/loader.py   data/dataset.py   eval/metrics.py            │
│  图像扫描与加载    数据集管理          评估指标计算                   │
│                                                                 │
│  output/tile_dataset/     output/dataset_v2/                   │
│  切片数据集                预处理后全图数据集                        │
└─────────────────────────────────────────────────────────────────┘
```

**设计原则**：
- **分层解耦**：每层仅依赖下层，禁止跨层调用（除用户层可调用任意层）
- **配置驱动**：所有算法参数通过 `PipelineParams` dataclass 统一管理，支持 YAML 覆盖
- **延迟加载**：深度学习模型（`torch`、`ultralytics`）采用懒加载，避免无 GPU 环境启动失败
- **接口标准化**：所有检测器继承 `BaseDetector` 抽象基类，保证可替换性

---

### 3.2 模块划分

#### 3.2.1 预处理模块 `src/darkfield_defects/preprocessing/`

| 文件 | 职责 |
|------|------|
| `pipeline.py` | **预处理主流水线**。暴露 `PreprocessPipeline` 类，封装标定（`calibrate()`）和处理（`process()`）完整流程；定义 `CalibrationResult`、`PipelineResult`、`QualityMetrics` 数据结构；支持批量处理（`process_batch()`）和标定结果持久化/加载。 |
| `background_fusion.py` | **背景融合与离焦模板生成**。实现 `generate_defocused_template()`：对多帧均值背景图进行迭代高斯模糊（多 pass 分解大 sigma），模拟镜片背景的离焦散焦效果，并施加亮度增益系数。 |
| `registration.py` | **图像配准**。实现 `register_to_template()` 函数：以高光 Ring 区域为引导，执行"相位相关粗估平移→ECC 仿射细配→缩放超限回退 Euclidean"的级联配准策略；`apply_warp()` 将 2×3 仿射矩阵应用于图像。 |
| `roi_builder.py` | **ROI 区域构建**。`build_highlight_structure_mask()` 基于中心相对亮度阈值检测高光 Ring 边缘掩码；`build_roi_from_highlight_mask()` 从 Ring 掩码推断镜片有效圆形 ROI 并进行形态学腐蚀收缩。 |
| `brightness_correction.py` | **亮度修正**。`apply_linear_correction()` 实现除法校正（`subtract` 模式）：`corrected = aligned - B_blur`；`finalize_image()` 将结果归一化到 uint8 并掩码 ROI 外区域为 0。 |
| `arc_extraction.py` | 圆弧提取辅助函数，用于 ROI 圆形边界精化。 |
| `circle_fitting.py` | 最小二乘圆拟合，从连通域边界估计镜片圆心和半径。 |

#### 3.2.2 检测模块 `src/darkfield_defects/detection/`

| 文件 | 职责 |
|------|------|
| `base.py` | **抽象基类与数据结构**。定义 `DefectType` 枚举（SCRATCH/SPOT/DAMAGE/CRASH）及 YOLO 3 类映射；`DefectInstance` 缺陷实例数据类（骨架坐标、面积、长度、散射强度、所属视区等）；`DetectionResult` 结果容器；`BaseDetector` 抽象接口。 |
| `classical.py` | **经典算法检测器**。`ClassicalDetector` 实现完整的传统计算机视觉检测流水线：gamma 增强→CLAHE→Frangi+TopHat+亮度三通道候选图合成→Otsu 阈值→形态学开运算→连通域分析→实例分类（长宽比/面积/圆度）→划痕断裂合并→密集区检测。 |
| `features.py` | **特征提取函数库**。Frangi 滤波封装、TopHat 变换、骨架化（`skeletonize`）、端点/交叉点检测、旋转包围盒计算、散射强度估计等底层图像特征函数。 |
| `params.py` | **参数管理**。定义 `PreprocessParams`、`DetectionParams`、`ScoringParams`、`OutputParams`、`ROIPipelineParams`、`PipelineParams` 等 dataclass；`load_params()` 从 YAML 文件加载并覆盖默认值。 |
| `preprocess.py` | 检测器内部预处理辅助（gamma 校正、CLAHE 封装）。 |
| `rendering.py` | **结果可视化与导出**。`render_overlay()` 在原图上绘制缺陷边框与标签；`save_detection_output()` 保存叠加图、二值掩码；`export_coco()` 导出 COCO 格式标注 JSON。 |

#### 3.2.3 评分模块 `src/darkfield_defects/scoring/`

| 文件 | 职责 |
|------|------|
| `quantify.py` | **磨损量化**。定义 `WearMetrics` 数据类（13 个物理量指标）；`compute_wear_metrics()` 从 `DetectionResult` 汇总各视区（中心/过渡/边缘）的划痕长度、条数、面积、密度、散射强度；`_count_crossings()` 统计骨架交叉点数量。 |
| `wear_score.py` | **磨损评分**。`compute_wear_score()` 对各物理量应用对数饱和变换后加权求和得到 WearScore；推导眩光指数（`_compute_glare_index()`）和雾化指数（`_compute_haze_index()`）；映射等级 A/B/C/D；`_generate_conclusion()` 生成可解释中文结论。 |
| `report.py` | **报告生成**。`generate_json_report()` 输出结构化 JSON；`generate_html_report()` 生成暗色主题 HTML 报告（含贡献因素横条图、场景风险表、结论区）。 |

#### 3.2.4 机器学习模块 `src/darkfield_defects/ml/`

| 文件 | 职责 |
|------|------|
| `models.py` | YOLOv12m 模型封装，处理权重加载和设备选择。 |
| `train.py` | 训练接口封装，调用 Ultralytics `YOLO.train()` API。 |
| `predict.py` | 推理接口封装，`load_model()` 返回模型和设备；批量推理辅助函数。 |

#### 3.2.5 GUI 模块 `src/darkfield_defects/viz/app.py`

基于 Gradio 构建的 Web 交互界面，详见第 6.2 节。

#### 3.2.6 CLI 模块 `src/darkfield_defects/cli/app.py`

基于 Typer 构建的命令行工具，详见第 6.1 节。

#### 3.2.7 数据与评估模块

| 模块 | 职责 |
|------|------|
| `data/loader.py` | `load_image()` 加载单张图像；`scan_directory()` 按命名规范（`bg-*.png`、左右镜片标识）扫描目录，返回 `ImageInfo` 列表。 |
| `data/dataset.py` | 数据集管理，支持 YOLO 格式数据集的构建与验证。 |
| `eval/metrics.py` | `compute_segmentation_metrics()` 像素级 P/R/F1/IoU；`compute_instance_metrics()` 实例级命中率/漏检率/虚警率。 |

---

### 3.3 数据流设计

从原始图像到最终报告的完整数据流如下：

```
【阶段 0：标定（一次性）】
  多帧背景图 (bg-*.png)
    → stack → 均值 → B_avg
    → 多 pass 高斯模糊(σ=30) → B_blur（离焦模板）
    → build_highlight_structure_mask() → ring_mask（高光环掩码）
    → build_roi_from_highlight_mask() → roi_mask（有效区域掩码）
    → 计算 ROI 内亮度统计 → correction_ref_median, ref_mad
    → CalibrationResult 持久化（.npy + calibration.json）

【阶段 1：单帧预处理】
  目标图像 (*.png, uint8)
    → 降采样 × 0.75 → 预模糊(σ=15)
    → 相位相关(ring_mask 引导) → 粗平移量 (dx_coarse, dy_coarse)
    → ECC 仿射细配(wide_band 掩码) → warp_matrix (2×3)
    → 缩放检查(>5% → 回退 Euclidean)
    → apply_warp() → aligned (float64)
    → apply_linear_correction(mode=subtract) → corrected
    → finalize_image(roi_mask) → image_final (uint8)
    → QualityMetrics（ECC 分数、配准幅度、背景 std）
    → PipelineResult

【阶段 2：切片生成（训练/推理）】
  image_final (4096×3000)
    → iter_tiles(stride=480, tile=640) → (y0, x0) 坐标序列
    → 裁切每个 640×640 tile
    → （训练用）gamma(0.55)+CLAHE → 增强 tile（仅驱动检测）
    → ClassicalDetector.detect() → DefectInstance 列表
    → YOLO 格式标注：class cx cy w h（坐标基于原始 tile）
    → 保存 images/{train,val}/*.jpg + labels/{train,val}/*.txt

【阶段 3：模型训练（离线）】
  tile_dataset/defects.yaml
    → Stage1: YOLO(yolo12m.pt).train(epochs=80, batch=32, imgsz=640)
    → stage1/weights/best.pt
    → Step1: 自动标注清洗（噪声小框、边缘截断、满切片大框、重叠去重）
    → Step2: 多尺度伪标签（640/960/1280 推理→一致性筛选→与GT融合）
    → Stage3: YOLO(stage1/best.pt).train(微调, lr0=0.002)
    → stage2_cleaned/weights/best.pt（最优权重）

【阶段 4：SAHI 全图推理】
  image_final + best.pt
    → generate_tile_positions(stride=480) → tile 坐标列表
    → batch 推理(batch=16, conf=0.20) → 每 tile 的 bbox 列表
    → tile_boxes_to_fullimage() → 全图像素坐标
    → nms_ios(ios_thresh=0.35) → 去重后检测框
    → connect_scratches(gap=100px, angle=30°) → 合并共线划痕碎片
    → 分类统计（scratch/spot/critical 数量与位置）

【阶段 5：磨损评分】
  全图检测框列表
    → BboxDefect 适配器（面积→长度代理，短边→宽度代理）
    → compute_wear_metrics() → WearMetrics（L/N/A/D/S 指标）
    → compute_wear_score() → WearAssessment（score/grade/结论）

【阶段 6：报告输出】
  WearMetrics + WearAssessment
    → generate_json_report() → report.json
    → generate_html_report() → report.html
    → render_overlay() → detection.png
```

---

### 3.4 目录结构说明

```
/home/bm/Dev/Microlens_DF/
│
├── src/darkfield_defects/          # Python 包主目录
│   ├── __init__.py
│   ├── exceptions.py               # 自定义异常（ConfigurationError 等）
│   ├── logging.py                  # 日志配置（get_logger 工厂函数）
│   ├── preprocessing/              # 预处理模块
│   ├── detection/                  # 经典检测模块
│   ├── scoring/                    # 磨损评分模块
│   ├── ml/                         # 机器学习模块
│   ├── data/                       # 数据加载模块
│   ├── eval/                       # 评估指标模块
│   ├── viz/                        # Gradio GUI
│   └── cli/                        # Typer CLI
│
├── scripts/                        # 独立运行脚本（训练/推理/工具）
│   ├── build_tile_dataset.py       # 切图 + 半自动标注
│   ├── train_stage1_baseline.py    # Stage1 基线训练
│   ├── step1_label_cleanup.py      # 标注清洗
│   ├── step2_pseudo_labels.py      # 伪标签生成
│   ├── step3_retrain.py            # Stage3 重训练
│   ├── infer_full_pipeline.py      # 完整推理流水线
│   ├── run_cnas_eval.py            # CNAS 标准评测
│   ├── review_fullimage.py         # 交互式标注审核
│   ├── fullimage_utils.py          # SAHI/NMS 工具函数库
│   └── ...（其他调试/辅助脚本）
│
├── configs/
│   ├── detect_default.yaml         # 默认检测参数
│   └── detect_tile.yaml            # 切片模式专用参数
│
├── output/                         # 运行时输出（不含于版本控制）
│   ├── dataset_v2/                 # 预处理后全图（247张PNG + roi_mask）
│   ├── tile_dataset/               # 切片数据集（~9400 tiles）
│   ├── training/                   # 训练输出（权重 + 日志）
│   │   ├── stage1/weights/best.pt
│   │   └── stage2_cleaned/weights/best.pt   ← CNAS 认证权重
│   ├── audit/                      # 人工审核相关
│   │   ├── test_set_v1.json        # 独立测试集（20张图）
│   │   └── predictions/            # 模型预测结果
│   ├── pipeline_results/           # infer_full_pipeline.py 输出
│   └── cnas_eval/                  # run_cnas_eval.py 输出
│
├── doc/                            # 项目文档
├── tests/                          # 单元测试
├── pyproject.toml                  # 包配置与依赖声明
├── uv.lock                         # 锁定依赖版本
├── yolo12m.pt                      # YOLOv12m COCO 预训练权重
└── .venv/                          # 虚拟环境
```

---

## 4. 核心算法说明

### 4.1 预处理流水线算法

#### 4.1.1 背景平均融合

**目标**：将多帧背景图（无样品）融合为稳定的参考背景，消除单帧随机噪声。

**算法流程**：

1. 加载目录内所有 `bg-*.png` 文件（灰度，uint8）
2. 转换为 float64，沿帧维度堆叠为 (N, H, W) 张量
3. 逐像素均值：`B_avg = np.mean(stack, axis=0)`
4. 调用 `generate_defocused_template(B_avg, sigma=30, per_pass_sigma=6, brightness_gain=1.08)`

**离焦模板生成**（`background_fusion.py`）：

为模拟背景的"低频成分"（去除高频噪声），对 B_avg 进行多次小 sigma 高斯模糊，等效于大 sigma 模糊：

```
n_pass = ceil((σ_target / σ_per_pass)²)  = ceil((30/6)²) = 25 次
σ_each = σ_target / sqrt(n_pass)         = 30 / sqrt(25) = 6.0
```

循环调用 `cv2.GaussianBlur(B_blur, (0,0), σ_each)` 共 25 次，最后乘以亮度增益 1.08：

```
B_blur = clip(B_blur × 1.08, 0, 255)
```

**设计理由**：单次大 sigma 高斯模糊需要极大的卷积核（σ=30 对应 ks=181），计算慢且边界效应显著；分多次小核模糊等效精度更好，且每次核尺寸仅为 37×37 (σ=6 时)。

#### 4.1.2 高光 Ring 检测

**目标**：在离焦背景模板 B_blur 中检测暗场照明形成的高光环形区域，作为图像配准的锚定特征。

**算法**（`roi_builder.py:build_highlight_structure_mask()`）：

1. **中心亮度采样**：取图像中心 40%×40% 区域（`center_sample_ratio=0.40`）的像素值分布
2. **相对阈值计算**：`threshold = center_level × center_ratio = center_level × 1.25`
   - `center_level` 为中心采样区的中位数亮度
   - 相对阈值确保阈值自适应图像整体亮度，无需手动调参
3. **二值化**：`ring_mask = B_blur > threshold`（高于阈值的区域为高光 Ring）
4. **形态学处理**：膨胀扩展 `edge_expand_px=25` 像素，填充 Ring 内细碎空洞
5. **步骤记录**：返回中间步骤掩码供调试可视化

#### 4.1.3 ECC 图像配准

**目标**：将目标图像对齐到背景模板坐标系，补偿采集时镜片位置微小偏移（平移、旋转、轻微缩放）。

**级联配准策略**（`registration.py:register_to_template()`）：

```
阶段 1 — 粗估平移（相位相关）
  输入：预模糊(σ=15) 的 src 和 template，wide_band 掩码（ring ±100px）
  方法：cv2.phaseCorrelate()
  输出：粗估平移量 (dx_coarse, dy_coarse)

阶段 2 — 仿射精配（ECC，以相位相关结果为初值）
  方法：cv2.findTransformECC(motion=MOTION_AFFINE,
        criteria=(MAX_ITER=200, eps=1e-6), inputMask=wide_band)
  输入图像归一化到 [0,1]
  输出：2×3 仿射矩阵，ECC 相关分数

阶段 3 — 低分回退（若 ECC < 0.75）
  从 identity 矩阵重新初始化，去掉掩码约束重试 ECC
  选择分数更高的结果

阶段 4 — 缩放幅度检查（若 |scale-1| > 5%）
  回退到 MOTION_EUCLIDEAN（纯旋转+平移，无缩放自由度）

阶段 5 — 坐标系转换
  降采样坐标系 → 全分辨率（平移量 ÷ scale）
  求仿射矩阵逆（bg→src 转换为 src→ref）
```

**预模糊的作用**：对 src 和 template 均施加 σ=15 的预模糊，平滑划痕/噪声的局部高频干扰，提升 ECC 对低频形态（Ring 边缘）的收敛稳定性。

**降采样的作用**：配准在 75% 分辨率（约 3072×2250）下进行，减少计算量约 44%，平移量事后按比例还原。

#### 4.1.4 背景减除（除法校正）

**模式**：`brightness_mode = "subtract"`（当前默认模式）

```python
corrected = aligned - B_blur    # 差值保留高频缺陷细节
corrected = clip(corrected + 128, 0, 255)  # 偏置到中灰，保留正负偏差
```

**固定增益设计**：所有目标图像使用固定的 `gain=1.0, bias=0.0`，不进行 per-image 亮度估计。原因：per-image 估计会被划痕区域的强亮度拉偏，导致背景减除不充分。固定增益依赖标定时得到的 `correction_ref_median` 确保模板亮度与目标图像匹配。

---

### 4.2 切图与标注生成

#### 4.2.1 切片策略

**参数**：
- 切片尺寸：640×640 px
- 滑窗步长（Stride）：480 px
- 重叠（Overlap）：160 px（注：`build_tile_dataset.py` 中 `OVERLAP=80`，`infer_full_pipeline.py` 中 `STRIDE=480`，两者 overlap 不同，分别为 80px 和 160px，推理时用 160px 更大 overlap）
- ROI 覆盖率过滤：tile 内 ROI 面积 < 20% 时跳过，不生成标注

**边界处理**：最后一个 tile 的位置向左/上补齐，确保完全覆盖图像边缘，不留盲区。

#### 4.2.2 三通道候选图生成

经典检测器在每个 tile 上合成三通道候选图：

| 通道 | 算法 | 参数 | 检测目标 |
|------|------|------|---------|
| Frangi | Hessian 管状滤波 | σ∈{1,2,3,5}，α=β=0.5，γ=15 | 线状划痕 |
| TopHat | 形态学白顶帽 | 核尺寸 15×15 矩形 | 点状斑点、短划痕 |
| 亮度 | gamma(0.55)+CLAHE | clip=3.0, tile=8×8 | 全局增强，辅助发现暗弱缺陷 |

三通道取最大值融合作为缺陷候选图，再进行阈值分割。

#### 4.2.3 Otsu 阈值与形态学后处理

1. **Otsu 阈值**：自动确定二值化阈值
   - `otsu_floor = 0.08`（tile 模式）：若 Otsu 结果低于此值，强制提升到 0.08，防止纯暗 tile 无双峰分布时产生极低阈值导致全图误检
2. **形态学开运算**：3×3 椭圆核，去除孤立噪声点
3. **连通域分析**：`cv2.connectedComponentsWithStats()`
4. **缺陷分类**（基于连通域特征）：
   - 长宽比 > 3.0 且长度 > 30px → SCRATCH
   - 面积 > 2000px 且密度 > 0.25 → CRASH（临界缺陷）
   - 否则 → SPOT（斑点）
5. **YOLO 格式标注**：DAMAGE/CRASH → critical(2)，输出 `class cx cy w h`（归一化到 [0,1]）

---

### 4.3 YOLOv12m 目标检测

#### 4.3.1 模型架构

**YOLOv12m** 是 Ultralytics YOLO 系列第 12 代 medium 规格变体，核心创新为 **Area Attention 机制**：将注意力计算限制在垂直或水平长条状局部区域内，降低全局注意力的 O(n²) 计算复杂度，同时保留对细长目标（划痕）的感受野优势。

**输入规格**：
- 图像尺寸：640×640 px
- 通道数：3（灰度图通过 `img.convert("RGB")` 或 OpenCV 通道复制扩展为 3 通道）
- 归一化：[0,255] → [0,1]

**检测头**：3 个尺度检测头
- P3（stride=8）：80×80 特征图，适合小目标（spot，典型尺寸 10-30px）
- P4（stride=16）：40×40 特征图，适合中目标（短划痕，critical）
- P5（stride=32）：20×20 特征图，适合大目标（长划痕跨度，大 critical）

#### 4.3.2 损失函数

训练使用 Ultralytics 标准损失组合：

| 损失项 | 说明 | 权重 |
|--------|------|------|
| Varifocal Loss | 分类损失，正样本用预测 IoU 质量加权，负样本降权 | `cls=0.5` |
| DFL（Distribution Focal Loss）| 边框回归，将连续坐标建模为离散概率分布 | `dfl=1.5` |
| CIoU Loss | 考虑中心距离和长宽比的 IoU 损失 | `box=7.5` |

#### 4.3.3 三阶段训练策略

**Stage 1：基线训练**

| 参数 | 值 |
|------|---|
| 预训练权重 | `yolo12m.pt`（COCO 预训练）|
| 数据集 | 原始半自动标注，112,929 框 |
| Epochs | 80（最优在 epoch 22）|
| Batch size | 32 |
| 图像尺寸 | 640 |
| 优化器 | SGD，lr0=0.01，momentum=0.937 |
| 数据增强 | mosaic=1.0, flipud=0.5, fliplr=0.5, hsv_v=0.4 |
| 结果 | mAP@0.5 = 0.5275，train/val gap = 0.73（过拟合）|

**Stage 2（Step1+Step2+Step3）：清洗+伪标签重训练**

- Step1 自动清洗：移除 15,338 个噪声框（噪声小框、边缘截断残留、满切片大框、高重叠冗余）
- Step2 伪标签：多尺度推理（640/960/1280）+ flip TTA → 一致性筛选（≥2 尺度同意，avg_conf ≥ 0.35）→ 补充 8,613 框
- Step3 重训练（实际标注为 Stage 3 即当前最优）：

| 参数 | 值 |
|------|---|
| 初始权重 | Stage1 best.pt |
| 数据集 | 清洗+伪标签，92,358 框 |
| Epochs | 80（最优在 epoch 56）|
| Batch size | 32 |
| 学习率 | lr0=0.002（比 Stage1 低 5×，微调策略）|
| 结果 | mAP@0.5 = 0.6765，train/val gap = 0.037（泛化良好）|

---

### 4.4 SAHI 推理流水线

#### 4.4.1 滑窗推理

对 4096×3000 全图进行切片推理：

```python
# fullimage_utils.py
TILE_SIZE = 640
STRIDE    = 480    # 对应 overlap = 160px

def generate_tile_positions(img_h, img_w):
    """生成覆盖全图的所有 (y0, x0) 起始坐标"""
    # 边界处理：末尾 tile 向回退，确保无盲区
```

典型 4096×3000 图像：约 (ceil(3000/480)+1) × (ceil(4096/480)+1) ≈ 8×9 = 72 个 tile

**批量推理**：tile 按 `batch_size=16` 分批送入模型，推理速度约 4.4ms/tile（RTX 4090D）。

#### 4.4.2 IOS NMS（Intersection over Smaller）

全图拼合后存在大量跨切片重叠检测框，标准 IoU NMS 对细长划痕效果差（两个端点相邻的短划痕碎片 IoU 极小但实为同一划痕）。本系统使用 IOS（Intersection over Smaller）：

```python
def nms_ios(boxes, ios_thresh=0.35):
    """
    IOS = 交集面积 / min(box1面积, box2面积)
    当一个框大部分被另一个框覆盖时（IOS高），抑制置信度较低的框
    适合处理：长划痕被切片截断后的碎片重叠
    """
```

IOS 阈值 0.35：两框中较小框超过 35% 面积被覆盖时触发抑制。

#### 4.4.3 connect_scratches 后处理

将多个切片中被边界截断的划痕碎片重新连接：

```python
def connect_scratches(boxes, max_gap=100, max_angle_diff=30):
    """
    对 scratch 类检测框：
    1. 计算每个框的主方向向量（通过旋转包围盒或长轴方向）
    2. 遍历框对：若端点距离 < max_gap 且方向差 < max_angle_diff
    3. 合并为外接矩形框，更新为更高置信度
    """
```

---

### 4.5 WearMetrics 与 WearScore

#### 4.5.1 BboxDefect 适配器

全图 YOLO 推理输出的是 bbox（边框），而 `WearMetrics` 需要骨架长度、面积等物理量。`BboxDefect` 数据类提供代理量化：

```python
@dataclass
class BboxDefect:
    cls: int           # 0=scratch, 1=spot, 2=critical
    x1, y1, x2, y2    # 全图像素坐标
    conf: float

    @property
    def length_px(self):
        # 以较长边作为长度代理
        return max(x2-x1, y2-y1)

    @property
    def area_px(self):
        return (x2-x1) * (y2-y1)

    @property
    def zone(self):
        # 根据框中心到 ROI 中心的相对距离确定视区
        # 中心 30% 半径内 → "center"
        # 30%-60% → "transition"
        # >60% → "edge"
```

**局限性**：bbox 长度代理对于碎片化划痕（连接前的短碎片）会严重低估真实骨架长度；面积代理包含了大量背景像素。详见第 9.2 节。

#### 4.5.2 WearScore 计算公式

```
WearScore = w₁·sat_log(L_center, 8.0)
          + w₂·sat_log(L_total, 5.0)
          + w₃·sat_log(N_total, 20.0)
          + w₄·sat_log(A_total, 3.0)
          + w₅·sat_log(S_scatter×10, 10.0)
```

其中对数饱和函数：
```
sat_log(x, scale) = min(scale × ln(1+x), 100)
```

默认权重（来自 `ScoringParams`）：

| 因子 | 变量 | 权重 | 说明 |
|------|------|------|------|
| 中心区划痕长度 | L_center | **0.35** | 最高权重，中心视区最影响视力 |
| 总划痕长度 | L_total | 0.15 | |
| 划痕总条数 | N_total | 0.10 | |
| 划痕总面积 | A_total | 0.15 | |
| 散射强度 | S_scatter | 0.25 | 与眩光直接相关 |

#### 4.5.3 等级映射

| 等级 | WearScore 范围 | 中文描述 |
|------|--------------|---------|
| A | < 15 | 极佳 — 无划痕或仅边缘微小划痕 |
| B | 15 ~ 40 | 良好 — 核心区外少量划痕，特定光线下偶有感知 |
| C | 40 ~ 70 | 警告 — 核心区有轻/中度划痕，夜间驾驶谨慎 |
| D | ≥ 70 | 报废 — 核心区严重磨损，强烈建议更换 |

> **注意**：当前 WearScore 参数基于定性调参，未经过大规模标准样品的统计标定。等级边界数值（15/40/70）为工程估计值，应在量产前通过标准磨损样品集重新标定。

#### 4.5.4 场景风险指数

**眩光指数**（Night Glare Index）：
```
glare_index = sat_log(L_center × S_scatter_center × 0.01, 15.0)
```
反映夜间行车强光下的眩光风险，与中心划痕长度和散射强度乘积正相关。

**雾化指数**（Haze Index）：
```
haze_index = sat_log(D_density × 1e6 + n_crossings × 5, 10.0)
```
反映细密划痕引起的整体透光率下降，与密度和交叉点数正相关。

---

## 5. 数据设计

### 5.1 数据集组织结构

```
output/tile_dataset/
├── images/
│   ├── train/          # 训练集切片（.jpg 格式）
│   │   └── {stem}_{row}_{col}.jpg    # 命名示例：13r_2_4.jpg
│   └── val/            # 验证集切片
│       └── {stem}_{row}_{col}.jpg
├── labels/
│   ├── train/          # 对应 YOLO 格式标注
│   │   └── {stem}_{row}_{col}.txt
│   └── val/
│       └── {stem}_{row}_{col}.txt
├── overlays/           # 标注叠加图（审核用）
│   └── {stem}_{row}_{col}_overlay.png
├── defects.yaml        # YOLO 数据集配置
├── defects_augmented.yaml  # 含 damage 过采样的配置
├── tile_index.csv      # 全量切片索引（原图stem、split、tile坐标）
└── cleanup_report.json # 标注清洗报告
```

**数据量**（Stage 3 训练集）：
- 总切片数：约 9,400 张（train: ~7,550, val: ~1,850）
- 总标注框数（清洗后）：92,358 框
- 原始图像数：247 张（全图）

---

### 5.2 标注格式

使用 **YOLO 格式**：每个 `.txt` 文件对应一张切片图像，每行一个目标：

```
<class_id> <cx> <cy> <w> <h>
```

- `class_id`：整数，0/1/2
- `cx, cy`：目标中心坐标，归一化到 [0,1]（相对于切片宽/高）
- `w, h`：目标宽高，归一化到 [0,1]
- 各字段空格分隔，精度建议 6 位小数

**示例**（`13r_2_4.txt`）：
```
0 0.432150 0.215300 0.234500 0.018700
1 0.671200 0.550100 0.045600 0.043200
2 0.125000 0.800000 0.234000 0.187000
```

无目标切片对应空 `.txt` 文件（0 字节）。

---

### 5.3 类别定义

| class_id | 名称 | 内部 DefectType | 物理特征 | 典型尺寸（tile 内）|
|----------|------|----------------|---------|----------------|
| 0 | scratch | SCRATCH | 细长线状散射，长宽比 > 3 | 长 20-600px，宽 2-15px |
| 1 | spot | SPOT | 近圆形点状散射 | 直径 5-50px |
| 2 | critical | DAMAGE + CRASH | 大面积不规则高亮 | 面积 > 500px²，尺寸 20-400px |

**合并原因**：DAMAGE 类在全数据集中仅 46 实例（占 0.045%），样本极度不平衡，无法独立训练；DAMAGE 和 CRASH 在切片级视觉特征（大片高亮区域）上高度相似，合并为 critical 后两类共享特征学习，提升检测稳定性。

---

### 5.4 关键配置文件说明

#### `configs/defects.yaml`（数据集配置，训练时自动生成）

```yaml
path: /home/bm/Dev/Microlens_DF/output/tile_dataset
train: images/train
val: images/val
nc: 3
names: ['scratch', 'spot', 'critical']
```

#### `configs/detect_tile.yaml`（切片模式检测参数）

关键差异（相比 `detect_default.yaml`）：

```yaml
detection:
  otsu_floor: 0.08         # 防止纯暗 tile 极低阈值误检
  min_area: 120            # 比全图模式更严格（50→120）
  min_length: 30.0         # 比全图模式更严格（20→30）
  enhance_gamma: 0.55      # 比全图模式保守（0.4→0.55）
  density_kernel_ratio: 0.15  # tile 更小，核比例放大
```

#### `configs/detect_default.yaml`（默认全图参数）

```yaml
detection:
  frangi_sigmas: [1.0, 2.0, 3.0, 5.0]
  otsu_floor: 0.0
  min_area: 50
  min_length: 20.0
  enhance_gamma: 0.4
```

---

### 5.5 重要数据文件说明

| 文件路径 | 说明 |
|---------|------|
| `output/audit/test_set_v1.json` | CNAS 独立测试集定义文件。包含 20 张图的 stem 列表，格式：`{"images": [{"stem": "13r", "split": "val"}, ...]}`。已排除 ljj 异型样本，确保测试图像与训练集不重叠。|
| `output/tile_dataset/cleanup_report.json` | 标注清洗报告。记录每轮清洗的删除框数量、删除原因分布（small_box/edge_truncation/full_tile_scratch/overlap_dedup）、清洗前后各类别框数统计。|
| `output/tile_dataset/tile_index.csv` | 切片索引文件。每行：`stem,split,row,col,tile_path,has_label,n_boxes`，用于追溯每个切片来自哪张原图、哪个位置，是数据集可溯源性的关键文件。|
| `output/training/stage2_cleaned/args.yaml` | Stage3 训练超参数快照（Ultralytics 自动保存）。记录训练时的全部参数，确保结果可复现。|
| `output/training/stage2_cleaned/weights/best.pt` | CNAS 认证最优权重文件。epoch 56 时 mAP@0.5 最优。|

---

## 6. 接口设计

### 6.1 CLI 接口（Typer Commands）

入口点：`darkfield-defects`（由 `pyproject.toml` 定义，安装后可直接调用）

#### `darkfield-defects detect`

```
检测镜片划痕并生成评估报告

参数:
  INPUT_PATH              输入图像或目录路径（位置参数，必须）

选项:
  --output, -o TEXT       输出目录（默认 ./output）
  --config, -c TEXT       YAML 配置文件路径（可选）
  --limit, -n INT         最大处理图像数（可选）
  --no-report             不生成 HTML 报告
  --calibration, --cal TEXT  标定结果目录（必须）
```

**处理流程**：加载标定 → 遍历图像 → `PreprocessPipeline.process()` → `ClassicalDetector.detect()` → `compute_wear_metrics()` → `compute_wear_score()` → 生成报告 → 导出 COCO JSON。

#### `darkfield-defects preprocess`

```
执行模板背景预处理并导出结果

参数:
  INPUT_PATH              输入图像路径

选项:
  --output, -o TEXT       输出目录（默认 ./output/preprocess）
  --config, -c TEXT       YAML 配置文件路径
  --calibration, --cal TEXT  标定结果目录（必须）
```

输出：`{stem}_original.png`、`{stem}_corrected.png`、`{stem}_roi.png`、标准输出打印 ECC 分数和修正参数。

#### `darkfield-defects info`

```
显示数据集统计信息

参数:
  INPUT_PATH              数据目录路径
```

输出：图像类型分布（背景/目标）、左右镜片分布、批次分布。

#### `darkfield-defects eval`

```
评估检测结果：像素级 + 实例级指标

参数:
  PRED_DIR                预测掩码目录（PNG 二值图）
  GT_DIR                  Ground Truth 掩码目录

选项:
  --output, -o TEXT       输出 JSON 路径
  --iou-threshold FLOAT   实例匹配 IoU 阈值（默认 0.3）
```

输出：每对文件的 Precision/Recall/F1/IoU（像素级）及命中率/漏检率/虚警率（实例级）。

---

### 6.2 Gradio GUI 接口（5 个标签页）

启动方式：
```python
from darkfield_defects.viz.app import build_app
app = build_app()
app.launch(server_port=7860)
```

#### Tab 1：单图检测（Single Detection）

**功能**：上传单张镜片图像，执行经典算法检测，展示结果。

**输入组件**：
- 图像上传（`gr.Image`，RGB numpy array）
- 背景图像上传（可选）
- 配置文件路径文本框

**输出组件**：
- 检测叠加图（`gr.Image`）
- 缺陷统计表格（`gr.DataFrame`，列：ID/类型/视区/长度/面积/平均宽度/散射强度/圆度/细长比）
- 摘要文本（检测到 N 条 scratch、M 个 spot）

#### Tab 2：批量分析（Batch Analysis）

**功能**：指定图像目录，批量运行检测，汇总统计。

**输入组件**：
- 目录路径文本框
- 最大处理数滑块
- 配置文件路径

**输出组件**：
- 汇总统计表（每张图的检测结果）
- 整体统计（总缺陷数分布）

#### Tab 3：对比审核（Compare Review）

**功能**：经典算法 vs 机器学习模型的并排对比视图。

**输入组件**：
- 图像上传
- ML 模型权重路径文本框

**输出组件**：
- 左：经典检测叠加图
- 右：ML 检测叠加图
- 各自的统计表格

#### Tab 4：磨损报告（Wear Report）

**功能**：基于检测结果计算 WearScore 并展示可视化报告。

**输入组件**：
- 图像上传
- 是否使用背景开关

**输出组件**：
- WearScore 数值展示（`gr.Number`）
- 等级标识（A/B/C/D，带颜色编码）
- 贡献因素条形图（`gr.BarPlot`）
- 眩光/雾化指数
- 可解释结论文本

#### Tab 5：预处理调试（Preprocessing Debug）

> 注意：当前代码注释中描述为 4 个 Tab，但架构设计预留第 5 个 Tab 用于预处理管线可视化调试（展示 B_avg、B_blur、ring_mask、roi_mask 等中间步骤图像）。

---

### 6.3 Python API 接口（主要类和函数签名）

#### 预处理流水线

```python
class PreprocessPipeline:
    def __init__(self,
        template_blur_sigma: float = 30.0,
        template_highlight_center_ratio: float = 1.25,
        max_scale_deviation: float = 0.05,
        ecc_max_iter: int = 200,
        registration_downsample: float = 0.75,
        brightness_mode: str = "subtract",
        ...
    ) -> None: ...

    def calibrate(self,
        bg_dir: str | Path,
        save_dir: str | Path | None = None
    ) -> CalibrationResult: ...

    def load_calibration(self, path: str | Path) -> None: ...

    def process(self,
        image: np.ndarray,   # (H,W) uint8 灰度图
        frame_index: int = -1
    ) -> PipelineResult: ...

    def process_batch(self,
        target_dir: str | Path,
        output_dir: str | Path,
        extensions: tuple[str,...] = ('.png', '.bmp'),
    ) -> list[dict[str, Any]]: ...
```

#### 经典检测器

```python
class ClassicalDetector(BaseDetector):
    def __init__(self,
        detection_params: DetectionParams,
        preprocess_params: PreprocessParams,
        scoring_params: ScoringParams | None = None
    ) -> None: ...

    def detect(self,
        image: np.ndarray,          # (H,W) 灰度图
        background: np.ndarray | None = None,
        roi_mask: np.ndarray | None = None,
        preprocessed_image: np.ndarray | None = None
    ) -> DetectionResult: ...
```

#### 磨损评分

```python
def compute_wear_metrics(
    result: DetectionResult,
    roi_mask: np.ndarray | None = None
) -> WearMetrics: ...

def compute_wear_score(
    metrics: WearMetrics,
    params: ScoringParams | None = None
) -> WearAssessment: ...
```

#### 报告生成

```python
def generate_json_report(
    filename: str,
    metrics: WearMetrics,
    assessment: WearAssessment,
    output_path: str | Path,
    extra: dict[str, Any] | None = None
) -> Path: ...

def generate_html_report(
    filename: str,
    metrics: WearMetrics,
    assessment: WearAssessment,
    output_path: str | Path,
    overlay_path: str | None = None
) -> Path: ...
```

#### 参数加载

```python
def load_params(config_path: str | Path | None = None) -> PipelineParams: ...
# 返回含 preprocess/detection/scoring/output/roi_pipeline 五个子参数组的 PipelineParams
```

---

### 6.4 CNAS 标准评测接口

**脚本**：`scripts/run_cnas_eval.py`

**评测流程**：

```python
# 1. 加载测试集定义
stems = load_test_set("output/audit/test_set_v1.json")
# → 返回 20 个 stem 字符串列表

# 2. 收集对应的 val 切片
tile_paths = collect_tile_paths(stems)
# → 从 output/tile_dataset/images/val/ 中按文件名前缀匹配
# → 约 860 个切片

# 3. 构建临时 val 数据集
yaml_path = build_val_dataset(tile_paths, save_dir)
# → 生成临时 dataset.yaml + 符号链接

# 4. 调用 Ultralytics 官方评测
model = YOLO(weights_path)
results = model.val(
    data=yaml_path,
    conf=0.001,      # 极低阈值，由 PR 曲线确定 AP
    iou=0.6,         # IoU 匹配阈值
    save_dir=save_dir,
    verbose=False
)

# 5. 格式化输出 CNAS 报告
print_cnas_report(results, pass_threshold=0.60)
# → 输出 mAP@0.5、per-class AP、PASS/FAIL 判定
# → 保存 output/cnas_eval/cnas_eval_results.json
```

**标准测试命令**：
```bash
cd /home/bm/Dev/Microlens_DF
python scripts/run_cnas_eval.py
# 可选：
# --weights  output/training/stage2_cleaned/weights/best.pt
# --test-set output/audit/test_set_v1.json
# --save-dir output/cnas_eval/
```

**输出文件**：
- `output/cnas_eval/cnas_eval_results.json`：完整指标 JSON（mAP、per-class AP/P/R、推理速度）
- `output/cnas_eval/ultralytics_output/`：混淆矩阵图（`confusion_matrix.png`）、PR 曲线（`PR_curve.png`）、F1 曲线等

---

## 7. 部署与运行

### 7.1 环境依赖

| 组件 | 版本 | 验证命令 |
|------|------|---------|
| Python | 3.13.5 | `python --version` |
| PyTorch | 2.10.0+cu128 | `python -c "import torch; print(torch.__version__)"` |
| CUDA | 12.8 | `nvcc --version` |
| Ultralytics | 8.4.24 | `python -c "import ultralytics; print(ultralytics.__version__)"` |
| OpenCV | ≥4.8（含 contrib）| `python -c "import cv2; print(cv2.__version__)"` |
| Gradio | ≥4.0 | `python -c "import gradio; print(gradio.__version__)"` |

> **重要**：Ultralytics 版本必须固定为 **8.4.24**。不同版本的 YOLO 权重格式和训练参数可能不兼容，升级前需完整重新验证 mAP 指标。

---

### 7.2 安装步骤

#### 方式 1：使用 uv（推荐）

```bash
# 1. 克隆项目
git clone <repo_url> /home/bm/Dev/Microlens_DF
cd /home/bm/Dev/Microlens_DF

# 2. 创建虚拟环境（uv 自动从 uv.lock 安装）
uv sync

# 3. 安装 ML 和 viz 可选依赖
uv pip install torch==2.10.0+cu128 torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install ultralytics==8.4.24
uv pip install gradio>=4.0

# 4. 以可编辑模式安装本包
uv pip install -e .

# 5. 验证安装
python -c "from darkfield_defects.preprocessing.pipeline import PreprocessPipeline; print('OK')"
darkfield-defects --help
```

#### 方式 2：使用 pip + venv

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pip install torch==2.10.0+cu128 torchvision --index-url https://download.pytorch.org/whl/cu128
pip install ultralytics==8.4.24 gradio>=4.0
```

---

### 7.3 关键路径配置

系统使用硬编码的项目相对路径（基于 `Path(__file__).parent.parent` 推导），关键路径如下：

| 路径变量 | 值 | 说明 |
|---------|---|------|
| `PROJECT_ROOT` | `/home/bm/Dev/Microlens_DF` | 项目根目录，脚本自动推导 |
| `WEIGHTS_PATH` | `output/training/stage2_cleaned/weights/best.pt` | CNAS 认证权重 |
| `IMAGES_DIR` | `output/dataset_v2/images` | 预处理后全图目录 |
| `TILES_DIR` | `output/tile_dataset/images/val` | 验证集切片目录 |
| `DEFAULT_TEST_SET` | `output/audit/test_set_v1.json` | CNAS 测试集定义 |

**标定结果目录**（首次运行需先标定）：

```bash
# 运行标定（需要有 bg-*.png 背景图）
python - <<'EOF'
from darkfield_defects.preprocessing.pipeline import PreprocessPipeline
p = PreprocessPipeline()
p.calibrate("path/to/bg_images", save_dir="output/calibration")
EOF
```

---

### 7.4 RustDesk 远程测试配置

CNAS 第三方测试通过 RustDesk 远程桌面进行，配置步骤：

**服务器端准备**：

```bash
# 1. 确认 RustDesk 已安装并后台运行
systemctl status rustdesk   # 或检查进程

# 2. 创建演示专用用户（避免暴露主账户）
sudo useradd -m -s /bin/bash demo_user
sudo cp -r /home/bm/Dev/Microlens_DF /home/demo_user/Microlens_DF
sudo chown -R demo_user:demo_user /home/demo_user/Microlens_DF

# 3. 确认防火墙端口（RustDesk 使用 21115-21119 TCP/UDP）
sudo ufw allow 21115:21119/tcp
sudo ufw allow 21115:21119/udp

# 4. 发送 RustDesk ID 给测试工程师
```

**演示步骤顺序**（约 30 分钟）：

1. `(5min)` 展示预处理：打开 `output/dataset_v2/images/` 样本图，运行 `darkfield-defects preprocess`
2. `(5min)` 展示 CNAS 评测：执行 `python scripts/run_cnas_eval.py`，等待约 2 分钟，展示结果
3. `(10min)` 展示全图推理：`python scripts/infer_full_pipeline.py --image output/dataset_v2/images/13r.png`
4. `(10min)` 展示 Gradio GUI：启动界面，上传样本图演示磨损报告

---

## 8. 性能指标与测试结果

### 8.1 检测精度

**测试配置**：
- 测试集：`test_set_v1.json`，20 张图，860 个切片
- 评测框架：Ultralytics `model.val()`
- conf 阈值：0.001（AP 计算）；实际推理使用 conf=0.20
- IoU 匹配阈值：0.6
- 评测时间：2026-03-22

**Stage 3（当前最优，CNAS 认证版本）测试结果**：

| 指标 | 值 |
|------|---|
| **mAP@0.5** | **0.6765** |
| mAP@0.5:0.95 | 0.4395 |
| Precision (conf=0.20) | 0.5944 |
| Recall (conf=0.20) | 0.6411 |

**Per-class 详细结果**：

| 类别 | AP@0.5 | AP@0.5:0.95 | P | R |
|------|--------|-------------|---|---|
| scratch (0) | **0.4525** | 0.2548 | 0.508 | 0.364 |
| spot (1) | **0.8180** | 0.6042 | 0.648 | 0.784 |
| critical (2) | **0.7589** | 0.4592 | 0.626 | 0.773 |
| **均值（mAP）** | **0.6765** | **0.4395** | 0.594 | 0.640 |

**多阶段训练对比**：

| 阶段 | 数据集 | Epochs（最优/总）| mAP@0.5 | train/val gap |
|------|--------|----------------|---------|--------------|
| Stage 1 基线 | 原始 112,929 框 | 22/38 | 0.5275 | 0.73（过拟合）|
| Stage 3 清洗+伪标签 | 92,358 框 | 56/80 | **0.6765** | 0.037（良好）|
| Stage 4（失败）| Step4 重标注 77,171 框 | 11/31（早停）| 0.4728 | — |

**Stage 4 失败原因**：step4 对 GT scratch 执行了 `connect_scratches`，将碎片合并为极长大框（如 600×15px），生成了模型无法预测的不合理 GT，导致 scratch AP 崩溃至 0.166（-59%）。

---

### 8.2 推理速度

**测试环境**：RTX 4090D 24GB VRAM，CUDA 12.8，batch=16

| 阶段 | 速度 | 备注 |
|------|------|------|
| 单 tile 推理 | **~4.4 ms/tile** | 640×640，包含前处理和后处理 |
| 全图推理（4096×3000）| ~320 ms | 约 72 个 tile，batch=16，含 NMS |
| 预处理（ECC 配准）| ~200-400 ms | 含降采样+预模糊+ECC 迭代 |
| 总端到端（单张）| ~600-800 ms | 预处理+推理+评分+报告 |

---

### 8.3 资源消耗

**推理时（batch=1，单张全图）**：

| 资源 | 占用 |
|------|------|
| GPU VRAM | ~2.5 GB |
| CPU RAM | ~1.5 GB |
| GPU 利用率 | ~35%（推理间歇）|

**推理时（batch=16，高吞吐）**：

| 资源 | 占用 |
|------|------|
| GPU VRAM | ~6 GB |
| CPU RAM | ~2 GB |
| GPU 利用率 | ~85% |

---

### 8.4 训练资源

**Stage 3 训练（RTX 4090D，batch=32，epochs=80）**：

| 参数 | 值 |
|------|---|
| 训练时长 | **~126 分钟** |
| GPU VRAM 峰值 | ~22 GB |
| 数据集缓存 | RAM 缓存（需 ~8 GB 内存）|
| 平均每 iter 时间 | ~0.055s |
| 存储空间（权重+日志）| ~500 MB |

---

## 9. 已知限制与待改进项

### 9.1 Scratch AP 偏低（AP@0.5 = 0.4525）

**根本原因**：

1. **细长 bbox 与 IoU 评估的天然不匹配**：长 200px 宽 8px 的划痕框，中心偏移 4px 会导致 IoU 从 0.8 骤降至 0.5（对相同偏移，正方形目标影响仅 ~5%）。这是几何上的固有问题，不能仅通过模型改进解决。

2. **GT 标注质量问题**：Frangi 验证显示 44.3% 的 GT scratch 框内无显著线状结构响应，部分框包含的是背景噪声或极弱细划痕（可能处于人眼辨别边界）。

3. **切片截断**：长划痕被切片边界截断成多个碎片，每个碎片的 GT 框是局部线段而非完整划痕，增加了 recall 计算的不确定性。

**待改进方案**：
- 使用旋转框（Oriented Bounding Box，OBB）替代轴对齐 bbox，理论提升 mAP@0.5:0.95 约 3-5%，但需重新标注全部 scratch
- 增大切片 overlap（80→160px），减少边缘截断比例
- 对 Frangi 弱响应的 scratch GT 进行人工审核，预期提升 AP 约 5-10%
- 推理端跨切片 NMS 优化（更激进的 connect_scratches），提升 scratch recall 约 5-8%

---

### 9.2 WearMetrics 为 Bbox 代理（非真实骨架量化）

**当前实现**：全图推理路径（`infer_full_pipeline.py`）使用 `BboxDefect` 适配器，以 bbox 长边作为划痕长度代理，面积作为实际面积代理。

**误差来源**：
- 对于经 `connect_scratches` 合并前的碎片框，长度代理严重低估（一条 500px 划痕被截为 3 段，每段长度约 170px，代理值之和与真实值接近但丢失了空洞间距信息）
- bbox 面积包含大量背景像素（划痕宽度仅 5-15px，但 bbox 高度与宽度相等）

**待改进方案**：
- 对检测框内区域执行实例分割（Segment Anything Model 或 YOLO-Seg），提取真实掩码后进行骨架化，得到精确的骨架长度和真实面积
- 短期：对 `connect_scratches` 后的合并框，用 Frangi 骨架化重新测量长度

---

### 9.3 未实现像素-毫米物理标定

**当前状态**：所有量化指标（L_total、L_center、A_total 等）均以像素为单位，未建立与实际物理尺寸（毫米）的对应关系。

**影响**：
- WearScore 公式中的饱和系数（如 `sat_log(L_center, scale=8.0)`）是按像素量经验调参，缺乏物理意义上的可解释性
- 等级边界（A<15、B<40、C<70）是定性估计，无法对应具体的物理损伤标准（如 ISO 标准中的划痕深度/宽度阈值）

**待改进方案**：
- 使用已知尺寸的标定板（如蚀刻有 1mm 间距网格的玻璃片）拍摄标定图像，建立像素/毫米换算系数
- 按物理尺寸（μm²、mm）重新标定 WearScore 参数，与光学行业标准（如 MIL-PRF-13830B）对接

---

### 9.4 Phase 3（API/Docker 部署）未完成

**当前状态**：系统处于本地离线工具阶段（CLI + Gradio 本地服务），未实现：
- RESTful HTTP API 服务（FastAPI/Flask）
- Docker 容器化打包
- 批量异步处理队列（Celery/Redis）
- 数据库存储（检测历史、磨损趋势追踪）
- 与生产线 PLC/相机控制系统的集成接口

**规划**（Phase 3）：
- FastAPI 服务：`POST /detect`（上传图像→返回 JSON 报告），`GET /health`
- Docker Compose 部署（含 NVIDIA GPU 容器运行时）
- 推理服务水平扩展（多 GPU 负载均衡）

---

## 附录 A：训练配置参数完整表

以下为 Stage 3（当前最优）的完整 Ultralytics 训练参数（来源：`output/training/stage2_cleaned/args.yaml`）：

| 参数 | Stage 1 | Stage 3 | 说明 |
|------|---------|---------|------|
| model | yolo12m.pt | stage1/best.pt | 初始权重 |
| data | defects.yaml | defects.yaml | 数据集配置 |
| epochs | 80 | 80 | 总训练轮数 |
| imgsz | 640 | 640 | 输入图像尺寸 |
| batch | 32 | 32 | 批大小 |
| lr0 | 0.01 | 0.002 | 初始学习率 |
| lrf | 0.01 | 0.01 | 最终学习率比 |
| momentum | 0.937 | 0.937 | SGD 动量 |
| weight_decay | 0.0005 | 0.0005 | 权重衰减 |
| warmup_epochs | 3.0 | 3.0 | 预热轮数 |
| mosaic | 1.0 | 1.0 | Mosaic 增强概率 |
| flipud | 0.5 | 0.5 | 垂直翻转概率 |
| fliplr | 0.5 | 0.5 | 水平翻转概率 |
| hsv_v | 0.4 | 0.4 | HSV 亮度扰动 |
| degrees | 0.0 | 5.0 | 旋转增强角度范围 |
| translate | 0.1 | 0.1 | 平移增强比例 |
| scale | 0.5 | 0.5 | 缩放增强范围 |
| cls | 0.5 | 0.5 | 分类损失权重 |
| box | 7.5 | 7.5 | 边框损失权重 |
| dfl | 1.5 | 1.5 | DFL 损失权重 |
| cache | ram | ram | 数据集缓存方式 |
| device | 0 | 0 | GPU 设备 ID |
| workers | 8 | 8 | 数据加载线程数 |
| patience | 50 | 50 | 早停耐心轮数 |
| optimizer | SGD | SGD | 优化器 |
| save_period | -1 | -1 | 每 N epoch 保存一次 |

---

## 附录 B：关键文件路径速查表

| 用途 | 路径 |
|------|------|
| 项目根目录 | `/home/bm/Dev/Microlens_DF/` |
| Python 包源码 | `src/darkfield_defects/` |
| CLI 入口 | `src/darkfield_defects/cli/app.py` |
| GUI 入口 | `src/darkfield_defects/viz/app.py` |
| 预处理流水线 | `src/darkfield_defects/preprocessing/pipeline.py` |
| 经典检测器 | `src/darkfield_defects/detection/classical.py` |
| 参数配置 | `src/darkfield_defects/detection/params.py` |
| WearScore | `src/darkfield_defects/scoring/wear_score.py` |
| **CNAS 认证权重** | `output/training/stage2_cleaned/weights/best.pt` |
| 预处理后全图 | `output/dataset_v2/images/` |
| 切片数据集 | `output/tile_dataset/` |
| 数据集配置 | `output/tile_dataset/defects.yaml` |
| CNAS 测试集定义 | `output/audit/test_set_v1.json` |
| 清洗报告 | `output/tile_dataset/cleanup_report.json` |
| CNAS 评测结果 | `output/cnas_eval/cnas_eval_results.json` |
| Stage1 训练日志 | `output/training/stage1/` |
| Stage3 训练日志 | `output/training/stage2_cleaned/` |
| 标定结果 | `output/calibration/`（首次运行后生成）|
| 默认检测参数 | `configs/detect_default.yaml` |
| 切片检测参数 | `configs/detect_tile.yaml` |
| 预训练 YOLO 权重 | `yolo12m.pt`（COCO 预训练）|
| SAHI 推理工具库 | `scripts/fullimage_utils.py` |
| CNAS 评测脚本 | `scripts/run_cnas_eval.py` |
| 全图推理脚本 | `scripts/infer_full_pipeline.py` |
| 标注审核工具 | `scripts/review_fullimage.py` |

---

## 附录 C：常用命令速查表

### 标定与预处理

```bash
# 运行标定（生成 CalibrationResult）
python - <<'EOF'
from pathlib import Path
from darkfield_defects.preprocessing.pipeline import PreprocessPipeline
p = PreprocessPipeline()
p.calibrate(
    bg_dir="path/to/bg_images",
    save_dir="output/calibration"
)
EOF

# 单张预处理
darkfield-defects preprocess \
    path/to/target.png \
    --output output/preprocess_test \
    --calibration output/calibration

# 批量预处理
darkfield-defects detect \
    output/dataset_v1/ \
    --output output/results \
    --calibration output/calibration
```

### 数据集构建

```bash
# 测试10张（验证参数）
python scripts/build_tile_dataset.py --n-sample 10

# 全量构建（247张）
python scripts/build_tile_dataset.py --all

# Shell 脚本方式
bash scripts/run_build_dataset.sh
```

### 模型训练

```bash
# Stage 1 基线训练
python scripts/train_stage1_baseline.py
python scripts/train_stage1_baseline.py --epochs 50 --batch 16

# 标注清洗（预览模式）
python scripts/step1_label_cleanup.py --dry-run

# 标注清洗（执行）
python scripts/step1_label_cleanup.py

# 伪标签生成
python scripts/step2_pseudo_labels.py

# Stage 3 重训练
python scripts/step3_retrain.py --epochs 80 --batch 32
```

### 推理与评测

```bash
# CNAS 标准评测（标准命令）
cd /home/bm/Dev/Microlens_DF
python scripts/run_cnas_eval.py

# 单张全图推理
python scripts/infer_full_pipeline.py \
    --image output/dataset_v2/images/13r.png

# 批量推理（指定 stem）
python scripts/infer_full_pipeline.py \
    --stems 123l 84r 51r

# 测试集 20 张推理
python scripts/infer_full_pipeline.py \
    --test-set output/audit/test_set_v1.json

# 自定义权重
python scripts/infer_full_pipeline.py \
    --stems 13r \
    --weights output/training/stage2_cleaned/weights/best.pt
```

### 标注审核

```bash
# 启动交互式审核工具（默认端口 7860）
python scripts/review_fullimage.py --split train
python scripts/review_fullimage.py --split val
python scripts/review_fullimage.py --split both --port 7861
```

### Gradio GUI

```bash
# 启动 Web 界面
python -c "
from darkfield_defects.viz.app import build_app
build_app().launch(server_port=7860, share=False)
"
# 访问 http://localhost:7860
```

### CLI 评估

```bash
# 像素级 + 实例级评估
darkfield-defects eval \
    output/predictions/ \
    output/ground_truth/ \
    --output output/eval_results.json \
    --iou-threshold 0.3

# 数据集统计
darkfield-defects info output/dataset_v1/
```

### 环境验证

```bash
# 验证 GPU 可用性
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

# 验证 Ultralytics
python -c "from ultralytics import YOLO; m = YOLO('yolo12m.pt'); print('YOLO OK')"

# 验证完整导入
python -c "
from darkfield_defects.preprocessing.pipeline import PreprocessPipeline
from darkfield_defects.detection.classical import ClassicalDetector
from darkfield_defects.scoring.wear_score import compute_wear_score
print('All imports OK')
"
```

---

*文档终*

**版权声明**：本文档为内部工程文档，仅供项目相关人员参阅，禁止对外发布。

*生成日期：2026-03-22 | 系统版本：darkfield-defects v0.1.0 | 最优模型：Stage3 mAP@0.5=0.6765*

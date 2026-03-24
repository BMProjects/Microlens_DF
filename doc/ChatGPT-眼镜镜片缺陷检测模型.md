# 眼镜镜片缺陷检测模型


## Prompt:
A lightweight and robust detection network for diverse glass surface defects via scale- and shape-aware feature extraction.pdf

PDF

现在有一篇类似的玻璃检测论文和相应的算法实现 https://github.com/FisherYuuri/GSDNet ，请进行分析，项目中用到的两个数据集，MSD Mobile phone screen surface defect segmentation dataset https://github.com/jianzhang96/MSD 和 SSGD: A SMARTPHONE SCREEN GLASS DATASET FOR DEFECT DETECTION https://github.com/VincentHancoder/SSGD 现在已经下载到本地。请仔细分析相关的算法，和数据集的特征。
与当前的项目相比
1. 本项目是否可以借鉴上面的GSDNet识别算法的优势，不是轻量化，而是高精度为目的，进行综合改进，以提升当前项目的技术水平和准确率指标？
2. 当前项目缺乏高质量标注，但是两个公开数据集有高质量标注，但是图像数据形式是否相似？可以利用先进的模型，进行预训练，再对当前项目的私有数据进行高质量标注或者标注改进？以提升当前模型的数据集标注质量？
3. 如前文的分割算法研究报告，如果使用分割算法，以分割结果为基础，通过形态识别缺陷，是否对于当前项目更加合适？有否快速实现的建议？

## Response:
Thought for 3m 41s

可以，而且这三个方向里，我的判断是：

**1）GSDNet 很值得借鉴，但更适合“模块化吸收”，不适合原样搬过去。**  
**2）两个公开数据集都能用，但用途不同：SSGD 更适合检测预训练，MSD 更适合分割预训练与弱监督标注改进。**  
**3）如果你的目标已经从“先过 60% mAP”升级到“更高精度 + 后续面积/分布/骨架分析”，那分割路线会比纯 bbox 更贴合问题本质。** 你们当前项目本身就是暗场显微镜单通道高分辨率图像，缺陷以细长亮线、亮点和大面积高亮区为主，图像尺寸 4096×3000，scratch 宽度约 2–20 px，当前系统虽然已经达到 `mAP@0.5 = 0.6765`，但 `scratch AP@0.5` 仍只有 `0.4525`，而且你们自己也已经把 “YOLO bbox → SAM 精细分割 → 实例 mask → 骨架化分析” 写成了桥接方向。

软件系统说明文档\_SDD\_v1

研究实验日志\_20260321\_Phase2C\_重标注迭代与S…

研究实验日志\_20260322\_推理流水线与测试准备

先说明一下：你说两个公开数据集已经下载到本地，但我在当前会话里**不能直接读取你本地那两份数据内容本身**；下面对 MSD / SSGD 的分析，依据的是它们的官方仓库、公开论文/说明，以及你上传的 GSDNet 论文和你们自己的项目文档。[GitHub+1](https://github.com/jianzhang96/MSD)

先看 GSDNet 到底强在哪里
----------------

GSDNet 不是简单换了个 backbone，而是围绕玻璃缺陷的两个核心问题做了专门设计：**形状不规则 / 细长结构** 和 **尺度变化 / 小目标**。它的主体仍是一阶段检测框架，但把 backbone 改成了带 **SFE block** 的结构，SFE 里用了带线性约束的可变形卷积 `LODConv`，专门强化细长、复杂形状的局部提取；在 SPPF 位置引入了 **SA attention** 去做尺度选择；在 neck 上用了 **EBiPAN**，通过双向拼接把浅层高分辨率信息带进来；loss 上则把 **CIoU + NWD** 用于框回归，并用 **Slide Loss** 处理困难样本。GSDNet 官方代码也是基于 Ultralytics 风格组织，训练默认 `imgsz=640`，这对你现在的 YOLO 系工程改造是友好的。

A lightweight and robust detect…

A lightweight and robust detect…

[GitHub+1](https://github.com/FisherYuuri/GSDNet)

更关键的是，论文里的消融结果并不是“玄学提升”。在 SSGD 上，YOLOv8 基线 `mAP50 = 61.6`，而 GSDNet 最终做到 `68.4`；其中混合损失先带来 `+1.6 mAP50`、`+3.9 mAPs`，SFE 是单模块里最强的增益项，带来 `+3.5 mAP50`，SA 对小目标尤其有效，`mAPs` 提升 `+3.9`，EBiPAN 则在 `+2.0 mAP50` 的同时还带来约 `+10 FPS` 的效率增益。换句话说，GSDNet 的可借鉴价值是真实存在的，而且**最有价值的不是“轻量化”，而是它对玻璃缺陷形状与尺度的针对性设计**。

A lightweight and robust detect…

A lightweight and robust detect…

A lightweight and robust detect…

MSD 和 SSGD，哪个更像你现在的数据
---------------------

**SSGD 明显更接近你现在的检测任务。**  
SSGD 是手机屏玻璃缺陷检测数据集，官方 README 写明它有 **2,504 张标注图像、3,914 个缺陷、7 个类别**，包括 crack、broken、spot、scratch、light-leakage、blot、broken-membrane，并建议用 **5-fold cross-validation**。另一篇公开研究对它的描述还提到：它是用**线扫工业相机**、在**黑色背景**和固定工作站条件下采集的，分辨率约 **1500×1000**，目标中有很多细小且 subtle 的缺陷。这个采集方式、背景风格和“玻璃表面亮缺陷”的视觉范式，和你现在的暗场镜片图像更接近，尽管它不是显微镜镜片，也没有圆形 ROI。[GitHub+2GitHub+2](https://github.com/VincentHancoder/SSGD)

**MSD 更像“可用于分割预训练的高质量标注源”，但域差更大。**  
MSD 官方 README 写的是 **1200 张 1920×1080 图像**，3 类缺陷：**Oil、Scratch、Stain**，由工业相机采集；公开研究进一步说明这些缺陷是**人工制造在玻璃盖板上**的，并且是**像素级 LabelMe 标注**。因此 MSD 的最大价值不是和你的镜片图“长得像”，而是它给了你一套**质量较高的 defect mask**，尤其对 scratch 类的分割预训练很有用。问题在于，它有油污类，缺陷成因和暗场散射机理并不一致，而且是盖板屏幕，不是圆形镜片；所以 MSD 更适合作为“先学会像素级缺陷边界”的源域，而不是直接拿来替代你私有数据。[GitHub+2Nature+2](https://github.com/jianzhang96/MSD)

你自己的数据和它们都存在明显域差：你们是**暗场显微镜、单通道灰度、圆形镜片 ROI、缺陷因散射呈亮、scratch 宽度 2–20 px、原图 4096×3000**；而这两个公开集都更偏手机盖板/屏幕表面检测。也就是说，**它们能帮你学“缺陷是什么样”，但不能直接替你学“暗场镜片缺陷物理成像”**。

背景技术综合研究报告\_v1

背景技术综合研究报告\_v1

软件系统说明文档\_SDD\_v1

1）是否值得借鉴 GSDNet 来提升你当前项目
------------------------

我的答案是：**值得，而且是高优先级，但要“拆模块吸收”，不要整套照搬。**

最值得借鉴的，是 **SFE / LODConv**。  
你现在最弱的是 scratch，而你们文档已经确认根因之一是：长划痕被 tile 截断后，bbox 的 IoU 非常脆弱，同时大量 scratch 标注又存在 Frangi 弱响应噪声。GSDNet 的 SFE/LODConv 专门针对“自由线状、细长、不规则”的玻璃缺陷设计，这和你当前 scratch 的痛点是高度对齐的。尤其论文里明确提到，普通 deformable conv 更容易受背景干扰，而线性约束的 LODConv 更能贴合长条缺陷、抑制类似背景杂质。对你来说，这一块是最值得优先迁入现有检测分支的。

研究实验日志\_20260321\_Phase2C\_重标注迭代与S…

研究实验日志\_20260320\_Phase2AB\_切图标注与基…

A lightweight and robust detect…

第二值得借鉴的是 **NWD + 困难样本分类损失**。  
你的 spot 很小、scratch 很细、bbox 又细长，普通 IoU 系列对这类目标很不稳定。GSDNet 用 NWD 来缓解小目标定位对尺度的敏感性，这一点对你是有现实意义的；而分类侧的 Slide Loss 本质上也是在强调 hard sample，只是你当前工程里已经在用 YOLOv12m + Focal/Varifocal/DFL 体系，所以我不建议第一步就把整套 loss 推翻，**更稳妥的是先试 “保持现有分类损失，替换/增加 bbox 回归里的 NWD 分支”**。

A lightweight and robust detect…

A lightweight and robust detect…

研究实验日志\_20260320\_Phase2AB\_切图标注与基…

**SA attention 可以借，但优先级低于 SFE。**  
它对 spot 和小缺陷会有帮助，因为论文里它主要提升的是 `mAPs`；但你当前真正的卡点不是 spot，而是 scratch 的结构表达。所以如果 AI 助手要排顺序，我会建议：  
**先做 SFE / 方向性感受野 → 再做 NWD → 再考虑 SA。**

A lightweight and robust detect…

A lightweight and robust detect…

**EBiPAN 可以参考，但不必按“轻量化脉络”照搬。**  
因为你现在的目标不是 68 FPS，而是更高精度。GSDNet 之所以做 EBiPAN，是在精度和速度之间找平衡；但你现在有 4090D，检测不是算力最紧的环节。对你更合理的做法可能是：  
保留“把更浅层高分辨率特征送进 neck”这个思想，但不必拘泥于轻量化版本，可以直接做更激进的 **P2 head / 更强高分辨率 neck / 更大的推理输入**。你们现在本来就在用 SAHI 和原图滑窗，系统约束和 GSDNet 论文里的实时工业检测不完全一样。

软件系统说明文档\_SDD\_v1

研究实验日志\_20260320\_Phase2AB\_切图标注与基…

A lightweight and robust detect…

但也要把限制说清楚：  
**GSDNet 再好，它仍然是 bbox detection。**  
而你当前最难的对象恰恰是“长、深、密、交叉”的划痕网络。GSDNet 会让 detector 更会“看线”和“看小目标”，但**它不能从根本上解决 bbox 对长划痕语义表达不够自然的问题**。所以我的建议不是“换成 GSDNet 就好了”，而是：  
把 GSDNet 作为你现有 YOLOv12/YOLOv13 检测分支的**强化模块来源**。

研究实验日志\_20260321\_Phase2C\_重标注迭代与S…

研究实验日志\_20260322\_推理流水线与测试准备

如果要给 AI 助手一条高精度导向的检测改造路线，我会写成这样：

*   **Backbone**：保留 YOLOv12/YOLOv13 主体，但在浅层和中层 `C2f/C3` 替换为 **SFE-like directional deformable bottleneck**。
*   **Neck**：优先增加 **P2 检测层** 或更高分辨率特征流，再决定是否引入 EBiPAN 思想。
*   **Loss**：bbox 回归尝试 **CIoU + NWD**；分类先保留现有 Focal/Varifocal，再做对照。
*   **Input**：从你们文档里的备选方案 `[gray, frangi, clahe]` 三通道输入重新拿回来做实验，因为这和 GSDNet 的“形状先验”是同向增强。
*   **Inference**：继续保留 SAHI + IOS NMS + connect\_scratches。

这条路线比“完全换模型”更符合你们现有系统成熟度。

研究实验日志\_20260320\_Phase2AB\_切图标注与基…

软件系统说明文档\_SDD\_v1

A lightweight and robust detect…

2）能否用 MSD / SSGD 做预训练，再反过来提升私有数据标注质量
------------------------------------

答案是：**可以，而且这是我最推荐你们马上做的事情之一。**

但要分开用：

**SSGD 用来做检测预训练。**  
因为它本身就是玻璃缺陷检测数据集，类别里就有 scratch、spot、crack、broken 等，且黑背景、工业采集、缺陷 subtle，这些都比 MSD 更接近你们的检测问题。最合理的方法不是直接混训，而是：  
先在 SSGD 上做 detector 预训练，让模型学会“玻璃表面的点、线、裂、块缺陷”这一类视觉先验；再迁移到你们私有数据上微调。这样比从 ImageNet/COCO 迁移更接近任务域。[GitHub+1](https://github.com/VincentHancoder/SSGD)

**MSD 用来做分割预训练。**  
因为 MSD 的核心价值是**像素级高质量 mask**，这正好弥补你们当前“bbox 多、mask 缺”的短板。最合适的路径是：  
先用 MSD 预训练一个 scratch/stain/oil 的 segmentation backbone，让模型学会“细线边界、污渍边界、弱对比缺陷边界”；然后再用你们自己的弱标签去做 domain adaptation。即使类别不完全一致，这个 pretrain 仍然有价值，因为它预训练的是“defectness / boundary / thin-structure perception”。[GitHub+2Nature+2](https://github.com/jianzhang96/MSD)

**它们不能直接给你“高质量私有标签”，但可以做 teacher。**  
我不建议“直接用公开集训练一个模型，然后全自动替换你们私有 GT”。域差太大，容易把手机屏幕的数据偏差带进镜片暗场。更稳妥的做法是 teacher-style 标注改进：

1.  用 **SSGD 预训练检测器**，在你们私有图上给出更稳的 box proposal。
2.  用 **MSD 预训练分割器** 或 promptable SAM，对 proposal 区域做 mask refinement。
3.  同时引入你们现有的 **Frangi / threshold / morphology** 结果作为形状先验；只有当“模型 mask”和“传统算法响应”一致时才提升置信度。
4.  最终输出的不只是新 bbox，还包括 **mask / skeleton / OBB 候选**。

这其实和你们已有技术积累非常契合，因为你们已经在用 Frangi 响应来审计 scratch 标注噪声，也已经把 SAM→骨架化写成长期路线。

研究实验日志\_20260320\_Phase2AB\_切图标注与基…

背景技术综合研究报告\_v1

研究实验日志\_20260322\_推理流水线与测试准备

如果只问“哪个数据集对标注改进更有用”，我的排序是：

*   **用于检测预训练**：SSGD > MSD
*   **用于分割预训练**：MSD > SSGD
*   **用于直接替代私有训练集**：都不行
*   **用于弱监督标注改进**：MSD + 你们自己的 Frangi 先验 最合适

因为你当前最大的短板不是“没数据”，而是“没有足够高质量的像素级结构表达”。[GitHub+1](https://github.com/VincentHancoder/SSGD)

研究实验日志\_20260321\_Phase2C\_重标注迭代与S…

3）如果改用分割，再做形态识别，是否更适合当前项目
-------------------------

我的判断是：**是的，从“问题定义”上说更适合；从“工程节奏”上说应该走渐进式，而不是一步推翻现有检测系统。**

原因很直接。你们当前的 hardest cases 具有这几个特点：

*   划痕很长，可能跨多个 tile
*   宽度很窄，但交叉很多
*   大面积报废镜片上会出现密集缺陷网络
*   后续还要做面积、长度、中心区密度、交叉点、散射 profile 等分析

这类问题用 bbox 不是做不了，但天然别扭；而 mask / skeleton 才是更自然的表达。你们自己文档里也已经承认 bbox 只是 WearMetrics 的代理，而精确路线应该是 mask 和骨架。

软件系统说明文档\_SDD\_v1

研究实验日志\_20260322\_推理流水线与测试准备

背景技术综合研究报告\_v1

而且你们的传统算法本来就是从分割思路出发的：  
`frangi_sigmas`、`threshold_method`、`morph_open_kernel`、`min_area` 这些都说明你们已经有一套“候选区域 / 形态区域”知识体系。换句话说，**你们并不是从 0 开始做分割，而是已经有弱监督分割老师了。**

背景技术综合研究报告\_v1

### 我建议的最快实现路线

**不是先做纯语义分割替代全部检测，而是做“检测 + 分割精修”的双层系统。**

### 第一步，最快可落地

保留现在的 YOLOv12 主线检测，继续负责：

*   高召回定位
*   CNAS / 当前工程交付

然后在其后串一个：

*   **SAM / SAM2 / MobileSAM 掩码精修**
*   输入可以是 YOLO bbox 或 Frangi 连通域框
*   输出实例 mask
*   再做 skeletonization、connected components、长度/面积统计

这条路最快，因为你几乎不需要先准备大规模人工 mask。你们文档里已经把这条桥接路径写出来了。

研究实验日志\_20260322\_推理流水线与测试准备

### 第二步，建立真正的 segmentation baseline

我会建议至少做 3 个基线：

*   **U-Net++**：实现最简单，适合先验证弱标签可不可学
*   **DeepLabV3+**：边界更稳，适合中等复杂度
*   **SegFormer-B2/B3**：对细节与全局都相对平衡，通常是工业分割里很务实的选择

训练时先不要直接上 4/5 类细分，先做：

*   `scratch / spot / critical / background` 四类  
    或者更快一点先做
*   `defect / background` 二类

因为你的第一目标是把“结构区域”学准，而不是立刻把所有亚类学全。这个建议是工程判断，不是来自某一篇文献。它是基于你们当前标注质量和任务目标得出的。

### 第三步，把形态学分析接到评分系统

一旦拿到 mask，就能比现在的 bbox proxy 更真实地计算：

*   scratch 骨架长度
*   交叉点数量
*   连通域面积
*   中心区占比
*   散射 profile
*   缺陷网络密度

这正好对应你们现在想做的 WearMetrics 升级。

研究实验日志\_20260322\_推理流水线与测试准备

我的最终建议
------

如果把你的三个问题合成一条研发路线，我会这样排优先级：

**第一优先级：用 GSDNet 的 SFE / NWD 思想增强现有检测主线。**  
这是最短路径，最可能先把 scratch AP 再往上拉。

A lightweight and robust detect…

A lightweight and robust detect…

**第二优先级：用 SSGD 做检测预训练，用 MSD 做分割预训练。**  
这是提升私有弱标签质量的最佳组合，但不要直接拿公开集替换私有数据。[GitHub+2GitHub+2](https://github.com/VincentHancoder/SSGD)

**第三优先级：走“检测→分割精修→骨架分析”的渐进式升级。**  
这条路最符合你们当前“已经能检出，但还想做精确量化”的状态。

研究实验日志\_20260322\_推理流水线与测试准备

背景技术综合研究报告\_v1

一句话概括：

> **GSDNet 适合提升你“检测器看线、看小目标”的能力；MSD/SSGD 适合提升你“模型初始化与弱监督标签质量”的能力；而真正让项目从“能检测”升级到“能精确分析”的，是分割 + 形态学这条线。**

下一步最值得做的，不是三件事同时上，而是先做一个小规模对照：

*   `YOLOv12 + SFE/NWD`
*   `SSGD 预训练 → 私有微调`
*   `YOLO bbox → SAM/SegFormer 精修 mask`

看哪条对 scratch 改善最大。  
我可以继续把这个整理成一份\*\*“公开数据迁移 + GSDNet模块借鉴 + 分割升级”的实施方案\*\*，细化到模型、训练顺序、数据组织和消融实验表。

## Prompt:
请将上述意见整理成一份**“公开数据迁移 + GSDNet模块借鉴 + 分割升级”的实施方案**，细化到模型、训练顺序、数据组织和消融实验表。方便后续给项目端的AI助手进行分析，结合具体的项目结构，自主实现。

## Response:
Thought for 38s

下面是一份可直接交给项目端 AI 助手的实施方案草案，目标是把你前面提到的三条路线统一成一个**可迭代、可对比、可逐步落地**的研发计划：

> **公开数据迁移 + GSDNet 模块借鉴 + 分割升级**

我先给总判断：

**短期最优路线**不是推翻现有 YOLOv12 主线，而是保留当前已达标的检测系统作为基线，在其上新增两条平行分支：**GSDNet 风格高精度检测分支** 和 **分割精修/量化分支**。  
这样做最符合你们当前状态：现有系统已经在独立测试集上达到 `mAP@0.5 = 0.6765`，超过 0.60 目标，但 `scratch AP@0.5 = 0.4525` 明显偏低；项目代码结构、训练脚本、推理脚本、评分模块都已成形，适合做模块化增量改造，而不是重建整套系统。

软件系统说明文档\_SDD\_v1

研究实验日志\_20260322\_推理流水线与测试准备

软件系统说明文档\_SDD\_v1

* * *

一、实施目标
======

1）总体目标
------

在不破坏现有可交付系统的前提下，完成三件事：

1.  **提升 scratch 类检测能力**
    *   重点解决细长、交叉、碎片化划痕
    *   目标是把 `scratch AP@0.5` 从当前约 `0.45` 提升到 `0.55+`
2.  **提高私有数据标注质量**
    *   利用公开数据集做任务相关预训练
    *   利用分割模型和现有 Frangi 弱监督结果，改进私有数据标签
3.  **为后续面积/长度/分布/骨架分析建立分割基础**
    *   从 bbox 代理量化逐步升级到 mask / skeleton 量化

当前系统的已知关键问题是：scratch 类受切片边界截断、细长 bbox 的 IoU 敏感性、以及 Frangi 弱响应噪声框影响最明显；同时当前 WearMetrics 仍以 bbox 长短边作为代理量。

研究实验日志\_20260322\_推理流水线与测试准备

软件系统说明文档\_SDD\_v1

研究实验日志\_20260322\_推理流水线与测试准备

* * *

二、公开数据与项目数据的角色分工
================

1）SSGD 的定位：检测预训练数据
------------------

SSGD 官方仓库说明它包含 **2,504 张标注图像、3,914 个缺陷、7 个类别**，并建议使用 **5-fold cross-validation**；公开论文摘要也说明它是用专业采集设备在固定工作站中采集的玻璃缺陷检测数据。这个数据集更适合做**检测器的任务相关预训练**，尤其适合学习玻璃表面的点、线、裂、块类缺陷外观。[GitHub+2IEEE Resource Center+2](https://github.com/VincentHancoder/SSGD?utm_source=chatgpt.com)

2）MSD 的定位：分割预训练数据
-----------------

MSD 官方仓库说明它是**手机屏表面缺陷分割数据集**，包含 **1200 张 1920×1080 图像**，3 类缺陷：Oil、Scratch、Stain，标注格式为 PASCAL VOC，并提供额外非缺陷图像。它更适合做**分割 backbone / segmentation head 的预训练**，特别是帮助模型学习缺陷边界和细长 scratch mask。[GitHub](https://github.com/jianzhang96/MSD?utm_source=chatgpt.com)

3）你们私有数据的定位：最终目标域
-----------------

你们私有数据是暗场显微镜镜片缺陷图像，当前训练体系是：

*   原图预处理后切为 `640×640` tiles
*   YOLO 格式 3 类：scratch / spot / critical
*   当前最优权重为 `output/training/stage2_cleaned/weights/best.pt`
*   切片数据集位于 `output/tile_dataset/`
*   训练/评测脚本已经成形。
    研究实验日志\_20260322\_推理流水线与测试准备
    软件系统说明文档\_SDD\_v1

* * *

三、总体技术路线
========

我建议把项目拆成三条并行工作流：

Track A：现有 YOLOv12 主线稳态保持
-------------------------

用途：

*   维持当前可交付方案
*   所有新方案都与它对比
*   作为 fallback

基线状态：

*   `mAP@0.5 = 0.6765`
*   `scratch AP@0.5 = 0.4525`
*   `spot AP@0.5 = 0.8180`
*   `critical AP@0.5 = 0.7589`。
    研究实验日志\_20260322\_推理流水线与测试准备

Track B：GSDNet 模块借鉴的高精度检测分支
---------------------------

用途：

*   在检测框架内，优先解决 scratch 的形状表达与小目标问题
*   不追求轻量化，追求精度

GSDNet 的关键可借鉴模块包括：

*   **SFE block**：面向复杂形状缺陷的 shape-aware 特征提取
*   **SA attention**：多尺度空间注意力
*   **EBiPAN**：强调高分辨率信息的双向特征聚合
*   **CIoU + NWD** 的混合回归损失
*   困难样本分类损失。  
    这些设计的论文与官方代码都已公开，且 GSDNet 在 SSGD 等玻璃缺陷数据上给出明显优于 YOLOv8 的结果。
    A lightweight and robust detect…
    A lightweight and robust detect…
    [GitHub+1](https://github.com/FisherYuuri/GSDNet?utm_source=chatgpt.com)

Track C：分割升级分支
--------------

用途：

*   弥补 bbox 对细长交叉划痕表达不足的问题
*   为后续面积、长度、交叉点、连通性分析提供真实 mask
*   反向服务于私有数据标注改进

你们自己的系统文档已把未来桥接路径明确为：  
`YOLO bbox → SAM 精细分割 → 实例 mask → 骨架化分析`。

研究实验日志\_20260322\_推理流水线与测试准备

* * *

四、推荐的数据组织方案
===========

建议在现有项目根目录 `/home/bm/Dev/Microlens_DF/` 下新增如下结构。这个方案与当前 `src/darkfield_defects/`、`output/tile_dataset/`、`scripts/` 的组织方式兼容。

软件系统说明文档\_SDD\_v1

```
data/
  public/
    SSGD/
      images/
      labels_yolo/
      annotations_coco/
      splits_5fold/
    MSD/
      images/
      masks/
      splits/
  private/
    full_images/
    preprocessed/
    tiles_v1_640/
    tiles_v2_overlap256/
    labels_yolo_stage3/
    labels_refined/
    masks_weak/
    masks_refined/
  interm/
    pseudo_labels/
    proposals/
    sam_masks/
    skeletons/
```

同时建议把数据版本显式化：

```
output/
  experiments/
    exp_det_baseline_y12/
    exp_det_gsd_sfe/
    exp_det_gsd_sfe_nwd/
    exp_det_ssgd_pretrain/
    exp_seg_msd_pretrain/
    exp_seg_private_weak/
```

* * *

五、模型实施方案
========

A. 检测主线基线模型
-----------

### A0：当前基线

*   模型：现有 `yolo12m.pt` 微调版本
*   数据：私有 tiles，Stage3 标签
*   作用：所有后续实验对照基线

你们当前训练参数中，`imgsz=640`、`batch=32`、`optimizer=SGD`、`mosaic=1.0`、`cls=0.5`、`box=7.5`、`dfl=1.5`，这套配置应先保持不变，避免比较不公平。

软件系统说明文档\_SDD\_v1

* * *

B. GSDNet 借鉴的检测升级线
------------------

### B1：YOLOv12 + SFE-like 模块

**目标**：优先提高 scratch 的线状表达能力。  
**做法**：

*   在 backbone 浅层与中层各插入一组 SFE-like block
*   核心思想：引入带方向约束的可变形卷积或可替代的方向感受野模块
*   第一版不替换整网，只替换最关键的 2–4 个 block

**实现建议**

*   若 AI 助手实现 LODConv 成本高，可先用简化版替代：
    *   `1×k + k×1` 条状卷积
    *   depthwise deformable conv
    *   directional bottleneck
*   保持 neck 与检测头不变

**优先级**：最高  
因为 GSDNet 论文里 SFE 是单模块中最明显提升精度的组件，尤其针对不规则形状和细长缺陷。

A lightweight and robust detect…

### B2：YOLOv12 + SFE-like + NWD 回归

**目标**：改善细长框和小目标框的定位稳定性。  
**做法**：

*   保留当前分类损失
*   将 bbox 回归改为：
    *   `L_box = λ1 * CIoU + λ2 * NWD`
*   先从 `λ1=1.0, λ2=0.5` 起做网格搜索

**原因**  
你当前 `mAP@0.5` 和 `mAP@0.5:0.95` 差值很大，说明模型“能找到”但定位不够精确；而 NWD 对小目标和细长目标通常比纯 IoU 更稳定。你们文档中也明确指出暗场边界本身有 ±5 px 不确定性，这使传统 IoU 更脆弱。

研究实验日志\_20260322\_推理流水线与测试准备

软件系统说明文档\_SDD\_v1

### B3：YOLOv12 + SFE-like + NWD + P2 Head

**目标**：兼顾 spot 类和更细 scratch 片段。  
**做法**：

*   在 neck 中增加高分辨率 P2 检测分支
*   优先用于小 spot 与极细小局部划痕
*   训练时对 spot 类可尝试轻度 class weight 提升

### B4：YOLOv12 + SFE-like + NWD + P2 + SA attention

**目标**：作为检测升级线的上限版本。  
**做法**：

*   在 SPPF 后或 neck 初段加入 SA-style attention
*   不建议一开始就在每个 stage 都加

* * *

C. 公开数据迁移的检测预训练线
----------------

### C1：SSGD 预训练 → 私有微调

**目标**：让检测器先学会玻璃缺陷的视觉先验。  
**训练顺序**：

1.  在 SSGD 上训练/微调 detector
2.  加载到私有数据上进行继续训练
3.  与随机初始化或 COCO 初始化对照

**迁移策略**

*   第一版只迁 backbone + neck
*   detection head 重新初始化
*   类别映射：
    *   SSGD 的 `scratch / spot / crack / broken / ...`
    *   私有数据只保留映射后共有抽象结构先验，不做强行一一对类

**判断标准**

*   看私有数据上 scratch AP 是否显著优于纯 COCO 初始化

* * *

D. 分割升级线
--------

### D1：MSD 预训练的二分类分割器

**目标**：快速得到一个“缺陷 vs 背景”的稳定 mask 模型。  
**模型推荐顺序**：

1.  U-Net++
2.  DeepLabV3+
3.  SegFormer-B2

**训练顺序**：

1.  在 MSD 上做预训练
2.  转到私有弱标签 `masks_weak/` 微调
3.  用 SAM/Frangi 一致性筛选高置信 mask 做第二轮训练

**标签生成**

*   由你们现有传统算法输出弱标签：
    *   Frangi + threshold + morphology
*   第一版只做二类 mask
*   不急于先做 scratch/spot/critical 三类精细实例分割

### D2：YOLO bbox → SAM 精修 → mask 回流

**目标**：快速构建高质量私有 mask。  
**流程**：

1.  用当前 YOLOv12 或 B 线模型生成 bbox proposal
2.  对 proposal 区域调用 SAM / SAM2 / MobileSAM
3.  用 Frangi 响应与灰度先验筛掉明显错误 mask
4.  生成 `masks_refined/`
5.  用这些 mask 训练 D1 分割器

### D3：mask → skeleton → WearMetrics 升级

**目标**：建立真正适用于镜片磨损分析的结构量化。  
**输出指标**：

*   骨架长度
*   平均宽度
*   交叉点数量
*   连通域面积
*   中心区缺陷占比
*   散射 profile

这条线直接服务于你们当前 P1 问题：bbox 代理指标。

研究实验日志\_20260322\_推理流水线与测试准备

* * *

六、训练顺序建议
========

Phase 0：冻结当前基线
--------------

固定以下内容：

*   当前 Stage3 数据版本
*   当前 `best.pt`
*   当前 CNAS 评估脚本
*   当前测试集定义

原因是你们已经明确：任何修改不能动 val/test 的 GT，也不能再对 GT 做 `connect_scratches`。

研究实验日志\_20260321\_Phase2C\_重标注迭代与S…

* * *

Phase 1：检测迁移预训练
---------------

顺序：

1.  `C1`: SSGD 预训练 detector
2.  私有 Stage3 数据微调
3.  与 A0 对照

这一步先不引入 GSDNet 模块，先回答一个简单问题：  
**“任务相关公开检测数据预训练，能否在私有数据上直接带来收益？”**

* * *

Phase 2：GSDNet 模块借鉴
-------------------

顺序：

1.  `B1`: SFE-like
2.  `B2`: SFE-like + NWD
3.  `B3`: SFE-like + NWD + P2
4.  `B4`: SFE-like + NWD + P2 + SA

这样安排的原因是：

*   先验证 shape-aware 模块本身是否有效
*   再看回归损失是否补充增益
*   最后再看小目标层和注意力是否有边际收益

* * *

Phase 3：分割预训练与私有弱监督
-------------------

顺序：

1.  `D1`: MSD 预训练二分类分割器
2.  私有弱标签微调
3.  `D2`: 用 YOLO + SAM 生成 refined masks
4.  用 refined masks 再训练分割器

* * *

Phase 4：检测与分割联动
---------------

顺序：

1.  用分割结果生成更紧的 bbox / OBB 候选
2.  回流到检测训练集的 `labels_refined/`
3.  再次训练检测器

这个阶段的目标不是替代检测器，而是**用分割帮助检测标签变好**。

* * *

七、与现有项目结构的整合建议
==============

基于当前代码结构，建议 AI 助手在以下位置扩展。

软件系统说明文档\_SDD\_v1

新增目录
----

```
src/darkfield_defects/
  detection/
    gsd_modules/
      sfe.py
      directional_conv.py
      nwd_loss.py
      neck_p2.py
  segmentation/
    datasets.py
    unetpp.py
    deeplabv3plus.py
    segformer.py
    sam_refine.py
    skeleton.py
  transfer/
    ssgd_pretrain.py
    msd_pretrain.py
    label_refine.py
```

新增脚本
----

```
scripts/
  run_pretrain_ssgd.py
  run_finetune_private.py
  run_train_gsd_variant.py
  run_train_segmentation.py
  run_refine_masks_with_sam.py
  run_bbox_from_masks.py
  run_skeleton_metrics.py
```

配置文件
----

```
configs/
  det_ssgd_pretrain.yaml
  det_gsd_sfe.yaml
  det_gsd_sfe_nwd.yaml
  det_gsd_full.yaml
  seg_unetpp_msd.yaml
  seg_segformer_private.yaml
```

* * *

八、消融实验表
=======

下面这个表建议直接按实验编号执行。

表 1：检测线主消融
----------

| Exp ID | 模型 | 预训练 | 关键改动 | 主要目的 | 主要指标 |
| --- | --- | --- | --- | --- | --- |
| A0 | YOLOv12 baseline | COCO/当前 | 无 | 基线 | mAP, scratch AP |
| C1 | YOLOv12 | SSGD | 公开检测预训练 | 验证任务相关迁移 | mAP, scratch AP |
| B1 | YOLOv12+SFE | COCO/当前 | SFE-like block | 验证 shape-aware | scratch AP |
| B2 | YOLOv12+SFE | COCO/当前 | \+ NWD | 验证细长框定位 | mAP50:95 |
| B3 | YOLOv12+SFE | COCO/当前 | \+ NWD + P2 | 验证 spot/小缺陷 | spot AP |
| B4 | YOLOv12+SFE | COCO/当前 | \+ NWD + P2 + SA | 上限版本 | 全指标 |
| C2 | B2/B3 最优 | SSGD | SSGD 预训练 + GSD 模块 | 组合增益 | 全指标 |

表 2：分割线主消融
----------

| Exp ID | 模型 | 预训练 | 标签来源 | 类别设置 | 目的 | 指标 |
| --- | --- | --- | --- | --- | --- | --- |
| D1 | U-Net++ | MSD | 私有弱标签 | defect/bg | 快速 baseline | IoU, Dice |
| D2 | DeepLabV3+ | MSD | 私有弱标签 | defect/bg | 边界更稳 | IoU, Dice |
| D3 | SegFormer-B2 | MSD | 私有弱标签 | defect/bg | 全局+局部平衡 | IoU, Dice |
| D4 | 最优分割器 | MSD | YOLO+SAM refined | defect/bg | 验证 refined mask | IoU, Dice |
| D5 | 最优分割器 | MSD | refined | scratch/spot/critical/bg | 多类升级 | class IoU |

表 3：检测-分割联动实验
-------------

| Exp ID | 检测器 | 分割器 | 联动方式 | 目的 | 指标 |
| --- | --- | --- | --- | --- | --- |
| DS1 | A0 | D4 | bbox→mask 精修 | 验证量化 | 骨架长度误差 |
| DS2 | B2/B3 | D4 | 精修后反投 bbox | 改善检测标签 | scratch AP |
| DS3 | B2/B3 | D5 | mask→OBB 候选 | 验证 OBB 可行性 | OBB AP / scratch AP |

* * *

九、评估指标建议
========

除了你们当前已有的：

*   mAP@0.5
*   mAP@0.5:0.95
*   Precision / Recall

建议新增三组专项指标：

1）scratch 专项
------------

*   scratch AP@0.5
*   长划痕召回率
*   跨 tile 重复框数量
*   `connect_scratches` 前后差异

2）分割专项
------

*   Dice
*   IoU
*   boundary F1
*   skeleton length error

3）最终业务专项
--------

*   WearScore 稳定性
*   损伤等级一致性
*   中心区损伤面积偏差

因为当前你们已经明确：`mAP@0.5` 对本项目比 `mAP@0.5:0.95` 更合理，但如果要做结构量化升级，分割和骨架指标必须单独引入。

研究实验日志\_20260322\_推理流水线与测试准备

* * *

十、优先级建议
=======

第一优先级
-----

**C1 + B1 + B2**

原因：

*   实现成本最低
*   最可能最快改善 scratch
*   不破坏现有系统

第二优先级
-----

**D1 + D2**

原因：

*   建立分割 baseline
*   为后续高质量私有标签打基础

第三优先级
-----

**D2 + DS2**

原因：

*   让分割反哺检测
*   真正提高标签质量

暂缓项
---

*   直接全量 OBB 重标注
*   一开始就做多类实例分割
*   彻底替换现有 YOLOv12 主线

* * *

十一、给 AI 助手的执行摘要
===============

可以直接把下面这段发给 AI 助手：

> 在现有 `Microlens_DF` 项目中，保留当前 YOLOv12 Stage3 系统作为基线，不修改现有 val/test GT。新增三条并行实验线：  
> 1）`SSGD` 检测预训练后迁移到私有数据；  
> 2）借鉴 `GSDNet` 的 `SFE-like block + NWD loss + P2 head` 做高精度检测升级；  
> 3）使用 `MSD` 做分割预训练，在私有弱标签上训练 `U-Net++ / DeepLabV3+ / SegFormer-B2`，并通过 `YOLO bbox -> SAM refine -> mask refined` 逐步构建高质量 mask。  
> 先做 `A0/C1/B1/B2`，再做 `D1/D2`，最后做 `DS2`。所有实验统一使用当前 Stage3 数据切分、CNAS 测试集和现有评估脚本，重点观察 `scratch AP@0.5`、长划痕召回和后续 skeleton 量化误差。





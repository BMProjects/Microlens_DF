"""把 V0.3 修订建议稿核心结构生成为 .docx，便于交付给检测机构对照修改。

- 公式使用文本表达（检测机构在 Word 中可改为公式编辑器），避免依赖 pandoc
- 表格使用 python-docx 直接构造
- 与 .md 修订稿一致；.md 仍是单一事实源
"""

from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.shared import Pt

DOCS_DIR = Path(__file__).resolve().parents[1] / "docs"
OUT_PATH = DOCS_DIR / "离焦微结构镜片磨损识别技术项目《软件评测报告》V0.3修订建议稿.docx"


def _font(run, size: int = 10) -> None:
    run.font.name = "Microsoft YaHei"
    run.font.size = Pt(size)


def _para(doc, text: str, *, bold: bool = False, size: int = 10) -> None:
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.bold = bold
    _font(r, size)


def _h1(doc, text: str) -> None:
    h = doc.add_heading(text, level=1)
    for r in h.runs:
        _font(r, 14)


def _h2(doc, text: str) -> None:
    h = doc.add_heading(text, level=2)
    for r in h.runs:
        _font(r, 12)


def _h3(doc, text: str) -> None:
    h = doc.add_heading(text, level=3)
    for r in h.runs:
        _font(r, 11)


def _table(doc, headers: list[str], rows: list[list[str]]) -> None:
    t = doc.add_table(rows=1 + len(rows), cols=len(headers))
    t.style = "Light Grid"
    for i, h in enumerate(headers):
        c = t.rows[0].cells[i]
        c.text = ""
        r = c.paragraphs[0].add_run(h)
        r.bold = True
        _font(r)
    for ri, row in enumerate(rows, start=1):
        for ci, val in enumerate(row):
            c = t.rows[ri].cells[ci]
            c.text = ""
            r = c.paragraphs[0].add_run(str(val))
            _font(r)


def build() -> Path:
    doc = Document()

    # 封面信息
    title = doc.add_heading("软件评测报告 V0.3 修订建议稿", level=0)
    for r in title.runs:
        _font(r, 18)

    for line in [
        "项目名称：离焦微结构镜片磨损识别技术项目",
        "系统名称：镜片磨损智能识别算法（LWIA-Det v1.0.0）",
        "委托单位：中国计量大学信息工程学院",
        "检测单位：浙江网新检测技术有限公司",
        "文档性质：本稿基于检测机构 V0.1 PDF 草稿 + 项目方 V0.2 修订稿 + 三次实测结果整理；"
        "检测机构原报告封面、声明、目录、报告编号、签章页保持不变。",
    ]:
        _para(doc, line)

    # 第一部分
    _h1(doc, "第一部分　可直接替换的文字（按 V0.1 章节顺序）")

    _h2(doc, "§1　《测试结论》单元格 - 评测结论")
    _para(
        doc,
        "受中国计量大学信息工程学院委托，于 2026 年 05 月 20 日至 05 月 22 日，根据评测依据栏"
        "所列文件，对其开发的「离焦微结构镜片磨损识别技术项目」所包含的技术指标进行测试。"
        "具体实测结果如下：",
    )
    _para(doc, "1. 磨损识别准确率（主指标）：mAP50 = 0.6966，即 69.66%；三次重复执行结果完全相同（差值 0.0000）。")
    _para(doc, "2. 辅助统计指标：Precision = 0.6122，Recall = 0.6811，mAP@0.5:0.95 = 0.4723（三次完全相同）。")
    _para(doc, "3. 符合性判定：上述主指标 mAP50 = 69.66% ≥ 委托测试文件规定的 60% 阈值，技术指标判定为 符合。")

    _h2(doc, "§2　《测试环境与配置》")
    _h3(doc, "服务器（被测算法运行主机，远程接入）")
    _table(
        doc,
        ["项目", "内容"],
        [
            ["主机标识", "BM-Server-Deb（开发单位提供）"],
            ["CPU", "AMD Ryzen 5 7500F，6 核 12 线程，3.7 GHz（睿频 5.03 GHz）"],
            ["GPU", "NVIDIA GeForce RTX 4090D，显存 24 GB（驱动 595.71.05 / CUDA 13.2）"],
            ["内存", "64 GB"],
            ["硬盘", "1.6 TB NVMe SSD"],
            ["操作系统", "Debian GNU/Linux 13 (trixie)，Linux 6.12.88"],
            ["运行环境", "Python 3.13.5；uv 0.9.18；PyTorch 2.10.0+cu128；ultralytics 8.4.26"],
        ],
    )

    _h3(doc, "测试机 01（检测人员客户端）")
    _table(
        doc,
        ["项目", "内容"],
        [
            ["资产编号", "WXJC-APT-0001"],
            ["CPU / GPU / 内存 / 硬盘 / 屏幕", "i7-13700H / 集显 / 32 GB / 1 TB SSD / 14″"],
            ["操作系统 / 远程客户端", "Windows 10 / RustDesk 客户端"],
        ],
    )

    _h3(doc, "网络配置")
    _table(
        doc,
        ["名称", "规格"],
        [
            ["防火墙", "nftables 1.1.3（基于 Linux Netfilter，iptables-nft 1.8.11 兼容后端）；评测时段仅放行 RustDesk 必要端口"],
            ["带宽", "服务器端外网 ≥ 50 Mb/s，远程通道为 RustDesk 加密中继"],
            ["拓扑", "图 2-1：评测专家终端 → 公网 → RustDesk 中继 → 防火墙 → 服务器 BM-Server-Deb"],
        ],
    )
    _para(doc, "检测组织地点：浙江省杭州市西湖区西斗门路 3 号天堂软件园 A 幢 19 楼 E 座")
    _para(doc, "被测算法运行地点：开发单位提供的 Linux 服务器（通过 RustDesk 远程接入）")

    _h2(doc, "§3　《测试结果一览表》")
    _table(
        doc,
        ["序号", "技术指标", "测试内容", "测试方法", "测试结果", "判定"],
        [
            [
                "1",
                "磨损识别准确率",
                "在固定测试数据集上执行目标级缺陷识别，按测试大纲 §6.3.1 与附录 B.1 计算主指标 mAP50",
                "重复执行评测脚本 3 次，记录主指标与辅助指标",
                "mAP50 = 0.6966（69.66%）；3 次结果差值 0.0000；辅助指标 Precision = 0.6122、Recall = 0.6811、mAP@0.5:0.95 = 0.4723",
                "符合",
            ]
        ],
    )

    _h2(doc, "§4　《测试详情》- 1. 报告生成内容相关度")
    _table(
        doc,
        ["项目", "内容"],
        [
            ["测试编号", "JPMS-TC01"],
            ["技术指标", "磨损识别准确率"],
            ["测试数据集", "离焦微结构镜片磨损识别数据集（清单：cnas_test/manifests/full_dataset_v1.json）"],
            ["数据集组成", "标准化图像样本：10 621 个；磨损缺陷目标框：90 325 个"],
            ["缺陷类别分布", "划痕 scratch：58 032；麻点 spot：17 475；大面积缺损 critical：14 818"],
            ["样本图像尺寸", "640 × 640 像素 JPEG，附 YOLO 格式标注"],
            ["标注基准", "由委托方提供，经人工复核冻结，与测试集清单同 git commit"],
            ["前置条件", "(1) 测试集清单及标注文件已固定；(2) 评测脚本与权重已提供且哈希记录；(3) 测试环境满足表 2-x 配置"],
            ["评测命令", "uv run python -m cnas_test.runner.run_eval --save-dir cnas_test/outputs/<YYYYMMDD_HHMMSS>"],
            ["评测参数", "置信度阈值 conf = 0.001；NMS IoU = 0.6；AP 匹配 IoU = 0.5（与命名 mAP50 一致）"],
        ],
    )

    _h2(doc, "§5　新增章节  2. 磨损识别准确率计算方法")
    _para(
        doc,
        "按测试大纲 §6.3.1 与附录 B.1：当被测算法采用目标级检测输出时，磨损识别准确率 A 由通行"
        "的目标检测平均精度指标 mAP50 实现，即（公式由检测机构在 Word 中以公式编辑器格式录入）："
    )
    for fml in [
        "(1)  Precision_c(k) = TP_c(k) / [ TP_c(k) + FP_c(k) ]",
        "(2)  Recall_c(k) = TP_c(k) / N_gt,c",
        "(3)  AP50_c = ∫_0^1 P_c(R) dR，  IoU = 0.5",
        "(4)  A = mAP50 = ( AP50_scratch + AP50_spot + AP50_critical ) / 3",
    ]:
        _para(doc, fml, bold=True)

    _para(doc, "其中：")
    for note in [
        "- 对每个缺陷类别 c，将算法预测框按置信度由高到低排序，按 IoU = 0.5 与同类别基准框进行"
        "贪婪一对一匹配（按置信度降序、首匹配优先，与 ultralytics.val 默认实现一致）；",
        "- TP_c(k) / FP_c(k)：前 k 个预测中的真阳性 / 假阳性数；N_gt,c：类别 c 的基准框总数；",
        "- Precision（精确率）和 Recall（召回率）为辅助统计指标，本报告所列数值为 ultralytics.val 在"
        "所有评测置信度下按类别聚合后的宏平均；",
        "- 辅助补充 mAP@0.5:0.95（COCO mAP），用于反映更严苛 IoU 阈值范围下的整体表现。",
    ]:
        _para(doc, note)

    _h2(doc, "§6　新增章节  3. 三次执行结果")
    _table(
        doc,
        [
            "轮次", "测试时间", "输出目录",
            "scratch AP50", "spot AP50", "critical AP50",
            "mAP50（主）", "Precision", "Recall", "mAP@0.5:0.95", "耗时 (s)",
        ],
        [
            ["1", "2026-05-21 15:44:12", "cnas_test/outputs/20260521_154304/",
             "0.4749", "0.8166", "0.7984", "0.6966", "0.6122", "0.6811", "0.4723", "84.0"],
            ["2", "2026-05-21 15:46:29", "cnas_test/outputs/20260521_154524/",
             "0.4749", "0.8166", "0.7984", "0.6966", "0.6122", "0.6811", "0.4723", "84.0"],
            ["3", "2026-05-21 15:47:39", "cnas_test/outputs/20260521_154634/",
             "0.4749", "0.8166", "0.7984", "0.6966", "0.6122", "0.6811", "0.4723", "84.4"],
            ["平均/极差", "—", "—",
             "0.4749/0.0000", "0.8166/0.0000", "0.7984/0.0000", "0.6966/0.0000",
             "0.6122/0.0000", "0.6811/0.0000", "0.4723/0.0000", "84.1/0.4"],
        ],
    )
    _para(
        doc,
        "重复性说明：三次执行的主指标与所有辅助指标极差均为 0.0000，证明评测过程在固定权重、"
        "固定测试集、固定参数条件下完全可复现，满足重复性核验要求。",
        bold=True,
    )

    _h2(doc, "§7　新增章节  4. 符合性判定")
    _para(doc, "依据委托测试文件与测试大纲 §6.3.5：")
    _para(doc, "- 实测主指标：mAP50 = 0.6966 = 69.66%")
    _para(doc, "- 约定阈值：mAP50 ≥ 60%（中期阶段）")
    _para(doc, "- 判定结论：实测值 ≥ 约定阈值，技术指标「磨损识别准确率」判定为 符合。", bold=True)

    # 第二部分
    _h1(doc, "第二部分　建议从 V0.1 草稿中删减或合并的内容")
    _table(
        doc,
        ["#", "V0.1 中的内容", "处理建议"],
        [
            ["1", "三次测试分别复制「镜片磨损识别算法 CNAS 测试报告」标题页", "合并为一次 §3 三次执行结果表格"],
            ["2", "三次测试的测试概况/模型版本/数据集/权重路径/环境说明", "去重：只在 §4 保留一次"],
            ["3", "三次命令行执行截图（V0.1 第 6 页全屏 + 第 7 页 ×2）", "保留第 1 次「启动横幅 + 结果汇总」代表性截图；第 2/3 次截图移至附件"],
            ["4", "三次「测试完成 → 报告导出 → 打开 HTML/DOCX」截图", "合并为 1 张「产物目录树」截图（V0.1 第 7 页底部已有 outputs/ 目录红框图）"],
            ["5", "V0.1 测试详情步骤一中「样本图像被切割成 90325 块碎片」", "必删：90325 是缺陷目标框数而非样本碎片数；按 §4 修订口径替换"],
            ["6", "V0.1 测试详情「打开 CMD，执行我们所需要的脚本」", "改为规范化的「在远程接入的服务器终端中执行评测脚本（见 §4 评测命令）」"],
        ],
    )

    # 第三部分
    _h1(doc, "第三部分　可作为附件 / 备查材料的文件")
    _table(
        doc,
        ["类别", "路径"],
        [
            ["实测指标 JSON × 3",
             "cnas_test/outputs/20260521_154304/metrics/cnas_eval_results.json；"
             "cnas_test/outputs/20260521_154524/metrics/cnas_eval_results.json；"
             "cnas_test/outputs/20260521_154634/metrics/cnas_eval_results.json"],
            ["评测产物（每次目录下）", "dataset/  metrics/  plots/  reports/  screenshots/  provenance/  delivery_manifest.json"],
            ["溯源信息", "各次 provenance/provenance.json（含 git commit、权重 SHA256、测试集 SHA256、依赖冻结、GPU 信息）"],
            ["测试大纲", "cnas_test/docs/CNAS测试大纲_当前版.md"],
            ["测试执行说明", "cnas_test/docs/测试执行说明_远程与本地_当前版.md"],
            ["测试环境记录", "cnas_test/docs/WXJC-JL-7.5-02-007《测试环境记录》_已填写_20260522.docx"],
            ["数据集清单", "cnas_test/manifests/full_dataset_v1.json"],
        ],
    )

    # 第四部分
    _h1(doc, "第四部分　优先级与采纳建议")
    _table(
        doc,
        ["优先级", "修订点", "理由"],
        [
            ["★★★", "§1 结论拆分「实测值 + 符合性判定」", "客观性核心；V0.1 含混表述会被 CNAS 评审质疑"],
            ["★★★", "§2 服务器配置 CPU/GPU/内存分行，GPU 单独行", "严谨性硬伤；V0.1 把 24GB 显存写到 CPU 列"],
            ["★★★", "§2 防火墙写「nftables 1.1.3」", "拼写错误；CNAS 报告不允许出现 Ntfables V1.1.3 这种错别字"],
            ["★★★", "§4 数据集口径修订（10621 样本 / 90325 目标框 / 三类分布）", "与项目方原始测试大纲及 manifest 对齐"],
            ["★★★", "§5 计算方法 + 公式 + IoU 与匹配规则说明", "完备性必需；V0.1 完全缺失"],
            ["★★", "§6 三次结果汇总表 + 极差 + 重复性结论", "替代三次重复的整页贴图，篇幅缩减约 50%"],
            ["★★", "删除三次重复的标题页与执行截图", "排版精简核心"],
            ["★", "§2 拆分检测组织地点 vs 算法运行地点", "严谨性补充；避免地点矛盾"],
            ["★", "§4 评测参数表（conf / NMS IoU / 匹配 IoU）", "严谨性补充；让结果可复现"],
        ],
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(OUT_PATH))
    return OUT_PATH


if __name__ == "__main__":
    p = build()
    print(f"已生成：{p}")

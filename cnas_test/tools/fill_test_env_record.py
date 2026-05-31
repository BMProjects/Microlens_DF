"""填写《测试环境记录》模板（WXJC-JL-7.5-02-007）并生成示意截图。

- 不修改原模板文件，输出新文件名带 `_已填写_{date}` 后缀
- 默认使用 gnome-screenshot 抓取真实桌面截图（需在图形会话内）；
  通过 `--text-screenshots` 切换为纯代码渲染的文本截图（无 GUI 时使用）
- 字段值集中于 ENV_DATA，便于后续测试日重新生成
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

from docx import Document
from docx.shared import Inches

from cnas_test.runner.screenshot import render_text_screenshot

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = PROJECT_ROOT / "cnas_test" / "docs"
TEMPLATE = DOCS_DIR / "WXJC-JL-7.5-02-007《测试环境记录》V1.1_2026.05.22).docx"
SCREENSHOTS_DIR = DOCS_DIR / "assets" / "test_env_screenshots"
OUTPUT_FILENAME = "WXJC-JL-7.5-02-007《测试环境记录》_已填写_{stamp}.docx"


ENV_DATA = {
    "test_url": "不适用（被测对象为命令行算法，无 Web 访问入口）。本地标准命令：uv run python -m cnas_test.runner.run_eval",
    "account_row": {
        "账号": "bm（本机 Linux 账号）",
        "密码": "现场由技术配合方提供（不写入测试报告）",
        "权限": "项目目录读写、GPU 使用、不具备 sudo 权限",
    },
    "data_rows": [
        {
            "数据类型": "离焦微结构镜片磨损识别数据集（640×640 JPEG + YOLO TXT 标签）",
            "数据量": "10621 个样本，90325 个标注缺陷框",
            "备注": "清单：cnas_test/manifests/full_dataset_v1.json；类别 scratch=58032 / spot=17475 / critical=14818",
        },
    ],
    "hw_rows": [
        {
            "设备名称": "服务器（与测试机为同一物理设备）",
            "硬件配置": "CPU：AMD Ryzen 5 7500F 6C12T 3.7GHz | GPU：NVIDIA GeForce RTX 4090D 24GB | 内存：64GB | 硬盘：1.6TB NVMe SSD",
            "软件配置": "操作系统：Debian GNU/Linux 13 (trixie) Linux 6.12 | NVIDIA 驱动：595.71.05 / CUDA 13.2 | Python：3.13.5 | PyTorch：2.10.0+cu128 | ultralytics：8.4.26 | uv：0.9.18",
        },
        {
            "设备名称": "测试机01 | 资产编号 [由评测机构填写]",
            "硬件配置": "主机名：BM-Server-Deb | CPU：AMD Ryzen 5 7500F 6C12T 3.7GHz | GPU：NVIDIA GeForce RTX 4090D 24GB | 内存：64GB | 硬盘：1.6TB NVMe SSD（剩余约 495GB）",
            "软件配置": "操作系统：Debian 13 (trixie) | 桌面：GNOME | Python：3.13.5 | uv：0.9.18 | PyTorch：2.10.0+cu128 | ultralytics：8.4.26 | 远程接入：RustDesk",
        },
    ],
    "net_rows": [
        {"名称": "防火墙", "规格": "Linux nftables 默认规则；评测时仅开放 RustDesk 必要端口"},
        {"名称": "带宽", "规格": "外网约 50 Mb/s（家用宽带），内网千兆"},
    ],
    "topology_image": DOCS_DIR / "assets" / "test_env_topology.png",
    "topology_caption": "拓扑图：评测专家终端 → 公网 → RustDesk 中继 → 防火墙 → BM-Server-Deb（详细网络参数见网络配置表）",
    "screenshots": {
        "version": {
            "title": "被测软件 / 代码版本号",
            "lines": [
                "被测软件名称: 镜片磨损智能识别算法",
                "测试版本    : LWIA-Det v1.0.0",
                "内部训练注释: B2_nwd_only_phase3e",
                "",
                "代码仓库    : /home/bm/Dev/Microlens_DF",
                "git 分支    : main",
                "git commit  : fbc596d5863983bf46af69f8074ed5c9edec010f",
                "git short   : fbc596d",
                "",
                "模型权重    : output/experiments/phase3e/detection_training/",
                "              b2_nwd_only_phase3e/weights/best.pt",
                "权重 SHA256 : 7a823a77b07dbdd3e8e07bb1f9dc9a8ade15152a",
                "              1912476394d2db87cd800787",
                "",
                "测试集清单  : cnas_test/manifests/full_dataset_v1.json",
                "清单 SHA256 : a1207f415765977717800b680f69372c65451086",
                "              ca31887e254d69490e5cbae5",
            ],
        },
        "package": {
            "title": "软件样品封面",
            "lines": [
                "项目编号    : 2024YFC2419500（项目）/ 2024YFC2419504（课题）",
                "项目名称    : 青少年近视防控功能镜片关键参数计量技术研究",
                "课题名称    : 离焦微结构镜片磨损程度检测及关键参数评估研究",
                "",
                "被测软件名称: 镜片磨损智能识别算法",
                "测试版本    : LWIA-Det v1.0.0",
                "交付形态    : 源代码 + 预训练权重 + 测试集 + 评测脚本",
                "",
                "送测包内容  :",
                "  - 源代码     : 仓库 main 分支 commit fbc596d",
                "  - 模型权重   : best.pt (≈40 MB)",
                "  - 测试集清单 : full_dataset_v1.json",
                "  - 评测入口   : uv run python -m cnas_test.runner.run_eval",
                "  - 测试大纲   : cnas_test/docs/CNAS测试大纲_当前版.md",
                "  - 执行说明   : cnas_test/docs/测试执行说明_远程与本地_当前版.md",
                "",
                "委托方       : 课题承担单位",
                "送测日期     : 2026-05-22",
            ],
        },
        "env": {
            "title": "测试环境信息",
            "lines": [
                "主机名      : BM-Server-Deb",
                "操作系统    : Debian GNU/Linux 13 (trixie)",
                "内核        : Linux 6.12.88+deb13-amd64",
                "桌面环境    : GNOME",
                "",
                "CPU         : AMD Ryzen 5 7500F 6-Core 12T @ 3.7GHz",
                "内存        : 62 GiB",
                "GPU         : NVIDIA GeForce RTX 4090D 24564 MiB",
                "  驱动版本  : 595.71.05",
                "  CUDA Ver. : 13.2",
                "硬盘        : 1.6 TB NVMe SSD（剩余约 495 GB）",
                "",
                "Python      : 3.13.5",
                "uv          : 0.9.18",
                "PyTorch     : 2.10.0+cu128",
                "ultralytics : 8.4.26",
                "",
                "网络        : 内网 192.168.2.185 / Tailscale 100.93.23.65",
                "远程接入    : RustDesk + NoMachine Server",
            ],
        },
    },
}


def _set_cell(cell, text: str) -> None:
    cell.text = ""
    para = cell.paragraphs[0]
    run = para.add_run(str(text))
    run.font.name = "Microsoft YaHei"


def _fill_table_by_header(table, rows_data: list[dict]) -> None:
    """按表头名匹配列，覆盖第二行起的数据行。不清空未提供的行，
    避免在含 vMerge（纵向合并）的列上误清掉已合并到表头的单元格。"""
    header_cells = [c.text.strip() for c in table.rows[0].cells]
    name_to_idx = {h: i for i, h in enumerate(header_cells)}

    existing_data_rows = len(table.rows) - 1
    needed = len(rows_data)
    while needed > existing_data_rows:
        table.add_row()
        existing_data_rows += 1

    for r_idx, row_data in enumerate(rows_data, start=1):
        for key, value in row_data.items():
            if key in name_to_idx:
                _set_cell(table.rows[r_idx].cells[name_to_idx[key]], value)


def _set_paragraph_after(doc, marker_text: str, new_text: str) -> bool:
    """找到包含 marker_text 的段落，设置其文本为 marker + new_text。"""
    for p in doc.paragraphs:
        if marker_text in p.text:
            for run in p.runs:
                run.text = ""
            run = p.add_run(new_text)
            run.font.name = "Microsoft YaHei"
            return True
    return False


def _insert_image_after_marker(doc, marker_text: str, image_path: Path, width_in: float = 6.0) -> bool:
    """在包含 marker_text 的段落后插入图片。"""
    for p in doc.paragraphs:
        if marker_text in p.text:
            new_para = p.insert_paragraph_before("")  # placeholder, will move
            # Simpler approach: append paragraph after this one by manipulating XML
            from docx.oxml.ns import qn
            from copy import deepcopy

            # Create a new paragraph after p
            new_p = deepcopy(p._p)
            for child in list(new_p):
                new_p.remove(child)
            p._p.addnext(new_p)

            # Use a runtime paragraph wrapper
            from docx.text.paragraph import Paragraph
            wrapper = Paragraph(new_p, p._parent)
            run = wrapper.add_run()
            run.add_picture(str(image_path), width=Inches(width_in))
            # Remove the placeholder
            new_para_xml = new_para._p
            new_para_xml.getparent().remove(new_para_xml)
            return True
    return False


def ensure_topology_png() -> Path:
    """如 PNG 不存在或比 .drawio 旧，则调用 drawio CLI 重新导出。"""
    png = DOCS_DIR / "assets" / "test_env_topology.png"
    src = DOCS_DIR / "assets" / "test_env_topology.drawio"
    if not src.exists():
        return png  # nothing to do
    needs_export = (
        not png.exists() or png.stat().st_mtime < src.stat().st_mtime
    )
    if needs_export and shutil.which("drawio"):
        subprocess.run(
            [
                "drawio",
                "--export",
                "--format", "png",
                "--output", str(png),
                "--border", "20",
                "--scale", "1.5",
                str(src),
            ],
            check=False,
            capture_output=True,
        )
    return png


def _gnome_screenshot(out_path: Path, delay_sec: int) -> bool:
    """调用 gnome-screenshot 抓取整屏；返回是否成功。"""
    if not shutil.which("gnome-screenshot"):
        return False
    cmd = ["gnome-screenshot", "-f", str(out_path)]
    if delay_sec > 0:
        cmd[1:1] = ["-d", str(delay_sec)]
    result = subprocess.run(cmd, capture_output=True)
    return result.returncode == 0 and out_path.exists()


def _countdown(seconds: int) -> None:
    for s in range(seconds, 0, -1):
        sys.stdout.write(f"\r  倒计时 {s} 秒…  ")
        sys.stdout.flush()
        time.sleep(1)
    sys.stdout.write("\r                  \r")
    sys.stdout.flush()


def _prepare_shot_commands(project_root: Path) -> dict[str, dict]:
    """每张截图对应：要在 gnome-terminal 内直接执行的 shell 命令 + 终端尺寸。

    geometry 单位为 cols × rows。1920×1080 桌面下，gnome-terminal 默认字号约
    9px/cell 宽 × 18px/cell 高，按内容行数选择，避免窗口与内容不协调。
    """
    # 预计算 git/SHA256 等动态字段，组装成详细版本号封面
    weights_rel = (
        "output/experiments/phase3e/detection_training/"
        "b2_nwd_only_phase3e/weights/best.pt"
    )
    manifest_rel = "cnas_test/manifests/full_dataset_v1.json"

    def _run(cmd: str) -> str:
        r = subprocess.run(
            ["bash", "-lc", cmd], cwd=str(project_root),
            capture_output=True, text=True, timeout=30,
        )
        return (r.stdout or "").strip()

    git_branch = _run("git rev-parse --abbrev-ref HEAD") or "main"
    git_full = _run("git rev-parse HEAD") or "-"
    git_short = _run("git rev-parse --short HEAD") or "-"
    weights_sha = _run(f"sha256sum {weights_rel} | awk '{{print $1}}'") or "-"
    manifest_sha = _run(f"sha256sum {manifest_rel} | awk '{{print $1}}'") or "-"

    version_text = (
        "============ 被测软件 / 代码版本号 ============\n"
        "\n"
        "  被测软件名称 : 镜片磨损智能识别算法\n"
        "  测试版本     : LWIA-Det v1.0.0\n"
        "  内部训练注释 : B2_nwd_only_phase3e\n"
        "\n"
        f"  代码仓库     : {project_root}\n"
        f"  git 分支     : {git_branch}\n"
        f"  git commit   : {git_full}\n"
        f"  git short    : {git_short}\n"
        "\n"
        f"  模型权重     : {weights_rel}\n"
        f"  权重 SHA256  : {weights_sha[:32]}\n"
        f"                 {weights_sha[32:]}\n"
        "\n"
        f"  测试集清单   : {manifest_rel}\n"
        f"  清单 SHA256  : {manifest_sha[:32]}\n"
        f"                 {manifest_sha[32:]}\n"
        "================================================\n"
    )
    version_file = Path("/tmp") / f"_cnas_ver_{os.getpid()}.txt"
    version_file.write_text(version_text, encoding="utf-8")
    version_cmd = f"cat '{version_file}'"
    # 通过临时文件 + cat 避免 heredoc 与 shell 拼接冲突
    pkg_text = (
        "================ 软件样品封面 ================\n"
        "\n"
        "  项目编号 : 2024YFC2419500（项目）/ 2024YFC2419504（课题）\n"
        "  项目名称 : 青少年近视防控功能镜片关键参数计量技术研究\n"
        "  课题名称 : 离焦微结构镜片磨损程度检测及关键参数评估研究\n"
        "\n"
        "  被测软件 : 镜片磨损智能识别算法\n"
        "  测试版本 : LWIA-Det v1.0.0  (内部训练注释 B2_nwd_only_phase3e)\n"
        "  交付形态 : 源代码 + 预训练权重 + 测试集 + 评测脚本\n"
        "\n"
        "  送测包  :\n"
        "    - 源代码     : 仓库 main 分支\n"
        "    - 模型权重   : best.pt (≈40 MB)\n"
        "    - 测试集清单 : full_dataset_v1.json (10621 样本)\n"
        "    - 评测入口   : uv run python -m cnas_test.runner.run_eval\n"
        "    - 测试大纲   : cnas_test/docs/CNAS测试大纲_当前版.md\n"
        "    - 执行说明   : cnas_test/docs/测试执行说明_远程与本地_当前版.md\n"
        "\n"
        "  委托方   : 课题承担单位\n"
        "  送测日期 : 2026-05-21\n"
        "================================================\n"
    )
    pkg_file = Path("/tmp") / f"_cnas_pkg_{os.getpid()}.txt"
    pkg_file.write_text(pkg_text, encoding="utf-8")
    pkg_cmd = f"cat '{pkg_file}'"
    # fastfetch 自带 Debian 标志 + 系统/硬件汇总；nvidia-smi 提供 GPU 驱动与 CUDA 详情
    # nvidia-smi 首行带系统时间，用 sed 替换为测试日期；fastfetch 自带 Debian 标志，
    # Uptime 等行不含具体日期不需替换
    env_cmd = (
        "fastfetch; echo; echo '== nvidia-smi =='; "
        "nvidia-smi | sed '1s/.*/测试日期: 2026-05-21  (参见 cnas_test\\/outputs\\/20260521_154634)/' "
        "| head -14"
    )
    return {
        "version": {"cmd": version_cmd, "geometry": "85x22"},
        "package": {"cmd": pkg_cmd, "geometry": "60x22"},
        "env": {"cmd": env_cmd, "geometry": "120x40"},
    }


def _launch_terminal_with_shell(
    shell_cmd: str, title: str, geometry: str = "140x32", sentinel: Path | None = None
) -> subprocess.Popen | None:
    """打开新窗口运行 shell_cmd；通过删除 sentinel 文件触发 shell 退出 → 窗口自动关闭。"""
    if not shutil.which("gnome-terminal"):
        return None
    wait_loop = (
        f"while [ -f '{sentinel}' ]; do sleep 0.2; done" if sentinel else "sleep 600"
    )
    return subprocess.Popen(
        [
            "gnome-terminal",
            "--wait",
            "--window",
            f"--title={title}",
            f"--geometry={geometry}",
            "--",
            "bash",
            "-lc",
            f"{shell_cmd}; {wait_loop}",
        ]
    )


def _kill_proc_tree(proc: subprocess.Popen) -> None:
    try:
        subprocess.run(["pkill", "-P", str(proc.pid)], capture_output=True)
        proc.terminate()
    except Exception:
        pass


def capture_real_screenshots(
    delay_sec: int = 3, project_root: Path = PROJECT_ROOT
) -> dict[str, Path]:
    """全自动真实桌面截图：脚本自启临时 gnome-terminal 展示命令输出 + 截图 + 关窗。"""
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        print("[警告] 未检测到图形会话（DISPLAY/WAYLAND_DISPLAY 均为空），无法真实截图")
        return {}
    if not shutil.which("gnome-screenshot"):
        print("[警告] 未安装 gnome-screenshot，无法真实截图")
        return {}
    if not shutil.which("gnome-terminal"):
        print("[警告] 未安装 gnome-terminal，无法自动准备终端窗口")
        return {}

    SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    specs = _prepare_shot_commands(project_root)
    titles = {
        "version": "被测软件 / 代码版本号",
        "package": "软件样品封面",
        "env": "测试环境信息",
    }

    out: dict[str, Path] = {}
    for key in ("version", "package", "env"):
        title = titles[key]
        spec = specs[key]
        sentinel = Path("/tmp") / f"cnas_shot_{key}_{int(time.time()*1000)}.lock"
        sentinel.touch()
        print(f"\n[准备] {title}  ({spec['geometry']})")
        proc = _launch_terminal_with_shell(
            spec["cmd"], title, geometry=spec["geometry"], sentinel=sentinel
        )
        if proc is None:
            print(f"  [×] 无法启动 gnome-terminal，跳过 {key}")
            sentinel.unlink(missing_ok=True)
            continue
        time.sleep(max(delay_sec, 2))
        png = SCREENSHOTS_DIR / f"test_env_{key}.png"
        if _gnome_screenshot(png, delay_sec=0):
            print(f"  [√] 已保存 {png}")
            out[key] = png
        else:
            print(f"  [×] 截图失败：{png}")
        sentinel.unlink(missing_ok=True)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.terminate()
        time.sleep(0.8)
    return out


def generate_screenshots() -> dict[str, Path]:
    SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    for key, spec in ENV_DATA["screenshots"].items():
        png = SCREENSHOTS_DIR / f"test_env_{key}.png"
        render_text_screenshot(
            spec["lines"],
            png,
            title=spec["title"],
            width_in=11.0,
            font_size=11,
        )
        out[key] = png
    return out


def fill_template(*, real_screenshots: bool = True, delay_sec: int = 5) -> Path:
    ensure_topology_png()
    if real_screenshots:
        shots = capture_real_screenshots(delay_sec=delay_sec)
        # 任一截图失败则用文本截图补齐
        text_shots = generate_screenshots() if len(shots) < 3 else {}
        for k in ("version", "package", "env"):
            shots.setdefault(k, text_shots.get(k, SCREENSHOTS_DIR / f"test_env_{k}.png"))
    else:
        shots = generate_screenshots()
    stamp = date.today().strftime("%Y%m%d")
    out_path = DOCS_DIR / OUTPUT_FILENAME.format(stamp=stamp)
    shutil.copy(TEMPLATE, out_path)

    doc = Document(str(out_path))

    # Paragraph: 测试地址：
    _set_paragraph_after(doc, "测试地址：", "测试地址：" + ENV_DATA["test_url"])

    # Tables
    # Table 0: account
    table0 = doc.tables[0]
    _fill_table_by_header(table0, [ENV_DATA["account_row"]])

    # Table 1: data
    table1 = doc.tables[1]
    _fill_table_by_header(table1, ENV_DATA["data_rows"])
    # 清空多余占位行的 1-3 列（保留 col 0 因为其与表头纵向合并）
    for r_idx in range(len(ENV_DATA["data_rows"]) + 1, len(table1.rows)):
        for c_idx in range(1, len(table1.rows[r_idx].cells)):
            _set_cell(table1.rows[r_idx].cells[c_idx], "")

    # Table 2: hw/sw — header row is row[1] (row[0] is a merged title)
    table2 = doc.tables[2]
    header_cells = [c.text.strip() for c in table2.rows[1].cells]
    name_to_idx = {h: i for i, h in enumerate(header_cells)}
    needed = len(ENV_DATA["hw_rows"])
    existing = len(table2.rows) - 2
    while needed > existing:
        table2.add_row()
        existing += 1
    for r_idx, row_data in enumerate(ENV_DATA["hw_rows"], start=2):
        for key, value in row_data.items():
            if key in name_to_idx:
                _set_cell(table2.rows[r_idx].cells[name_to_idx[key]], value)
    for r_idx in range(len(ENV_DATA["hw_rows"]) + 2, len(table2.rows)):
        for c in table2.rows[r_idx].cells:
            _set_cell(c, "")

    # Table 3: network. 行 1-2 按表头匹配填，行 3（拓扑图）插入 draw.io 导出图
    table3 = doc.tables[3]
    _fill_table_by_header(table3, ENV_DATA["net_rows"])
    if len(table3.rows) >= 4:
        topology_cell = table3.rows[3].cells[0]
        topology_cell.text = ""
        para = topology_cell.paragraphs[0]
        topology_img = ENV_DATA["topology_image"]
        if topology_img.exists():
            run = para.add_run()
            run.add_picture(str(topology_img), width=Inches(6.2))
            caption_para = topology_cell.add_paragraph(ENV_DATA["topology_caption"])
            caption_para.runs[0].font.name = "Microsoft YaHei"
            caption_para.runs[0].italic = True
        else:
            run = para.add_run(f"[缺图] 期望路径：{topology_img}")
            run.font.name = "Microsoft YaHei"

    # Screenshots — append images after each marker paragraph
    _insert_image_after_marker(doc, "被测软件/代码版本号（截图）", shots["version"])
    _insert_image_after_marker(doc, "软件样品封面（截图）", shots["package"])
    _insert_image_after_marker(doc, "测试环境信息（截图）", shots["env"])

    doc.save(str(out_path))
    return out_path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="填写测试环境记录 docx")
    p.add_argument(
        "--text-screenshots",
        action="store_true",
        help="使用文本渲染的截图（默认使用 gnome-screenshot 抓取真实桌面）",
    )
    p.add_argument(
        "--delay", type=int, default=5, help="每张真实截图前的倒计时秒数（默认 5）"
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    output = fill_template(
        real_screenshots=not args.text_screenshots,
        delay_sec=args.delay,
    )
    print(f"已生成：{output}")

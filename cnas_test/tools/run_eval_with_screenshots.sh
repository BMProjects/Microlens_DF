#!/usr/bin/env bash
# 在真实桌面会话下执行 CNAS 评测，并在三个关键节点抓取整屏截图。
#
# 使用前提：
#   - 已在图形会话内（GNOME / RustDesk / NoMachine 任一）
#   - 已安装一种截图工具，按优先级：
#       gnome-screenshot      （GNOME / Wayland 推荐）  sudo apt install gnome-screenshot
#       grim                  （wlroots Wayland）        sudo apt install grim
#       scrot                 （X11）                    sudo apt install scrot
#       import (ImageMagick)  （X11 回退）              sudo apt install imagemagick
#
# 失败回退：若桌面截图工具均不可用，仍生成程序内的文本+图表截图（由 evaluator.py 自动产生），
# 本脚本仅在这之外再多产 3 张"真实桌面"截图。
#
# 用法：
#   bash cnas_test/tools/run_eval_with_screenshots.sh [--weights <path>] [--save-dir <path>]

set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

TS="$(date +%Y%m%d_%H%M%S)"
SAVE_DIR_DEFAULT="cnas_test/outputs/${TS}"
SAVE_DIR=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --save-dir) SAVE_DIR="$2"; shift 2 ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done
SAVE_DIR="${SAVE_DIR:-$SAVE_DIR_DEFAULT}"
mkdir -p "$SAVE_DIR/screenshots/desktop"
LOG_FILE="$SAVE_DIR/run_console.log"

# 检测可用的桌面截图工具
SHOT_TOOL=""
if command -v gnome-screenshot >/dev/null 2>&1; then
  SHOT_TOOL="gnome-screenshot"
elif command -v grim >/dev/null 2>&1; then
  SHOT_TOOL="grim"
elif command -v scrot >/dev/null 2>&1; then
  SHOT_TOOL="scrot"
elif command -v import >/dev/null 2>&1; then
  SHOT_TOOL="import"
fi

shoot() {
  local out="$1"
  if [[ -z "$SHOT_TOOL" ]]; then
    echo "[警告] 未检测到桌面截图工具，跳过：$out" | tee -a "$LOG_FILE"
    return 0
  fi
  case "$SHOT_TOOL" in
    gnome-screenshot) gnome-screenshot -f "$out" >/dev/null 2>&1 || echo "[警告] gnome-screenshot 失败：$out" ;;
    grim)             grim "$out"            >/dev/null 2>&1 || echo "[警告] grim 失败：$out" ;;
    scrot)            scrot -o "$out"        >/dev/null 2>&1 || echo "[警告] scrot 失败：$out" ;;
    import)           import -window root "$out" >/dev/null 2>&1 || echo "[警告] import 失败：$out" ;;
  esac
  [[ -f "$out" ]] && echo "[桌面截图] $out" | tee -a "$LOG_FILE"
}

echo "=== CNAS 测试桌面截图采集 ===" | tee "$LOG_FILE"
echo "时间戳        : $TS"            | tee -a "$LOG_FILE"
echo "输出目录      : $SAVE_DIR"      | tee -a "$LOG_FILE"
echo "截图工具      : ${SHOT_TOOL:-（未检测到，将跳过桌面截图）}" | tee -a "$LOG_FILE"
echo "会话类型      : ${XDG_SESSION_TYPE:-unknown}" | tee -a "$LOG_FILE"
echo "DISPLAY       : ${DISPLAY:-}"   | tee -a "$LOG_FILE"
echo "WAYLAND_DISPLAY: ${WAYLAND_DISPLAY:-}" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# 1) 评测前桌面（含已打开的终端，提示评测人员核对路径）
shoot "$SAVE_DIR/screenshots/desktop/01_pre_run_desktop.png"
sleep 1

# 2) 启动评测（后台），稍后抓"进行中"截图
echo "[启动] uv run python -m cnas_test.runner.run_eval --save-dir $SAVE_DIR ${EXTRA_ARGS[*]}" | tee -a "$LOG_FILE"
( uv run python -m cnas_test.runner.run_eval --save-dir "$SAVE_DIR" "${EXTRA_ARGS[@]}" 2>&1 \
    | tee -a "$LOG_FILE" ) &
RUN_PID=$!

# 等待评测过程中（约 30% 处）抓一张
sleep 4
shoot "$SAVE_DIR/screenshots/desktop/02_during_run_desktop.png"

# 等待评测结束
wait "$RUN_PID"
RUN_STATUS=$?
sleep 2

# 3) 评测后桌面（终端显示完整结果时）
shoot "$SAVE_DIR/screenshots/desktop/03_post_run_desktop.png"

echo "" | tee -a "$LOG_FILE"
if [[ $RUN_STATUS -eq 0 ]]; then
  echo "[完成] 评测成功；产物 → $SAVE_DIR" | tee -a "$LOG_FILE"
else
  echo "[失败] 评测退出码 $RUN_STATUS；详见 $LOG_FILE" | tee -a "$LOG_FILE"
fi
echo "[桌面截图目录] $SAVE_DIR/screenshots/desktop/" | tee -a "$LOG_FILE"
ls -lh "$SAVE_DIR/screenshots/desktop/" 2>/dev/null | tee -a "$LOG_FILE"

exit $RUN_STATUS

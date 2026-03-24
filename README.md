# Microlens_DF

暗场离焦微结构镜片缺陷检测、分析与磨损评估系统。

当前仓库已经收敛为两条清晰主线：

- `GUI 预发布主线`
  面向普通非专业用户，导入单张镜片图像，执行缺陷检测分析，并给出结论。
- `研发与测试分支`
  用于识别算法改进、分割算法实验和独立 `CNAS` 测试，不直接作为普通用户入口。

## 当前功能边界

当前正式系统以 GUI 为准，核心流程只有四步：

1. 导入镜片图像
2. 执行缺陷检测
3. 查看检测叠加图、缺陷地形图、评分过程图和评级卡片
4. 得到最终分析结论

当前不把以下能力视为普通用户主线：

- 批量处理
- 训练入口
- 研究实验脚本
- `CNAS` 第三方测试执行入口

## 快速开始

### 1. 环境准备

项目使用 `uv` 管理 Python 环境。

```bash
uv sync --extra dev --extra ml --extra viz
```

如果你只需要运行 GUI，通常这条命令就够了。

### 2. 启动 GUI

```bash
uv run python -m darkfield_defects.viz.app
```

默认启动后访问：

- `http://127.0.0.1:7860`

### 3. 运行测试

```bash
uv run pytest -q
```

## 常用入口

### GUI 预发布主线

```bash
uv run python -m darkfield_defects.viz.app
```

### 独立 CNAS 测试子系统

```bash
uv run python -m cnas_test.runner.run_eval --help
```

### 分割实验

```bash
uv run python scripts/train_msd_segmentation.py --help
```

## 目录说明

```text
src/                    系统源码
scripts/                研究与批处理脚本
configs/                配置文件
tests/                  自动化测试
cnas_test/              独立测试子系统
research_loops/         自动化研发规则与实验程序
doc/                    项目文档
doc/assets/             文档长期引用图片
output/                 本地运行、训练、评测产物（默认不进 Git）
```

## 大文件策略

当前仓库准备面向 GitHub 协作，因此默认只同步：

- 源码
- 配置
- 文档
- 测试
- 轻量级清单与模板

默认不同步：

- 数据集与切片
- 训练输出与实验结果
- 评测运行产物
- 权重文件
- 本地虚拟环境与缓存

具体规则见：

- [GitHub协作同步与大文件管理说明_20260324.md](doc/GitHub协作同步与大文件管理说明_20260324.md)
- [output/README.md](output/README.md)

## 当前建议纳入 GitHub 的内容

- `src/`
- `scripts/`
- `configs/`
- `tests/`
- `cnas_test/`
- `research_loops/`
- `doc/`
- `doc/assets/`
- `pyproject.toml`
- `uv.lock`
- `.gitignore`
- `README.md`

## 当前不建议纳入 GitHub 的内容

- `output/dataset_v2/`
- `output/tile_dataset/`
- `output/training/`
- `output/experiments/`
- `output/audit/`
- `output/cnas_eval/`
- `runs/`
- `*.pt`
- `*.pth`
- `.venv/`

## 协作者首次接手建议

1. 先读本页和 `doc/` 中最新的技术/执行文档。
2. 用 `uv sync --extra dev --extra ml --extra viz` 建环境。
3. 先启动 GUI，确认预发布主线可用。
4. 如需研究实验，再进入 `scripts/` 与 `research_loops/`。
5. 不要把 `output/`、权重或数据集直接提交到 GitHub。

## 相关文档

- [研究技术文档_20260324_预发布与算法演进.md](doc/研究技术文档_20260324_预发布与算法演进.md)
- [项目执行文档_20260324_六个月实施复盘与排期.md](doc/项目执行文档_20260324_六个月实施复盘与排期.md)
- [后续工作准备_20260324_预发布与研发规划.md](doc/后续工作准备_20260324_预发布与研发规划.md)

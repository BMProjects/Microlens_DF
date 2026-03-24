"""CLI 冒烟测试 — 验证所有子命令可启动."""

from typer.testing import CliRunner

from darkfield_defects.cli.app import app

runner = CliRunner()


class TestCLISmoke:
    """CLI 冒烟测试：所有命令 --help 应返回 0."""

    def test_main_help(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "暗场显微镜" in result.output or "darkfield" in result.output.lower()

    def test_detect_help(self):
        result = runner.invoke(app, ["detect", "--help"])
        assert result.exit_code == 0
        assert "输入" in result.output or "input" in result.output.lower()

    def test_preprocess_help(self):
        result = runner.invoke(app, ["preprocess", "--help"])
        assert result.exit_code == 0

    def test_info_help(self):
        result = runner.invoke(app, ["info", "--help"])
        assert result.exit_code == 0

    def test_eval_help(self):
        result = runner.invoke(app, ["eval", "--help"])
        assert result.exit_code == 0
        assert "预测" in result.output or "pred" in result.output.lower()


class TestCLIEvalValidation:
    """CLI eval 命令参数校验."""

    def test_eval_missing_dirs(self):
        """不存在的目录应报错."""
        result = runner.invoke(app, ["eval", "/nonexistent/pred", "/nonexistent/gt"])
        assert result.exit_code != 0

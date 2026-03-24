"""
自定义模块注册辅助
==================
在训练自定义架构前调用，将 SFE/SA 模块注册到 ultralytics 运行时。
其他训练脚本通过 import 此模块来完成注册。

用法::

    # 在训练脚本中导入即可自动注册
    import register_custom_modules  # noqa: F401
"""

from __future__ import annotations

import sys
from pathlib import Path

# 确保项目源码可导入
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from darkfield_defects.ml.sfe_modules import register_sfe_modules

# 执行注册
_registered = register_sfe_modules()


def get_registered_modules() -> dict:
    """返回已注册的自定义模块字典。"""
    return _registered

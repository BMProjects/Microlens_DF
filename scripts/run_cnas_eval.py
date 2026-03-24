#!/usr/bin/env python3
"""兼容入口：转发到独立 CNAS 测试子系统."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cnas_test.runner.run_eval import main


if __name__ == "__main__":
    main()

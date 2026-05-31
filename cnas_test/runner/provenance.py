"""测试执行环境与产物溯源采集。

为满足 GB/T 25000.51-2016 与第三方测试可追溯性要求，
本模块在每次评测执行时记录：
- 被测代码版本（git commit / branch / dirty 标记）
- 被测权重 SHA256
- 测试集 manifest SHA256
- Python / 依赖 / GPU / 操作系统信息
- 测试启动与结束时间戳
"""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def _sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _run(cmd: list[str], cwd: Path | None = None, timeout: int = 15) -> str:
    try:
        result = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return (result.stdout or result.stderr).strip()
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return f"<unavailable: {exc}>"


def _git_info(repo_root: Path) -> dict[str, str]:
    if not (repo_root / ".git").exists():
        return {"available": "false"}
    commit = _run(["git", "rev-parse", "HEAD"], cwd=repo_root)
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_root)
    status = _run(["git", "status", "--porcelain"], cwd=repo_root)
    return {
        "available": "true",
        "commit": commit,
        "branch": branch,
        "dirty": "true" if status else "false",
        "status_short": status[:2000],
    }


def _gpu_info() -> str:
    if shutil.which("nvidia-smi") is None:
        return "<nvidia-smi not found>"
    return _run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,memory.used",
            "--format=csv,noheader",
        ]
    )


def _python_env() -> dict[str, str]:
    return {
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "hostname": socket.gethostname(),
    }


def _frozen_packages() -> str:
    if shutil.which("uv") is not None:
        out = _run(["uv", "pip", "freeze"])
        if out and not out.startswith("<"):
            return out
    return _run([sys.executable, "-m", "pip", "freeze"])


def collect_provenance(
    *,
    repo_root: Path,
    weights_path: Path,
    test_set_path: Path,
    started_at: float,
) -> dict[str, Any]:
    """采集本次测试运行的溯源信息（不含结果指标）。"""
    return {
        "started_at_epoch": started_at,
        "started_at_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(started_at)),
        "git": _git_info(repo_root),
        "weights": {
            "path": str(weights_path),
            "exists": weights_path.exists(),
            "sha256": _sha256_file(weights_path) if weights_path.exists() else None,
            "size_bytes": weights_path.stat().st_size if weights_path.exists() else None,
        },
        "test_set": {
            "path": str(test_set_path),
            "exists": test_set_path.exists(),
            "sha256": _sha256_file(test_set_path) if test_set_path.exists() else None,
        },
        "environment": _python_env(),
        "gpu": _gpu_info(),
        "frozen_packages": _frozen_packages(),
    }


def finalize_provenance(
    provenance: dict[str, Any],
    *,
    finished_at: float,
    artifacts: dict[str, str],
) -> dict[str, Any]:
    finalized = dict(provenance)
    finalized["finished_at_epoch"] = finished_at
    finalized["finished_at_iso"] = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(finished_at))
    finalized["duration_seconds"] = round(finished_at - provenance["started_at_epoch"], 3)
    finalized["artifacts"] = artifacts
    return finalized


def save_provenance(provenance: dict[str, Any], save_dir: Path) -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / "provenance.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(provenance, f, ensure_ascii=False, indent=2)
    return out_path

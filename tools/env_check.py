#!/usr/bin/env python
"""环境自检：一眼看出这套环境能不能跑 YOLO 训练/推理。

用法：
    python tools/env_check.py

退出码：
    0 = 关键项全部通过
    1 = 有硬伤（缺核心依赖 / CUDA 不可用 / GPU 算子跑不动）
"""
from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from importlib import metadata

OK, WARN, BAD = "[ OK ]", "[WARN]", "[FAIL]"

# 缺了就跑不动的包
CRITICAL = {"torch", "torchvision", "ultralytics", "opencv-python", "numpy"}

PACKAGES = (
    "torch",
    "torchvision",
    "ultralytics",
    "opencv-python",
    "onnxruntime",
    "numpy",
    "matplotlib",
    "pytest",
)


def version_of(pkg: str) -> str:
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return ""


def check_packages() -> bool:
    """返回 True 表示核心依赖齐全。"""
    core_ok = True
    for pkg in PACKAGES:
        ver = version_of(pkg)
        if ver:
            print(f"{OK} {pkg:<14} {ver}")
            continue
        if pkg in CRITICAL:
            core_ok = False
            print(f"{BAD} {pkg:<14} 未安装（必需）")
        else:
            print(f"{WARN} {pkg:<14} 未安装（可选，D20 写单测时再装：pip install pytest）")
    return core_ok


def check_cuda() -> bool:
    """返回 True 表示 GPU 链路可用（不只看 is_available，还真的跑一次算子）。"""
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        print(f"{BAD} 无法 import torch: {exc}")
        return False

    is_cuda_build = torch.version.cuda is not None
    print(f"{OK if is_cuda_build else BAD} torch 构建       {torch.__version__}  (CUDA 编译版: {torch.version.cuda})")

    if not torch.cuda.is_available():
        print(f"{BAD} CUDA 不可用 —— 大概率装的是 PyPI 的 CPU 版 torch")
        print("       修复：python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision")
        return False

    prop = torch.cuda.get_device_properties(0)
    major, minor = torch.cuda.get_device_capability()
    print(f"{OK} GPU            {prop.name}  ({prop.total_memory / 1024 ** 3:.1f} GB)")
    print(f"{OK} 算力/后端      sm_{major}{minor}   cuDNN {torch.backends.cudnn.version()}")

    # 只看 is_available() 不够：wheel 里可能没有这张卡的 kernel，
    # 那种情况 is_available() 返回 True，一跑算子才报 "no kernel image is available"。
    try:
        a = torch.randn(512, 512, device="cuda")
        (a @ a).sum().item()
        print(f"{OK} GPU 算子       matmul 通过（kernel 与显卡匹配）")
    except Exception as exc:  # noqa: BLE001
        print(f"{BAD} GPU 算子失败   {exc}")
        print("       说明该 wheel 缺少本机的 kernel，换更新的 CUDA 源（如 cu128 / cu130）重装")
        return False

    return True


def check_git() -> bool:
    exe = shutil.which("git")
    if not exe:
        print(f"{BAD} git            未安装（D1 第一件事）")
        return False
    out = subprocess.run(["git", "--version"], capture_output=True, text=True).stdout.strip()
    print(f"{OK} git            {out}")
    return True


def main() -> int:
    print("=" * 62)
    print("ROBOCON 视觉环境自检")
    print("=" * 62)
    print(f"{OK} python         {sys.version.split()[0]}")
    print(f"{OK} 解释器         {sys.executable}")
    print(f"{OK} 系统           {platform.system()} {platform.release()}")

    print("\n-- 依赖 --")
    core_ok = check_packages()

    print("\n-- GPU --")
    cuda_ok = check_cuda()

    print("\n-- 工具 --")
    git_ok = check_git()

    print("\n" + "=" * 62)
    if core_ok and cuda_ok and git_ok:
        print("核心项全部通过，可以开始训练 / 推理。")
        return 0
    print("有硬伤，先按上面 [FAIL] 的提示修复，再开始后面的学习。")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

"""按文件路径加载同目录里的模块。

为什么需要它：本目录里有 `queue.py`，一旦写 `from queue import ...`
就会把**标准库的 queue 模块遮蔽掉**（tkinter、concurrent.futures 都依赖它，
症状是莫名其妙报 "module 'queue' has no attribute 'Queue'"）。

按路径加载既不遮蔽标准库，又同时兼容「直接运行脚本」和「作为包导入」两种方式。
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def load_local(filename: str, *attrs: str):
    """加载与本文件同目录的 filename；传 1 个属性名返回单值，多个返回元组。"""
    path = Path(__file__).resolve().parent / filename
    if not path.is_file():
        raise ImportError(f"找不到 {path}")

    spec = importlib.util.spec_from_file_location(f"_local_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 {path}")

    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))  # 让被加载文件里的平级 import（from linked_list import ...）成立
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(str(path.parent))

    if not attrs:
        return mod
    values = tuple(getattr(mod, attr) for attr in attrs)
    return values[0] if len(values) == 1 else values

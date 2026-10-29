#!/usr/bin/env python
"""最小堆可视化：把 MinHeap 的 push / pop 拆成「一行代码一步」。

和前两个可视化器同一套交互（← → 单步、空格 自动播放），但这次画的是
**完全二叉树 + 它底层的那个数组**，重点看四件事：
    1. 堆就是数组：下标 i 的父 = (i-1)//2，左孩子 = 2i+1，右孩子 = 2i+2
    2. push：先把新元素放到数组末尾（树的最后一个叶子），然后一路上浮
    3. pop：堆顶弹出，把末尾元素搬到堆顶，然后一路下沉
    4. 上浮 / 下沉都是「和父节点（或最小孩子）比大小，不满足就交换」

用法（在仓库根目录下）：
    python -m src.utils.data_structure.heap_visualizer           # 打开窗口
    python -m src.utils.data_structure.heap_visualizer --check   # 逻辑自检
    python -m src.utils.data_structure.heap_visualizer --smoke   # 渲染一次就退出
"""
from __future__ import annotations

import argparse
import random
import sys
import tkinter as tk
from pathlib import Path
from tkinter import font as tkfont

try:  # 作为包的一部分被 import
    from ._loader import load_local
except ImportError:  # 直接运行本文件
    from _loader import load_local

MinHeapImpl = None
_IMPORT_ERROR: str | None = None
try:
    MinHeapImpl = load_local("heap.py", "MinHeap")
except Exception as exc:  # noqa: BLE001
    _IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

# --------------------------------------------------------------------------- #
# 配色
# --------------------------------------------------------------------------- #
BG = "#f6f7f9"
CARD = "#ffffff"
INK = "#1f2328"
MUTED = "#6e7781"
EDGE = "#c9d1d9"
HOT = "#e5534b"
ROOT_C = "#1f6feb"
LEAF_FILL = "#f7fbff"


# --------------------------------------------------------------------------- #
# 一、状态模型：就是一个 list（等价于 MinHeap.heap）
# --------------------------------------------------------------------------- #
def trace_push(state: list, value: int) -> list[dict]:
    a = list(state)
    steps: list[dict] = []

    def rec(desc: str, code: str, hl: tuple = (), swap: tuple | None = None) -> None:
        steps.append({"desc": desc, "code": code, "state": list(a), "hl": tuple(hl), "swap": swap})

    a.append(value)
    rec(
        f"先放到数组末尾（下标 {len(a) - 1}），也就是树的最后一个叶子",
        "self.heap.append(val)",
        hl=(len(a) - 1,),
    )

    i = len(a) - 1
    while i > 0:
        father = (i - 1) // 2
        rec(
            f"和父节点比：a[{i}] = {a[i]} 与 a[{father}] = {a[father]}",
            "if self.heap[index] < self.heap[father]:",
            hl=(i, father),
        )
        child, parent = a[i], a[father]
        if child < parent:
            a[i], a[father] = parent, child
            rec(
                f"{child} < {parent} → 交换；index 上移到父节点 {father} 继续比",
                "self.heap[index], self.heap[father] = self.heap[father], self.heap[index]",
                hl=(i, father),
                swap=(i, father),
            )
            i = father
        else:
            rec(
                f"{child} 不小于父节点 {parent} → 不用换，堆性质已满足，结束",
                "else: break",
                hl=(i, father),
            )
            break

    rec(f"push 完成：size = {len(a)}，堆顶仍是最小值 {a[0]}", "return（隐式）")
    return steps


def trace_pop(state: list) -> list[dict]:
    a = list(state)
    steps: list[dict] = []

    def rec(desc: str, code: str, hl: tuple = (), swap: tuple | None = None) -> None:
        steps.append({"desc": desc, "code": code, "state": list(a), "hl": tuple(hl), "swap": swap})

    if not a:
        rec("堆是空的 → 你的代码会 raise ValueError('堆没有值')", "if not self.heap: raise ValueError")
        return steps

    if len(a) == 1:
        only = a[0]
        rec(f"只有一个元素 {only} → 直接 pop 出去", "if self.size() == 1: return self.heap.pop()", hl=(0,))
        a.pop()
        rec("堆空了", "")
        return steps

    top = a[0]
    rec(f"堆顶 a[0] = {top} 就是整个堆的最小值，先记下来", "value = self.heap[0]", hl=(0,))

    last = a.pop()
    a[0] = last
    rec(
        f"把末尾的 {last} 摘下来填到堆顶（长度减一），它现在是「假根」，必须往下沉",
        "self.heap[0] = self.heap.pop()",
        hl=(0,),
    )

    i = 0
    while True:
        n = len(a)
        left, right = 2 * i + 1, 2 * i + 2
        hl = (i,) + tuple(x for x in (left, right) if x < n)

        rec(
            f"看 a[{i}] = {a[i]} 的最小孩子",
            "smallest = index; left_child = 2*index+1; right_child = 2*index+2",
            hl=hl,
        )
        smallest = i
        if left < n and a[left] < a[smallest]:
            smallest = left
        if right < n and a[right] < a[smallest]:
            smallest = right

        if smallest == i:
            rec(
                f"孩子都不比 {a[i]} 小（或者没有孩子）→ 位置对了，停",
                "if smallest == index: break",
                hl=hl,
            )
            break

        child = a[smallest]
        rec(
            f"孩子 a[{smallest}] = {child} 更小 → 交换，index 下沉到 {smallest}",
            "self.heap[index], self.heap[smallest] = self.heap[smallest], self.heap[index]",
            hl=(i, smallest),
            swap=(i, smallest),
        )
        a[i], a[smallest] = child, a[i]
        i = smallest

    rec(f"pop 完成：这次弹出 {top}，size = {len(a)}", "return value")
    return steps


# --------------------------------------------------------------------------- #
# 二、界面
# --------------------------------------------------------------------------- #
class HeapApp:
    R = 22          # 节点半径
    ARRAY_H = 46    # 数组格高度

    def __init__(self, root: tk.Tk):
        self.root = root
        self.state: list[int] = []
        self.base_state: list[int] = []
        self.steps: list[dict] = []
        self.idx = -1
        self.playing = False
        self._after_id = None

        self.f_body = tkfont.Font(family="Microsoft YaHei UI", size=10)
        self.f_bold = tkfont.Font(family="Microsoft YaHei UI", size=11, weight="bold")
        self.f_small = tkfont.Font(family="Microsoft YaHei UI", size=9)
        self.f_tiny = tkfont.Font(family="Consolas", size=8)
        self.f_node = tkfont.Font(family="Consolas", size=14, weight="bold")
        self.f_code = tkfont.Font(family="Consolas", size=10)

        self._build_ui()
        self._bind_keys()
        self.render()

    # ------------------------------------------------------------------ 布局
    def _build_ui(self) -> None:
        self.root.title("最小堆可视化 · MinHeap")
        self.root.geometry("1220x620")
        self.root.configure(bg=BG)

        bar = tk.Frame(self.root, bg=CARD, padx=14, pady=10)
        bar.pack(fill="x")

        tk.Label(bar, text="插入值", bg=CARD, fg=MUTED, font=self.f_body).pack(side="left")
        self.var_value = tk.StringVar(value="7")
        tk.Entry(
            bar, textvariable=self.var_value, width=5, font=self.f_code,
            relief="solid", bd=1, justify="center",
        ).pack(side="left", padx=(6, 12))

        def btn(text, cmd, color=CARD, fg=INK):
            b = tk.Button(
                bar, text=text, command=cmd, font=self.f_body, bg=color, fg=fg,
                activebackground=color, relief="solid", bd=1, padx=10, pady=3, cursor="hand2",
            )
            b.pack(side="left", padx=4)
            return b

        btn("插入 ↓", lambda: self.do_push(1), "#e9f7ee", "#2ea043")
        btn("插入 5 个随机", lambda: self.do_push(5), "#e9f7ee", "#2ea043")
        btn("弹出堆顶 ↑", lambda: self.do_pop(), "#eaf2ff", ROOT_C)
        tk.Frame(bar, bg="#e4e7eb", width=1, height=22).pack(side="left", padx=10, fill="y")
        btn("演示一遍", self.demo)
        btn("清空", self.reset)

        self.canvas = tk.Canvas(self.root, bg=BG, highlightthickness=0)
        self.canvas.bind("<Configure>", lambda _e: self.render())

        foot = tk.Frame(self.root, bg=CARD, padx=14, pady=10)

        self.lbl_desc = tk.Label(
            foot, text="", bg=CARD, fg=INK, font=self.f_bold, anchor="w",
            justify="left", wraplength=1120,
        )
        self.lbl_desc.pack(fill="x")
        self.lbl_code = tk.Label(foot, text="", bg=CARD, fg="#57606a", font=self.f_code, anchor="w")
        self.lbl_code.pack(fill="x", pady=(4, 2))

        row = tk.Frame(foot, bg=CARD)
        row.pack(fill="x", pady=(6, 0))
        self.lbl_stat = tk.Label(row, text="", bg=CARD, fg=MUTED, font=self.f_body, anchor="w")
        self.btn_play = tk.Button(
            row, text="▶ 自动播放", command=self.toggle_play, font=self.f_body,
            relief="solid", bd=1, padx=10, pady=2, cursor="hand2",
        )
        self.btn_next = tk.Button(
            row, text="下一步 →", command=self.next_step, font=self.f_body,
            relief="solid", bd=1, padx=10, pady=2, cursor="hand2",
        )
        self.btn_prev = tk.Button(
            row, text="← 上一步", command=self.prev_step, font=self.f_body,
            relief="solid", bd=1, padx=10, pady=2, cursor="hand2",
        )

        self.btn_play.pack(side="right", padx=(4, 0))
        self.btn_next.pack(side="right", padx=4)
        self.btn_prev.pack(side="right", padx=4)
        self.lbl_stat.pack(side="left", fill="x", expand=True)

        for w in (bar, self.canvas, foot):
            w.pack_forget()
        bar.pack(side="top", fill="x")
        foot.pack(side="bottom", fill="x")
        self.canvas.pack(side="top", fill="both", expand=True)

    def _bind_keys(self) -> None:
        self.root.bind("<Right>", lambda _e: self.next_step())
        self.root.bind("<Left>", lambda _e: self.prev_step())
        self.root.bind("<space>", lambda _e: self.toggle_play())
        self.root.bind("<Escape>", lambda _e: self.root.destroy())

    # ------------------------------------------------------------------ 交互
    def _value(self) -> int:
        raw = self.var_value.get().strip()
        try:
            return int(raw)
        except ValueError:
            return random.randint(1, 99)

    def _commit(self) -> None:
        if self.steps:
            self.state = list(self.steps[-1]["state"])

    def _play(self, ops: list[tuple]) -> None:
        st = list(self.state)
        base = list(st)
        all_steps: list[dict] = []
        for op, value in ops:
            s = trace_push(st, value) if op == "push" else trace_pop(st)
            if s:
                st = list(s[-1]["state"])
            all_steps.extend(s)
        self.base_state = base
        self.steps = all_steps
        self.idx = 0
        self.render()

    def do_push(self, times: int = 1) -> None:
        self.stop_play()
        self._commit()
        if times == 1:
            ops = [("push", self._value())]
        else:
            ops = [("push", random.randint(1, 99)) for _ in range(times)]
        self._play(ops)
        self.var_value.set(str(random.randint(1, 99)))

    def do_pop(self) -> None:
        self.stop_play()
        self._commit()
        self._play([("pop", None)])

    def demo(self) -> None:
        """固定剧本：插入 [5, 2, 8, 1, 9, 3]，然后弹三次堆顶。"""
        self.reset()
        ops = [("push", v) for v in (5, 2, 8, 1, 9, 3)] + [("pop", None)] * 3
        self._play(ops)
        self.toggle_play()

    def reset(self) -> None:
        self.stop_play()
        self.state = []
        self.base_state = []
        self.steps = []
        self.idx = -1
        self.render()

    def next_step(self) -> None:
        if self.steps and self.idx < len(self.steps) - 1:
            self.idx += 1
            self.render()

    def prev_step(self) -> None:
        if self.steps and self.idx > 0:
            self.idx -= 1
            self.render()

    def toggle_play(self) -> None:
        if self.playing:
            self.stop_play()
            return
        if not self.steps:
            return
        self.playing = True
        self.btn_play.config(text="⏸ 暂停")
        self._after_id = self.root.after(900, self._tick)

    def stop_play(self) -> None:
        self.playing = False
        self.btn_play.config(text="▶ 自动播放")
        if self._after_id is not None:
            try:
                self.root.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None

    def _tick(self) -> None:
        if not self.playing:
            return
        if self.idx < len(self.steps) - 1:
            self.idx += 1
            self.render()
            self._after_id = self.root.after(900, self._tick)
        else:
            self.stop_play()

    # ------------------------------------------------------------------ 绘制
    def _snap(self) -> dict:
        if self.steps:
            return self.steps[self.idx]
        return {
            "desc": "点「插入」或「演示一遍」：每次 push/pop 都会被拆成若干步，用「下一步」看它怎么上浮/下沉",
            "code": "",
            "state": self.state,
            "hl": (),
            "swap": None,
        }

    def render(self) -> None:
        c = self.canvas
        c.delete("all")

        snap = self._snap()
        a: list[int] = snap["state"]
        n = len(a)
        hot = set(snap.get("hl", ()))
        swap = snap.get("swap")

        w = max(c.winfo_width(), 900)
        h = c.winfo_height() if c.winfo_height() > 200 else 420

        self._draw_header(w, n, a)

        if n == 0:
            c.create_text(w / 2, h / 2 - 20, text="（空堆）", font=self.f_bold, fill="#a0a8b0")
            c.create_text(w / 2, h / 2 + 10, text="点「插入 ↓」放第一个元素",
                          font=self.f_small, fill="#a0a8b0")
            self._update_footer(snap, a)
            return

        # ---- 树 + 数组条的几何 ----
        # 最后一个节点的层 = n.bit_length() - 1，所以整棵树的层数 = n.bit_length()
        levels = n.bit_length()

        array_top = h - self.ARRAY_H - 34
        tree_top = 96.0
        tree_bottom = array_top - (64 if levels > 1 else 48)
        gap = (tree_bottom - tree_top) / (levels - 1) if levels > 1 else 0.0
        gap = min(gap, 86.0)
        used = gap * (levels - 1)
        top_offset = tree_top + max(0.0, (tree_bottom - tree_top - used) / 2)

        x_left, x_right = 80.0, w - 80.0
        tree_w = x_right - x_left
        pos: dict[int, tuple[float, float]] = {}
        for i in range(n):
            level = (i + 1).bit_length() - 1
            first = (1 << level) - 1
            idx_in_level = i - first
            slots = 1 << level
            x = x_left + (idx_in_level + 0.5) * tree_w / slots
            y = top_offset + level * gap
            pos[i] = (x, y)

        # 边先画，节点后画（这样线被节点盖住，看起来是连着的）
        for i in range(1, n):
            parent = (i - 1) // 2
            c.create_line(*pos[parent], *pos[i], fill=EDGE, width=2)

        # 节点
        for i in range(n):
            x, y = pos[i]
            is_hot = i in hot
            outline = HOT if is_hot else (ROOT_C if i == 0 else "#8b949e")
            fill = "#fff1f0" if is_hot else (LEAF_FILL if i == 0 else CARD)
            c.create_oval(
                x - self.R, y - self.R, x + self.R, y + self.R,
                fill=fill, outline=outline, width=3 if is_hot else 2,
            )
            # 下标写在圆外右上角，别挤在圆里
            c.create_text(x + self.R - 2, y - self.R - 2, anchor="w", text=str(i),
                          font=self.f_tiny, fill=MUTED)
            c.create_text(x, y + 2, text=str(a[i]), font=self.f_node, fill=INK)

        c.create_text(pos[0][0], pos[0][1] - self.R - 16, text="堆顶 = 全局最小值",
                      font=self.f_small, fill=ROOT_C)

        # 交换提示（两个节点都被标红了，这里再画一个双向箭头文字，避免遮挡）
        if swap:
            (i, j) = swap
            mx = (pos[i][0] + pos[j][0]) / 2
            my = (pos[i][1] + pos[j][1]) / 2
            c.create_text(mx, my - 12, text="⇅ 交换", font=self.f_small, fill=HOT)

        self._draw_array(w, a, hot, array_top)
        self._update_footer(snap, a)

    def _draw_header(self, w: float, n: int, a: list[int]) -> None:
        c = self.canvas
        c.create_text(
            34, 22, anchor="w", font=self.f_small, fill=MUTED,
            text="堆 = 完全二叉树，底层就是一个数组：下标 i 的父 = (i-1)//2，左孩子 = 2i+1，右孩子 = 2i+2",
        )
        c.create_text(w - 34, 22, anchor="e", font=self.f_small, fill=MUTED,
                      text=f"size = {n}　堆顶 = {a[0] if n else '—'}")

    def _draw_array(self, w: float, a: list[int], hot: set[int], top: float) -> None:
        c = self.canvas
        n = len(a)
        if n == 0:
            return
        cell = min(58.0, max(38.0, (w - 160) / n - 6))
        gap = 6.0
        total = n * cell + (n - 1) * gap
        x0 = (w - total) / 2

        c.create_text(x0 - 12, top + self.ARRAY_H / 2, anchor="e", font=self.f_small,
                      fill=MUTED, text="底层数组")

        for i in range(n):
            x = x0 + i * (cell + gap)
            is_hot = i in hot
            c.create_rectangle(
                x, top, x + cell, top + self.ARRAY_H,
                fill="#fff1f0" if is_hot else CARD,
                outline=HOT if is_hot else EDGE,
                width=2,
            )
            c.create_text(x + 6, top + 8, anchor="w", text=str(i), font=self.f_tiny, fill=MUTED)
            c.create_text(x + cell / 2, top + self.ARRAY_H / 2 + 5, text=str(a[i]),
                          font=self.f_node, fill=INK)

    def _update_footer(self, snap: dict, a: list[int]) -> None:
        total = len(self.steps)
        head = f"第 {self.idx + 1} / {total} 步 　" if total else ""
        self.lbl_desc.config(text=head + snap["desc"])
        self.lbl_code.config(text=("　　" + snap["code"]) if snap["code"] else "")

        # 顺手自检：每个节点都必须 <= 它的孩子，否则堆性质被破坏
        bad = [str(i) for i in range(len(a))
               if (2 * i + 1 < len(a) and a[2 * i + 1] < a[i])
               or (2 * i + 2 < len(a) and a[2 * i + 2] < a[i])]
        health = "堆性质 ✓" if not bad else f"⚠ 暂未满足（正在调整节点 {','.join(bad)}）"
        self.lbl_stat.config(
            text=f"size = {len(a)}　数组 = {a}　{health}"
        )
        self.btn_prev.config(state="normal" if self.idx > 0 else "disabled")
        self.btn_next.config(state="normal" if total and self.idx < total - 1 else "disabled")


# --------------------------------------------------------------------------- #
# 三、自检 / 截图
# --------------------------------------------------------------------------- #
def run_self_check() -> int:
    if MinHeapImpl is None:
        print("⚠ 跳过自检：导入 heap.py 里的 MinHeap 失败")
        print(f"  {_IMPORT_ERROR}")
        return 2

    seed_values = [5, 2, 8, 1, 9, 3, 7, 4, 12, 6]
    real = MinHeapImpl()
    model: list[int] = []
    ok = True

    for value in seed_values:
        real.push(value)
        model = trace_push(model, value)[-1]["state"]
        if model != real.heap:
            ok = False
            print(f"✗ push({value})：模型 {model} vs 真实 {real.heap}")

    for _ in range(len(seed_values)):
        expect = real.pop()
        got = model[0]
        steps = trace_pop(model)
        model = steps[-1]["state"]
        if got != expect:
            ok = False
            print(f"✗ pop：模型弹出 {got}，真实弹出 {expect}")
        if model != real.heap:
            ok = False
            print(f"✗ pop 后数组不一致：模型 {model} vs 真实 {real.heap}")

    print()
    print(
        f"PASS —— 可视化模型与 MinHeap 完全一致（{len(seed_values)} 次 push + {len(seed_values)} 次 pop，数组逐步对齐）"
        if ok
        else "FAIL —— 见上面的差异"
    )
    return 0 if ok else 1


def _dpi_aware() -> None:
    try:
        from ctypes import windll

        windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        pass


def run_shot(path: str, step: int = 3) -> int:
    import time

    _dpi_aware()
    root = tk.Tk()
    root.overrideredirect(True)
    root.geometry("1220x620+60+60")
    app = HeapApp(root)
    root.update_idletasks()
    root.update()

    app.demo()
    app.stop_play()
    for _ in range(step):
        app.next_step()

    root.attributes("-topmost", True)
    root.update_idletasks()
    root.update()
    time.sleep(0.4)

    try:
        from PIL import ImageGrab
    except ImportError:
        root.destroy()
        print("需要 Pillow：python -m pip install pillow")
        return 1

    x, y = root.winfo_rootx(), root.winfo_rooty()
    ImageGrab.grab(bbox=(x, y, x + root.winfo_width(), y + root.winfo_height())).save(path)
    root.destroy()
    print(f"已保存截图：{path}")
    return 0


def run_smoke() -> int:
    root = tk.Tk()
    app = HeapApp(root)
    root.update_idletasks()
    app.demo()
    for _ in range(12):
        app.next_step()
        root.update()
    app.stop_play()
    root.update()
    root.destroy()
    print("smoke OK —— 窗口能创建、能绘制、能单步")
    return 0


def main() -> int:
    _dpi_aware()
    ap = argparse.ArgumentParser(description="最小堆可视化")
    ap.add_argument("--check", action="store_true", help="只做逻辑自检，不开窗口")
    ap.add_argument("--smoke", action="store_true", help="渲染一次就退出")
    ap.add_argument("--shot", metavar="PNG", help="渲染并截图保存")
    args = ap.parse_args()

    if args.check:
        return run_self_check()
    if args.smoke:
        return run_smoke()
    if args.shot:
        return run_shot(args.shot)

    root = tk.Tk()
    HeapApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

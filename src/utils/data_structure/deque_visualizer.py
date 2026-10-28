#!/usr/bin/env python
"""双向队列可视化：把 LinkedListDeque 的入队 / 出队拆成「一行代码一步」，逐步看指针怎么变。

用法（在仓库根目录下）：
    python -m src.utils.data_structure.deque_visualizer           # 打开窗口
    python -m src.utils.data_structure.deque_visualizer --check   # 只做逻辑自检，不开窗口
    python -m src.utils.data_structure.deque_visualizer --smoke   # 渲染一次就退出（排错用）

    也可以直接运行：python src\\utils\\data_structure\\deque_visualizer.py

窗口里：
    上方按钮 = 一次操作；操作会被拆成若干「步骤」
    下一步 / 上一步  单步前进后退（键盘 → ←）
    自动播放         每 0.9 秒走一步（键盘空格）
    红色实线 = 本步新增的指针   红色虚线 = 本步被剪断的指针
    紫色虚线框 = 刚 new 出来、还没接进链表的节点
"""
from __future__ import annotations

import argparse
import copy
import random
import sys
import time
import tkinter as tk
from pathlib import Path
from tkinter import font as tkfont

# --------------------------------------------------------------------------- #
# 导入你自己写的 LinkedListDeque（在 data_structure/queue.py 里）
#
# 这里不能写 `from queue import LinkedListDeque` —— 那会把标准库的 queue 模块遮蔽掉，
# tkinter 之类会莫名其妙炸掉。所以统一按「文件路径」加载，顺便兼容
# 「直接跑脚本」和「python -m 包名」两种方式。
# --------------------------------------------------------------------------- #
import importlib.util

_here = Path(__file__).resolve().parent
LinkedListDeque = None
_IMPORT_ERROR: str | None = None

try:
    _spec = importlib.util.spec_from_file_location("deque_impl", _here / "queue.py")
    _mod = importlib.util.module_from_spec(_spec)
    sys.path.insert(0, str(_here))      # queue.py 里是 `from linked_list import ...`
    try:
        _spec.loader.exec_module(_mod)
    finally:
        sys.path.remove(str(_here))
    LinkedListDeque = _mod.LinkedListDeque
except Exception as exc:  # noqa: BLE001
    # 窗口本身不依赖它；只有 --check 需要。queue.py 有语法错误时也能正常看动画。
    _IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

# --------------------------------------------------------------------------- #
# 配色
# --------------------------------------------------------------------------- #
BG = "#f6f7f9"
CARD = "#ffffff"
INK = "#1f2328"
MUTED = "#6e7781"
LINE = "#8b949e"
HEAD_C = "#1f6feb"
TAIL_C = "#2ea043"
HOT = "#e5534b"
NEW_C = "#8250df"
HOT_FILL = "#fff1f0"


# --------------------------------------------------------------------------- #
# 一、状态模型：只用字典表示，便于整份快照 + 前进后退
# --------------------------------------------------------------------------- #
def empty_state() -> dict:
    return {"nodes": [], "head": None, "tail": None, "size": 0, "next_id": 1}


def node_of(st: dict, nid: int) -> dict:
    for n in st["nodes"]:
        if n["id"] == nid:
            return n
    raise KeyError(f"节点 {nid} 不存在")


def index_of(st: dict, nid: int) -> int:
    for i, n in enumerate(st["nodes"]):
        if n["id"] == nid:
            return i
    raise KeyError(f"节点 {nid} 不存在")


def drop(st: dict, nid: int) -> None:
    st["nodes"] = [n for n in st["nodes"] if n["id"] != nid]


def values(st: dict) -> list[int]:
    return [n["value"] for n in st["nodes"]]


def trace(state: dict, op: str, value: int | None = None) -> list[dict]:
    """把一次操作拆成若干步；每步都带一份完整快照，所以可以随便前进后退。"""
    st = copy.deepcopy(state)
    steps: list[dict] = []

    def rec(desc: str, code: str, floating: dict | None = None, hl: tuple = ()) -> None:
        steps.append(
            {
                "desc": desc,
                "code": code,
                "state": copy.deepcopy(st),
                "floating": floating,
                "hl": list(hl),
            }
        )

    # ---------------------------------------------------------------- 队头入队
    if op == "push_first":
        nid = st["next_id"]
        st["next_id"] += 1
        node = {"id": nid, "value": value, "prev": None, "next": None}
        rec(
            f"new 出一个节点 {value}，此刻它的 prev / next 都还是 NULL",
            "node = DoublyListNode(num)",
            floating={"id": nid, "value": value},
        )

        if not st["nodes"]:
            st["nodes"].insert(0, node)
            st["head"] = st["tail"] = nid
            rec(
                "队列本来是空的：head 和 tail 同时指向它，一步就完成",
                "self._head = self._tail = node",
                hl=(nid,),
            )
        else:
            old = st["head"]
            node_of(st, old)["prev"] = nid
            st["nodes"].insert(0, node)
            rec(
                f"第 1 根指针：让旧 head（{node_of(st, old)['value']}）的 prev 指回新节点 —— 注意，"
                "此时新节点的 next 还是空的，链表只是「半通」",
                "self._head.prev = node",
                hl=(old, nid),
            )
            node["next"] = old
            rec(
                "第 2 根指针：新节点的 next 指向旧 head —— 到这里双向才真正接通",
                "node.next = self._head",
                hl=(old, nid),
            )
            st["head"] = nid
            rec(
                "最后只是把 head 这个「标记」挪过去，链表结构本身没有再动",
                "self._head = node",
                hl=(nid,),
            )

        st["size"] += 1
        rec(f"长度加一：size = {st['size']}", "self._size += 1")
        return steps

    # ---------------------------------------------------------------- 队尾入队
    if op == "push_last":
        nid = st["next_id"]
        st["next_id"] += 1
        node = {"id": nid, "value": value, "prev": None, "next": None}
        rec(
            f"new 出一个节点 {value}，此刻它的 prev / next 都还是 NULL",
            "node = DoublyListNode(num)",
            floating={"id": nid, "value": value},
        )

        if not st["nodes"]:
            st["nodes"].append(node)
            st["head"] = st["tail"] = nid
            rec(
                "队列本来是空的：head 和 tail 同时指向它",
                "self._head = self._tail = node",
                hl=(nid,),
            )
        else:
            old = st["tail"]
            node_of(st, old)["next"] = nid
            st["nodes"].append(node)
            rec(
                f"第 1 根指针：让旧 tail（{node_of(st, old)['value']}）的 next 指向新节点",
                "self._tail.next = node",
                hl=(old, nid),
            )
            node["prev"] = old
            rec(
                "第 2 根指针：新节点的 prev 指向旧 tail —— 双向接通",
                "node.prev = self._tail",
                hl=(old, nid),
            )
            st["tail"] = nid
            rec(
                "最后把 tail 标记挪到新节点上",
                "self._tail = node",
                hl=(nid,),
            )

        st["size"] += 1
        rec(f"长度加一：size = {st['size']}", "self._size += 1")
        return steps

    # ---------------------------------------------------------------- 队头出队
    if op == "pop_first":
        if not st["nodes"]:
            raise IndexError("双向队列为空")
        nid = st["head"]
        n = node_of(st, nid)
        rec(
            f"先把队头的值读出来：num = {n['value']}（这一步只取值，不动任何指针）",
            "num = self._head.value",
            hl=(nid,),
        )

        if st["size"] == 1:
            st["head"] = st["tail"] = None
            rec(
                "队列里只剩这一个节点：head / tail 一起置空 —— 瞧，已经没有任何人指向它了",
                "self._head = self._tail = None",
                hl=(nid,),
            )
            drop(st, nid)
            rec("节点被回收（Python 里没人引用后自动垃圾回收）", "return num")
        else:
            nxt = n["next"]
            rec(
                f"记住队头的下一个节点（{node_of(st, nxt)['value']}），它下一步要当新 head",
                "fnext = self._head.next",
                hl=(nxt,),
            )
            node_of(st, nxt)["prev"] = None
            rec(
                "剪第 1 根线：下一个节点的 prev 置空（它以后是头了，前面不该有人）",
                "fnext.prev = None",
                hl=(nxt,),
            )
            n["next"] = None
            rec(
                "剪第 2 根线：旧 head 的 next 置空 —— 两条线都断了，它才真正脱离链表",
                "self._head.next = None",
                hl=(nid,),
            )
            drop(st, nid)
            st["head"] = nxt
            rec(f"head 标记挪到 {node_of(st, nxt)['value']}", "self._head = fnext", hl=(nxt,))

        st["size"] -= 1
        rec(f"长度减一：size = {st['size']}", "self._size -= 1")
        return steps

    # ---------------------------------------------------------------- 队尾出队
    if op == "pop_last":
        if not st["nodes"]:
            raise IndexError("双向队列为空")
        nid = st["tail"]
        n = node_of(st, nid)
        rec(
            f"先把队尾的值读出来：num = {n['value']}（只取值，不动指针）",
            "num = self._tail.value",
            hl=(nid,),
        )

        if st["size"] == 1:
            st["head"] = st["tail"] = None
            rec(
                "队列里只剩这一个节点：head / tail 一起置空",
                "self._head = self._tail = None",
                hl=(nid,),
            )
            drop(st, nid)
            rec("节点被回收", "return num")
        else:
            prv = n["prev"]
            rec(
                f"记住队尾的前一个节点（{node_of(st, prv)['value']}），它下一步要当新 tail",
                "tprev = self._tail.prev",
                hl=(prv,),
            )
            node_of(st, prv)["next"] = None
            rec(
                "剪第 1 根线：前一个节点的 next 置空",
                "tprev.next = None",
                hl=(prv,),
            )
            n["prev"] = None
            rec(
                "剪第 2 根线：旧 tail 的 prev 置空 —— 它彻底脱链",
                "self._tail.prev = None",
                hl=(nid,),
            )
            drop(st, nid)
            st["tail"] = prv
            rec(f"tail 标记挪到 {node_of(st, prv)['value']}", "self._tail = tprev", hl=(prv,))

        st["size"] -= 1
        rec(f"长度减一：size = {st['size']}", "self._size -= 1")
        return steps

    raise ValueError(f"未知操作: {op}")


def pointer_map(st: dict) -> set[tuple[str, int, int | None]]:
    """把当前所有指针拆成集合：("next", 从, 到) / ("prev", 从, 到)。NULL 用 None。"""
    out: set[tuple[str, int, int | None]] = set()
    for n in st["nodes"]:
        out.add(("next", n["id"], n["next"]))
        out.add(("prev", n["id"], n["prev"]))
    return out


# --------------------------------------------------------------------------- #
# 二、界面
# --------------------------------------------------------------------------- #
class DequeApp:
    NODE_W = 94
    NODE_H = 62
    NULL_W = 64
    MARGIN = 34
    CY = 156

    def __init__(self, root: tk.Tk):
        self.root = root
        self.state = empty_state()
        self.base_state = copy.deepcopy(self.state)
        self.steps: list[dict] = []
        self.idx = -1
        self.playing = False
        self._after_id = None

        self.f_body = tkfont.Font(family="Microsoft YaHei UI", size=10)
        self.f_bold = tkfont.Font(family="Microsoft YaHei UI", size=11, weight="bold")
        self.f_small = tkfont.Font(family="Microsoft YaHei UI", size=9)
        self.f_node = tkfont.Font(family="Consolas", size=15, weight="bold")
        self.f_code = tkfont.Font(family="Consolas", size=10)

        self._build_ui()
        self._bind_keys()
        self.render()

    # ------------------------------------------------------------------ 布局
    def _build_ui(self) -> None:
        self.root.title("双向队列可视化 · LinkedListDeque")
        self.root.geometry("1220x560")
        self.root.configure(bg=BG)

        # ---- 顶部工具条 ----
        bar = tk.Frame(self.root, bg=CARD, padx=14, pady=10)
        bar.pack(fill="x")

        tk.Label(bar, text="数值", bg=CARD, fg=MUTED, font=self.f_body).pack(side="left")
        self.var_value = tk.StringVar(value=str(random.randint(10, 99)))
        tk.Entry(
            bar, textvariable=self.var_value, width=6, font=self.f_code,
            relief="solid", bd=1, justify="center",
        ).pack(side="left", padx=(6, 14))

        def btn(text, cmd, color="#ffffff", fg=INK):
            b = tk.Button(
                bar, text=text, command=cmd, font=self.f_body, bg=color, fg=fg,
                activebackground=color, relief="solid", bd=1, padx=10, pady=3, cursor="hand2",
            )
            b.pack(side="left", padx=4)
            return b

        btn("队头入队 ↑", lambda: self.do("push_first"), "#eaf2ff", HEAD_C)
        btn("队尾入队 ↓", lambda: self.do("push_last"), "#e9f7ee", TAIL_C)
        btn("队头出队 ×", lambda: self.do("pop_first"), "#fff1f0", HOT)
        btn("队尾出队 ×", lambda: self.do("pop_last"), "#fff1f0", HOT)

        tk.Frame(bar, bg="#e4e7eb", width=1, height=22).pack(side="left", padx=10, fill="y")
        btn("演示一遍", self.demo)
        btn("清空", self.reset)

        # ---- 画布 ----
        self.canvas = tk.Canvas(self.root, bg=BG, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda _e: self.render())

        # ---- 底部信息 ----
        foot = tk.Frame(self.root, bg=CARD, padx=14, pady=10)
        foot.pack(fill="x")

        self.lbl_desc = tk.Label(
            foot, text="", bg=CARD, fg=INK, font=self.f_bold, anchor="w",
            justify="left", wraplength=1120,
        )
        self.lbl_desc.pack(fill="x")

        self.lbl_code = tk.Label(
            foot, text="", bg=CARD, fg="#57606a", font=self.f_code, anchor="w",
        )
        self.lbl_code.pack(fill="x", pady=(4, 2))

        row = tk.Frame(foot, bg=CARD)
        row.pack(fill="x", pady=(6, 0))

        self.lbl_stat = tk.Label(row, text="", bg=CARD, fg=MUTED, font=self.f_body, anchor="w")
        self.lbl_stat.pack(side="left")

        self.btn_play = tk.Button(
            row, text="▶ 自动播放", command=self.toggle_play, font=self.f_body,
            relief="solid", bd=1, padx=10, pady=2, cursor="hand2",
        )
        self.btn_play.pack(side="right", padx=4)
        self.btn_next = tk.Button(
            row, text="下一步 →", command=self.next_step, font=self.f_body,
            relief="solid", bd=1, padx=10, pady=2, cursor="hand2",
        )
        self.btn_next.pack(side="right", padx=4)
        self.btn_prev = tk.Button(
            row, text="← 上一步", command=self.prev_step, font=self.f_body,
            relief="solid", bd=1, padx=10, pady=2, cursor="hand2",
        )
        self.btn_prev.pack(side="right", padx=4)

        # 按钮先占位（从右往左），状态文字再吃掉剩下的宽度，
        # 否则长文字会把按钮挤成一条缝。
        for w in (self.lbl_stat, self.btn_prev, self.btn_next, self.btn_play):
            w.pack_forget()
        self.btn_play.pack(side="right", padx=(4, 0))
        self.btn_next.pack(side="right", padx=4)
        self.btn_prev.pack(side="right", padx=4)
        self.lbl_stat.pack(side="left", fill="x", expand=True)

        # pack 的顺序决定谁先拿到空间：底栏必须先占位，
        # 否则 canvas 的 expand=True 会把底部区域吃光，size 行和按钮就被挤出窗口。
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
    def _int_value(self) -> int:
        raw = self.var_value.get().strip()
        try:
            return int(raw)
        except ValueError:
            v = random.randint(10, 99)
            self.var_value.set(str(v))
            return v

    def do(self, op: str) -> None:
        self.stop_play()
        if self.steps:  # 先把上一组步骤落到最终状态，再接着做
            self.state = copy.deepcopy(self.steps[-1]["state"])
        self.base_state = copy.deepcopy(self.state)
        try:
            steps = trace(self.state, op, self._int_value())
        except IndexError as exc:
            self.lbl_desc.config(text=f"⚠ {exc} —— 先入队，再出队")
            return
        self.steps = steps
        self.idx = 0
        if op.startswith("push"):
            self.var_value.set(str(random.randint(10, 99)))
        self.render()

    def demo(self) -> None:
        """一遍看完整故事：入队 → 入队 → 入队 → 出队。"""
        self.stop_play()
        self.reset()
        st = self.state
        base = copy.deepcopy(st)
        script = [("push_first", 5), ("push_last", 8), ("push_first", 2), ("pop_first", None)]
        all_steps: list[dict] = []
        for op, val in script:
            s = trace(st, op, val)
            st = s[-1]["state"]
            all_steps.extend(s)
        self.steps = all_steps
        self.base_state = base
        self.idx = 0
        self.render()
        self.toggle_play()

    def next_step(self) -> None:
        if not self.steps:
            return
        if self.idx < len(self.steps) - 1:
            self.idx += 1
            self.render()

    def prev_step(self) -> None:
        if not self.steps or self.idx <= 0:
            return
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
        # 注意：这里要"排一次"，不能直接调 _tick()，
        # 否则 _tick 会立刻把 idx 推进一步，画面会跳过当前这一格。
        self._after_id = self.root.after(950, self._tick)

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
            self._after_id = self.root.after(950, self._tick)
        else:
            self.stop_play()

    def reset(self) -> None:
        self.stop_play()
        self.state = empty_state()
        self.base_state = copy.deepcopy(self.state)
        self.steps = []
        self.idx = -1
        self.render()

    # ------------------------------------------------------------------ 绘制
    def _snap(self) -> dict:
        if self.steps:
            return self.steps[self.idx]
        return {
            "desc": "点击上方按钮开始 —— 每个操作都会被拆成若干步，用「下一步」看每根指针的变化",
            "code": "",
            "state": self.state,
            "floating": None,
            "hl": [],
        }

    def render(self) -> None:
        c = self.canvas
        c.delete("all")

        snap = self._snap()
        st = snap["state"]
        prev_st = self.steps[self.idx - 1]["state"] if (self.steps and self.idx > 0) else self.base_state
        prev_ids = {nd["id"] for nd in prev_st["nodes"]}
        cur_ids = {nd["id"] for nd in st["nodes"]}
        # 只有「本来就存在的节点」上发生的指针变化才算本步的变化。
        # 否则新接入的节点会把它天然为 NULL 的 prev/next 也标成"本步新增"，画出两条假箭头。
        added = {e for e in (pointer_map(st) - pointer_map(prev_st)) if e[1] in prev_ids}
        removed = {
            e for e in (pointer_map(prev_st) - pointer_map(st))
            if e[2] is not None            # 指向 NULL 的"剪断"没有信息量，只会糊住画面
            and e[1] in cur_ids and e[2] in cur_ids
        }

        w = max(c.winfo_width(), 980)
        h = c.winfo_height() if c.winfo_height() > 120 else 390
        nodes = st["nodes"]
        n = len(nodes)
        avail = w - 2 * self.MARGIN - 2 * self.NULL_W - n * self.NODE_W
        gap = max(58.0, min(112.0, avail / (n + 1)))

        x_null_l = float(self.MARGIN)
        x_first = x_null_l + self.NULL_W + gap
        xs = [x_first + i * (self.NODE_W + gap) for i in range(n)]
        x_null_r = x_first + n * (self.NODE_W + gap)
        cy = max(146.0, min(h * 0.52, 268.0))

        self._draw_legend(w)

        # NULL 框
        for x in (x_null_l, x_null_r):
            self._round_rect(
                x, cy - self.NODE_H / 2, self.NULL_W, self.NODE_H, 8,
                fill="#eceff3", outline="#c9d1d9", width=2, dash=(4, 3),
            )
            c.create_text(x + self.NULL_W / 2, cy, text="NULL", font=self.f_small, fill=MUTED)

        # 被剪断的指针（幽灵线）：端点还在就画出来，否则跳过
        for kind, a, b in removed:
            seg = self._segment(st, xs, kind, a, b, x_null_l, x_null_r, cy)
            if seg:
                x1, y1, x2, y2, bend = seg
                self._arrow(x1, y1, x2, y2, bend, HOT, 2, dash=(5, 4))

        # 现存的指针
        for nd in nodes:
            for kind, tgt in (("next", nd["next"]), ("prev", nd["prev"])):
                # NULL 端只画在真正的队头 / 队尾上：
                # 否则"新节点刚接入、还没连好"的那一步会到处飞箭头
                if tgt is None:
                    if (kind == "prev" and nd["id"] != st["head"]) or (
                        kind == "next" and nd["id"] != st["tail"]
                    ):
                        continue
                seg = self._segment(st, xs, kind, nd["id"], tgt, x_null_l, x_null_r, cy)
                if not seg:
                    continue
                x1, y1, x2, y2, bend = seg
                hot = (kind, nd["id"], tgt) in added
                self._arrow(x1, y1, x2, y2, bend, HOT if hot else LINE, 3 if hot else 2)

        # 节点
        for i, nd in enumerate(nodes):
            self._draw_node(
                xs[i] + self.NODE_W / 2, cy, nd["value"],
                head=(nd["id"] == st["head"]),
                tail=(nd["id"] == st["tail"]),
                hot=(nd["id"] in snap.get("hl", [])),
            )

        # 刚 new 出来、还没接上的节点
        if snap.get("floating"):
            fx = w / 2 - self.NODE_W / 2
            fy = 74
            self._round_rect(
                fx, fy - self.NODE_H / 2, self.NODE_W, self.NODE_H, 10,
                fill="#faf5ff", outline=NEW_C, width=2, dash=(5, 4),
            )
            c.create_text(fx + self.NODE_W / 2, fy, text=str(snap["floating"]["value"]),
                          font=self.f_node, fill=NEW_C)
            c.create_text(fx + self.NODE_W / 2, fy - self.NODE_H / 2 - 14,
                          text="new 出来的节点（还没接进链表）", font=self.f_small, fill=NEW_C)

        self._update_footer(snap, st, len(nodes))

    def _draw_legend(self, w: float) -> None:
        c = self.canvas
        y = 20
        x = float(self.MARGIN)
        c.create_text(x, y, text="图例", font=self.f_small, fill=MUTED, anchor="w")
        x += 36

        def text_after(marker, text, color):
            nonlocal x
            marker(x, y)
            x += 19
            tid = c.create_text(x, y, text=text, font=self.f_small, fill=color, anchor="w")
            x = c.bbox(tid)[2] + 20

        def box(color, dash=None):
            def draw(mx, my):
                c.create_rectangle(mx, my - 5, mx + 11, my + 5, outline=color, width=2, dash=dash)
            return draw

        def line(color, dash=None):
            def draw(mx, my):
                c.create_line(mx, my, mx + 14, my, fill=color, width=3, dash=dash,
                              arrow="last", arrowshape=(7, 9, 3))
            return draw

        text_after(box(HEAD_C), "head", HEAD_C)
        text_after(box(TAIL_C), "tail", TAIL_C)
        text_after(line(HOT), "本步新增的指针", HOT)
        text_after(line(HOT, dash=(4, 3)), "本步剪断的指针", HOT)
        text_after(box(NEW_C, dash=(3, 2)), "刚 new、还没接上", NEW_C)

    def _segment(self, st, xs, kind, a, b, x_null_l, x_null_r, cy):
        """返回一根指针的坐标 (x1, y1, x2, y2, bend)；画不出来就返回 None。"""
        ids = [nd["id"] for nd in st["nodes"]]
        if a not in ids:
            return None
        ia = ids.index(a)

        if kind == "next":
            x1 = xs[ia] + self.NODE_W - 2
            y1 = y2 = cy - 19
            bend = -26
            if b is None:
                x2 = x_null_r + 2
            else:
                if b not in ids or ids.index(b) != ia + 1:
                    return None
                x2 = xs[ia + 1] + 2
            return x1, y1, x2, y2, bend

        x1 = xs[ia] + 2
        y1 = y2 = cy + 19
        bend = 26
        if b is None:
            x2 = x_null_l + self.NULL_W - 2
        else:
            if b not in ids or ids.index(b) != ia - 1:
                return None
            x2 = xs[ia - 1] + self.NODE_W - 2
        return x1, y1, x2, y2, bend

    def _round_rect(self, x, y, w, h, r, **kw):
        pts = [
            x + r, y, x + w - r, y, x + w, y, x + w, y + r,
            x + w, y + h - r, x + w, y + h, x + w - r, y + h,
            x + r, y + h, x, y + h, x, y + h - r, x, y + r, x, y,
        ]
        return self.canvas.create_polygon(pts, smooth=True, **kw)

    def _arrow(self, x1, y1, x2, y2, bend, color, width, dash=None):
        mx = (x1 + x2) / 2
        my = (y1 + y2) / 2 + bend
        self.canvas.create_line(
            x1, y1, mx, my, x2, y2, smooth=True, arrow="last",
            arrowshape=(11, 13, 4), fill=color, width=width, dash=dash,
        )

    def _draw_node(self, cx, cy, value, head, tail, hot):
        x, y = cx - self.NODE_W / 2, cy - self.NODE_H / 2
        outline = HOT if hot else (HEAD_C if head else TAIL_C if tail else "#c9d1d9")
        self._round_rect(
            x, y, self.NODE_W, self.NODE_H, 10, fill=CARD, outline=outline,
            width=3 if (head or tail or hot) else 2,
        )
        self.canvas.create_text(cx, cy, text=str(value), font=self.f_node, fill=INK)

        label_y = y - 13
        if head and tail:
            self.canvas.create_text(cx, label_y, text="head / tail", font=self.f_small, fill=HEAD_C)
        elif head:
            self.canvas.create_text(cx, label_y, text="head", font=self.f_small, fill=HEAD_C)
        elif tail:
            self.canvas.create_text(cx, label_y, text="tail", font=self.f_small, fill=TAIL_C)

    def _update_footer(self, snap: dict, st: dict, n: int) -> None:
        total = len(self.steps)
        if total:
            head = f"第 {self.idx + 1} / {total} 步 　"
        else:
            head = ""
        self.lbl_desc.config(text=head + snap["desc"])
        self.lbl_code.config(text=("　　" + snap["code"]) if snap["code"] else "")

        extra = ""
        if st["size"] != len(st["nodes"]):
            extra = "　（size 最后一步才 +1）"
        self.lbl_stat.config(
            text=f"size = {st['size']}　｜　节点序列 = {values(st)}{extra}"
        )
        self.btn_prev.config(state="normal" if self.idx > 0 else "disabled")
        self.btn_next.config(
            state="normal" if total and self.idx < total - 1 else "disabled"
        )


# --------------------------------------------------------------------------- #
# 三、逻辑自检：可视化模型 vs 你自己的 LinkedListDeque
# --------------------------------------------------------------------------- #
def run_self_check() -> int:
    if LinkedListDeque is None:
        print("⚠ 跳过自检：导入 data_structure/queue.py 失败")
        print(f"  {_IMPORT_ERROR}")
        print("\n  最常见的原因是 queue.py 有语法错误（比如某个 def 只写了名字没写函数体）。")
        print("  先让下面这条命令跑通，再回来做自检：")
        print("      python src\\utils\\data_structure\\queue.py")
        return 2

    script = [
        ("push_first", 1), ("push_last", 2), ("push_first", 3), ("push_last", 4),
        ("pop_first", None), ("pop_last", None), ("pop_first", None),
        ("push_last", 5), ("pop_last", None), ("push_first", 6), ("push_last", 7),
        ("pop_last", None), ("pop_first", None), ("pop_first", None), ("pop_last", None),
    ]
    real = LinkedListDeque()
    st = empty_state()
    ok = True
    done = 0

    for op, val in script:
        # 两边分别执行，空队列出队时应当「都抛同样的异常」
        real_err = model_err = None
        try:
            if op == "push_first":
                real.push_first(val)
            elif op == "push_last":
                real.push_last(val)
            elif op == "pop_first":
                real.pop_first()
            else:
                real.pop_last()
        except IndexError as exc:
            real_err = exc
        try:
            steps = trace(st, op, val)
        except IndexError as exc:
            steps = []
            model_err = exc

        if real_err is not None or model_err is not None:
            if real_err is None or model_err is None:
                ok = False
                print(
                    f"✗ {op}({val}): 真实实现{'抛错' if real_err else '正常'}，"
                    f"可视化模型{'抛错' if model_err else '正常'}"
                )
            continue

        done += 1
        st = steps[-1]["state"]
        got, want = values(st), real.to_list()
        if got != want or st["size"] != real.size():
            ok = False
            print(f"✗ {op}({val}): 模型 {got} size={st['size']} | 真实 {want} size={real.size()}")
            continue

        nodes = st["nodes"]
        if nodes:
            if nodes[0]["prev"] is not None or st["head"] != nodes[0]["id"]:
                ok = False
                print(f"✗ {op}: 头节点 prev 应为 NULL 且等于 head")
            if nodes[-1]["next"] is not None or st["tail"] != nodes[-1]["id"]:
                ok = False
                print(f"✗ {op}: 尾节点 next 应为 NULL 且等于 tail")
            for i in range(len(nodes) - 1):
                if nodes[i]["next"] != nodes[i + 1]["id"] or nodes[i + 1]["prev"] != nodes[i]["id"]:
                    ok = False
                    print(f"✗ {op}: 第 {i} / {i + 1} 个节点之间双向指针不一致")

    print()
    print(
        f"PASS —— 可视化模型与 LinkedListDeque 行为完全一致"
        f"（{done} 次有效操作 + 1 次空队列异常，指针结构全部通过）"
        if ok
        else "FAIL —— 见上面的差异"
    )
    return 0 if ok else 1


def run_smoke() -> int:
    root = tk.Tk()
    app = DequeApp(root)
    root.update_idletasks()
    app.demo()
    for _ in range(8):
        app.next_step()
        root.update()
    app.stop_play()
    root.update()
    root.destroy()
    print("smoke OK —— 窗口能创建、能绘制、能单步")
    return 0


def _dpi_aware() -> None:
    """让 Tk 和屏幕截图用同一套物理像素坐标，否则截图会偏移。"""
    try:
        from ctypes import windll

        windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        pass


def run_shot(path: str, step: int = 3) -> int:
    """把窗口渲染结果存成 PNG，用来检查排版（不用人工盯屏幕）。"""
    _dpi_aware()
    root = tk.Tk()
    root.overrideredirect(True)  # 去掉标题栏，否则 ImageGrab 会把窗口边框也框进来
    root.geometry("1220x560+60+60")
    app = DequeApp(root)
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
    w, h = root.winfo_width(), root.winfo_height()
    ImageGrab.grab(bbox=(x, y, x + w, y + h)).save(path)
    root.destroy()
    print(f"已保存截图：{path}")
    return 0


def main() -> int:
    _dpi_aware()

    ap = argparse.ArgumentParser(description="双向队列可视化")
    ap.add_argument("--check", action="store_true", help="只做逻辑自检，不开窗口")
    ap.add_argument("--smoke", action="store_true", help="渲染一次就退出")
    ap.add_argument("--shot", metavar="PNG", help="渲染并截图保存到指定 PNG")
    args = ap.parse_args()

    if args.check:
        return run_self_check()
    if args.smoke:
        return run_smoke()
    if args.shot:
        return run_shot(args.shot)

    root = tk.Tk()
    DequeApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

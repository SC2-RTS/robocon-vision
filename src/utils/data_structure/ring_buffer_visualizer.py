#!/usr/bin/env python
"""环形缓冲可视化：把 RingBuffer 的 write / read 拆成「一行代码一步」。

和 deque_visualizer 同一套交互（← → 单步、空格 自动播放），
但画的是**定长数组 + 两个指针绕圈**，重点看三件事：
    1. write 时 tail 前移，read 时 head 前移，到末尾就取模回到 0
    2. size == capacity 时再写：head 先前移 → 最旧的数据被覆盖
    3. head == tail 既可能是「空」也可能是「满」，只能靠 size 区分

用法（在仓库根目录下）：
    python -m src.utils.data_structure.ring_buffer_visualizer           # 打开窗口
    python -m src.utils.data_structure.ring_buffer_visualizer --check   # 逻辑自检
    python -m src.utils.data_structure.ring_buffer_visualizer --smoke   # 渲染一次就退出
"""
from __future__ import annotations

import argparse
import copy
import sys
import tkinter as tk
from pathlib import Path
from tkinter import font as tkfont

try:  # 作为包的一部分被 import
    from ._loader import load_local
except ImportError:  # 直接运行本文件
    from _loader import load_local

RingBufferImpl = None
_IMPORT_ERROR: str | None = None
try:
    RingBufferImpl = load_local("queue.py", "RingBuffer")
except Exception as exc:  # noqa: BLE001
    _IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

# --------------------------------------------------------------------------- #
# 配色
# --------------------------------------------------------------------------- #
BG = "#f6f7f9"
CARD = "#ffffff"
INK = "#1f2328"
MUTED = "#6e7781"
LINE = "#8b949e"
HEAD_C = "#1f6feb"   # 读指针
TAIL_C = "#2ea043"   # 写指针
HOT = "#e5534b"      # 本步改动
EMPTY_C = "#c9d1d9"
WARN_C = "#bf6a02"


# --------------------------------------------------------------------------- #
# 一、状态模型（纯字典，方便整份快照 + 前进后退）
# --------------------------------------------------------------------------- #
def empty_state(capacity: int) -> dict:
    return {"capacity": capacity, "buffer": [None] * capacity, "head": 0, "tail": 0, "size": 0}


def state_to_list(st: dict) -> list:
    """等价于 RingBuffer.to_list()：从旧到新。"""
    cap = st["capacity"]
    return [st["buffer"][(st["head"] + i) % cap] for i in range(st["size"])]


def trace_write(state: dict, value) -> list[dict]:
    st = copy.deepcopy(state)
    cap = st["capacity"]
    steps: list[dict] = []

    def rec(desc: str, code: str, hl: tuple = (), note: str | None = None, ptr: tuple | None = None) -> None:
        steps.append(
            {
                "desc": desc,
                "code": code,
                "state": copy.deepcopy(st),
                "hl": list(hl),      # 数据发生变化的槽 → 红色
                "ptr": ptr,          # 指针移动到的槽 → 彩色虚线框（不是数据变化）
                "note": note,
            }
        )

    rec(
        f"要写入 {value}：先判断满没满（size = {st['size']} / capacity = {cap}）",
        "if self.size == self.capacity:",
    )

    if st["size"] == cap:
        slot = st["tail"]           # 满的时候 head == tail，被覆盖的就是这一格
        dropped = st["buffer"][slot]
        st["head"] = (st["head"] + 1) % cap
        rec(
            f"已经满了 → 读指针 head 前移一格到 {st['head']}，槽 {slot} 里最旧的 {dropped} 就此作废",
            "self.head = (self.head + 1) % self.capacity",
            ptr=("head", st["head"]),
            note=f"即将被覆盖：{dropped}（在槽 {slot}）",
        )
    else:
        st["size"] += 1
        rec(
            f"没满 → size 加一（{st['size'] - 1} → {st['size']}），两根指针都先不动",
            "self.size += 1",
        )

    where = st["tail"]
    st["buffer"][where] = value
    rec(f"真正写进去：buffer[{where}] = {value}", "self.buffer[self.tail] = data", hl=(where,))

    st["tail"] = (st["tail"] + 1) % cap
    wrapped = "，到末尾了 → 取模回到 0" if st["tail"] == 0 and cap > 1 else ""
    rec(
        f"写指针 tail 前移到 {st['tail']}{wrapped}",
        "self.tail = (self.tail + 1) % self.capacity",
        ptr=("tail", st["tail"]),
    )
    return steps


def trace_read(state: dict) -> list[dict]:
    st = copy.deepcopy(state)
    cap = st["capacity"]
    steps: list[dict] = []

    def rec(desc: str, code: str, hl: tuple = (), note: str | None = None, ptr: tuple | None = None) -> None:
        steps.append(
            {
                "desc": desc,
                "code": code,
                "state": copy.deepcopy(st),
                "hl": list(hl),
                "ptr": ptr,
                "note": note,
            }
        )

    if st["size"] == 0:
        rec("缓冲区是空的 → 直接返回 None，两根指针都不动", "if self.size == 0: return None")
        return steps

    slot = st["head"]
    value = st["buffer"][slot]
    rec(f"从 head = {slot} 处取值：{value}", "value = self.buffer[self.head]", hl=(slot,))

    st["buffer"][slot] = None
    rec(
        f"把槽 {slot} 清空（置 None）—— 不清的话，下次写进来之前它就是脏数据",
        "self.buffer[self.head] = None",
        hl=(slot,),
    )

    st["head"] = (slot + 1) % cap
    wrapped = "，到末尾了 → 取模回到 0" if st["head"] == 0 and cap > 1 else ""
    rec(
        f"读指针 head 前移到 {st['head']}{wrapped}",
        "self.head = (self.head + 1) % self.capacity",
        ptr=("head", st["head"]),
    )

    st["size"] -= 1
    rec(f"size 减一 → {st['size']}", "self.size -= 1")
    return steps


# --------------------------------------------------------------------------- #
# 二、界面
# --------------------------------------------------------------------------- #
class RingBufferApp:
    SLOT_W = 96
    SLOT_H = 76
    GAP = 18

    def __init__(self, root: tk.Tk):
        self.root = root
        self.state = empty_state(4)
        self.base_state = copy.deepcopy(self.state)
        self.steps: list[dict] = []
        self.idx = -1
        self.playing = False
        self._after_id = None
        self._next_value = 11

        self.f_body = tkfont.Font(family="Microsoft YaHei UI", size=10)
        self.f_bold = tkfont.Font(family="Microsoft YaHei UI", size=11, weight="bold")
        self.f_small = tkfont.Font(family="Microsoft YaHei UI", size=9)
        self.f_node = tkfont.Font(family="Consolas", size=16, weight="bold")
        self.f_code = tkfont.Font(family="Consolas", size=10)

        self._build_ui()
        self._bind_keys()
        self.render()

    # ------------------------------------------------------------------ 布局
    def _build_ui(self) -> None:
        self.root.title("环形缓冲可视化 · RingBuffer")
        self.root.geometry("1220x580")
        self.root.configure(bg=BG)

        bar = tk.Frame(self.root, bg=CARD, padx=14, pady=10)
        bar.pack(fill="x")

        tk.Label(bar, text="容量", bg=CARD, fg=MUTED, font=self.f_body).pack(side="left")
        self.var_cap = tk.IntVar(value=4)
        tk.Spinbox(
            bar, from_=1, to=8, width=3, textvariable=self.var_cap, font=self.f_code,
            justify="center", relief="solid", bd=1, command=self.rebuild,
        ).pack(side="left", padx=(6, 12))

        tk.Label(bar, text="写入值", bg=CARD, fg=MUTED, font=self.f_body).pack(side="left")
        self.var_value = tk.StringVar(value=str(self._next_value))
        tk.Entry(
            bar, textvariable=self.var_value, width=5, font=self.f_code,
            relief="solid", bd=1, justify="center",
        ).pack(side="left", padx=(6, 14))

        def btn(text, cmd, color=CARD, fg=INK):
            b = tk.Button(
                bar, text=text, command=cmd, font=self.f_body, bg=color, fg=fg,
                activebackground=color, relief="solid", bd=1, padx=10, pady=3, cursor="hand2",
            )
            b.pack(side="left", padx=4)
            return b

        btn("写入 ↓", lambda: self.do_write(1), "#e9f7ee", TAIL_C)
        btn("连写 3 个", lambda: self.do_write(3), "#e9f7ee", TAIL_C)
        btn("读出 ↑", lambda: self.do_read(), "#eaf2ff", HEAD_C)
        tk.Frame(bar, bg="#e4e7eb", width=1, height=22).pack(side="left", padx=10, fill="y")
        btn("演示一遍", self.demo)
        btn("重置", self.rebuild)

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

        # 按钮先占位（从右往左），状态文字再吃掉剩下的宽度
        self.btn_play.pack(side="right", padx=(4, 0))
        self.btn_next.pack(side="right", padx=4)
        self.btn_prev.pack(side="right", padx=4)
        self.lbl_stat.pack(side="left", fill="x", expand=True)

        # 底栏必须先占位，否则 canvas 的 expand=True 会把底栏挤出去
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
            return self._next_value

    def _bump_value(self) -> None:
        self._next_value += 1
        self.var_value.set(str(self._next_value))

    def _commit(self) -> None:
        if self.steps:
            self.state = copy.deepcopy(self.steps[-1]["state"])

    def _play(self, ops: list[tuple]) -> None:
        """把一串操作连成一条长步骤链。ops: [("write", 11), ("read", None), ...]"""
        self.stop_play()
        st = copy.deepcopy(self.state)
        base = copy.deepcopy(st)
        all_steps: list[dict] = []
        for op, value in ops:
            s = trace_write(st, value) if op == "write" else trace_read(st)
            st = s[-1]["state"]
            all_steps.extend(s)
        self.base_state = base
        self.steps = all_steps
        self.idx = 0
        self.render()

    def do_write(self, times: int = 1) -> None:
        self._commit()
        ops = [("write", self._value() + i) for i in range(times)]
        self._play(ops)
        self._next_value = self._value() + times
        self.var_value.set(str(self._next_value))

    def do_read(self) -> None:
        self._commit()
        self._play([("read", None)])

    def demo(self) -> None:
        """固定剧本：写满 → 再写一个（覆盖最旧）→ 读 → 写 → 读到空。"""
        self.rebuild()
        cap = self.state["capacity"]
        base = self._value()
        ops = [("write", base + i) for i in range(cap + 1)]    # 写 cap+1 个 → 必然覆盖一次
        ops += [("read", None), ("write", base + 99)]
        ops += [("read", None)] * (cap + 1)                    # 读到底，最后一次返回 None
        self._next_value = base + 100
        self.var_value.set(str(self._next_value))
        self._play(ops)
        self.toggle_play()

    def rebuild(self) -> None:
        self.stop_play()
        cap = max(1, min(8, int(self.var_cap.get())))
        self.state = empty_state(cap)
        self.base_state = copy.deepcopy(self.state)
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
        self._after_id = self.root.after(950, self._tick)   # 排一次，别直接调 _tick

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

    # ------------------------------------------------------------------ 绘制
    def _snap(self) -> dict:
        if self.steps:
            return self.steps[self.idx]
        return {
            "desc": "点击上方按钮：写入 / 读出都会被拆成若干步，用「下一步」看指针怎么绕圈",
            "code": "",
            "state": self.state,
            "hl": [],
            "ptr": None,
            "note": None,
        }

    def render(self) -> None:
        c = self.canvas
        c.delete("all")

        snap = self._snap()
        st = snap["state"]
        cap = st["capacity"]
        hot = set(snap.get("hl", []))

        w = max(c.winfo_width(), 900)
        h = c.winfo_height() if c.winfo_height() > 120 else 360
        total = cap * self.SLOT_W + (cap - 1) * self.GAP
        x0 = (w - total) / 2
        cy = max(150.0, min(h * 0.48, 230.0))
        top = cy - self.SLOT_H / 2
        bottom = cy + self.SLOT_H / 2

        self._draw_legend(w)
        self._draw_badge(w, st)

        # ---- 槽位 ----
        centers = []
        for i in range(cap):
            x = x0 + i * (self.SLOT_W + self.GAP)
            centers.append(x + self.SLOT_W / 2)
            self._draw_slot(x, top, st["buffer"][i], i, is_hot=(i in hot))

        # ---- 逻辑序号（从旧到新的第几个）----
        for order, slot in enumerate(range(st["head"], st["head"] + st["size"]), start=1):
            idx = slot % cap
            if st["buffer"][idx] is None:      # 中间态（先清空、后 size-=1）里那段空位不标序号
                continue
            x = x0 + idx * (self.SLOT_W + self.GAP) + self.SLOT_W - 16
            c.create_oval(x - 10, top + 6, x + 10, top + 26, fill="#eef4ff", outline=HEAD_C)
            c.create_text(x, top + 16, text=str(order), font=self.f_small, fill=HEAD_C)

        # ---- 本步移动的指针：彩色虚线框（和数据变化区分开）----
        ptr = snap.get("ptr")
        if ptr:
            kind, slot = ptr
            color = HEAD_C if kind == "head" else TAIL_C
            px = x0 + slot * (self.SLOT_W + self.GAP) - 5
            c.create_rectangle(
                px, top - 5, px + self.SLOT_W + 10, bottom + 5,
                outline=color, width=2, dash=(5, 3),
            )

        # ---- 两根指针 ----
        hx = centers[st["head"]]
        tx = centers[st["tail"]]
        self._arrow(hx, top - 40, hx, top - 8, HEAD_C)
        c.create_text(hx, top - 52, text="head（读）", font=self.f_small, fill=HEAD_C)
        self._arrow(tx, bottom + 40, tx, bottom + 8, TAIL_C)
        c.create_text(tx, bottom + 52, text="tail（写）", font=self.f_small, fill=TAIL_C)

        # head == tail 是「空」还是「满」？必须靠 size 才能区分
        hint_y = bottom + 78
        if st["head"] == st["tail"]:
            state_text = "满（size == capacity）" if st["size"] == cap else "空（size == 0）"
            c.create_text(
                w / 2, hint_y,
                text=f"head 与 tail 重合 → 只能靠 size 判断：{state_text}",
                font=self.f_small, fill=WARN_C,
            )
            hint_y += 22

        if snap.get("note"):
            c.create_text(w / 2, hint_y, text=snap["note"], font=self.f_small, fill=WARN_C)

        self._update_footer(snap, st)

    def _draw_legend(self, w: float) -> None:
        c = self.canvas
        y = 20
        x = 34.0
        c.create_text(x, y, text="图例", font=self.f_small, fill=MUTED, anchor="w")
        x += 36

        def text_after(marker, text, color):
            nonlocal x
            marker(x, y)
            x += 19
            tid = c.create_text(x, y, text=text, font=self.f_small, fill=color, anchor="w")
            x = c.bbox(tid)[2] + 20

        def arrow_marker(color):
            def draw(mx, my):
                c.create_line(mx, my - 6, mx, my + 6, fill=color, width=2,
                              arrow="last", arrowshape=(7, 9, 3))
            return draw

        def circle_marker(color):
            def draw(mx, my):
                c.create_oval(mx, my - 6, mx + 12, my + 6, outline=color, fill="#eef4ff")
            return draw

        text_after(arrow_marker(HEAD_C), "head 读指针", HEAD_C)
        text_after(arrow_marker(TAIL_C), "tail 写指针", TAIL_C)
        text_after(circle_marker(HEAD_C), "逻辑顺序（从旧到新第几个）", HEAD_C)
        text_after(
            lambda mx, my: c.create_rectangle(mx, my - 6, mx + 12, my + 6, outline=HOT, width=2),
            "本步改动", HOT,
        )

    def _draw_badge(self, w: float, st: dict) -> None:
        cap = st["capacity"]
        if st["size"] == 0:
            text, color = "空", MUTED
        elif st["size"] == cap:
            text, color = "满（再写就要覆盖最旧的）", WARN_C
        else:
            text, color = f"占用 {st['size']} / {cap}", INK
        self.canvas.create_text(w - 34, 20, text=text, font=self.f_small, fill=color, anchor="e")

    def _draw_slot(self, x: float, y: float, value, index: int, is_hot: bool) -> None:
        c = self.canvas
        outline = HOT if is_hot else (EMPTY_C if value is None else "#8b949e")
        fill = "#fff1f0" if is_hot else ("#f2f4f7" if value is None else CARD)
        dash = (4, 3) if value is None else None
        c.create_rectangle(
            x, y, x + self.SLOT_W, y + self.SLOT_H, fill=fill, outline=outline,
            width=3 if is_hot else 2, dash=dash,
        )
        c.create_text(x + 10, y + 12, text=f"[{index}]", font=self.f_small, fill=MUTED, anchor="w")
        if value is None:
            c.create_text(x + self.SLOT_W / 2, y + self.SLOT_H / 2 + 4, text="None",
                          font=self.f_small, fill="#a0a8b0")
        else:
            c.create_text(x + self.SLOT_W / 2, y + self.SLOT_H / 2 + 6, text=str(value),
                          font=self.f_node, fill=INK)

    def _arrow(self, x1, y1, x2, y2, color) -> None:
        self.canvas.create_line(x1, y1, x2, y2, fill=color, width=3,
                                arrow="last", arrowshape=(11, 13, 4))

    def _update_footer(self, snap: dict, st: dict) -> None:
        total = len(self.steps)
        head = f"第 {self.idx + 1} / {total} 步 　" if total else ""
        self.lbl_desc.config(text=head + snap["desc"])
        self.lbl_code.config(text=("　　" + snap["code"]) if snap["code"] else "")
        self.lbl_stat.config(
            text=f"capacity = {st['capacity']}　size = {st['size']}　"
                 f"head = {st['head']}　tail = {st['tail']}　"
                 f"to_list() = {state_to_list(st)}"
        )
        self.btn_prev.config(state="normal" if self.idx > 0 else "disabled")
        self.btn_next.config(state="normal" if total and self.idx < total - 1 else "disabled")


# --------------------------------------------------------------------------- #
# 三、自检 / 截图
# --------------------------------------------------------------------------- #
def run_self_check() -> int:
    if RingBufferImpl is None:
        print("⚠ 跳过自检：导入 queue.py 里的 RingBuffer 失败")
        print(f"  {_IMPORT_ERROR}")
        return 2

    ops = [
        ("write", 1), ("write", 2), ("write", 3), ("write", 4),
        ("write", 5),            # 满之后再写 → 覆盖最旧
        ("read", None), ("write", 6), ("read", None),
        ("read", None), ("read", None),
        ("write", 7), ("read", None), ("read", None), ("read", None),
    ]
    real = RingBufferImpl(4)
    st = empty_state(4)
    ok = True

    for op, value in ops:
        if op == "write":
            real.write(value)
            steps = trace_write(st, value)
            expect_value = None
        else:
            expect_value = real.read()
            model_value = st["buffer"][st["head"]] if st["size"] else None
            steps = trace_read(st)
            if model_value != expect_value:
                ok = False
                print(f"✗ read：模型给出 {model_value}，真实实现给出 {expect_value}")

        st = steps[-1]["state"]
        if st["buffer"] != real.buffer or st["head"] != real.head or st["tail"] != real.tail or st["size"] != real.size:
            ok = False
            print(f"✗ {op}({value})：模型 buffer={st['buffer']} head={st['head']} tail={st['tail']} size={st['size']}")
            print(f"           真实 buffer={real.buffer} head={real.head} tail={real.tail} size={real.size}")
        if state_to_list(st) != real.to_list():
            ok = False
            print(f"✗ {op}({value})：to_list 不一致 {state_to_list(st)} vs {real.to_list()}")

    print()
    print(f"PASS —— 可视化模型与 RingBuffer 完全一致（{len(ops)} 次写入/读取，指针与数据全部对齐）"
          if ok else "FAIL —— 见上面的差异")
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
    root.geometry("1220x580+60+60")
    app = RingBufferApp(root)
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
    app = RingBufferApp(root)
    root.update_idletasks()
    app.demo()
    for _ in range(10):
        app.next_step()
        root.update()
    app.stop_play()
    root.update()
    root.destroy()
    print("smoke OK —— 窗口能创建、能绘制、能单步")
    return 0


def main() -> int:
    _dpi_aware()
    ap = argparse.ArgumentParser(description="环形缓冲可视化")
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
    RingBufferApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

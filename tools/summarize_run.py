#!/usr/bin/env python
"""把 runs/detect/<某次训练>/ 翻译成人话：哪一轮最好、有没有过拟合、下一步改什么。

为什么要有这个脚本（三个最容易看错的点）：
    1. **"最后一轮"不等于"最好的一轮"** —— best.pt 是按 fitness 挑的。本机 ultralytics
       8.4.x 的 fitness 直接取 **mAP50-95**（老版本才是 0.1×mAP50 + 0.9×mAP50-95）。
       脚本会把最佳轮次、以及"如果按 mAP50 挑会挑到哪一轮"都算给你。
    2. **光看 mAP 看不出过拟合** —— 必须成对看 `train/` 与 `val/` 的 loss 走向。
       这里用最小二乘斜率判方向（降/升），再据此给判词。
    3. **一堆超参数看不出差异** —— `--compare` 会自动 diff 两个 run 的 args.yaml，
       只列出**不同的**那几个参数（这就是"一次只改一个变量"的核验工具）。

用法：
    python tools/summarize_run.py                            # 最近一次训练
    python tools/summarize_run.py --list                     # 列出所有 run（按时间倒序）
    python tools/summarize_run.py --run train-coco8           # 指定，名字或路径都行
    python tools/summarize_run.py --compare train-coco8 train-leaves   # 对照表 + 参数 diff
    python tools/summarize_run.py --per-class                # 另跑一次 val，出每类 mAP（最差类）

    --per-class 需要 ultralytics（会加载 best.pt 重新验证，几十秒）；不加则只读
    args.yaml + results.csv，纯标准库、秒出、不占用显存。

配套文档：docs/术语速查.md
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "detect"
BAR = "─" * 64
BARS = "▁▂▃▄▅▆▇█"

# 输出被重定向（> file.md / 管道）时强制 UTF-8，否则 Windows 会按 cp936 写，粘进文档全是乱码。
# 直接在终端里跑（tty）时不动它，交给控制台自己的编码。
if not sys.stdout.isatty():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------- 小工具


def dwidth(s: str) -> int:
    """字符串的显示宽度（中文算 2 格），用来对齐含中文的表格。"""
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)


def pad(s: str, width: int) -> str:
    return s + " " * max(0, width - dwidth(s))


def is_num(v: str) -> bool:
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


def slope(xs: list[float], ys: list[float]) -> float:
    """最小二乘斜率（单位：每轮变化量）。点太少或没有横轴变化时返回 0。"""
    n = len(xs)
    if n < 2:
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    if den == 0:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den


def spark(vals: list[float], width: int = 56) -> str:
    """把一条序列压成一行方块图（▁ = 最低，█ = 最高）。"""
    if not vals:
        return ""
    step = max(1, math.ceil(len(vals) / width))
    buckets = [sum(vals[i:i + step]) / len(vals[i:i + step]) for i in range(0, len(vals), step)]
    lo, hi = min(buckets), max(buckets)
    if hi - lo < 1e-12:
        return BARS[0] * len(buckets)
    return "".join(BARS[min(len(BARS) - 1, int((v - lo) / (hi - lo) * (len(BARS) - 1)))] for v in buckets)


# ---------------------------------------------------------------- 读数据


@dataclass
class Run:
    path: Path
    args: dict = field(default_factory=dict)
    header: list = field(default_factory=list)
    rows: list = field(default_factory=list)

    def col(self, *needles: str, exclude: tuple = ()) -> str | None:
        """按子串找列名（results.csv 的列名随版本变，硬编码会挂）。"""
        for h in self.header:
            hl = h.lower()
            if all(n.lower() in hl for n in needles) and not any(e.lower() in hl for e in exclude):
                return h
        return None

    def series(self, *needles: str, exclude: tuple = ()) -> list[float]:
        c = self.col(*needles, exclude=exclude)
        if c is None:
            return []
        return [float(r[c]) for r in self.rows if is_num(r.get(c, ""))]

    def arg(self, key: str, default: str = "?") -> str:
        return str(self.args.get(key, default)).strip("'\"")


def load_args(path: Path) -> dict:
    """args.yaml 是扁平的 key: value，手解足够（不引入 pyyaml 依赖）。"""
    out: dict = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if ":" not in line or line.lstrip().startswith("#"):
            continue
        k, v = line.split(":", 1)
        out[k.strip()] = v.strip()
    return out


def load_run(spec: str | Path) -> Run:
    p = Path(str(spec))
    if p.suffix.lower() == ".csv":                      # 直接给了 results.csv
        p = p.parent
    elif not p.is_absolute() and not (p / "results.csv").exists():
        p = RUNS / p                                    # 给了 run 名字
    if not (p / "results.csv").exists():
        raise SystemExit(f"✗ 找不到 run：{spec}（没有 results.csv）")
    run = Run(path=p, args=load_args(p / "args.yaml"))
    with (p / "results.csv").open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        run.header = list(reader.fieldnames or [])
        run.rows = list(reader)
    if not run.rows:
        raise SystemExit(f"✗ {p / 'results.csv'} 是空的")
    return run


def all_runs() -> list[Run]:
    if not RUNS.exists():
        raise SystemExit(f"✗ 没有 {RUNS}，还没训练过？")
    items = []
    for csv_path in RUNS.rglob("results.csv"):
        try:
            items.append(load_run(csv_path.parent))
        except SystemExit:
            continue
    items.sort(key=lambda r: (r.path / "results.csv").stat().st_mtime, reverse=True)
    return items


# ---------------------------------------------------------------- 分析


@dataclass
class Stats:
    """一次 run 的关键指标（全部算好，打印与对比共用）。"""
    n: int = 0
    best_idx: int = 0
    best_epoch: int = 0
    best_map50: float = 0.0
    best_map: float = 0.0              # mAP50-95，= 本版本 fitness
    last_epoch: int = 0
    last_map50: float = 0.0
    last_map: float = 0.0
    last_p: float = 0.0
    last_r: float = 0.0
    best_p: float = 0.0
    best_r: float = 0.0
    sec_per_epoch: float = 0.0
    train_slope: float = 0.0
    val_slope: float = 0.0
    alt_best_epoch: int = 0            # 如果按 mAP50 挑
    converge_epoch: int = 0            # 首次达到 best 的 98%
    verdict: str = ""
    notes: list = field(default_factory=list)
    todo: list = field(default_factory=list)
    missing: list = field(default_factory=list)


def analyze(run: Run) -> Stats:
    s = Stats(n=len(run.rows))
    s.missing = [name for name, ok in [
        ("train/box_loss", run.col("train/", "box_loss")),
        ("train/cls_loss", run.col("train/", "cls_loss")),
        ("val/box_loss", run.col("val/", "box_loss")),
        ("mAP50-95", run.col("mAP50-95")),
    ] if not ok]

    m50 = run.series("mAP50", exclude=("95",))
    m5095 = run.series("mAP50-95")
    if not m5095:
        raise SystemExit("✗ results.csv 里没有 mAP50-95 列，版本太老？")
    s.n = min(len(run.rows), len(m5095))
    epochs = [int(float(r.get("epoch", i + 1))) for i, r in enumerate(run.rows[:s.n])]

    # fitness：本机 8.4.x 就是 mAP50-95；老版本是 0.1*mAP50 + 0.9*mAP50-95
    fit = m5095[:s.n]
    # 并列时取最早的：best.pt 是在第一次达到该分数的那个 epoch 存下来的
    s.best_idx = max(range(s.n), key=lambda i: fit[i])
    s.best_epoch, s.best_map = epochs[s.best_idx], fit[s.best_idx]
    s.last_epoch, s.last_map = epochs[-1], fit[-1]
    if m50 and len(m50) >= s.n:
        s.best_map50 = m50[s.best_idx]
        s.last_map50 = m50[-1]
        s.alt_best_epoch = epochs[max(range(s.n), key=lambda i: m50[i])]

    p = run.series("precision")
    r = run.series("recall")
    if p and r and min(len(p), len(r)) > s.best_idx:
        s.best_p, s.best_r = p[s.best_idx], r[s.best_idx]
        s.last_p, s.last_r = p[-1], r[-1]

    t = run.series("time")
    s.sec_per_epoch = (t[s.n - 1] / epochs[-1]) if t and epochs[-1] else 0.0

    tb = run.series("train/", "box_loss")[:s.n]
    vb = run.series("val/", "box_loss")[:s.n]
    s.train_slope = slope(epochs[:len(tb)], tb) if len(tb) == s.n else 0.0
    s.val_slope = slope(epochs[:len(vb)], vb) if len(vb) == s.n else 0.0

    thresh = 0.98 * s.best_map
    s.converge_epoch = next((epochs[i] for i in range(s.n) if fit[i] >= thresh), s.best_epoch)

    _diagnose(run, s, fit, tb, vb, m50, m5095)
    return s


def _diagnose(run: Run, s: Stats, fit, tb, vb, m50, m5095) -> None:
    """判词 + 建议。规则都写在这一个函数里，方便以后加。"""
    n, best = s.n, s.best_epoch
    s.verdict = "健康（train/val 同向下降）"

    if n == 1:
        s.verdict = "只有 1 轮 —— 只能验证管道通不通，不能评判好坏"
        s.notes.append("1 轮的训练不足以收敛；这不是调参结果，只是冒烟测试。")
        s.todo.append("正经训练：epochs ≥ 30，patience ≥ 10")
        return

    if s.val_slope > 0 and s.train_slope < 0:
        s.verdict = "过拟合（train 降 + val 升）"
    elif s.val_slope > 0 and s.train_slope > 0:
        s.verdict = "训练异常（两边都在升，看 lr / 数据是否坏）"
    elif s.val_slope < 0 and s.train_slope >= 0:
        s.verdict = "train loss 未下降（可能在 warmup 期）—— 别急着判欠拟合，先加 epochs"

    # ---- 实验设计体检：这两条经常让整组对照白做
    if n > 2 and s.best_epoch == s.last_epoch:
        s.notes.append(
            f"最佳出现在**最后一轮**（epoch {s.last_epoch} / 共 {n} 轮）→ 训练轮数不够，"
            "模型还在进步。此时比较各组 ≈ 比“谁跑得快”，不是比“谁最终更好”。")

    epochs_arg = int(float(run.arg("epochs", "0") or 0))
    cm = int(float(run.arg("close_mosaic", "0") or 0))
    if epochs_arg and cm >= epochs_arg - 3:
        s.notes.append(
            f"close_mosaic={cm} 而 epochs={epochs_arg} → mosaic 只在最初 "
            f"{max(0, epochs_arg - cm)} 轮生效，"
            "“关掉 mosaic”这组实验几乎没有对照意义（要 epochs ≥ 25 才做得出）。")

    dropped = s.best_map - s.last_map
    if best < n * 0.3 and dropped > 0.05:
        s.verdict += f"｜严重：最佳在 {best / n:.0%} 处，后 {n - best} 轮是负收益"

    s.notes.append(
        f"最佳轮次只占总轮数的 {best / n:.0%}，首次达到最佳 98% 是在 epoch {s.converge_epoch}。")
    if s.best_map - s.last_map > 1e-9:
        s.notes.append(
            f"末轮 mAP50-95 {s.last_map:.4f} 比最佳 {s.best_map:.4f} 低 {s.best_map - s.last_map:.4f}。")

    patience = int(float(run.arg("patience", "0") or 0))
    if patience >= n > 1:
        s.notes.append(f"patience={patience} ≥ epochs={n} → **早停等于被关掉**，注定跑满。")

    gap = s.best_map50 - s.best_map
    if s.best_map50 > 0 and gap > 0.20:
        s.notes.append(
            f"mAP50 {s.best_map50:.3f} 比 mAP50-95 {s.best_map:.3f} 高 {gap:.2f} → "
            "IoU 0.5 能对上但位置不够准，检查标注框是否贴合。")

    if s.best_p and s.best_r and abs(s.best_p - s.best_r) > 0.15:
        if s.best_p < s.best_r:
            s.notes.append(f"最佳轮 P={s.best_p:.3f} 远低于 R={s.best_r:.3f} → 报的框里假框多。")
        else:
            s.notes.append(f"最佳轮 P={s.best_p:.3f} 远高于 R={s.best_r:.3f} → 太保守，漏检多。")

    # val 指标连续多轮完全不变 → warmup 期 lr 极小，模型没动
    frozen = 0
    for i in range(1, len(m5095)):
        if abs(m5095[i] - m5095[i - 1]) < 1e-9:
            frozen += 1
        else:
            frozen = 0
        if frozen >= 3:
            s.notes.append(
                "val 指标连续 ≥4 轮完全不变 → 模型已经不在学（warmup 期 lr 太小，"
                "或后期 lr 衰减到 ~0，看 lr/pg0 列）。")
            break

    # ---------------- 建议（按优先级排）
    if "过拟合" in s.verdict:
        s.todo.append("加数据（最有效）；数据不动就加增强：mosaic、scale、degrees、mixup")
        s.todo.append(f"epochs {n} → {max(2 * best, 10)} 左右（best={best}，留一点余量）")
        s.todo.append(f"patience 设 {max(10, best * 3)}，让它自己在低谷停")
    if s.best_p and s.best_r and s.best_p < s.best_r - 0.15:
        s.todo.append("补「无目标」背景图当负样本，并适当提 conf（推理时）")
    if s.best_p and s.best_r and s.best_r < s.best_p - 0.15:
        s.todo.append("补难例（遮挡/小目标/暗光），并试 imgsz 640 → 960")
    imgsz = int(float(run.arg("imgsz", "640") or 640))
    if imgsz <= 640:
        s.todo.append(f"imgsz={imgsz}：若小目标漏检，这一项比 lr 值得先试（640 → 960）")
    s.todo.append("固定 batch 再动其它参数（改 batch 会顺带改有效学习率，nbs=64 累积）")


# ---------------------------------------------------------------- 打印


def print_run(run: Run, s: Stats, show_table: int = 0) -> None:
    name = run.path.name
    print(BAR)
    print(f"运行目录：{run.path.relative_to(ROOT) if run.path.is_relative_to(ROOT) else run.path}")
    print(BAR)

    print("\n【1】这是什么实验（args.yaml 是唯一可信的参数记录）")
    for k in ("model", "data", "epochs", "batch", "imgsz", "patience",
              "optimizer", "lr0", "lrf", "seed", "fraction", "close_mosaic", "mosaic"):
        print(f"  {pad(k, 12)} {run.arg(k)}")
    device = run.arg("device", "")
    amp = run.arg("amp", "")
    print(f"  {pad('device/amp', 12)} {device} / {amp}")

    print("\n【2】最佳轮次（best.pt 的挑选依据 = fitness = mAP50-95，8.4.x 起）")
    print(f"  ★ best  epoch {s.best_epoch:<4} mAP50 {s.best_map50:.4f}   mAP50-95 {s.best_map:.4f}"
          f"   ← best.pt 存的是这一轮")
    print(f"    末轮  epoch {s.last_epoch:<4} mAP50 {s.last_map50:.4f}   mAP50-95 {s.last_map:.4f}"
          f"   ← last.pt 存的是这一轮")
    if s.alt_best_epoch and s.alt_best_epoch != s.best_epoch:
        print(f"    注：如果按 mAP50 挑，会挑到 epoch {s.alt_best_epoch}（不是同一个）")
    if s.sec_per_epoch:
        print(f"    均摊 {s.sec_per_epoch:.2f} s/轮（含首轮启动开销），总共 {s.n} 轮")

    if s.n > 2:
        m5095 = run.series("mAP50-95")
        tb = run.series("train/", "box_loss")
        vb = run.series("val/", "box_loss")
        print("\n【3】曲线形状（左=第1轮，右=最后一轮）")
        print(f"  mAP50-95   {spark(m5095)}  {min(m5095):.3f}~{max(m5095):.3f}")
        if tb:
            print(f"  train_box  {spark(tb)}  {tb[0]:.3f}→{tb[-1]:.3f}  "
                  f"斜率 {s.train_slope:+.5f}/轮")
        if vb:
            print(f"  val_box    {spark(vb)}  {vb[0]:.3f}→{vb[-1]:.3f}  "
                  f"斜率 {s.val_slope:+.5f}/轮")
        print("  口诀：train 降 + val 降 = 健康；train 降 + val 升 = 过拟合")
    else:
        print("\n【3】曲线形状：轮数太少（≤2），看不出趋势，跳过")

    print("\n【4】精度/召回")
    if s.best_p or s.best_r:
        f1 = 2 * s.best_p * s.best_r / (s.best_p + s.best_r) if (s.best_p + s.best_r) else 0.0
        print(f"  最佳轮  P {s.best_p:.3f}  R {s.best_r:.3f}  F1 {f1:.3f}")
        f1 = 2 * s.last_p * s.last_r / (s.last_p + s.last_r) if (s.last_p + s.last_r) else 0.0
        print(f"  末轮    P {s.last_p:.3f}  R {s.last_r:.3f}  F1 {f1:.3f}")
    else:
        print("  （CSV 里没有 precision/recall 列）")

    print(f"\n【5】判词：{s.verdict}")
    for note in s.notes:
        print(f"  · {note}")
    if s.missing:
        print(f"  · 缺失列：{', '.join(s.missing)}（不同版本列名不同，已跳过）")

    print("\n【6】下一步该改什么（按优先级）")
    for i, item in enumerate(s.todo, 1):
        print(f"  {i}. {item}")

    print("\n【7】还要亲眼看这三张图（脚本不替你看）")
    print(f"  results.png                    判断健康/过拟合的整体形状")
    print(f"  confusion_matrix_normalized.png 最右列=误检，最后一行=漏检，对角线≈召回率")
    print(f"  val_batch0_labels.jpg vs _pred  真值 vs 预测，用眼睛找失败模式")
    print(f"  （都在 {run.path.name}/ 下；这三张默认用 best.pt 生成）")

    if show_table > 0 and s.n > 1:
        step = max(1, s.n // show_table)
        print(f"\n【8】抽样明细（每 {step} 轮一行）")
        m50 = run.series("mAP50", exclude=("95",))
        m5095 = run.series("mAP50-95")
        vb2 = run.series("val/", "box_loss")
        keep = list(range(0, s.n, step))
        if keep[-1] != s.n - 1:
            keep.append(s.n - 1)
        hdr = f"  {'epoch':>6}{'mAP50':>9}{'mAP50-95':>10}{'val_box':>9}"
        print(hdr)
        print("  " + "-" * (dwidth(hdr) - 2))
        for i in keep:
            e = run.rows[i].get("epoch", i + 1)
            a = f"{m50[i]:.4f}" if m50 and len(m50) > i else "     -"
            b = f"{m5095[i]:.4f}" if len(m5095) > i else "     -"
            c = f"{vb2[i]:.4f}" if vb2 and len(vb2) > i else "     -"
            mark = "  ★" if i == s.best_idx else ""
            print(f"  {str(e):>6}{a:>9}{b:>10}{c:>9}{mark}")


def print_table(runs: list[Run], stats: list[Stats], title: str, delta: bool = False) -> None:
    print(f"\n{title}")
    head = f"{'run':<24}{'model':<11}{'ep':>4}{'imgsz':>6}{'batch':>6}{'best':>6}" \
           f"{'mAP50':>8}{'mAP50-95':>10}"
    head += f"{'Δ vs 基线':>11}" if delta else ""
    head += f"{'s/轮':>7}  判词"
    print(head)
    print("-" * (dwidth(head) + 20))
    base = stats[0].best_map if stats else 0.0
    for i, (run, s) in enumerate(zip(runs, stats)):
        line = (f"{pad(run.path.name, 24)}{pad(run.arg('model'), 11)}"
                f"{run.arg('epochs'):>4}{run.arg('imgsz'):>6}{run.arg('batch'):>6}"
                f"{s.best_epoch:>6}{s.best_map50:>8.3f}{s.best_map:>10.3f}")
        if delta:
            d = s.best_map - base
            line += f"{('—' if i == 0 else f'{d:+.3f}'):>11}"
        line += f"{s.sec_per_epoch:>7.2f}  {s.verdict.split('｜')[0]}"
        print(line)


# 这些是 ultralytics 自己填的元数据，不属于实验设计，diff 时要忽略。
# 注意：seed / device / workers 不忽略 —— 它们也可能就是你的实验变量（D1 就是只改 seed）。
SKIP_ARGS = {"name", "project", "save_dir", "source", "cfg", "tracker"}


def arg_diffs(base: Run, run: Run) -> list[str]:
    keys = list(dict.fromkeys([*base.args, *run.args]))
    return [k for k in keys
            if k not in SKIP_ARGS and str(base.args.get(k)) != str(run.args.get(k))]


def print_diff(runs: list[Run]) -> None:
    """以第一组为基线，逐组打印"只改了哪个参数"。"""
    base = runs[0]
    print(f"\n各组 vs 基线（{base.path.name}）：")
    for run in runs[1:]:
        diffs = arg_diffs(base, run)
        if not diffs:
            print(f"  {pad(run.path.name, 24)} 与基线完全相同 —— 这不是对照实验")
        elif len(diffs) == 1:
            k = diffs[0]
            print(f"  {pad(run.path.name, 24)} ✅ 只改了「{k}」: "
                  f"{base.args.get(k)} → {run.args.get(k)}")
        else:
            print(f"  {pad(run.path.name, 24)} ⚠ 改了 {len(diffs)} 个参数，结论不可归因：")
            for k in diffs:
                print(f"      {pad(k, 14)} {base.args.get(k)} → {run.args.get(k)}")


def print_groups(groups) -> None:
    """分组比较：一组 = 同一个配置跑多个 seed。这是唯一能扛住抖动的比法。

    判定规则（很粗，但比拍脑袋强）：
        噪声 ≈ max(基线组 std, 本组 std)
        |Δ| ≤ 噪声      → 无显著差异
        噪声 < |Δ| ≤ 2σ → 弱（要补 seed）
        |Δ| > 2σ        → 显著
        n=1             → 连噪声都估不出来，一律判"无法判定"
    """
    import statistics as st

    print(f"\n分组比较（第一组 = 基线；每组内的 run 应该是同一配置的不同 seed）\n")
    print(f"  {'组':<12}{'n':>3}{'mAP50-95 均值':>15}{'±std':>9}{'Δ vs 基线':>12}")
    print("  " + "-" * 50)

    base_name, _, base_stats = groups[0]
    base_vals = [s.best_map for s in base_stats]
    base_mean = st.mean(base_vals)
    base_std = st.pstdev(base_vals) if len(base_vals) > 1 else 0.0

    for name, runs, stats in groups:
        vals = [s.best_map for s in stats]
        mean = st.mean(vals)
        std = st.pstdev(vals) if len(vals) > 1 else 0.0
        delta = mean - base_mean

        if len(vals) < 2 or len(base_vals) < 2:
            judge = "n=1，无法判定（噪声地板未知）"
        else:
            noise = max(base_std, std)
            if abs(delta) <= noise:
                judge = "无显著差异"
            elif abs(delta) <= 2 * noise:
                judge = "弱（<2σ），要补 seed"
            else:
                judge = "显著（>2σ）"
        d_txt = "—" if name == base_name else f"{delta:+.3f}"
        print(f"  {pad(name, 12)}{len(vals):>3}{mean:>15.3f}{std:>9.3f}{d_txt:>12}")
        members = " ".join(f"{r.path.name}({s.best_map:.3f})" for r, s in zip(runs, stats))
        print(f"      判定：{judge}")
        print(f"      成员：{members}")

    if len(base_vals) < 2:
        print(f"\n⚠ 基线组只有 {len(base_vals)} 个 seed → 噪声地板未知，上面所有 Δ 都不能当结论。")
        print("   先补：同一个配置换 seed 再跑 2~4 次，放进基线组。")


def print_per_class(run: Run, s: Stats) -> None:
    best = run.path / "weights" / "best.pt"
    if not best.exists():
        print(f"\n【每类表现】跳过：找不到 {best}")
        return
    try:
        from ultralytics import YOLO
    except Exception as exc:  # noqa: BLE001
        print(f"\n【每类表现】跳过：没装 ultralytics（{exc}）")
        return
    print(f"\n【每类表现】正在用 {best.name} 跑一次 val（几十秒）…")
    model = YOLO(str(best))
    res = model.val(data=run.arg("data"), imgsz=int(float(run.arg("imgsz", "640") or 640)),
                    split="val", plots=False, verbose=False,
                    project=str(RUNS), name="summarize_val", exist_ok=True)
    box = res.box
    maps = getattr(box, "maps", None)
    idx = list(getattr(box, "ap_class_index", []))
    names = res.names if isinstance(res.names, dict) else dict(enumerate(res.names))
    if maps is None or not idx:
        print("  这个版本拿不到每类 mAP")
        return
    rows = sorted(((names.get(c, str(c)), float(maps[c])) for c in idx), key=lambda t: t[1])
    print(f"  {'类别':<20}{'mAP50-95':>10}   备注")
    worst = rows[0] if len(rows) > 1 else None
    for cname, val in rows:
        note = "← 最差，优先补这类数据" if worst and (cname, val) == worst else ""
        print(f"  {pad(cname, 20)}{val:>10.4f}   {note}")
    if len(rows) == 1:
        print("  （验证集里只有 1 个类别，没有可对比的“最差类”）")
    print("  （脚本只报数；为什么差，要么看混淆矩阵，要么自己翻 val 图）")


# ---------------------------------------------------------------- 入口


def main() -> int:
    ap = argparse.ArgumentParser(
        description="把 runs/detect/<run> 翻译成人话", formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", help="run 名字或路径；缺省取最近一次")
    ap.add_argument("--list", action="store_true", help="列出所有 run")
    ap.add_argument("--compare", nargs="+", help="对照表 + 超参数 diff（至少 2 个）")
    ap.add_argument("--per-class", action="store_true", help="另跑一次 val，出每类 mAP（需 ultralytics）")
    ap.add_argument("--table", type=int, default=0, metavar="N", help="额外打印 N 行抽样明细")
    ap.add_argument("--groups", nargs="+", metavar="NAME=run1,run2",
                    help="分组比较（组内放多个 seed），如 base=b-s0,b-s1 img320=i-s0,i-s1")
    args = ap.parse_args()

    if args.groups:
        groups = []
        for spec in args.groups:
            if "=" not in spec:
                raise SystemExit(f"✗ 写法是 组名=run1,run2 ，你给的是 {spec}")
            name, members = spec.split("=", 1)
            runs = [load_run(m.strip()) for m in members.split(",") if m.strip()]
            if not runs:
                raise SystemExit(f"✗ 组 {name} 里没有 run")
            groups.append((name, runs, [analyze(r) for r in runs]))
        if len(groups) < 2:
            raise SystemExit("✗ --groups 至少给两组（第一组当基线）")
        print_groups(groups)
        return 0

    if args.compare:
        if len(args.compare) < 2:
            raise SystemExit("✗ --compare 至少给两个 run（第一个当基线）")
        runs = [load_run(x) for x in args.compare]
        stats = [analyze(r) for r in runs]
        print_table(runs, stats, f"对照表（基线 = {runs[0].path.name}）", delta=True)
        print_diff(runs)
        print("\n各组判词与体检：")
        for run, s in zip(runs, stats):
            print(f"  {run.path.name}：{s.verdict}")
            for note in s.notes:
                print(f"      · {note}")
        return 0

    if args.list:
        runs = all_runs()
        print_table(runs, [analyze(r) for r in runs], f"共 {len(runs)} 个 run（按时间倒序）")
        return 0

    run = load_run(args.run) if args.run else all_runs()[0]
    s = analyze(run)
    print_run(run, s, show_table=args.table)
    if args.per_class:
        print_per_class(run, s)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())

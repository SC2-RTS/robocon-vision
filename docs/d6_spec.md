# D6 · 流水线规格书

> 这份文件回答的是"**每个文件必须满足什么**"，不是"怎么写"。
> 判据都写成"跑一条命令 + 看一眼结果"的形式，做不到就是没做完。

---

## 0. 目标与非目标

**目标**：把 D5 的 `find_targets()` 变成一个**能交给队友跑**的程序。算法一行不改。

**非目标**（做了算跑偏）：

- 不追求更高的检测精度（D5 已经验收过）
- 不换算法、不加新模型
- 不做 GUI 界面（命令行 + 窗口就够）
- 不做单元测试全覆盖（只测 3 个纯函数就够，见 §7）

---

## 1. 数据契约（先定这个，其余都围着它转）

### 1.1 `Target`（贯穿全程的字典）

```python
Target = {
    "id":    int,     # 稳定编号（帧内唯一；不要指望跨帧一致，见 §5.2）
    "cx":    float,   # 中心 x（**原图**坐标系，不是裁剪图）
    "cy":    float,   # 中心 y（同上）
    "w":     float,   # 旋转矩形的长边
    "h":     float,   # 旋转矩形的短边
    "angle": float,   # 长边相对水平轴，[-90, 90)
    "area":  float,   # 轮廓像素面积
}
```

**为什么坐标必须是原图坐标**：ROI 裁剪/缩放之后检测出来的坐标是"小图坐标"，画到原图上会错位。还原是 `preprocess` 和 `detect` 之间的契约。

### 1.2 `Frame`

```python
@dataclass
class Frame:
    image: np.ndarray   # BGR，原始分辨率
    index: int          # 帧序号，从 0 开始（含跳过的帧）
    t: float            # 时间戳（秒，从开始播放算，不是 wall clock）
```

### 1.3 `configs/classic.yaml`（完整 schema）

```yaml
source:
  kind: video                  # image | video | camera
  path: data/samples/wobble.mp4
  step: 1                      # 跳帧：每 N 帧取 1（1 = 不跳）

preprocess:
  resize: 1.0                  # 0.5 = 半分辨率（快 4 倍，精度略降）
  roi: null                    # [x, y, w, h] 或 null（null = 全图）

detect:
  params:                      # 直接对应 D5 的 Params，六个字段一个不少
    median_k: 5
    otsu_delta: 0
    open_k: 0
    close_k: 0
    min_area: 2000
    use_clahe: false

smooth:
  method: median               # none | median | ema
  window: 5                    # median 用
  ema_alpha: 0.4               # ema 用

rank:
  enable: false                # 多目标时只留前 N 个
  key: area                    # area | conf
  top_n: 3

visualize:
  show: true
  save: null                   # 例：data/out/d6.mp4
  draw_raw: true               # 红框：原始检测
  draw_smoothed: true          # 绿框：平滑后
  show_mask: false             # 是否再开一个 mask 窗口

log:
  level: INFO
  fps_window: 30               # FPS 用最近多少帧的平均
```

**判据**：程序里搜不到 `2000` 这类数字（除了 yaml 和默认值），全部从配置来。

---

## 2. 模块职责与接口

### 2.1 `capture.py` —— 把"输入"变成一个统一的帧迭代器

```python
class FrameSource:
    def __init__(self, cfg: dict) -> None
    def __enter__(self) -> "FrameSource"
    def __exit__(self, *exc) -> None
    def read(self) -> Frame | None      # 读完/断流 → None
    @property
    def fps(self) -> float              # 视频上报帧率；图片源返回 0
```

必须支持三种 `kind`：

| kind | 行为 | 边界 |
|---|---|---|
| `image` | 同一张图只产出 1 帧（index=0） | 文件不存在 → 抛 `FileNotFoundError`，别返回 None |
| `video` | 顺序读，读完返回 None | 路径错 → 抛错；帧数 0 → 抛错 |
| `camera` | `VideoCapture(0)` | 打不开 → 抛错（**别静默返回空**） |

`step` 的含义：读 `step` 次只产出 1 帧，但 `index` 仍按真实帧号累加。

**验收**：

```bat
python -c "from src.classic.capture import FrameSource; ..."   # 三种源各读 5 帧，打印 index/t/fps
```

三种源都能读到帧，且 `t` 单调递增。

---

### 2.2 `preprocess.py` —— ROI + 缩放 + 转灰度，**并吐出坐标变换**

```python
@dataclass(frozen=True)
class Transform:
    offset: tuple[int, int]        # ROI 左上角在原图的 (x, y)
    scale: float                   # resize 比例
    full_size: tuple[int, int]     # 原图 (w, h)

    def to_full(self, cx: float, cy: float) -> tuple[float, float]

def prepare(frame_bgr: np.ndarray, cfg: dict) -> tuple[np.ndarray, Transform]
```

`to_full` 的公式（自己推导，不要背）：

```
cx_full = cx_small / scale + offset_x
cy_full = cy_small / scale + offset_y
```

**验收**：

| 配置 | 期望 |
|---|---|
| `resize: 1.0, roi: null` | 目标数 3，中心与不裁剪时误差 0 |
| `resize: 0.5` | 目标数 3，中心误差 < 3px |
| `roi` 取左上半边 | 落在 ROI 内的目标被检出，ROI 外的没有；坐标是原图坐标 |

**这是最容易错的地方**：如果 `to_full` 写错，框会整体偏移一个 `offset`，而"目标数量"仍然正确 —— 评分器只能靠中心误差抓到。

---

### 2.3 `detect.py` —— 一层薄包装，只做两件事

```python
def detect_frame(gray: np.ndarray, transform: Transform, cfg: dict) -> tuple[list[Target], np.ndarray]
```

职责只有两个：

1. 从 `cfg["detect"]["params"]` 构造 `Params`，调 `find_targets(gray, params)`
2. 用 `transform.to_full()` 把每个目标的 `cx/cy` 还原成原图坐标

**不许**在这里做阈值、形态学、过滤 —— 那些都在 `find_targets` 里。

**验收**：`resize=1.0, roi=null` 时，`detect_frame` 的结果与直接调 `find_targets` **完全一致**（逐字段相等）。

---

### 2.4 `smooth.py` —— 只能按"位置"关联，不能按 id

```python
class Smoother:
    def __init__(self, cfg: dict) -> None
    def update(self, targets: list[Target]) -> list[Target]
```

支持 `method: none | median | ema`。

**核心难点**：`find_targets` 的 `id` 是按 `area` 降序编的，**帧与帧之间会变**（目标面积稍微一抖，顺序就换）。所以：

- 不能 `self.buffers[id]`
- 必须做**最近邻关联**：新目标中心到上帧各目标的距离 < 阈值（建议 40px）里取最近的，才认为是同一个目标
- 关联不上（新出现）→ 新开一个 RingBuffer；连续 N 帧（建议 5）没出现 → 释放

用你 D4 的 `RingBuffer`（`src/utils/data_structure/queue.py`）存最近 `window` 帧的中心，输出中位数。

> ⚠️ `queue.py` 里是 `from linked_list import ...` 的平级导入，直接 `import src.utils.data_structure.queue` 会失败。
> 用现成的 `src/utils/data_structure/_loader.py`：
> `RingBuffer = load_local("queue.py", "RingBuffer")`

**验收**：接 §7 的 `tools/score_track.py`，改成走你的 `Smoother`，必须仍然是 **误差降低 ≥40%**。

---

### 2.5 `rank.py` —— 多目标时取前 N

```python
def top_k(targets: list[Target], k: int, key: str = "area") -> list[Target]
```

用你 D4 的 `MinHeap`（`src/utils/data_structure/heap.py`），不用 `sorted()`。

> ⚠️ **堆里不能直接塞 dict**：你的 `MinHeap` 内部用 `<` 比较元素，而 `dict` 之间不支持 `<`，会抛
> `TypeError: '<' not supported between instances of 'dict' and 'dict'`。
> 正确做法是 push `(key_value, index)` 元组：元组先比 `key_value`，相等时比 `index`（都是数字，可比），
> 弹出来之后再按下标去原列表取 `Target`。

**边界**：

| 情况 | 行为 |
|---|---|
| `k >= len(targets)` | 原样返回（别报错，也别截断） |
| `k <= 0` | 返回 `[]` |
| `targets` 为空 | 返回 `[]` |

**验收**：与 `sorted(targets, key=lambda t: t[key], reverse=True)[:k]` 结果一致（跑 100 组随机数据）。

---

### 2.6 `visualize.py` —— 画框 + 状态栏

```python
@dataclass
class Stats:
    fps: float
    n_raw: int
    n_smoothed: int
    t_pre_ms: float
    t_det_ms: float
    t_post_ms: float

def render(frame_bgr: np.ndarray, raw: list[Target], smoothed: list[Target], stats: Stats, cfg: dict) -> np.ndarray
```

要求：

1. **画旋转矩形**，不是 `rectangle` —— 用 `cv2.boxPoints(((cx, cy), (w, h), angle))`
2. 原始检测 **红色** (0,0,255)，平滑后 **绿色** (0,255,0)，线宽 2
3. 中心画一个小实心点（能看出"抖 vs 稳"）
4. 左上角状态栏：`FPS 28.4 | raw 3 | smooth 3 | pre 0.4 det 1.2 post 0.3 ms`
5. **状态栏要加半透明底条** —— 白字压在亮背景上根本看不清（这是必踩的坑）
6. `show_mask: true` 时再开一个窗口 `cv2.imshow("mask", mask)`

**验收**：

- 同一个目标，红框和绿框位置**明显不同**（抖动被压掉），肉眼能看出来
- 状态栏在深色和浅色背景上都读得清
- `save` 指定路径时，输出视频**能播放**且长度等于帧数

---

### 2.7 `main.py` —— 只做编排，不做算法

```python
def main() -> int
```

职责：

1. `argparse`：`--config`（默认 `configs/classic.yaml`）、`--source`（覆盖 yaml）、`--save`、`--no-show`
2. 主循环按固定顺序走：`read → prepare → detect_frame → update → top_k → render → imshow → waitKey`
3. 计时：每个阶段用 `time.perf_counter()` 单独计时，累计求平均
4. FPS：最近 `fps_window` 帧的平均（**不要用"总帧数/总时间"**，启动卡顿会长期拉低读数）
5. 退出条件：`q` / `ESC` / 视频读完 / `Ctrl+C`
6. 退出时打印汇总：

```
处理 150 帧，耗时 5.31s，平均 28.2 FPS
平均每帧：pre 0.41ms | detect 1.17ms | post 0.28ms | render 0.55ms
平均目标数：raw 3.00 | smoothed 3.00
输出视频：data/out/d6.mp4
```

**验收**：一条命令跑完 150 帧不报错，退出时打出上面那张表。

---

## 3. 主循环的固定顺序（别自己改）

```
FrameSource.read()  →  Frame(BGR, index, t)
        │
        ▼
prepare(frame, cfg) →  (gray, Transform)
        │
        ▼
detect_frame(gray, Transform, cfg)  →  (raw_targets, mask)      # 坐标已还原
        │
        ├──► Smoother.update(raw_targets)  →  smoothed_targets
        │
        ├──► top_k(...)  （可选，rank.enable=true 时）
        │
        ▼
render(frame, raw, smoothed, stats, cfg)  →  annotated（画在**原始帧**上）
        │
        ▼
imshow / VideoWriter.write  →  waitKey(1)  →  退出判断
```

**注意**：`render` 接收的是**原始 BGR 帧**，不是 `gray`，也不是 `mask`。去噪和阈值只影响检测，不影响输出画面。

---

## 4. 全局约束

| 约束 | 要求 |
|---|---|
| 依赖 | 只允许 `cv2`、`numpy`、`yaml`、标准库，加上你自己的三个容器 |
| 参数 | 全部来自 yaml，代码里不能出现阈值/尺寸/路径常量 |
| 编码 | UTF-8，中文注释 |
| 异常 | 单帧出错只跳过该帧并打印帧号，不能让整个视频崩 |
| 退出 | `Ctrl+C` 要能干净退出（关窗口、释放 VideoWriter） |

---

## 5. 三处最容易做错的地方

### 5.1 坐标没还原（最隐蔽）

现象：目标数对、面积对、但框整体偏移。
原因：`transform.to_full()` 漏了 `offset` 或忘了除 `scale`。
抓法：用 `resize=0.5` 跑一遍，如果框位置不对但 `resize=1.0` 正常 → 就是这里。

### 5.2 平滑按 id 关联（第二隐蔽）

现象：目标一多，平滑结果开始乱跳、绿框跑到别的目标上。
原因：`id` 是"按面积排序"的序号，帧间不稳定。
抓法：打印每帧的 `id` 和 `area`，看 `id` 对应的 `area` 是否在跳。

### 5.3 时间统计用错口径

现象：FPS 显示 12，但你说"明明感觉很快"。
原因：用"总帧数/总耗时"，把第一帧的 `import` + 模型初始化的时间算进去了。
抓法：前 5 帧单独丢弃不计入 FPS（warmup）。

---

## 6. 实现顺序（重要：先跑通，再拆）

| 步 | 做什么 | 跑通的标准 |
|---|---|---|
| 1 | **单文件 `main.py`**：读视频 → `find_targets` → 画红框 → FPS → 按 q 退出（约 30 行） | 窗口能动，按 q 能退 |
| 2 | 把读视频抽成 `capture.py`，`main.py` 调它 | 行为和第 1 步完全一样 |
| 3 | 抽 `preprocess.py` + `detect.py`（先只支持 `resize=1.0, roi=null`） | 目标数、坐标和第 1 步一致 |
| 4 | 打开 `resize` / `roi`，验证坐标还原 | `resize=0.5` 时框仍然对 |
| 5 | 加 `smooth.py`，画绿框 | `score_track.py` 显示降低 ≥40% |
| 6 | 加 `rank.py`、`logging`、`--save`、README | 一条命令跑完 + 汇总表 + 视频能播 |

**第 1 步不做完，不要碰第 2 步。** 拆模块的动机是"改一处要动三个文件"，不是"文件看起来太少"。

---

## 7. 验收清单（逐条打勾）

- [ ] `python -m src.classic.main --source data/samples/wobble.mp4` 跑完 150 帧无报错
- [ ] 退出时打印汇总表（FPS / 各阶段耗时 / 目标数）
- [ ] 窗口里能同时看到红框（raw）和绿框（smoothed），且绿框更稳
- [ ] `--save data/out/d6.mp4` 生成的视频能播放，帧数 = 150
- [ ] 改 `configs/classic.yaml` 的 `min_area` 为 `27000` → 目标数变 0（不用改代码）
- [ ] `resize: 0.5` → 目标数仍 3，框位置正确
- [ ] `roi` 取半边 → 只检出 ROI 内的目标，位置仍是原图坐标
- [ ] `python tools/score_track.py` 的平滑误差降低 ≥ 40%
- [ ] `top_k` 与 `sorted()[:k]` 结果一致（随机 100 组）
- [ ] `--source 0` 能打开摄像头（没有摄像头就跳过这项）
- [ ] README 里 3 条命令，队友照抄能跑
- [ ] `Ctrl+C` 能干净退出，不留僵尸窗口

---

## 8. 只有 4 小时怎么办（裁剪版）

**必做**（不做等于没做 D6）：1 能跑、2 能看、3 能量、4 能改

| 砍掉 | 理由 |
|---|---|
| `preprocess.py` 的 ROI | resize 就够练坐标还原 |
| `rank.py` | 单目标场景用不上；D4 的堆你已经练过了 |
| `show_mask` 窗口 | 调试用，不影响交付 |
| 单测 | 用 §7 的命令行验收代替 |
| `--save` | 录屏也能交 |

**保留下来的顺序**：`main.py` → `capture.py` → `detect.py`(含 preprocess 的 resize) → `smooth.py` → `visualize.py` → README。

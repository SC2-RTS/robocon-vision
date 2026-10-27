# robocon-vision

ROBOCON 视觉相关代码仓库（SC2-RTS）。

## 环境要求

| 项 | 版本 | 说明 |
|---|---|---|
| Python | 3.12 | 用 `D:\my_tool\python3.12.1`，不要用 anaconda base（3.13） |
| torch | 2.11.0+cu128 | **必须走 cu128 源**，PyPI 上的 torch 在 Windows 是 CPU 版 |
| torchvision | 0.26.0+cu128 | 与 torch 2.11 配对 |
| CUDA | 12.8 | 随 torch wheel 自带，**不需要**另装 CUDA Toolkit |
| GPU | RTX 5060 Laptop 8 GB（算力 12.0） | 训练 `yolo26n` 从 `batch=8` 起步，OOM 就降 `imgsz=512` |

## 安装

```bash
# 1) 先装 GPU 版 torch（顺序不能反，否则依赖会把 torch 换成 CPU 版）
python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.11.0 torchvision==0.26.0

# 2) 再装其余依赖
python -m pip install -r requirements.txt

# 3) 自检，必须看到「全部通过」
python tools/env_check.py
```

> ⚠️ 不要执行 `pip install -U torch`：PyPI 的 torch 在 Windows 上是 CPU 版，会静默覆盖掉 GPU 版。

## 目录结构

```
robocon-vision/
├── README.md
├── .gitignore
├── requirements.txt          # 依赖清单（torch 单独装，见上）
├── configs/                  # yaml 配置，禁止硬编码
├── data/                     # 只放 data.yaml 与样例图，数据集本体 ignore
├── src/
│   ├── classic/              # 项目①：经典 CV 流水线
│   ├── deep/                 # 训练 / 推理封装
│   └── utils/                # letterbox / nms / 可视化 / 计时
├── scripts/                  # 训练、导出、评估的入口脚本
├── tests/                    # pytest 单测
├── tools/
│   └── env_check.py          # 环境自检
└── docs/
    ├── daily/                # 每日总结
    ├── notes/                # 知识笔记（pr-flow.md 等）
    └── summary/              # 阶段性总结
```

## 当前状态

- [x] 仓库初始化、骨架就位
- [x] 环境可跑（torch 2.11.0+cu128，GPU 算子验证通过）
- [ ] 项目① 经典 CV 流水线（D5–D6）
- [ ] YOLOv8 / YOLO26 训练与部署（D9–D13）
- [ ] ROS 2 检测节点（D19）

## 仓库约定

- commit message 用 `feat: / fix: / docs: / chore: / test:` 前缀
- 模型权重、数据集、`runs/`、`.venv/` **不进仓库**
- 版本号用 annotated tag（`git tag -a`），且**合并到 main 之后**再打
- 空目录必须放 `.gitkeep`（或 Python 包的 `__init__.py`），否则 Git 不跟踪

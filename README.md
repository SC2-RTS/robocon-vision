# robocon-vision

ROBOCON 视觉相关代码仓库。

## 说明

仓库已初始化，代码待补充。

## 仓库骨架
```
robocon-vision/
├── README.md
├── .gitignore
├── requirements.txt
├── configs/          # yaml 配置，不硬编码
├── data/             # 只放 data.yaml 和样例图，数据集本体 ignore
├── src/
│   ├── classic/      # 项目①：经典 CV 流水线
│   ├── deep/         # 训练、推理封装
│   └── utils/        # letterbox / nms / 可视化 / 计时
├── scripts/          # 训练、导出、评估的入口脚本
├── tests/            # pytest
├── docs/             # 每日总结、notes、图
└── tools/            # env_check.py 等
```
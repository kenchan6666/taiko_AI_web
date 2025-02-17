TaikoAI/
├── data/
│   ├── raw/                     # 原始数据：原始音频、TJA 文件等
│   ├── processed/               # 预处理后生成的数据（例如音频特征、标签）
│   └── external/                # 外部数据集或引用的公共数据（如公开谱面数据）
├── docs/                        # 项目相关文档、设计说明、调研报告等
├── experiments/                 # 实验记录、日志、模型调优的配置文件
├── models/                      # 模型文件和训练后保存的权重
│   └── taiko_transformer.pt     # 示例模型权重文件（训练后生成）
├── notebooks/                   # Jupyter Notebook，用于实验、数据探索、模型验证等
├── scripts/                     # 辅助脚本
│   ├── preprocess/              # 数据预处理相关脚本（例如 TJA 解析、音频特征提取）
│   │   └── preprocess_taiko.py  # 预处理入口脚本
│   ├── train.py                 # 模型训练脚本
│   ├── inference.py             # 模型推理脚本
│   └── utils.py                 # 常用工具函数（如路径处理、日志、编码检测等）
├── src/                         # 项目核心代码
│   ├── __init__.py
│   ├── dataset.py               # 定义数据集类，加载预处理数据
│   ├── model.py                 # 定义 Transformer 模型及相关组件
│   ├── train.py                 # 训练入口（可与 scripts/train.py 合并或作为调用接口）
│   └── inference.py             # 推理入口
├── .gitignore                   # Git 忽略文件（如虚拟环境、日志、临时文件等）
├── requirements.txt             # 项目依赖包列表（例如 torch、librosa、numpy、pandas、transformers 等）
├── README.md                    # 项目说明文档（项目背景、安装、使用说明等）
└── setup.py                     # （可选）安装脚本，方便打包和分发

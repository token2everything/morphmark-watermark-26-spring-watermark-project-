# MorphMark 文本水印复现

对 MorphMark 自适应文本水印算法的独立复现与改进。在 OPT-1.3B + C4 上完成了从算法实现、评测、消融、鲁棒性测试到改进探索的全流程实验。

## 项目结构

```
watermark/
├── src/
│   ├── watermarking/       # 水印算法核心
│   │   ├── kgw.py          # KGW 水印 (LogitsProcessor + 检测)
│   │   ├── morphmark.py    # MorphMark 水印 (三种 φ 函数变体)
│   │   ├── adaptive.py     # 自适应强度函数 φ(PG): exp / linear / log
│   │   ├── hash_split.py   # 绿红词表分割 (基于哈希)
│   │   └── extensions.py   # 改进变体: MorphMarkFloor, EntropyAdaptive
│   ├── detection/          # 检测算法
│   │   ├── zscore.py       # 标准 z-score 检测
│   │   └── ewd.py          # 熵加权检测 (EWD)
│   ├── evaluation/         # 评测工具
│   │   ├── metrics.py      # TPR@FPR, AUC-ROC
│   │   ├── quality.py      # PPL 计算
│   │   └── robustness.py   # 鲁棒性攻击: WordNet 同义替换, 随机删除
│   ├── data/
│   │   └── loader.py       # C4 数据集加载
│   └── utils/
│       ├── hashing.py      # 哈希工具
│       └── config.py       # 配置管理
├── scripts/                # 实验脚本
│   ├── generate.py         # 单次水印生成演示
│   ├── evaluate.py         # 完整评测 (4 方法 × 50 样本)
│   ├── ablation.py         # 参数消融实验 (5 suites, 26 数据点)
│   ├── robustness.py       # 鲁棒性测试
│   ├── validate_ewd.py     # EWD 验证
│   ├── evaluate_improved.py # 改进方法评测
│   ├── analyze_pg.py       # PG 分布分析
│   └── failure_cases.py    # 失败案例分析
├── outputs/                # 实验结果和报告
│   ├── results/            # 4 方法基础评测结果 (JSON)
│   ├── ablation/           # 消融、鲁棒性、改进实验数据
│   └── report/             # LaTeX 实验报告 (main.tex + main.pdf)
├── paper_understanding.md  # 论文精读笔记
├── requirements.txt
└── configs/                # YAML 配置文件
```

## 实现的方法

| 方法 | 描述 |
|------|------|
| **KGW** | Kirchenbauer et al. (ICML 2023) — 固定 logit 偏置 + z-score 检测 |
| **MorphMark_exp** | 自适应水印，指数衰减 $\phi(P_G) = p_0 \cdot e^{k(1-P_G)}$ |
| **MorphMark_linear** | 自适应水印，线性衰减 $\phi(P_G) = p_0 \cdot \max(0, 1 - kP_G)$ |
| **MorphMark_log** | 自适应水印，对数衰减 $\phi(P_G) = p_0 \cdot \log(1 + k(1-P_G))$ |
| **MorphMarkFloor** | 改进：对低水印位置加最小偏置 $\delta_{\min}$ |
| **EntropyAdaptive** | 改进：用局部 entropy 动态调整基础强度 $p_0$ |

## 核心结果

50 样本 C4 评测, OPT-1.3B, $\gamma=0.5$:

| 方法 | TPR@1% | AUC | PPL_w | ΔPPL |
|------|--------|-----|-------|------|
| KGW (δ=2.0) | 0.94 | 0.9946 | 7.52 | +15.1% |
| MorphMark_exp | 0.88 | 0.9874 | 7.03 | +7.6% |
| MorphMark_linear | 0.86 | 0.9910 | 6.89 | +5.5% |
| MorphMark_log | 0.86 | 0.9914 | 6.91 | +5.8% |
| **EntropyAdaptive (α=0.3)** | **0.94** | **0.9936** | **5.59** | **-2.7%** |

## 快速开始

```bash
# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 生成一段水印文本（快速验证）
python scripts/generate.py

# 运行完整评测
python scripts/evaluate.py --methods kgw,morphmark_exp --n_samples 50

# 运行消融实验
python scripts/ablation.py --suite gamma_kgw --n_samples 20

# 编译实验报告
cd outputs/report && tectonic main.tex
```

## 依赖

- Python 3.10+
- PyTorch 2.x + CUDA
- transformers, datasets (HuggingFace)
- nltk (WordNet, 鲁棒性测试用)
- tectonic (LaTeX 编译, 可选)

## 实验报告

完整 LaTeX 报告见 `outputs/report/main.pdf`（ACL 单栏格式，中文）。包含方法推导、全部实验数据、消融分析、失败案例和踩坑记录。

## 许可

仅供学习和研究使用。

# eval 项目说明

## 1. 项目简介

本项目用于完成 comment 情感分析的 baseline 推理与结果评测，主要包含两条 baseline 路线：

- BERT baseline：`nlptown/bert-base-multilingual-uncased-sentiment`
- FLAN-T5 baseline：`google/flan-t5-base`

项目支持两种数据模式：

- `test`：在固定测试集上推理，用于和队友的微调模型做公平对比
- `full`：在完整数据集上推理，用于补充分析

项目统一标签定义为：

- `0 = negative`
- `1 = positive`
- `2 = neutral`

## 2. 目录结构与文件用途

### 2.1 根目录文件

- `README.md`
  当前这份项目说明文档。

- `requirements.txt`
  项目运行所需的 Python 依赖列表，不包含单独从 PyTorch 官方安装的 `torch / torchvision / torchaudio`。

- `run_bert_baseline.py`
  BERT baseline 推理入口脚本。负责：
  - 读取数据集
  - 加载本地 BERT 权重
  - 批量推理
  - 保存预测结果
  - 计算指标并导出图表

- `run_flan_t5_baseline.py`
  FLAN-T5 baseline 推理入口脚本。负责：
  - 读取数据集
  - 加载本地 FLAN-T5 权重
  - 批量生成式分类
  - 保存预测结果
  - 计算指标并导出图表

- `model_downloader.py`
  用于从 Hugging Face 下载并缓存模型权重到本地目录，目前配置为下载 `google/flan-t5-base`。

- `run.sh`
  记录本项目常用命令行的简单脚本文件，可作为命令参考。

- `plan.md`
  项目计划文档，记录思路、任务拆分或阶段性安排。

- `note.txt`
  临时说明或个人记录文件。

- `bert_sa.png`
  与项目相关的图片资源。

### 2.2 `baseline/` 目录

这个目录存放 baseline 推理的核心公共代码。

- `baseline/__init__.py`
  baseline 包初始化文件。

- `baseline/constants.py`
  项目常量配置，包括：
  - 路径常量
  - 默认模型目录
  - 标签定义与显示顺序

- `baseline/data_utils.py`
  数据读取与基础清洗工具，包括：
  - `full/test` 模式切换
  - 输入 CSV 校验
  - 标签合法性检查
  - 输出目录创建

- `baseline/evaluation.py`
  指标计算与结果导出模块，包括：
  - Accuracy / Precision / Recall / F1
  - classification report
  - confusion matrix
  - 错分样本导出

- `baseline/README.md`
  baseline 结构的简要说明。

### 2.3 `baseline/models/` 目录

这个目录封装不同模型的推理逻辑。

- `baseline/models/__init__.py`
  模型封装子包初始化文件。

- `baseline/models/bert_baseline.py`
  BERT baseline 推理逻辑。
  重要说明：
  - 模型原生输出 `1_star ~ 5_star`
  - 当前逻辑是：先取 5 星输出中的 argmax，再映射到三分类
    - `1_star, 2_star -> negative -> 0`
    - `3_star -> neutral -> 2`
    - `4_star, 5_star -> positive -> 1`

- `baseline/models/flan_t5_baseline.py`
  FLAN-T5 baseline 推理逻辑。
  重要说明：
  - 使用 prompt 让模型直接生成 `negative / positive / neutral`
  - 再将生成文本解析为标签 id

### 2.4 `dataset/` 目录

这个目录存放数据集构建与切分相关文件。

- `dataset/Dataset.py`
  构建 `merged_sentiment_clean.csv` 的脚本。负责：
  - 读取多个公开数据集
  - 统一标签空间
  - 基础清洗与去重
  - 导出最终合并数据集

- `dataset/merged_sentiment_clean.csv`
  合并、清洗后的完整情感数据集。

- `dataset/dataset_analyze.txt`
  数据集分析记录文件。

- `dataset/splits/train.csv`
  训练集。

- `dataset/splits/valid.csv`
  验证集。

- `dataset/splits/test.csv`
  测试集，用于最终 baseline 和微调模型对比。

### 2.5 `scripts/` 目录

- `scripts/create_eval_split.py`
  用于从 `merged_sentiment_clean.csv` 中按固定随机种子做分层切分，生成 `train/valid/test`。

### 2.6 `bert/` 与 `T5/` 目录

- `bert/nlptown_bert-base-multilingual-uncased-sentiment/`
  本地 BERT 权重目录。

- `T5/google_flan-t5-base/`
  本地 FLAN-T5 权重目录。

说明：
这两个目录体积很大，建议只保留在本地，不纳入 Git 版本控制。

### 2.7 `outputs/` 目录

这个目录存放推理结果。

- `outputs/bert_baseline/test/`
  BERT 在 test 集上的推理结果。

- `outputs/flan_t5_baseline/test/`
  FLAN-T5 在 test 集上的推理结果。

每次运行通常会生成：

- `predictions.csv`
  每条样本的预测结果
- `metrics.json`
  总体指标
- `per_class_metrics.csv`
  每一类指标
- `classification_report.csv`
  sklearn 分类报告
- `confusion_matrix_raw.png`
  原始计数混淆矩阵
- `confusion_matrix_normalized.png`
  归一化混淆矩阵
- `misclassified_examples.csv`
  错分样本

此外还有：

- `outputs/bert_baseline/bert推理结果分析.md`
  BERT 结果分析文档
- `outputs/flan_t5_baseline/T5推理结果分析.md`
  T5 结果分析文档

## 3. 环境配置命令

本项目建议在 **Windows + conda** 环境中运行，并使用 `pip` 安装依赖。

### 3.1 创建并激活 conda 环境

```powershell
conda create -n comment-sa python=3.10 -y
conda activate comment-sa
```

### 3.2 进入项目目录

```powershell
Set-Location D:\PyCharmProjects\comment_sa
```

### 3.3 升级 pip

```powershell
python -m pip install --upgrade pip
```

### 3.4 安装 PyTorch

推荐直接使用 pip 安装：

```powershell
pip install torch torchvision torchaudio
```

如果希望使用 PyTorch 官方 CUDA wheel，也可以使用：

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

### 3.5 安装项目其余依赖

```powershell
pip install -r requirements.txt
```

### 3.6 验证环境

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"
python -c "import transformers, pandas, sklearn, sentencepiece, seaborn; print(transformers.__version__)"
```

## 4. 数据准备命令

### 4.1 如果需要重新生成合并数据集

```powershell
python dataset\Dataset.py
```

### 4.2 生成固定 train / valid / test 切分

```powershell
python scripts\create_eval_split.py
```

## 5. 推理命令

### 5.1 BERT baseline

#### 在 test 集上运行

```powershell
python run_bert_baseline.py --data-mode test
```

#### 在 full 数据集上运行

```powershell
python run_bert_baseline.py --data-mode full
```

#### 小样本冒烟测试

```powershell
python run_bert_baseline.py --data-mode test --max-samples 1000
```

#### 指定 batch size

```powershell
python run_bert_baseline.py --data-mode test --batch-size 32
```

#### 指定设备

```powershell
python run_bert_baseline.py --data-mode test --device cuda
python run_bert_baseline.py --data-mode test --device cpu
```

### 5.2 FLAN-T5 baseline

#### 在 test 集上运行

```powershell
python run_flan_t5_baseline.py --data-mode test
```

#### 在 full 数据集上运行

```powershell
python run_flan_t5_baseline.py --data-mode full
```

#### 小样本冒烟测试

```powershell
python run_flan_t5_baseline.py --data-mode test --max-samples 200
```

#### 调整 batch size

```powershell
python run_flan_t5_baseline.py --data-mode test --batch-size 16
python run_flan_t5_baseline.py --data-mode test --batch-size 24
```

#### 调整输入长度

```powershell
python run_flan_t5_baseline.py --data-mode test --batch-size 16 --max-input-length 128
```

#### 指定设备

```powershell
python run_flan_t5_baseline.py --data-mode test --device cuda
python run_flan_t5_baseline.py --data-mode test --device cpu
```

## 6. 输出结果怎么看

### 6.1 查看总体指标

```powershell
Get-Content outputs\bert_baseline\test\metrics.json
Get-Content outputs\flan_t5_baseline\test\metrics.json
```

### 6.2 查看分类别指标

```powershell
Get-Content outputs\bert_baseline\test\per_class_metrics.csv
Get-Content outputs\flan_t5_baseline\test\per_class_metrics.csv
```

### 6.3 查看错分样本

```powershell
python -c "import pandas as pd; df=pd.read_csv('outputs/bert_baseline/test/misclassified_examples.csv'); print(df.head().to_string())"
python -c "import pandas as pd; df=pd.read_csv('outputs/flan_t5_baseline/test/misclassified_examples.csv'); print(df.head().to_string())"
```

### 6.4 查看预测分布

```powershell
python -c "import pandas as pd; df=pd.read_csv('outputs/bert_baseline/test/predictions.csv'); print(df['pred_text'].value_counts().to_string())"
python -c "import pandas as pd; df=pd.read_csv('outputs/flan_t5_baseline/test/predictions.csv'); print(df['pred_text'].value_counts().to_string())"
```

## 7. 目前项目中需要特别注意的点

### 7.1 标签定义

项目统一标签必须始终保持：

- `0 = negative`
- `1 = positive`
- `2 = neutral`

不能写成 `3 = neutral`。

### 7.2 BERT 当前映射逻辑

当前 BERT baseline 已修正为：

- 先在 `1_star ~ 5_star` 中取 argmax
- 再映射到三分类：
- 1_star,2_star:negative 3_star:neutral 4_star,5_star:positive

这意味着：

- `raw_prediction_text = 3_star` 时，最终一定对应 `pred_text = neutral`

### 7.3 T5 当前主要问题

FLAN-T5 当前结果显示：

- overall accuracy 不低
- 但 `neutral` 类 recall 非常低
- 模型明显更倾向预测为 `negative` 或 `positive`

因此分析结果时，不要只看 accuracy，更要看：

- `macro_f1`
- `per_class_metrics.csv`
- `confusion_matrix_normalized.png`

## 8. 常见问题

### 8.1 `ModuleNotFoundError: No module named 'baseline'`

如果你运行：

```powershell
python scripts\create_eval_split.py
```

出现这个错误，当前项目里的脚本已经做了兼容处理。正常情况下重新运行即可。

### 8.2 T5 推理太慢

建议优先尝试：

```powershell
python run_flan_t5_baseline.py --data-mode test --batch-size 16
python run_flan_t5_baseline.py --data-mode test --batch-size 16 --max-input-length 128
```

### 8.3 GPU 没被识别

检查：

```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

如果输出是 `False`，说明当前 PyTorch 环境没有正确启用 CUDA。

## 9. 推荐执行顺序

如果你要从头跑一遍项目，推荐顺序如下：

```powershell
conda create -n comment-sa python=3.10 -y
conda activate comment-sa
Set-Location D:\PyCharmProjects\comment_sa
python -m pip install --upgrade pip
pip install torch torchvision torchaudio
pip install -r requirements.txt
python scripts\create_eval_split.py
python run_bert_baseline.py --data-mode test --max-samples 1000
python run_flan_t5_baseline.py --data-mode test --max-samples 200
python run_bert_baseline.py --data-mode test
python run_flan_t5_baseline.py --data-mode test
```

## 10. 本地大模型目录说明

以下目录仅建议保留在本地，不建议提交到 Git：

- `bert/nlptown_bert-base-multilingual-uncased-sentiment/`
- `T5/google_flan-t5-base/`

原因：

- 文件体积大
- 不适合纳入版本控制
- 同组成员可以各自在本地下载

如果项目中存在 `.gitignore`，应将这些目录加入忽略规则。

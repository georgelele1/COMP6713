# Baseline Models Evaluation Report

## 1. 实验目的

本实验使用两种现成预训练模型在相同测试集上进行直接推理（direct inference），比较它们在评论情感三分类任务中的表现。本文采用的三分类标签定义为 `0 = negative`、`1 = positive`、`2 = neutral`。参与比较的两个 baseline 模型分别是 `nlptown/bert-base-multilingual-uncased-sentiment` 和 `google/flan-t5-base`。由于两组结果都在相同测试集上评测，因此本文中的对比是公平的。

---

## 2. 为什么选择 `nlptown/bert-base-multilingual-uncased-sentiment`

### 2.1 与任务形式高度相关

本项目任务是评论情感分析，而 `nlptown/bert-base-multilingual-uncased-sentiment` 本身就是一个已经完成情感相关训练的 BERT 模型。与普通 BERT 相比，它并不是只输出通用语义向量，而是可以直接给出情感倾向预测，因此非常适合作为现成权重 baseline。

### 2.2 支持多语言文本

本项目数据集 `merged_sentiment_clean.csv` 并非纯英文，而是包含英语、法语、德语、意大利语等多语言评论文本。该模型属于 multilingual BERT 路线，能够比纯英文情感模型更好地覆盖本项目的数据分布，因此在数据适配性上更合理。

### 2.3 适合同时用于 baseline 与 downstream adaptation

该模型原始输出形式更接近评分型情感预测，例如 1-star 到 5-star，与本项目的三分类标签空间并不完全一致。这一点恰好带来两个优势：一方面，它足够接近情感分析任务，其输出的情感强度可以直接映射为情感三分类，因此可以直接用于 baseline eval；另一方面，它又没有与本项目标签体系完全重合，因此仍然保留了进一步适配本项目数据集的空间，适合继续用于 finetune。换句话说，这个模型既不是“完全无关任务的普通编码器”，也不是“已经对本项目标签体系过度贴合的现成答案”，因此非常适合作为本项目中 BERT 路线的统一起点。

### 2.4 便于和生成式模型形成互补对比

本项目另一条 baseline 路线选择的是 `google/flan-t5-base`，它属于 prompt-based generative model。将其与 `nlptown/bert-base-multilingual-uncased-sentiment` 对比，可以自然形成“判别式分类模型与生成式分类模型”的比较，也能够对应“现成情感分类头”与“prompt-based label generation”两种不同技术路线。这种对比不仅能比较最终指标，也能比较两类模型在 `neutral` 类、短文本以及多语言文本上的差异表现。

---

## 3. 总体量化结果对比

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 | Test Samples |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `nlptown/bert-base-multilingual-uncased-sentiment` | 0.5078 | 0.4740 | 0.4729 | 0.4730 | 0.5110 | 257,771 |
| `google/flan-t5-base` | 0.5946 | 0.5127 | 0.5033 | 0.4513 | 0.5263 | 257,771 |

### 3.1 表面现象

如果只看 `Accuracy`，`FLAN-T5` 高于 `BERT baseline`，前者为 `0.5946`，后者为 `0.5078`。如果只看 `Weighted F1`，`FLAN-T5` 也略高，分别为 `0.5263` 和 `0.5110`。从这些表面指标来看，FLAN-T5 似乎整体表现更好。

### 3.2 更关键的判断

然而，如果进一步观察 `Macro F1`，情况正好相反：BERT baseline 为 `0.4730`，而 FLAN-T5 为 `0.4513`。`Macro F1` 对每个类别一视同仁，不会因为样本数较多的类别占优势而掩盖问题。由于本项目是三分类任务，且 `neutral` 类相对更难，`Macro F1` 比单纯的 `Accuracy` 更能反映模型是否真正学会了三类情感边界。因此，尽管 `FLAN-T5` 的总体准确率更高，但其三分类平衡能力实际上弱于 `BERT baseline`。

---

## 4. 分类别结果对比

| Model | Negative F1 | Positive F1 | Neutral F1 |
| --- | ---: | ---: | ---: |
| `nlptown/bert-base-multilingual-uncased-sentiment` | 0.5993 | 0.5596 | 0.2599 |
| `google/flan-t5-base` | 0.7063 | 0.6131 | 0.0345 |

### 4.1 Negative 类

`FLAN-T5` 在 negative 类上明显更强，BERT 的 F1 为 `0.5993`，而 FLAN-T5 为 `0.7063`。这说明 `FLAN-T5` 更倾向于将文本判断为带有明确负面倾向的类别。

### 4.2 Positive 类

`FLAN-T5` 在 positive 类上也优于 BERT，二者的 F1 分别为 `0.5596` 和 `0.6131`。因此，在“正负两极情感”的判断上，FLAN-T5 的直接生成式推理比当前 BERT baseline 更激进，也更容易给出明显极性的标签。

### 4.3 Neutral 类

最关键的问题出现在 `neutral` 类。BERT 的 neutral F1 为 `0.2599`，虽然不高，但仍保留了一定识别能力；相比之下，FLAN-T5 的 neutral F1 仅为 `0.0345`。其中 FLAN-T5 的 neutral recall 只有 `0.0182`，几乎等于“基本不预测 neutral”。这也是为什么它的 `Accuracy` 看起来更高，但 `Macro F1` 反而更差。

---

## 5. 预测分布、混淆矩阵与类别偏置分析

单看预测分布只能看出模型“更愿意输出什么标签”，而结合混淆矩阵后，才能进一步判断模型“究竟把哪些真实类别压向了哪些预测类别”，从而更完整地观察两个模型的类别偏置问题。

### 5.1 BERT baseline

BERT baseline 的预测分布为 negative `103,733`、positive `92,934`、neutral `61,104`。从数量上看，这个分布虽然与真实标签分布并不完全一致，但至少三类都有较明显覆盖，说明该模型并没有系统性放弃某一个类别。其混淆矩阵如下：

```text
[[64514 22493 24540]
 [17250 51071 21264]
 [21969 19370 15300]]
```

结合分布与混淆矩阵可以看出，negative 经常被误判为 positive 或 neutral，positive 也大量被误判为 negative 或 neutral，而 neutral 又同时被压向 negative 和 positive 两边。因此，BERT baseline 的主要问题并不是类别塌缩，而是整体类别边界仍然较模糊。它保留了三分类结构，但对弱情绪、边界样本和多语言评论的区分还不够稳定。

### 5.2 FLAN-T5 baseline

FLAN-T5 的预测分布为 negative `155,905`、positive `98,912`、neutral `2,954`。这个分布已经明显表现出类别偏置，说明模型几乎不愿意输出 `neutral`。其混淆矩阵如下：

```text
[[94450 16681   416]
 [30292 57783  1510]
 [31163 24448  1028]]
```

结合分布与混淆矩阵可以看出，true negative 被大量识别为 negative，true positive 也有相当一部分能识别为 positive，但真实 neutral 中只有 `1,028` 条被正确识别为 neutral，大量 neutral 被压向 negative 或 positive。因此，FLAN-T5 的主要问题不是整体分类边界都模糊，而是存在非常明显的“二极化”偏置。它更擅长把样本判断成带有明确极性的情感，但难以保留中性类。

### 5.3 小结

综合来看，BERT baseline 的问题是三类之间普遍混淆，但预测分布相对均衡；FLAN-T5 的问题则是类别偏置高度集中，几乎放弃 `neutral`。这也解释了为什么 FLAN-T5 的 `Accuracy` 更高，却没有带来更好的 `Macro F1`。它在正负极性判断上更强，但在三分类任务中失去了类别平衡性；相比之下，BERT baseline 虽然整体准确率较低，但更符合“完整三分类情感分析”的任务要求。

---

## 6. 错误样本与模型行为分析

### 6.1 BERT baseline 的错误模式

从 [misclassified_examples.csv](/D:/PyCharmProjects/comment_sa/outputs/bert_baseline/test/misclassified_examples.csv) 可观察到，BERT baseline 的错误主要体现在多语言文本语义理解不稳定、短文本缺少上下文、弱情绪表达容易被判为 neutral 或相邻极性，以及一些带有隐含态度、反讽或复杂语境的评论容易被错分。总体来看，它的错误更像是“理解不够精确”，但仍然保留了三分类结构。

### 6.2 FLAN-T5 的错误模式

从 [misclassified_examples.csv](/D:/PyCharmProjects/comment_sa/outputs/flan_t5_baseline/test/misclassified_examples.csv) 可观察到，FLAN-T5 的错误集中在将 neutral 评论误判成 negative、将 neutral 评论误判成 positive，以及对短文本和模糊表达过度极化。此外，FLAN-T5 还存在一定的输出解析问题。当前结果中 `parse_error = 2,598`，最常见的非标准生成结果包括 `sentimental`、`sentiment`、`i`、`a`、`i like it` 和 `i love it`。这表明生成式模型即使被要求只输出 `negative / positive / neutral`，仍然可能生成非标准标签文本，从而增加后处理的不稳定性。不过需要指出的是，FLAN-T5 当前的主要问题并不是 parse error 本身，而是其对 `neutral` 类的系统性回避。

---

## 7. 两个模型的优缺点总结

### 7.1 `nlptown/bert-base-multilingual-uncased-sentiment`

`nlptown/bert-base-multilingual-uncased-sentiment` 的主要优点在于它与情感分析任务高度相关，多语言适配性较好，预测分布相对平衡，并且在三分类任务上保留了 `neutral` 类识别能力，因此更适合作为后续进一步适配本项目数据集的起点。它的不足之处在于直接 eval 时总体指标仍然一般，三类边界依旧较模糊，尤其 `neutral` 类 F1 仍然偏低。

### 7.2 `google/flan-t5-base`

`google/flan-t5-base` 的主要优点在于它在正负两极分类上更强，Accuracy 和 Weighted F1 均高于当前 BERT baseline，同时作为生成式 baseline，也体现了与判别式模型不同的建模范式。它的缺点则更加明显，包括几乎不预测 neutral、Macro F1 低于 BERT baseline、prompt 输出存在解析不稳定性，以及对三分类平衡任务不够理想。

---

**发现**

1. [predictions.csv](/D:/PyCharmProjects/comment_sa/outputs/bert_baseline/test/predictions.csv) 现在已经是新逻辑结果，不再有 `3_star -> 非 neutral` 的映射错误。  
我核对了交叉表：

- `1_star -> negative`
- `2_star -> negative`
- `3_star -> neutral`
- `4_star -> positive`
- `5_star -> positive`

这说明你刚才指出的 BERT 逻辑问题已经修正，并且这批输出文件是修正后重新生成的。

2. 主要性能瓶颈在 `neutral` 类，导致整体指标不高。  
从 [metrics.json](/D:/PyCharmProjects/comment_sa/outputs/bert_baseline/test/metrics.json) 和 [per_class_metrics.csv](/D:/PyCharmProjects/comment_sa/outputs/bert_baseline/test/per_class_metrics.csv) 看：

- Accuracy: `0.5078`
- Macro F1: `0.4730`

分类别表现：
- negative F1: `0.5993`
- positive F1: `0.5596`
- neutral F1: `0.2599`

其中 `neutral recall` 只有 `0.2701`，这是当前最大问题。

3. 错分不是单一偏向，而是三类之间都在互相混淆，但 `neutral` 最严重。  
按混淆矩阵统计：

- true negative -> predicted neutral: `24,540`
- true positive -> predicted neutral: `21,264`
- true neutral -> predicted negative: `21,969`
- true neutral -> predicted positive: `19,370`

这说明模型并不是简单“全都预测成负面”，而是对中性和弱情绪句子边界判断很差。

4. [misclassified_examples.csv](/D:/PyCharmProjects/comment_sa/outputs/bert_baseline/test/misclassified_examples.csv) 显示出明显的多语言和语义边界问题。  
我抽看的错分样本里有：
- 法语、德语、西语、意大利语
- 很多句子是“弱情绪 / 隐含情绪 / 上下文不足”
- 有些文本像叙述、感受描述、主观看法，天然接近中性与正负面边界

这和这个 checkpoint 的背景是匹配的：它本质上是“多语言星级评分模型”，不是专门针对你们这种 `comment sentiment 3-class` 数据训练的。

**结果文件概览**

当前 BERT 输出目录是 [outputs/bert_baseline/test](/D:/PyCharmProjects/comment_sa/outputs/bert_baseline/test) ，包含：

- [predictions.csv](/D:/PyCharmProjects/comment_sa/outputs/bert_baseline/test/predictions.csv)
- [misclassified_examples.csv](/D:/PyCharmProjects/comment_sa/outputs/bert_baseline/test/misclassified_examples.csv)
- [metrics.json](/D:/PyCharmProjects/comment_sa/outputs/bert_baseline/test/metrics.json)
- [per_class_metrics.csv](/D:/PyCharmProjects/comment_sa/outputs/bert_baseline/test/per_class_metrics.csv)
- [classification_report.csv](/D:/PyCharmProjects/comment_sa/outputs/bert_baseline/test/classification_report.csv)
- [confusion_matrix_raw.png](/D:/PyCharmProjects/comment_sa/outputs/bert_baseline/test/confusion_matrix_raw.png)
- [confusion_matrix_normalized.png](/D:/PyCharmProjects/comment_sa/outputs/bert_baseline/test/confusion_matrix_normalized.png)

这批文件时间一致，都是 `2026-04-04 02:26` 左右生成，说明是一套完整同步的结果，不是混着旧文件。

**为什么会这样**

核心不是代码 bug 了，而是 baseline 本身的适配性问题：

- 这个 BERT checkpoint 原生任务是 `1~5 star sentiment`
- 你们任务是 `negative / positive / neutral`
- 你们数据还是 comment 场景，而且混有多语言、短文本、弱情绪文本
- 所以它对 `neutral` 的边界非常不稳定

这其实正好说明了你们后续“同模型微调”的价值：  
微调最可能提升的，不一定是总 accuracy，而是 `neutral` 类和边界样本的识别能力。

**一句话结论**

现在 output 里的 BERT 结果从“代码逻辑”上看是正常的；真正的问题已经变成“模型作为现成 baseline 对你们数据不够贴合”，尤其在 `neutral` 类上表现明显偏弱。

如果你愿意，我下一步可以继续帮你：
1. 直接分析 [misclassified_examples.csv](/D:/PyCharmProjects/comment_sa/outputs/bert_baseline/test/misclassified_examples.csv) 并归纳 error types  
2. 再把 T5 的 output 也按同样方式做一遍对照分析
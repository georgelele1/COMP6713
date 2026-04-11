# Baseline Models Evaluation Report

## 1. Experimental Objective

This experiment evaluates two off-the-shelf pretrained models on the same test set through direct inference and compares their performance on a three-class comment sentiment classification task. The label definition used in this report is `0 = negative`, `1 = positive`, and `2 = neutral`. The two baseline models under comparison are `nlptown/bert-base-multilingual-uncased-sentiment` and `google/flan-t5-base`. Since both result sets were evaluated on the same test split, the comparison in this section is fair.

---

## 2. Why `nlptown/bert-base-multilingual-uncased-sentiment` Was Selected

### 2.1 Strong Relevance to the Task

The task of this project is comment sentiment analysis, and `nlptown/bert-base-multilingual-uncased-sentiment` is already a sentiment-related BERT model rather than a generic encoder. Unlike a standard pretrained BERT model that only produces general semantic representations, this model can directly output sentiment-oriented predictions. For this reason, it is well suited to serve as an off-the-shelf baseline.

### 2.2 Support for Multilingual Text

The project dataset `merged_sentiment_clean.csv` is not purely English. It contains multilingual comments, including English, French, German, and Italian. Since this model follows a multilingual BERT route, it is better aligned with the language distribution of the dataset than an English-only sentiment model, making it a more reasonable choice in terms of data compatibility.

### 2.3 Suitable for Both Baseline Evaluation and Downstream Adaptation

The original output format of this model is closer to rating-style sentiment prediction, such as 1-star to 5-star outputs, and therefore does not fully overlap with the three-class label space used in this project. This characteristic brings two important advantages. First, the model remains highly relevant to sentiment analysis, so its outputs can be mapped directly into a three-class setting and used for baseline evaluation. Second, because it is not perfectly aligned with the target label system, it still leaves room for further adaptation to the project dataset and is therefore also suitable for finetuning. In other words, this model is neither a task-irrelevant generic encoder nor an overly task-specific ready-made answer. It provides a strong and balanced starting point for the BERT line in this project.

### 2.4 Complementary Comparison with a Generative Model

The other baseline route in this project is `google/flan-t5-base`, which is a prompt-based generative model. Comparing it with `nlptown/bert-base-multilingual-uncased-sentiment` naturally creates a contrast between a discriminative classification model and a generative classification model, as well as between a sentiment-specific classification head and prompt-based label generation. This comparison is useful not only for reporting final metrics, but also for analyzing how the two model families differ on the `neutral` class, short texts, and multilingual comments.

---

## 3. Overall Quantitative Results

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 | Test Samples |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `nlptown/bert-base-multilingual-uncased-sentiment` | 0.5078 | 0.4740 | 0.4729 | 0.4730 | 0.5110 | 257,771 |
| `google/flan-t5-base` | 0.5946 | 0.5127 | 0.5033 | 0.4513 | 0.5263 | 257,771 |

### 3.1 Surface-Level Observation

If only `Accuracy` is considered, `FLAN-T5` performs better than the `BERT baseline`, with scores of `0.5946` and `0.5078` respectively. A similar pattern appears in `Weighted F1`, where `FLAN-T5` reaches `0.5263` while the BERT baseline reaches `0.5110`. From these surface-level metrics alone, FLAN-T5 may appear to be the stronger model overall.

### 3.2 More Important Interpretation

However, the situation changes when `Macro F1` is examined. The BERT baseline achieves `0.4730`, while FLAN-T5 reaches only `0.4513`. Since `Macro F1` gives equal importance to each class, it is more informative than raw accuracy for a three-class task, especially when the `neutral` class is more difficult and easier to suppress. Therefore, although FLAN-T5 achieves a higher overall accuracy, its class-balanced performance is actually weaker than that of the BERT baseline.

---

## 4. Per-Class Results

| Model | Negative F1 | Positive F1 | Neutral F1 |
| --- | ---: | ---: | ---: |
| `nlptown/bert-base-multilingual-uncased-sentiment` | 0.5993 | 0.5596 | 0.2599 |
| `google/flan-t5-base` | 0.7063 | 0.6131 | 0.0345 |

### 4.1 Negative Class

`FLAN-T5` is clearly stronger on the negative class. The BERT baseline obtains an F1 score of `0.5993`, while FLAN-T5 reaches `0.7063`. This suggests that FLAN-T5 is more inclined to assign texts to a strongly negative category.

### 4.2 Positive Class

`FLAN-T5` also outperforms BERT on the positive class, with F1 scores of `0.6131` versus `0.5596`. This indicates that, for strongly polarized positive and negative sentiment, FLAN-T5’s direct generative inference is more aggressive and more likely to output clearly polarized labels.

### 4.3 Neutral Class

The most critical issue appears in the `neutral` class. The BERT baseline reaches a neutral F1 of `0.2599`, which is not strong but still preserves some recognition ability. In contrast, FLAN-T5 reaches only `0.0345`. Its neutral recall is just `0.0182`, which is effectively equivalent to almost never predicting `neutral`. This is the main reason why FLAN-T5 achieves higher `Accuracy` yet lower `Macro F1`.

---

## 5. Prediction Distribution, Confusion Matrix, and Class Bias Analysis

Prediction distribution alone shows which labels a model prefers to produce, while the confusion matrix shows which true classes are being pushed into which predicted classes, this enables a more comprehensive observation of the category bias issues of the two models.

### 5.1 BERT Baseline

The prediction distribution of the BERT baseline is negative `103,733`, positive `92,934`, and neutral `61,104`. Although this distribution does not perfectly match the true label distribution, all three classes are still substantially represented, which indicates that the model does not systematically abandon any class. Its confusion matrix is shown below:

```text
[[64514 22493 24540]
 [17250 51071 21264]
 [21969 19370 15300]]
```

Taken together, the prediction distribution and confusion matrix show that negative samples are often misclassified as positive or neutral, positive samples are also frequently pushed toward negative or neutral, and neutral samples are split toward both sides. Therefore, the main weakness of the BERT baseline is not class collapse but rather blurred decision boundaries across all three classes. It preserves the three-class structure, but it is still unstable on weak sentiment, boundary cases, and multilingual comments.

### 5.2 FLAN-T5 Baseline

The prediction distribution of FLAN-T5 is negative `155,905`, positive `98,912`, and neutral `2,954`. This already reveals an obvious class bias, as the model is almost unwilling to output `neutral`. Its confusion matrix is shown below:

```text
[[94450 16681   416]
 [30292 57783  1510]
 [31163 24448  1028]]
```

The combination of the distribution and confusion matrix shows that true negative samples are often correctly mapped to negative, and a considerable portion of true positive samples are also mapped to positive. However, among true neutral samples, only `1,028` are correctly classified as neutral, while a large number are pushed toward either negative or positive. Therefore, the main issue of FLAN-T5 is not general ambiguity across all classes, but a strong polarity bias. It is more effective at making polarized sentiment decisions, but much less effective at preserving the neutral class.

### 5.3 Summary

Overall, the BERT baseline mainly suffers from widespread confusion among all three classes, although its prediction distribution remains relatively balanced. FLAN-T5, by contrast, exhibits a much more concentrated bias and almost abandons the neutral class. This explains why FLAN-T5 achieves a higher `Accuracy` but not a better `Macro F1`. It is stronger at positive-versus-negative polarity decisions, but in a three-class sentiment task it loses class balance. By comparison, although the BERT baseline has lower raw accuracy, it better fits the requirement of full three-class sentiment analysis.

---

## 6. Misclassified Samples and Model Behavior Analysis

### 6.1 Error Patterns of the BERT Baseline

Inspection of [misclassified_examples.csv](/D:/PyCharmProjects/comment_sa/outputs/bert_baseline/test/misclassified_examples.csv) shows that the BERT baseline mainly struggles with unstable semantic understanding in multilingual text, short comments with limited context, weak sentiment expressions that are easily confused with neutral or adjacent polarity, and comments containing implicit attitudes, irony, or more complex discourse context. Overall, its errors are better described as insufficiently precise understanding, while the three-class structure is still preserved.

### 6.2 Error Patterns of FLAN-T5

Inspection of [misclassified_examples.csv](/D:/PyCharmProjects/comment_sa/outputs/flan_t5_baseline/test/misclassified_examples.csv) shows that FLAN-T5’s errors are concentrated in predicting negative for neutral comments, predicting positive for neutral comments, and over-polarizing short or ambiguous expressions. In addition, FLAN-T5 also shows a degree of output parsing instability. In the current results, `parse_error = 2,598`, and the most common non-standard outputs include `sentimental`, `sentiment`, `i`, `a`, `i like it`, and `i love it`. This indicates that even when a generative model is instructed to output only `negative / positive / neutral`, it may still produce non-standard label text, which increases post-processing instability. However, it is important to note that the main problem of FLAN-T5 is not the parse errors themselves, but its systematic avoidance of the `neutral` class.

---

## 7. Summary of Strengths and Weaknesses

### 7.1 `nlptown/bert-base-multilingual-uncased-sentiment`

The main strengths of `nlptown/bert-base-multilingual-uncased-sentiment` are its strong relevance to sentiment analysis, its better multilingual compatibility, its relatively balanced prediction distribution, and its ability to retain at least some recognition of the `neutral` class in a three-class setting. These properties make it a suitable starting point for further adaptation to the project dataset. Its weaknesses are that its direct-evaluation performance is still only moderate, its class boundaries remain blurry, and its `neutral` F1 is still relatively low.

### 7.2 `google/flan-t5-base`

The main strengths of `google/flan-t5-base` are its stronger performance on polarized positive-versus-negative classification, its higher Accuracy and Weighted F1 than the current BERT baseline, and its value as a generative baseline that represents a different modeling paradigm. Its weaknesses are more pronounced: it almost never predicts neutral, its Macro F1 is lower than the BERT baseline, its prompt outputs are not always stable to parse, and it is not ideal for a balanced three-class classification task.

---

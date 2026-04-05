"""BERT baseline 封装。

这里封装的是本地 checkpoint：
nlptown/bert-base-multilingual-uncased-sentiment

这个模型原生输出 1~5 星评分，因此这里会把 5 分类结果重新映射为：
- 1, 2 星 -> negative
- 3 星    -> neutral
- 4, 5 星 -> positive

封装成类之后，外层运行脚本只需要关心：
- 传入文本列表
- 获取预测标签、标签文本、置信度
不用反复处理 tokenizer、device、batch 推理细节。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from baseline.constants import DEFAULT_BERT_MODEL_DIR, LABEL_ID_TO_NAME


STAR_TO_LABEL_ID = {
    1: 0,  # 1-star -> negative
    2: 0,  # 2-star -> negative
    3: 2,  # 3-star -> neutral
    4: 1,  # 4-star -> positive
    5: 1,  # 5-star -> positive
}


@dataclass
class BertPredictionBatch:
    """保存一批文本的预测结果。"""

    pred_label: list[int]
    pred_text: list[str]
    confidence: list[float]
    raw_prediction_text: list[str]


class BertSentimentBaseline:
    """BERT 情感分类 baseline。

    注意：
    这里不是训练代码，而是“直接加载现成权重做推理”的包装器。
    """

    def __init__(
        self,
        model_dir: str | Path = DEFAULT_BERT_MODEL_DIR,
        batch_size: int = 32,
        max_length: int = 256,
        device: str | None = None,
    ) -> None:
        # 记录运行配置，后续推理阶段会直接使用。
        self.model_dir = Path(model_dir)
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = self._resolve_device(device)

        # 从本地目录读取 tokenizer 与模型，避免联网下载。
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_dir,
            local_files_only=True,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_dir,
            local_files_only=True,
        )

        # 将模型放到指定设备，并切换到评估模式。
        self.model.to(self.device)
        self.model.eval()

        # nlptown 这版模型原生是 5 分类输出。
        # 如果权重目录不是预期模型，这里会直接报错提醒。
        if getattr(self.model.config, "num_labels", None) != 5:
            raise ValueError(
                "This baseline expects the nlptown sentiment checkpoint with 5 star outputs."
            )

    @staticmethod
    def _resolve_device(device: str | None) -> torch.device:
        """解析运行设备。

        优先级：
        1. 用户手动指定
        2. 自动检测 CUDA
        3. 否则退回 CPU
        """
        if device:
            return torch.device(device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def predict(self, texts: list[str]) -> BertPredictionBatch:
        """对输入文本列表做批量推理。"""
        # 用列表逐步收集所有 batch 的结果。
        pred_label: list[int] = []
        pred_text: list[str] = []
        confidence: list[float] = []
        raw_prediction_text: list[str] = []

        with torch.inference_mode():
            # 分批处理，避免一次把所有文本送进显存。
            batch_starts = range(0, len(texts), self.batch_size)
            total_batches = (len(texts) + self.batch_size - 1) // self.batch_size
            progress_bar = tqdm(
                batch_starts,
                total=total_batches,
                desc="BERT inference progress",
                unit="batch",
            )
            for start in progress_bar:
                batch_texts = texts[start : start + self.batch_size]

                # tokenizer 负责分词、补齐、截断，并返回张量。
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )

                # 把输入张量统一移动到 GPU 或 CPU。
                encoded = {key: value.to(self.device) for key, value in encoded.items()}

                # 前向推理，得到每个 1~5 星类别的 logits。
                outputs = self.model(**encoded)

                # softmax 后得到每个星级的概率。
                probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()

                # 先在 5 星输出中取 argmax，再将 winning star 映射到三分类。
                # 这比将多个星级概率相加后再做 argmax 更符合
                # “先预测星级，再映射成情感类别”的直觉。
                raw_star_pred = probs.argmax(axis=1) + 1
                raw_star_confidence = probs.max(axis=1)

                for star, score in zip(
                    raw_star_pred.tolist(),
                    raw_star_confidence.tolist(),
                ):
                    prediction_id = STAR_TO_LABEL_ID[int(star)]

                    # 保存三分类 id。
                    pred_label.append(int(prediction_id))

                    # 保存标签文本，如 negative / positive / neutral。
                    pred_text.append(LABEL_ID_TO_NAME[int(prediction_id)])

                    # 保存 winning star 对应的概率作为置信度。
                    confidence.append(float(score))

                    # 保存原始 5 星标签，便于回溯模型原始输出。
                    raw_prediction_text.append(f"{star}_star")

                progress_bar.set_postfix_str(
                    f"processed {min(start + len(batch_texts), len(texts))}/{len(texts)}"
                )

        return BertPredictionBatch(
            pred_label=pred_label,
            pred_text=pred_text,
            confidence=confidence,
            raw_prediction_text=raw_prediction_text,
        )

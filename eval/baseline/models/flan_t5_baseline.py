"""FLAN-T5 baseline 封装。

与 BERT 不同，FLAN-T5 属于生成式模型。
这里的思路不是直接输出分类 logits，而是：
1. 把情感分类任务写成一段 prompt；
2. 让模型生成 negative / positive / neutral 之一；
3. 再把生成文本解析回整数标签。

这也是项目里“判别式模型 vs 生成式模型”对比的一部分。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, T5ForConditionalGeneration

from baseline.constants import DEFAULT_T5_MODEL_DIR, LABEL_NAME_TO_ID


# 统一 prompt 模板。
# 保持所有样本提示词一致，能让实验更可复现，也更方便和队友对比。
PROMPT_TEMPLATE = (
    "Classify the sentiment of the following comment as negative, positive, or neutral.\n\n"
    'Comment: "{text}"\n\n'
    "Answer with only one word: negative, positive, or neutral."
)


@dataclass
class FlanT5PredictionBatch:
    """保存 FLAN-T5 的一批预测结果。"""

    pred_label: list[int]
    pred_text: list[str]
    confidence: list[float | None]
    raw_prediction_text: list[str]
    parse_error: list[bool]


class FlanT5SentimentBaseline:
    """FLAN-T5 文本生成式情感分类 baseline。"""

    def __init__(
        self,
        model_dir: str | Path = DEFAULT_T5_MODEL_DIR,
        batch_size: int = 8,
        max_input_length: int = 256,
        max_new_tokens: int = 4,
        device: str | None = None,
    ) -> None:
        # 记录配置参数。
        self.model_dir = Path(model_dir)
        self.batch_size = batch_size
        self.max_input_length = max_input_length
        self.max_new_tokens = max_new_tokens
        self.device = self._resolve_device(device)

        # 从本地目录加载 tokenizer 与模型。
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_dir,
            local_files_only=True,
        )
        self.model = T5ForConditionalGeneration.from_pretrained(
            self.model_dir,
            local_files_only=True,
        )

        # 切换到目标设备并设为评估模式。
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _resolve_device(device: str | None) -> torch.device:
        """解析实际运行设备。"""
        if device:
            return torch.device(device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def predict(self, texts: list[str]) -> FlanT5PredictionBatch:
        """对输入文本列表做批量生成式分类。"""
        pred_label: list[int] = []
        pred_text: list[str] = []
        confidence: list[float | None] = []
        raw_prediction_text: list[str] = []
        parse_error: list[bool] = []

        with torch.inference_mode():
            # 分批推理，避免 prompt 太多时显存占用过高。
            batch_starts = range(0, len(texts), self.batch_size)
            total_batches = (len(texts) + self.batch_size - 1) // self.batch_size
            progress_bar = tqdm(
                batch_starts,
                total=total_batches,
                desc="FLAN-T5 inference progress",
                unit="batch",
            )
            for start in progress_bar:
                batch_texts = texts[start : start + self.batch_size]

                # 把原始评论包装成统一 prompt。
                prompts = [PROMPT_TEMPLATE.format(text=text) for text in batch_texts]

                # 编码成模型可接收的输入张量。
                encoded = self.tokenizer(
                    prompts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_input_length,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}

                # 使用确定性生成：
                # do_sample=False 表示不采样，便于实验复现。
                generated = self.model.generate(
                    **encoded,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                )

                # 把 token 序列还原成文本。
                decoded = self.tokenizer.batch_decode(
                    generated,
                    skip_special_tokens=True,
                )

                for generated_text in decoded:
                    # 将生成文本解析为项目统一的三分类标签。
                    label_name, had_parse_error = self._parse_label(generated_text)
                    pred_label.append(LABEL_NAME_TO_ID[label_name])
                    pred_text.append(label_name)

                    # 生成式输出这里暂不计算概率分数，因此用 None 占位。
                    confidence.append(None)

                    # 保留原始生成文本，便于分析模型到底说了什么。
                    raw_prediction_text.append(generated_text)

                    # 记录是否发生了解析兜底。
                    parse_error.append(had_parse_error)

                progress_bar.set_postfix_str(
                    f"processed {min(start + len(batch_texts), len(texts))}/{len(texts)}"
                )

        return FlanT5PredictionBatch(
            pred_label=pred_label,
            pred_text=pred_text,
            confidence=confidence,
            raw_prediction_text=raw_prediction_text,
            parse_error=parse_error,
        )

    @staticmethod
    def _parse_label(generated_text: str) -> tuple[str, bool]:
        """把生成文本解析成 negative / positive / neutral。"""
        # 先做最基础的规范化，去掉标点和大小写差异。
        normalized = re.sub(r"[^a-z]+", " ", generated_text.lower()).strip()
        tokens = normalized.split()

        # 优先精确匹配 token。
        for token in tokens:
            if token in LABEL_NAME_TO_ID:
                return token, False

        # 如果模型输出了一段更长的句子，再做包含关系匹配。
        if "negative" in normalized:
            return "negative", False
        if "positive" in normalized:
            return "positive", False
        if "neutral" in normalized:
            return "neutral", False

        # 如果完全没法解析，就固定回退到 neutral，
        # 同时把 parse_error 标为 True，便于后续排查。
        return "neutral", True

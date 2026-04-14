"""BERT baseline wrapper.

This module wraps the local checkpoint:
nlptown/bert-base-multilingual-uncased-sentiment

The model natively outputs a 1-to-5 star rating. The implementation here
maps the five-class output back into the project label space:
- 1, 2 stars -> negative
- 3 stars    -> neutral
- 4, 5 stars -> positive

After wrapping this logic in a class, the outer scripts only need to:
- pass in a list of texts
- receive predicted labels, label names, and confidence scores
without repeating tokenizer, device, and batched inference details.
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
    """Store prediction results for a batch of texts."""

    pred_label: list[int]
    pred_text: list[str]
    confidence: list[float]
    raw_prediction_text: list[str]


class BertSentimentBaseline:
    """BERT sentiment classification baseline.

    Note:
    This is not training code. It is a wrapper for directly loading
    existing weights and running inference.
    """

    def __init__(
        self,
        model_dir: str | Path = DEFAULT_BERT_MODEL_DIR,
        batch_size: int = 32,
        max_length: int = 256,
        device: str | None = None,
    ) -> None:
        # Store runtime configuration for later inference calls.
        self.model_dir = Path(model_dir)
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = self._resolve_device(device)

        # Load the tokenizer and model from the local directory to avoid
        # downloading anything from the internet.
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_dir,
            local_files_only=True,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_dir,
            local_files_only=True,
        )

        # Move the model to the target device and switch to evaluation mode.
        self.model.to(self.device)
        self.model.eval()

        # This nlptown checkpoint should always have 5 output labels.
        # If another checkpoint is passed in by mistake, fail early.
        if getattr(self.model.config, "num_labels", None) != 5:
            raise ValueError(
                "This baseline expects the nlptown sentiment checkpoint with 5 star outputs."
            )

    @staticmethod
    def _resolve_device(device: str | None) -> torch.device:
        """Resolve the runtime device.

        Priority:
        1. Manually specified by the user
        2. Automatically detected CUDA
        3. CPU fallback
        """
        if device:
            return torch.device(device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def predict(self, texts: list[str]) -> BertPredictionBatch:
        """Run batched inference over a list of input texts."""
        # Collect prediction outputs batch by batch.
        pred_label: list[int] = []
        pred_text: list[str] = []
        confidence: list[float] = []
        raw_prediction_text: list[str] = []

        with torch.inference_mode():
            # Process texts in batches to avoid loading everything into memory at once.
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

                # The tokenizer handles tokenization, padding, truncation,
                # and returns tensors.
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )

                # Move all input tensors onto the selected GPU or CPU.
                encoded = {key: value.to(self.device) for key, value in encoded.items()}

                # Forward pass to obtain logits for the 1~5 star classes.
                outputs = self.model(**encoded)

                # Apply softmax to convert logits into star probabilities.
                probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()

                # First take the argmax over the 5-star output, then map the
                # winning star back to the 3-class project label.
                # This is more intuitive than summing multiple star probabilities
                # and then taking an argmax over the merged classes.
                raw_star_pred = probs.argmax(axis=1) + 1
                raw_star_confidence = probs.max(axis=1)

                for star, score in zip(
                    raw_star_pred.tolist(),
                    raw_star_confidence.tolist(),
                ):
                    prediction_id = STAR_TO_LABEL_ID[int(star)]

                    # Save the three-class label id.
                    pred_label.append(int(prediction_id))

                    # Save the label name such as negative / positive / neutral.
                    pred_text.append(LABEL_ID_TO_NAME[int(prediction_id)])

                    # Use the winning-star probability as the confidence score.
                    confidence.append(float(score))

                    # Preserve the original 5-star prediction for later analysis.
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

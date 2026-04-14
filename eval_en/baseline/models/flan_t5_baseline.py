"""FLAN-T5 baseline wrapper.

Unlike BERT, FLAN-T5 is a generative model.
The workflow here is not to output classification logits directly, but to:
1. express the sentiment task as a prompt,
2. ask the model to generate one of negative / positive / neutral,
3. parse the generated text back into an integer label.

This is also part of the project's comparison between discriminative and
generative modeling approaches.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, T5ForConditionalGeneration

from baseline.constants import DEFAULT_T5_MODEL_DIR, LABEL_NAME_TO_ID


# Shared prompt template.
# Keeping the same prompt for every sample makes the experiment more reproducible
# and easier to compare with other runs.
PROMPT_TEMPLATE = (
    "Classify the sentiment of the following comment as negative, positive, or neutral.\n\n"
    'Comment: "{text}"\n\n'
    "Answer with only one word: negative, positive, or neutral."
)


@dataclass
class FlanT5PredictionBatch:
    """Store a batch of FLAN-T5 prediction results."""

    pred_label: list[int]
    pred_text: list[str]
    confidence: list[float | None]
    raw_prediction_text: list[str]
    parse_error: list[bool]


class FlanT5SentimentBaseline:
    """FLAN-T5 generative sentiment classification baseline."""

    def __init__(
        self,
        model_dir: str | Path = DEFAULT_T5_MODEL_DIR,
        batch_size: int = 8,
        max_input_length: int = 256,
        max_new_tokens: int = 4,
        device: str | None = None,
    ) -> None:
        # Store configuration values.
        self.model_dir = Path(model_dir)
        self.batch_size = batch_size
        self.max_input_length = max_input_length
        self.max_new_tokens = max_new_tokens
        self.device = self._resolve_device(device)

        # Load the tokenizer and model from the local directory.
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_dir,
            local_files_only=True,
        )
        self.model = T5ForConditionalGeneration.from_pretrained(
            self.model_dir,
            local_files_only=True,
        )

        # Move to the selected device and switch to evaluation mode.
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _resolve_device(device: str | None) -> torch.device:
        """Resolve the actual runtime device."""
        if device:
            return torch.device(device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def predict(self, texts: list[str]) -> FlanT5PredictionBatch:
        """Run batched generative classification over a list of texts."""
        pred_label: list[int] = []
        pred_text: list[str] = []
        confidence: list[float | None] = []
        raw_prediction_text: list[str] = []
        parse_error: list[bool] = []

        with torch.inference_mode():
            # Process prompts in batches to avoid excessive GPU memory usage.
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

                # Wrap the raw comments into the shared prompt format.
                prompts = [PROMPT_TEMPLATE.format(text=text) for text in batch_texts]

                # Tokenize the prompts into model-ready tensors.
                encoded = self.tokenizer(
                    prompts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_input_length,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}

                # Use deterministic generation.
                # do_sample=False disables sampling and improves reproducibility.
                generated = self.model.generate(
                    **encoded,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                )

                # Convert generated token ids back into text.
                decoded = self.tokenizer.batch_decode(
                    generated,
                    skip_special_tokens=True,
                )

                for generated_text in decoded:
                    # Parse the generated string into the project's three-class label.
                    label_name, had_parse_error = self._parse_label(generated_text)
                    pred_label.append(LABEL_NAME_TO_ID[label_name])
                    pred_text.append(label_name)

                    # This baseline does not compute token-level confidence scores,
                    # so None is used as a placeholder.
                    confidence.append(None)

                    # Preserve the raw generated output for later analysis.
                    raw_prediction_text.append(generated_text)

                    # Record whether fallback parsing was required.
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
        """Parse generated text into negative / positive / neutral."""
        # Apply simple normalization first by removing punctuation and
        # lowercasing the text.
        normalized = re.sub(r"[^a-z]+", " ", generated_text.lower()).strip()
        tokens = normalized.split()

        # Prefer exact token matches first.
        for token in tokens:
            if token in LABEL_NAME_TO_ID:
                return token, False

        # If the model generated a longer sentence, fall back to substring matching.
        if "negative" in normalized:
            return "negative", False
        if "positive" in normalized:
            return "positive", False
        if "neutral" in normalized:
            return "neutral", False

        # If parsing completely fails, fall back to neutral and mark
        # parse_error=True so that the sample can be inspected later.
        return "neutral", True

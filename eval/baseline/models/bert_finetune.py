from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import os
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from baseline.constants import LABEL_ID_TO_NAME


@dataclass
class BertPredictionBatch:
    pred_label: list[int]
    pred_text: list[str]
    confidence: list[float]
    raw_prediction_text: list[str]


def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SentimentTextDataset(Dataset):
    """Simple torch dataset for sentiment CSV rows."""

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.df.iloc[index]
        return {
            "id": int(row["id"]),
            "text": str(row["text"]),
            "label": int(row["label"]),
        }


class BertFineTuner:
    """Fine-tunable BERT 3-class sentiment classifier."""

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        batch_size: int = 16,
        max_length: int = 256,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        device: str | None = None,
        seed: int = 42,
        dropout: float = 0.1,
        scheduler_type: str = "linear",
        warmup_ratio: float = 0.1,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.seed = seed
        self.device = self._resolve_device(device)
        self.dropout = dropout
        self.scheduler_type = scheduler_type
        self.warmup_ratio = warmup_ratio

        set_global_seed(self.seed)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        config = AutoConfig.from_pretrained(model_name)
        config.num_labels = 3
        config.id2label = {0: "negative", 1: "positive", 2: "neutral"}
        config.label2id = {"negative": 0, "positive": 1, "neutral": 2}
        config.hidden_dropout_prob = self.dropout
        config.attention_probs_dropout_prob = self.dropout

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            config=config,
        )
        self.model.to(self.device)

        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

    @staticmethod
    def _resolve_device(device: str | None) -> torch.device:
        if device:
            return torch.device(device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _tokenize_batch(self, texts: list[str]) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
        )
        return {key: value.to(self.device) for key, value in encoded.items()}

    def _build_loader(
        self,
        df: pd.DataFrame,
        shuffle: bool,
    ) -> torch.utils.data.DataLoader:
        dataset = SentimentTextDataset(df)

        def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
            texts = [item["text"] for item in batch]
            labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
            ids = [item["id"] for item in batch]

            encoded = self.tokenizer(
                texts,
                truncation=True,
                max_length=self.max_length,
                padding=True,
                return_tensors="pt",
            )

            encoded["labels"] = labels
            encoded["ids"] = ids
            return encoded

        generator = torch.Generator()
        generator.manual_seed(self.seed)

        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            generator=generator,
        )

    def train(
        self,
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame,
        epochs: int = 3,
        output_dir: str | Path = "outputs/bert_finetune/train",
        early_stopping_patience: int = 2,
        grad_clip: float = 1.0,
    ) -> dict[str, Any]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        train_loader = self._build_loader(train_df, shuffle=True)
        valid_loader = self._build_loader(valid_df, shuffle=False)

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        total_steps = len(train_loader) * epochs
        num_warmup_steps = int(self.warmup_ratio * total_steps)

        if self.scheduler_type == "cosine":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=total_steps,
            )
        elif self.scheduler_type == "linear":
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=total_steps,
            )
        else:
            raise ValueError(f"Unsupported scheduler_type: {self.scheduler_type}")

        history: list[dict[str, float]] = []
        best_valid_f1 = -1.0
        best_epoch = -1
        patience_counter = 0
        best_checkpoint_dir = output_dir / "best_checkpoint"

        for epoch in range(1, epochs + 1):
            train_metrics = self._train_one_epoch(
                train_loader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                grad_clip=grad_clip,
                epoch=epoch,
                num_epochs=epochs,
            )

            valid_metrics = self._evaluate_loader(
                data_loader=valid_loader,
                desc=f"Validation epoch {epoch}/{epochs}",
            )
            valid_macro_f1 = float(valid_metrics["macro_f1"])
            current_lr = float(optimizer.param_groups[0]["lr"])

            history.append(
                {
                    "epoch": float(epoch),
                    "learning_rate": current_lr,
                    "train_loss": float(train_metrics["loss"]),
                    "train_accuracy": float(train_metrics["accuracy"]),
                    "train_macro_precision": float(train_metrics["macro_precision"]),
                    "train_macro_recall": float(train_metrics["macro_recall"]),
                    "train_macro_f1": float(train_metrics["macro_f1"]),
                    "valid_loss": float(valid_metrics["loss"]),
                    "valid_accuracy": float(valid_metrics["accuracy"]),
                    "valid_macro_precision": float(valid_metrics["macro_precision"]),
                    "valid_macro_recall": float(valid_metrics["macro_recall"]),
                    "valid_macro_f1": float(valid_metrics["macro_f1"]),
                }
            )

            print(
                f"[Epoch {epoch}/{epochs}] "
                f"train_loss={train_metrics['loss']:.4f} "
                f"train_acc={train_metrics['accuracy']:.4f} "
                f"train_f1={train_metrics['macro_f1']:.4f} "
                f"valid_loss={valid_metrics['loss']:.4f} "
                f"valid_acc={valid_metrics['accuracy']:.4f} "
                f"valid_f1={valid_metrics['macro_f1']:.4f} "
                f"lr={current_lr:.8f}"
            )

            history_df = pd.DataFrame(history)
            history_path = output_dir / "training_history.csv"
            history_df.to_csv(history_path, index=False, encoding="utf-8-sig")

            if valid_macro_f1 > best_valid_f1:
                best_valid_f1 = valid_macro_f1
                best_epoch = epoch
                patience_counter = 0
                self.save(best_checkpoint_dir)
                print(f"Saved new best checkpoint to: {best_checkpoint_dir}")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print("Early stopping triggered.")
                    break

        history_df = pd.DataFrame(history)
        history_path = output_dir / "training_history.csv"
        history_df.to_csv(history_path, index=False, encoding="utf-8-sig")

        return {
            "best_epoch": best_epoch,
            "best_valid_macro_f1": best_valid_f1,
            "history_path": str(history_path),
            "best_checkpoint_dir": str(best_checkpoint_dir),
        }

    def _train_one_epoch(
        self,
        train_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        grad_clip: float,
        epoch: int,
        num_epochs: int,
    ) -> dict[str, float]:
        self.model.train()

        running_loss = 0.0
        all_preds: list[int] = []
        all_labels: list[int] = []

        progress_bar = tqdm(
            train_loader,
            total=len(train_loader),
            desc=f"Training epoch {epoch}/{num_epochs}",
            unit="batch",
        )

        for batch in progress_bar:
            labels = batch.pop("labels").to(self.device)
            batch.pop("ids", None)
            batch = {key: value.to(self.device) for key, value in batch.items()}

            optimizer.zero_grad()

            outputs = self.model(**batch, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)

            optimizer.step()
            scheduler.step()

            preds = torch.argmax(logits, dim=-1)

            running_loss += float(loss.item())
            all_preds.extend(preds.detach().cpu().tolist())
            all_labels.extend(labels.detach().cpu().tolist())

            avg_loss = running_loss / max(1, (progress_bar.n + 1))
            progress_bar.set_postfix_str(f"loss={avg_loss:.4f}")

        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)

        accuracy = float((y_true == y_pred).mean()) if len(y_true) > 0 else 0.0
        macro_precision, macro_recall, macro_f1 = self._macro_prf(
            y_true=y_true,
            y_pred=y_pred,
            labels=[0, 1, 2],
        )

        return {
            "loss": running_loss / max(1, len(train_loader)),
            "accuracy": accuracy,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
        }

    def _evaluate_loader(
        self,
        data_loader: torch.utils.data.DataLoader,
        desc: str = "Evaluating",
    ) -> dict[str, float]:
        self.model.eval()

        running_loss = 0.0
        all_preds: list[int] = []
        all_labels: list[int] = []

        progress_bar = tqdm(
            data_loader,
            total=len(data_loader),
            desc=desc,
            unit="batch",
        )

        with torch.inference_mode():
            for batch in progress_bar:
                labels = batch.pop("labels").to(self.device)
                batch.pop("ids", None)
                batch = {key: value.to(self.device) for key, value in batch.items()}

                outputs = self.model(**batch, labels=labels)
                loss = outputs.loss
                logits = outputs.logits

                preds = torch.argmax(logits, dim=-1)

                running_loss += float(loss.item())
                all_preds.extend(preds.detach().cpu().tolist())
                all_labels.extend(labels.detach().cpu().tolist())

                avg_loss = running_loss / max(1, (progress_bar.n + 1))
                progress_bar.set_postfix_str(f"loss={avg_loss:.4f}")

        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)

        accuracy = float((y_true == y_pred).mean()) if len(y_true) > 0 else 0.0
        macro_precision, macro_recall, macro_f1 = self._macro_prf(
            y_true=y_true,
            y_pred=y_pred,
            labels=[0, 1, 2],
        )

        return {
            "loss": running_loss / max(1, len(data_loader)),
            "accuracy": accuracy,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
        }

    def evaluate(self, df: pd.DataFrame) -> dict[str, float]:
        data_loader = self._build_loader(df, shuffle=False)
        return self._evaluate_loader(data_loader, desc="Evaluating")

    @staticmethod
    def _macro_prf(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: list[int],
    ) -> tuple[float, float, float]:
        precisions = []
        recalls = []
        f1s = []

        for label in labels:
            tp = int(((y_true == label) & (y_pred == label)).sum())
            fp = int(((y_true != label) & (y_pred == label)).sum())
            fn = int(((y_true == label) & (y_pred != label)).sum())

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )

            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)

        return (
            float(np.mean(precisions)),
            float(np.mean(recalls)),
            float(np.mean(f1s)),
        )

    def predict(self, texts: list[str]) -> BertPredictionBatch:
        self.model.eval()

        pred_label: list[int] = []
        pred_text: list[str] = []
        confidence: list[float] = []
        raw_prediction_text: list[str] = []

        with torch.inference_mode():
            batch_starts = range(0, len(texts), self.batch_size)
            total_batches = (len(texts) + self.batch_size - 1) // self.batch_size

            progress_bar = tqdm(
                batch_starts,
                total=total_batches,
                desc="BERT fine-tuned inference progress",
                unit="batch",
            )

            for start in progress_bar:
                batch_texts = texts[start : start + self.batch_size]
                encoded = self._tokenize_batch(batch_texts)

                outputs = self.model(**encoded)
                probs = torch.softmax(outputs.logits, dim=-1).cpu()
                labels = probs.argmax(dim=-1).tolist()
                scores = probs.max(dim=-1).values.tolist()

                for label_id, score in zip(labels, scores):
                    pred_label.append(int(label_id))
                    pred_text.append(LABEL_ID_TO_NAME[int(label_id)])
                    confidence.append(float(score))
                    raw_prediction_text.append(f"logit_class_{int(label_id)}")

                progress_bar.set_postfix_str(
                    f"processed {min(start + len(batch_texts), len(texts))}/{len(texts)}"
                )

        return BertPredictionBatch(
            pred_label=pred_label,
            pred_text=pred_text,
            confidence=confidence,
            raw_prediction_text=raw_prediction_text,
        )

    def save(self, save_dir: str | Path) -> None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save_pretrained(save_dir)
        self.model.save_pretrained(save_dir)

    @classmethod
    def load(
        cls,
        model_dir: str | Path,
        batch_size: int = 16,
        max_length: int = 256,
        device: str | None = None,
        seed: int = 42,
    ) -> "BertFineTuner":
        model_dir = Path(model_dir)
        instance = cls(
            model_name=str(model_dir),
            batch_size=batch_size,
            max_length=max_length,
            device=device,
            seed=seed,
        )
        instance.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        instance.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        instance.model.to(instance.device)
        return instance
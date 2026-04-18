"""
Sentiment Analysis Demo

This demo compares:
 1. Baseline (rule-based)
 2. Pre-trained models (BERT, FLAN-T5)
 3. Fine-tuned models


Fine-tuned models are not included due to size limit.
Please download models and place them in:
  models/finetuned_bert/
  models/finetuned_xlmr/
  (see README for details)
"""


from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path



import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from baseline.models.bert_baseline import BertSentimentBaseline
from baseline.models.flan_t5_baseline import FlanT5SentimentBaseline

try:
    import nltk
    from nltk.corpus import opinion_lexicon
except Exception:
    nltk = None
    opinion_lexicon = None



BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
EVAL_EN_DIR = PROJECT_ROOT / "eval_en"

if str(EVAL_EN_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_EN_DIR))


APP_TITLE = "Sentiment Classification Demo"


LABEL_ID_TO_NAME = {
    0: "negative",
    1: "positive",
    2: "neutral",
}

MODELS_DIR = BASE_DIR / "models"

DEFAULT_BERT_DIR = MODELS_DIR / "bert/nlptown_bert-base-multilingual-uncased-sentiment"
DEFAULT_T5_DIR = MODELS_DIR / "T5/google_flan-t5-base"
DEFAULT_FINETUNED_BERT_DIR = MODELS_DIR / "finetuned_bert"
DEFAULT_FINETUNED_XLMR_DIR = MODELS_DIR / "finetuned_xlmr"

EXAMPLES = [
    " Just sang 'Shine' by Newsboys in the car with my kids.",
    "It works fine, but I wouldn't recommend it.",
    "I absolutely love this product, it works perfectly.",
    "Siento que siempre sospecho."
]

MODEL_CACHE = {}


def get_device(device_name: str | None):
    # Select GPU if available, otherwise fallback to CPU
    if device_name is not None:
        return torch.device(device_name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# Check if model folder exists and contains required config
def model_exists(model_dir: str | Path):
    model_dir = Path(model_dir)
    return model_dir.exists() and model_dir.is_dir() and (model_dir / "config.json").exists()


def clean_text(text: str):
    # Basic preprocessing for lexicon-based baseline
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_lexicon():
    if nltk is None or opinion_lexicon is None:
        raise RuntimeError("NLTK opinion lexicon is not available.")

    try:
        nltk.data.find("corpora/opinion_lexicon")
    except LookupError:
        nltk.download("opinion_lexicon", quiet=True)

    pos_words = set(opinion_lexicon.positive())
    neg_words = set(opinion_lexicon.negative())
    return pos_words, neg_words


def predict_baseline(text: str):
    # Predict sentiment using rule-based lexicon method.
    
    # The method counts positive and negative words in the input text.
    # The final label is determined by comparing these counts. 

    if "lexicon" not in MODEL_CACHE:
        MODEL_CACHE["lexicon"] = load_lexicon()

    pos_words, neg_words = MODEL_CACHE["lexicon"]
    tokens = clean_text(text).split()

    pos_count = 0
    neg_count = 0

    for token in tokens:
        if token in pos_words:
            pos_count += 1
        if token in neg_words:
            neg_count += 1

    if pos_count > neg_count:
        pred_id = 1
    elif neg_count > pos_count:
        pred_id = 0
    else:
        pred_id = 2

    return {
        "label": LABEL_ID_TO_NAME[pred_id],
        "confidence": None,
        "raw": f"pos_count={pos_count}, neg_count={neg_count}"
    }


def get_pretrained_bert(model_dir, device_name):
    # Load pretrained BERT model 
    key = f"bert::{model_dir}"
    if key not in MODEL_CACHE:
        MODEL_CACHE[key] = BertSentimentBaseline(
            model_dir=model_dir,
            device=device_name
        )
    return MODEL_CACHE[key]


def get_pretrained_t5(model_dir, device_name):
    
     # Load pretrained FLAN-T5 model (cached for efficiency)
    key = f"t5::{model_dir}"
    if key not in MODEL_CACHE:
        MODEL_CACHE[key] = FlanT5SentimentBaseline(
            model_dir=model_dir,
            device=device_name
        )
    return MODEL_CACHE[key]



 # Load fine-tuned model from local directory (no further training)
def load_finetuned_model(model_key: str, model_dir: str | Path, device: torch.device):
    if model_key in MODEL_CACHE:
        return MODEL_CACHE[model_key]

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)

    model.to(device)
    model.eval()

    MODEL_CACHE[model_key] = (tokenizer, model)
    return tokenizer, model



# Perform inference using fine-tuned transformer model.
#  Uses softmax to obtain predicted label and confidence.
def predict_finetuned(text: str, model_key: str, model_dir: str | Path, device: torch.device) -> dict:
    tokenizer, model = load_finetuned_model(model_key, model_dir, device)

    inputs = tokenizer(
        [text],
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).cpu()

    pred_id = int(torch.argmax(probs, dim=-1).item())
    confidence = float(torch.max(probs).item())

    return {
        "label": LABEL_ID_TO_NAME[pred_id],
        "confidence": confidence,
        "raw": f"class_{pred_id}"
    }


def build_app(args):
    device = get_device(args.device)

    # Run all models on the same input to ensure fair comparison
    def run_models(text):
        text = text.strip()

        if text == "":
            return "Please enter some text.", []

        rows = []

        # Baseline
        try:
            result = predict_baseline(text)
            rows.append(["Baseline", result["label"], result["confidence"], result["raw"]])
        except Exception as e:
            rows.append(["Baseline", "error", None, str(e)])

        # Pre-trained BERT
        if model_exists(args.bert_model_dir):
            try:
                bert_model = get_pretrained_bert(args.bert_model_dir, args.device)
                result = bert_model.predict([text])
                rows.append([
                    "Pre-trained BERT",
                    result.pred_text[0],
                    result.confidence[0],
                    result.raw_prediction_text[0]
                ])
            except Exception as e:
                rows.append(["Pre-trained BERT", "error", None, str(e)])
        else:
            rows.append(["Pre-trained BERT", "missing", None, str(args.bert_model_dir)])

        # Pre-trained FLAN-T5
        if model_exists(args.t5_model_dir):
            try:
                t5_model = get_pretrained_t5(args.t5_model_dir, args.device)
                result = t5_model.predict([text])
                rows.append([
                    "Pre-trained FLAN-T5",
                    result.pred_text[0],
                    result.confidence[0],
                    result.raw_prediction_text[0]
                ])
            except Exception as e:
                rows.append(["Pre-trained FLAN-T5", "error", None, str(e)])
        else:
            rows.append(["Pre-trained FLAN-T5", "missing", None, str(args.t5_model_dir)])

        # Fine-tuned BERT
        if model_exists(args.finetuned_bert_dir):
            try:
                result = predict_finetuned(
                    text,
                    "finetuned_bert",
                    args.finetuned_bert_dir,
                    device
                )
                rows.append(["Fine-tuned BERT", result["label"], result["confidence"], result["raw"]])
            except Exception as e:
                rows.append(["Fine-tuned BERT", "error", None, str(e)])
        else:
            rows.append(["Fine-tuned BERT", "missing", None, str(args.finetuned_bert_dir)])

        # Fine-tuned XLM-R
        if model_exists(args.finetuned_xlmr_dir):
            try:
                result = predict_finetuned(
                    text,
                    "finetuned_xlmr",
                    args.finetuned_xlmr_dir,
                    device
                )
                rows.append(["Fine-tuned XLM-R", result["label"], result["confidence"], result["raw"]])
            except Exception as e:
                rows.append(["Fine-tuned XLM-R", "error", None, str(e)])
        else:
            rows.append(["Fine-tuned XLM-R", "missing", None, str(args.finetuned_xlmr_dir)])


        # Generate summary for quick comparison of model outputs
        summary_lines = []
        for row in rows:
            line = f"- {row[0]}: {row[1]}"
            if row[2] is not None:
                line += f" (confidence={row[2]:.4f})"
            summary_lines.append(line)

        summary = "\n".join(summary_lines)
        return summary, rows

    with gr.Blocks(title=APP_TITLE) as demo:
        gr.Markdown("# Sentiment Classification Demo")
        gr.Markdown("Compare the baseline, pre-trained models, and fine-tuned models.")

        text_input = gr.Textbox(
            label="Input text",
            lines=5,
            placeholder="Enter a comment here..."
        )

        gr.Examples(examples=EXAMPLES, inputs=text_input)

        run_button = gr.Button("Run Prediction")

        summary_output = gr.Markdown(label="Summary")
        table_output = gr.Dataframe(
            headers=["Model", "Predicted Label", "Confidence", "Raw Output"],
            interactive=False,
            label="Results"
        )

        run_button.click(
            fn=run_models,
            inputs=text_input,
            outputs=[summary_output, table_output]
        )

    return demo


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--device", default=None)

    parser.add_argument("--bert-model-dir", default=str(DEFAULT_BERT_DIR))
    parser.add_argument("--t5-model-dir", default=str(DEFAULT_T5_DIR))
    parser.add_argument("--finetuned-bert-dir", default=str(DEFAULT_FINETUNED_BERT_DIR))
    parser.add_argument("--finetuned-xlmr-dir", default=str(DEFAULT_FINETUNED_XLMR_DIR))

    return parser.parse_args()


def main():
    args = parse_args()
    demo = build_app(args)
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()

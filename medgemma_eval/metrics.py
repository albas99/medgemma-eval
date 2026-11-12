"""Metrics mirroring LO-VLM's metrics.py so scores are directly comparable.

LO-VLM's evaluation:
  - BERTScore  : torchmetrics BERTScore with
                 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext',
                 all_layers=True
  - BLEU       : torchmetrics BLEUScore(n_gram=1) (unigram)
  - ROUGE      : torchmetrics ROUGEScore() (rouge1/2/L/Lsum), per sample
  - parsing    : generated text must contain
                 "diagnosed disease : X, layer information : Y"
                 which is regex-parsed into disease + report; text is
                 lowercased and newlines stripped before scoring.

This module adds exact-match accuracy (normalized full text and parsed
disease label) on top of those.
"""

import argparse
import json
import re
from collections import defaultdict
from statistics import mean, pstdev
from typing import Optional

import torch
from torchmetrics.text import BLEUScore, ROUGEScore
from torchmetrics.text.bert import BERTScore

PUBMED_BERT = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"

DISEASE_RE = re.compile(
    r"diagnosed disease\s*:\s*(?P<disease>.*?)(?=,?\s*layer information)",
    re.IGNORECASE,
)
LAYER_RE = re.compile(
    r"layer information\s*:\s*(?P<report>.*)", re.IGNORECASE | re.DOTALL
)


def normalize(text: str) -> str:
    """LO-VLM-style cleanup: lowercase, strip newlines, collapse whitespace."""
    return re.sub(r"\s+", " ", str(text).replace("\n", " ").strip()).lower()


def parse_generation(text: str) -> tuple[str, str]:
    """Parse 'diagnosed disease : X, layer information : Y' style output.

    Returns (disease, report); falls back to the whole text as the report
    (empty disease) if the format is missing, so no sample is silently dropped.
    """
    disease_match = DISEASE_RE.search(text)
    layer_match = LAYER_RE.search(text)
    disease = disease_match.group("disease").strip(" ,") if disease_match else ""
    report = layer_match.group("report").strip() if layer_match else text.strip()
    return disease, report


class OctMetrics:
    def __init__(self, bert_model: str = PUBMED_BERT, device: str = "cpu"):
        self.bert = BERTScore(model_name_or_path=bert_model, all_layers=True).to(device)
        self.bleu = BLEUScore(n_gram=1)
        self.bleu_per_sample = BLEUScore(n_gram=1)
        self.rouge = ROUGEScore()

    @torch.no_grad()
    def bert_per_sample(self, preds: list[str], refs: list[str]) -> torch.Tensor:
        """(num_layers, num_samples) tensor of BERTScore F1 values."""
        return self.bert(preds=preds, refs=refs)

    def rouge_per_sample(self, preds: list[str], refs: list[str]) -> list[dict]:
        out = []
        for pred, ref in zip(preds, refs):
            scores = self.rouge(preds=pred, refs=ref)
            if isinstance(scores, torch.Tensor):
                scores = scores.detach().cpu()
            out.append(scores)
        return out

    @staticmethod
    def bleu_per_sample_scores(preds: list[str], refs: list[str]) -> list[float]:
        metric = BLEUScore(n_gram=1)
        return [float(metric(preds=[p], refs=[r])) for p, r in zip(preds, refs)]


def evaluate(
    predictions: list[dict],
    report_column: str = "desc",
    bert_model: str = PUBMED_BERT,
    device: str = "cpu",
    limit: Optional[int] = None,
) -> dict:
    """Compute all metrics over a list of prediction dicts.

    Each dict needs the generated text and the ground-truth report.
    """
    if limit:
        predictions = predictions[:limit]

    # Parse + normalize, mirroring LO-VLM's metrics.py.
    records = []
    for p in predictions:
        gen_text = p.get("generated_text") or ""
        ref_text = p.get("ground_truth") or p.get(report_column) or ""
        gen_disease, gen_report = parse_generation(gen_text)
        records.append(
            {
                "index": p.get("index"),
                "img_path": p.get("img_path"),
                "gen_disease": normalize(gen_disease),
                "gen_report": normalize(gen_report),
                "gen_raw": gen_text,
                "ref_disease_raw": str(p.get("label") or ""),
                "ref_report": normalize(ref_text),
            }
        )

    preds = [r["gen_report"] for r in records]
    refs = [r["ref_report"] for r in records]

    metrics = OctMetrics(bert_model=bert_model, device=device)

    # --- BERTScore (all layers, PubMedBERT) ---
    bert_layers = metrics.bert_per_sample(preds=preds, refs=refs).cpu()
    per_sample_bert = bert_layers.mean(dim=0).tolist()  # mean across layers

    # --- BLEU ---
    bleu_corpus = float(metrics.bleu(preds=preds, refs=refs))
    bleu_sample = OctMetrics.bleu_per_sample_scores(preds, refs)

    # --- ROUGE (per sample) ---
    rouge_sample = metrics.rouge_per_sample(preds, refs)
    rouge_keys = sorted(rouge_sample[0].keys()) if rouge_sample else []
    rouge_scores = {k: [] for k in rouge_keys}
    for scores in rouge_sample:
        for k in rouge_keys:
            rouge_scores[k].append(float(scores[k]))

    # --- Exact match ---
    em_full = [float(r["gen_report"] == r["ref_report"]) for r in records]
    em_disease = [float(r["gen_disease"] == normalize(r["ref_disease_raw"])) for r in records]

    def agg(values: list[float]) -> dict:
        return {"mean": mean(values), "std": pstdev(values) if len(values) > 1 else 0.0}

    result = {
        "n_samples": len(records),
        "bert_score_pubmed": {
            "mean": mean(per_sample_bert),
            "std": pstdev(per_sample_bert) if len(per_sample_bert) > 1 else 0.0,
            "note": "F1, averaged across all encoder layers per sample",
        },
        "bleu_1": {
            "corpus": bleu_corpus,
            "per_sample": agg(bleu_sample),
        },
        "rouge": {k: agg(v) for k, v in rouge_scores.items()},
        "exact_match": {
            "full_text": agg(em_full),
            "disease_label": agg(em_disease),
        },
        "per_sample": [
            {
                "index": r["index"],
                "img_path": r["img_path"],
                "bert": bert_layers[:, i].tolist(),  # layer-wise, like LO-VLM per-sample dumps
                "bleu_1": bleu_sample[i],
                "rouge": {k: rouge_scores[k][i] for k in rouge_keys},
                "exact_match": em_full[i],
                "disease_exact_match": em_disease[i],
                "generated": r["gen_raw"],
                "reference": r["ref_report"],
            }
            for i, r in enumerate(records)
        ],
    }
    return result


def load_predictions(path: str) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records:
        raise ValueError(f"No predictions found in {path}")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", help="JSONL produced by run_inference.py")
    parser.add_argument(
        "--report_column",
        default="desc",
        help="Column holding the ground-truth report inside each prediction record",
    )
    parser.add_argument(
        "--bert_model",
        default=PUBMED_BERT,
        help="Encoder for BERTScore (default: PubMedBERT, same as LO-VLM)",
    )
    parser.add_argument("--device", default="cpu", help="cpu or cuda for BERTScore")
    parser.add_argument("--output_json", required=True, help="Where to write metric scores")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    predictions = load_predictions(args.predictions)
    results = evaluate(
        predictions,
        report_column=args.report_column,
        bert_model=args.bert_model,
        device=args.device,
        limit=args.limit,
    )

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    summary_keys = [
        "n_samples",
        "bert_score_pubmed",
        "bleu_1",
        "rouge",
        "exact_match",
    ]
    print(json.dumps({k: results[k] for k in summary_keys}, indent=2))
    print(f"\nFull per-sample scores written to {args.output_json}")


if __name__ == "__main__":
    main()

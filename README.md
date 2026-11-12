# MedGemma Zero-Shot Evaluation Pipeline (OCT Report Generation)

Zero-shot evaluation of `google/medgemma-4b-it` on OCT report generation,
built to be **directly comparable with the LO-VLM evaluation setup**:

| | LO-VLM | This pipeline |
|---|---|---|
| Dataset | `load_from_disk` HF dataset, string columns | Same (see below) |
| Columns | `img_path` + `label` + `desc`, images in separate folder | Same defaults, configurable |
| Metrics | torchmetrics `BERTScore` (PubMedBERT, all layers), `BLEUScore(n_gram=1)`, `ROUGEScore()` | Same, plus exact-match accuracy |
| Output format | `"Diagnosed disease: X, Layer Information: Y"` parsed by regex | MedGemma is prompted to produce the same format |

## Project layout

```
run_inference.py            Stage 1: generate reports (checkpointed, resumable)
run_metrics.py              Stage 2: BERTScore / BLEU / ROUGE / exact match
medgemma_eval/
    data.py                 LO-VLM-style dataset loading (img_path + images dir)
    prompts.py              Zero-shot prompts (system + user, matching LO-VLM format)
    metrics.py              Metric implementation (mirrors LO-VLM metrics.py)
scripts/run_medgemma.sh     Convenience wrapper for both stages
```

## Setup (on your GPU server)

```bash
git clone <your-repo-url> medgemma-eval
cd medgemma-eval

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# optional, only if you use --load_in_4bit:
pip install bitsandbytes
```

**MedGemma is a gated model.** Accept the license once at
https://huggingface.co/google/medgemma-4b-it with your HF account, then on
the server:

```bash
huggingface-cli login        # or: hf auth login
```

## Data

Point the pipeline at your LO-VLM dataset exactly as `inference.py` does:

- `--dataset` — a `save_to_disk` directory (e.g. `lo-vlm/dataset_full/hf_dataset/test`) or an HF hub name (`--split` then applies)
- `--images_dir` — the folder the `img_path` values are relative to (e.g. `lo-vlm/dataset_full/images`)
- Columns default to `img_path` / `label` / `desc`; override with `--image_column`, `--label_column`, `--report_column` if needed.

## Run

```bash
# Stage 1 — generation (greedy decoding by default; interrupt-safe and resumable)
python run_inference.py \
    --dataset <path-to-hf_dataset-test> \
    --images_dir <path-to-images> \
    --output outputs/medgemma4b_test.jsonl \
    --limit 50          # smoke test first; omit for the full set

# Stage 2 — metrics
python run_metrics.py outputs/medgemma4b_test.jsonl \
    --output_json outputs/medgemma4b_metrics.json
```

Or both stages at once:

```bash
bash scripts/run_medgemma.sh <dataset_path> <images_dir>
```

### VRAM

- bf16 full precision: ~9 GB
- `--load_in_4bit`: ~6 GB (requires `bitsandbytes`)

`--max_new_tokens 256` matches LO-VLM's `max_length=256`. Generation is
greedy (deterministic) by default; pass `--temperature 0.7` to sample.

## Output

`run_inference.py` writes JSONL records:

```json
{"index": 0, "img_path": "sample_0001.png", "label": "AMD_late,ERM",
 "ground_truth": "...", "generated_text": "diagnosed disease: ..., layer information: ..."}
```

`run_metrics.py` writes a JSON with a summary block plus `per_sample`
entries containing **layer-wise BERTScore** (`"bert"`), BLEU, ROUGE and
exact-match per sample — the same granularity as LO-VLM's per-sample
metric dumps, so your `m.py` aggregation style works unchanged.

## Notes on comparability

- LO-VLM's BLIP generates text *unconditionally* from the image; MedGemma
  is instructed to follow the `Diagnosed disease: ..., Layer Information: ...`
  format so `metrics.py`'s regex parsing behaves identically. Samples whose
  output lacks the format fall back to using the full text as the report
  (nothing is silently dropped) — watch `exact_match.disease_label` to see
  how well zero-shot MedGemma follows the format.
- BERTScore is averaged across all encoder layers per sample; the full
  layer-wise values are kept in the `per_sample` records, so you can
  re-aggregate (e.g. last layer) to match any LO-VLM variant exactly.

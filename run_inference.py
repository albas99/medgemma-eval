#!/usr/bin/env python3
"""Stage 1: zero-shot MedGemma report generation on the LO-VLM OCT dataset.

Produces a JSONL file with one record per sample:
    {"index", "img_path", "label", "ground_truth", "generated_text"}

The run is checkpointed: it can be interrupted and resumed, skipping
samples already present in the output file.

Example (on your GPU server):
    python run_inference.py \
        --dataset lo-vlm/dataset_full/hf_dataset/test \
        --images_dir lo-vlm/dataset_full/images \
        --output outputs/medgemma4b_test.jsonl
"""

import argparse
import json
import os
import sys
import time

import torch
from tqdm import tqdm

from medgemma_eval.data import DatasetSpec, check_columns, load_dataset_any, load_image
from medgemma_eval.prompts import (
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_USER_PROMPT,
    build_messages,
)

MODEL_ID = "google/medgemma-4b-it"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Data (LO-VLM conventions)
    p.add_argument("--dataset", required=True, help="Path to a save_to_disk dataset (dir, optionally .../split) or an HF hub name")
    p.add_argument("--split", default=None, help="Split name when loading from the HF hub")
    p.add_argument("--images_dir", required=True, help="Folder the img_path values are relative to")
    p.add_argument("--image_column", default="img_path")
    p.add_argument("--label_column", default="label", help="Disease column (None to disable with --label_column '')")
    p.add_argument("--report_column", default="desc", help="Ground-truth report column")
    # Model
    p.add_argument("--model_id", default=MODEL_ID)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--load_in_4bit", action="store_true", help="NF4 4-bit quantization (~9GB VRAM) via bitsandbytes")
    # Generation
    p.add_argument("--max_new_tokens", type=int, default=256, help="Match LO-VLM's max_length=256")
    p.add_argument("--num_beams", type=int, default=1)
    p.add_argument("--temperature", type=float, default=None, help="Set >0 for sampling; default is greedy (deterministic)")
    # Prompting
    p.add_argument("--system_prompt", default=DEFAULT_SYSTEM_PROMPT)
    p.add_argument("--user_prompt", default=DEFAULT_USER_PROMPT)
    # Bookkeeping
    p.add_argument("--output", required=True, help="Output JSONL path (resumable)")
    p.add_argument("--limit", type=int, default=None, help="Evaluate only the first N samples")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_model(args):
    kwargs = {}
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
        kwargs["device_map"] = "auto"
    else:
        kwargs["torch_dtype"] = getattr(torch, args.dtype)
        kwargs["device_map"] = "auto"

    from transformers import AutoModelForImageTextToText, AutoProcessor

    print(f"Loading {args.model_id} ...")
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = AutoModelForImageTextToText.from_pretrained(args.model_id, **kwargs)
    model.eval()
    return model, processor


def already_done(output_path: str) -> set[int]:
    done: set[int] = set()
    if not os.path.exists(output_path):
        return done
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(int(json.loads(line)["index"]))
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
    return done


def main() -> None:
    args = parse_args()
    if args.temperature is not None and args.temperature > 0:
        torch.manual_seed(args.seed)

    spec = DatasetSpec(
        image_column=args.image_column,
        label_column=args.label_column,
        report_column=args.report_column,
        images_dir=args.images_dir,
    )

    dataset = load_dataset_any(args.dataset, args.split)
    check_columns(dataset, spec)
    if args.limit:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
    print(f"Dataset: {len(dataset)} samples, columns: {dataset.column_names}")

    model, processor = load_model(args)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    done = already_done(args.output)
    if done:
        print(f"Resuming: {len(done)} samples already generated.")

    generation_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        do_sample=bool(args.temperature and args.temperature > 0),
    )
    if generation_kwargs["do_sample"]:
        generation_kwargs["temperature"] = args.temperature

    mode = "a" if done else "w"
    with open(args.output, mode, encoding="utf-8") as out:
        for idx, example in enumerate(tqdm(dataset, desc="Generating")):
            if idx in done:
                continue
            try:
                image = load_image(example, spec)
            except Exception as exc:  # keep the run going on a single bad image
                tqdm.write(f"[{idx}] image error: {exc}")
                continue

            messages = build_messages(
                user_prompt=args.user_prompt,
                system_prompt=args.system_prompt,
                image=image,
            )
            inputs = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(model.device)

            t0 = time.time()
            with torch.no_grad():
                outputs = model.generate(**inputs, **generation_kwargs)
            prompt_len = inputs["input_ids"].shape[-1]
            text = processor.decode(outputs[0][prompt_len:], skip_special_tokens=True).strip()

            record = {
                "index": idx,
                "img_path": example.get(spec.image_column),
                "label": example.get(spec.label_column) if spec.label_column else None,
                "ground_truth": example.get(spec.report_column),
                "generated_text": text,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            tqdm.write(f"[{idx}] {time.time() - t0:.1f}s -> {text[:90]}...")

    print(f"Done. Predictions written to {args.output}")
    print("Next step:")
    print(f"  python run_metrics.py {args.output} --output_json outputs/medgemma4b_metrics.json")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted — progress is saved, rerun the same command to resume.", file=sys.stderr)
        sys.exit(130)

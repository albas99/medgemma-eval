#!/usr/bin/env python3
"""Stage 2: score MedGemma predictions with the LO-VLM metric stack.

BERTScore (PubMedBERT, all layers) + BLEU (n_gram=1) + ROUGE + exact match.

Example:
    python run_metrics.py outputs/medgemma4b_test.jsonl \
        --output_json outputs/medgemma4b_metrics.json
"""

from medgemma_eval.metrics import main

if __name__ == "__main__":
    main()

"""MedGemma zero-shot evaluation pipeline for OCT report generation.

Built to be directly comparable with the LO-VLM evaluation setup:
same dataset format (HF save_to_disk with img_path / label / desc columns),
same metrics (torchmetrics BERTScore w/ PubMedBERT, BLEU n_gram=1, ROUGE).
"""

__version__ = "0.1.0"

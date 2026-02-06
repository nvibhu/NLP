# Design Document: Context-Aware Entity Recognition and Sensitivity Masking

## Overview
This project fine-tunes a transformer-based model for Named Entity Recognition (NER) and applies sensitivity masking to detected entities. It supports single-text and batch processing via a Streamlit interface.

## Architecture
- Data: Hugging Face datasets (default: CoNLL-2003). The data loader expects `tokens` and `ner_tags`.
- Model: Transformer encoder (default: BERT base) fine-tuned for token classification.
- Training: Hugging Face Trainer with standard token classification setup. Labels aligned with subword tokens.
- Inference: BIO decoding from token predictions into character spans; merged and masked in output.
- UI: Streamlit app for interactive usage; CSV export for batch results.

## Training Approach
- Tokenization: Fast tokenizer with `is_split_into_words=True` for proper alignment of labels.
- Loss Masking: Non-wordpiece tokens and padding positions are set to label `-100` to be ignored by loss.
- Hyperparameters: LR 5e-5, batch size 16, epochs 3 (modifiable). Early stopping can be added if needed.
- Metrics: Seqeval precision/recall/F1/accuracy on validation and test.

## Evaluation
- We report seqeval metrics per entity type and overall.
- A token-level confusion matrix (excluding the O tag) is generated for qualitative analysis.

## Literature Survey (Selected)
- Early NER: Rule-based and feature-engineered CRFs (Lafferty et al., 2001).
- Neural Approaches: BiLSTM-CRF (Lample et al., 2016) showed strong performance without heavy feature engineering.
- Contextual Embeddings: ELMo (Peters et al., 2018) brought deep contextualized representations.
- Pretrained Transformers: BERT (Devlin et al., 2019) and RoBERTa (Liu et al., 2019) established new SOTA via fine-tuning.
- Domain Adaptation: BioBERT, SciBERT demonstrate domain-specific benefits.
- Relation Extraction: Pipeline vs. joint models; transformers with classification heads, sequence-to-sequence for structured outputs; prompt-based extraction and instruction tuning with LLMs.

## Limitations
- Subword alignment can fragment entities; naive BIO decoding can produce label inconsistencies.
- Out-of-domain PII forms (e.g., custom user IDs, exotic URLs) may be missed if training data lacks such patterns.
- Transformer inference latency and memory usage may limit real-time, high-throughput scenarios.
- Confusion matrix is token-level; span-level errors (boundary mistakes) need additional analysis.

## Improvements
- Add CRF decoding layer for better label consistency across subwords.
- Perform domain adaptation by continued pretraining on in-domain unlabeled data.
- Augment training with synthetic PII patterns (emails, phone, URLs, IDs) mixed with hard negatives.
- Calibrate thresholds and apply post-processing heuristics for specific entity types (e.g., URL/email regex confirmation).
- Distill model to a smaller student (e.g., DistilBERT) for speed; or quantize for deployment.

## LLM-based Approaches
- Zero/Few-shot extraction with instruction-tuned LLMs for cold-start domains.
- Use LLMs to generate synthetic training corpora with diverse PII contexts.
- Hybrid pipeline: LLM proposes candidate entities; NER model verifies/classifies using constrained decoding.
- RAG-enhanced guidance: Provide schema/examples to LLM for better consistency; cache and reuse prompts.

## Data Privacy & Masking
- Strictly avoid logging raw inputs in production.
- Offer configurable masking tokens or type-preserving placeholders (e.g., [EMAIL], [URL]).
- Consider reversible hashing for audit scenarios while keeping plain text hidden.

## Runbook
- Train: `python -m src.ner_masker.train --dataset_name conll2003 --output_dir models/bert-finetuned-ner`
- Evaluate: `python -m src.ner_masker.evaluate --model_dir models/bert-finetuned-ner --output_dir outputs`
- UI: `streamlit run webapp/app.py`
- CLI batch: `python -m src.ner_masker.cli --model_dir models/bert-finetuned-ner --input_dir sample_data --output_dir outputs/masked`

## Screenshots
Add screenshots under `outputs/` and embed here once generated.

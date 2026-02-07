Title: Design and Research: BERT-based Extractive Question Answering

1. Overview
This document outlines the theoretical approach, key design decisions, and research survey for an extractive QA system built on BERT. We fine-tune a pre-trained BERT encoder on SQuAD 1.1 to predict start and end token positions for answer spans in a given context.

2. Problem Formulation
Given a context passage C and a natural language question Q, the model predicts a span (i, j) within C such that the substring C[i:j] answers Q. We cast this as two token classification tasks over the concatenated input: [CLS] Q [SEP] C [SEP], training with cross-entropy on start and end indices.

3. Word Embeddings and Representation Choices
- Static embeddings (e.g., GloVe) encode words context-independently and struggle with polysemy.
- Contextual embeddings (ELMo, BERT) condition representations on sentence-level context, improving QA performance substantially.
- Subword tokenization (WordPiece/BPE) reduces out-of-vocabulary issues and enables fine-grained span alignment.

Impact on performance: Contextual, transformer-based embeddings generally yield higher EM/F1 on QA due to better handling of ambiguity, long-range dependencies, and domain generalization.

4. Model Architecture: BERT and Alternatives
- BERT: Bidirectional encoder using multi-head self-attention; excellent for extractive QA.
- RoBERTa: Robustly optimized BERT with larger pretraining corpus and no NSP objective; often improves QA scores.
- ALBERT: Parameter sharing and factorized embeddings; lighter while competitive in QA.
- DeBERTa(v3): Disentangled attention and enhanced pre-training; strong QA performance.
- Longformer/BigBird: Sparse attention for long contexts; useful when contexts exceed 512 tokens.

5. Training Procedure
- Inputs: [CLS] Q [SEP] C [SEP], with truncation and sliding window (doc_stride) for long contexts.
- Objectives: Start and end position cross-entropy losses.
- Hyperparameters: batch size, learning rate (e.g., 3e-5), weight decay, warmup ratio, epochs. Early stopping or evaluation per epoch helps monitor EM/F1.

6. Evaluation
- Metrics: Exact Match (EM) and token-level F1 on validation set (SQuAD 1.1).
- Post-processing: Use start/end logits to extract spans from the context via offset mappings. Aggregate across overlapping windows and select the highest-scoring span.

7. Enhancements and Research Directions
7.1 Knowledge Integration
- Retrieval-Augmented QA: Retrieve passages from external corpora and feed top-k contexts to the reader (BERT), improving coverage (e.g., RAG).
- Knowledge Graphs: Link entities in context to KG nodes and incorporate KG-aware embeddings or constraints.

7.2 Cross-lingual Adaptation
- Use multilingual encoders (mBERT, XLM-R) for cross-lingual QA, zero-shot transfer, or fine-tune on translated SQuAD variants.

7.3 Robustness Improvements
- Adversarial training and data augmentation (back-translation, paraphrasing).
- Answerability (SQuAD v2): Model a null answer, calibrate thresholds for unanswerable questions.
- Calibration: Temperature scaling for confidence estimates.

7.4 Efficient Inference
- Knowledge distillation (e.g., DistilBERT) for faster serving.
- Quantization and pruning to reduce latency and memory.

8. System Design
- Data: Download via Hugging Face datasets. Preprocess with sliding windows (max_length=384, doc_stride=128).
- Model: Fine-tune BERT; save artifacts (config, tokenizer, weights) to models/qa-bert.
- Serving: Flask API /predict receives JSON {context, question}; uses the fine-tuned model for inference and returns the best span.
- Frontend: Minimal HTML+JS for user input and response rendering.

9. Limitations
- Long contexts (>512 tokens) require windowing which may fragment evidence; consider Longformer/BigBird.
- Domain shift may reduce accuracy; domain-adaptive pretraining (DAPT) can help.
- Extractive QA cannot synthesize answers beyond spans; consider generative models (T5, FLAN-T5) for abstractive QA.

10. References
- Devlin, J. et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.
- Liu, Y. et al. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach.
- Lan, Z. et al. (2019). ALBERT: A Lite BERT for Self-supervised Learning of Language Representations.
- He, P. et al. (2020/2021). DeBERTa: Decoding-enhanced BERT with Disentangled Attention.
- Beltagy, I. et al. (2020). Longformer: The Long-Document Transformer.
- Zaheer, M. et al. (2020). Big Bird: Transformers for Longer Sequences.
- Karpukhin, V. et al. (2020). Dense Passage Retrieval for Open-Domain QA.
- Lewis, P. et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP.
- Conneau, A. et al. (2020). Unsupervised Cross-lingual Representation Learning at Scale (XLM-R).

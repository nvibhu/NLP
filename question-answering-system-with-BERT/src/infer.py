from __future__ import annotations
from typing import Optional, Dict, Any

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering


class QAInference:
    def __init__(
        self,
        model_dir: Optional[str] = None,
        model_name: str = "bert-base-uncased",
        device: Optional[str] = None,
        max_length: int = 384,
        doc_stride: int = 128,
        max_answer_length: int = 30,
    ) -> None:
        self.model_dir = model_dir
        self.model_name = model_name
        self.max_length = max_length
        self.doc_stride = doc_stride
        self.max_answer_length = max_answer_length

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        load_from = model_dir if (model_dir and len(model_dir) > 0) else model_name
        self.tokenizer = AutoTokenizer.from_pretrained(load_from, use_fast=True)
        self.model = AutoModelForQuestionAnswering.from_pretrained(load_from)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, context: str, question: str) -> Dict[str, Any]:
        pad_on_right = self.tokenizer.padding_side == "right"
        enc = self.tokenizer(
            question if pad_on_right else context,
            context if pad_on_right else question,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation="only_second" if pad_on_right else "only_first",
            max_length=self.max_length,
            stride=self.doc_stride,
            padding=True,
        )

        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)
        token_type_ids = enc.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(self.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            start_logits = outputs.start_logits.cpu().numpy()
            end_logits = outputs.end_logits.cpu().numpy()

        offsets = enc["offset_mapping"].numpy()
        sequence_ids_fn = self.tokenizer.create_token_type_ids_from_sequences  # not used directly; kept for clarity

        best_text = ""
        best_score = -1e9

        for i in range(input_ids.shape[0]):
            input_id_row = input_ids[i].tolist()
            cls_index = input_id_row.index(self.tokenizer.cls_token_id)

            # sequence_ids like in fast tokenizers (requires calling on encoded batch)
            seq_ids = enc.sequence_ids(i)
            context_index = 1 if (self.tokenizer.padding_side == "right") else 0

            # Consider top indices
            start_log = start_logits[i]
            end_log = end_logits[i]

            start_indexes = np.argsort(start_log)[-20:][::-1]
            end_indexes = np.argsort(end_log)[-20:][::-1]

            for s in start_indexes:
                for e in end_indexes:
                    if s >= len(offsets[i]) or e >= len(offsets[i]):
                        continue
                    if offsets[i][s] is None or offsets[i][e] is None:
                        continue
                    if seq_ids[s] != context_index or seq_ids[e] != context_index:
                        continue
                    if e < s or (e - s + 1) > self.max_answer_length:
                        continue
                    start_char, end_char = offsets[i][s]
                    text_span = context[start_char:end_char]
                    score = start_log[s] + end_log[e]
                    if score > best_score:
                        best_score = score
                        best_text = text_span

        return {"answer": best_text, "score": float(best_score)}


def predict(model_dir: Optional[str], context: str, question: str) -> str:
    qa = QAInference(model_dir=model_dir)
    return qa.predict(context, question)["answer"]

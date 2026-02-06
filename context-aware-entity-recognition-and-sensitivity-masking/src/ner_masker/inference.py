from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import re

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification


@dataclass
class EntitySpan:
    start: int
    end: int
    label: str


class NERMasker:
    def __init__(self, model_dir: str, use_regex_fallback: bool = True):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForTokenClassification.from_pretrained(model_dir)
        self.model.eval()
        self.id2label = self.model.config.id2label
        self.use_regex_fallback = use_regex_fallback

    @torch.no_grad()
    def predict_entities(self, text: str) -> List[EntitySpan]:
        tokens = self.tokenizer(text, return_offsets_mapping=True, return_tensors="pt", truncation=True)
        outputs = self.model(**{k: v for k, v in tokens.items() if k in ["input_ids", "attention_mask"]})
        logits = outputs.logits.squeeze(0)  # (seq_len, num_labels)
        preds = torch.argmax(logits, dim=-1).tolist()
        offsets = tokens["offset_mapping"].squeeze(0).tolist()

        spans: List[EntitySpan] = []
        current_label = None
        current_start = None
        for idx, (start, end) in enumerate(offsets):
            if start == end:  # special tokens
                continue
            label = self.id2label[preds[idx]]
            if label.startswith("B-"):
                if current_label is not None:
                    spans.append(EntitySpan(current_start, prev_end, current_label))
                current_label = label[2:]
                current_start = start
            elif label.startswith("I-") and current_label == label[2:]:
                pass  # continue span
            else:
                if current_label is not None:
                    spans.append(EntitySpan(current_start, prev_end, current_label))
                current_label = None
                current_start = None
            prev_end = end

        if current_label is not None:
            spans.append(EntitySpan(current_start, prev_end, current_label))

        # Optional regex fallback for common PII types (URLs, emails, simple IDs)
        if self.use_regex_fallback:
            spans.extend(find_regex_spans(text))

        return merge_overlaps(spans)

    def mask_text(self, text: str, mask_token: str = "[MASK]") -> str:
        spans = self.predict_entities(text)
        if not spans:
            return text
        spans = sorted(spans, key=lambda s: s.start)
        out = []
        last = 0
        for s in spans:
            out.append(text[last:s.start])
            out.append(mask_token)
            last = s.end
        out.append(text[last:])
        return "".join(out)


def merge_overlaps(spans: List[EntitySpan]) -> List[EntitySpan]:
    if not spans:
        return spans
    spans = sorted(spans, key=lambda s: s.start)
    merged: List[EntitySpan] = []
    cur = spans[0]
    for s in spans[1:]:
        if s.start <= cur.end:
            cur = EntitySpan(cur.start, max(cur.end, s.end), cur.label if cur.label == s.label else cur.label)
        else:
            merged.append(cur)
            cur = s
    merged.append(cur)
    return merged


def find_regex_spans(text: str) -> List[EntitySpan]:
    spans: List[EntitySpan] = []
    # URL
    for m in re.finditer(r"\bhttps?://[^\s)]+", text):
        spans.append(EntitySpan(m.start(), m.end(), "URL"))
    # Email
    for m in re.finditer(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", text):
        spans.append(EntitySpan(m.start(), m.end(), "EMAIL"))
    # Simple UID pattern (e.g., UID: 8829-X)
    for m in re.finditer(r"\b(?:UID|User\s*ID|ID)[:\s]+([A-Za-z0-9._-]+)\b", text, flags=re.IGNORECASE):
        # Mask only the value part
        val_span = m.span(1)
        spans.append(EntitySpan(val_span[0], val_span[1], "ID"))
    return spans

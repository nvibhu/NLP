from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, DataCollatorForTokenClassification


def set_seed(seed: int = 42):
    random.seed(seed)


def load_ner_dataset(dataset_name: Optional[str] = "conll2003", dataset_path: Optional[str] = None) -> DatasetDict:
    """
    Load a Hugging Face NER dataset. Defaults to 'conll2003'.
    The dataset must provide 'tokens' and 'ner_tags'.
    """
    if dataset_path:
        ds = load_dataset(dataset_path)
    else:
        if dataset_name is None:
            dataset_name = "conll2003"
        try:
            ds = load_dataset(dataset_name)
        except RuntimeError as e:
            msg = str(e)
            # Fall back to hub datasets that don't rely on local scripts
            if "Dataset scripts are no longer supported" in msg or "scripts are no longer supported" in msg:
                alt = "tner/conll2003" if dataset_name == "conll2003" else "wnut_17"
                ds = load_dataset(alt)
            else:
                raise
    # Normalize common alternative column names (e.g., 'tags' -> 'ner_tags')
    for split in ds.keys():
        cols = ds[split].column_names
        if "ner_tags" not in cols and "tags" in cols:
            ds[split] = ds[split].rename_column("tags", "ner_tags")

    # Basic schema check
    for split in ds.keys():
        cols = ds[split].column_names
        if not ("tokens" in cols and "ner_tags" in cols):
            raise ValueError(
                f"Dataset split '{split}' must contain 'tokens' and 'ner_tags' columns. Found: {cols}"
            )
    return ds


def get_label_list(dataset: DatasetDict) -> List[str]:
    features = dataset["train"].features
    names = features["ner_tags"].feature.names
    return list(names)


def build_tokenizer(model_name_or_path: str) -> AutoTokenizer:
    tok = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    return tok


def tokenize_and_align_labels(tokenizer: AutoTokenizer, label_all_tokens: bool = True):
    def _inner(examples: Dict[str, List[List[str]]]):
        tokenized_inputs = tokenizer(
            examples["tokens"], truncation=True, is_split_into_words=True
        )

        labels = []
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(label[word_idx] if label_all_tokens else -100)
                previous_word_idx = word_idx
            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    return _inner


def prepare_splits(
    dataset: DatasetDict,
    tokenizer: AutoTokenizer,
    label_all_tokens: bool = True,
) -> Tuple[DatasetDict, List[str]]:
    label_list = get_label_list(dataset)
    fn = tokenize_and_align_labels(tokenizer, label_all_tokens=label_all_tokens)
    tokenized = dataset.map(fn, batched=True)
    return tokenized, label_list


def build_data_collator(tokenizer: AutoTokenizer) -> DataCollatorForTokenClassification:
    return DataCollatorForTokenClassification(tokenizer=tokenizer)

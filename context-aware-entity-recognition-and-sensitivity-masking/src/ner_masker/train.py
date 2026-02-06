from __future__ import annotations

import argparse
import json
import os
from typing import List

import numpy as np
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
)
import inspect

from .data_prep import (
    load_ner_dataset,
    build_tokenizer,
    prepare_splits,
    build_data_collator,
    set_seed,
)


def compute_metrics_builder(label_list: List[str]):
    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        return {
            "precision": precision_score(true_labels, true_predictions),
            "recall": recall_score(true_labels, true_predictions),
            "f1": f1_score(true_labels, true_predictions),
            "accuracy": accuracy_score(true_labels, true_predictions),
        }

    return compute_metrics


def main():
    parser = argparse.ArgumentParser(description="Fine-tune a transformer for NER and masking")
    parser.add_argument("--dataset_name", type=str, default="conll2003", help="HF dataset name (e.g., conll2003)")
    parser.add_argument("--dataset_path", type=str, default=None, help="Local dataset path or HF script path")
    parser.add_argument("--base_model", type=str, default="bert-base-cased", help="Base model checkpoint")
    parser.add_argument("--output_dir", type=str, default="models/bert-finetuned-ner", help="Where to save the model")
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    dataset = load_ner_dataset(dataset_name=args.dataset_name, dataset_path=args.dataset_path)
    tokenizer = build_tokenizer(args.base_model)
    tokenized, label_list = prepare_splits(dataset, tokenizer)

    id2label = {i: l for i, l in enumerate(label_list)}
    label2id = {l: i for i, l in enumerate(label_list)}

    model = AutoModelForTokenClassification.from_pretrained(
        args.base_model, num_labels=len(label_list), id2label=id2label, label2id=label2id
    )

    data_collator = build_data_collator(tokenizer)

    # Backward-compatible TrainingArguments construction: filter unsupported kwargs
    ta_kwargs = dict(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
    )

    sig = inspect.signature(TrainingArguments.__init__)
    supported = set(sig.parameters.keys())
    # Map evaluation_strategy -> evaluate_during_training for very old HF
    if "evaluation_strategy" not in supported and "evaluate_during_training" in supported:
        val = ta_kwargs.get("evaluation_strategy", "no")
        ta_kwargs["evaluate_during_training"] = val != "no"
        ta_kwargs.pop("evaluation_strategy", None)
    # Drop keys not accepted by this Transformers version
    ta_kwargs = {k: v for k, v in ta_kwargs.items() if k in supported}

    training_args = TrainingArguments(**ta_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized.get("validation", tokenized.get("test")),
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_builder(label_list),
    )

    trainer.train()

    print("Evaluating on validation/test split...")
    metrics = trainer.evaluate()
    print(metrics)

    print(f"Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    with open(os.path.join(args.output_dir, "labels.json"), "w") as f:
        json.dump({"label_list": label_list}, f, indent=2)


if __name__ == "__main__":
    main()

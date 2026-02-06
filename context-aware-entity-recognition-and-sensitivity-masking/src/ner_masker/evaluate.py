from __future__ import annotations

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from seqeval.metrics import classification_report
from sklearn.metrics import confusion_matrix
from transformers import AutoModelForTokenClassification, AutoTokenizer, Trainer

from .data_prep import load_ner_dataset, build_tokenizer, prepare_splits, build_data_collator


def main():
    parser = argparse.ArgumentParser(description="Evaluate NER model and plot confusion matrix")
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default="conll2003")
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    dataset = load_ner_dataset(dataset_name=args.dataset_name, dataset_path=args.dataset_path)
    base_tokenizer = build_tokenizer(args.model_dir)
    tokenized, label_list = prepare_splits(dataset, base_tokenizer)

    model = AutoModelForTokenClassification.from_pretrained(args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    data_collator = build_data_collator(tokenizer)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    preds_output = trainer.predict(tokenized.get("test", tokenized["validation"]))
    preds = np.argmax(preds_output.predictions, axis=2)
    labels = preds_output.label_ids

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(preds, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(preds, labels)
    ]

    print("Classification report (seqeval):")
    print(classification_report(true_labels, true_predictions))

    # Build a token-level confusion matrix for non-O labels
    y_true_flat, y_pred_flat = [], []
    for tl, tp in zip(true_labels, true_predictions):
        for t, p in zip(tl, tp):
            if t != "O":
                y_true_flat.append(t)
                y_pred_flat.append(p)

    unique_labels = sorted(list({*y_true_flat, *y_pred_flat}))
    cm = confusion_matrix(y_true_flat, y_pred_flat, labels=unique_labels)
    plt.figure(figsize=(max(6, len(unique_labels)), max(4, len(unique_labels) * 0.4)))
    sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=unique_labels, yticklabels=unique_labels, fmt="d")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (token-level, excluding O)")
    out_path = os.path.join(args.output_dir, "confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved confusion matrix to {out_path}")


if __name__ == "__main__":
    main()

import argparse
import os
from typing import Dict, Any

import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    set_seed,
)

from datasets import DatasetDict

from .data import load_squad
from .preprocess import prepare_train_features, prepare_validation_features
from .evaluate import postprocess_qa_predictions, compute_squad_metrics


def train(
    model_name: str = "bert-base-uncased",
    output_dir: str = "models/qa-bert",
    epochs: int = 2,
    batch_size: int = 8,
    learning_rate: float = 3e-5,
    seed: int = 42,
    squad_version: str = "1.1",
    max_length: int = 384,
    doc_stride: int = 128,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    train_samples: int | None = None,
    eval_samples: int | None = None,
) -> Dict[str, Any]:
    set_seed(seed)

    ds: DatasetDict = load_squad(version=squad_version)
    # Optionally subsample for quicker experimentation
    if train_samples is not None and train_samples > 0:
        ds["train"] = ds["train"].select(range(min(train_samples, len(ds["train"]))))
    if eval_samples is not None and eval_samples > 0:
        ds["validation"] = ds["validation"].select(range(min(eval_samples, len(ds["validation"]))))
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    tokenized_train = ds["train"].map(
        lambda x: prepare_train_features(x, tokenizer, max_length=max_length, doc_stride=doc_stride),
        batched=True,
        remove_columns=ds["train"].column_names,
        desc="Tokenize train",
    )

    validation_features = ds["validation"].map(
        lambda x: prepare_validation_features(x, tokenizer, max_length=max_length, doc_stride=doc_stride),
        batched=True,
        remove_columns=ds["validation"].column_names,
        desc="Tokenize validation",
    )

    model = AutoModelForQuestionAnswering.from_pretrained(model_name)

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        num_train_epochs=epochs,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        report_to=[],
        fp16=False,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # We'll need access to the raw validation set (examples) to compute metrics properly.
    validation_examples = ds["validation"]

    def compute_metrics_fn(eval_pred):
        start_logits, end_logits = eval_pred.predictions
        predictions = (np.array(start_logits), np.array(end_logits))
        # Post-process to text answers
        pred_texts = postprocess_qa_predictions(
            examples=validation_examples,
            features=validation_features,
            predictions=predictions,
        )
        metrics = compute_squad_metrics(pred_texts, validation_examples)
        return metrics

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_train,
        eval_dataset=validation_features,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
    )

    trainer.train()
    os.makedirs(output_dir, exist_ok=True)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    eval_metrics = trainer.evaluate()
    return {"metrics": eval_metrics, "output_dir": output_dir}


def main():
    parser = argparse.ArgumentParser(description="Fine-tune BERT for Question Answering on SQuAD")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--output_dir", type=str, default="models/qa-bert")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--squad_version", type=str, default="1.1", choices=["1.1", "2.0"])
    parser.add_argument("--max_length", type=int, default=384)
    parser.add_argument("--doc_stride", type=int, default=128)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--train_samples", type=int, default=0, help="Optional limit of training samples for quick runs")
    parser.add_argument("--eval_samples", type=int, default=0, help="Optional limit of eval samples for quick runs")
    args = parser.parse_args()

    result = train(
        model_name=args.model_name,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        squad_version=args.squad_version,
        max_length=args.max_length,
        doc_stride=args.doc_stride,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        train_samples=args.train_samples if args.train_samples > 0 else None,
        eval_samples=args.eval_samples if args.eval_samples > 0 else None,
    )
    print(result)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from typing import List

from .data_prep import load_ner_dataset
from .inference import NERMasker


def main():
    parser = argparse.ArgumentParser(description="Preview a few NER predictions from the dataset")
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--dataset_name", default="conll2003")
    parser.add_argument("--dataset_path", default=None)
    parser.add_argument("--num_examples", type=int, default=3)
    args = parser.parse_args()

    ds = load_ner_dataset(dataset_name=args.dataset_name, dataset_path=args.dataset_path)
    samples = ds["test"][: args.num_examples]

    masker = NERMasker(args.model_dir)
    for i, (tokens, _) in enumerate(zip(samples["tokens"], samples["ner_tags"])):
        text = " ".join(tokens)
        spans = masker.predict_entities(text)
        print(f"Example {i+1}:")
        print("TEXT:", text)
        if not spans:
            print("ENTITIES: (none)")
        else:
            print("ENTITIES:")
            for s in spans:
                print(f" - {s.label}: '{text[s.start:s.end]}' @ [{s.start}, {s.end}]")
        print()


if __name__ == "__main__":
    main()

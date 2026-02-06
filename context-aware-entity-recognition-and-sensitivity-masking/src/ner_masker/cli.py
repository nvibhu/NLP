from __future__ import annotations

import argparse
import os
from pathlib import Path

from .inference import NERMasker


def iter_text_files(input_dir: str):
    for p in Path(input_dir).glob("**/*.txt"):
        if p.is_file():
            yield p


def main():
    parser = argparse.ArgumentParser(description="Batch mask text files using a fine-tuned NER model")
    parser.add_argument("--model_dir", required=True, help="Path to fine-tuned model directory")
    parser.add_argument("--input_dir", required=True, help="Directory containing .txt files")
    parser.add_argument("--output_dir", required=True, help="Directory to write masked files")
    parser.add_argument("--mask_token", default="[MASK]", help="Mask token text")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    masker = NERMasker(args.model_dir)

    for p in iter_text_files(args.input_dir):
        text = p.read_text(encoding="utf-8", errors="ignore")
        masked = masker.mask_text(text, mask_token=args.mask_token)
        rel = p.relative_to(args.input_dir)
        out_path = Path(args.output_dir) / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(masked, encoding="utf-8")
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

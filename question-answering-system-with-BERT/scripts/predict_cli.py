import argparse
import os
import sys

# Ensure project root is on sys.path so we can import src.* when running this script directly
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.infer import QAInference  # type: ignore


def main():
    parser = argparse.ArgumentParser(description="CLI for QA prediction using a fine-tuned BERT model")
    parser.add_argument("--context", type=str, required=True, help="Context passage")
    parser.add_argument("--question", type=str, required=True, help="Question to ask")
    parser.add_argument("--model_dir", type=str, default=os.environ.get("QA_MODEL_DIR", ""))
    args = parser.parse_args()

    qa = QAInference(model_dir=args.model_dir if args.model_dir else None)
    out = qa.predict(args.context, args.question)
    print(out["answer"])


if __name__ == "__main__":
    main()

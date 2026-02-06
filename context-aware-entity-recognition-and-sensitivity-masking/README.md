# Context-Aware Entity Recognition and Sensitivity Masking

This project implements a deep learning-based NER pipeline that distinguishes contextual entities and masks sensitive tokens in text. It includes:
- Fine-tuning a transformer model for token classification (NER)
- Evaluation with standard metrics and a confusion matrix
- An inference/masking pipeline
- A Streamlit web app for single-text and batch file processing with color-coded highlights
- A design document with architecture, literature survey, limitations, and LLM-based extensions

## Quickstart

1) Create a Python environment and install dependencies (Python 3.10+ recommended)

```
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

2) Fine-tune a model on a public dataset (default: conll2003)

```
python -m src.ner_masker.train \
  --dataset_name conll2003 \
  --base_model bert-base-cased \
  --output_dir models/bert-finetuned-ner \
  --num_train_epochs 3 \
  --batch_size 16
```

Notes:
- You can substitute another Hugging Face NER dataset (e.g., wnut_17). Ensure it provides `tokens` and `ner_tags` or adapt the loader.
- Training artifacts and logs will be saved under the specified output directory.

3) Evaluate and generate a confusion matrix

```
python -m src.ner_masker.evaluate \
  --model_dir models/bert-finetuned-ner \
  --dataset_name conll2003 \
  --output_dir outputs
```

This produces metrics on the test set and saves `outputs/confusion_matrix.png`.

4) Run the Streamlit app (single text + batch file upload)

```
streamlit run webapp/app.py
```

The app supports:
- Text input and live masking/highlighting
- Uploading multiple .txt files for batch processing
- Download masked outputs as a CSV

5) CLI for masking text files

```
python -m src.ner_masker.cli --model_dir models/bert-finetuned-ner --input_dir sample_data --output_dir outputs/masked
```

This will read all `.txt` files under `sample_data/`, mask entities, and write masked files to `outputs/masked/`.

## Project Structure

- src/ner_masker/
  - data_prep.py: Load dataset, tokenize, align labels
  - train.py: Fine-tune token classification model
  - evaluate.py: Evaluate and plot confusion matrix
  - inference.py: Inference pipeline and masking utilities
  - utils.py: Helpers for highlighting and span operations
  - cli.py: Batch masking command line utility
- webapp/app.py: Streamlit app (Part B)
- design/Design_Document.md: Architecture, research, limitations, LLM usage (Part C)
- outputs/: Evaluation artifacts (created after running scripts)
- requirements.txt: Dependencies

## Screenshots

Place your execution screenshots in `outputs/` or embed them in the design document. Suggested screenshots:
- Training logs/metrics
- Confusion matrix
- Web UI with example input and masked output
- Batch processing results

## Notes on Data and PII

- This template defaults to public datasets (e.g., `conll2003`, `wnut_17`). If you have a provided/attached dataset, adapt `data_prep.py` to load it and map labels accordingly.
- For masking behavior, the app replaces detected entity spans with `[MASK]`. You can customize this behavior (e.g., preserve entity type, partial masking) in `inference.py`.

## License
For coursework/educational use.
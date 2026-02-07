Question Answering System with BERT

A complete, end-to-end extractive QA project using Hugging Face Transformers. You can fine-tune BERT on SQuAD 1.1, explore the data, evaluate performance, and serve a simple web application for interactive QA.

Contents
- notebooks/Question_Answering_with_BERT.ipynb — end-to-end Jupyter Notebook (EDA → Fine-tune → Evaluate → Predict)
- src/ — reusable Python modules for data, preprocessing, training, evaluation, and inference
- webapp/ — Flask backend with a minimal HTML+JS frontend (index.html). Endpoint: POST /predict
- models/ — saved fine-tuned model artifacts (created after training)
- design/Design_Document.md — design and research document template
- scripts/predict_cli.py — quick CLI for single QA predictions
 - sample_data/ — small example payload
 - screenshots/ — put your CSIS Lab and test-case screenshots here

Quickstart
1) Create and activate a virtual environment (recommended)
   python3 -m venv .venv
   source .venv/bin/activate

2) Install dependencies
   pip install -r requirements.txt

3) Run the Jupyter Notebook
   jupyter notebook notebooks/Question_Answering_with_BERT.ipynb

4) Train and save the model (from the notebook or via src/train.py)
   python -m src.train --model_name bert-base-uncased --output_dir models/qa-bert \
     --epochs 2 --batch_size 8

5) Serve the web app
   export HF_HOME=.cache/huggingface
   export TRANSFORMERS_CACHE=.cache/huggingface
   export QA_MODEL_DIR=models/qa-bert
   python webapp/app.py
   # Then open http://127.0.0.1:5000

6) CLI prediction example
   python scripts/predict_cli.py --context "<your context>" \
     --question "Who wrote the book?" --model_dir models/qa-bert

Notes
- The notebook downloads SQuAD 1.1 from the datasets hub.
- Training can be compute-intensive. Reduce epochs/batch sizes for quick local runs.
- If you cannot fine-tune locally, you can still run the web app using the pretrained model; it will fall back to bert-base-uncased.
- Add your CSIS Lab screenshots to the screenshots/ directory.

Submission Hints
- Zip the entire question-answering-system-with-BERT directory including the executed notebook, screenshots, design document, and a saved model directory.

Assignment rubric mapping
- Part 1: Data Preprocessing
  - EDA, distributions, 5 Q&A prints: notebooks/Question_Answering_with_BERT.ipynb
  - Simple web page: webapp/templates/index.html served by webapp/app.py
  - Code is executable: use the notebook; cells are ordered from EDA → Train → Inference
- Part 2: Model Fine-Tuning
  - Fine-tune BERT: src/train.py (used in the notebook)
  - Hyperparams: CLI flags in src/train.py and in notebook
  - Inference from trained model: src/infer.py, CLI, and /predict endpoint
- Part 3: Web Application
  - Backend: webapp/app.py (/predict)
  - Frontend: webapp/templates/index.html (+ static CSS)
- Part 4: Design and Research
  - design/Design_Document.md


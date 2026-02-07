from __future__ import annotations
from typing import Dict, List, Tuple, Any
import collections
import numpy as np
import evaluate as hf_evaluate


def postprocess_qa_predictions(
    examples: Any,
    features: Any,
    predictions: Tuple[np.ndarray, np.ndarray],
    n_best_size: int = 20,
    max_answer_length: int = 30,
) -> Dict[str, str]:
    """Post-processes the predictions of a question-answering model to convert them to answers.

    Args:
        examples: The non-preprocessed dataset (with columns like id, context, question, answers)
        features: The processed dataset (with columns like input_ids, attention_mask, example_id, offset_mapping)
        predictions: The predictions of the model: two arrays containing the start logits and the end logits
        n_best_size: The total number of n-best predictions to generate when looking for an answer.
        max_answer_length: The maximum length of an answer that can be generated.

    Returns:
        A dictionary mapping example id to the final predicted answer text.
    """
    all_start_logits, all_end_logits = predictions

    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[feature["example_id"]].append(i)

    predictions_dict = {}
    for example in examples:
        example_id = example["id"]
        context = example["context"]

        feature_indices = features_per_example[example_id]

        min_null_score = None  # for SQuAD v2 (not used here but kept for completeness)
        valid_answers = []

        for feature_index in feature_indices:
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            offset_mapping = features[feature_index]["offset_mapping"]

            # Go through all possibilities for the start and end logits to find the best span.
            start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
            end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()

            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Skip answers that are not fully in the context
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or offset_mapping[end_index] is None
                    ):
                        continue
                    # Skip answers with a length that is either < 0 or > max_answer_length
                    if end_index < start_index or (end_index - start_index + 1) > max_answer_length:
                        continue
                    start_char = offset_mapping[start_index][0]
                    end_char = offset_mapping[end_index][1]
                    text = context[start_char:end_char]
                    score = start_logits[start_index] + end_logits[end_index]
                    valid_answers.append({"score": float(score), "text": text})

        if len(valid_answers) > 0:
            best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0]
        else:
            # In case no valid answer is found, return empty string
            best_answer = {"text": ""}

        predictions_dict[example_id] = best_answer["text"]

    return predictions_dict


def compute_squad_metrics(pred_texts: Dict[str, str], references: Any) -> Dict[str, float]:
    """Compute EM/F1 metrics using the 'squad' metric from evaluate.

    Args:
        pred_texts: mapping from example id to predicted answer text
        references: the examples split (each example has id and answers)
    Returns:
        Dict with 'exact_match' and 'f1'
    """
    metric = hf_evaluate.load("squad")
    preds = [{"id": k, "prediction_text": v} for k, v in pred_texts.items()]
    refs = [{"id": ex["id"], "answers": ex["answers"]} for ex in references]
    return metric.compute(predictions=preds, references=refs)

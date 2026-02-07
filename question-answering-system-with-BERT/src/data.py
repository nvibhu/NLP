from typing import Dict, List, Tuple
from datasets import load_dataset, DatasetDict
from collections import Counter


def load_squad(version: str = "1.1") -> DatasetDict:
    """Load the SQuAD dataset via Hugging Face datasets.

    Args:
        version: '1.1' or '2.0'. Defaults to '1.1'.

    Returns:
        A DatasetDict with 'train' and 'validation' splits.
    """
    if version == "1.1":
        ds = load_dataset("squad")
    elif version == "2.0":
        ds = load_dataset("squad_v2")
    else:
        raise ValueError("Unsupported SQuAD version. Use '1.1' or '2.0'.")
    return ds


def question_type_distribution(dataset_split) -> Dict[str, int]:
    """Compute distribution of question types by first token (e.g., 'what', 'who')."""
    counter: Counter = Counter()
    for q in dataset_split["question"]:
        first = (q.strip().split(" ")[0] if q.strip() else "").lower()
        counter[first] += 1
    return dict(counter)


def answer_length_distribution(dataset_split) -> Dict[int, int]:
    """Return distribution of answer lengths (in tokens/words) for first answer."""
    counter: Counter = Counter()
    for ans in dataset_split["answers"]:
        # answers is dict with 'text' (list) and 'answer_start' (list)
        text = ans.get("text", [])
        if text:
            length = len(text[0].split())
            counter[length] += 1
    return dict(counter)


def sample_qas(dataset_split, n: int = 5) -> List[Tuple[str, str, List[str]]]:
    """Return n sample (context, question, answers) tuples from a split."""
    n = min(n, len(dataset_split))
    samples: List[Tuple[str, str, List[str]]] = []
    for i in range(n):
        context = dataset_split[i]["context"]
        question = dataset_split[i]["question"]
        answers = dataset_split[i]["answers"]["text"]
        samples.append((context, question, answers))
    return samples

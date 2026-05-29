"""
dataset.py - MixLoRA training dataset loaders aligned with TUDB-Labs/MoE-PEFT.

The prompt templates and label formats follow MoE-PEFT's commonsense QA tasks:
  arc_c, arc_e, boolq, obqa, piqa, siqa, hellaswag, winogrande

Important:
  - Training examples append only the target label token(s), not the full answer text.
  - Padding is dynamic inside each batch instead of fixed at dataset build time.
"""

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from datasets import load_dataset
from datasets.exceptions import DatasetNotFoundError
from requests.exceptions import RequestException

logger = logging.getLogger(__name__)


COMMONSENSE_DATASET_CANDIDATES = {
    "arc_c": [("allenai/ai2_arc", "ARC-Challenge")],
    "arc_e": [("allenai/ai2_arc", "ARC-Easy")],
    "boolq": [
        ("google/boolq", None),
        ("super_glue", "boolq"),
    ],
    "obqa": [("allenai/openbookqa", "main")],
    "piqa": [("1-800-LLMs/piqa", None)],
    "siqa": [("baber/social_i_qa", None)],
    "hellaswag": [("Rowan/hellaswag", None)],
    "winogrande": [("allenai/winogrande", "winogrande_debiased")],
}


def _load_hf_split(
    dataset_name: str,
    candidates: Sequence[Tuple[str, Optional[str]]],
    split: str,
):
    errors = []
    for ds_id, config_name in candidates:
        try:
            if config_name:
                return load_dataset(ds_id, config_name, split=split)
            return load_dataset(ds_id, split=split)
        except DatasetNotFoundError as exc:
            target = f"{ds_id}/{config_name}" if config_name else ds_id
            logger.warning(
                "  %s: dataset loader %s is unavailable; trying fallback",
                dataset_name,
                target,
            )
            errors.append(f"{target}: {exc}")
        except RuntimeError as exc:
            if "Dataset scripts are no longer supported" not in str(exc):
                raise
            target = f"{ds_id}/{config_name}" if config_name else ds_id
            logger.warning(
                "  %s: dataset loader %s uses an unsupported dataset script; trying fallback",
                dataset_name,
                target,
            )
            errors.append(f"{target}: {exc}")
        except (ConnectionError, TimeoutError, OSError, RequestException) as exc:
            target = f"{ds_id}/{config_name}" if config_name else ds_id
            logger.warning(
                "  %s: dataset loader %s failed due to network/IO issue; trying fallback",
                dataset_name,
                target,
            )
            errors.append(f"{target}: {exc}")
        except Exception as exc:
            message = str(exc).lower()
            if not any(
                key in message
                for key in (
                    "timed out",
                    "timeout",
                    "connection",
                    "temporarily unavailable",
                    "max retries exceeded",
                    "name resolution",
                )
            ):
                raise
            target = f"{ds_id}/{config_name}" if config_name else ds_id
            logger.warning(
                "  %s: dataset loader %s failed due to network issue; trying fallback",
                dataset_name,
                target,
            )
            errors.append(f"{target}: {exc}")

    joined = " | ".join(errors) if errors else "no candidates configured"
    raise RuntimeError(f"Failed to load dataset '{dataset_name}' for split '{split}': {joined}")


def _fmt_arc(ex) -> Optional[dict]:
    choices = ex.get("choices", {})
    labels = choices.get("label", [])
    texts = choices.get("text", [])
    answer_key = ex.get("answerKey", "")
    if answer_key not in labels:
        return None

    prompt = "Please choose the correct answer to the question: " + ex["question"]
    for label, text in zip(labels, texts):
        prompt += f" ({label}) {text}"
    prompt += "\nAnswer:"
    return {"prompt": prompt, "response": answer_key}


def _fmt_boolq(ex) -> Optional[dict]:
    prompt = (
        "Please answer the following question with true or false: "
        + f"{ex['question']}?\nAnswer:"
    )
    response = "true" if ex["answer"] else "false"
    return {"prompt": prompt, "response": response}


def _fmt_obqa(ex) -> Optional[dict]:
    choices = ex.get("choices", {})
    labels = choices.get("label", [])
    texts = choices.get("text", [])
    answer_key = ex.get("answerKey", "")
    if answer_key not in labels:
        return None

    prompt = "Please choose the correct answer to the question: " + ex["question_stem"]
    for label, text in zip(labels, texts):
        prompt += f" ({label}) {text}"
    prompt += "\nAnswer:"
    return {"prompt": prompt, "response": answer_key}


def _fmt_piqa(ex) -> Optional[dict]:
    label = ex.get("label", -1)
    if label not in (0, 1):
        return None

    prompt = "Below is a common task along with two possible solutions labeled (A) and (B)."
    prompt += f" Please select the appropriate solution to achieve the task:\n{ex['goal']}\n"
    prompt += f"\n(A) {ex['sol1']}\n(B) {ex['sol2']}\n"
    prompt += "\nCorrect solution:"
    response = "A" if label == 0 else "B"
    return {"prompt": prompt, "response": response}


def _fmt_siqa(ex) -> Optional[dict]:
    label = ex.get("label", "")
    try:
        label_idx = int(label) - 1
    except (ValueError, TypeError):
        return None
    if label_idx not in (0, 1, 2):
        return None

    answers = [ex.get("answerA", ""), ex.get("answerB", ""), ex.get("answerC", "")]
    prompt = "Please choose the correct answer to the question.\n"
    prompt += f"Question: {ex['context']} {ex['question']}"
    prompt += f"\n(A) {answers[0]}"
    prompt += f"\n(B) {answers[1]}"
    prompt += f"\n(C) {answers[2]}"
    prompt += "\nAnswer:"
    response = chr(ord("A") + label_idx)
    return {"prompt": prompt, "response": response}


def _fmt_hellaswag(ex) -> Optional[dict]:
    label = ex.get("label", "")
    try:
        label_idx = int(label)
    except (ValueError, TypeError):
        return None

    endings = ex.get("endings", [])
    if label_idx < 0 or label_idx >= len(endings):
        return None

    prompt = "Please choose the correct ending to complete the given sentence.\n"
    prompt += f"Sentence: {ex['activity_label']}. {ex['ctx']}"
    for idx, ending in enumerate(endings):
        prompt += f"\n({chr(ord('A') + idx)}) {ending}"
    prompt += "\nAnswer:"
    response = chr(ord("A") + label_idx)
    return {"prompt": prompt, "response": response}


def _fmt_winogrande(ex) -> Optional[dict]:
    answer = ex.get("answer", "")
    if answer not in ("1", "2"):
        return None

    prompt = "Please choose the correct answer to fill in the blank to complete the given sentence.\n"
    prompt += f"Sentence: {ex['sentence']}"
    prompt += f"\n(A) {ex['option1']}\n(B) {ex['option2']}"
    prompt += "\nAnswer:"
    response = "A" if answer == "1" else "B"
    return {"prompt": prompt, "response": response}


COMMONSENSE_DATASET_FORMATTERS = {
    "arc_c": _fmt_arc,
    "arc_e": _fmt_arc,
    "boolq": _fmt_boolq,
    "obqa": _fmt_obqa,
    "piqa": _fmt_piqa,
    "siqa": _fmt_siqa,
    "hellaswag": _fmt_hellaswag,
    "winogrande": _fmt_winogrande,
}


def format_commonsense_example(dataset_name: str, ex) -> Optional[Dict[str, str]]:
    if dataset_name not in COMMONSENSE_DATASET_FORMATTERS:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. Choose from {list(COMMONSENSE_DATASET_FORMATTERS)}"
        )
    return COMMONSENSE_DATASET_FORMATTERS[dataset_name](ex)


class _TokenizedCommonsenseDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        raw_ds,
        dataset_name: str,
        tokenizer,
        max_length: int,
        tokenize_batch_size: int = 1024,
    ) -> None:
        self.pad_token_id = tokenizer.pad_token_id
        self.padding_side = getattr(tokenizer, "padding_side", "right")
        self.input_ids = []
        self.attention_mask = []

        pending_texts = []

        for ex in raw_ds:
            result = format_commonsense_example(dataset_name, ex)
            if result is None:
                continue
            pending_texts.append(result["prompt"] + " " + result["response"])

            if len(pending_texts) >= tokenize_batch_size:
                self._tokenize_chunk(tokenizer, max_length, pending_texts)
                pending_texts.clear()

        if pending_texts:
            self._tokenize_chunk(tokenizer, max_length, pending_texts)

        if not self.input_ids:
            raise RuntimeError(
                f"No valid examples loaded for dataset '{dataset_name}'"
            )

        logger.info("  %s: %s examples", dataset_name, len(self.input_ids))

    def _tokenize_chunk(self, tokenizer, max_length: int, texts) -> None:
        enc = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        self.input_ids.extend(enc["input_ids"])
        self.attention_mask.extend(enc["attention_mask"])

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "pad_token_id": self.pad_token_id,
            "padding_side": self.padding_side,
        }


def _load_commonsense_dataset(
    dataset_name: str,
    tokenizer,
    max_length: int,
    split: str = "train",
):
    if dataset_name not in COMMONSENSE_DATASET_CANDIDATES:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. Choose from {list(COMMONSENSE_DATASET_CANDIDATES)}"
        )

    raw = _load_hf_split(
        dataset_name,
        COMMONSENSE_DATASET_CANDIDATES[dataset_name],
        split,
    )
    return _TokenizedCommonsenseDataset(raw, dataset_name, tokenizer, max_length)


AVAILABLE_DATASETS = [
    *COMMONSENSE_DATASET_CANDIDATES.keys(),
]


def build_train_dataset(
    dataset_names: List[str],
    tokenizer,
    max_length: int,
):
    if not dataset_names:
        raise ValueError("dataset_names must be non-empty")

    datasets = []
    for name in dataset_names:
        logger.info("Loading training split: %s", name)
        ds = _load_commonsense_dataset(name, tokenizer, max_length, split="train")
        datasets.append(ds)

    if len(datasets) == 1:
        combined = datasets[0]
    else:
        combined = torch.utils.data.ConcatDataset(datasets)

    logger.info("Total training examples: %s", f"{len(combined):,}")
    return combined


def _left_pad_tensors(sequences, pad_value: int):
    max_len = max(seq.size(0) for seq in sequences)
    padded = []
    for seq in sequences:
        pad_len = max_len - seq.size(0)
        if pad_len > 0:
            pad = torch.full(
                (pad_len,),
                pad_value,
                dtype=seq.dtype,
                device=seq.device,
            )
            seq = torch.cat([pad, seq], dim=0)
        padded.append(seq)
    return torch.stack(padded, dim=0)


def collate_fn(batch):
    pad_token_id = batch[0]["pad_token_id"]
    padding_side = batch[0].get("padding_side", "right")

    input_tensors = [
        torch.as_tensor(b["input_ids"], dtype=torch.long)
        for b in batch
    ]
    mask_tensors = [
        torch.as_tensor(b["attention_mask"], dtype=torch.long)
        for b in batch
    ]

    if padding_side == "left":
        input_ids = _left_pad_tensors(input_tensors, pad_token_id)
        attention_mask = _left_pad_tensors(mask_tensors, 0)
    else:
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_tensors,
            batch_first=True,
            padding_value=pad_token_id,
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            mask_tensors,
            batch_first=True,
            padding_value=0,
        )

    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

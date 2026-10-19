"""Dual-crop inference with a (merged or LoRA-adapted) SmolVLM.

``predict_row`` mirrors the notebooks' single-image evaluation (greedy decoding, 12 new
tokens, per-image latency of ``model.generate`` only). ``predict_batch_rows`` is the batched
variant used for the per-epoch macro-F1 check during training.

Note: this calls the model directly - it does *not* go through the LangGraph pipeline, so
there is no retry, decision gate or temporal vote (the same scope as the notebooks' test
evaluation of the 500M model).
"""

from __future__ import annotations

import time
from types import ModuleType
from typing import Callable, Optional

from edgevlm.constants import IMAGE_SIZE, INSTRUCTION_V2, LABELS, MAX_NEW_TOKENS
from edgevlm.data.dataset import build_user_messages, make_dual_crop
from edgevlm.evaluation.metrics import compute_metrics, parse_prediction


def _load_dual_crop(row, split_name: str, resolver: Callable, preprocessing: ModuleType):
    import cv2
    from PIL import Image

    image_path = resolver(row["frame_path"], split_name=split_name, label=row["label"])
    raw_bgr = cv2.imread(str(image_path))
    if raw_bgr is None:
        raise FileNotFoundError(f"cv2 could not read: {image_path}")
    full_crop, face_crop = make_dual_crop(raw_bgr, preprocessing)
    return Image.fromarray(full_crop), Image.fromarray(face_crop), str(image_path)


def predict_row(
    model,
    processor,
    row,
    split_name: str,
    resolver: Callable,
    preprocessing: ModuleType,
    instruction: str = INSTRUCTION_V2,
    image_size: int = IMAGE_SIZE,
) -> tuple[str, str, bool, float, str]:
    """Return ``(raw_output, prediction, exactly_valid, latency_ms, image_path)`` for one row."""
    import torch

    full_image, face_image, image_path = _load_dual_crop(row, split_name, resolver, preprocessing)
    prompt = processor.apply_chat_template(
        build_user_messages(instruction), add_generation_prompt=True, tokenize=False
    )
    inputs = processor(
        text=prompt,
        images=[[full_image, face_image]],
        return_tensors="pt",
        size={"longest_edge": image_size},
        max_image_size={"longest_edge": image_size},
    ).to(next(model.parameters()).device)

    start = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    latency_ms = (time.perf_counter() - start) * 1000

    input_length = inputs["input_ids"].shape[-1]
    raw_output = processor.batch_decode(generated[:, input_length:], skip_special_tokens=True)[0].strip()
    prediction, exact_valid = parse_prediction(raw_output)
    return raw_output, prediction, exact_valid, latency_ms, image_path


def predict_batch_rows(
    model,
    processor,
    rows_df,
    split_name: str,
    resolver: Callable,
    preprocessing: ModuleType,
    instruction: str = INSTRUCTION_V2,
    image_size: int = IMAGE_SIZE,
    batch_size: int = 8,
) -> list[str]:
    """Batched predictions (labels only). Left padding is forced for decoder-only batched
    ``generate`` and restored afterwards, so it does not affect training."""
    import torch

    predictions: list[str] = []
    original_padding_side = processor.tokenizer.padding_side
    processor.tokenizer.padding_side = "left"
    try:
        prompt = processor.apply_chat_template(
            build_user_messages(instruction), add_generation_prompt=True, tokenize=False
        )
        for start in range(0, len(rows_df), batch_size):
            batch_rows = rows_df.iloc[start : start + batch_size]
            image_pairs = []
            for _, row in batch_rows.iterrows():
                full_image, face_image, _ = _load_dual_crop(row, split_name, resolver, preprocessing)
                image_pairs.append([full_image, face_image])

            inputs = processor(
                text=[prompt] * len(image_pairs),
                images=image_pairs,
                return_tensors="pt",
                padding=True,
                size={"longest_edge": image_size},
                max_image_size={"longest_edge": image_size},
            ).to(next(model.parameters()).device)

            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)

            input_length = inputs["input_ids"].shape[-1]
            for raw in processor.batch_decode(generated[:, input_length:], skip_special_tokens=True):
                predictions.append(parse_prediction(raw.strip())[0])
    finally:
        processor.tokenizer.padding_side = original_padding_side
    return predictions


def evaluate_split(
    model,
    processor,
    dataframe,
    split_name: str,
    resolver: Callable,
    preprocessing: ModuleType,
    instruction: str = INSTRUCTION_V2,
    image_size: int = IMAGE_SIZE,
    progress: Optional[Callable] = None,
):
    """Evaluate every row one image at a time. Returns ``(metrics_dict, per_row_dataframe)``."""
    iterator = dataframe.iterrows()
    if progress is not None:
        iterator = progress(iterator, total=len(dataframe), desc=f"Evaluating {split_name}")

    true_labels, predictions, raw_outputs, valid_flags, latencies, paths = [], [], [], [], [], []
    for _, row in iterator:
        raw, pred, exact, latency, path = predict_row(
            model, processor, row, split_name, resolver, preprocessing, instruction, image_size
        )
        true_labels.append(str(row["label"]).strip().upper())
        predictions.append(pred)
        raw_outputs.append(raw)
        valid_flags.append(exact)
        latencies.append(latency)
        paths.append(path)

    metrics = compute_metrics(true_labels, predictions, valid_flags, latencies, labels=LABELS)

    results = dataframe.copy()
    results["resolved_path"] = paths
    results["raw_output"] = raw_outputs
    results["prediction"] = predictions
    results["exact_label_valid"] = valid_flags
    results["latency_ms"] = latencies
    return metrics, results

"""Dual-crop dataset and multimodal collator for SmolVLM fine-tuning.

Each sample is the ``(full_crop, face_crop)`` pair produced by the *same* preprocessing
functions the deployed pipeline uses (``enhance_frame`` -> ``resize_full_frame`` /
``extract_face_crop``), so the model is trained on the pixels it will see at inference.
The collator masks the prompt and padding tokens so the loss covers only the label tokens.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any, Callable

try:  # torch is only needed to *use* the dataset in training; keep the module importable without it
    from torch.utils.data import Dataset as _Dataset
except ImportError:  # pragma: no cover - exercised only in torch-less environments
    _Dataset = object

from edgevlm.constants import IMAGE_SIZE, INSTRUCTION_V2


def build_user_messages(instruction: str) -> list[dict[str, Any]]:
    """Chat-template user turn: two image placeholders (full scene, face) followed by the instruction."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "image"},
                {"type": "text", "text": instruction},
            ],
        }
    ]


def make_dual_crop(raw_bgr, preprocessing: ModuleType):
    """Run one BGR frame through the pipeline preprocessing; return ``(full_crop, face_crop)`` RGB arrays."""
    enhanced_rgb, _frame_quality_ok = preprocessing.enhance_frame(raw_bgr)
    full_crop = preprocessing.resize_full_frame(enhanced_rgb)
    face_crop = preprocessing.extract_face_crop(enhanced_rgb, landmarks=None)
    return full_crop, face_crop


class DriverStateDualCropDataset(_Dataset):
    """Rows of a split CSV -> ``{full_image, face_image, messages, target, path}``."""

    def __init__(
        self,
        dataframe,
        split_name: str,
        resolver: Callable[..., Any],
        preprocessing: ModuleType,
        instruction: str = INSTRUCTION_V2,
    ) -> None:
        self.dataframe = dataframe.reset_index(drop=True)
        self.split_name = split_name
        self.resolver = resolver
        self.preprocessing = preprocessing
        self.instruction = instruction

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> dict[str, Any]:
        import cv2
        from PIL import Image

        row = self.dataframe.iloc[index]
        image_path = self.resolver(row["frame_path"], split_name=self.split_name, label=row["label"])

        raw_bgr = cv2.imread(str(image_path))
        if raw_bgr is None:
            raise FileNotFoundError(f"cv2 could not read: {image_path}")

        full_crop, face_crop = make_dual_crop(raw_bgr, self.preprocessing)
        return {
            "full_image": Image.fromarray(full_crop),
            "face_image": Image.fromarray(face_crop),
            "messages": build_user_messages(self.instruction),
            "target": str(row["label"]).strip(),
            "path": str(image_path),
        }


class SmolVLMDualCropCollator:
    """Batch ``DriverStateDualCropDataset`` items; labels = answer tokens only (prompt/pad = -100)."""

    def __init__(self, processor, image_size: int = IMAGE_SIZE) -> None:
        self.processor = processor
        self.image_size = image_size

    def __call__(self, features: list[dict[str, Any]]):
        images_for_processor = [[item["full_image"], item["face_image"]] for item in features]
        full_texts, prompt_texts = [], []

        for item in features:
            user_messages = item["messages"]
            full_messages = user_messages + [
                {"role": "assistant", "content": [{"type": "text", "text": item["target"]}]}
            ]
            prompt_texts.append(
                self.processor.apply_chat_template(user_messages, add_generation_prompt=True, tokenize=False)
            )
            full_texts.append(
                self.processor.apply_chat_template(full_messages, add_generation_prompt=False, tokenize=False)
            )

        processor_arguments = {
            "return_tensors": "pt",
            "padding": True,
            "truncation": True,
            "size": {"longest_edge": self.image_size},
            "max_image_size": {"longest_edge": self.image_size},
        }

        batch = self.processor(text=full_texts, images=images_for_processor, **processor_arguments)
        prompt_batch = self.processor(text=prompt_texts, images=images_for_processor, **processor_arguments)

        labels = batch["input_ids"].clone()
        prompt_lengths = prompt_batch["attention_mask"].sum(dim=1)
        for row_number, prompt_length in enumerate(prompt_lengths):
            labels[row_number, : int(prompt_length)] = -100

        pad_token_id = self.processor.tokenizer.pad_token_id
        if pad_token_id is not None:
            labels[labels == pad_token_id] = -100

        batch["labels"] = labels
        return batch

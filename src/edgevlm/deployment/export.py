"""Merge the best LoRA adapter into SmolVLM-500M and export a standalone FP16 package.

The package is a plain PyTorch / Hugging Face model (safetensors, no PEFT needed at
runtime, **no ONNX**). ONNX was tried for the Jetson Nano (JetPack 4.6.6) and abandoned
because of operator-compatibility barriers - see the interim report, Sec. VI-D.
"""

from __future__ import annotations

import gc
import shutil
from pathlib import Path
from typing import Optional

from edgevlm.constants import IMAGE_SIZE, LABELS, MODEL_ID

REQUIRED_ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")
REQUIRED_EXPORT_FILES = ("config.json", "preprocessor_config.json", "tokenizer.json", "tokenizer_config.json")


def check_adapter_dir(adapter_dir: Path) -> None:
    adapter_dir = Path(adapter_dir)
    if not adapter_dir.is_dir():
        raise FileNotFoundError(
            f"Best adapter was not found at: {adapter_dir}\n"
            "Run training first and make sure the macro-F1 checkpoint was saved."
        )
    missing = [name for name in REQUIRED_ADAPTER_FILES if not (adapter_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing adapter files: {missing}\nAdapter directory: {adapter_dir}")


def write_deployment_info(
    export_dir: Path, adapter_dir: Path, model_id: str = MODEL_ID, image_size: int = IMAGE_SIZE
) -> Path:
    """Write ``deployment_info.txt`` (``key: value`` lines, same format as the notebooks)."""
    info = {
        "model_id": model_id,
        "model_type": "SmolVLM-500M-Instruct",
        "deployment_backend": "PyTorch",
        "onnx_used": False,
        "adapter_merged": True,
        "image_size": image_size,
        "labels": list(LABELS),
        "source_adapter": str(adapter_dir),
    }
    path = Path(export_dir) / "deployment_info.txt"
    with open(path, "w", encoding="utf-8") as handle:
        for key, value in info.items():
            handle.write(f"{key}: {value}\n")
    return path


def merge_and_export(
    adapter_dir: Path,
    export_dir: Path,
    model_id: str = MODEL_ID,
    model_cache: Optional[Path] = None,
    image_size: int = IMAGE_SIZE,
    make_zip: bool = False,
) -> Path:
    """Merge ``adapter_dir`` into the base model and save a standalone FP16 model to ``export_dir``."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForVision2Seq, AutoProcessor

    adapter_dir, export_dir = Path(adapter_dir), Path(export_dir)
    check_adapter_dir(adapter_dir)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    cache = {"cache_dir": str(model_cache)} if model_cache else {}
    base = AutoModelForVision2Seq.from_pretrained(
        model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True, **cache
    )
    adapter_model = PeftModel.from_pretrained(base, str(adapter_dir), is_trainable=False)
    adapter_model.eval()

    merged = adapter_model.merge_and_unload()
    merged.eval()
    print("Merged parameter count:", sum(p.numel() for p in merged.parameters()))

    export_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(export_dir), safe_serialization=True, max_shard_size="2GB")
    AutoProcessor.from_pretrained(str(adapter_dir)).save_pretrained(str(export_dir))
    write_deployment_info(export_dir, adapter_dir, model_id, image_size)

    verify_export(export_dir)

    if make_zip:
        archive = shutil.make_archive(str(export_dir), "zip", root_dir=str(export_dir))
        print("Deployment archive created:", archive)
    return export_dir


def verify_export(export_dir: Path) -> None:
    """Check the required files exist and the package loads with ``local_files_only``."""
    import torch
    from transformers import AutoModelForVision2Seq, AutoProcessor

    export_dir = Path(export_dir)
    missing = [name for name in REQUIRED_EXPORT_FILES if not (export_dir / name).is_file()]
    if missing or not list(export_dir.glob("*.safetensors")):
        raise FileNotFoundError(f"Incomplete export in {export_dir}: missing {missing or 'weights'}")

    AutoProcessor.from_pretrained(str(export_dir), local_files_only=True)
    model = AutoModelForVision2Seq.from_pretrained(
        str(export_dir),
        local_files_only=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
    )
    model.eval()
    size_mb = sum(f.stat().st_size for f in export_dir.glob("*.safetensors")) / (1024**2)
    print(f"Exported model loads successfully ({size_mb:.0f} MB of weights): {export_dir}")

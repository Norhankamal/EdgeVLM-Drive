"""LoRA on BOTH the vision encoder and the text decoder of SmolVLM.

Adapting the vision encoder lets the model learn a *new visual cue* (e.g. eye closure)
instead of only re-mapping cues it could already see; the text-decoder adapters handle the
label mapping. Targets are the attention projections (q/k/v/o and the vision ``out_proj``).
Target modules are classified by name (``.text_model.`` / ``.vision_model.``) and the counts
are checked, because an empty vision list would mean a SmolVLM revision with different
module names and would silently train a text-only adapter.
"""

from __future__ import annotations

from typing import Iterable, Optional

ATTENTION_PROJ_SUFFIXES = ("q_proj", "k_proj", "v_proj", "o_proj", "out_proj")


def find_lora_targets(
    named_modules: Iterable[tuple[str, object]],
    linear_cls: Optional[type] = None,
    suffixes: tuple[str, ...] = ATTENTION_PROJ_SUFFIXES,
) -> tuple[list[str], list[str], list[str]]:
    """Return ``(text_targets, vision_targets, other_targets)`` module names.

    ``linear_cls`` defaults to ``torch.nn.Linear`` (injectable so the logic is testable
    without torch).
    """
    if linear_cls is None:
        import torch

        linear_cls = torch.nn.Linear

    text_targets: list[str] = []
    vision_targets: list[str] = []
    other_targets: list[str] = []
    for name, module in named_modules:
        if not (isinstance(module, linear_cls) and name.endswith(suffixes)):
            continue
        if ".text_model." in name:
            text_targets.append(name)
        elif ".vision_model." in name:
            vision_targets.append(name)
        else:
            other_targets.append(name)
    return text_targets, vision_targets, other_targets


def select_lora_targets(model, linear_cls: Optional[type] = None) -> list[str]:
    """All text + vision attention targets of ``model``; raises if either group is empty."""
    text, vision, other = find_lora_targets(model.named_modules(), linear_cls=linear_cls)
    print("Text-decoder attention layers matched:", len(text))
    print("Vision-encoder attention layers matched:", len(vision))
    if other:
        print("Matched but NOT classified as text or vision (review these):", other[:10])
    if not text:
        raise RuntimeError("No text-model attention layers matched - inspect model.named_modules().")
    if not vision:
        raise RuntimeError(
            "No vision-model attention layers matched - this SmolVLM revision's module names may differ "
            "from what ATTENTION_PROJ_SUFFIXES / '.vision_model.' assume. Print "
            "[n for n, _ in model.named_modules() if 'vision' in n.lower()][:30] to inspect the real names."
        )
    return sorted(text + vision)


def attach_lora(model, r: int = 16, alpha: int = 32, dropout: float = 0.05):
    """Attach vision+text LoRA adapters; enable gradient checkpointing; return the PEFT model."""
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model

    if isinstance(model, PeftModel):
        print("A LoRA adapter is already attached.")
        model.train()
        model.print_trainable_parameters()
        return model

    targets = select_lora_targets(model)
    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=targets,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()  # extra memory headroom: the vision encoder trains too
    model.enable_input_require_grads()  # needed for gradient checkpointing with a frozen base model
    model = get_peft_model(model, config)
    model.train()
    model.print_trainable_parameters()
    return model

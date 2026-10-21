import json
import re
from pathlib import Path

import nbformat

from edgevlm.constants import INSTRUCTION_V1, INSTRUCTION_V2

NB_DIR = Path(__file__).resolve().parents[1] / "notebooks"
NOTEBOOKS = sorted(NB_DIR.rglob("*.ipynb"))


def _instruction_from(nb_path):
    nb = json.load(open(nb_path))
    # Locate the prompt cell by content, so adding or removing documentation cells cannot break the test.
    src = next("".join(c["source"]) for c in nb["cells"]
               if c["cell_type"] == "code" and "INSTRUCTION = (" in "".join(c["source"]))
    m = re.search(r"INSTRUCTION = \((.*?)\n\)\n", src, re.S)
    ns = {}
    exec("INSTRUCTION = (" + m.group(1) + "\n)", ns)
    return ns["INSTRUCTION"]


def test_prompts_match_notebooks():
    d = NB_DIR / "finetuning_smolvlm500"
    assert _instruction_from(d / "01_phase1_baseline_dualcrop_lora.ipynb") == INSTRUCTION_V1
    assert _instruction_from(d / "02_phase2_extended_data_v1.ipynb") == INSTRUCTION_V1
    assert _instruction_from(d / "04_phase2_extended_data_final.ipynb") == INSTRUCTION_V2


def test_notebooks_are_valid_and_clean():
    assert NOTEBOOKS, "no notebooks found"
    for path in NOTEBOOKS:
        nb = nbformat.read(path, as_version=4)
        nbformat.validate(nb)
        raw = path.read_text()
        assert "image/png" not in raw and "image/jpeg" not in raw, f"{path.name}: embedded image output"
        assert not re.search(r"hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_\-]{20,}|gh[pousr]_[A-Za-z0-9]{20,}", raw), path.name
        assert "widgets" not in nb.metadata

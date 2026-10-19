from pathlib import Path

import pytest

from edgevlm.config import load_config

CONFIGS = sorted((Path(__file__).resolve().parents[1] / "configs").glob("*.yaml"))


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_every_config_loads(path, monkeypatch):
    monkeypatch.setenv("EDGEVLM_DATA_ROOT", "/data")
    cfg = load_config(path)
    assert cfg.csv("train").is_absolute()
    assert cfg.instruction_text


def test_phase1_matches_documented_settings(monkeypatch):
    monkeypatch.setenv("EDGEVLM_DATA_ROOT", "/data")
    c = load_config(CONFIGS[0].parent / "smolvlm500m_phase1_baseline.yaml")
    assert (c.lora.r, c.lora.alpha) == (8, 16)
    assert c.validation_check.samples_per_class * 4 == 160 and c.validation_check.patience == 2
    assert c.training.sampler_weight_power == 1.0 and c.data.preprocessing == "v1"


def test_final_matches_documented_settings(monkeypatch):
    monkeypatch.setenv("EDGEVLM_DATA_ROOT", "/data")
    c = load_config(CONFIGS[0].parent / "smolvlm500m_phase2_final.yaml")
    assert (c.lora.r, c.lora.alpha) == (16, 32)
    assert c.validation_check.samples_per_class * 4 == 600 and c.validation_check.patience == 3
    assert c.training.sampler_weight_power == 0.5 and c.data.preprocessing == "v2"


def test_unknown_yaml_key_rejected(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("lora: {rank: 8}\n")
    with pytest.raises(ValueError, match="Unknown key"):
        load_config(p)

import json
from collections import Counter

import numpy as np
import pandas as pd
import pytest

from edgevlm.constants import LABELS
from edgevlm.modeling.lora import find_lora_targets
from edgevlm.training.callbacks import MacroF1EarlyStopping, balanced_validation_subset
from edgevlm.training.resume import find_resume_checkpoint
from edgevlm.training.trainer import inverse_frequency_weights


def test_sampler_weights_balance_classes():
    labels = ["N"] * 90 + ["D"] * 10
    for power in (1.0, 0.5):
        w = np.array(inverse_frequency_weights(labels, power))
        share_d = w[90:].sum() / w.sum()
        assert share_d > 0.1  # rare class is up-weighted
    full = np.array(inverse_frequency_weights(labels, 1.0))
    assert full[90:].sum() / full.sum() == pytest.approx(0.5)             # perfectly balanced
    damp = np.array(inverse_frequency_weights(labels, 0.5))
    assert 0.1 < damp[90:].sum() / damp.sum() < 0.5                        # damped, not fully balanced


def test_lora_target_classification():
    class Lin: ...
    mods = [
        ("model.text_model.layers.0.self_attn.q_proj", Lin()),
        ("model.text_model.layers.0.mlp.up_proj", Lin()),
        ("model.vision_model.encoder.layers.0.self_attn.out_proj", Lin()),
        ("model.connector.o_proj", Lin()),
        ("model.text_model.layers.0.self_attn.k_proj", object()),   # not a Linear
    ]
    text, vision, other = find_lora_targets(mods, linear_cls=Lin)
    assert text == ["model.text_model.layers.0.self_attn.q_proj"]
    assert vision == ["model.vision_model.encoder.layers.0.self_attn.out_proj"]
    assert other == ["model.connector.o_proj"]


def test_balanced_subset_caps_small_classes():
    df = pd.DataFrame({"label": ["NORMAL"] * 50 + ["DROWSY"] * 50 + ["LOOKING_AWAY"] * 5 + ["PHONE_USAGE"] * 50})
    sub = balanced_validation_subset(df, 20)
    assert Counter(sub["label"]) == {"NORMAL": 20, "DROWSY": 20, "LOOKING_AWAY": 5, "PHONE_USAGE": 20}


class _FakeModel:
    def __init__(self): self.saved = []
    def eval(self): return self
    def train(self): return self
    def save_pretrained(self, path): self.saved.append(path)


class _FakeProcessor:
    def save_pretrained(self, path): pass


class _State:
    def __init__(self, epoch): self.epoch = epoch


class _Control:
    should_training_stop = False


def _run_epochs(cb, scores_by_epoch, check_df):
    model, control = _FakeModel(), _Control()
    for i, correct_fraction in enumerate(scores_by_epoch, start=1):
        true = check_df["label"].tolist()
        n_wrong = int(round(len(true) * (1 - correct_fraction)))
        preds = ["NORMAL" if j < n_wrong and t != "NORMAL" else t for j, t in enumerate(true)]
        cb.predict_fn = lambda m, p, df, preds=preds: preds
        cb.on_epoch_end(None, _State(float(i)), control, model=model)
    return model, control


@pytest.fixture
def check_df():
    return pd.DataFrame({"label": list(LABELS) * 10})


def test_best_checkpoint_saved_only_on_improvement_and_patience_stops(tmp_path, check_df):
    cb = MacroF1EarlyStopping(_FakeProcessor(), check_df, tmp_path, predict_fn=None, patience=2)
    # epoch1 good, epoch2 worse, epoch3 worse -> stop after 2 non-improving epochs
    model, control = _run_epochs(cb, [0.9, 0.2, 0.1], check_df)
    assert len(model.saved) == 1                       # only the first epoch improved
    assert control.should_training_stop is True
    log = json.loads((tmp_path / "macro_f1_epoch_log.json").read_text())
    assert [r["improved"] for r in log] == [True, False, False]


def test_resume_restores_best_so_far(tmp_path, check_df):
    (tmp_path / "macro_f1_epoch_log.json").write_text(json.dumps(
        [{"epoch": 1.0, "macro_f1": 0.5}, {"epoch": 2.0, "macro_f1": 0.8}, {"epoch": 3.0, "macro_f1": 0.6}]))
    cb = MacroF1EarlyStopping(_FakeProcessor(), check_df, tmp_path, predict_fn=None, patience=3, resume=True)
    assert cb.best_macro_f1 == 0.8 and cb.epochs_without_improvement == 1
    # a worse epoch after restart must NOT overwrite the saved best
    model, _ = _run_epochs(cb, [0.3], check_df)
    assert model.saved == []


def test_failed_check_does_not_stop_training(tmp_path, check_df):
    def boom(*a): raise RuntimeError("gpu hiccup")
    cb = MacroF1EarlyStopping(_FakeProcessor(), check_df, tmp_path, predict_fn=boom)
    control = _Control()
    cb.on_epoch_end(None, _State(1.0), control, model=_FakeModel())
    assert control.should_training_stop is False


def test_find_resume_checkpoint_skips_unusable_newest(tmp_path):
    for step in (100, 300, 200):
        (tmp_path / f"checkpoint-{step}").mkdir()
    (tmp_path / "checkpoint-final").mkdir()  # ignored: not checkpoint-<int>
    usable = lambda p: p.name != "checkpoint-300"
    assert find_resume_checkpoint(tmp_path, usable=usable).name == "checkpoint-200"
    assert find_resume_checkpoint(tmp_path, usable=lambda p: False) is None

import importlib.util
from pathlib import Path

import torch

spec = importlib.util.spec_from_file_location(
    "baseline", Path(__file__).resolve().parents[1] / "rsna_knee_baseline.py"
)
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)


def test_confidence_scales_single_study_loss_and_gradient():
    logits = torch.zeros(1, 12, requires_grad=True)
    targets = torch.ones_like(logits)
    gold = baseline.weighted_bce(logits, targets, torch.ones_like(logits))
    gold_gradient = torch.autograd.grad(gold, logits)[0]
    pseudo = baseline.weighted_bce(logits, targets, torch.full_like(logits, 0.55))
    pseudo_gradient = torch.autograd.grad(pseudo, logits)[0]
    torch.testing.assert_close(pseudo, gold * 0.55)
    torch.testing.assert_close(pseudo_gradient, gold_gradient * 0.55)


def test_zero_confidence_has_zero_gradient():
    logits = torch.zeros(1, 12, requires_grad=True)
    loss = baseline.weighted_bce(logits, torch.ones_like(logits), torch.zeros_like(logits))
    loss.backward()
    assert torch.count_nonzero(logits.grad) == 0

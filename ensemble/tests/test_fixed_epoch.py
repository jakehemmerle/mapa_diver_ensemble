import math

import numpy as np
import pytest

from ensemble.members import diver


def test_fixed_epoch_head_matches_upstream_training_prefix(monkeypatch, tmp_path):
    """Compare the real CUDA optimization paths without any V/T inputs."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        pytest.skip("the pinned DIVER path requires CUDA with bfloat16")

    backend = diver.Backend()
    backend._imports()
    import finetune_trainer as upstream

    monkeypatch.setattr(diver.paths, "scratch", lambda: tmp_path)
    epochs = 8
    features = torch.randn(105, diver.TOKENS, generator=torch.Generator().manual_seed(53)).bfloat16()
    x, x_select, x_check = features[:73], features[73:97], features[97:]
    y, y_select = np.arange(len(x)) % 2, np.arange(len(x_select)) % 2
    prefix_weights = {}
    optimizer_lrs, scheduler_lrs, horizons = [], [], []

    class PrefixPruner(upstream.EarlyStopPruner):
        def __call__(self, epoch, model_state_dict, val_loss, val_metrics):
            stopped = super().__call__(epoch, model_state_dict, val_loss, val_metrics)
            assert not stopped  # patience is longer than this prefix
            if epoch + 1 == epochs:
                prefix_weights.update({k: v.detach().clone() for k, v in model_state_dict.items()})
                return True
            return False

    class TracedAdamW(torch.optim.AdamW):
        def step(self, *args, **kwargs):
            optimizer_lrs.append(self.param_groups[0]["lr"])
            return super().step(*args, **kwargs)

    class TracedCosineAnnealingLR(torch.optim.lr_scheduler.CosineAnnealingLR):
        def step(self, *args, **kwargs):
            super().step(*args, **kwargs)
            scheduler_lrs.append(self.get_last_lr()[0])
            horizons.append(self.T_max)

    monkeypatch.setattr(upstream, "EarlyStopPruner", PrefixPruner)
    monkeypatch.setattr(torch.optim, "AdamW", TracedAdamW)
    monkeypatch.setattr(torch.optim.lr_scheduler, "CosineAnnealingLR", TracedCosineAnnealingLR)

    normal = backend.fit(x, y, x_select, y_select, seed=0)
    assert np.isfinite(normal.select_auroc)
    assert 1 <= normal.best_epoch <= epochs
    # Trainer restored its real AUROC winner. Compare with the genuine eighth
    # checkpoint captured before that restoration, not necessarily the winner.
    normal.head.load_state_dict(prefix_weights)
    normal_scores = normal.predict(x_check)
    expected_path = optimizer_lrs[:], scheduler_lrs[:], horizons[:]
    optimizer_lrs.clear()
    scheduler_lrs.clear()
    horizons.clear()

    def unexpected_evaluation(*_args, **_kwargs):
        pytest.fail("fixed training ran a validation or test evaluator")

    monkeypatch.setattr(upstream.Evaluator, "evaluate", unexpected_evaluation)
    fixed = backend.fit(x, y, x_select, np.zeros_like(y_select), seed=0, fixed_epochs=epochs)
    assert fixed.best_epoch is None and fixed.select_auroc is None
    batches = math.ceil(len(x) / backend.config["batch_size"])
    updates = epochs * batches
    horizon = backend.config["epochs"] * batches
    assert len(optimizer_lrs) == updates
    assert len(scheduler_lrs) == updates + 1  # includes initialization at step zero
    assert horizons == [horizon] * (updates + 1)
    assert (optimizer_lrs, scheduler_lrs, horizons) == expected_path
    assert optimizer_lrs == scheduler_lrs[:-1]
    lr = backend.config["lr"]
    expected_final_lr = lr * 0.01 + (lr - lr * 0.01) * (1 + math.cos(math.pi * updates / horizon)) / 2
    assert scheduler_lrs[-1] == pytest.approx(expected_final_lr, abs=1e-15)
    for name, value in fixed.head.state_dict().items():
        torch.testing.assert_close(value, prefix_weights[name], rtol=0, atol=0)
    np.testing.assert_array_equal(fixed.predict(x_check), normal_scores)

import numpy as np
import pytest
from sklearn.linear_model import Ridge

from common import ridge
from common.ridge import LAM_MULTS, KernelSet, fit_predict


def _data(seed=0, n=60, n_select=40, n_eval=30, dims=(200, 15)):
    rng = np.random.default_rng(seed)
    y = (np.arange(n) % 2).astype(float)
    y_select = (np.arange(n_select) % 2).astype(float)
    y_eval = (np.arange(n_eval) % 2).astype(float)
    blocks = []
    for d in dims:
        direction = rng.standard_normal(d)

        def draw(labels):
            z = rng.standard_normal((len(labels), d)) + 0.6 * np.outer(labels - 0.5, direction)
            return z
        train, select, evaluate = draw(y), draw(y_select), draw(y_eval)
        mean, scale = train.mean(0), train.std(0)
        blocks.append(tuple((z - mean) / scale for z in (train, select, evaluate)))
    kernels = [KernelSet(train=t @ t.T, cross={"select": s @ t.T, "eval": e @ t.T}) for t, s, e in blocks]
    return blocks, kernels, y, y_select, y_eval


def _direct(kernels, y, multiplier, weights=None):
    n = len(y)
    weights = np.ones(len(kernels)) if weights is None else np.asarray(weights)
    gram = sum(w * n * k.train / np.trace(k.train) for w, k in zip(weights, kernels))
    alpha = np.linalg.solve(gram + multiplier * np.trace(gram) / n * np.eye(n), y)
    return {part: sum(w * n * k.cross[part] / np.trace(k.train) for w, k in zip(weights, kernels)) @ alpha
            for part in kernels[0].cross}


def test_grid_is_mapas():
    assert len(LAM_MULTS) == 25
    np.testing.assert_allclose(LAM_MULTS, np.logspace(-4, 4, 25))


@pytest.mark.parametrize("strict", [False, True])
def test_matches_direct_closed_form_and_grid_search(strict):
    _, kernels, y, y_select, _ = _data()
    out = fit_predict(kernels, y, y_select, strict=strict)
    aurocs = [ridge.auroc(_direct(kernels, y, m)["select"], y_select) for m in LAM_MULTS]
    expected = int(np.flatnonzero(np.array(aurocs) == max(aurocs))[0])
    assert out["lam_index"] == expected and out["lam"] == LAM_MULTS[expected]
    assert out["select_auroc"] == pytest.approx(max(aurocs))
    direct = _direct(kernels, y, out["lam"])
    for part in ("select", "eval"):
        np.testing.assert_allclose(out["scores"][part], direct[part], rtol=1e-8, atol=1e-10)


def test_single_kernel_matches_sklearn_ridge():
    blocks, kernels, y, y_select, _ = _data(seed=1, dims=(200,))
    train, select, evaluate = blocks[0]
    out = fit_predict(kernels, y, y_select)
    penalty = out["lam"] * np.trace(train @ train.T) / len(y)   # MAPA's scale: multiplier * trace(G) / n
    model = Ridge(alpha=penalty, fit_intercept=False, solver="cholesky").fit(train, y)
    np.testing.assert_allclose(out["scores"]["select"], model.predict(select), rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(out["scores"]["eval"], model.predict(evaluate), rtol=1e-6, atol=1e-8)


def test_fixed_weights_are_applied_not_searched():
    _, kernels, y, y_select, _ = _data(seed=2)
    out = fit_predict(kernels, y, y_select, weights=(1.0, 0.25))
    direct = _direct(kernels, y, out["lam"], weights=(1.0, 0.25))
    np.testing.assert_allclose(out["scores"]["eval"], direct["eval"], rtol=1e-8, atol=1e-10)
    equal = fit_predict(kernels, y, y_select)
    assert not np.allclose(equal["scores"]["eval"], out["scores"]["eval"])
    with pytest.raises(ValueError):
        fit_predict(kernels, y, y_select, weights=(1.0,))


def test_ties_take_the_smallest_penalty():
    _, kernels, y, _, _ = _data(seed=3)
    select = KernelSet(train=kernels[0].train, cross={"select": kernels[0].cross["select"][:2]})
    out = fit_predict([select], y, np.array([0.0, 1.0]))   # two select rows: AUROC is 0 or 1, so ties abound
    grid = np.array([ridge.auroc(_direct([select], y, m)["select"], [0, 1]) for m in LAM_MULTS])
    assert out["n_tied"] == int(np.sum(grid == grid.max())) > 1
    assert out["lam_index"] == int(np.argmax(grid == grid.max()))


def test_out_of_fold_scores_never_see_the_held_block():
    _, kernels, y, y_select, _ = _data(seed=4)
    positions = np.arange(len(y))
    blocks = [(np.setdiff1d(positions, held), held) for held in np.array_split(positions, 5)]
    out = fit_predict(kernels, y, y_select, oof_blocks=blocks)
    assert np.isfinite(out["scores"]["train"]).all()
    n = len(y)
    gram = sum(n * k.train / np.trace(k.train) for k in kernels)
    for fit, held in blocks:     # ridge with an intercept on features centred on the fit rows
        h = np.eye(len(fit)) - 1.0 / len(fit)
        sub = h @ gram[np.ix_(fit, fit)] @ h
        cross = (gram[np.ix_(held, fit)] - gram[np.ix_(fit, fit)].mean(axis=0)[None, :]) @ h
        alpha = np.linalg.solve(sub + out["lam"] * np.trace(sub) / len(fit) * np.eye(len(fit)), y[fit] - y[fit].mean())
        np.testing.assert_allclose(out["scores"]["train"][held], cross @ alpha + y[fit].mean(), rtol=1e-7, atol=1e-9)
    fit, held = blocks[2]
    flipped = y.copy()
    flipped[held] = 1 - flipped[held]
    again = ridge._out_of_fold(gram, flipped, blocks, out["lam"])
    np.testing.assert_allclose(again[held], out["scores"]["train"][held], rtol=1e-9, atol=1e-12)


def test_fixed_multiplier_skips_grid_and_scores_all_parts(monkeypatch):
    features, kernels, y, y_select, _ = _data(seed=6, dims=(15,))
    train, _, _ = features[0]
    positions = np.arange(len(y))
    blocks = [(np.setdiff1d(positions, held), held) for held in np.array_split(positions, 5)]
    lam = LAM_MULTS[13]

    def unexpected_auc(*args):
        pytest.fail("the fixed fit must not compute AUROC")

    monkeypatch.setattr(ridge, "auroc", unexpected_auc)
    out = fit_predict(kernels, y, np.zeros_like(y_select), oof_blocks=blocks, fixed_lam=lam)
    assert out["lam"] == lam
    assert out["select_auroc"] is out["lam_index"] is out["n_tied"] is None
    for part, expected in _direct(kernels, y, lam).items():
        np.testing.assert_allclose(out["scores"][part], expected, rtol=1e-8, atol=1e-10)
    for fit, held in blocks:
        centered = train[fit] - train[fit].mean(axis=0)
        penalty = lam * np.sum(centered ** 2) / len(fit)
        model = Ridge(alpha=penalty, solver="cholesky").fit(train[fit], y[fit])
        np.testing.assert_allclose(out["scores"]["train"][held], model.predict(train[held]), atol=1e-10)


def test_fixed_multiplier_rejects_other_degeneracy():
    _, kernels, y, y_select, _ = _data(seed=7, dims=(15,))
    lam = LAM_MULTS[11]
    one_class = np.ones_like(y_select)
    with pytest.raises(ValueError, match="declared grid"):
        fit_predict(kernels, y, one_class, fixed_lam=2.0)
    with pytest.raises(ValueError, match="one-class select"):
        fit_predict(kernels, y, y_select, fixed_lam=lam)
    for malformed in (np.array([]), np.full_like(y_select, np.nan)):
        with pytest.raises(ridge.Degenerate, match="select labels"):
            fit_predict(kernels, y, malformed, fixed_lam=lam)
    for malformed in (np.ones_like(y), np.full_like(y, np.nan)):
        with pytest.raises(ridge.Degenerate, match="training labels"):
            fit_predict(kernels, malformed, one_class, fixed_lam=lam)
    fit, held = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    with pytest.raises(ridge.Degenerate, match="out-of-fold fit labels"):
        fit_predict(kernels, y, one_class, oof_blocks=[(fit, held)], fixed_lam=lam)
    evaluate = kernels[0].cross["eval"]
    for malformed in (evaluate[:0], np.full_like(evaluate, np.inf)):
        kernels[0].cross["eval"] = malformed
        with pytest.raises(ridge.Degenerate, match="eval cross-kernel"):
            fit_predict(kernels, y, one_class, fixed_lam=lam)


def test_degenerate_inputs_raise():
    _, kernels, y, y_select, _ = _data(seed=5)
    with pytest.raises(ridge.Degenerate):
        fit_predict(kernels, np.ones_like(y), y_select)
    with pytest.raises(ridge.Degenerate):
        fit_predict(kernels, y, np.ones_like(y_select))

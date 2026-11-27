import numpy as np
import pytest

from classical import controls, features
from common import ridge, splits
from common.splits import Cell
from synthetic import Provider, make_recording, make_table

SESSION = "btbank9_0"          # recorded at 1024 Hz: half the samples and the non-default rate path
RATE = 1024
ALL_BLOCKS = features.BLOCKS + features.SURROGATE_BLOCKS


@pytest.fixture(scope="module")
def planted(tmp_path_factory):
    """One within-session Tier P cell with a phase-locked response on one contact, features cached."""
    root = tmp_path_factory.mktemp("features")
    table = make_table(RATE, 420, seed=7)
    recordings = {SESSION: make_recording(table, RATE, seed=11)}
    return root, table, recordings


@pytest.fixture
def cell(planted):
    root, table, recordings = planted
    splits.set_provider(Provider({SESSION: table}))
    features.set_recording_source(recordings.__getitem__)
    features.extract(SESSION, {("window", "bipolar"): ALL_BLOCKS, ("session", "bipolar"): features.BLOCKS,
                               ("window", "laplacian"): ("base6",), ("window", "shaft_mean"): ("base6",)},
                     root=root, workers=1)
    return Cell("P", "within", SESSION, SESSION, "speech", 0), root


def _eval_auroc(kernel, split) -> float:
    fit = ridge.fit_predict([kernel], split.labels("train"), split.labels("select"))
    return ridge.auroc(fit["scores"]["eval"], split.labels("eval"))


def test_channel_definitions():
    labels = ("A1", "A2", "A3", "A4", "B1", "B2", "B3", "C10", "T1cIf12")
    regions = {"A1": "ctx-lh-insula", "A2": "Left-Hippocampus", "A3": "Unknown", "C10": "ctx-rh-insula"}
    bipolar = features.channels(labels, regions, "bipolar")
    assert [c.name for c in bipolar] == ["A1-A2", "A2-A3", "A3-A4", "B1-B2", "B2-B3"]
    assert bipolar[0].weights == (1.0, -1.0) and bipolar[0].region == "ctx-lh-insula"
    assert bipolar[2].region == "" and bipolar[4].shaft == "B"       # outside MAPA's 74-region vocabulary
    laplacian = features.channels(labels, regions, "laplacian")
    assert [c.name for c in laplacian] == ["A2", "A3", "B2"] and laplacian[0].weights == (1.0, -0.5, -0.5)
    shaft_mean = features.channels(labels, regions, "shaft_mean")
    assert [c.name for c in shaft_mean] == ["A1", "A2", "A3", "A4", "B1", "B2", "B3"]
    assert sum(shaft_mean[0].weights) == pytest.approx(0.0)
    assert features.split_label("T1cIf12") == ("T1cIf", 12) and features.split_label("F10Fa10") == ("F10Fa", 10)
    assert len(features.REGION_VOCABULARY) == 74


def test_cache_shapes_and_dtypes(cell):
    _, root = cell
    store = features.Store(SESSION, "window", "bipolar", root)
    assert store.names == ["A1-A2", "A2-A3", "A3-A4", "B1-B2", "B2-B3"]
    positions = np.arange(5)
    for block, width in features.BLOCK_WIDTH.items():
        values = store.rows(block, positions)
        assert values.shape == (5, 5, width) and np.isfinite(values).all()
    assert store.rows("env5", positions).shape == (5, 5, 50) and store.rows("wave10", positions).shape == (5, 5, 10)
    assert store.rows("B4", positions).shape == (5, 5 * (3 * 4 // 2 + 2 * 3 // 2))
    assert features.Store(SESSION, "session", "bipolar", root).rows("B1", positions).shape == (5, 5, 50)
    assert "B1~phaserand" not in features.Store(SESSION, "session", "bipolar", root).meta["blocks"]
    assert features.Store(SESSION, "window", "laplacian", root).names == ["A2", "A3", "B2"]


def test_waveform_block_finds_a_phase_locked_response_and_magnitude_blocks_do_not(cell):
    cell, root = cell
    split = splits.get_cell(cell)
    assert len(split.rows["eval"]) >= 80
    scores = {block: _eval_auroc(features.kernels_and_dim(cell, block, "window", "bipolar", root=root)[0], split)
              for block in ("B1", "B2", "wave10", "base6", "B3", "env5", "B4", "B5")}
    assert scores["B1"] > 0.9 and scores["wave10"] > 0.9 and scores["B2"] > 0.8, scores
    assert scores["base6"] > 0.8, scores                              # the base keeps the 1-30 Hz stream
    for magnitude_only in ("B3", "env5"):                             # the response's polarity is the label
        assert 0.3 < scores[magnitude_only] < 0.7, scores
    assert np.isfinite(scores["B4"]) and np.isfinite(scores["B5"])
    session = _eval_auroc(features.kernels_and_dim(cell, "B1", "session", "bipolar", root=root)[0], split)
    assert session > 0.9


def test_phase_surrogate_and_shifted_sham_remove_it(cell):
    cell, root = cell
    split = splits.get_cell(cell)
    real = _eval_auroc(features.kernels_and_dim(cell, "B1", "window", "bipolar", root=root)[0], split)
    surrogate = _eval_auroc(controls.phase_randomized(cell, "B1", "bipolar", root=root)[0], split)
    surrogate_b2 = _eval_auroc(controls.phase_randomized(cell, "B2", "bipolar", root=root)[0], split)
    sham = _eval_auroc(controls.shifted_sham(cell, "B1", "window", "bipolar", root=root)[0], split)
    assert real > 0.9
    for control in (surrogate, surrogate_b2, sham):
        assert 0.3 < control < 0.7, (real, surrogate, surrogate_b2, sham)


def test_shifted_positions_move_every_row_at_least_30_s():
    rng = np.random.default_rng(0)
    onsets = np.unique((np.sort(rng.uniform(0, 600, 800)) * RATE).astype(np.int64))
    positions = rng.permutation(len(onsets))[:300]
    partner = controls.shifted_positions(onsets, positions, RATE)
    span = onsets[-1] - onsets[0] + 1
    forward = (onsets[partner] - onsets[positions]) % span
    assert forward.min() >= 30 * RATE and forward.max() < 40 * RATE
    assert (span - forward).min() >= 30 * RATE                         # and not close from the other side
    with pytest.raises(ValueError):
        controls.shifted_positions(onsets[onsets < 50 * RATE], np.arange(5), RATE)


def test_window_mode_ignores_every_sample_outside_the_window():
    rng = np.random.default_rng(3)
    x = rng.standard_normal((3, 40 * RATE)).astype(np.float32)
    starts = np.array([2 * RATE + 17, 9 * RATE + 400, 9 * RATE + 900, 30 * RATE + 5], dtype=np.int64)
    blocks = features.BLOCKS + features.SURROGATE_BLOCKS
    stats = features.window_stats(x, RATE, want_bands=True)
    inside = np.zeros(x.shape[1], dtype=bool)
    for start in starts:
        inside[start:start + RATE] = True
    noisy = x.copy()
    noisy[:, ~inside] = 50.0 * rng.standard_normal((3, int((~inside).sum()))).astype(np.float32)
    original = features.window_features(x, starts, RATE, blocks, stats, seed=5)
    replaced = features.window_features(noisy, starts, RATE, blocks, stats, seed=5)
    assert set(original) == set(blocks)
    for block in blocks:
        assert np.array_equal(original[block], replaced[block]), block
    # The same replacement does change session-mode features, so the check above has teeth.
    a = features.session_features(x, starts, RATE, ("base6", "B1"))
    b = features.session_features(noisy, starts, RATE, ("base6", "B1"))
    assert not np.allclose(a["B1"].astype(np.float32), b["B1"].astype(np.float32), atol=1e-2)


def test_session_mode_is_refused_outside_tier_p():
    features.check_mode("P", "session")
    for tier in ("V", "T"):
        features.check_mode(tier, "window")
        with pytest.raises(ValueError):
            features.check_mode(tier, "session")
    with pytest.raises(ValueError):
        features.kernels(Cell("T", "within", "btbank3_0", "btbank3_0", "speech", 0), "base6", "session", "bipolar")


def test_standardization_uses_training_rows_only(cell):
    cell, root = cell
    matrices = features.cell_matrices(cell, "base6", "window", "bipolar", root=root)
    np.testing.assert_allclose(matrices["train"].mean(axis=0), 0.0, atol=1e-9)
    np.testing.assert_allclose(matrices["train"].std(axis=0), 1.0, atol=1e-9)
    assert abs(matrices["eval"].std(axis=0).mean() - 1.0) > 1e-6       # not refit on the evaluation rows
    kernel = features.kernel_set(matrices, cross_gram=True)
    np.testing.assert_allclose(kernel.cross["eval"], matrices["eval"] @ matrices["train"].T)
    np.testing.assert_allclose(kernel.cross_gram["eval_select"], matrices["eval"] @ matrices["select"].T)


def test_cross_subject_pools_shared_regions_and_transfers(tmp_path):
    anchor, target = "btbank2_6", SESSION
    tables = {anchor: make_table(2048, 260, seed=1), target: make_table(RATE, 260, seed=2)}
    # The target has the same anatomy on differently named shafts; only regions can align the two.
    target_recording = make_recording(tables[target], RATE, seed=4)
    renamed = {"A1": "X1", "A2": "X2", "A3": "X3", "A4": "X4", "B1": "Y1", "B2": "Y2", "B3": "Y3"}
    target_recording.data = {renamed[k]: v for k, v in target_recording.data.items()}
    target_recording.regions = {renamed[k]: v for k, v in target_recording.regions.items()}
    target_recording.regions["Y1"] = "Right-Amygdala"                  # a region the anchor lacks
    recordings = {anchor: make_recording(tables[anchor], 2048, seed=3), target: target_recording}
    splits.set_provider(Provider(tables))
    features.set_recording_source(recordings.__getitem__)
    for session in (anchor, target):
        features.extract(session, {("window", "bipolar"): ("B1", "B4")}, root=tmp_path, workers=1)
    cell = Cell("P", "cross_subject", anchor, target, "speech", 0)
    kernel, dim = features.kernels_and_dim(cell, "B1", "window", "bipolar", root=tmp_path)
    # Shared: superiortemporal, middletemporal, Left-Hippocampus (pair Y2-Y3). Right-Amygdala (Y1-Y2) exists
    # only in the target and the anchor's second hippocampal pair has no counterpart to add a column.
    assert dim == 3 * 50
    assert _eval_auroc(kernel, splits.get_cell(cell)) > 0.85
    with pytest.raises(features.BlockUnavailable):
        features.kernels_and_dim(cell, "B4", "window", "bipolar", root=tmp_path)


def test_random_projection_control_matches_its_source_in_expectation(cell):
    cell, root = cell
    source, _ = features.kernels_and_dim(cell, "base6", "window", "bipolar", root=root, cross_gram=True)
    projector = controls.RandomProjector(source)
    assert projector.exact
    drawn = projector.kernel(4000, seed=1)
    scale = np.trace(source.train) / np.trace(drawn.train)
    assert 0.9 < scale < 1.1
    error = np.abs(drawn.cross["eval"] - source.cross["eval"]).mean() / np.abs(source.cross["eval"]).mean()
    assert error < 0.6                                                 # converges to the source kernel as dim grows
    small = projector.kernel(8, seed=1)
    assert np.linalg.matrix_rank(small.train) == 8
    approximate = controls.RandomProjector(ridge.KernelSet(train=source.train, cross=source.cross))
    assert not approximate.exact and approximate.kernel(8, seed=1).cross["select"].shape == source.cross["select"].shape

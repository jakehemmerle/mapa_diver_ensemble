import numpy as np
import pytest

from common import predictions, splits, stats
from ensemble import controls, fuse
from ensemble.tests import synthetic

REGIME = "cross_subject"


def test_normalize_is_monotone_and_inductive():
    rng = np.random.default_rng(0)
    reference = rng.normal(size=200)
    scores = np.sort(rng.normal(scale=3.0, size=500))          # reaches far outside the reference
    out = fuse.normalize(scores, reference)
    assert np.all(np.diff(out) > 0)
    # each score is mapped on its own: the result does not depend on the other scores
    assert fuse.normalize(scores[:7], reference) == pytest.approx(out[:7])
    inside = fuse.normalize(reference, reference)
    assert abs(inside.mean()) < 1e-9 and inside.std() == pytest.approx(1.0, abs=0.01)


def test_subsets():
    combos = fuse.subsets(["a", "b", "c", "d", "e"])
    assert len(combos) == 10 + 10 + 5 and all(2 <= len(c) <= 4 for c in combos)


def test_uniform_fusion_of_two_noisy_copies_beats_each(root):
    for member in ("copy1", "copy2"):
        synthetic.write_member(member, "P", REGIME, root, effect=1.0)
    name = fuse.fuse(["copy1", "copy2"], "a", "P", REGIME, root)
    assert name == "ens_a__copy1+copy2"
    for member in ("copy1", "copy2"):
        gain = stats.compare(name, member, "P", REGIME, root, descriptive=True)
        assert gain.delta > 0.03 and gain.subjects_up == gain.n_subjects
    meta = predictions.read(splits.list_cells("P", REGIME)[0], name, root).meta
    assert meta["weights"] == {"copy1": 0.5, "copy2": 0.5} and meta["normalization"] == "select_cdf"


def test_fusion_reads_no_select_or_eval_labels(root, monkeypatch):
    for member in ("copy1", "copy2"):
        synthetic.write_member(member, "P", REGIME, root)
    original = splits.CellSplit.labels

    def guarded(self, part):
        assert part == "train", f"fusion asked for {part} labels"
        return original(self, part)

    monkeypatch.setattr(splits.CellSplit, "labels", guarded)
    fuse.fuse(["copy1", "copy2"], "a", "P", REGIME, root)
    fuse.fuse(["copy1", "copy2"], "b", "P", REGIME, root)


def test_stacker_weights():
    rng = np.random.default_rng(3)
    y = np.arange(2000) % 2
    informative = y + rng.normal(size=2000)
    noise = rng.normal(size=2000)
    x = np.column_stack([fuse.normalize(informative, informative), fuse.normalize(noise, noise)])
    weights = fuse.fit_stacker(x, y)
    assert weights.sum() == pytest.approx(1.0) and weights.min() >= 0 and weights[0] > 0.8
    shuffled = fuse.fit_stacker(x, rng.permutation(y))
    assert np.abs(shuffled - 0.5).max() < 0.05
    assert fuse.fit_stacker(x, np.ones(2000)) == pytest.approx([0.5, 0.5])


def test_stacker_on_shuffled_labels_matches_uniform_mean(root):
    synthetic.write_member("good", "P", REGIME, root, effect=1.2)
    synthetic.write_member("fair", "P", REGIME, root, effect=0.6)
    synthetic.write_member("poor", "P", REGIME, root, effect=0.1)
    members = ["good", "fair", "poor"]
    report = controls.shuffled_stacker(members, "P", REGIME, root)
    assert report["passed"] and abs(report["delta"]) < controls.SHUFFLE_TOLERANCE
    cell = splits.list_cells("P", REGIME)[0]
    shuffled = predictions.read(cell, "ctl_shuf_b__fair+good+poor", root).meta
    assert shuffled["shuffled_training_labels"] == 0
    assert max(abs(w - 1 / 3) for w in shuffled["weights"].values()) < 0.08
    # with the real training labels the stacker moves weight to the better member and helps
    stacked = fuse.fuse(members, "b", "P", REGIME, root)
    real = predictions.read(cell, stacked, root).meta["weights"]
    assert real["good"] > real["fair"] > real["poor"]
    uniform = fuse.fused_name(members, "a")
    assert stats.compare(stacked, uniform, "P", REGIME, root, descriptive=True).delta > 0.003


def test_self_ensemble_of_identical_members_shows_no_gain(root):
    synthetic.write_member("mapa12", "P", REGIME, root, effect=1.0)
    replicas = controls.boot_members("mapa12")
    assert len(replicas) == 5
    for name in replicas:
        synthetic.write_member(name, "P", REGIME, root, effect=1.0, noise_of="mapa12")
    result = controls.self_ensemble("P", REGIME, root, members=replicas)
    assert result.candidate == "ctl_self__mapa12x5" and result.baseline == "mapa12"
    assert abs(result.delta) < 1e-12 and result.subjects_up == 0


def test_self_ensemble_of_independent_replicates_shows_the_variance_gain(root):
    synthetic.write_member("mapa12", "P", REGIME, root, effect=1.0)
    replicas = controls.boot_members("mapa12")
    for name in replicas:
        synthetic.write_member(name, "P", REGIME, root, effect=1.0)       # independent noise each
    assert controls.self_ensemble("P", REGIME, root, members=replicas).delta > 0.05


def test_leave_one_member_out_and_subject_concentration(root):
    synthetic.write_member("good", "P", REGIME, root, effect=1.2)
    synthetic.write_member("also", "P", REGIME, root, effect=1.2)
    synthetic.write_member("junk", "P", REGIME, root, effect=0.0)
    report = controls.leave_one_member_out(["good", "also", "junk"], "a", "P", REGIME, root)
    assert report["good"]["keep"] and report["also"]["keep"] and not report["junk"]["keep"]
    assert report["junk"]["without"] == "ens_a__also+good"
    comparison = stats.compare("ens_a__also+good", "good", "P", REGIME, root, descriptive=True)
    spread = controls.subject_concentration(comparison)
    assert spread["carried_by"] == [] and len(spread["leave_one_subject_out"]) == 7
    comparison.delta, comparison.leave_one_subject_out = 0.02, {"1": 0.02, "3": 0.004}
    assert controls.subject_concentration(comparison)["carried_by"] == ["3"]


class _FakeBackend:
    """Stands in for a trained member: 'training' is a mean difference, no GPU."""
    name = "fake"
    regimes = ("within", "cross_subject")
    early_stop = {"metric": "select AUROC", "patience": 10, "max_epochs": 40}
    autocast = "bfloat16 autocast"
    config = {"kind": "fake"}
    seconds_per_train_row = {"within": 0.001, "cross_subject": 0.001}
    vram_mib = 0

    def __init__(self):
        self.fits = 0

    def session_minutes(self, session):
        return 0.0

    def missing_inputs(self, session):
        return []

    def open_pair(self, train, target, regime):
        backend = self

        class Pair:
            meta = {"aligned_by": "test"}

            def train(self, onsets):
                return backend._features(train, onsets)

            def target(self, onsets):
                return backend._features(target, onsets)

        return Pair()

    def close_pair(self, train, target):
        pass

    @staticmethod
    def _features(session, onsets):
        rows = np.asarray(onsets) // (synthetic.SPACING * splits.native_rate(session) // 2048)
        return synthetic.features(12, session, "any")[rows]

    def fit(self, x, y, x_select, y_select, seed):
        self.fits += 1
        direction = x[y == 1].mean(0) - x[y == 0].mean(0) + np.random.default_rng(seed).normal(scale=0.05, size=x.shape[1])

        class Model:
            best_epoch, select_auroc = 3, None

            def predict(self, z):
                return z @ direction

        return Model()


def test_trained_member_export_design(root, monkeypatch):
    """Seed 0 carries out-of-fold training scores; seeds 1 and 2 only select and eval.
    No eval label is read, and the export is resumable."""
    from ensemble.members import trained
    original = splits.CellSplit.labels

    def guarded(self, part):
        assert part != "eval", "a member export asked for eval labels"
        return original(self, part)

    monkeypatch.setattr(splits.CellSplit, "labels", guarded)
    backend = _FakeBackend()
    plan = trained.plan(backend, "T", REGIME, root)
    assert plan["cells"] == 150 and plan["fits"] == 150 * 8 and plan["missing"] == []
    assert trained.export(backend, "T", REGIME, root) == 150
    assert backend.fits == 150 * 8                                   # 3 seeds + 5 out-of-fold blocks
    assert trained.export(backend, "T", REGIME, root) == 0
    cell = splits.list_cells("T", REGIME)[0]
    split = splits.get_cell(cell)
    first = predictions.read(cell, "fake", root)
    assert set(first.scores) == {"train", "select", "eval"}
    assert np.array_equal(first.rows["train"], split.rows["train"])
    assert first.meta["early_stop"]["patience"] == 10 and first.meta["early_stop"]["max_epochs"] == 40
    assert first.meta["autocast"] == "bfloat16 autocast"
    assert first.meta["selected_on_select"] == ["stopping epoch (AUROC)"] and first.meta["seed"] == 0
    for seed in (1, 2):
        replica = predictions.read(cell, f"fake@s{seed}", root)
        assert set(replica.scores) == {"select", "eval"} and replica.meta["seed"] == seed
    monkeypatch.undo()
    assert stats.cell_auroc("fake", cell, root, part="train") > 0.6      # out-of-fold scores carry signal
    assert trained.plan(backend, "T", "cross_session", root)["note"]

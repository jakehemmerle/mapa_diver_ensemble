import numpy as np
import pytest

torch = pytest.importorskip("torch")

from common import mapa_features, ridge, splits  # noqa: E402
from ensemble.mapa import features as feature_files  # noqa: E402
from ensemble.mapa import gram, kernel_job  # noqa: E402
from ensemble.members import mapa  # noqa: E402
from ensemble.tests import synthetic  # noqa: E402


def _reference(a, b):
    """MAPA's readout arithmetic on whole arrays: standardize on training rows, linear kernel."""
    a, b = a.astype(np.float64), b.astype(np.float64)
    mean, scale = a.mean(0), a.std(0)
    scale[scale == 0] = 1
    a, b = (a - mean) / scale, (b - mean) / scale
    return a @ a.T, b @ a.T, b @ b.T


def test_blocked_kernels_match_whole_array_arithmetic():
    rng = np.random.default_rng(0)
    columns, width = 5, 7
    train = rng.normal(size=(40, columns, width)).astype(np.float16)
    target = rng.normal(size=(30, columns, width)).astype(np.float16)
    train[:, 2, 3] = 1.0                                   # a constant feature: deviation 0 becomes 1
    jobs = [gram.Job("one", np.arange(0, 40, 2), np.arange(30), want_target=True),
            gram.Job("two", np.arange(5, 35), np.arange(3, 20))]
    gram.accumulate(jobs, columns, lambda i: (train[:, i], target[:, i]), torch.device("cpu"))
    for job in jobs:
        a = train[job.train_pos].reshape(len(job.train_pos), -1)
        b = target[job.target_pos].reshape(len(job.target_pos), -1)
        k_train, k_cross, k_target = _reference(a, b)
        assert job.out["train"].dtype == np.float64
        assert np.allclose(job.out["train"], k_train, rtol=1e-5, atol=1e-4)
        assert np.allclose(job.out["cross"], k_cross, rtol=1e-5, atol=1e-4)
        if job.want_target:
            assert np.allclose(job.out["target"], k_target, rtol=1e-5, atol=1e-4)
        else:
            assert "target" not in job.out


def test_passes_respect_the_accumulator_budget():
    jobs = [gram.Job(str(i), np.arange(100), np.arange(50)) for i in range(7)]
    each = jobs[0].nbytes()
    groups = gram.passes(jobs, 3 * each)
    assert [len(g) for g in groups] == [3, 3, 1]
    with pytest.raises(MemoryError):
        gram.passes(jobs, each - 1)


def test_kernels_round_trip_and_tier_v_is_served_by_the_evaluation_store(root):
    work = root.parent / "work"
    synthetic.write_kernels("T", "cross_subject", work)
    cell_t = splits.list_cells("T", "cross_subject")[0]
    cell_v = splits.Cell("V", *list(cell_t.__dict__.values())[1:])
    for cell in (cell_t, cell_v):
        split = splits.get_cell(cell)
        kernel = mapa_features.kernels(cell, 12)
        a = synthetic.features(12, cell.train_session, cell.task)[split.rows["train"]]
        for part in ("select", "eval"):
            b = synthetic.features(12, cell.target_session, cell.task)[split.rows[part]]
            k_train, k_cross, _ = _reference(a, b)
            assert np.allclose(kernel.train, k_train, rtol=1e-5, atol=1e-4)
            assert kernel.cross[part].shape == (len(split.rows[part]), len(split.rows["train"]))
            assert np.allclose(kernel.cross[part], k_cross, rtol=1e-5, atol=1e-4)
        assert kernel.cross_gram is None
    assert len(splits.get_cell(cell_v).rows["eval"]) < len(splits.get_cell(cell_t).rows["select"])
    frontend = mapa_features.kernels(cell_t, 0)            # the evaluation store keeps no target kernel...
    assert frontend.cross_gram is not None                 # ...unless the file has one (synthetic files do)
    assert set(frontend.cross_gram) == {"select", "eval", "eval_select"}


def test_missing_kernel_names_the_job(root):
    with pytest.raises(mapa_features.KernelMissing, match="mapa_kernels.sh"):
        mapa_features.kernels(splits.list_cells("P", "within")[0], 12)


def test_mapa_member_reads_no_eval_labels_and_reports_its_own_auroc(root, monkeypatch):
    from common import predictions, stats
    work = root.parent / "work"
    synthetic.write_kernels("T", "cross_subject", work)
    original = splits.CellSplit.labels

    def guarded(self, part):
        assert part != "eval", "a member export asked for eval labels"
        return original(self, part)

    monkeypatch.setattr(splits.CellSplit, "labels", guarded)
    assert mapa.export("mapa12", "T", "cross_subject", root) == 150
    assert mapa.export("mapa12", "T", "cross_subject", root) == 0          # resumable
    cell = splits.list_cells("T", "cross_subject")[0]
    pred = predictions.read(cell, "mapa12", root)
    assert set(pred.scores) == {"train", "select", "eval"}
    assert pred.meta["lam"] in ridge.LAM_MULTS and pred.meta["tap"] == 12
    assert stats.cell_auroc("mapa12", cell, root, part="select") == pytest.approx(pred.meta["select_auroc"], abs=1e-6)
    assert pred.meta["select_auroc"] > 0.6


def _simulate(ops):
    """Replay a plan; return the largest number of sessions holding per-contact features."""
    held, peak, done = {}, 0, []
    for op in ops:
        kind = op[0]
        if kind == "encode":
            held[op[1]] = op[2]
        elif kind == "delete":
            held.pop(op[1], None)
        elif kind == "delete_elec":
            held[op[1]] = False
        elif kind == "within":
            assert held.get(op[1]) is True, f"{op} without per-contact features"
            done.append(op)
        elif kind == "cross_session":
            assert held.get(op[1]) is True and held.get(op[2]) is True, f"{op} without both sessions"
            done.append(op)
        elif kind == "cross_subject":
            assert op[1] in held and op[2] in held, f"{op} without region features"
            done.append(op)
        peak = max(peak, sum(1 for v in held.values() if v))
    assert not held, f"features left on disk: {held}"
    return peak, done


@pytest.mark.parametrize("store", ["P", "E"])
def test_plan_holds_at_most_two_sessions_and_covers_every_pair(root, store):
    regimes = ["within", "cross_session", "cross_subject"]
    ops = kernel_job.build_plan(store, regimes)
    peak, done = _simulate(ops)
    assert peak == 2
    tier = kernel_job.TIER_OF_STORE[store]
    assert sorted(op[1] for op in done if op[0] == "within") == sorted(t for _, t in splits.pairs(tier, "within"))
    for regime in ("cross_session", "cross_subject"):
        assert sorted((op[1], op[2]) for op in done if op[0] == regime) == sorted(splits.pairs(tier, regime))
    assert len(done) == len(set(done)), "an operation is planned twice"
    encodes = [op[1] for op in ops if op[0] == "encode"]
    assert "btbank6_0" not in encodes
    sessions = splits.DEV_SESSIONS if store == "P" else splits.EVAL_SESSIONS
    assert set(encodes) == set(sessions)
    if store == "E":
        assert len(encodes) == 12                         # one encode per session
    else:
        assert len(encodes) <= len(sessions) + 7          # subject 2's five sessions need re-encodes


def test_plan_is_empty_once_kernels_exist_and_skips_done_regimes(root):
    work = root.parent / "work"
    synthetic.write_kernels("P", "cross_subject", work)
    assert kernel_job.build_plan("P", ["cross_subject"]) == []
    ops = kernel_job.build_plan("P", ["within", "cross_subject"])
    assert not [op for op in ops if op[0] == "cross_subject"]
    assert _simulate(ops)[0] == 1                           # within-session alone holds one session


def test_excluded_sessions_are_refused():
    from ensemble import paths
    with pytest.raises(ValueError, match="excluded"):
        paths.parse_session("btbank6_0")
    with pytest.raises(ValueError, match="excluded"):
        paths.parse_session("btbank7_100")


def test_feature_rows_are_matched_by_onset(root, tmp_path):
    import json
    folder = tmp_path / "features" / "btbank3_0"
    folder.mkdir(parents=True)
    onsets = np.array([100, 5000, 9000, 20000])
    rng = np.random.default_rng(0)
    contact_major = rng.normal(size=(3, 4, 6)).astype(np.float16)          # (contacts, windows, width)
    region_major = rng.normal(size=(4, 2, 6)).astype(np.float16)           # (windows, regions, width)
    np.save(folder / "enc12_elec.npy", contact_major)
    np.save(folder / "enc12.npy", region_major)
    (folder / "meta.json").write_text(json.dumps({"session": "btbank3_0", "onsets": onsets.tolist(),
                                                  "elec_labels": ["A1", "A2", "B1"], "present_parcels": [4, 74]}))
    store = feature_files.OwnFeatures(folder)
    assert store.positions([9000, 100]).tolist() == [2, 0]
    with pytest.raises(KeyError, match="no encoded window"):
        store.positions([101])
    assert np.array_equal(store.column("enc12_elec", 1), contact_major[1])
    assert np.array_equal(store.column("enc12", 1), region_major[:, 1])
    other = feature_files.OwnFeatures(folder)
    other.elec_labels, other.parcels = np.array(["B1", "A1", "C9"]), np.array([74, 4, 9])
    a, b, shared = feature_files.shared_columns(store, other, "enc12_elec")
    assert shared == ["A1", "B1"] and a == [0, 2] and b == [1, 0]
    a, b, shared = feature_files.shared_columns(store, other, "enc12")
    assert shared == [4] and a == [0] and b == [1]          # the unknown region 74 never aligns subjects


def test_kernel_store_must_be_on_its_filesystem(root, monkeypatch):
    """Without the pool mounted nothing is read or written, and nothing falls back to /."""
    work = root.parent / "work"
    cell = splits.list_cells("P", "cross_subject")[0]
    monkeypatch.setenv("NPEXP_KERNEL_STORE_FS", "zfs")
    with pytest.raises(mapa_features.KernelStoreUnavailable, match="does not exist"):
        mapa_features.kernel_file("P", "cross_subject", cell.key(), 12, work)
    (work / "kernels").mkdir(parents=True)                  # an empty mountpoint on the wrong filesystem
    with pytest.raises(mapa_features.KernelStoreUnavailable, match="not zfs"):
        mapa_features.kernels(cell, 12)
    with pytest.raises(mapa_features.KernelStoreUnavailable, match="not zfs"):
        mapa_features.save(work / "kernels" / "P" / "x" / "tap12" / "k.npz", train=np.eye(2), cross=np.ones((1, 2)),
                           train_rows=np.arange(2), target_rows=np.arange(1), meta={})
    assert not (work / "kernels" / "P").exists()
    with pytest.raises(SystemExit, match="KERNEL STORE"):
        monkeypatch.setattr("sys.argv", ["kernel_job", "--store", "P", "--regimes", "cross_subject", "--dry-run"])
        kernel_job.main()


def test_single_slot_mode_excludes_cross_session(root, monkeypatch):
    monkeypatch.setattr(kernel_job, "MAX_HELD", 1)
    with pytest.raises(SystemExit, match="two sessions at once"):
        kernel_job.build_plan("E", ["cross_session"])
    ops = kernel_job.build_plan("P", ["within", "cross_subject"])
    assert _simulate(ops)[0] == 1


def test_available_only_plans_what_is_on_disk(root, monkeypatch):
    data = root.parent / "data_dev"
    data.mkdir()
    monkeypatch.setenv("ROOT_DIR_BRAINTREEBANK", str(data))
    monkeypatch.setattr(kernel_job, "AVAILABLE_ONLY", True)
    assert kernel_job.build_plan("P", ["within", "cross_subject"]) == []
    for name in ("sub_2_trial006.h5", "sub_5_trial000.h5", "sub_1_trial000.h5"):
        (data / name).touch()
    ops = kernel_job.build_plan("P", ["within", "cross_subject"])
    assert sorted(op[1] for op in ops if op[0] == "within") == ["btbank1_0", "btbank2_6", "btbank5_0"]
    assert sorted(op[2] for op in ops if op[0] == "cross_subject") == ["btbank1_0", "btbank5_0"]
    _simulate(ops)


def test_kernel_store_refuses_the_root_mount(root, monkeypatch):
    """An ext4 kernel store on / (scratch not mounted, empty directory left behind) is refused."""
    store = root.parent / "work" / "kernels"
    store.mkdir(parents=True)
    monkeypatch.setenv("NPEXP_KERNEL_STORE_FS", "ext4")
    monkeypatch.setattr(mapa_features, "_mount", lambda path: ("/", "ext4"))
    with pytest.raises(mapa_features.KernelStoreUnavailable, match="not ext4 on its own disk"):
        mapa_features.require_store(root.parent / "work")
    monkeypatch.setattr(mapa_features, "_mount", lambda path: ("/scratch", "ext4"))
    assert mapa_features.require_store(root.parent / "work") == store


def test_scratch_refuses_an_unmounted_disk(root, monkeypatch, tmp_path):
    """NPEXP_SCRATCH on a disk that is not mounted resolves to /: refuse, never write there."""
    from ensemble import paths

    mount = (tmp_path / "scratch").resolve()
    mount.mkdir()
    monkeypatch.setenv("NPEXP_SCRATCH", str(mount / "neuroprobe-experiments"))
    monkeypatch.setattr(paths.os.path, "ismount", lambda p: str(p) == "/")
    with pytest.raises(SystemExit) as stop:
        paths.scratch()
    assert stop.value.code == 6
    with pytest.raises(SystemExit):
        paths.require_disk(1, "encoding")
    monkeypatch.setattr(paths.os.path, "ismount", lambda p: str(p) in ("/", str(mount)))
    assert paths.scratch() == mount / "neuroprobe-experiments"
    monkeypatch.delenv("NPEXP_SCRATCH")
    assert paths.scratch() == paths.work()


def test_cuda_out_of_memory_exits_cleanly():
    from ensemble import paths

    def boom():
        raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")

    with pytest.raises(SystemExit) as stop:
        paths.exit_on_cuda_oom(boom)
    assert stop.value.code == 4
    with pytest.raises(RuntimeError, match="other"):
        paths.exit_on_cuda_oom(lambda: (_ for _ in ()).throw(RuntimeError("other")))


def test_penalty_grid_agrees_with_the_readout(root):
    work = root.parent / "work"
    synthetic.write_kernels("T", "cross_subject", work)
    cell = splits.list_cells("T", "cross_subject")[3]
    split = splits.get_cell(cell)
    kernel = mapa_features.kernels(cell, 12)
    grid = kernel_job.penalty_grid([kernel], split.labels("train"), split.labels("select"))
    fit = ridge.fit_predict([kernel], split.labels("train"), split.labels("select"))
    assert len(grid) == 25 and grid[fit["lam_index"]] == pytest.approx(fit["select_auroc"], abs=1e-12)
    assert fit["lam_index"] == int(np.flatnonzero(grid == np.nanmax(grid))[0])

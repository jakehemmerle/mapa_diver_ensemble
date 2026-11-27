"""M5: Neuroprobe's CNN on Laplacian-referenced STFT magnitudes.

Adapted from the reproduction's ``cnn_benchmark.py``
(/Users/jake/ws/code/personal/spark/experiments/neuroprobe/cnn_benchmark.py,
sha256 2aa5f2d072e51cd958d4508a26f1206b3d0520efbb23669097a903afacbb388d), which matched
the published within-session macro to 0.004. Kept from it: ``laplacian-stft_abs`` with
nperseg 512, 75% overlap, Hann, 0-150 Hz; a StandardScaler fit on the training rows;
Neuroprobe's ``CNNClassifier`` (Adam 1e-4, batch 128, at most 100 epochs, patience 10 on
validation AUROC) with the official validation rows placed at its validation boundary;
cross-subject pooling of contacts to the Desikan-Killiany regions both subjects have.

Changed: windows come from ``common.splits`` onsets; scores are written per trial for
train (out-of-fold), select and eval and are the logit difference instead of the
softmax probability (same ranking, no saturation); eval labels are never read.
The reproduction's window caches for the 12 evaluation sessions are reused read-only.
Subject 9 is recorded at 1024 Hz: its one-second windows are resampled to 2048 samples.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

from ensemble import paths

STFT = {"stft": dict(nperseg=512, poverlap=0.75, window="hann", min_frequency=0, max_frequency=150)}


class Model:
    def __init__(self, classifier, scaler, best_epoch, select_auroc):
        self.classifier, self.scaler = classifier, scaler
        self.best_epoch, self.select_auroc = best_epoch, select_auroc

    def predict(self, x) -> np.ndarray:
        import torch
        net, device = self.classifier.model, self.classifier.device
        net.eval()
        out = []
        with torch.no_grad():
            for lo in range(0, len(x), 128):
                block = self.scaler.transform(x[lo:lo + 128].reshape(len(x[lo:lo + 128]), -1)).reshape(x[lo:lo + 128].shape)
                logits = net(torch.from_numpy(block.astype(np.float32)).to(device)).float()
                out.append((logits[:, 1] - logits[:, 0]).cpu().numpy())
        return np.concatenate(out)


class Pair:
    def __init__(self, train, target, regime):
        self._train, self._target = train, target
        self._pool = None
        self.meta = {"train_contacts": len(train["labels"]), "target_contacts": len(target["labels"])}
        if regime == "cross_subject":
            a, b = np.asarray(train["regions"]), np.asarray(target["regions"])
            common = np.intersect1d(a, b)
            if common.size == 0:
                raise ValueError("no shared regions")
            self._pool = (a, b, common)
            self.meta.update(aligned_by="Desikan-Killiany region mean", regions_used=int(common.size))
        elif train is not target and list(train["labels"]) != list(target["labels"]):
            raise ValueError("cross-session CNN expects the same Lite contact order in both sessions")

    def _take(self, store, onsets, regions):
        rows = np.array([store["lookup"][int(o)] for o in onsets])
        order = np.argsort(rows)
        x = np.empty((len(rows), *store["features"].shape[1:]), dtype=np.float32)
        x[order] = store["features"][rows[order]]
        if self._pool is not None:
            x = np.stack([x[:, regions == r].mean(axis=1) for r in self._pool[2]], axis=1)
        return x

    def train(self, onsets):
        return self._take(self._train, onsets, None if self._pool is None else self._pool[0])

    def target(self, onsets):
        return self._take(self._target, onsets, None if self._pool is None else self._pool[1])


class Backend:
    name = "cnn"
    regimes = ("within", "cross_session", "cross_subject")
    vram_mib = 3072          # the reproduction measured at most 2.6 GiB
    early_stop = {"metric": "select AUROC", "patience": 10, "max_epochs": 100, "min_delta": 1e-4}
    autocast = "disabled; float32 train and predict forward, weights and optimizer state"
    config = {"stft": STFT["stft"], "lr": 1e-4, "batch_size": 128, "max_epochs": 100, "patience": 10,
              "revision": paths.REVISIONS["neuroprobe"], "autocast": autocast}
    # reproduction file times: 30 within fits in 3.7 min; 15 cross-session fits in 5.5 min;
    # 15 cross-subject fits in 1.0 min (training rows 1,750 / 3,500 / 3,500)
    seconds_per_train_row = {"within": 7.3 / 1750, "cross_session": 22.0 / 3500, "cross_subject": 4.0 / 3500}

    def __init__(self):
        self._sessions: dict = {}

    def session_minutes(self, session: str) -> float:
        return 0.0 if self.cache_dir(session)[1] else 7.0     # window cache: about 7 min per session

    def cache_dir(self, session: str) -> tuple[Path, bool]:
        subject, trial = paths.parse_session(session)
        theirs = paths.repro() / "work/cnn" / f"s{subject}_t{trial}"
        if (theirs / "features.npy").is_file():
            return theirs, True
        ours = paths.scratch() / "cnn" / session
        return ours, (ours / "features.npy").is_file()

    def missing_inputs(self, session: str) -> list[Path]:
        needed = [] if self.cache_dir(session)[1] else [paths.recording(session)]
        try:
            needed.append(paths.source("neuroprobe") / "examples" / "eval_utils.py")
        except FileNotFoundError as error:
            needed.append(Path(str(error)))
        return [p for p in needed if not p.exists()]

    def _imports(self):
        folder = str(paths.source("neuroprobe") / "examples")
        if folder not in sys.path:
            sys.path.insert(0, folder)

    def prepare(self, session: str) -> Path:
        import scipy.signal
        import torch
        from neuroprobe import BrainTreebankSubject
        from common import splits
        self._imports()
        from eval_utils import preprocess_data

        out, ready = self.cache_dir(session)
        if ready:
            return out
        subject_id, trial = paths.parse_session(session)
        rate = splits.native_rate(session)
        source = splits.provider()
        onsets = np.unique(np.concatenate([source.table(session, task).onsets for task in splits.TASKS]))
        labels = list(splits.dataset(session, splits.TASKS[0])[0]["electrode_labels"])
        subject = BrainTreebankSubject(subject_id, cache=True, dtype=torch.float32)
        subject.set_electrode_subset(labels)
        subject.load_neural_data(trial)
        regions = subject.get_all_electrode_metadata().set_index("Electrode").loc[labels, "DesikanKilliany"]
        out.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=out.name + ".", dir=out.parent))
        np.save(staging / "starts.npy", onsets)
        (staging / "metadata.json").write_text(json.dumps(dict(labels=labels, regions=regions.tolist())))
        data = None
        for lo in range(0, len(onsets), 16):
            times = onsets[lo:lo + 16]
            raw = torch.stack([subject.get_all_electrode_data(trial, int(start), int(start) + rate) for start in times])
            if rate != 2048:
                raw = torch.from_numpy(scipy.signal.resample(raw.numpy(), 2048, axis=-1).astype(np.float32))
            features = preprocess_data(raw, labels, "laplacian-stft_abs", STFT).float().numpy()
            if data is None:
                paths.require_disk(len(onsets) * features[0].size * 4 / paths.GIB, f"CNN windows of {session}")
                data = np.lib.format.open_memmap(staging / "features.npy", mode="w+", dtype=np.float32,
                                                 shape=(len(onsets), *features.shape[1:]))
            data[lo:lo + len(times)] = features
        data.flush()
        os.replace(staging, out)
        print(f"[cnn] cached {len(onsets)} windows for {session}", flush=True)
        return out

    def session(self, session: str) -> dict:
        if session not in self._sessions:
            folder = self.prepare(session)
            starts = np.load(folder / "starts.npy")
            meta = json.loads((folder / "metadata.json").read_text())
            self._sessions[session] = {"features": np.load(folder / "features.npy", mmap_mode="r"),
                                       "lookup": {int(v): i for i, v in enumerate(starts)},
                                       "labels": meta["labels"], "regions": meta["regions"], "source": str(folder)}
        return self._sessions[session]

    def open_pair(self, train: str, target: str, regime: str) -> Pair:
        a = self.session(train)
        return Pair(a, a if target == train else self.session(target), regime)

    def close_pair(self, train: str, target: str) -> None:
        for name in [s for s in self._sessions if s not in (train, target)]:
            del self._sessions[name]

    def fit(self, x, y, x_select, y_select, seed: int) -> Model:
        import torch
        from sklearn.preprocessing import StandardScaler
        self._imports()
        from eval_utils import CNNClassifier

        scaler = StandardScaler().fit(x.reshape(len(x), -1))          # training rows only
        a = scaler.transform(x.reshape(len(x), -1)).reshape(x.shape)
        v = scaler.transform(x_select.reshape(len(x_select), -1)).reshape(x_select.shape)
        np.random.seed(seed % (2 ** 32))
        torch.manual_seed(seed)
        classifier = CNNClassifier(random_state=seed)
        # The upstream classifier takes the last val_size fraction as validation.
        # Concatenate the official sets so the exact split falls at that boundary.
        classifier.val_size = (len(v) + 0.25) / (len(a) + len(v))
        classifier.fit(np.concatenate((a, v)), np.concatenate((np.asarray(y, dtype=np.int64),
                                                               np.asarray(y_select, dtype=np.int64))))
        # The upstream classifier does not expose its best epoch or validation AUROC.
        return Model(classifier, scaler, None, None)

"""M4: DIVER-1 tiny, frozen encoder, linear head on flattened tokens.

Adapted from the reproduction's ``diver_pilot.py``
(/Users/jake/ws/code/personal/spark/experiments/neuroprobe/diver_pilot.py,
sha256 d4f3868b1b23e0529ca91e5c9b6f6782af538990e5dea1fde0aac6e2fba92fa2), which matched
the published within-session macro to 0.0015. Kept from it: 0.5 Hz high-pass and
60/120/180 Hz notch on the whole channel, clip to +-200, Laplacian reference and 500 Hz
downsampling per window through Neuroprobe's ``preprocess_data``, the /2 and /100
scalings, bfloat16 encoder features, and DIVER's own ``Trainer`` for the head (AdamW,
lr 0.002, weight decay 0.01, 40 epochs, cosine schedule, patience 10 on validation AUROC).

Changed: windows come from ``common.splits`` onsets; scores are written per trial for
train (out-of-fold), select and eval; the trainer's test loader receives the select rows,
so no eval label is read; scores are the float32 class-1 logit instead of its
bfloat16 sigmoid (same ranking up to rounding ties). Cross-session uses the contacts the
two sessions share by label (not part of DIVER's submission, which is within-session
only). There is no cross-subject protocol for a flattened per-contact head.
Subject 9 is recorded at 1024 Hz: its one-second windows are resampled to 2048 samples
before the unchanged preprocessing.
Within V/T confirmation only, the one-class-select amendment keeps the optimizer
and cosine schedule but exports the head after a fixed prefix of training passes.
"""
from __future__ import annotations

import contextlib
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ensemble import paths

TOKENS = 10 * 256       # ten 0.1 s patches of width 256 per contact


class Model:
    def __init__(self, head, best_epoch, select_auroc):
        self.head, self.best_epoch, self.select_auroc = head, best_epoch, select_auroc

    def predict(self, x) -> np.ndarray:
        import torch
        out = []
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):    # as in training
            for lo in range(0, len(x), 256):
                out.append(self.head(x[lo:lo + 256].cuda().float())[:, 1].float().cpu())
        return torch.cat(out).numpy()


class Pair:
    def __init__(self, train, target, shared):
        self._train, self._target = train, target
        self._columns = None
        if train is not target:
            a, b = list(train["labels"]), list(target["labels"])
            self._columns = ([a.index(label) for label in shared], [b.index(label) for label in shared])
        self.meta = {"train_contacts": len(train["labels"]), "target_contacts": len(target["labels"]),
                     "contacts_used": len(shared), "aligned_by": "contact label"}

    def _take(self, store, onsets, columns):
        lookup = store["lookup"]
        rows = [lookup[int(o)] for o in onsets]
        x = store["features"][rows]
        if columns is not None:
            x = x.reshape(len(rows), -1, TOKENS)[:, columns].reshape(len(rows), -1)
        return x

    def train(self, onsets):
        return self._take(self._train, onsets, None if self._columns is None else self._columns[0])

    def target(self, onsets):
        return self._take(self._target, onsets, None if self._columns is None else self._columns[1])


def _train_fixed_epochs(trainer, epochs: int) -> None:
    """Run the pinned Trainer's Neuroprobe head updates without an evaluator."""
    import torch

    for _ in range(epochs):
        trainer.model.train()
        trainer.model.backbone.eval()
        for x, y, data_info in trainer.data_loader["train"]:
            trainer.optimizer.zero_grad()
            x, y = x.cuda(), y.cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=trainer.params.use_amp):
                pred = trainer.model(x, data_info)
                loss = trainer.criterion(pred, torch.nn.functional.one_hot(y, num_classes=2).float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), trainer.clip_value)
            trainer.optimizer.step()
            trainer.optimizer_scheduler.step()

        # Even the normal unshuffled validation loader consumes a CPU RNG draw
        # when its iterator is created. Use an empty loader to preserve the next
        # train shuffle without loading or evaluating any select examples.
        iter(trainer.data_loader["val"])


class Backend:
    name = "diver"
    regimes = ("within", "cross_session")
    vram_mib = 6144          # estimate: the reproduction did not log DIVER's VRAM
    early_stop = {"metric": "select AUROC", "patience": 10, "max_epochs": 40}
    # DIVER's Trainer runs its forward passes under bf16 autocast (use_amp); predict matches it
    autocast = "bfloat16 autocast (encoder, head train and predict forward); float32 weights and optimizer state"
    config = {"lr": 0.002, "weight_decay": 0.01, "epochs": 40, "patience": 10, "batch_size": 32,
              "patch": 50, "rate": 500, "revision": paths.REVISIONS["DIVER-1"],
              "weights_sha256": paths.WEIGHTS["diver_ieeg.pt"], "autocast": autocast}
    # 8.8 s per fit with 1,750 training rows (reproduction: 30 fits of s3_t0 in 4.4 min)
    seconds_per_train_row = {"within": 8.8 / 1750, "cross_session": 8.8 / 1750}

    def __init__(self):
        self._sessions: dict = {}
        self._settings = None

    def session_minutes(self, session: str) -> float:
        return 5.0           # filter + encode; reproduction: 13-14 min per session including 30 fits

    def missing_inputs(self, session: str) -> list[Path]:
        needed = [paths.recording(session), paths.weights("diver_ieeg.pt")]
        try:
            needed += [paths.source("DIVER-1"), paths.source("neuroprobe") / "examples" / "eval_utils.py"]
        except FileNotFoundError as error:
            needed.append(Path(str(error)))
        return [p for p in needed if not p.exists()]

    def cache_file(self, session: str) -> Path:
        return paths.scratch() / "diver" / f"{session}.pt"

    # -- encoder features -------------------------------------------------------------
    def _imports(self):
        os.environ.setdefault("TQDM_DISABLE", "1")
        for folder in (paths.source("DIVER-1"), paths.source("neuroprobe") / "examples"):
            if str(folder) not in sys.path:
                sys.path.insert(0, str(folder))

    def _base_settings(self, model_dir: Path) -> SimpleNamespace:
        return SimpleNamespace(
            width=256, depth=12, patch_size=50, patch_sampling_rate=500, model_dir=str(model_dir),
            mup_weights=True, ft_mup=False, ft_config="flatten_linear",
            foundation_dir=str(paths.weights("diver_ieeg.pt")), deepspeed_pth_format=True,
            downstream_dataset="Neuroprobe", dataset_name="Neuroprobe", frozen=True,
            precompute_features=True, lr=0.002, weight_decay=0.01, epochs=40, use_amp=True,
            early_stop_criteria="val_auroc", early_stop_patience=10, label_smoothing=0.1)

    def encode(self, session: str) -> dict:
        import h5py
        import mne
        import scipy.signal
        import torch
        from concurrent.futures import ThreadPoolExecutor
        from neuroprobe import BrainTreebankSubject
        from common import splits
        self._imports()
        from eval_utils import preprocess_data
        from models.diver import DIVER
        from models.finetune_model import flatten_linear_finetune
        from utils.mup_utils import ensure_base_shapes

        subject_id, _ = paths.parse_session(session)
        rate = splits.native_rate(session)
        subject = BrainTreebankSubject(subject_id, cache=False)
        example = splits.dataset(session, splits.TASKS[0])[0]
        labels = list(example["electrode_labels"])
        info = {"xyz_id": example["electrode_coordinates"], "modality": "iEEG"}
        source = splits.provider()
        onsets = np.unique(np.concatenate([source.table(session, task).onsets for task in splits.TASKS]))
        samples = onsets[:, None] + np.arange(rate)[None, :]          # one second at the native rate
        folder = paths.scratch() / "diver"
        folder.mkdir(parents=True, exist_ok=True)
        paths.require_disk(len(labels) * len(onsets) * rate * 4 / paths.GIB + 8, f"DIVER windows of {session}")
        scratch = folder / f"{session}.voltage.npy"
        windows = np.lib.format.open_memmap(scratch, mode="w+", dtype="float32",
                                            shape=(len(labels), len(onsets), rate))

        def filter_channel(index):
            with h5py.File(paths.recording(session), "r") as h5:
                trace = h5["data"][subject.h5_neural_data_keys[labels[index]]][:].astype(np.float64)
            trace = mne.filter.filter_data(trace, rate, 0.5, None, verbose=False)
            trace = mne.filter.notch_filter(trace, rate, [60, 120, 180], verbose=False)
            np.clip(trace, -200, 200, out=trace)
            windows[index] = trace[samples]

        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(filter_channel, range(len(labels))))
        windows.flush()
        model_dir = folder / "head"
        model_dir.mkdir(exist_ok=True)
        ensure_base_shapes(lambda width, depth: DIVER(d_model=width, e_layer=depth, mup=True, patch_size=50),
                           "DIVER_iEEG_FINAL_model_patch50", 256, 12, str(model_dir))
        paths.verify_weights("diver_ieeg.pt")
        task = dict(consistent_channels=True, num_channels=len(labels), num_seconds=1,
                    num_targets=2, patch_sampling_rate=500, target_dynamics="discrete")
        model = flatten_linear_finetune(self._base_settings(model_dir), task).eval().cuda()
        features = torch.empty((len(onsets), len(labels) * TOKENS), dtype=torch.bfloat16)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for lo in range(0, len(onsets), 8):
                hi = min(len(onsets), lo + 8)
                block = np.ascontiguousarray(windows[:, lo:hi].transpose(1, 0, 2))
                if rate != 2048:
                    block = scipy.signal.resample(block, 2048, axis=-1).astype(np.float32)
                voltage = torch.from_numpy(block)
                # Upstream downsamples each example separately and divides by 2 in the
                # dataset and 100 in the trainer. Keep both operations in float32.
                x = torch.stack([preprocess_data(v, labels, "laplacian-downsample_500", None)
                                 for v in voltage]).float().div(2).div(100).cuda()
                output = model.backbone(x.reshape(hi - lo, len(labels), 10, 50),
                                        data_info_list=[info] * (hi - lo), use_mask=False,
                                        return_encoder_output=True)
                features[lo:hi] = model.feature_extraction_func(output).flatten(1).cpu()
        del windows, model
        scratch.unlink()
        torch.cuda.empty_cache()
        record = {"features": features, "onsets": onsets, "labels": labels, "native_rate": rate}
        temp = self.cache_file(session).with_suffix(".tmp")
        torch.save(record, temp)
        os.replace(temp, self.cache_file(session))
        print(f"[diver] encoded {session}: {tuple(features.shape)}", flush=True)
        return record

    def session(self, session: str) -> dict:
        import torch
        if session not in self._sessions:
            file = self.cache_file(session)
            # Cross-session pairs gather only the current cell's rows. Map the immutable
            # scratch tensors so clean pages can be reclaimed under the unit's RAM cap.
            record = torch.load(file, mmap=True, weights_only=False) if file.is_file() else self.encode(session)
            record["lookup"] = {int(o): i for i, o in enumerate(record["onsets"])}
            self._sessions[session] = record
        return self._sessions[session]

    def open_pair(self, train: str, target: str, regime: str) -> Pair:
        a = self.session(train)
        b = a if target == train else self.session(target)
        shared = [label for label in a["labels"] if label in set(b["labels"])]
        if not shared:
            raise ValueError(f"{train} and {target} share no contacts")
        return Pair(a, b, shared)

    def close_pair(self, train: str, target: str) -> None:
        """Keep at most the two sessions just used in memory; cached files stay on disk
        until the launcher's cleanup."""
        for name in [s for s in self._sessions if s not in (train, target)]:
            del self._sessions[name]

    # -- head -------------------------------------------------------------------------
    def fit(self, x, y, x_select, y_select, seed: int, *, fixed_epochs: int | None = None) -> Model:
        import torch
        from torch.utils.data import DataLoader
        self._imports()
        from finetune_trainer import Trainer

        class Head(torch.nn.Module):
            """The linear head of DIVER's ``flatten_linear_finetune`` on precomputed tokens."""

            def __init__(self, width):
                super().__init__()
                self.backbone = torch.nn.Identity()
                self.ft_core_model = torch.nn.Linear(width, 2)

            def forward(self, features, data_info_list=None):
                return self.ft_core_model(features.flatten(1))

        torch.manual_seed(seed)
        np.random.seed(seed % (2 ** 32))
        head = Head(x.shape[1])
        fit_rows = [(x[i], int(y[i]), {}) for i in range(len(y))]
        select_rows = ()
        if fixed_epochs is None:
            select_rows = [(x_select[i], int(y_select[i]), {}) for i in range(len(y_select))]
        loaders = {"train": DataLoader(fit_rows, batch_size=32, shuffle=True),
                   "val": DataLoader(select_rows, batch_size=32, shuffle=False),
                   # On the normal path, DIVER scores the select rows again after
                   # training. Eval labels never enter either path.
                   "test": DataLoader(select_rows, batch_size=32, shuffle=False)}
        model_dir = paths.scratch() / "diver" / "head"
        model_dir.mkdir(parents=True, exist_ok=True)
        task = dict(consistent_channels=True, num_channels=x.shape[1] // TOKENS, num_seconds=1,
                    num_targets=2, patch_sampling_rate=500, target_dynamics="discrete")
        # The upstream trainer saves best_state.pth after restoring the best weights in
        # memory. It is not an export: do not let parallel shards write the same file.
        with tempfile.TemporaryDirectory(dir=model_dir) as scratch, open(os.devnull, "w") as sink:
            trainer = Trainer(self._base_settings(Path(scratch)), loaders, head, downstream_task_info=task, clip_value=1)
            if fixed_epochs is not None:
                # Keep params.epochs unchanged: the scheduler spans all 40
                # planned passes even when exporting an earlier prefix.
                _train_fixed_epochs(trainer, fixed_epochs)
                head.eval()
                return Model(head, None, None)
            with contextlib.redirect_stdout(sink):
                metrics = trainer.train()
        head.eval()
        return Model(head, int(metrics.get("best_epoch", -1)), float(metrics.get("val_auroc", float("nan"))))

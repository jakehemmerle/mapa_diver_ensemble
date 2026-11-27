"""Classical feature blocks, their per-session cache, and per-cell kernels.

DECLARED DEFAULTS (fixed before any run; see ``DEFAULTS`` below, hashed into every
prediction file and ledger row). The only specification of Ridge 6-band is PR #20's
``metadata.json``; what it states is marked [PR20], everything else is our choice
and is NOT tuned against published numbers (plan, honesty rule 8).

Base block ``base6`` (the Ridge 6-band reimplementation)
  * [PR20] Lite electrodes as adjacent bipolar pairs on the same shaft.
    Ours: both contacts must be in the benchmark's Lite set after its corrupted-
    electrode filter; shaft = label minus trailing digits; channel = x[n] - x[n+1].
  * Ours: every signal is first decimated to a 512 Hz working rate with a polyphase FIR
    (scipy ``resample_poly``; x4 from 2048 Hz, x2 from 1024 Hz); all filters run at 512 Hz.
  * [PR20] line noise removed. Ours: IIR notches at 60/120/180 Hz, Q = 30, zero phase.
  * [PR20] six streams at 200 Hz: 1-30 Hz waveform; theta, alpha, beta, low-gamma
    (30-70 Hz) log envelopes; 70-170 Hz high-gamma log amplitude.
    Ours: theta 4-8, alpha 8-13, beta 13-30 Hz; Butterworth order 4 band-pass as SOS,
    zero phase (forward-backward); envelope = |Hilbert analytic signal|; envelope and
    waveform resampled to 200 Hz with a polyphase FIR (scipy ``resample_poly``), then
    log(max(envelope, 1e-9)).
  * [PR20] each stream normalized with median/MAD per channel over the session.
    Ours: (x - median) / (1.4826 MAD), clipped to +-50.
  * [PR20] every stream averaged in ten 100 ms bins over the 1 s after onset.
  * [PR20] features z-scored on the training set; ridge penalty chosen on validation
    AUROC. Ours: MAPA's 25-point grid, smallest penalty on ties (``common.ridge``).
  * [PR20] cross-subject: features of each channel averaged within its DKT region.
    Ours: column "DKT" of depth-wm.csv restricted to MAPA's 74-region vocabulary,
    hemispheres distinct; a bipolar pair takes the region of its lower-numbered
    contact; only regions present in both anchor and target are used.
  * Ours: cross-session uses the channels present in both sessions, matched by name.

Window discipline (plan, honesty rule 4)
  * ``window_mode="window"`` (the only mode allowed on Tiers V and T): the one-second
    window [onset, onset + 1 s) is cut from the re-referenced recording first; every
    filter runs on those samples alone, padded by odd reflection of the window itself
    (pad length = window - 1); the Hilbert transform and the resampler see only the
    window. Normalization statistics are the per-channel median/MAD of the same
    in-window pipeline applied to one-second tiles taken every 4 s across the whole
    session (label-free session statistics, allowed in both modes).
  * ``window_mode="session"`` (Tier P measurement only): filters run on the continuous
    recording, streams are cut afterwards, statistics come from the continuous streams.
  * Subject 9 is recorded at 1024 Hz; the benchmark still slices 2048 samples (2 s).
    We use the first second (1024 samples).

Added blocks (plan, "Classical track"); all per re-referenced channel unless noted
  * B1  1-30 Hz normalized waveform, 50 bins of 20 ms.
  * B2  cos and sin of theta and alpha Hilbert phase, ten 100 ms bins each.
  * B3  70-170 Hz log amplitude at 256 Hz, normalized, 32 bins of 31.25 ms.
  * B4  per shaft and band (theta..high-gamma): covariance between the shaft's channels
    over the window (each channel scaled by its session MAD), shrunk 10% towards the
    scaled identity, mapped to the tangent space at the arithmetic mean covariance of
    the TRAINING rows (so nothing is fit on target-session data). Per shaft, hence not
    defined for cross-subject.
  * B5  theta-phase to high-gamma-amplitude coupling: real part, imaginary part and
    length of sum(A exp(i phi)) / sum(A) over the window.
  * Re-referencing ("bipolar" | "laplacian" | "shaft_mean") is a factor, not a block.
  * ``env5`` and ``wave10`` are views of base6 (magnitude-only and waveform-only parts).
  * ``B1~phaserand`` / ``B2~phaserand``: the same blocks computed from windows whose
    Fourier phases were randomized (classical/controls.py).

Features are cached per session (``<cache>/features/<session>/<mode>__<ref>/``), kernels are
built per cell from that cache (optionally cached with NPEXP_KERNEL_CACHE=1). ``<cache>`` is
``$NPEXP_CLASSICAL_CACHE`` when set, else ``$NPEXP_ROOT/work/classical``.
"""
from __future__ import annotations

import functools
import json
import math
import multiprocessing
import os
import re
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Callable, Protocol

import numpy as np
import scipy.fft
import scipy.signal

import common
from common import splits
from common.ridge import KernelSet
from common.splits import Cell, MissingInput

DEFAULTS = {
    "electrodes": "neuroprobe-lite-after-corrupted-filter",
    "reference": "bipolar",
    "window_mode": "window",
    "notch_hz": [60.0, 120.0, 180.0],
    "notch_q": 30.0,
    "bands_hz": {"wave": [1.0, 30.0], "theta": [4.0, 8.0], "alpha": [8.0, 13.0], "beta": [13.0, 30.0],
                 "lgamma": [30.0, 70.0], "hgamma": [70.0, 170.0]},
    "band_filter": "butterworth-order4-sos-zero-phase",
    "window_padding": "odd-reflection-of-the-window, padlen=window-1",
    "envelope": "abs-hilbert",
    "log_floor": 1e-9,
    "work_hz": 512,
    "stream_hz": 200,
    "fine_hz": 256,
    "resampler": "scipy.signal.resample_poly(padtype='line')",
    "normalization": "median/1.4826*MAD per channel and stream over the session",
    "norm_tile_stride_s": 4.0,
    "norm_clip": 50.0,
    "base_bins": 10,
    "b1_bins": 50,
    "b3_bins": 32,
    "b4_bands": ["theta", "alpha", "beta", "lgamma", "hgamma"],
    "b4_shrinkage": 0.1,
    "b4_reference": "arithmetic mean covariance of training rows",
    "region_column": "DKT",
    "region_vocabulary": "MAPA V14_DKT_REGION_LABELS (62 cortical + 12 subcortical)",
    "bipolar_region": "lower-numbered contact",
    "penalty_grid": "MAPA LAM_MULTS = logspace(-4, 4, 25), ties -> smallest",
    "feature_dtype": "float16 (B4 covariances float32)",
}
BANDS = {name: tuple(edges) for name, edges in DEFAULTS["bands_hz"].items()}
ENVELOPE_BANDS = ("theta", "alpha", "beta", "lgamma", "hgamma")
B4_BANDS = tuple(DEFAULTS["b4_bands"])
STREAM_HZ, FINE_HZ, WORK_HZ = DEFAULTS["stream_hz"], DEFAULTS["fine_hz"], DEFAULTS["work_hz"]
BLOCKS = ("base6", "B1", "B2", "B3", "B4", "B5")
SURROGATE_BLOCKS = ("B1~phaserand", "B2~phaserand")
VIEWS = {"env5": ("base6", slice(10, 60)), "wave10": ("base6", slice(0, 10))}
REFERENCES = ("bipolar", "laplacian", "shaft_mean")
WINDOW_MODES = ("window", "session")
BLOCK_WIDTH = {"base6": 60, "B1": 50, "B2": 40, "B3": 32, "B5": 3, "B1~phaserand": 50, "B2~phaserand": 40}
NORMALIZED_STREAMS = ("wave", "hg256") + tuple(f"env_{band}" for band in ENVELOPE_BANDS)
SURROGATE_SEED = 20261125

_DKT_CORTICAL = (
    "caudalanteriorcingulate", "caudalmiddlefrontal", "cuneus", "entorhinal", "fusiform", "inferiorparietal",
    "inferiortemporal", "insula", "isthmuscingulate", "lateraloccipital", "lateralorbitofrontal", "lingual",
    "medialorbitofrontal", "middletemporal", "paracentral", "parahippocampal", "parsopercularis", "parsorbitalis",
    "parstriangularis", "pericalcarine", "postcentral", "posteriorcingulate", "precentral", "precuneus",
    "rostralanteriorcingulate", "rostralmiddlefrontal", "superiorfrontal", "superiorparietal", "superiortemporal",
    "supramarginal", "transversetemporal",
)
_ASEG = ("Hippocampus", "Amygdala", "Caudate", "Putamen", "Pallidum", "Thalamus-Proper")
# Same 74 labels as MAPA's V14_DKT_REGION_LABELS (mapa/data/anatomy.py at dc582b2).
REGION_VOCABULARY = frozenset([f"ctx-{h}-{b}" for h in ("lh", "rh") for b in _DKT_CORTICAL]
                              + [f"{side}-{b}" for side in ("Left", "Right") for b in _ASEG])


class BlockUnavailable(ValueError):
    """The block is not defined for this regime (B4 across subjects)."""


def defaults_hash() -> str:
    return common.config_hash(DEFAULTS)


# --------------------------------------------------------------------------- recordings and montage

@dataclass(frozen=True)
class Channel:
    name: str
    shaft: str
    region: str                    # DKT label of the channel, "" when outside the vocabulary
    contacts: tuple[str, ...]
    weights: tuple[float, ...]


class Recording(Protocol):
    rate: int
    n_samples: int
    labels: tuple[str, ...]        # Lite contacts in benchmark order
    regions: dict[str, str]        # contact -> DKT label

    def read(self, labels: tuple[str, ...]) -> np.ndarray: ...   # (len(labels), n_samples) float32


@dataclass(frozen=True)
class H5Recording:
    path: str
    keys: tuple[tuple[str, str], ...]   # (contact label, h5 dataset key)
    rate: int
    n_samples: int
    labels: tuple[str, ...]
    regions_items: tuple[tuple[str, str], ...]

    @property
    def regions(self) -> dict[str, str]:
        return dict(self.regions_items)

    def read(self, labels: tuple[str, ...]) -> np.ndarray:
        import h5py
        keys = dict(self.keys)
        out = np.empty((len(labels), self.n_samples), dtype=np.float32)
        with h5py.File(self.path, "r") as handle:
            for i, label in enumerate(labels):
                handle["data"][keys[label]].read_direct(out[i])
        return out


@dataclass
class ArrayRecording:
    """In-memory recording for tests."""
    rate: int
    data: dict[str, np.ndarray]
    regions: dict[str, str]

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(self.data)

    @property
    def n_samples(self) -> int:
        return len(next(iter(self.data.values())))

    def read(self, labels: tuple[str, ...]) -> np.ndarray:
        return np.stack([self.data[label] for label in labels]).astype(np.float32)


def _h5_recording(session: str) -> H5Recording:
    import h5py
    source = splits.provider()
    if not isinstance(source, splits.NeuroprobeProvider):
        raise RuntimeError("no recording source registered for a non-Neuroprobe split provider")
    source.require_recording(session)
    subject_id, trial = splits.parse_session(session)
    subject = source.subject(subject_id)
    from neuroprobe.config import NEUROPROBE_LITE_ELECTRODES
    labels = tuple(e for e in NEUROPROBE_LITE_ELECTRODES[subject.subject_identifier] if e in subject.electrode_labels)
    localization = subject.localization_data
    column = DEFAULTS["region_column"]
    regions = {str(row["Electrode"]): str(row[column]) for _, row in localization.iterrows()}
    path = common.data_root() / f"sub_{subject_id}_trial{trial:03d}.h5"
    with h5py.File(path, "r") as handle:
        n_samples = int(handle["data"][subject.h5_neural_data_keys[labels[0]]].shape[0])
    return H5Recording(path=str(path), keys=tuple((label, subject.h5_neural_data_keys[label]) for label in labels),
                       rate=splits.native_rate(session), n_samples=n_samples, labels=labels,
                       regions_items=tuple((label, regions.get(label, "")) for label in labels))


_recording_source: Callable[[str], Recording] = _h5_recording


def set_recording_source(source: Callable[[str], Recording] | None) -> None:
    global _recording_source
    _recording_source = source or _h5_recording


def recording(session: str) -> Recording:
    return _recording_source(session)


_LABEL = re.compile(r"^(.*?)(\d+)$")


def split_label(label: str) -> tuple[str, int]:
    match = _LABEL.match(label)
    if not match:
        raise ValueError(f"contact label without a trailing number: {label!r}")
    return match.group(1), int(match.group(2))


def _region(regions: dict[str, str], contact: str) -> str:
    label = regions.get(contact, "")
    return label if label in REGION_VOCABULARY else ""


def channels(labels: tuple[str, ...], regions: dict[str, str], reference: str) -> list[Channel]:
    """Re-referenced channels, grouped by shaft in order of first appearance, contacts ascending."""
    shafts: dict[str, dict[int, str]] = {}
    for label in labels:
        stem, number = split_label(label)
        shafts.setdefault(stem, {})[number] = label
    out: list[Channel] = []
    for stem, contacts in shafts.items():
        numbers = sorted(contacts)
        if reference == "bipolar":
            for n in numbers:
                if n + 1 in contacts:
                    out.append(Channel(f"{contacts[n]}-{contacts[n + 1]}", stem, _region(regions, contacts[n]),
                                       (contacts[n], contacts[n + 1]), (1.0, -1.0)))
        elif reference == "laplacian":
            for n in numbers:
                if n - 1 in contacts and n + 1 in contacts:
                    out.append(Channel(contacts[n], stem, _region(regions, contacts[n]),
                                       (contacts[n], contacts[n - 1], contacts[n + 1]), (1.0, -0.5, -0.5)))
        elif reference == "shaft_mean":
            if len(numbers) >= 2:
                members = tuple(contacts[n] for n in numbers)
                for n in numbers:
                    weights = tuple((1.0 if m == contacts[n] else 0.0) - 1.0 / len(members) for m in members)
                    out.append(Channel(contacts[n], stem, _region(regions, contacts[n]), members, weights))
        else:
            raise ValueError(f"unknown reference {reference!r}; choose from {REFERENCES}")
    return out


# --------------------------------------------------------------------------- signal processing

@functools.lru_cache(maxsize=None)
def _notch_sos(rate: int) -> np.ndarray:
    return np.vstack([scipy.signal.tf2sos(*scipy.signal.iirnotch(f, DEFAULTS["notch_q"], fs=rate))
                      for f in DEFAULTS["notch_hz"]])


@functools.lru_cache(maxsize=None)
def _band_sos(band: str, rate: int) -> np.ndarray:
    return scipy.signal.butter(4, BANDS[band], btype="bandpass", fs=rate, output="sos")


def _filter(sos: np.ndarray, x: np.ndarray, in_window: bool) -> np.ndarray:
    if in_window:
        return scipy.signal.sosfiltfilt(sos, x, axis=-1, padtype="odd", padlen=x.shape[-1] - 1)
    return scipy.signal.sosfiltfilt(sos, x, axis=-1)


def _analytic(x: np.ndarray, in_window: bool) -> np.ndarray:
    if in_window:
        return scipy.signal.hilbert(x, axis=-1)
    n = x.shape[-1]
    return scipy.signal.hilbert(x, N=scipy.fft.next_fast_len(n), axis=-1)[..., :n]


def _resample(x: np.ndarray, rate: int, target: int) -> np.ndarray:
    ratio = Fraction(target, rate)
    return scipy.signal.resample_poly(x, ratio.numerator, ratio.denominator, axis=-1, padtype="line")


def streams(x: np.ndarray, rate: int, in_window: bool, keep_bands: bool = False) -> dict[str, np.ndarray]:
    """All intermediate streams of ``x`` (..., n_samples at ``rate``), processed along the last axis.

    With ``in_window`` every operation, the decimation included, uses only the samples of ``x`` itself.
    """
    floor = DEFAULTS["log_floor"]
    work = np.asarray(x, dtype=np.float64)
    if rate != WORK_HZ:
        work = _resample(work, rate, WORK_HZ)
    rate = WORK_HZ
    notched = _filter(_notch_sos(rate), work, in_window)
    out: dict[str, np.ndarray] = {}
    out["wave"] = _resample(_filter(_band_sos("wave", rate), notched, in_window), rate, STREAM_HZ).astype(np.float32)
    for band in ENVELOPE_BANDS:
        passed = _filter(_band_sos(band, rate), notched, in_window)
        if keep_bands and band in B4_BANDS:
            out[f"band_{band}"] = passed.astype(np.float32)
        analytic = _analytic(passed, in_window)
        amplitude = np.abs(analytic)
        coarse = _resample(amplitude, rate, STREAM_HZ)
        out[f"env_{band}"] = np.log(np.maximum(coarse, floor)).astype(np.float32)
        if band == "hgamma":
            out["amp_hgamma"] = np.maximum(coarse, 0.0).astype(np.float32)
            out["hg256"] = np.log(np.maximum(_resample(amplitude, rate, FINE_HZ), floor)).astype(np.float32)
        if band in ("theta", "alpha"):
            unit = analytic / np.maximum(amplitude, 1e-300)
            out[f"cos_{band}"] = _resample(unit.real, rate, STREAM_HZ).astype(np.float32)
            out[f"sin_{band}"] = _resample(unit.imag, rate, STREAM_HZ).astype(np.float32)
    return out


def robust_stats(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Median and 1.4826 MAD along all axes but the first (channels)."""
    flat = np.asarray(values).reshape(values.shape[0], -1)
    median = np.median(flat, axis=1)
    scale = 1.4826 * np.median(np.abs(flat - median[:, None]), axis=1)
    scale[~(scale > 0)] = 1.0
    return median.astype(np.float64), scale.astype(np.float64)


def phase_randomize(windows: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Surrogate with each window's amplitude spectrum and independent uniform phases."""
    spectrum = np.fft.rfft(windows, axis=-1)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=spectrum.shape)
    surrogate = np.abs(spectrum) * np.exp(1j * phases)
    surrogate[..., 0] = spectrum[..., 0]
    if windows.shape[-1] % 2 == 0:
        surrogate[..., -1] = spectrum[..., -1]
    return np.fft.irfft(surrogate, n=windows.shape[-1], axis=-1)


def _bins(values: np.ndarray, count: int) -> np.ndarray:
    return values.reshape(*values.shape[:-1], count, -1).mean(axis=-1)


def assemble(cut: dict[str, np.ndarray], stats: dict[str, tuple[np.ndarray, np.ndarray]],
             blocks: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Block features (n_windows, n_channels, width) from cut streams (n_channels, n_windows, samples)."""
    clip = DEFAULTS["norm_clip"]

    def normalized(name: str) -> np.ndarray:
        median, scale = stats[name]
        return np.clip((cut[name] - median[:, None, None]) / scale[:, None, None], -clip, clip)

    out: dict[str, np.ndarray] = {}
    for block in blocks:
        base = block.split("~")[0]
        if base == "base6":
            parts = [_bins(normalized("wave"), DEFAULTS["base_bins"])]
            parts += [_bins(normalized(f"env_{band}"), DEFAULTS["base_bins"]) for band in ENVELOPE_BANDS]
        elif base == "B1":
            parts = [_bins(normalized("wave"), DEFAULTS["b1_bins"])]
        elif base == "B2":
            parts = [_bins(cut[f"{fn}_{band}"], 10) for band in ("theta", "alpha") for fn in ("cos", "sin")]
        elif base == "B3":
            parts = [_bins(normalized("hg256"), DEFAULTS["b3_bins"])]
        elif base == "B5":
            amplitude = cut["amp_hgamma"].astype(np.float64)
            total = amplitude.sum(axis=-1) + 1e-300
            real = (amplitude * cut["cos_theta"]).sum(axis=-1) / total
            imaginary = (amplitude * cut["sin_theta"]).sum(axis=-1) / total
            parts = [np.stack([real, imaginary, np.hypot(real, imaginary)], axis=-1)]
        else:
            continue
        out[block] = np.concatenate(parts, axis=-1).transpose(1, 0, 2).astype(np.float16)
    return out


def _covariance(signals: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Shrunk covariance upper triangles (n_windows, tri) of one band's (C, W, T) signals."""
    gamma = DEFAULTS["b4_shrinkage"]
    z = signals.astype(np.float64) / scale[:, None, None]
    cov = np.einsum("cwt,dwt->wcd", z, z, optimize=True) / z.shape[-1]
    n = cov.shape[-1]
    mean_variance = np.trace(cov, axis1=1, axis2=2) / n
    cov = (1.0 - gamma) * cov + gamma * mean_variance[:, None, None] * np.eye(n)
    rows, cols = np.triu_indices(n)
    return cov[:, rows, cols].astype(np.float32)


def _gather(x: np.ndarray, starts: np.ndarray, length: int) -> np.ndarray:
    return x[..., starts[:, None] + np.arange(length)[None, :]]


def _check_starts(starts: np.ndarray, length: int, n_samples: int) -> None:
    if len(starts) and (starts.min() < 0 or starts.max() + length > n_samples):
        raise ValueError("a window runs past the recording")


def window_chunks(x: np.ndarray, starts: np.ndarray, rate: int, keep_bands: bool = False,
                  rng: np.random.Generator | None = None, chunk: int = 128):
    """Yield in-window streams (dict of (C, W, samples)) for consecutive chunks of windows of ``x`` (C, N).

    Depends on no sample outside [start, start + 1 s) of any window. ``rng`` switches on the
    phase-randomized surrogate.
    """
    length = int(round(splits.WINDOW_S * rate))
    _check_starts(starts, length, x.shape[-1])
    for low in range(0, len(starts), chunk):
        windows = _gather(x, starts[low:low + chunk], length).astype(np.float64)
        if rng is not None:
            windows = phase_randomize(windows, rng)
        yield streams(windows, rate, in_window=True, keep_bands=keep_bands)


def window_stats(x: np.ndarray, rate: int, want_bands: bool) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Session statistics for window mode: in-window streams of one-second tiles every 4 s."""
    length = int(round(splits.WINDOW_S * rate))
    stride = int(round(DEFAULTS["norm_tile_stride_s"] * rate))
    tiles = np.arange(0, x.shape[-1] - length + 1, stride, dtype=np.int64)
    names = NORMALIZED_STREAMS + (tuple(f"band_{band}" for band in B4_BANDS) if want_bands else ())
    pieces: dict[str, list[np.ndarray]] = {name: [] for name in names}
    for cut in window_chunks(x, tiles, rate, keep_bands=want_bands):
        for name in names:
            pieces[name].append(cut[name][..., ::2] if name.startswith("band_") else cut[name])
    return {name: robust_stats(np.concatenate(parts, axis=1)) for name, parts in pieces.items()}


def window_features(x: np.ndarray, starts: np.ndarray, rate: int, blocks: tuple[str, ...],
                    stats: dict[str, tuple[np.ndarray, np.ndarray]], seed: int | None = None) -> dict[str, np.ndarray]:
    """Window-mode block features of one shaft given fixed normalization statistics.

    For fixed ``stats`` the result is a function of the windows' own samples only.
    """
    real = tuple(b for b in blocks if "~" not in b)
    surrogate = tuple(b for b in blocks if b.endswith("~phaserand"))
    pieces: dict[str, list[np.ndarray]] = {}
    if real:
        for cut in window_chunks(x, starts, rate, keep_bands="B4" in real):
            for name, value in assemble(cut, stats, real).items():
                pieces.setdefault(name, []).append(value)
            if "B4" in real:
                pieces.setdefault("B4", []).append(np.concatenate(
                    [_covariance(cut[f"band_{b}"], stats[f"band_{b}"][1]) for b in B4_BANDS], axis=1))
    if surrogate:
        for cut in window_chunks(x, starts, rate, rng=np.random.default_rng(seed)):
            for name, value in assemble(cut, stats, surrogate).items():
                pieces.setdefault(name, []).append(value)
    return {name: np.concatenate(parts, axis=0) for name, parts in pieces.items()}


def session_features(x: np.ndarray, starts: np.ndarray, rate: int, blocks: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Session-mode block features of one shaft: filter the continuous recording, then cut.

    Tier P measurement only. Surrogate blocks are not produced in this mode.
    """
    length = int(round(splits.WINDOW_S * rate))
    _check_starts(starts, length, x.shape[-1])
    real = tuple(b for b in blocks if "~" not in b)
    per_channel = []
    for channel in x:
        cut, stats = {}, {}
        for name, value in streams(channel, rate, in_window=False).items():
            hz = FINE_HZ if name == "hg256" else STREAM_HZ
            first = np.minimum(np.floor(starts * (hz / rate) + 0.5).astype(np.int64), len(value) - hz)
            cut[name] = _gather(value, first, hz)[None]
            if name in NORMALIZED_STREAMS:
                stats[name] = robust_stats(value[None, :])
        per_channel.append(assemble(cut, stats, real))
    out = {block: np.concatenate([part[block] for part in per_channel], axis=1) for block in per_channel[0]}
    if "B4" in blocks:
        work = np.stack([_resample(channel.astype(np.float64), rate, WORK_HZ) for channel in x])
        notched = _filter(_notch_sos(WORK_HZ), work, False)
        first = np.minimum(np.floor(starts * (WORK_HZ / rate) + 0.5).astype(np.int64), work.shape[-1] - WORK_HZ)
        per_band = []
        for band in B4_BANDS:
            passed = _filter(_band_sos(band, WORK_HZ), notched, False).astype(np.float32)
            scale = robust_stats(passed[:, ::2])[1]
            per_band.append(np.concatenate([_covariance(_gather(passed, first[low:low + 256], WORK_HZ), scale)
                                            for low in range(0, len(starts), 256)], axis=0))
        out["B4"] = np.concatenate(per_band, axis=1)
    return out


# --------------------------------------------------------------------------- per-session feature cache

def cache_root() -> Path:
    """The one setting for where classical caches live: NPEXP_CLASSICAL_CACHE, else work/classical.

    On the server both point at the scratch NVMe (``work/classical`` as a symlink). That disk
    is not mounted again after a reboot, so when the root is set by NPEXP_CLASSICAL_CACHE or
    is a symlink, it must resolve onto a mount other than ``/``; otherwise this refuses
    (exit 6) rather than write the cache to the system disk."""
    override = os.environ.get("NPEXP_CLASSICAL_CACHE")
    root = Path(override).expanduser() if override else common.work_root() / "classical"
    if override or root.is_symlink():
        mount = root.resolve()
        while not mount.exists():
            mount = mount.parent
        while not os.path.ismount(mount):
            mount = mount.parent
        if mount == Path("/"):
            print(f"CLASSICAL CACHE: {root} resolves to {root.resolve()} on /, not on the scratch disk. "
                  "Mount it. Nothing was started.", flush=True)
            raise SystemExit(6)
    return root


def features_root(root: Path | None = None) -> Path:
    return Path(root) if root is not None else cache_root() / "features"


def free_gib(path: Path) -> float:
    """Free space, in GiB, of the filesystem that holds ``path`` (its nearest existing ancestor)."""
    import shutil
    path = Path(path).resolve()
    while not path.exists():
        path = path.parent
    return shutil.disk_usage(path).free / 1024 ** 3


def store_dir(session: str, window_mode: str, reference: str, root: Path | None = None) -> Path:
    return features_root(root) / session / f"{window_mode}__{reference}"


def session_onsets(session: str) -> tuple[np.ndarray, list[str]]:
    """Sorted unique window starts over the 15 tasks of a session, and the tasks whose dataset failed to build."""
    source = splits.provider()
    starts, failed = [], []
    for task in splits.TASKS:
        try:
            starts.append(source.table(session, task).onsets)
        except MissingInput:
            raise
        except Exception as error:   # a task the benchmark cannot build for this session; reported, not hidden
            failed.append(f"{task}: {type(error).__name__}: {error}")
    if not starts:
        raise RuntimeError(f"{session}: no task dataset could be built: {failed}")
    return np.unique(np.concatenate(starts)).astype(np.int64), failed


def _onsets_digest(onsets: np.ndarray) -> str:
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(onsets, dtype=np.int64).tobytes()).hexdigest()[:16]


def _reference(rec: Recording, shaft_channels: list[Channel]) -> np.ndarray:
    """Re-referenced signals (channels, samples), accumulated one contact at a time to bound memory."""
    x = np.zeros((len(shaft_channels), rec.n_samples), dtype=np.float32)
    for contact in dict.fromkeys(c for channel in shaft_channels for c in channel.contacts):
        values = rec.read((contact,))[0]
        for row, channel in enumerate(shaft_channels):
            for member, weight in zip(channel.contacts, channel.weights):
                if member == contact and weight != 0.0:
                    x[row] += np.float32(weight) * values
    return x


def _shaft_job(job: dict) -> tuple[str, dict[tuple[str, str], dict[str, np.ndarray]]]:
    """Features of one shaft for every requested (mode, reference)."""
    rec: Recording = job["recording"]
    out: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for reference in dict.fromkeys(reference for _, reference in job["plan"]):
        shaft_channels: list[Channel] = job["channels"][reference]
        if not shaft_channels:
            continue
        x = _reference(rec, shaft_channels)
        for (mode, other), blocks in job["plan"].items():
            if other != reference:
                continue
            if mode == "window":
                stats = window_stats(x, rec.rate, want_bands="B4" in blocks)
                out[(mode, reference)] = window_features(x, job["onsets"], rec.rate, blocks, stats, seed=job["seed"])
            else:
                out[(mode, reference)] = session_features(x, job["onsets"], rec.rate, blocks)
        del x
    return job["shaft"], out


def extract(session: str, plan: dict[tuple[str, str], tuple[str, ...]], onsets: np.ndarray | None = None,
            root: Path | None = None, workers: int = 1, max_shafts: int | None = None) -> dict[tuple[str, str], Path]:
    """Compute and cache block features of one session.

    ``plan`` maps ``(window_mode, reference)`` to the blocks wanted. Blocks already cached for
    the same onsets and declared defaults are skipped. ``onsets`` defaults to every benchmark
    window of the session; ``max_shafts`` restricts to the first shafts (smoke tests only).
    A file lock admits one extraction per session at a time; a second caller waits and then
    finds the blocks cached.
    """
    import fcntl
    folder = features_root(root) / session
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / ".lock", "w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        return _extract(session, plan, onsets, root, workers, max_shafts)


def _extract(session: str, plan: dict[tuple[str, str], tuple[str, ...]], onsets: np.ndarray | None,
             root: Path | None, workers: int, max_shafts: int | None) -> dict[tuple[str, str], Path]:
    import zlib
    rec = recording(session)
    failed: list[str] = []
    if onsets is None:
        onsets, failed = session_onsets(session)
    onsets = np.unique(np.asarray(onsets, dtype=np.int64))
    digest = _onsets_digest(onsets)
    todo: dict[tuple[str, str], tuple[str, ...]] = {}
    metas: dict[tuple[str, str], dict] = {}
    layout: dict[str, list[Channel]] = {}
    for (mode, reference), blocks in plan.items():
        if mode not in WINDOW_MODES:
            raise ValueError(f"unknown window mode {mode!r}")
        blocks = tuple(b for b in blocks if not (mode == "session" and "~" in b))
        unknown = [b for b in blocks if b not in BLOCKS + SURROGATE_BLOCKS]
        if unknown:
            raise ValueError(f"unknown blocks {unknown}")
        if reference not in layout:
            chans = channels(rec.labels, rec.regions, reference)
            if max_shafts is not None:
                keep = list(dict.fromkeys(c.shaft for c in chans))[:max_shafts]
                chans = [c for c in chans if c.shaft in keep]
            layout[reference] = chans
        folder = store_dir(session, mode, reference, root)
        meta_path = folder / "meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else None
        if meta is not None and (meta["onsets_digest"] != digest or meta["defaults_hash"] != defaults_hash()
                                 or meta["max_shafts"] != max_shafts):
            raise RuntimeError(f"{folder} holds features for other onsets or defaults; remove it to recompute")
        if meta is None:
            shafts = list(dict.fromkeys(c.shaft for c in layout[reference]))
            b4, offset = [], 0
            for shaft in shafts:
                names = [c.name for c in layout[reference] if c.shaft == shaft]
                b4.append({"shaft": shaft, "channels": names, "offset": offset})
                offset += len(B4_BANDS) * len(names) * (len(names) + 1) // 2
            meta = {"session": session, "window_mode": mode, "reference": reference, "rate": rec.rate,
                    "defaults_hash": defaults_hash(), "defaults": DEFAULTS, "onsets_digest": digest,
                    "n_onsets": int(len(onsets)), "max_shafts": max_shafts, "failed_tasks": failed,
                    "channels": [{"name": c.name, "shaft": c.shaft, "region": c.region} for c in layout[reference]],
                    "b4": b4, "b4_width": offset, "blocks": {}}
        metas[(mode, reference)] = meta
        missing = tuple(b for b in blocks if b not in meta["blocks"])
        if missing:
            todo[(mode, reference)] = missing
    folders = {key: store_dir(session, key[0], key[1], root) for key in plan}
    if not todo:
        return folders

    maps: dict[tuple[str, str, str], np.memmap] = {}
    for (mode, reference), blocks in todo.items():
        folder = folders[(mode, reference)]
        folder.mkdir(parents=True, exist_ok=True)
        np.save(folder / "onsets.npy", onsets)
        for block in blocks:
            if block == "B4":
                shape, dtype = (len(onsets), metas[(mode, reference)]["b4_width"]), np.float32
            else:
                shape, dtype = (len(onsets), len(layout[reference]), BLOCK_WIDTH[block]), np.float16
            maps[(mode, reference, block)] = np.lib.format.open_memmap(
                folder / f"{block}.npy.part", mode="w+", dtype=dtype, shape=shape)

    shafts = list(dict.fromkeys(c.shaft for chans in layout.values() for c in chans))
    jobs = []
    for shaft in shafts:
        per_reference = {reference: [c for c in chans if c.shaft == shaft] for reference, chans in layout.items()}
        needed = {reference: per_reference[reference] for _, reference in todo}
        contacts = tuple(dict.fromkeys(contact for chans in needed.values() for c in chans for contact in c.contacts))
        if not contacts:
            continue
        jobs.append({"recording": rec, "shaft": shaft, "contacts": contacts, "channels": needed, "plan": todo,
                     "onsets": onsets, "seed": SURROGATE_SEED + zlib.crc32(f"{session}/{shaft}".encode())})

    def place(shaft: str, result: dict) -> None:
        for (mode, reference), blocks in result.items():
            names = [c.name for c in layout[reference]]
            members = [i for i, c in enumerate(layout[reference]) if c.shaft == shaft]
            for block, values in blocks.items():
                target = maps[(mode, reference, block)]
                if block == "B4":
                    entry = next(e for e in metas[(mode, reference)]["b4"] if e["shaft"] == shaft)
                    target[:, entry["offset"]:entry["offset"] + values.shape[1]] = values
                else:
                    assert [names[i] for i in members] == [c.name for c in layout[reference] if c.shaft == shaft]
                    target[:, members, :] = values

    if workers <= 1:
        for job in jobs:
            place(*_shaft_job(job))
    else:
        # h5py state must not be forked; in-memory test recordings only exist in this process.
        context = multiprocessing.get_context("spawn" if isinstance(rec, H5Recording) else "fork")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
            for shaft, result in pool.map(_shaft_job, jobs):
                place(shaft, result)
    for (mode, reference, block), target in maps.items():
        target.flush()
        folder = folders[(mode, reference)]
        os.replace(folder / f"{block}.npy.part", folder / f"{block}.npy")
        metas[(mode, reference)]["blocks"][block] = list(target.shape)
    del maps
    for key in todo:
        (folders[key] / "meta.json").write_text(json.dumps(metas[key], indent=1))
    return folders


class Store:
    """Read access to one session's cached features for a (window mode, reference)."""

    def __init__(self, session: str, window_mode: str, reference: str, root: Path | None = None) -> None:
        self.folder = store_dir(session, window_mode, reference, root)
        meta_path = self.folder / "meta.json"
        if not meta_path.exists():
            raise MissingInput(f"classical features not extracted: {self.folder}")
        self.meta = json.loads(meta_path.read_text())
        if self.meta["defaults_hash"] != defaults_hash():
            raise MissingInput(f"{self.folder} was extracted with other declared defaults")
        self.onsets = np.load(self.folder / "onsets.npy")
        self.names = [c["name"] for c in self.meta["channels"]]
        self.regions = [c["region"] for c in self.meta["channels"]]

    def positions(self, onsets: np.ndarray) -> np.ndarray:
        onsets = np.asarray(onsets, dtype=np.int64)
        found = np.minimum(np.searchsorted(self.onsets, onsets), len(self.onsets) - 1)
        if not np.array_equal(self.onsets[found], onsets):
            raise MissingInput(f"{self.folder} lacks {int((self.onsets[found] != onsets).sum())} requested windows")
        return found

    def rows(self, block: str, positions: np.ndarray) -> np.ndarray:
        base, columns = VIEWS.get(block, (block, slice(None)))
        if base not in self.meta["blocks"]:
            raise MissingInput(f"block {base} not extracted in {self.folder}")
        data = np.load(self.folder / f"{base}.npy", mmap_mode="r")
        order = np.argsort(positions, kind="stable")
        out = np.empty((len(positions),) + data.shape[1:], dtype=np.float32)
        out[order] = data[positions[order]]
        return out if base == "B4" else out[..., columns]


# --------------------------------------------------------------------------- per-cell kernels

def _tangent(reference: np.ndarray, covariances: np.ndarray) -> np.ndarray:
    """Tangent vectors (n, c(c+1)/2) of SPD matrices at ``reference``, off-diagonals weighted by sqrt 2."""
    values, vectors = np.linalg.eigh(reference)
    values = np.maximum(values, 1e-10 * values.max())
    whitener = (vectors / np.sqrt(values)) @ vectors.T
    eigenvalues, eigenvectors = np.linalg.eigh(whitener @ covariances @ whitener)
    logs = (eigenvectors * np.log(np.maximum(eigenvalues, 1e-12))[:, None, :]) @ eigenvectors.transpose(0, 2, 1)
    rows, cols = np.triu_indices(reference.shape[0])
    return logs[:, rows, cols] * np.where(rows == cols, 1.0, np.sqrt(2.0))


def _full(triangles: np.ndarray, size: int) -> np.ndarray:
    rows, cols = np.triu_indices(size)
    out = np.empty((len(triangles), size, size), dtype=np.float64)
    out[:, rows, cols] = triangles
    out[:, cols, rows] = triangles
    return out


def _b4_matrices(cell: Cell, stores: dict[str, Store], raw: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    if cell.regime == "cross_subject":
        raise BlockUnavailable("B4 is defined per shaft and cannot be aligned across subjects")
    target_part = next((p for p in raw if p != "train"), "train")
    target_layout = {e["shaft"]: e for e in stores[target_part].meta["b4"]}
    columns: dict[str, list[np.ndarray]] = {part: [] for part in raw}
    for entry in stores["train"].meta["b4"]:
        other = target_layout.get(entry["shaft"])
        if other is None:
            continue
        shared = [name for name in entry["channels"] if name in other["channels"]]
        if not shared:
            continue
        for band in range(len(B4_BANDS)):
            matrices = {}
            for part, values in raw.items():
                layout = entry if part == "train" else other
                size = len(layout["channels"])
                width = size * (size + 1) // 2
                low = layout["offset"] + band * width
                keep = [layout["channels"].index(name) for name in shared]
                matrices[part] = _full(values[:, low:low + width], size)[:, keep][:, :, keep]
            reference = matrices["train"].mean(axis=0)
            for part in raw:
                columns[part].append(_tangent(reference, matrices[part]))
    if not columns["train"]:
        raise BlockUnavailable("no shaft shared between the training and target sessions")
    return {part: np.concatenate(parts, axis=1) for part, parts in columns.items()}


def _align(cell: Cell, stores: dict[str, Store], raw: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Bring per-channel features (n, C, F) of each part onto a common column axis, flattened."""
    train = stores["train"]
    target = next((stores[p] for p in raw if p != "train"), train)
    if cell.regime == "within":
        index = {"train": slice(None), "target": slice(None)}
        pooled = raw
    elif cell.regime == "cross_session":
        shared = [name for name in train.names if name in set(target.names)]
        if not shared:
            raise BlockUnavailable("no channel shared between the two sessions")
        index = {"train": [train.names.index(n) for n in shared], "target": [target.names.index(n) for n in shared]}
        pooled = {part: values[:, index["train" if part == "train" else "target"]] for part, values in raw.items()}
    else:
        shared = sorted((set(train.regions) & set(target.regions)) - {""})
        if not shared:
            raise BlockUnavailable("no DKT region shared between anchor and target")
        pooled = {}
        for part, values in raw.items():
            regions = np.array(train.regions if part == "train" else target.regions)
            pooled[part] = np.stack([values[:, regions == region].mean(axis=1) for region in shared], axis=1)
    return {part: values.reshape(len(values), -1) for part, values in pooled.items()}


def check_mode(tier: str, window_mode: str) -> None:
    if window_mode not in WINDOW_MODES:
        raise ValueError(f"unknown window mode {window_mode!r}")
    if tier != "P" and window_mode != "window":
        raise ValueError("window_mode='session' is a Tier P measurement only; Tiers V and T use 'window'")


def cell_matrices(cell: Cell, block: str, window_mode: str, reference: str, *, root: Path | None = None,
                  onsets: dict[str, np.ndarray] | None = None,
                  remap: Callable[[Store, np.ndarray], np.ndarray] | None = None) -> dict[str, np.ndarray]:
    """Feature matrices per part, standardized with the training rows' mean and standard deviation.

    ``onsets`` overrides the cell's own rows (smoke tests pass train and select subsets only).
    ``remap`` rewrites cache positions before reading (the shifted-sham control).
    """
    check_mode(cell.tier, window_mode)
    if onsets is None:
        split = splits.get_cell(cell)
        onsets = {part: split.onsets(part) for part in splits.PARTS}
    if "train" not in onsets:
        raise ValueError("training rows are required")
    stores = {part: Store(cell.train_session if part == "train" else cell.target_session, window_mode, reference, root)
              for part in onsets}
    raw = {}
    for part, starts in onsets.items():
        positions = stores[part].positions(starts)
        if remap is not None:
            positions = remap(stores[part], positions)
        raw[part] = stores[part].rows(block, positions)
    matrices = _b4_matrices(cell, stores, raw) if block == "B4" else _align(cell, stores, raw)
    train = matrices["train"].astype(np.float64)
    mean, scale = train.mean(axis=0), train.std(axis=0)
    scale[scale == 0] = 1.0
    return {part: (values.astype(np.float64) - mean) / scale for part, values in matrices.items()}


def gram(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """``left @ right.T`` in float64; on the GPU named by NPEXP_GRAM_DEVICE when set (for example "cuda")."""
    device = os.environ.get("NPEXP_GRAM_DEVICE", "cpu")
    if device == "cpu":
        return left @ right.T
    import torch
    a = torch.as_tensor(left, dtype=torch.float64, device=device)
    b = a if right is left else torch.as_tensor(right, dtype=torch.float64, device=device)
    return (a @ b.T).cpu().numpy()


def kernel_set(matrices: dict[str, np.ndarray], cross_gram: bool = False) -> KernelSet:
    train = matrices["train"]
    others = {part: values for part, values in matrices.items() if part != "train"}
    extra = None
    if cross_gram:
        extra = {part: gram(values, values) for part, values in others.items()}
        if "eval" in others and "select" in others:
            extra["eval_select"] = gram(others["eval"], others["select"])
    return KernelSet(train=gram(train, train), cross={part: gram(values, train) for part, values in others.items()},
                     cross_gram=extra)


def kernels_and_dim(cell: Cell, block: str, window_mode: str, reference: str, **options) -> tuple[KernelSet, int]:
    """Kernels of a cell and block plus the feature dimension behind them.

    Options: ``root``, ``onsets``, ``remap`` (see ``cell_matrices``) and ``cross_gram``. With
    NPEXP_KERNEL_CACHE=1 plain requests are cached under work/classical/kernels/.
    """
    cross_gram = options.pop("cross_gram", False)
    plain = not options and not cross_gram
    path = (cache_root() / "kernels" / cell.tier / cell.regime
            / f"{window_mode}__{reference}__{block}__{defaults_hash()}" / f"{cell.key()}.npz")
    if plain and os.environ.get("NPEXP_KERNEL_CACHE") == "1" and path.exists():
        with np.load(path) as saved:
            return KernelSet(train=saved["train"].astype(np.float64),
                             cross={part: saved[part].astype(np.float64) for part in ("select", "eval")
                                    if part in saved}), int(saved["dim"])
    matrices = cell_matrices(cell, block, window_mode, reference, **options)
    result = kernel_set(matrices, cross_gram=cross_gram)
    dim = int(matrices["train"].shape[1])
    if plain and os.environ.get("NPEXP_KERNEL_CACHE") == "1":
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp.npz")
        np.savez(temporary, train=result.train.astype(np.float32), dim=dim,
                 **{part: values.astype(np.float32) for part, values in result.cross.items()})
        os.replace(temporary, path)
    return result, dim


def kernels(cell: Cell, block: str, window_mode: str, reference: str) -> KernelSet:
    """Linear kernels of one block for one cell (contract entry point).

    block: "base6" | "B1".."B5" (also "env5", "wave10", "B1~phaserand", "B2~phaserand");
    window_mode: "window" | "session" (Tier P only); reference: "bipolar" | "laplacian" | "shaft_mean".
    Raises ``MissingInput`` when the features are not extracted and ``BlockUnavailable`` when the
    block does not exist in the cell's regime.
    """
    return kernels_and_dim(cell, block, window_mode, reference)[0]

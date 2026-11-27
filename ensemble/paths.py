"""Locations on the GPU server and the read-only inputs taken from the reproduction."""
from __future__ import annotations

import hashlib
import os
import re
import shutil
from pathlib import Path

# GiB that must stay free on the filesystem that holds the scratch root. On the root
# filesystem the floor is never below 250; a dedicated scratch disk may set a lower one.
DISK_FLOOR_GIB = int(os.environ.get("NPEXP_DISK_FLOOR_GIB", "250"))
ROOT_FLOOR_GIB = 250
GIB = 1 << 30
# sha256 from the reproduction's neuroprobe-results/weights.sha256
WEIGHTS = {
    "mapa_vits384.pt": "2d236089a2f1a3cc2827e3f150c4a2ba14c51bbfaf0ce0888f84b92a6eb25a7a",
    "diver_ieeg.pt": "b812093779fb5cf1b76e18f8307df4136b67b008b6cd94c6be75fa286a87864c",
}
# Pinned revisions (reproduction's neuroprobe-results/revisions.txt)
REVISIONS = {"MAPA": "dc582b2ab18fa57dce3e88f15240f50d5727c1c1",
             "neuroprobe": "f9b084241aeeaab548ca45ec98368b70141465a8",
             "DIVER-1": "25638eb38ef297b582ab79ae1c96260f57c155b3"}
EXCLUDED_SESSIONS = {"btbank6_0"}     # shows an evaluation film; not a Tier P session


def root() -> Path:
    return Path(os.environ.get("NPEXP_ROOT", Path.home() / "runs/neuroprobe-experiments"))


def results() -> Path:
    return root() / "results"


def work() -> Path:
    return root() / "work"


def repro() -> Path:
    """The reproduction's tree. Read-only for us."""
    return Path(os.environ.get("NPEXP_REPRO_ROOT", Path.home() / "runs/neuroprobe-reproduction"))


def data_root() -> Path:
    return Path(os.environ.get("ROOT_DIR_BRAINTREEBANK", root() / "data_dev"))


def source(name: str) -> Path:
    """Source tree of MAPA, neuroprobe or DIVER-1: ours if present, else the reproduction's."""
    override = os.environ.get(f"NPEXP_{name.upper().replace('-1', '')}_SRC")
    for candidate in ([Path(override)] if override else []) + [root() / "src" / name, repro() / "src" / name]:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"source tree {name} not found under {root() / 'src'} or {repro() / 'src'}")


def weights(name: str) -> Path:
    return repro() / "weights" / name


def verify_weights(name: str) -> str:
    digest = hashlib.sha256()
    with weights(name).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    if digest.hexdigest() != WEIGHTS[name]:
        raise RuntimeError(f"{weights(name)} sha256 {digest.hexdigest()} differs from the pinned {WEIGHTS[name]}")
    return digest.hexdigest()


def parse_session(session: str) -> tuple[int, int]:
    match = re.fullmatch(r"btbank(\d+)_(\d+)", session)
    if not match:
        raise ValueError(f"unrecognized session {session!r}")
    subject, trial = int(match.group(1)), int(match.group(2))
    if session in EXCLUDED_SESSIONS or trial >= 100:
        raise ValueError(f"{session} is excluded from every tier (evaluation-film or partial session)")
    return subject, trial


def recording(session: str) -> Path:
    subject, trial = parse_session(session)
    return data_root() / f"sub_{subject}_trial{trial:03d}.h5"


def scratch() -> Path:
    """Root of the bulk, deletable intermediates: per-contact MAPA features, prepared
    sessions, DIVER and CNN window caches. ``NPEXP_SCRATCH`` moves it off the system
    disk; kernels do not live here (see ``common.mapa_features``). A scratch root set
    there must sit on its own mounted filesystem: if that disk is not mounted (it is
    not mounted again after a reboot) the job refuses rather than write to ``/``."""
    override = os.environ.get("NPEXP_SCRATCH")
    if not override:
        return work()
    path = Path(override)
    mount = _existing(path).resolve()
    while not os.path.ismount(mount):
        mount = mount.parent
    if mount == Path("/"):
        print(f"SCRATCH: NPEXP_SCRATCH={path} is not on a mounted scratch filesystem (it resolves to /). "
              "Mount the scratch disk or unset NPEXP_SCRATCH. Nothing was started.", flush=True)
        raise SystemExit(6)     # not a resource refusal: a queue stops here
    return path


def _existing(path: Path) -> Path:
    path = Path(path)
    while not path.exists():
        path = path.parent
    return path


def free_gib(path: Path | None = None) -> float:
    """Free space of the filesystem that holds ``path`` (default: the scratch root)."""
    return shutil.disk_usage(_existing(path or scratch())).free / GIB


def floor_gib(path: Path | None = None) -> int:
    """Floor for the filesystem that holds ``path``: the configured one, and at least 250
    when that filesystem is the system disk."""
    on_root = os.stat(_existing(path or scratch())).st_dev == os.stat("/").st_dev
    return max(ROOT_FLOOR_GIB, DISK_FLOOR_GIB) if on_root else DISK_FLOOR_GIB


def require_disk(need_gib: float, what: str, path: Path | None = None) -> None:
    """Refuse a step that would leave less than the floor free on the filesystem that
    holds ``path`` (default: the scratch root)."""
    target = _existing(path or scratch())
    free, floor = free_gib(target), floor_gib(target)
    if free - need_gib < floor:
        print(f"DISK: {what} needs {need_gib:.0f} GiB on the filesystem of {target}, "
              f"{free:.0f} GiB are free there and {floor} GiB must stay free. Nothing was started.", flush=True)
        raise SystemExit(5)     # resources: a queue retries later


def exit_on_cuda_oom(function):
    """Run ``function``; turn a CUDA out-of-memory error into a clean exit (code 7).
    Every output is written per cell and atomically, so a rerun resumes where it stopped."""
    try:
        return function()
    except RuntimeError as error:
        text = str(error)
        if type(error).__name__ != "OutOfMemoryError" and "CUDA out of memory" not in text \
                and "CUBLAS_STATUS_ALLOC_FAILED" not in text:
            raise
        print(f"CUDA OUT OF MEMORY: {text.splitlines()[0][:300]}. Another process probably took the card "
              "(gpu-inference.service holds about 56 GiB when it runs). Completed cells are kept; rerun the "
              "same command to resume.", flush=True)
        raise SystemExit(4) from None     # resources: a queue retries later

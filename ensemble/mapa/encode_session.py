"""Encode one session with the released MAPA encoder at caller-given window onsets.

Adapted from the reproduction's ``mapa_encode.py``
(/Users/jake/ws/code/personal/spark/experiments/neuroprobe/mapa_encode.py,
sha256 06f7a995bc83538d92acd35dda9d2453826f17fa666b7ebd0c41ee9a1907676a) and from the
session loop of MAPA's ``evals/neuroprobe/encode.py`` at dc582b2, whose helpers are
called unchanged. Differences, none of which touch a feature value:

- windows are the onsets passed in (the benchmark's own, from ``common.splits``), so
  permitted sessions outside MAPA's 12-session board can be encoded and no label,
  split or task enters the feature files;
- only block 12 is tapped (as in ``mapa_encode.py``);
- outputs are written straight to ``.npy`` memory maps in the destination, per-contact
  taps contact-major, so a session is written once (the upstream path writes scratch
  files and then a second full copy through ``torch.save``).

Contact selection is MAPA's: its voltage order with its per-session exclusion list,
intersected with the Neuroprobe Lite electrodes of the subject.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from ensemble import paths
from ensemble.mapa.features import native_rate

WIDTH_BYTES = 2    # float16


def estimate_gib(n_windows: int, n_contacts: int, n_regions: int, elec: bool) -> float:
    """Bytes written for one session: 52 tokens x 384 dims for block 12, 348 frontend bins."""
    per_unit = n_windows * (19968 + 348) * WIDTH_BYTES
    return per_unit * (n_regions + (n_contacts if elec else 0)) / paths.GIB


def encode(session: str, prepared: Path, out: Path, onsets: np.ndarray, *, elec: bool,
           batch_size: int = 8) -> Path:
    subject, trial = paths.parse_session(session)
    mapa = paths.source("MAPA")
    sys.path.insert(0, str(mapa))
    from evals.neuroprobe import encode as upstream
    from mapa.data import anatomy
    from mapa.data.region_fn import make_bt_region_fn
    from mapa.data.session_loader import load_v3_sessions
    from mapa.models.pack_r4 import build_r4_grid

    if torch.cuda.device_count() != 1:
        raise SystemExit("expected exactly one visible CUDA device (set CUDA_VISIBLE_DEVICES to one UUID)")
    device = torch.device("cuda")
    bt_root = str(paths.data_root())
    onsets = np.unique(np.asarray(onsets, dtype=np.int64))
    starts = onsets / float(native_rate(subject))
    clip_frames = round(upstream.CLIP_DUR_S * upstream.FPS)
    region_fn = make_bt_region_fn(bt_root)
    spec = load_v3_sessions(
        sessions=[(subject, trial)],
        band_cache_dirs=[str(prepared / "slow"), str(prepared / "mid"), str(prepared / "fast")],
        span_dir=str(prepared / "spans"), region_fn=region_fn, lof_report_path=None,
        keep_labels_fn=upstream._lite_keep_labels_fn(bt_root))[0]
    kept = list(spec.setup.sidecar.labels)
    if not set(kept) <= set(anatomy.lite_electrode_set(subject)):
        raise RuntimeError("the montage kept a non-Lite electrode; refusing to encode")
    bands = upstream._window_bands(spec, starts, clip_frames)
    grid = build_r4_grid(spec.setup.geom.to(device), n_time=clip_frames)
    region_true = region_fn(subject, trial, kept).long()
    if not torch.equal(region_true, spec.setup.region_id.cpu()):
        raise RuntimeError(f"{session}: recomputed atlas ids disagree with the session spec")
    region_packed = spec.setup.region_id.to(device)[grid.contact]
    canon, region_canon, present = upstream._canon_regions(grid, region_true)
    lengths = upstream._enc0_band_lengths(bands, None)
    if lengths != tuple(int(x) for x in grid.band_lengths):
        raise RuntimeError(f"frontend band lengths {lengths} != model grid {grid.band_lengths}")

    sha = paths.verify_weights("mapa_vits384.pt")
    state, recorded = upstream._load_ckpt(str(paths.weights("mapa_vits384.pt")))
    if recorded is None:
        raise RuntimeError("the checkpoint does not record its relative-position setting")
    encoder = upstream._load_encoder(state, device=device, space_rope=recorded)
    del state

    n = bands[0].shape[0]
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=out.name + ".", dir=out.parent))
    np.save(staging / "enc0.npy", upstream._enc0_pooled(bands, canon, region_canon, present).numpy())
    if elec:
        frontend = upstream._enc0_elec(bands, canon).numpy()             # (windows, contacts, 348)
        np.save(staging / "enc0_elec.npy", np.ascontiguousarray(frontend.transpose(1, 0, 2)))
        del frontend
    pooled = contact = None
    with torch.inference_mode():
        for lo in range(0, n, batch_size):
            hi = min(lo + batch_size, n)
            x = [band[lo:hi].to(device) for band in bands]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, taps = encoder.forward(x, grid, region_packed, tap_blocks=(12,))
            y = taps[12].float().reshape(hi - lo, -1, grid.k_full, taps[12].shape[-1]).cpu()
            block = upstream._pool_regions(y, region_canon, present).numpy()
            if pooled is None:
                pooled = np.lib.format.open_memmap(staging / "enc12.npy", mode="w+", dtype=np.float16,
                                                   shape=(n, *block.shape[1:]))
                if elec:
                    contact = np.lib.format.open_memmap(
                        staging / "enc12_elec.npy", mode="w+", dtype=np.float16,
                        shape=(y.shape[1], n, y.shape[2] * y.shape[3]))
            pooled[lo:hi] = block
            if elec:
                contact[:, lo:hi] = y.flatten(2).half().numpy().transpose(1, 0, 2)
    pooled.flush()
    if elec:
        contact.flush()
    listed = (subject, trial) in anatomy._BT_V14_EXTRA_BAD_ELECTRODES_PER_SESSION
    meta = {"session": session, "onsets": onsets.tolist(), "native_rate": native_rate(subject),
            "elec_labels": np.asarray(kept)[canon].tolist(),
            "present_parcels": np.asarray(present).tolist(),
            "parcel_canon": np.asarray(region_canon).tolist(),
            "band_lengths": list(lengths), "elec": bool(elec), "prepared": str(prepared),
            "checkpoint_sha256": sha, "mapa_revision": paths.REVISIONS["MAPA"],
            "flags": {} if listed else {"no_mapa_exclusion_list": True}}
    (staging / "meta.json").write_text(json.dumps(meta))
    os.replace(staging, out)
    print(f"[encode] {session}: windows={n} contacts={len(kept)} regions={len(present)} "
          f"elec={elec}{'' if listed else ' NO-EXCLUSION-LIST'} -> {out}", flush=True)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session")
    parser.add_argument("prepared", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("onsets", type=Path, help=".npy of window-start sample indices")
    parser.add_argument("--elec", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    encode(args.session, args.prepared, args.out, np.load(args.onsets), elec=args.elec, batch_size=args.batch_size)

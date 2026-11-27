# Copied unchanged from the reproduction workstream (read-only source):
#   /Users/jake/ws/code/personal/spark/experiments/neuroprobe/prepare_session.py
#   sha256 1fce714cda0706a8854c10b6aafb97d5f0f0ddab0ea60f533047fcf79c5a8267 (copied 2026-11-25)
# Usage: prepare_session.py <subject> <trial> <output dir>, with ROOT_DIR_BRAINTREEBANK set.
# For sessions without a per-session exclusion list in MAPA (only 5:0 among the permitted
# sessions) voltage_electrode_order drops nothing extra; the kernel job flags that session.
"""Prepare a NeuroProbe Lite recording using MAPA's whole-session frontend."""
import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import h5py
import numpy as np
import torch

from mapa import build_sidecar
from mapa.data.anatomy import lite_electrode_set, voltage_electrode_order
from mapa.data.manifest import bt_subject_native_rate_hz
from mapa.data.normalize import SessionRobustZNormalizer
from mapa.data.region_fn import make_bt_region_fn
from mapa.preprocessing import prepare_recording
from mapa.preprocessing.frontend import PreparedRecording
from neuroprobe import BrainTreebankSubject


def prepare_shafts(labels, regions, rate, read_rows, workers=2):
    # MAPA references within shafts. Filtering, STFTs, and normalization are
    # per contact. Keep the complete time axis and preserve original contact order.
    sidecar = build_sidecar(labels, region_id=regions)
    groups = [np.flatnonzero(sidecar.array_id.numpy() == i) for i in range(sidecar.n_arrays)]

    def process(rows):
        part = prepare_recording(read_rows(rows), rate, [labels[i] for i in rows], regions[rows])
        return rows, part

    bands, medians, scales = [], [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for rows, part in pool.map(process, groups):
            if not bands:
                for values, norm in zip(part.bands, part.normalizers):
                    bands.append(torch.empty((len(labels), *values.shape[1:]), dtype=values.dtype))
                    medians.append(torch.empty((len(labels), *norm.median.shape[1:]),dtype=norm.median.dtype))
                    scales.append(torch.empty_like(medians[-1]))
            for b, values in enumerate(part.bands):
                bands[b][rows] = values
                medians[b][rows] = part.normalizers[b].median
                scales[b][rows] = part.normalizers[b].sigma
            print(f"prepared shaft containing {labels[rows[0]]}",flush=True)
    norms = tuple(SessionRobustZNormalizer.from_stats(median=m,sigma=s) for m,s in zip(medians,scales))
    metadata = dict(part.metadata, input_labels=list(labels), output_labels=list(labels),
                    input_indices=list(range(len(labels))), region_ids=regions.tolist())
    return PreparedRecording(tuple(bands),norms,sidecar,np.arange(len(labels)),part.duration_s,metadata)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("subject", type=int)
    parser.add_argument("trial", type=int)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = Path(os.environ["ROOT_DIR_BRAINTREEBANK"])
    subject = BrainTreebankSubject(subject_id=args.subject, cache=False)
    lite = lite_electrode_set(args.subject)
    labels = [label for label in voltage_electrode_order(root, args.subject, args.trial)
              if label in lite]
    if not set(labels).issubset(subject.electrode_labels):
        raise ValueError("MAPA contact selection differs from the NeuroProbe recording")
    h5path = root / f"sub_{args.subject}_trial{args.trial:03d}.h5"
    def read_rows(rows):
        keys = [subject.h5_neural_data_keys[labels[i]] for i in rows]
        with h5py.File(h5path,"r") as recording:
            data = recording["data"]
            voltage = np.empty((len(keys), data[keys[0]].shape[0]), dtype=np.float32)
            for i, key in enumerate(keys):
                data[key].read_direct(voltage[i])
        return voltage

    regions = make_bt_region_fn(root)(args.subject, args.trial, labels).numpy()
    prepared = prepare_shafts(labels,regions,bt_subject_native_rate_hz(args.subject),read_rows)
    prepared.save(args.output, subject_id=args.subject, trial_id=args.trial)
    print(f"saved {args.output}", flush=True)


if __name__ == "__main__":
    main()

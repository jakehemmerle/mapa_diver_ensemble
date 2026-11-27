#!/usr/bin/env bash
# MAPA kernel job: encode, reduce to per-cell kernels (taps 0 and 12), delete features.
#   mapa_kernels.sh <GPU-UUID> --store P|E [--regimes within,cross_session,cross_subject]
#                   [--max-held 1-5] [--disk-floor GiB] [--available-only] [--skip-parity] [--claim] [--dry-run]
#   --claim: one shard of a Tier P within/cross_subject plan shared with other --claim jobs
#            (per-session claims on /scratch); the unit name carries the card.
#   mapa_kernels.sh <GPU-UUID|cpu> --store E --parity-only [--parity-tasks speech ...]
# VRAM declared: 6144 MiB (encode <= 3.1 GiB measured by the reproduction; kernel
# accumulation about 4.8 GiB estimated). First step is the parity check against the
# reproduction's cross-subject result for 3:0 (training and validation labels only).
memory_default=24G   # prepare + encode peaked at 23-27 GiB in the reproduction's units (measured there)
source "$(dirname "$0")/_common.sh"
need_mib=6144
gpu=${1:-}; shift || true
store=$(flag_value --store "$@")
[[ -n "$gpu" && -n "$store" ]] || die "usage: $0 <GPU-UUID> --store P|E [--regimes a,b] [--skip-parity] [--dry-run]" 64
regimes=$(flag_value --regimes "$@"); regimes=${regimes:-within,cross_session,cross_subject}
if has_flag --dry-run "$@"; then
    [[ "$gpu" == cpu ]] || echo "$(gpu_free_mib "$gpu") MiB free on $gpu now; the job declares $need_mib MiB"
    dry_run ensemble.mapa.kernel_job "$@"
    exit
fi
visible=""
if [[ "$gpu" != cpu ]]; then require_gpu "$gpu" "$need_mib"; visible=$gpu; fi
suffix=$(unit_part "$store-$regimes"); if has_flag --parity-only "$@"; then suffix="parity"; fi
if has_flag --claim "$@"; then suffix="$suffix-claim-${gpu:4:8}"; fi
launch "spark-npexp-ek-mapa-$suffix" "$visible" ensemble.mapa.kernel_job "$@"

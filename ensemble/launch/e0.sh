#!/usr/bin/env bash
# E0: export members' per-trial scores for a tier and regime, then check each export
# (AUROC recomputed from the file equals the member's own to 1e-6). Not a decision.
#   wave 1 (CPU, needs the MAPA kernels):  e0.sh cpu --wave 1 --tier P --regime within
#   wave 2 (one trained member per GPU):   e0.sh <GPU-UUID> --wave 2 --member diver|cnn --tier P --regime within [--gpu-shard i/n]
#   check only (e.g. after the classical track exported ridge6):  e0.sh cpu --check-only --wave 1 --tier P --regime within
# VRAM declared: diver 6144 MiB (estimate, not logged by the reproduction), cnn 3072 MiB
# (2.6 GiB measured by the reproduction), wave 1 none.
source "$(dirname "$0")/_common.sh"
gpu=${1:-}; shift || true
tier=$(flag_value --tier "$@"); regime=$(flag_value --regime "$@"); wave=$(flag_value --wave "$@")
member=$(flag_value --member "$@"); shard=$(flag_value --gpu-shard "$@")
[[ -n "$gpu" && -n "$tier" && -n "$regime" && -n "$wave" ]] || \
    die "usage: $0 <GPU-UUID|cpu> --wave 1|2 [--member diver|cnn] --tier P|V|T --regime R [--gpu-shard i/n] [--available-only] [--export-only|--check-only] [--dry-run]" 64
common=(--tier "$tier" --regime "$regime")
case "$wave" in
    1) members=(mapa12 mapa0); checked=(mapa12 mapa0 ridge6); need_mib=0
       [[ "$gpu" == cpu ]] || die "wave 1 runs on the CPU: pass 'cpu' instead of a GPU UUID" 64 ;;
    2) case "$member" in diver) need_mib=6144; memory_max=${NPEXP_MEMORY_MAX:-24G} ;; cnn) need_mib=3072; memory_max=${NPEXP_MEMORY_MAX:-16G} ;;
          *) die "wave 2 needs --member diver|cnn" 64 ;; esac
       members=("$member"); checked=("$member") ;;
    *) die "--wave must be 1 or 2" 64 ;;
esac
export_args=(export "${common[@]}" --members "${members[@]}")
[[ -n "$shard" ]] && export_args+=(--gpu-shard "$shard")
has_flag --available-only "$@" && export_args+=(--available-only)
has_flag --quick "$@" && export_args+=(--quick)     # trained members: seed 0 select/eval only
mode=""
has_flag --seeds-only "$@" && { export_args+=(--seeds-only); mode=-seeds; }   # trained members: seeds 1-2
has_flag --oof-only "$@" && { export_args+=(--oof-only); mode=-oof; }         # trained members: seed 0 out-of-fold only
# --export-only: write predictions without the E0 check (used while recordings are still arriving)
check_args=(e0 "${common[@]}" --wave "$wave" --members "${checked[@]}")
if has_flag --dry-run "$@"; then
    [[ "$gpu" == cpu ]] || echo "$(gpu_free_mib "$gpu") MiB free on $gpu now; the job declares $need_mib MiB"
    status=0
    has_flag --check-only "$@" || dry_run ensemble.run "${export_args[@]}" --dry-run || status=$?
    has_flag --export-only "$@" || dry_run ensemble.run "${check_args[@]}" --dry-run || status=$?
    exit "$status"
fi
visible=""
if (( need_mib > 0 )); then require_gpu "$gpu" "$need_mib"; visible=$gpu; fi
name="spark-npexp-e0-w$wave-$(unit_part "$tier-$regime")${member:+-$member}$mode${shard:+-$(unit_part "$shard")}"
if has_flag --check-only "$@"; then
    launch "$name-check" "" ensemble.run "${check_args[@]}"
elif has_flag --export-only "$@"; then
    launch "$name" "$visible" ensemble.run "${export_args[@]}"
else
    launch "$name" "$visible" ensemble.e0_job "$(printf '%s\x1f' "${export_args[@]}")" "$(printf '%s\x1f' "${check_args[@]}")"
fi

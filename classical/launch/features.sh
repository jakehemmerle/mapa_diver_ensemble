#!/usr/bin/env bash
# Cache classical features per session (CPU only, reads no labels).
#   features.sh <cpu|GPU-UUID> [--dry-run] [--sessions eval|dev|btbankS_T ...] [--exact]
# eval: 12 evaluation sessions, in-window, bipolar, all blocks (feeds C0, C4, C5).
# dev:  13 Tier P sessions, everything C1-C3 read (in-window + surrogates, session-filtered,
#       base under the two other references). Sessions whose recording is absent are skipped.
# No GPU memory is used; the device argument exists so every launcher has the same shape.
source "$(dirname "$0")/_lib.sh"
name=eval
for ((i = 1; i <= $#; i++)); do
    if [[ ${!i} == --sessions ]]; then j=$((i + 1)); name=${!j:-eval}; fi
done
npexp_launch cache "cfeat-$name" 0 "$@"

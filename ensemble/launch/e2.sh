#!/usr/bin/env bash
# E2 fusion ladder on Tier P: uniform mean vs fixed-penalty stacker, with the shuffled-label control.
#   e2.sh --regime R --wave 1|2 [--dry-run]
# CPU only (VRAM 0). Reads prediction files; appends to results/ledger.csv.
source "$(dirname "$0")/_common.sh"
cpu_phase e2 "$@"

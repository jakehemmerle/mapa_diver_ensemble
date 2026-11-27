#!/usr/bin/env bash
# E3 controls on the chosen design and the ensemble track's proposal for the regime (results/frozen/proposals/).
#   e3.sh --regime R --wave 1|2 [--no-propose] [--dry-run]
# CPU only (VRAM 0). Reads prediction files; appends to results/ledger.csv.
source "$(dirname "$0")/_common.sh"
cpu_phase e3 "$@"

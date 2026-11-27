#!/usr/bin/env bash
# E5 score: the frozen candidate on Tier T, once, behind results/LOCK.json. Refuses without a Tier V Go.
#   e5.sh --regime R [--dry-run]
# CPU only (VRAM 0). Reads prediction files; appends to results/ledger.csv.
source "$(dirname "$0")/_common.sh"
cpu_phase e5 "$@"

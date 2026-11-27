#!/usr/bin/env bash
# E1 uniform screen on Tier P (every member subset of size 2-4), with the leave-one-member-out and self-ensemble controls.
#   e1.sh --regime R --wave 1|2 [--dry-run]
# CPU only (VRAM 0). Reads prediction files; appends to results/ledger.csv.
source "$(dirname "$0")/_common.sh"
cpu_phase e1 "$@"

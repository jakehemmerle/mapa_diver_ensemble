#!/usr/bin/env bash
# E4 veto check: the regime's frozen candidate on Tier V, one pass. Refuses if the candidate is not the ensemble's, or the ledger holds a Tier V row for another candidate.
#   e4.sh --regime R [--dry-run]
# CPU only (VRAM 0). Reads prediction files; appends to results/ledger.csv.
source "$(dirname "$0")/_common.sh"
cpu_phase e4 "$@"

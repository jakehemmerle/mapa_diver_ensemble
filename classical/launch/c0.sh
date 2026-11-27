#!/usr/bin/env bash
# C0 parity: Ridge 6-band reimplementation on Tier T (official protocol), once, declared defaults.
# Fitting reads train and val labels only. Scoring needs results/LOCK.json with a
# "reproduction" grant for member ridge6 (the dry run prints the config hash) and a clean
# committed sync; without it the run stops after writing predictions (exit 3).
#   c0.sh <cpu|GPU-UUID> [--dry-run] [--regimes within cross_subject cross_session] [--stages extract fit score]
# VRAM declared when a GPU is given: 2048 MiB per fit worker (Gram products only).
source "$(dirname "$0")/_lib.sh"
npexp_launch c0 "c0${NPEXP_TAG:+-$NPEXP_TAG}" 2048 "$@"

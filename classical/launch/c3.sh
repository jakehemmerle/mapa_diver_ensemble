#!/usr/bin/env bash
# C3 controls report for the C1 and C2 candidates; refits any missing control arm.
#   c3.sh <cpu|GPU-UUID> [--dry-run] [--regimes within cross_subject cross_session] [--stages extract fit score]
# VRAM declared when a GPU is given: 2048 MiB per fit worker (Gram products only).
source "$(dirname "$0")/_lib.sh"
npexp_launch c3 "c3${NPEXP_TAG:+-$NPEXP_TAG}" 2048 "$@"

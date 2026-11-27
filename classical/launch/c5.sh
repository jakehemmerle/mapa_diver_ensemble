#!/usr/bin/env bash
# C5 Tier T, once, through the lock; if the frozen candidate belongs to the ensemble track it only
# exports the classical members' Tier T scores (no labels read).
#   c5.sh <cpu|GPU-UUID> [--dry-run] [--regimes within cross_subject cross_session] [--stages extract fit score]
# VRAM declared when a GPU is given: 2048 MiB per fit worker (Gram products only).
source "$(dirname "$0")/_lib.sh"
npexp_launch c5 "c5${NPEXP_TAG:+-$NPEXP_TAG}" 2048 "$@"

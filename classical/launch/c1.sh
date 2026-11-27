#!/usr/bin/env bash
# C1 block screen on Tier P without MAPA, with its controls and the window-discipline measurement.
#   c1.sh <cpu|GPU-UUID> [--dry-run] [--regimes within cross_subject cross_session] [--stages extract fit score]
# VRAM declared when a GPU is given: 2048 MiB per fit worker (Gram products only).
source "$(dirname "$0")/_lib.sh"
npexp_launch c1 "c1${NPEXP_TAG:+-$NPEXP_TAG}" 2048 "$@"

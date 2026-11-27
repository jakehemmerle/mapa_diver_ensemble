#!/usr/bin/env bash
# ridge6 (the Ridge 6-band reimplementation, in-window, bipolar) on every Tier P cell: the C1
# baseline and the ensemble's Ridge 6-band member on Tier P. Reads Tier P labels only.
#   p0.sh <cpu|GPU-UUID> [--dry-run] [--regimes within cross_subject cross_session] [--stages extract fit score]
# VRAM declared when a GPU is given: 2048 MiB per fit worker (Gram products only).
source "$(dirname "$0")/_lib.sh"
npexp_launch p0 "cp0${NPEXP_TAG:+-$NPEXP_TAG}" 2048 "$@"

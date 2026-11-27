#!/usr/bin/env bash
# C2 kept blocks stacked on MAPA's last layer and frontend, Tier P, with controls; writes the
# classical proposal (results/frozen/proposals/<regime>.classical.json). Needs MAPA kernels for
# Tier P (ensemble/launch) and the mapa12 member's Tier P predictions.
#   c2.sh <cpu|GPU-UUID> [--dry-run] [--regimes within cross_subject cross_session] [--stages extract fit score]
# VRAM declared when a GPU is given: 2048 MiB per fit worker (Gram products only).
source "$(dirname "$0")/_lib.sh"
npexp_launch c2 "c2${NPEXP_TAG:+-$NPEXP_TAG}" 2048 "$@"

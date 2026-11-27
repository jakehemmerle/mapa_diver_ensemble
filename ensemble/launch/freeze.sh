#!/usr/bin/env bash
# Freeze the regime's single candidate from the tracks' Tier P proposals (largest gain
# among Go, or Hold with clean controls; cross_session inherits within). Written once.
#   freeze.sh --regime R [--dry-run]
# Foreground, CPU, seconds.
source "$(dirname "$0")/_common.sh"
dry_run ensemble.run freeze "$@"

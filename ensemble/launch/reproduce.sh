#!/usr/bin/env bash
# Member parity against the reproduction and the leaderboard on the 12 evaluation sessions.
#   reproduce.sh parity    --regime R [--dry-run]              no test label: refit mapa12/mapa0 from the stored kernels
#                                                              and compare the chosen penalty (and validation AUROC)
#                                                              with the reproduction's finished sessions
#   reproduce.sh reproduce --regime R --member M [--dry-run]   official-protocol test scores; reads test labels
#                                                              through LOCK.json (purpose "reproduction"), logged
# CPU only (VRAM 0). "reproduce" needs the member's Tier T export (e0.sh ... --tier T); "parity" only the kernels.
source "$(dirname "$0")/_common.sh"
mode=${1:-}; shift || true
[[ "$mode" == parity || "$mode" == reproduce ]] || die "usage: $0 parity|reproduce --regime R [--member M] [--dry-run]" 64
regime=$(flag_value --regime "$@"); member=$(flag_value --member "$@")
[[ -n "$regime" ]] || die "--regime is required" 64
if has_flag --dry-run "$@"; then
    dry_run ensemble.run "$mode" "$@"
else
    launch "spark-npexp-erepro-$mode-$(unit_part "$regime")${member:+-$member}" "" ensemble.run "$mode" "$@"
fi

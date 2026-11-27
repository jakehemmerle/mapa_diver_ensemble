#!/usr/bin/env bash
# Unattended chains of ensemble jobs. Each chain runs in a watcher unit named
# npexp-queue-<chain> (outside the spark-npexp-* glob); every job it starts is a normal
# spark-npexp-* unit with its own GPU, memory and disk checks. A job that is refused for
# resources (exit 4 or 5) is retried every 5 minutes; any other failure stops the chain.
#   queue.sh start [chain ...]     start watcher units (default: all chains)
#   queue.sh status                each chain's state, its current job, last log line, failures
#   queue.sh run <chain>           chain body (what the watcher unit executes)
# Chains: card0 card1 card2 card3 wave1 xsess; kshard0-3 kjoin (Tier P kernels sharded over the cards)
source "$(dirname "$0")/_common.sh"
here=$(cd "$(dirname "$0")" && pwd)
state_dir="$logs/queue"
mkdir -p "$state_dir"
CARD0=GPU-013cce7e-9628-49c2-f849-251f3304d038
CARD1=GPU-ad6718d6-1cf2-f3e7-2f2b-515f9c00859f
CARD2=GPU-b8d6fafd-0270-eac0-2f13-e430c3850c6b
CARD3=GPU-5b9864f6-979f-d311-8cd3-85810ad3c919
CHAINS=(card0 card1 card2 card3 wave1 xsess kshard0 kshard1 kshard2 kshard3 kjoin)
# Bulk intermediates and kernels live on /scratch (_common.sh sets NPEXP_SCRATCH; paths.py and
# common/mapa_features.py refuse if it is not mounted). --max-held counts per-contact feature
# sets over every kernel job on the machine; at 2, concurrent jobs refused each other (Nov 25),
# and /scratch has room for five (about 50-70 GiB each).
KERNEL_LIMITS=(--max-held 5 --disk-floor 100)
PARITY="$NPEXP_ROOT/results/parity/mapa_kernels_cs_S3T0.json"

note() { echo "$(date -Is) $*" | tee -a "$state_dir/$chain.state"; }

step() {  # description command...  (retries resource refusals; stops the chain on failure)
    local what=$1 status; shift
    while true; do
        note "start: $what"
        status=0; NPEXP_WAIT=1 "$@" || status=$?
        case $status in
            0) note "done: $what"; return 0 ;;
            4|5) note "waiting: $what refused for resources (exit $status); retry in 5 min"; sleep 300 ;;
            *) note "FAILED: $what (exit $status); chain stopped"; exit "$status" ;;
        esac
    done
}

downloading() { systemctl --user is-active --quiet npexp-data-download.service; }

until_downloaded() {  # description command...  rerun with --available-only until the download is done
    local what=$1; shift
    while downloading; do
        step "$what (recordings present so far)" "$@" --available-only
        note "waiting: $what for more recordings; recheck in 10 min"; sleep 600
    done
    step "$what (all recordings)" "$@"
}

wait_for() {  # description test-command...
    local what=$1; shift
    until "$@"; do note "waiting: $what"; sleep 600; done
}

parity_passed() { [[ -f "$PARITY" ]] && "$py" -c "import json,sys; sys.exit(0 if json.load(open('$PARITY'))['passed'] else 1)"; }
chain_done() { grep -q "^.* CHAIN DONE" "$state_dir/$1.state" 2>/dev/null; }
p_kernels_complete() { (cd "$code" && dry_run ensemble.mapa.kernel_job --store P --regimes within,cross_subject --dry-run) 2>&1 | tee -a "$state_dir/$chain.state" | grep -q "nothing to do"; }
eval_cs_idle() { ! systemctl --user is-active --quiet spark-npexp-ek-mapa-E.cross_subject.service; }
stop_unsharded_card1() {  # the card1 job started before sharding: stop it right after it deletes a session
    local unit=spark-npexp-ek-mapa-P.within.cross_subject.service log="$logs/ek-mapa-P.within.cross_subject.log" n
    n=$(grep -c '^\[delete\]' "$log" 2>/dev/null || true)
    while systemctl --user is-active --quiet $unit && (( $(grep -c '^\[delete\]' "$log") == n )); do sleep 15; done
    systemctl --user stop $unit 2>/dev/null || true
    note "unsharded card1 kernel job stopped at a session boundary"
}
clean_tree() { "$py" -c "import json,sys; sys.exit(1 if json.load(open('$code/COMMIT.json'))['dirty'] else 0)"; }
memory_for_diver() { (( $(mem_available_gib) >= 60 )); }
e0_inputs() { (cd "$code" && dry_run ensemble.run e0 --tier P --regime "$1" --wave 1 --dry-run > /dev/null 2>&1); }

run_chain() {
    case $chain in
    card0)
        step "evaluation cross_subject kernels with parity" bash "$here/mapa_kernels.sh" $CARD0 --store E --regimes cross_subject "${KERNEL_LIMITS[@]}"
        step "evaluation within kernels" bash "$here/mapa_kernels.sh" $CARD0 --store E --regimes within --skip-parity "${KERNEL_LIMITS[@]}" ;;
    card1)
        wait_for "parity has not passed yet ($PARITY)" parity_passed
        until_downloaded "Tier P within+cross_subject kernels" bash "$here/mapa_kernels.sh" $CARD1 --store P --regimes within,cross_subject --skip-parity "${KERNEL_LIMITS[@]}" ;;
    card2)
        for regime in within cross_session; do
            wait_for "MemAvailable below 60 GiB for DIVER" memory_for_diver
            until_downloaded "DIVER Tier P $regime export (seed 0, select/eval)" bash "$here/e0.sh" $CARD2 --wave 2 --member diver --tier P --regime $regime --export-only --quick
        done ;;
    card3)
        for regime in within cross_session cross_subject; do
            until_downloaded "CNN Tier P $regime export (seed 0, select/eval)" bash "$here/e0.sh" $CARD3 --wave 2 --member cnn --tier P --regime $regime --export-only --quick
        done ;;
    wave1)
        wait_for "Tier P kernels (chain card1) not finished" chain_done card1
        for regime in within cross_subject; do
            step "mapa12+mapa0 Tier P $regime export" bash "$here/e0.sh" cpu --wave 1 --tier P --regime $regime --export-only
            wait_for "ridge6 (classical track) Tier P $regime predictions incomplete" e0_inputs $regime
            step "E0 check Tier P $regime" bash "$here/e0.sh" cpu --wave 1 --tier P --regime $regime --check-only
            wait_for "E1 needs a clean committed tree (code/COMMIT.json dirty=true)" clean_tree
            step "E1 Tier P $regime" bash "$here/e1.sh" --regime $regime --wave 1
        done ;;
    xsess)      # cross-session stores, after the within and cross_subject stores; descriptive E1 on Tier P
        wait_for "within/cross_subject kernels (chains card0, card1) not finished" eval 'chain_done card0 && chain_done card1'
        wait_for "a Tier P cross_session kernel job is already running" eval '! systemctl --user is-active --quiet spark-npexp-ek-mapa-P.cross_session.service'
        step "Tier P cross_session kernels" bash "$here/mapa_kernels.sh" $CARD1 --store P --regimes cross_session --skip-parity "${KERNEL_LIMITS[@]}"
        step "evaluation cross_session kernels" bash "$here/mapa_kernels.sh" $CARD0 --store E --regimes cross_session --skip-parity "${KERNEL_LIMITS[@]}"
        step "mapa12+mapa0 Tier P cross_session export" bash "$here/e0.sh" cpu --wave 1 --tier P --regime cross_session --export-only
        wait_for "ridge6 (classical track) Tier P cross_session predictions incomplete" e0_inputs cross_session
        step "E0 check Tier P cross_session" bash "$here/e0.sh" cpu --wave 1 --tier P --regime cross_session --check-only
        wait_for "E1 needs a clean committed tree (code/COMMIT.json dirty=true)" clean_tree
        step "E1 Tier P cross_session (descriptive)" bash "$here/e1.sh" --regime cross_session --wave 1 ;;
    kshard0|kshard1|kshard2|kshard3)    # Tier P within+cross_subject kernels; shards share the plan by per-session claims
        local card=${chain#kshard} var; var=CARD$card
        case $card in
            0) wait_for "card 0 busy with the evaluation cross_subject kernels" eval_cs_idle ;;
            1) stop_unsharded_card1 ;;
            3) sleep 300 ;;        # stagger the prepare steps of the shards on cards 2 and 3
        esac
        step "Tier P within+cross_subject kernels, shard on card $card" bash "$here/mapa_kernels.sh" "${!var}" --store P --regimes within,cross_subject --skip-parity --claim --max-held 5 --disk-floor 100 ;;
    kjoin)      # releases wave1 (waits on card1) and xsess (waits on card0 and card1)
        wait_for "kernel shards not finished" eval 'chain_done kshard0 && chain_done kshard1 && chain_done kshard2 && chain_done kshard3'
        step "every Tier P within+cross_subject kernel present (dry run)" p_kernels_complete
        for c in card0 card1; do
            echo "$(date -Is) CHAIN DONE (set by kjoin: Tier P kernels written by kshard0-3; evaluation within kernels deferred until a candidate freezes)" >> "$state_dir/$c.state"
        done ;;
    *) die "unknown chain $chain (chains: ${CHAINS[*]})" 64 ;;
    esac
    note "CHAIN DONE"
}

status() {
    local c unit
    for c in "${CHAINS[@]}"; do
        printf '== %-6s watcher %-9s ' "$c" "$(systemctl --user is-active "npexp-queue-$c.service" 2>/dev/null || true)"
        tail -n 1 "$state_dir/$c.state" 2>/dev/null || echo "(not started)"
        grep -h "FAILED" "$state_dir/$c.state" 2>/dev/null | tail -n 2 | sed 's/^/   /' || true
    done
    echo "== job units"
    systemctl --user list-units --plain --no-legend --all 'spark-npexp-e*' 2>/dev/null | awk '{print "   " $1, $3, $4}'
    for unit in $(systemctl --user list-units --plain --no-legend 'spark-npexp-e*' 2>/dev/null | awk '{print $1}'); do
        local log="$logs/${unit%.service}"; log="$logs/${log#"$logs/spark-npexp-"}.log"
        [[ -f "$log" ]] && echo "   ${unit%.service}: $(grep -v '^\s*$' "$log" | tail -n 1 | cut -c1-180)"
    done
    echo "== parity: $( [[ -f "$PARITY" ]] && "$py" -c "import json; r=json.load(open('$PARITY')); print('PASS' if r['passed'] else 'FAIL', 'max forced gap', r.get('max_abs_gap'), 'near-tie', r.get('max_near_tie_gap'), 'penalty equal', r['lam_equal'], '/', r['n'])" || echo 'not run')"
    echo "== disk: / $(df -h / | awk 'NR==2{print $4}') free; /scratch $(df -h "$NPEXP_SCRATCH" | awk 'NR==2{print $4}') free ($(mountpoint -q /scratch && echo mounted || echo NOT MOUNTED)); MemAvailable $(mem_available_gib) GiB"
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | sed 's/^/   gpu /'
}

case ${1:-} in
    start)
        shift
        for chain in "${@:-${CHAINS[@]}}"; do
            if systemctl --user is-active --quiet "npexp-queue-$chain.service"; then echo "npexp-queue-$chain already running"; continue; fi
            systemd-run --user --unit="npexp-queue-$chain" --collect --working-directory="$code" \
                -p StandardOutput="append:$state_dir/$chain.log" -p StandardError="append:$state_dir/$chain.log" \
                --setenv=NPEXP_ROOT="$NPEXP_ROOT" /run/current-system/sw/bin/bash "$here/queue.sh" run "$chain"
        done ;;
    status) status ;;
    run) chain=${2:?chain}; run_chain ;;
    *) die "usage: $0 start [chain ...] | status | run <chain>" 64 ;;
esac

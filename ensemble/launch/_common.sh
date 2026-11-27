# Shared plumbing for the ensemble launchers. Sourced, never executed.
# Rules it enforces: units are named spark-npexp-e*; a GPU job names one GPU by UUID,
# declares its VRAM need and refuses to start if the card has less free; --dry-run runs
# in the foreground on the CPU and starts nothing; no launcher touches gpu-inference.service.
# Optional environment: NPEXP_SCRATCH (bulk intermediates; default /scratch/neuroprobe-experiments,
# the provisioned NVMe; ensemble/paths.py refuses if it resolves onto /), NPEXP_DISK_FLOOR_GIB
# (GiB kept free on the scratch filesystem; default 100, and never less than 250 on the system
# disk), NPEXP_MEMORY_MAX, NPEXP_THREADS.
set -euo pipefail

export NPEXP_ROOT=${NPEXP_ROOT:-$HOME/runs/neuroprobe-experiments}
export NPEXP_SCRATCH=${NPEXP_SCRATCH:-/scratch/neuroprobe-experiments}
export NPEXP_DISK_FLOOR_GIB=${NPEXP_DISK_FLOOR_GIB:-100}
code="$NPEXP_ROOT/code"
py="$NPEXP_ROOT/.venv/bin/python"
logs="$NPEXP_ROOT/logs"
threads=${NPEXP_THREADS:-6}
memory_max=${NPEXP_MEMORY_MAX:-${memory_default:-16G}}   # launchers set memory_default before sourcing
lib_path="/run/opengl-driver/lib:/run/current-system/sw/share/nix-ld/lib"

die() { echo "$1" >&2; exit "${2:-1}"; }

has_flag() {  # flag args...
    local flag=$1; shift
    for arg in "$@"; do [[ "$arg" == "$flag" ]] && return 0; done
    return 1
}

flag_value() {  # --name args... -> value or empty
    local name=$1; shift
    while (($#)); do
        if [[ "$1" == "$name" ]]; then echo "${2:-}"; return 0; fi
        shift
    done
}

unit_part() { echo "$1" | tr -c 'A-Za-z0-9._\n' '.' ; }

gpu_free_mib() {  # uuid -> free MiB, or exit 4 if the card is unknown
    local row
    row=$(nvidia-smi --query-gpu=uuid,memory.total,memory.used --format=csv,noheader,nounits \
          | awk -F', *' -v u="$1" '$1 == u { print $2 - $3 }')
    [[ -n "$row" ]] || die "GPU $1 not found by nvidia-smi. Nothing was started." 4
    echo "$row"
}

require_gpu() {  # uuid need_mib
    [[ "$1" == GPU-* ]] || die "first argument must be a GPU UUID (nvidia-smi --query-gpu=uuid --format=csv)" 64
    local free
    free=$(gpu_free_mib "$1")
    if (( free < $2 )); then
        die "GPU $1 has $free MiB free; this job declares $2 MiB. Nothing was started." 4
    fi
    echo "GPU $1: $free MiB free, job declares $2 MiB"
}

dry_run() {  # module args...  (foreground, CPU only, starts nothing)
    echo "MemAvailable $(mem_available_gib) GiB now; the job is capped at $memory_max and needs that plus 20 GiB available"
    cd "$code"
    CUDA_VISIBLE_DEVICES="" PYTHONPATH="$code" ROOT_DIR_BRAINTREEBANK="$NPEXP_ROOT/data_dev" \
        OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
        nice -n 19 "$py" -m "$@"
}

mem_available_gib() { awk '/^MemAvailable:/ { printf "%d", $2 / 1048576 }' /proc/meminfo; }

require_memory() {  # refuse unless MemAvailable >= this unit's MemoryMax + 20 GiB
    local cap=${memory_max%G} have
    have=$(mem_available_gib)
    [[ "$cap" =~ ^[0-9]+$ ]] || die "memory cap $memory_max must be whole GiB, e.g. 24G" 64
    if (( have < cap + 20 )); then
        die "MemAvailable is $have GiB; this job is capped at $cap GiB and needs $((cap + 20)) GiB available. Nothing was started." 4
    fi
    echo "MemAvailable $have GiB, job capped at $cap GiB (needs $((cap + 20)) GiB available)"
}

launch() {  # unit gpu_uuid_or_empty module args...
    local unit=$1 gpu=$2; shift 2
    [[ "$unit" == spark-npexp-e* ]] || die "unit name $unit does not start with spark-npexp-e" 64
    require_memory
    if systemctl --user is-active --quiet "$unit.service"; then
        die "$unit is already running. Nothing was started." 5
    fi
    mkdir -p "$logs"
    local log="$logs/${unit#spark-npexp-}.log"
    local passed=()   # optional settings: scratch root, disk floor, kernel-store filesystem, test purpose
    local name
    for name in NPEXP_SCRATCH NPEXP_DISK_FLOOR_GIB NPEXP_KERNEL_STORE_FS NPEXP_REPRO_ROOT; do
        [[ -n "${!name:-}" ]] && passed+=(--setenv="$name=${!name}")
    done
    local status=0 wait=()     # NPEXP_WAIT=1 (set by queue.sh): block until the unit ends, return its status
    [[ -n "${NPEXP_WAIT:-}" ]] && wait=(--wait --quiet)
    systemd-run --user --unit="$unit" --collect "${wait[@]}" --working-directory="$code" \
        -p Nice=10 -p MemoryMax="$memory_max" \
        -p StandardOutput="append:$log" -p StandardError="append:$log" \
        --setenv=NPEXP_ROOT="$NPEXP_ROOT" --setenv=PYTHONPATH="$code" \
        --setenv=ROOT_DIR_BRAINTREEBANK="$NPEXP_ROOT/data_dev" \
        --setenv=CUDA_VISIBLE_DEVICES="$gpu" --setenv=LD_LIBRARY_PATH="$lib_path" \
        --setenv=OMP_NUM_THREADS="$threads" --setenv=OPENBLAS_NUM_THREADS="$threads" \
        --setenv=MKL_NUM_THREADS="$threads" --setenv=PYTHONUNBUFFERED=1 --setenv=TQDM_DISABLE=1 \
        "${passed[@]}" \
        "$py" -m "$@" || status=$?
    echo "${wait:+finished}${wait:-started} $unit (status $status); log $log"
    return $status
}

cpu_phase() {  # phase args...   (decision phases: CPU only, VRAM 0)
    local phase=$1; shift
    local regime wave
    regime=$(flag_value --regime "$@"); wave=$(flag_value --wave "$@")
    [[ -n "$regime" ]] || die "usage: $0 --regime within|cross_session|cross_subject [--wave 1|2] [--dry-run]" 64
    if has_flag --dry-run "$@"; then
        dry_run ensemble.run "$phase" "$@"
    else
        launch "spark-npexp-$phase-$(unit_part "$regime")${wave:+-w$wave}" "" ensemble.run "$phase" "$@"
    fi
}

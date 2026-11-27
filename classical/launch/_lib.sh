# Shared by the classical launchers; sourced, not run.
#
#   <launcher> <cpu|GPU-UUID> [--dry-run] [classical.run options]
#
# The first argument is where Gram products run: "cpu" (default design) or the UUID of one
# card. With a UUID the launcher declares its VRAM need (VRAM_MIB_PER_FIT_WORKER x fit
# workers), reads free memory from nvidia-smi and exits 3 without starting if it is short.
# --dry-run validates inputs, lists cells and prints time, disk, RAM and VRAM estimates in
# the foreground, CPU only, and starts nothing. A real launch is a transient user unit
# named spark-npexp-<tag>; it never touches gpu-inference.service or any system unit.
#
# Environment: NPEXP_ROOT, NPEXP_WORKERS (feature processes, default 12), NPEXP_FIT_WORKERS
# (default 8), NPEXP_THREADS (BLAS threads per fit worker, default 4), NPEXP_MEMORY_MAX
# (default 64G), NPEXP_CLASSICAL_CACHE (classical cache root, default $NPEXP_ROOT/work/classical;
# the disk floor is checked on the filesystem that holds it).
set -euo pipefail

npexp_launch() {
    local phase=$1 tag=$2 vram_per_worker=$3
    shift 3
    local usage="usage: $(basename "$0") <cpu|GPU-UUID> [--dry-run] [classical.run options]"
    local gpu=${1:?$usage}
    shift
    local root=${NPEXP_ROOT:-$HOME/runs/neuroprobe-experiments}
    local workers=${NPEXP_WORKERS:-12} fit_workers=${NPEXP_FIT_WORKERS:-8} threads=${NPEXP_THREADS:-4}
    local memory=${NPEXP_MEMORY_MAX:-64G}
    local dry=0 args=() arg
    for arg in "$@"; do
        if [[ $arg == --dry-run ]]; then dry=1; else args+=("$arg"); fi
    done
    local python="$root/.venv/bin/python"
    [[ -x $python ]] || { echo "missing $python: run server/setup.sh" >&2; exit 2; }
    [[ -f $root/code/classical/run.py ]] || { echo "missing $root/code: run server/sync.sh" >&2; exit 2; }

    local device=cpu visible="" need=0
    if [[ $gpu != cpu ]]; then
        need=$((vram_per_worker * fit_workers))
        local line total used free
        line=$(nvidia-smi --query-gpu=uuid,memory.total,memory.used --format=csv,noheader,nounits | grep -F "$gpu" || true)
        [[ -n $line ]] || { echo "no GPU with UUID $gpu (nvidia-smi -L lists them)" >&2; exit 3; }
        total=$(cut -d, -f2 <<<"$line" | tr -d ' ')
        used=$(cut -d, -f3 <<<"$line" | tr -d ' ')
        free=$((total - used))
        echo "GPU $gpu: ${free} MiB free of ${total}; this job declares ${need} MiB (${vram_per_worker} MiB x ${fit_workers} fit workers)"
        if ((free < need)); then
            echo "refusing to start: not enough free GPU memory" >&2
            exit 3
        fi
        device=cuda visible=$gpu
    else
        echo "Gram products on CPU; no GPU memory is used"
    fi

    local environment=(
        "NPEXP_ROOT=$root" "ROOT_DIR_BRAINTREEBANK=$root/data_dev" "PYTHONPATH=$root/code"
        "HDF5_USE_FILE_LOCKING=FALSE" "NPEXP_GRAM_DEVICE=$device"
        "OMP_NUM_THREADS=$threads" "OPENBLAS_NUM_THREADS=$threads" "MKL_NUM_THREADS=$threads"
        "LD_LIBRARY_PATH=/run/opengl-driver/lib:/run/current-system/sw/share/nix-ld/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        ${NPEXP_CLASSICAL_CACHE:+"NPEXP_CLASSICAL_CACHE=$NPEXP_CLASSICAL_CACHE"}
    )
    local command=("$python" -u -m classical.run "$phase" --workers "$workers" --fit-workers "$fit_workers" "${args[@]}")
    local unit="spark-npexp-$tag"
    local cores=$((workers > fit_workers * threads ? workers : fit_workers * threads))

    if ((dry)); then
        echo "would start unit $unit: MemoryMax=$memory CPUQuota=$((cores * 100))% Nice=10; log $root/logs/$tag-<time>.log"
        cd "$root/code"
        # The dry run itself is CPU only whatever the device argument.
        exec env "${environment[@]}" CUDA_VISIBLE_DEVICES="" nice -n 19 "${command[@]}" --dry-run
    fi

    if systemctl --user is-active --quiet "$unit.service"; then
        echo "$unit is already running" >&2
        exit 4
    fi
    mkdir -p "$root/logs"
    local log="$root/logs/$tag-$(date +%Y%m%dT%H%M%S).log"
    local setenv=() pair
    for pair in "${environment[@]}" "CUDA_VISIBLE_DEVICES=$visible"; do setenv+=("--setenv=$pair"); done
    systemd-run --user --unit="$unit" --collect -p Nice=10 -p MemoryMax="$memory" -p CPUQuota="$((cores * 100))%" \
        -p StandardOutput="append:$log" -p StandardError="append:$log" --working-directory="$root/code" \
        "${setenv[@]}" "${command[@]}"
    echo "started $unit; log $log; stop with: systemctl --user stop $unit"
}

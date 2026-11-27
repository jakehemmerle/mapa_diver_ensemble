#!/usr/bin/env bash
# Build ~/runs/neuroprobe-experiments on the GPU server. Run ON the server after
# server/sync.sh has copied the code:  bash ~/runs/neuroprobe-experiments/code/server/setup.sh
# Modeled on ~/runs/neuroprobe-reproduction/setup.sh (same pinned MAPA and Neuroprobe
# revisions; torch pinned to the version that venv ended up with, see its logs/setup.log).
# CPU only; safe to re-run. Touches nothing under ~/runs/neuroprobe-reproduction.
set -euo pipefail
root=${NPEXP_ROOT:-$HOME/runs/neuroprobe-experiments}
repro=${NPEXP_REPRO:-$HOME/runs/neuroprobe-reproduction}
MAPA_REV=dc582b2ab18fa57dce3e88f15240f50d5727c1c1
NEUROPROBE_REV=f9b084241aeeaab548ca45ec98368b70141465a8
export CUDA_VISIBLE_DEVICES=""
export UV_CONCURRENT_DOWNLOADS=4 UV_CONCURRENT_BUILDS=2 UV_CONCURRENT_INSTALLS=4

mkdir -p "$root"/{code,data_dev,work,results,logs,src}
cd "$root"

fetch_rev() {  # dir url rev
    if [[ "$(git -C "$1" rev-parse HEAD 2>/dev/null || true)" != "$3" ]]; then
        git init -q "$1"
        git -C "$1" fetch -q --depth 1 "$2" "$3"
        git -C "$1" checkout -q FETCH_HEAD
    fi
    [[ "$(git -C "$1" rev-parse HEAD)" == "$3" ]]
}
fetch_rev src/MAPA https://github.com/bentang18/MAPA "$MAPA_REV"
fetch_rev src/neuroprobe https://github.com/insight-neuro/neuroprobe "$NEUROPROBE_REV"

[[ -x .venv/bin/python ]] || uv venv --python 3.11 .venv
export VIRTUAL_ENV="$root/.venv"
py=.venv/bin/python
uv pip install --python $py "torch==2.15.0" "torchvision==0.29.2"
uv pip install --python $py -e 'src/MAPA[preprocessing]' -e src/neuroprobe
for req in code/classical/requirements.txt code/ensemble/requirements.txt; do
    if [[ -f "$req" ]]; then uv pip install --python $py -r "$req"; else echo "setup: $req not present, skipped"; fi
done

# data_dev is the merged ROOT_DIR_BRAINTREEBANK: read-only symlinks to everything the
# reproduction already holds (metadata for all 26 sessions and the 12 evaluation
# recordings); server/download_dev.py adds the 13 Tier P recordings as real files.
for entry in "$repro"/data/*; do
    name=$(basename "$entry")
    [[ -e "data_dev/$name" || -L "data_dev/$name" ]] || ln -s "$entry" "data_dev/$name"
done

ROOT_DIR_BRAINTREEBANK="$root/data_dev" PYTHONPATH="$root/code" nice -n 19 $py - <<'EOF'
import numpy, scipy, sklearn, pandas, h5py, torch
import neuroprobe, mapa
import common, common.splits, common.ridge, classical.features
print("setup ok:", "numpy", numpy.__version__, "scipy", scipy.__version__, "sklearn", sklearn.__version__,
      "torch", torch.__version__, "cuda visible:", torch.cuda.is_available())
EOF
uv pip freeze --python $py > "$root/logs/venv-freeze.txt"

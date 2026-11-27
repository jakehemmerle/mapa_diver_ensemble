#!/usr/bin/env bash
# Copy neuroprobe_experiments/ to the GPU server (never --delete) and stamp the copy
# with the commit it came from. Run locally:  bash server/sync.sh
set -euo pipefail
here=$(cd "$(dirname "$0")/.." && pwd)
host=${NPEXP_HOST:-gpu}
dest=${NPEXP_REMOTE_ROOT:-runs/neuroprobe-experiments}
ssh_opts=(-o BatchMode=yes -o ConnectTimeout=8 -o StrictHostKeyChecking=yes)

commit=$(git -C "$here" rev-parse HEAD 2>/dev/null || echo unknown)
if [[ -n "$(git -C "$here" status --porcelain -- . 2>/dev/null)" ]]; then dirty=true; else dirty=false; fi
# Content hash of the code actually shipped, so a dirty tree is still identifiable.
tree=$(cd "$here" && find . -type f \( -name '*.py' -o -name '*.sh' -o -name '*.txt' -o -name '*.md' \) \
        -not -path '*/__pycache__/*' -not -path './.pytest_cache/*' | LC_ALL=C sort | xargs shasum -a 256 | shasum -a 256 | cut -d' ' -f1)
stamp=$(mktemp /tmp/npexp-commit.XXXXXX)
frozen=""; prefix=$(git -C "$here" rev-parse --show-prefix)
for regime in within cross_subject cross_session; do
    blob="$commit:${prefix}frozen-designs/$regime.json"
    if git -C "$here" cat-file -e "$blob" 2>/dev/null; then
        hash=$(git -C "$here" show "$blob" | shasum -a 256 | cut -d' ' -f1)
        frozen="${frozen:+$frozen,}\"$regime\":\"$hash\""
    fi
done
printf '{"commit":"%s","dirty":%s,"tree_sha256":"%s","synced":"%s","frozen_design_sha256":{%s}}\n' \
    "$commit" "$dirty" "$tree" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$frozen" > "$stamp"

ssh "${ssh_opts[@]}" "$host" "mkdir -p ~/$dest/code"
rsync -a -e "ssh ${ssh_opts[*]}" --exclude '__pycache__' --exclude '.pytest_cache' --exclude 'COMMIT.json' \
    "$here/" "$host:$dest/code/"
rsync -a -e "ssh ${ssh_opts[*]}" --chmod=F644 "$stamp" "$host:$dest/code/COMMIT.json"
rm -f "$stamp"
echo "synced $here -> $host:~/$dest/code  commit=$commit dirty=$dirty tree=$tree"

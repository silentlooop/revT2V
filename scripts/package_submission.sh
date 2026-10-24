#!/usr/bin/env bash
# Build submission.zip: code, configs, prompts, notebooks, tests, docs,
# README, report_assets/, results/final/ (if present), logs/ (if present).
# Excludes: data/, checkpoints/, *.pt, *.safetensors, caches, .git, .env,
# node_modules/, .venv/, and anything over 20 MB.
set -euo pipefail
cd "$(dirname "$0")/.."

OUT="submission.zip"
MAX_BYTES=$((20 * 1024 * 1024))
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

INCLUDE_DIRS=(revt2v scripts configs prompts notebooks tests docs client server kaggle)
INCLUDE_FILES=(README.md STRUCTURE.md pyproject.toml SUBMISSION_CHECKLIST.md)
EXTRA_DIRS=(report_assets results/final logs)

EXCLUDE_PRUNE=(
  -path "*/.git" -o
  -path "*/.venv" -o
  -path "*/node_modules" -o
  -path "*/__pycache__" -o
  -path "*/.pytest_cache" -o
  -path "*/.mypy_cache" -o
  -path "*/.ruff_cache" -o
  -path "*/*.egg-info" -o
  -path "*/data" -o
  -path "*/checkpoints" -o
  -path "*/.ipynb_checkpoints"
)

file_list=()

add_path() {
  local src="$1"
  [ -e "$src" ] || return 0
  while IFS= read -r -d '' f; do
    case "$f" in
      *.pt|*.pth|*.safetensors|*.ckpt|*.DS_Store) continue ;;
      *.env|*/.env) continue ;;
    esac
    local size
    size=$(stat -f%z "$f" 2>/dev/null || stat -c%s "$f" 2>/dev/null || echo 0)
    if [ "$size" -gt "$MAX_BYTES" ]; then
      echo "SKIP (>20MB, $((size / 1024 / 1024)) MB): $f" >&2
      continue
    fi
    file_list+=("$f")
  done < <(find "$src" \( "${EXCLUDE_PRUNE[@]}" \) -prune -o -type f -print0)
}

for d in "${INCLUDE_DIRS[@]}" "${EXTRA_DIRS[@]}"; do
  add_path "$d"
done
for f in "${INCLUDE_FILES[@]}"; do
  [ -f "$f" ] && file_list+=("$f")
done

if [ -f results/final.zip ]; then
  size=$(stat -f%z results/final.zip 2>/dev/null || stat -c%s results/final.zip 2>/dev/null || echo 0)
  if [ "$size" -le "$MAX_BYTES" ]; then
    file_list+=("results/final.zip")
  else
    echo "SKIP (>20MB, $((size / 1024 / 1024)) MB): results/final.zip" >&2
  fi
fi

if [ "${#file_list[@]}" -eq 0 ]; then
  echo "Nothing to package -- check that revt2v/, scripts/, etc. exist." >&2
  exit 1
fi

rm -f "$OUT"
printf '%s\n' "${file_list[@]}" | zip -q "$OUT" -@

echo
echo "=== Files in $OUT (${#file_list[@]}) ==="
printf '%s\n' "${file_list[@]}" | sort

total_size=$(stat -f%z "$OUT" 2>/dev/null || stat -c%s "$OUT" 2>/dev/null)
echo
echo "=== $OUT: $((total_size / 1024 / 1024)) MB ($total_size bytes) ==="

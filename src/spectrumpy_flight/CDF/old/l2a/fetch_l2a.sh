#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TMP_DATA_DIR="$(mktemp -d)"

cleanup() {
  rm -rf "$TMP_DATA_DIR"
}
trap cleanup EXIT

mapfile -t remote_files < <(
  imap-data-access query \
    --instrument idex \
    --data-level l2a \
    --descriptor sci-1week \
    --version latest \
    --extension cdf \
    --output-format json \
  | python3 -c 'import ast, sys; print("\n".join(item["file_path"] for item in ast.literal_eval(sys.stdin.read())))'
)

if [[ ${#remote_files[@]} -eq 0 ]]; then
  echo "No matching l2a sci-1week files found." >&2
  exit 1
fi

declare -A wanted_files=()

for remote_path in "${remote_files[@]}"; do
  base_name="$(basename "$remote_path")"
  wanted_files["$base_name"]=1

  echo "Downloading $base_name"
  imap-data-access --data-dir "$TMP_DATA_DIR" download "$remote_path"

  downloaded_file="$TMP_DATA_DIR/$remote_path"
  if [[ ! -f "$downloaded_file" ]]; then
    echo "Expected downloaded file not found: $downloaded_file" >&2
    exit 1
  fi

  mv -f "$downloaded_file" "$SCRIPT_DIR/$base_name"
done

shopt -s nullglob
for local_file in "$SCRIPT_DIR"/imap_idex_l2a_sci-1week_*.cdf; do
  local_base="$(basename "$local_file")"
  if [[ -z "${wanted_files[$local_base]:-}" ]]; then
    echo "Removing stale file $local_base"
    rm -f "$local_file"
  fi
done
shopt -u nullglob

echo "Fetch complete. Current latest sci-1week files are in $SCRIPT_DIR"

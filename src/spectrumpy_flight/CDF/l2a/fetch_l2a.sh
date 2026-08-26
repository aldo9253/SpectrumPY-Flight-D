#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TMP_DATA_DIR="$(mktemp -d)"
OLD_DIR="$(cd "$SCRIPT_DIR/.." && pwd)/old/$(basename "$SCRIPT_DIR")"
DATA_LEVEL="l2a"
DESCRIPTOR="sci-10days"

cleanup() {
  rm -rf "$TMP_DATA_DIR"
}
trap cleanup EXIT

mkdir -p "$OLD_DIR"

require_imap_data_access_version() {
  python3 -c '
from importlib.metadata import version

installed = tuple(map(int, version("imap-data-access").split(".")[:3]))
required = (0, 42, 0)
if installed < required:
    raise SystemExit(
        "imap-data-access >= 0.42.0 is required for vMMM.mmmm CDF filenames; "
        f"found {installed[0]}.{installed[1]}.{installed[2]}."
    )
'
}

require_imap_data_access_version

if ! query_json="$(
  imap-data-access query \
    --instrument idex \
    --data-level "$DATA_LEVEL" \
    --descriptor "$DESCRIPTOR" \
    --version latest \
    --extension cdf \
    --output-format json
)"; then
  echo "Unable to query the IMAP data archive." >&2
  exit 1
fi

if ! remote_file_output="$(
  python3 -c '
import json
import sys

items = json.loads(sys.stdin.read())
if not isinstance(items, list):
    raise SystemExit("The archive query did not return a JSON list.")
for item in items:
    file_path = item.get("file_path")
    if file_path:
        print(file_path)
' <<< "$query_json"
)"; then
  echo "The archive query returned invalid JSON." >&2
  exit 1
fi

remote_files=()
if [[ -n "$remote_file_output" ]]; then
  mapfile -t remote_files <<< "$remote_file_output"
fi

if [[ ${#remote_files[@]} -eq 0 ]]; then
  echo "No matching $DATA_LEVEL $DESCRIPTOR files found." >&2
  exit 1
fi

parse_cdf_name() {
  local file_name="$1"
  if [[ "$file_name" =~ ^(.+)_v([0-9]{3})\.([0-9]{4})\.cdf$ ]]; then
    printf '%s %s %s\n' "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}" "${BASH_REMATCH[3]}"
  else
    return 1
  fi
}

version_key() {
  local major="$1"
  local minor="$2"
  printf '%d\n' "$((10#$major * 10000 + 10#$minor))"
}

archive_file() {
  local file_path="$1"
  local file_name dest stem n

  file_name="$(basename "$file_path")"
  dest="$OLD_DIR/$file_name"

  if [[ -e "$dest" ]]; then
    if cmp -s "$file_path" "$dest"; then
      rm -f "$file_path"
      return
    fi

    stem="${file_name%.cdf}"
    n=1
    while [[ -e "$OLD_DIR/${stem}.old${n}.cdf" ]]; do
      ((n++))
    done
    dest="$OLD_DIR/${stem}.old${n}.cdf"
  fi

  mv "$file_path" "$dest"
}

archive_legacy_cdfs() {
  local local_file local_base

  shopt -s nullglob
  for local_file in "$SCRIPT_DIR"/*_v*.cdf; do
    local_base="$(basename "$local_file")"
    if [[ "$local_base" =~ ^.+_v[0-9]+\.cdf$ ]]; then
      echo "Archiving legacy CDF name $local_base"
      archive_file "$local_file"
    fi
  done
  shopt -u nullglob
}

archive_legacy_cdfs

for remote_path in "${remote_files[@]}"; do
  base_name="$(basename "$remote_path")"
  if ! read -r remote_stem remote_major remote_minor < <(parse_cdf_name "$base_name"); then
    echo "Skipping unrecognized remote CDF name: $base_name" >&2
    continue
  fi

  remote_version_key="$(version_key "$remote_major" "$remote_minor")"
  remote_version_label="v$remote_major.$remote_minor"
  latest_local_file=""
  latest_local_version_key=-1
  latest_local_version_label=""

  shopt -s nullglob
  for local_file in "$SCRIPT_DIR/$remote_stem"_v*.cdf; do
    local_base="$(basename "$local_file")"
    if ! read -r local_stem local_major local_minor < <(parse_cdf_name "$local_base"); then
      continue
    fi
    if [[ "$local_stem" != "$remote_stem" ]]; then
      continue
    fi

    local_version_key="$(version_key "$local_major" "$local_minor")"
    if (( local_version_key > latest_local_version_key )); then
      latest_local_version_key="$local_version_key"
      latest_local_version_label="v$local_major.$local_minor"
      latest_local_file="$local_file"
    fi
  done
  shopt -u nullglob

  if (( latest_local_version_key >= remote_version_key )); then
    echo "Skipping $base_name; local $(basename "$latest_local_file") is version $latest_local_version_label"
    continue
  fi

  if [[ -n "$latest_local_file" ]]; then
    echo "Downloading $base_name ($remote_version_label); newer than local $(basename "$latest_local_file")"
  else
    echo "Downloading $base_name ($remote_version_label); no local copy found"
  fi
  imap-data-access --data-dir "$TMP_DATA_DIR" download "$remote_path"

  downloaded_file="$TMP_DATA_DIR/$remote_path"
  if [[ ! -f "$downloaded_file" ]]; then
    echo "Expected downloaded file not found: $downloaded_file" >&2
    exit 1
  fi

  mv -f "$downloaded_file" "$SCRIPT_DIR/$base_name"

  shopt -s nullglob
  for local_file in "$SCRIPT_DIR/$remote_stem"_v*.cdf; do
    local_base="$(basename "$local_file")"
    if [[ "$local_base" == "$base_name" ]]; then
      continue
    fi
    if ! read -r local_stem local_major local_minor < <(parse_cdf_name "$local_base"); then
      continue
    fi
    if [[ "$local_stem" != "$remote_stem" ]]; then
      continue
    fi

    local_version_key="$(version_key "$local_major" "$local_minor")"
    if (( local_version_key < remote_version_key )); then
      echo "Archiving older version $local_base"
      archive_file "$local_file"
    fi
  done
  shopt -u nullglob
done

echo "Fetch complete. Current latest $DESCRIPTOR files are in $SCRIPT_DIR"

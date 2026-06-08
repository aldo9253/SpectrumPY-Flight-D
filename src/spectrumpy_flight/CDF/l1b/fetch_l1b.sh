#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TMP_DATA_DIR="$(mktemp -d)"
DATA_LEVEL="l1b"
DESCRIPTOR="sci-10days"

cleanup() {
  rm -rf "$TMP_DATA_DIR"
}
trap cleanup EXIT

mapfile -t remote_files < <(
  imap-data-access query \
    --instrument idex \
    --data-level "$DATA_LEVEL" \
    --descriptor "$DESCRIPTOR" \
    --version latest \
    --extension cdf \
    --output-format json \
  | python3 -c 'import ast, sys; print("\n".join(item["file_path"] for item in ast.literal_eval(sys.stdin.read())))'
)

if [[ ${#remote_files[@]} -eq 0 ]]; then
  echo "No matching $DATA_LEVEL $DESCRIPTOR files found." >&2
  exit 1
fi

parse_cdf_name() {
  local file_name="$1"
  if [[ "$file_name" =~ ^(.+)_v([0-9]+)\.cdf$ ]]; then
    printf '%s %s\n' "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}"
  else
    return 1
  fi
}

version_number() {
  local version="$1"
  printf '%d\n' "$((10#$version))"
}

for remote_path in "${remote_files[@]}"; do
  base_name="$(basename "$remote_path")"
  if ! read -r remote_stem remote_version < <(parse_cdf_name "$base_name"); then
    echo "Skipping unrecognized remote CDF name: $base_name" >&2
    continue
  fi

  remote_version_num="$(version_number "$remote_version")"
  latest_local_file=""
  latest_local_version_num=-1

  shopt -s nullglob
  for local_file in "$SCRIPT_DIR/$remote_stem"_v*.cdf; do
    local_base="$(basename "$local_file")"
    if ! read -r local_stem local_version < <(parse_cdf_name "$local_base"); then
      continue
    fi
    if [[ "$local_stem" != "$remote_stem" ]]; then
      continue
    fi

    local_version_num="$(version_number "$local_version")"
    if (( local_version_num > latest_local_version_num )); then
      latest_local_version_num="$local_version_num"
      latest_local_file="$local_file"
    fi
  done
  shopt -u nullglob

  if (( latest_local_version_num >= remote_version_num )); then
    echo "Skipping $base_name; local $(basename "$latest_local_file") is version $latest_local_version_num"
    continue
  fi

  if [[ -n "$latest_local_file" ]]; then
    echo "Downloading $base_name; newer than local $(basename "$latest_local_file")"
  else
    echo "Downloading $base_name; no local copy found"
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
    if ! read -r local_stem local_version < <(parse_cdf_name "$local_base"); then
      continue
    fi
    if [[ "$local_stem" != "$remote_stem" ]]; then
      continue
    fi

    local_version_num="$(version_number "$local_version")"
    if (( local_version_num < remote_version_num )); then
      echo "Removing older version $local_base"
      rm -f "$local_file"
    fi
  done
  shopt -u nullglob
done

echo "Fetch complete. Current latest $DESCRIPTOR files are in $SCRIPT_DIR"

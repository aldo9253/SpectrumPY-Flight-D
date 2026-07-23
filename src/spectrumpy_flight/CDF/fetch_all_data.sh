#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

fetch_scripts=(
  "l1a/fetch_l1a.sh"
  "l1b/fetch_l1b.sh"
  "l2a/fetch_l2a.sh"
  "l1a_msg/fetch_l1a_msg.sh"
  "l1b_msg/fetch_l1b_msg.sh"
)

for fetch_script in "${fetch_scripts[@]}"; do
  script_path="$SCRIPT_DIR/$fetch_script"

  if [[ ! -x "$script_path" ]]; then
    echo "Fetch script is missing or not executable: $script_path" >&2
    exit 1
  fi

  echo "Running $fetch_script"
  "$script_path"
  echo
done

echo "All CDF data fetches complete."

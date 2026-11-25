#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
contact_python="${CONTACT_PYTHON:-/home/yihong/supermimic_ws/supermimic/.venv/bin/python}"
exec "$contact_python" "$repo_root/scripts/deploy/contact_sim2sim.py" "$@"

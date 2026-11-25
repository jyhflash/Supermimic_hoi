#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
isaac_python="${ISAAC_PYTHON:-/home/yihong/anaconda3/envs/env_isaaclab/bin/python}"
exec "$isaac_python" "$repo_root/scripts/deploy/contact_isaac.py" "$@"

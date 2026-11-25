#!/usr/bin/env bash
# Loopback-only commissioning test. Never uses DDS domain 0.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
contact_python="${CONTACT_DEPLOY_PYTHON:-$repo_root/.venv-contact-deploy/bin/python}"
scenario="${1:-normal}"
case "$scenario" in normal|vio-loss|nan|stall) ;; *) echo 'Use normal|vio-loss|nan|stall'; exit 2;; esac
run_dir="$(mktemp -d "$repo_root/deploy_artifacts/contact_dds_${scenario}_XXXXXX")"
art="$repo_root/deploy_artifacts/hoi_contact_pose_sub7_largebox_016"
"$contact_python" - "$art" "$run_dir" <<'PY'
import sys,json
from pathlib import Path
art,out=map(Path,sys.argv[1:]);c=json.loads((art/'deploy_config.json').read_text())
(out/'vio_sim.json').write_text(json.dumps(dict(frame_id=c['clip']['clip_id'],imu_yaw_offset_rad=0.,verified=True,simulation_only=True)))
PY
touch "$run_dir/band"
: > "$run_dir/buttons"
sim_pid=''
cleanup() { if [[ -n "$sim_pid" ]]; then kill "$sim_pid" 2>/dev/null || true; wait "$sim_pid" 2>/dev/null || true; fi; }
trap cleanup EXIT
"$contact_python" scripts/deploy/contact_dds_sim.py --artifacts "$art" --band-file "$run_dir/band" --buttons-file "$run_dir/buttons" --vio-drop-file "$run_dir/drop_vio" --duration 70 > "$run_dir/sim.log" 2>&1 &
sim_pid=$!
extra=()
if [[ "$scenario" == nan ]]; then extra+=(--sim-inject-nan-step 30); fi
if [[ "$scenario" == stall ]]; then extra+=(--sim-stall-at-s 0.2 --sim-stall-s 0.5); fi
if [[ "$scenario" == vio-loss ]]; then
  ( for _ in {1..500}; do
      if [[ -f "$run_dir/runner.log" ]] && rg -q 'STATE: STANDBY_PRE -> POLICY' "$run_dir/runner.log"; then touch "$run_dir/drop_vio"; exit; fi
      sleep .1
    done ) &
  fault_pid=$!
fi
set +e
timeout --signal=INT --kill-after=5 55 "$contact_python" scripts/deploy/contact_g1_runner.py --sim --domain 1 --iface lo --stage ground \
  --artifacts "$art" --clip "$art/reference.npz" --vio-calibration "$run_dir/vio_sim.json" \
  --auto-sequence --sim-band-file "$run_dir/band" --sim-exit-after-damp 1 \
  --log-dir "$run_dir" "${extra[@]}" > "$run_dir/runner.log" 2>&1
result=$?
set -e
if [[ -n "${fault_pid:-}" ]]; then kill "$fault_pid" 2>/dev/null || true; wait "$fault_pid" 2>/dev/null || true; fi
printf 'Logs: %s\nRunner exit: %s\n' "$run_dir" "$result"
# Exit alone is NOT a PASS: read STATE/ABORT/DAMP and verify the expected transition/fault.
exit "$result"

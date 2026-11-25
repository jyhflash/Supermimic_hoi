"""Read-only local VIO acquisition-clock monitor; never imports a DDS publisher."""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from supermimic.deploy.contact import UdpOdometry

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--frame-id", required=True)
p.add_argument("--port", type=int, default=15151)
p.add_argument("--seconds", type=float, default=10.0)
a = p.parse_args()
o = UdpOdometry(a.frame_id, a.port)
end = time.monotonic() + a.seconds
startup_until = time.monotonic() + 1.0
seen = set()
errors = []
max_age = 0.0
try:
    while time.monotonic() < end:
        try:
            pos, vel = o.sample()
            seen.add(o.latest["seq"])
            max_age = max(max_age, time.monotonic() - o.latest["stamp"])
        except ValueError as e:
            if seen or time.monotonic() > startup_until:
                errors.append(str(e))
        time.sleep(0.01)
finally:
    o.close()
result = dict(
    unique_packets=len(seen),
    max_age_ms=1000 * max_age,
    invalid_samples=len(errors),
    last_error=errors[-1] if errors else None,
    passed=len(seen) >= a.seconds * 40 and not errors,
)
print(json.dumps(result, indent=2))
sys.exit(0 if result["passed"] else 1)

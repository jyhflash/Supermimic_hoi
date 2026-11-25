"""Score synchronized contact-run traces; missing object/contact evidence stays unknown."""

import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--trace", type=Path, required=True)
p.add_argument("--reference", type=Path, required=True)
p.add_argument(
    "--object-track",
    type=Path,
    help="JSONL: t (same host monotonic seconds), pose_wxyz[7], actual_contacts[10] optional",
)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
if a.output.exists():
    raise FileExistsError(a.output)
rows = [json.loads(x) for x in a.trace.read_text().splitlines() if x.strip()]
rows = [r for r in rows if r["advance"]]
if not rows:
    raise ValueError("No POLICY frames in trace")
with np.load(a.reference) as z:
    state = z["state"].copy()
    labels = z["contact_labels"].copy()
frames = np.array([r["frame"] for r in rows])
actual = np.array([r["robot_state"] for r in rows])
if np.any(frames < 0) or np.any(frames >= len(state)):
    raise ValueError("Invalid frame indices")
complete = np.array_equal(frames, np.arange(len(state) - 1))
root = float(np.mean(np.linalg.norm(actual[:, :3] - state[frames, :3], axis=1)))
result = dict(
    policy_frames=len(rows),
    complete_policy_frame_sequence=complete,
    mean_root_error_m=root,
    mean_joint_rmse_rad=float(
        np.mean(np.sqrt(np.mean((actual[:, 7:36] - state[frames, 7:36]) ** 2, axis=1)))
    ),
    object_pose_verified=False,
    object_contact_coverage=None,
    tracking_criteria_pass=None,
    note="Pre-action samples vs same reference frame. Completeness alone does not prove safe termination; review runner STATE/ABORT logs and post-standby.",
)
if a.object_track:
    obs = [json.loads(x) for x in a.object_track.read_text().splitlines() if x.strip()]
    times = np.array([r["t"] for r in obs])
    if not len(times) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Object timestamps must be finite and increasing")
    selected = []
    for r in rows:
        i = int(np.argmin(abs(times - r["t"])))
        if abs(times[i] - r["t"]) > 0.02:
            raise ValueError("Object sample gap exceeds 20ms; synchronize acquisition clocks")
        selected.append(obs[i])
    pose = np.asarray([r["pose_wxyz"] for r in selected], float)
    if (
        pose.shape != (len(rows), 7)
        or not np.isfinite(pose).all()
        or np.max(abs(np.linalg.norm(pose[:, 3:7], axis=1) - 1)) > 0.05
    ):
        raise ValueError("Invalid object pose")
    ref = state[frames, 71:78]
    pos = float(np.mean(np.linalg.norm(pose[:, :3] - ref[:, :3], axis=1)))
    ori = float(
        np.degrees(
            (
                Rotation.from_quat(pose[:, [4, 5, 6, 3]]).inv()
                * Rotation.from_quat(ref[:, [4, 5, 6, 3]])
            ).magnitude()
        ).mean()
    )
    result.update(
        object_pose_verified=True,
        mean_object_position_error_m=pos,
        mean_object_orientation_error_deg=ori,
    )
    if all("actual_contacts" in r for r in selected):
        contact = np.asarray([r["actual_contacts"] for r in selected], float)
        if contact.shape != (len(rows), 10) or not np.isin(contact, [0, 1]).all():
            raise ValueError("Actual contacts must be measured binary 10D values")
        wanted = labels[frames, 1::2]
        count = wanted.sum()
        coverage = float((wanted * contact[:, 1::2]).sum() / count) if count else 1.0
        result.update(
            object_contact_coverage=coverage,
            tracking_criteria_pass=bool(
                complete and root < 0.2 and pos < 0.15 and ori < 30 and coverage >= 0.9
            ),
        )
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))

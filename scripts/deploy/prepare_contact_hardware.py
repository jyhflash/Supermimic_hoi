"""Freeze hardware constants; never opens DDS or sends motor commands."""

import argparse
import hashlib
import json
import re
from pathlib import Path
import mujoco
import numpy as np
from mjlab.asset_zoo.robots.unitree_g1 import g1_constants as g
from prepare_contact_isaac import effort_limit

p = argparse.ArgumentParser()
p.add_argument(
    "--artifacts",
    type=Path,
    default=Path(__file__).resolve().parents[2]
    / "deploy_artifacts/hoi_contact_pose_sub7_largebox_016",
)
a = p.parse_args()
c = json.loads((a.artifacts / "deploy_config.json").read_text())
m = mujoco.MjModel.from_binary_path(str(a.artifacts / "scene.mjb"))
vel = {
    id(g.G1_ACTUATOR_5020): g.ACTUATOR_5020.velocity_limit,
    id(g.G1_ACTUATOR_7520_14): g.ACTUATOR_7520_14.velocity_limit,
    id(g.G1_ACTUATOR_7520_22): g.ACTUATOR_7520_22.velocity_limit,
    id(g.G1_ACTUATOR_4010): g.ACTUATOR_4010.velocity_limit,
    id(g.G1_ACTUATOR_WAIST): g.ACTUATOR_5020.velocity_limit,
    id(g.G1_ACTUATOR_ANKLE): g.ACTUATOR_5020.velocity_limit,
}
velocity = []
for name in c["joint_names"]:
    hits = [
        ac
        for ac in g.G1_ARTICULATION.actuators
        if any(re.fullmatch(expr, name) for expr in ac.target_names_expr)
    ]
    if len(hits) != 1:
        raise ValueError(name)
    velocity.append(vel[id(hits[0])])
ids = np.array(c["actuator_ids"])
joints = m.actuator_trnid[ids, 0]
d = dict(
    joint_names=c["joint_names"],
    default_joint_pos=c["action_offset"],
    action_scale=c["action_scale"],
    kp=m.actuator_gainprm[ids, 0].tolist(),
    kd=(-m.actuator_biasprm[ids, 2]).tolist(),
    effort_limit=[effort_limit(m, j) for j in joints],
    joint_lower=m.jnt_range[joints, 0].tolist(),
    joint_upper=m.jnt_range[joints, 1].tolist(),
    velocity_limit=velocity,
    control_dt=c["control_dt"],
    soft_limit_factor=0.9,
)
# The Unitree protocol uses this exact motor order; never silently reorder commands.
expected = (
    [
        f"{side}_{joint}_joint"
        for side in ("left", "right")
        for joint in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
    ]
    + ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
    + [
        f"{side}_{joint}_joint"
        for side in ("left", "right")
        for joint in (
            "shoulder_pitch",
            "shoulder_roll",
            "shoulder_yaw",
            "elbow",
            "wrist_roll",
            "wrist_pitch",
            "wrist_yaw",
        )
    ]
)
assert d["joint_names"] == expected
out = a.artifacts / "hardware_config.json"
if out.exists():
    raise FileExistsError(out)
out.write_text(json.dumps(d, indent=2))
review = dict(
    schema="contact-hardware-review-v1",
    hashes={
        n: hashlib.sha256((a.artifacts / n).read_bytes()).hexdigest()
        for n in ["policy.onnx", "reference.npz", "deploy_config.json", "hardware_config.json"]
    },
    robot_variant="UNCONFIRMED",
    reviewer="",
    vio_frame_calibrated=False,
    shared_core_sim_pass=False,
    dds_fault_tests_pass=False,
    transitions_and_standby_pass=False,
    hoi_task_success=False,
)
(a.artifacts / "hardware_review.example.json").write_text(json.dumps(review, indent=2))
print(out)

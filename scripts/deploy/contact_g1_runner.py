"""Contact runtime extension to the existing G1 watchdog/state-machine runner."""

import hashlib
import json
import math
import time
from pathlib import Path
import numpy as np
import g1_runner as base
from supermimic.deploy.core import DeployConfig
from supermimic.deploy.contact import (
    ContactClip,
    ContactOnnx,
    ContactCore,
    static_contact_clip,
    UdpOdometry,
)


class Runtime:
    def __init__(self):
        self.trace = []
        self.odom = None

    def add_args(self, p):
        p.add_argument("--vio-port", type=int, default=15151)
        p.add_argument(
            "--vio-calibration", type=Path, help="JSON: frame_id, imu_yaw_offset_rad, verified=true"
        )
        p.add_argument(
            "--contact-review",
            type=Path,
            help="Recorded, completed checks for this exact artifact; required on domain 0 with policy",
        )

    def validate(self, a):
        self.args = a
        self.meta = json.loads((a.artifacts / "deploy_config.json").read_text())
        if self.meta["schema"] != "supermimic-contact-native-v1":
            raise ValueError("Wrong artifact schema")
        if a.lock_waist:
            raise ValueError("Contact policy has no validated locked-waist adapter")
        if a.log_obs:
            raise ValueError("Use contact_trace.jsonl; legacy --log-obs has the wrong dimensions")
        if len(a.clip) != 1 or a.clip[0].resolve() != (a.artifacts / "reference.npz").resolve():
            raise ValueError("Use the bundled reference.npz only")
        self.frame_id = self.meta["clip"]["clip_id"]
        self.yaw = 0.0
        if not a.no_policy:
            if a.vio_calibration is None:
                raise ValueError("Policy requires a verified VIO calibration file")
            cal = json.loads(a.vio_calibration.read_text())
            if a.domain == 0 and cal.get("simulation_only"):
                raise ValueError("Simulated VIO calibration cannot be used on hardware")
            self.yaw = float(cal["imu_yaw_offset_rad"])
            if (
                cal["frame_id"] != self.frame_id
                or cal.get("verified") is not True
                or not math.isfinite(self.yaw)
            ):
                raise ValueError("VIO calibration is invalid")
        if a.domain == 0 and not a.no_policy:
            if a.contact_review is None:
                raise ValueError(
                    "Hardware policy requires --contact-review; see README_CONTACT_HARDWARE.md"
                )
            review = json.loads(a.contact_review.read_text())
            if (
                review.get("schema") != "contact-hardware-review-v1"
                or review.get("robot_variant") != "g1_29dof"
                or not review.get("reviewer")
            ):
                raise ValueError("Hardware review/robot variant unconfirmed")
            for field in [
                "vio_frame_calibrated",
                "shared_core_sim_pass",
                "dds_fault_tests_pass",
                "transitions_and_standby_pass",
                "hoi_task_success",
            ]:
                if review.get(field) is not True:
                    raise ValueError(f"Unpassed hardware prerequisite: {field}")
            for name in [
                "policy.onnx",
                "reference.npz",
                "deploy_config.json",
                "hardware_config.json",
            ]:
                if (
                    review["hashes"].get(name)
                    != hashlib.sha256((a.artifacts / name).read_bytes()).hexdigest()
                ):
                    raise ValueError(f"Artifact changed after review: {name}")
        self.odom = UdpOdometry(self.frame_id, a.vio_port)

    def load_config(self, path):
        cfg = DeployConfig.load(path / "hardware_config.json")
        expected = (
            [
                f"{s}_{j}_joint"
                for s in ("left", "right")
                for j in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
            ]
            + ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
            + [
                f"{s}_{j}_joint"
                for s in ("left", "right")
                for j in (
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
        if cfg.joint_names != expected:
            raise ValueError("Unitree motor order mismatch")
        for k in [
            "kp",
            "kd",
            "effort_limit",
            "velocity_limit",
            "action_scale",
            "default_joint_pos",
            "joint_lower",
            "joint_upper",
        ]:
            v = getattr(cfg, k)
            if v.shape != (29,) or not np.isfinite(v).all():
                raise ValueError(f"Invalid {k}")
        if min(cfg.kp) <= 0 or min(cfg.effort_limit) <= 0 or min(cfg.velocity_limit) <= 0:
            raise ValueError("Nonpositive limits/gains")
        return cfg

    load_clip = staticmethod(ContactClip.load)
    static_clip = staticmethod(static_contact_clip)

    def load_policy(self, path, threads):
        return ContactOnnx(path / "policy.onnx", threads)

    def warmup(self, policy):
        cl = ContactClip.load(self.args.artifacts / "reference.npz")
        idx = np.minimum(np.arange(10) * 5, len(cl.reference) - 1)
        policy(cl.reference[0], (cl.reference[idx], cl.labels[idx]))

    def make_core(self, cfg, policy, clip, safety):
        return ContactCore(cfg, policy, clip, safety, self.odom, self.yaw, self.record)

    def record(self, row):
        if len(self.trace) >= 100000:
            raise RuntimeError("Contact trace full; restart session")
        self.trace.append(row)

    def close(self):
        if self.odom:
            self.odom.close()
        if self.trace:
            self.args.log_dir.mkdir(parents=True, exist_ok=True)
            path = self.args.log_dir / f"contact_trace_{time.time_ns()}.jsonl"
            with path.open("x") as f:
                for row in self.trace:
                    f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    base.main(runtime=Runtime())

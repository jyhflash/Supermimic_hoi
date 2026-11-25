"""Contact ONNX deployment core. Robot odometry is mandatory, never filled with zeros."""

from dataclasses import dataclass
import json
import socket
import time
import threading
from pathlib import Path
import numpy as np
from .core import Clip, DeployCore, StepResult, projected_gravity


def rotate(q, v):
    return v + 2 * q[0] * np.cross(q[1:], v) + 2 * np.cross(q[1:], np.cross(q[1:], v))


@dataclass
class ContactClip(Clip):
    reference: np.ndarray
    labels: np.ndarray

    @classmethod
    def load(cls, path):
        with np.load(path) as z:
            reference = z["state"][:, :71].copy().astype(np.float32)
            labels = z["contact_labels"].copy().astype(np.float32)
        if (
            reference.ndim != 2
            or reference.shape[1] != 71
            or labels.shape != (len(reference), 10)
            or len(reference) < 2
        ):
            raise ValueError("Invalid contact reference dimensions")
        if not np.isfinite(reference).all() or not np.isfinite(labels).all():
            raise ValueError("Nonfinite reference")
        goal = np.zeros((len(reference), 68), np.float32)
        goal[:, 29:58] = reference[:, 42:71]
        return cls(
            Path(path).stem,
            goal,
            reference[:, 7:36].astype(float),
            np.array([projected_gravity(q) for q in reference[:, 3:7]]),
            reference,
            labels,
        )


def static_contact_clip(clip, frame):
    ref = np.repeat(clip.reference[frame : frame + 1], 2, axis=0)
    ref[:, 36:71] = 0
    return ContactClip(
        clip.name,
        np.zeros((2, 68), np.float32),
        ref[:, 7:36].astype(float),
        np.repeat(clip.proj_grav[frame : frame + 1], 2, axis=0),
        ref,
        np.repeat(clip.labels[frame : frame + 1], 2, axis=0),
    )


class ContactOnnx:
    def __init__(self, path, threads=1):
        import onnxruntime as ort

        opt = ort.SessionOptions()
        opt.intra_op_num_threads = threads
        opt.inter_op_num_threads = 1
        opt.add_session_config_entry("session.intra_op.allow_spinning", "0")
        self.session = ort.InferenceSession(
            str(path), sess_options=opt, providers=["CPUExecutionProvider"]
        )
        expected = {
            "robot_state": [1, 71],
            "reference_robot": [1, 10, 71],
            "contact_labels": [1, 10, 10],
        }
        if {i.name: i.shape for i in self.session.get_inputs()} != expected:
            raise ValueError("Wrong ONNX input contract")

    def __call__(self, state, goals):
        ref, labels = goals
        return self.session.run(
            ["action"],
            {
                "robot_state": np.asarray(state, np.float32)[None],
                "reference_robot": np.asarray(ref, np.float32)[None],
                "contact_labels": np.asarray(labels, np.float32)[None],
            },
        )[0][0].astype(float)


class Odometry:
    """Local acquisition-clock packets, already expressed in the reference frame.

    Sender must use this host's monotonic_ns at measurement acquisition, not at
    retransmission. Repeating an old packet never refreshes its age.
    """

    def __init__(self, frame_id, max_age=0.05):
        self.frame_id = frame_id
        self.max_age = max_age
        self.latest = None
        self.error = None

    def accept(self, p, now=None):
        now = time.monotonic() if now is None else now
        try:
            if (
                p["schema"] != "contact-vio-v1"
                or p["frame_id"] != self.frame_id
                or p["tracking_valid"] is not True
            ):
                raise ValueError("VIO tracking/frame invalid")
            stamp = float(p["acquisition_monotonic_ns"]) * 1e-9
            seq = int(p["seq"])
            pos = np.asarray(p["root_pos_w"], float)
            vel = np.asarray(p["root_lin_vel_w"], float)
            if (
                pos.shape != (3,)
                or vel.shape != (3,)
                or not np.isfinite(np.r_[pos, vel, stamp]).all()
            ):
                raise ValueError("VIO values invalid")
            if now - stamp > self.max_age or stamp > now + 0.005:
                raise ValueError("VIO acquisition timestamp stale/future")
            if self.latest:
                old = self.latest
                if seq <= old["seq"] or stamp <= old["stamp"]:
                    raise ValueError("VIO duplicate/out-of-order")
                if np.linalg.norm(pos - old["pos"]) > 0.1 + 5 * (stamp - old["stamp"]):
                    raise ValueError("VIO pose jump/reset")
            self.latest = dict(seq=seq, stamp=stamp, pos=pos, vel=vel)
            self.error = None
        except (ValueError, KeyError, TypeError, OverflowError) as e:
            self.error = str(e)
            raise ValueError(self.error) from e

    def sample(self, now=None):
        now = time.monotonic() if now is None else now
        if self.error or self.latest is None:
            raise ValueError(self.error or "No VIO measurement")
        if now - self.latest["stamp"] > self.max_age:
            raise ValueError("VIO stale")
        return self.latest["pos"].copy(), self.latest["vel"].copy()


class UdpOdometry(Odometry):
    """Continuously drain packets during PASSIVE/INTERP as well as POLICY."""

    def __init__(self, frame_id, port, max_age=0.05):
        super().__init__(frame_id, max_age)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind(("127.0.0.1", port))
        self.socket.settimeout(0.02)
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._receive, daemon=True)
        self.thread.start()

    def _receive(self):
        while not self.stop.is_set():
            try:
                payload, _ = self.socket.recvfrom(8192)
            except socket.timeout:
                continue
            except OSError:
                return
            with self.lock:
                try:
                    self.accept(json.loads(payload))
                except (ValueError, UnicodeError) as e:
                    self.error = str(e)

    def sample(self, now=None):
        with self.lock:
            return super().sample(now)

    def close(self):
        self.stop.set()
        self.socket.close()
        self.thread.join(timeout=0.1)


class ContactCore(DeployCore):
    def __init__(self, cfg, policy, clip, safety, odometry, yaw_offset=0.0, log=None):
        self.odometry = odometry
        self.yaw_offset = yaw_offset
        self.log = log
        super().__init__(cfg, policy, clip, safety)

    def step(self, q, dq, gyro, quat_wxyz, advance=True):
        q, dq, gyro, quat = map(lambda x: np.asarray(x, float), (q, dq, gyro, quat_wxyz))
        # Keep the original input checks before any inference.
        abort = self._check(q, dq, gyro, quat)
        if abort:
            return StepResult(q.copy(), np.zeros(29), self.frame, False, abort, 0)
        try:
            pos, vel = self.odometry.sample()
        except ValueError as e:
            return StepResult(q.copy(), np.zeros(29), self.frame, False, str(e), 0)
        h = self.yaw_offset / 2
        yaw = np.array([np.cos(h), 0, 0, np.sin(h)])
        w, v = quat[0], quat[1:]
        quat = np.r_[
            yaw[0] * w - np.dot(yaw[1:], v), yaw[0] * v + w * yaw[1:] + np.cross(yaw[1:], v)
        ]
        quat /= np.linalg.norm(quat)
        ref = self.clip.reference
        if (
            abs(pos[2] - ref[self.frame, 2]) > 0.25
            or np.linalg.norm(pos[:2] - ref[self.frame, :2]) > 0.5
        ):
            return StepResult(
                q.copy(), np.zeros(29), self.frame, False, "Root odometry tracking error", 0
            )
        state = np.r_[pos, quat, q, vel, rotate(quat, gyro), dq]
        indices = np.minimum(self.frame + np.arange(10) * 5, len(ref) - 1)
        action = np.asarray(self.policy(state, (ref[indices], self.clip.labels[indices])), float)
        if (
            action.shape != (29,)
            or not np.isfinite(action).all()
            or np.max(abs(action)) > self.safety.max_abs_action
        ):
            return StepResult(
                q.copy(), np.zeros(29), self.frame, False, "Invalid contact action", 0
            )
        target, n = self.targets(action, q, dq)
        if self.log:
            self.log(
                dict(
                    t=time.monotonic(),
                    frame=self.frame,
                    advance=advance,
                    robot_state=state.tolist(),
                    action=action.tolist(),
                    target=target.tolist(),
                )
            )
        self.last_action = action
        if advance:
            self.frame = min(self.frame + 1, len(ref) - 1)
        return StepResult(
            target, action, self.frame, advance and self.frame >= len(ref) - 1, None, n
        )

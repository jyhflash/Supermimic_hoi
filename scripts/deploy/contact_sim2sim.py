"""Contact-policy native MuJoCo/ONNX validation. No torch or training imports.

This is a training-model transfer gate, NOT the hardware DeployCore interface.
The dynamic object's state never enters the policy. Root odometry is required.
"""

from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
from queue import SimpleQueue, Empty

import mujoco
import numpy as np
import onnxruntime as ort
from scipy.spatial.transform import Rotation


def rotation(q):
    return Rotation.from_quat(np.asarray(q)[[1, 2, 3, 0]])


class Simulation:
    def __init__(self, artifacts, deploy_core=False):
        self.cfg = c = json.loads((artifacts / "deploy_config.json").read_text())
        if c["schema"] != "supermimic-contact-native-v1":
            raise ValueError("Requires contact export, not the legacy single-person ONNX")
        if mujoco.__version__ != c["mujoco_version"]:
            raise ValueError(f"MJB requires MuJoCo {c['mujoco_version']}; got {mujoco.__version__}")
        self.model = mujoco.MjModel.from_binary_path(str(artifacts / "scene.mjb"))
        self.data = mujoco.MjData(self.model)
        self.reference_data = mujoco.MjData(self.model)
        with np.load(artifacts / "reference.npz") as ref:
            self.state = ref["state"].copy()
            self.labels = ref["contact_labels"].copy()
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        self.policy = ort.InferenceSession(
            str(artifacts / "policy.onnx"), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.q = np.array(c["joint_qadr"])
        self.v = np.array(c["joint_vadr"])
        self.rq = np.array(c["robot_root_qadr"])
        self.rv = np.array(c["robot_root_vadr"])
        self.oq = np.array(c["object_root_qadr"])
        self.ov = np.array(c["object_root_vadr"])
        self.act = np.array(c["actuator_ids"])
        self.substeps = round(c["control_dt"] / c["physics_dt"])
        self.reset()
        self.deploy_core = None
        if deploy_core:
            import sys
            sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
            from supermimic.deploy.contact import ContactCore, ContactClip, ContactOnnx
            from supermimic.deploy.core import DeployConfig, SafetyCfg
            simulation = self
            class SimOdometry:
                def sample(self):
                    state = simulation.robot_state()[0]
                    return state[:3], state[36:39]
            self.deploy_core = ContactCore(DeployConfig.load(artifacts / "hardware_config.json"),
                ContactOnnx(artifacts / "policy.onnx"), ContactClip.load(artifacts / "reference.npz"),
                SafetyCfg(), SimOdometry())

    def set_reference(self, data, s, velocities=False):
        data.qpos[self.rq] = s[:7]
        data.qpos[self.q] = s[7:36]
        data.qpos[self.oq] = s[71:78]
        if velocities:
            data.qvel[self.rv[:3]] = s[36:39]
            data.qvel[self.rv[3:]] = rotation(s[3:7]).inv().apply(s[39:42])
            data.qvel[self.v] = s[42:71]
            data.qvel[self.ov[:3]] = s[78:81]
            data.qvel[self.ov[3:]] = rotation(s[74:78]).inv().apply(s[81:84])
        mujoco.mj_forward(self.model, data)

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.set_reference(self.data, self.state[0], velocities=True)
        if getattr(self, "deploy_core", None) is not None:
            self.deploy_core.reset()
        self.frame = 0
        self.reason = "running"
        self.rows = []
        self.set_reference(self.reference_data, self.state[0])

    def robot_state(self):
        d = self.data
        root = d.qpos[self.rq]
        return np.concatenate(
            (
                root,
                d.qpos[self.q],
                d.qvel[self.rv[:3]],
                rotation(root[3:7]).apply(d.qvel[self.rv[3:]]),
                d.qvel[self.v],
            )
        ).astype(np.float32)[None]

    def step(self):
        if self.reason != "running":
            return
        ids = np.minimum(
            self.frame + np.arange(self.cfg["goal_frames"]) * self.cfg["goal_stride"],
            len(self.state) - 1,
        )
        inputs = dict(
            robot_state=self.robot_state(),
            reference_robot=self.state[ids, :71][None],
            contact_labels=self.labels[ids][None],
        )
        action = self.policy.run(None, inputs)[0][0]
        if not np.isfinite(action).all():
            raise FloatingPointError("Non-finite action")
        target = np.array(self.cfg["action_offset"]) + np.array(self.cfg["action_scale"]) * action
        if self.deploy_core is not None:
            state = self.robot_state()[0]
            result = self.deploy_core.step(state[7:36], state[42:71],
                rotation(state[3:7]).inv().apply(state[39:42]), state[3:7])
            if result.abort:
                self.reason = "deploy_abort: " + result.abort
                return
            target = result.q_target
            action = result.action
        self.data.ctrl[self.act] = target
        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.frame += 1
        ref = self.state[self.frame]
        self.set_reference(self.reference_data, ref)
        actual = self.data.qpos
        if not np.isfinite(actual).all():
            raise FloatingPointError("Non-finite physics")
        robot = actual[self.rq]
        obj = actual[self.oq]
        height = abs(robot[2] - ref[2])
        tilt = abs(
            rotation(robot[3:]).inv().apply([0, 0, -1])[2]
            - rotation(ref[3:7]).inv().apply([0, 0, -1])[2]
        )
        self.rows.append(
            dict(
                frame=self.frame,
                root_position_error_m=float(np.linalg.norm(robot[:3] - ref[:3])),
                object_position_error_m=float(np.linalg.norm(obj[:3] - ref[71:74])),
                object_orientation_error_deg=float(
                    np.degrees((rotation(obj[3:]).inv() * rotation(ref[74:78])).magnitude())
                ),
                object_displacement_m=float(np.linalg.norm(obj[:3] - self.state[0, 71:74])),
                reference_displacement_m=float(np.linalg.norm(ref[71:74] - self.state[0, 71:74])),
                max_abs_action=float(np.max(np.abs(action))),
                root_height_error_m=float(height),
                root_tilt_error=float(tilt),
            )
        )
        if height > 0.25 or tilt > 0.8:
            self.reason = "robot_failure"
        elif self.frame >= len(self.state) - 1:
            self.reason = "reference_end"

    def report(self):
        keys = ["root_position_error_m", "object_position_error_m", "object_orientation_error_deg"]
        return dict(
            clip=self.cfg["clip"]["clip_id"],
            checkpoint_step=self.cfg["step"],
            simulation="native MuJoCo, exported training model",
            shared_hardware_core=self.deploy_core is not None,
            dynamic_object=True,
            object_state_in_policy=False,
            real_robot_ready=False,
            stop_reason=self.reason,
            executed_steps=self.frame,
            expected_steps=len(self.state) - 1,
            mean_errors={k: float(np.mean([r[k] for r in self.rows])) for k in keys}
            if self.rows
            else {},
            trajectory=self.rows,
        )


class ReferenceOverlay:
    def __init__(self, server, model):
        from mjviser.conversions import merge_geoms

        self.handles = {}
        for body in range(1, model.nbody):
            if model.body_rootid[body] == 0:
                continue
            ids = np.flatnonzero(
                (model.geom_bodyid == body) & (model.geom_rgba[:, 3] > 0) & (model.geom_group <= 2)
            ).tolist()
            if not ids:
                continue
            mesh = merge_geoms(model, ids)
            self.handles[body] = server.scene.add_mesh_simple(
                f"/reference/{body}",
                vertices=mesh.vertices,
                faces=mesh.faces,
                color=(40, 210, 255),
                opacity=0.3,
                cast_shadow=False,
                receive_shadow=False,
            )

    def update(self, data, offset, visible):
        for body, handle in self.handles.items():
            handle.visible = visible
            handle.position = data.xpos[body] + offset
            handle.wxyz = data.xquat[body]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "deploy_artifacts/hoi_contact_pose_sub7_largebox_016",
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--deploy-core", action="store_true", help="Use shared hardware core and its safety/target clamps")
    parser.add_argument(
        "--steps", type=int, default=0, help="0: whole clip (headless) or interactive browser"
    )
    parser.add_argument("--port", type=int, default=8084)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.steps < 0:
        parser.error("--steps must be nonnegative")
    if args.report and args.report.exists():
        raise FileExistsError(args.report)
    sim = Simulation(args.artifacts, deploy_core=args.deploy_core)
    if args.headless:
        while sim.reason == "running" and (not args.steps or sim.frame < args.steps):
            sim.step()
    else:
        import viser
        from mjviser import ViserMujocoScene

        server = viser.ViserServer(
            host="0.0.0.0", port=args.port, label="Contact ONNX · native MuJoCo"
        )
        server.scene.set_up_direction("+z")
        scene = ViserMujocoScene(server, sim.model, num_envs=1)
        scene.site_groups_visible = [False] * 6
        scene.create_visualization_gui(camera_distance=3.5)
        overlay = ReferenceOverlay(server, sim.model)
        paused = server.gui.add_checkbox("Paused", initial_value=False)
        reference = server.gui.add_checkbox("Show reference", initial_value=True)
        speed = server.gui.add_slider("Speed", min=0.1, max=2.0, step=0.1, initial_value=1.0)
        reset = server.gui.add_button("Reset clip")
        commands = SimpleQueue()
        reset.on_click(lambda _: commands.put("reset"))
        status = server.gui.add_markdown("")

        @server.on_client_connect
        def center(client):
            root = sim.data.qpos[sim.rq[:3]]
            client.camera.position = root + [3, -3, 1.5]
            client.camera.look_at = root

        print(f"Open http://localhost:{server.get_port()}", flush=True)
        try:
            while True:
                start = time.monotonic()
                try:
                    while commands.get_nowait():
                        sim.reset()
                except Empty:
                    pass
                if (server.get_clients() or args.steps) and not paused.value:
                    sim.step()
                scene.update_from_mjdata(sim.data)
                overlay.update(sim.reference_data, scene._scene_offset, reference.value)
                status.content = f"**{sim.cfg['clip']['clip_id']}** · frame {sim.frame}/{len(sim.state) - 1}\n\n{sim.reason} · cyan = reference; solid = physical simulation"
                if args.steps and (sim.frame >= args.steps or sim.reason != "running"):
                    break
                time.sleep(max(0, sim.cfg["control_dt"] / speed.value - (time.monotonic() - start)))
        except KeyboardInterrupt:
            pass
        finally:
            server.stop()
    report = sim.report()
    if sim.reason == "running":
        report["stop_reason"] = "step_cap" if args.steps else "user_stop"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("x") as f:
            json.dump(report, f, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != "trajectory"}, indent=2))


if __name__ == "__main__":
    main()

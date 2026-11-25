"""Browser replay of recorded PhysX states; MuJoCo is used ONLY for mesh poses."""

import argparse
import json
import time
from pathlib import Path
import mujoco
import numpy as np
import viser
from mjviser import ViserMujocoScene
from contact_sim2sim import ReferenceOverlay

p = argparse.ArgumentParser(description=__doc__)
p.add_argument(
    "--artifacts",
    type=Path,
    default=Path(__file__).resolve().parents[2]
    / "deploy_artifacts/hoi_contact_pose_sub7_largebox_016",
)
p.add_argument("--report", type=Path)
p.add_argument("--port", type=int, default=8085)
p.add_argument(
    "--steps", type=int, default=0, help="Finite viewer smoke test; 0 opens interactive replay"
)
a = p.parse_args()
r = json.loads((a.report or a.artifacts / "isaac_aligned_report.json").read_text())
c = json.loads((a.artifacts / "deploy_config.json").read_text())
if r["engine"] != "Isaac Lab / PhysX" or r["checkpoint_sha256"] != c["checkpoint_sha256"]:
    raise ValueError("Report does not match artifact")
if r.get("invalid_reason"):
    raise ValueError(r["invalid_reason"])
m = mujoco.MjModel.from_binary_path(str(a.artifacts / "scene.mjb"))
d = mujoco.MjData(m)
ref = mujoco.MjData(m)
with np.load(a.artifacts / "reference.npz") as data:
    reference = data["state"].copy()
server = viser.ViserServer(port=a.port, label="Isaac PhysX trajectory replay")
server.scene.set_up_direction("+z")
scene = ViserMujocoScene(server, m, num_envs=1)
overlay = ReferenceOverlay(server, m)
scene.create_visualization_gui(camera_distance=3.5)
slider = server.gui.add_slider(
    "Recorded step", min=0, max=len(r["trajectory"]), step=1, initial_value=0
)
paused = server.gui.add_checkbox("Paused", initial_value=False)
show = server.gui.add_checkbox("Show reference", initial_value=True)
speed = server.gui.add_slider("Speed", min=0.1, max=2.0, step=0.1, initial_value=1.0)
server.gui.add_markdown(
    f"**Recorded Isaac Lab / PhysX trajectory**\n\n{r['clip']} · {r['stop_reason']}\n\nSolid = PhysX result; cyan = reference. This viewer does not simulate physics."
)


@server.on_client_connect
def center(client):
    client.camera.position = reference[0, :3] + [3, -3, 1.5]
    client.camera.look_at = reference[0, :3]


def pose(data, s):
    data.qpos[c["robot_root_qadr"]] = s[:7]
    data.qpos[c["joint_qadr"]] = s[7:36]
    data.qpos[c["object_root_qadr"]] = s[71:78]
    mujoco.mj_kinematics(m, data)


print(f"Open http://localhost:{server.get_port()} (recorded PhysX replay)", flush=True)
ticks = 0
try:
    while True:
        begin = time.monotonic()
        i = int(slider.value)
        state = reference[0] if i == 0 else np.asarray(r["trajectory"][i - 1]["state"])
        frame = 0 if i == 0 else r["trajectory"][i - 1]["frame"]
        pose(d, state)
        pose(ref, reference[frame])
        scene.update_from_mjdata(d)
        overlay.update(ref, scene._scene_offset, show.value)
        if (server.get_clients() or a.steps) and not paused.value and i < len(r["trajectory"]):
            slider.value = i + 1
        ticks += 1
        if a.steps and ticks >= a.steps:
            break
        time.sleep(max(0, 0.02 / speed.value - (time.monotonic() - begin)))
except KeyboardInterrupt:
    pass
finally:
    server.stop()

"""Loopback-only DDS contact harness, dynamic object + simulated VIO.

This exercises the hardware runner, not just ONNX. Requires unitree_sdk2py.
Only domain 1 / lo are used. Never run on the robot network.
"""

import argparse
import json
import socket
import threading
import time
from pathlib import Path
import mujoco
import numpy as np
from contact_sim2sim import Simulation, rotation
import g1_common as gc

p = argparse.ArgumentParser(description=__doc__)
p.add_argument(
    "--artifacts",
    type=Path,
    default=Path(__file__).resolve().parents[2]
    / "deploy_artifacts/hoi_contact_pose_sub7_largebox_016",
)
p.add_argument("--band-file", type=Path, required=True)
p.add_argument("--buttons-file", type=Path, required=True)
p.add_argument("--vio-drop-file", type=Path, default=Path("/tmp/contact_drop_vio"))
p.add_argument("--vio-port", type=int, default=15151)
p.add_argument("--duration", type=float, default=90.0)
a = p.parse_args()
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_
from unitree_sdk2py.utils.crc import CRC

gc.init_dds(1, "lo")
sim = Simulation(a.artifacts)
m, d = sim.model, sim.data
m.actuator_gainprm[:] = 0
m.actuator_gainprm[:, 0] = 1
m.actuator_biasprm[:] = 0
m.actuator_biastype[:] = 0
m.actuator_ctrllimited[:] = False
lock = threading.Lock()
command = None
received = 0.0
crc = CRC()


def receive(msg):
    global command, received
    if msg.crc != crc.Crc(msg):
        return
    with lock:
        command = np.array([[x.q, x.dq, x.kp, x.kd, x.tau] for x in msg.motor_cmd[:29]])
        received = time.monotonic()


sub = ChannelSubscriber("rt/lowcmd", LowCmd_)
sub.Init(receive, 10)
pub = ChannelPublisher("rt/lowstate", LowState_)
pub.Init()
udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
ls = unitree_hg_msg_dds__LowState_()
anchor = d.qpos[sim.rq[:3]].copy()
pelvis = m.jnt_bodyid[np.flatnonzero(m.jnt_qposadr == sim.rq[0])[0]]
start = time.monotonic()
seq = 0
print("DDS simulator domain=1 interface=lo; no real-robot interface is used", flush=True)
try:
    while time.monotonic() - start < a.duration:
        t = time.monotonic()
        with lock:
            cmd = None if command is None else command.copy()
            age = t - received
        if cmd is not None:
            if age > 0.1:
                cmd = np.column_stack(
                    (d.qpos[sim.q], np.zeros(29), np.zeros(29), np.full(29, 3.0), np.zeros(29))
                )
            d.ctrl[sim.act] = (
                cmd[:, 4]
                + cmd[:, 2] * (cmd[:, 0] - d.qpos[sim.q])
                + cmd[:, 3] * (cmd[:, 1] - d.qvel[sim.v])
            )
            d.xfrc_applied[pelvis, :3] = (
                2000 * (anchor - d.qpos[sim.rq[:3]]) - 200 * d.qvel[sim.rv[:3]]
                if a.band_file.exists()
                else 0
            )
            mujoco.mj_step(m, d)
            mujoco.mj_forward(m, d)
        state = sim.robot_state()[0]
        ls.mode_machine = 5
        ls.mode_pr = 0
        ls.tick = int((t - start) * 1000)
        for i in range(29):
            ls.motor_state[i].q = float(state[7 + i])
            ls.motor_state[i].dq = float(state[42 + i])
            ls.motor_state[i].temperature = [30, 30]
            ls.motor_state[i].tau_est = float(d.actuator_force[sim.act[i]])
        ls.imu_state.quaternion = state[3:7].tolist()
        ls.imu_state.gyroscope = rotation(state[3:7]).inv().apply(state[39:42]).tolist()
        keys = a.buttons_file.read_text().strip().split("+") if a.buttons_file.exists() else []
        ls.wireless_remote[2], ls.wireless_remote[3] = gc.keys_from_names([k for k in keys if k])
        ls.crc = crc.Crc(ls)
        pub.Write(ls)
        if not a.vio_drop_file.exists():
            packet = dict(
                schema="contact-vio-v1",
                frame_id=sim.cfg["clip"]["clip_id"],
                tracking_valid=True,
                acquisition_monotonic_ns=time.monotonic_ns(),
                seq=seq,
                root_pos_w=state[:3].tolist(),
                root_lin_vel_w=state[36:39].tolist(),
            )
            udp.sendto(json.dumps(packet).encode(), ("127.0.0.1", a.vio_port))
        seq += 1
        time.sleep(max(0, m.opt.timestep - (time.monotonic() - t)))
finally:
    udp.close()

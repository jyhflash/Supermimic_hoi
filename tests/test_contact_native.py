"""Native state/frame and reference-overlay boundary checks."""

import importlib.util
from pathlib import Path

import mujoco
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "contact_native", ROOT / "scripts/deploy/contact_sim2sim.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.fixture
def sim():
    artifacts = ROOT / "deploy_artifacts/hoi_contact_pose_sub7_largebox_016"
    if not (artifacts / "deploy_config.json").exists():
        pytest.skip("Export the contact artifact first")
    return module.Simulation(artifacts)


def test_native_state_roundtrip_and_world_angular_velocity(sim):
    state = sim.state[100].copy()
    state[3:7] = [0.5, 0.5, 0.5, 0.5]
    state[39:42] = [1.0, 2.0, 3.0]
    sim.set_reference(sim.data, state, velocities=True)
    np.testing.assert_allclose(sim.robot_state()[0], state[:71], atol=1e-6)
    # MuJoCo free-joint rotational qvel is local, not the actor's world velocity.
    assert not np.allclose(sim.data.qvel[sim.rv[3:]], state[39:42])


def test_reference_never_overwrites_physical_object(sim):
    qpos = sim.data.qpos.copy()
    sim.set_reference(sim.reference_data, sim.state[100])
    np.testing.assert_array_equal(sim.data.qpos, qpos)
    before = sim.robot_state()
    sim.data.qpos[sim.oq[:3]] += [1.0, 2.0, 3.0]
    mujoco.mj_forward(sim.model, sim.data)
    np.testing.assert_array_equal(sim.robot_state(), before)


def test_actuator_mapping_is_by_joint_identity(sim):
    m = sim.model
    for qadr, actuator in zip(sim.q, sim.act, strict=True):
        joint = m.actuator_trnid[actuator, 0]
        assert m.jnt_qposadr[joint] == qadr
        assert m.actuator_gainprm[actuator, 0] > 0
        assert m.actuator_biasprm[actuator, 1] < 0


def test_isaac_motor_limits_come_from_actuators(sim):
    spec = importlib.util.spec_from_file_location(
        "prepare_contact", ROOT / "scripts/deploy/prepare_contact_isaac.py"
    )
    prep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prep)
    for a in sim.act:
        j = sim.model.actuator_trnid[a, 0]
        limit = prep.effort_limit(sim.model, j)
        assert limit > 0
        assert limit == np.max(np.abs(sim.model.actuator_forcerange[a]))


def test_isaac_urdf_preserves_robot_kinematics(sim):
    path = ROOT / "deploy_artifacts/hoi_contact_pose_sub7_largebox_016/isaac/robot.urdf"
    if not path.exists():
        pytest.skip("Prepare Isaac assets first")
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    sim.data.qpos[sim.rq] = [0, 0, 0, 1, 0, 0, 0]
    rng = np.random.default_rng(7)
    for _ in range(10):
        for name, q in zip(sim.cfg["joint_names"], rng.normal(0, 0.1, 29), strict=True):
            sim.data.qpos[sim.model.joint("robot/" + name).qposadr] = q
            data.qpos[model.joint(name).qposadr] = q
        mujoco.mj_forward(sim.model, sim.data)
        mujoco.mj_forward(model, data)
        for i in range(1, model.nbody):
            j = sim.model.body("robot/" + model.body(i).name).id
            np.testing.assert_allclose(sim.data.xpos[j], data.xpos[i], atol=1e-8)
            np.testing.assert_allclose(sim.data.xmat[j], data.xmat[i], atol=1e-8)

# ruff: noqa: E402
import sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from supermimic.deploy.contact import (
    Odometry,
    ContactClip,
    ContactCore,
    ContactOnnx,
    static_contact_clip,
    rotate,
)
from supermimic.deploy.core import DeployConfig, SafetyCfg

ART = ROOT / "deploy_artifacts/hoi_contact_pose_sub7_largebox_016"


def packet(t=10.0, seq=1):
    return dict(
        schema="contact-vio-v1",
        frame_id="clip",
        tracking_valid=True,
        acquisition_monotonic_ns=int(t * 1e9),
        seq=seq,
        root_pos_w=[0, 0, 0.8],
        root_lin_vel_w=[0, 0, 0],
    )


def test_vio_timestamp_and_replay_rejection():
    o = Odometry("clip")
    o.accept(packet(), now=10.01)
    np.testing.assert_allclose(o.sample(now=10.02)[0], [0, 0, 0.8])
    with pytest.raises(ValueError, match="stale"):
        o.sample(now=10.06)
    with pytest.raises(ValueError, match="duplicate"):
        o.accept(packet(), now=10.03)
    with pytest.raises(ValueError):
        o.sample(now=10.03)


@pytest.mark.parametrize(
    "field,value",
    [
        ("tracking_valid", False),
        ("frame_id", "wrong"),
        ("root_pos_w", [float("nan"), 0, 0]),
        ("acquisition_monotonic_ns", 11_000_000_000),
    ],
)
def test_bad_vio_rejected(field, value):
    p = packet()
    p[field] = value
    with pytest.raises(ValueError):
        Odometry("clip").accept(p, now=10.01)


def setup_core(provider, policy):
    if not (ART / "hardware_config.json").exists():
        pytest.skip("Prepare artifact first")
    clip = ContactClip.load(ART / "reference.npz")
    return ContactCore(
        DeployConfig.load(ART / "hardware_config.json"),
        policy,
        clip,
        SafetyCfg(max_tilt_err=3.15, max_vel_frac=100, clamp_torque=False),
        provider,
    )


def test_no_inference_on_missing_odometry():
    calls = []
    core = setup_core(Odometry("clip"), lambda *x: calls.append(x))
    s = core.clip.reference[0]
    r = core.step(s[7:36], s[42:71], np.zeros(3), s[3:7])
    assert r.abort and not calls and core.frame == 0


def test_core_matches_onnx_and_window_clamps():
    policy = ContactOnnx(ART / "policy.onnx")
    holder = {}

    class Provider:
        def sample(self):
            return holder["s"][:3], holder["s"][36:39]

    core = setup_core(Provider(), policy)
    for frame in [0, 15, 300, 319]:
        s = core.clip.reference[frame].astype(float)
        holder["s"] = s
        core.reset(frame)
        q = s[3:7]
        gyro = rotate(q * np.array([1, -1, -1, -1]), s[39:42])
        ids = np.minimum(frame + np.arange(10) * 5, len(core.clip.reference) - 1)
        expected = policy(s, (core.clip.reference[ids], core.clip.labels[ids]))
        r = core.step(s[7:36], s[42:71], gyro, q)
        assert r.abort is None
        np.testing.assert_allclose(r.action, expected, atol=2e-5, rtol=2e-5)
        assert r.done == (frame == 319)


def test_static_clip_has_zero_rates_preserved_labels():
    clip = ContactClip.load(ART / "reference.npz")
    static = static_contact_clip(clip, 50)
    assert not np.any(static.reference[:, 36:71])
    np.testing.assert_array_equal(static.labels[0], clip.labels[50])
    np.testing.assert_array_equal(static.reference[0, :36], clip.reference[50, :36])


def test_udp_receives_before_policy_and_rejects_invalid_packet():
    import json
    import socket
    import time
    from supermimic.deploy.contact import UdpOdometry

    o = UdpOdometry("clip", 0, max_age=1.0)
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        address = o.socket.getsockname()
        sender.sendto(json.dumps(packet(time.monotonic())).encode(), address)
        deadline = time.monotonic() + 1
        while o.latest is None and time.monotonic() < deadline:
            time.sleep(0.005)
        np.testing.assert_allclose(o.sample()[0], [0, 0, 0.8])
        sender.sendto(b"[]", address)
        deadline = time.monotonic() + 1
        while not o.error and time.monotonic() < deadline:
            time.sleep(0.005)
        with pytest.raises(ValueError):
            o.sample()
        assert o.thread.is_alive()
    finally:
        sender.close()
        o.close()
    assert not o.thread.is_alive()

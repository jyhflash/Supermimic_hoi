"""Cross-engine contact ONNX evaluation in Isaac Lab / PhysX, one dynamic object."""

# ruff: noqa: E402
# Isaac imports must follow AppLauncher initialization.
import argparse
import json
from pathlib import Path
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--artifacts",
    type=Path,
    default=Path(__file__).resolve().parents[2]
    / "deploy_artifacts/hoi_contact_pose_sub7_largebox_016",
)
parser.add_argument("--steps", type=int, default=0)
parser.add_argument("--report", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.report.exists():
    raise FileExistsError(args.report)
app = AppLauncher(args).app

import numpy as np
import torch
import onnxruntime as ort
from pxr import UsdGeom, UsdPhysics, PhysxSchema, Gf, Vt, UsdShade
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.actuators import IdealPDActuatorCfg
from scipy.spatial.transform import Rotation


def rotation(q):
    return Rotation.from_quat(np.asarray(q)[[1, 2, 3, 0]])


def main():
    c = json.loads((args.artifacts / "deploy_config.json").read_text())
    assets = args.artifacts / "isaac"
    p = json.loads((assets / "physics.json").read_text())
    if min(p["effort"]) <= 0:
        raise ValueError("Motor effort limits must be positive; regenerate Isaac assets")
    o = json.loads((assets / "object_physics.json").read_text())
    with np.load(args.artifacts / "reference.npz") as r:
        states = r["state"].copy()
        labels = r["contact_labels"].copy()
    dt = c["physics_dt"]
    substeps = round(c["control_dt"] / dt)
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=dt, device=args.device))
    ground = sim_utils.GroundPlaneCfg(
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0, dynamic_friction=1.0, restitution=0.0, friction_combine_mode="max"
        )
    )
    ground.func("/World/Ground", ground)
    light = sim_utils.DomeLightCfg(intensity=2000)
    light.func("/World/Light", light)
    names = c["joint_names"]

    def mapping(key):
        return dict(zip(names, p[key], strict=True))

    spawn = sim_utils.UrdfFileCfg(
        asset_path=str(assets / "robot.urdf"),
        usd_dir=str(assets / "usd"),
        fix_base=False,
        force_usd_conversion=True,
        merge_fixed_joints=False,
        self_collision=False,
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            target_type="none",
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False, max_depenetration_velocity=10.0
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=2,
        ),
    )
    robot = Articulation(
        ArticulationCfg(
            prim_path="/World/Robot",
            spawn=spawn,
            actuators={
                "joints": IdealPDActuatorCfg(
                    joint_names_expr=[".*"],
                    stiffness=mapping("kp"),
                    damping=mapping("kd"),
                    effort_limit=mapping("effort"),
                    effort_limit_sim=mapping("effort"),
                    velocity_limit_sim=100.0,
                    armature=mapping("armature"),
                )
            },
        )
    )
    stage = sim.stage
    robot_mat = UsdShade.Material.Define(stage, "/World/RobotMaterial")
    rp = UsdPhysics.MaterialAPI.Apply(robot_mat.GetPrim())
    rp.CreateStaticFrictionAttr(0.6)
    rp.CreateDynamicFrictionAttr(0.6)
    rp.CreateRestitutionAttr(0.0)
    PhysxSchema.PhysxMaterialAPI.Apply(robot_mat.GetPrim()).CreateFrictionCombineModeAttr("max")
    UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath("/World/Robot")).Bind(
        robot_mat,
        bindingStrength=UsdShade.Tokens.strongerThanDescendants,
        materialPurpose="physics",
    )
    root = UsdGeom.Xform.Define(stage, "/World/Object").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(root)
    mass = UsdPhysics.MassAPI.Apply(root)
    mass.CreateMassAttr(o["mass"])
    mass.CreateCenterOfMassAttr(Gf.Vec3f(*o["com"]))
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*o["inertia"]))
    q = o["principal_axes"]
    mass.CreatePrincipalAxesAttr(Gf.Quatf(q[0], Gf.Vec3f(*q[1:])))
    mat = UsdShade.Material.Define(stage, "/World/ObjectMaterial")
    phys = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    phys.CreateStaticFrictionAttr(o["friction"])
    phys.CreateDynamicFrictionAttr(o["friction"])
    phys.CreateRestitutionAttr(0.0)
    PhysxSchema.PhysxMaterialAPI.Apply(mat.GetPrim()).CreateFrictionCombineModeAttr("max")
    with np.load(assets / "object_meshes.npz") as mesh:
        for key in mesh.files:
            if not key.endswith("_vertices"):
                continue
            prefix = key.removesuffix("_vertices")
            v = mesh[key]
            f = mesh[prefix + "_faces"]
            m = UsdGeom.Mesh.Define(stage, f"/World/Object/{prefix}")
            m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(v.astype(np.float32)))
            m.CreateFaceVertexCountsAttr([3] * len(f))
            m.CreateFaceVertexIndicesAttr(f.flatten().tolist())
            if prefix.startswith("collision"):
                m.CreateVisibilityAttr("invisible")
                UsdPhysics.CollisionAPI.Apply(m.GetPrim())
                UsdPhysics.MeshCollisionAPI.Apply(m.GetPrim()).CreateApproximationAttr("convexHull")
                UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(mat, materialPurpose="physics")
                co = PhysxSchema.PhysxCollisionAPI.Apply(m.GetPrim())
                co.CreateContactOffsetAttr(0.005)
                co.CreateRestOffsetAttr(0.0)
            else:
                m.CreateDisplayColorAttr([Gf.Vec3f(0.6, 0.45, 0.25)])
    obj = RigidObject(RigidObjectCfg(prim_path="/World/Object", spawn=None))
    sim.reset()
    if set(robot.joint_names) != set(names):
        raise ValueError(f"Joint mismatch: {robot.joint_names}")
    order = [robot.joint_names.index(n) for n in names]
    device = sim.device

    def tensor(x):
        return torch.as_tensor(x, dtype=torch.float32, device=device)

    s = states[0]
    robot.write_root_link_pose_to_sim(tensor(s[:7][None]))
    robot.write_root_link_velocity_to_sim(tensor(s[36:42][None]))
    robot.write_joint_state_to_sim(tensor(s[7:36][None]), tensor(s[42:71][None]), joint_ids=order)
    obj.write_root_link_pose_to_sim(tensor(s[71:78][None]))
    obj.write_root_link_velocity_to_sim(tensor(s[78:84][None]))
    robot.reset()
    obj.reset()
    sim.set_camera_view(s[:3] + [3, -3, 1.5], s[:3])
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    policy = ort.InferenceSession(
        str(args.artifacts / "policy.onnx"),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )

    def observation():
        root = robot.data.root_link_state_w
        return (
            torch.cat(
                (
                    root[:, :7],
                    robot.data.joint_pos[:, order],
                    root[:, 7:13],
                    robot.data.joint_vel[:, order],
                ),
                -1,
            )
            .cpu()
            .numpy()
        )

    initial = observation()
    np.testing.assert_allclose(initial, states[:1, :71], atol=2e-5, rtol=2e-5)
    actual_mass = float(robot.root_physx_view.get_masses().sum())
    np.testing.assert_allclose(actual_mass, p["robot_mass"], rtol=1e-4)
    rows = []
    reason = "reference_end"
    limit = min(len(states) - 1, args.steps or len(states) - 1)
    for frame in range(limit):
        if not app.is_running():
            reason = "application_closed"
            break
        ids = np.minimum(frame + np.arange(c["goal_frames"]) * c["goal_stride"], len(states) - 1)
        action = policy.run(
            None,
            dict(
                robot_state=observation(),
                reference_robot=states[ids, :71][None],
                contact_labels=labels[ids][None],
            ),
        )[0]
        if not np.isfinite(action).all():
            raise FloatingPointError("Nonfinite policy action")
        target = tensor(np.array(c["action_offset"]) + np.array(c["action_scale"]) * action)
        robot.set_joint_position_target(target, joint_ids=order)
        for _ in range(substeps):
            robot.write_data_to_sim()
            obj.write_data_to_sim()
            sim.step(render=not args.headless)
            robot.update(dt)
            obj.update(dt)
        root = robot.data.root_link_state_w[0].cpu().numpy()
        pose = obj.data.root_link_state_w[0, :7].cpu().numpy()
        ref = states[frame + 1]
        if not np.isfinite(root).all() or not np.isfinite(pose).all():
            raise FloatingPointError("Nonfinite PhysX state")
        height = abs(root[2] - ref[2])
        tilt = abs(
            rotation(root[3:7]).inv().apply([0, 0, -1])[2]
            - rotation(ref[3:7]).inv().apply([0, 0, -1])[2]
        )
        rows.append(
            dict(
                frame=frame + 1,
                state=np.concatenate(
                    (observation()[0], obj.data.root_link_state_w[0].cpu().numpy())
                ).tolist(),
                root_position_error_m=float(np.linalg.norm(root[:3] - ref[:3])),
                object_position_error_m=float(np.linalg.norm(pose[:3] - ref[71:74])),
                object_orientation_error_deg=float(
                    np.degrees((rotation(pose[3:]).inv() * rotation(ref[74:78])).magnitude())
                ),
                joint_rmse_rad=float(np.sqrt(np.mean((observation()[0, 7:36] - ref[7:36]) ** 2))),
                object_displacement_m=float(np.linalg.norm(pose[:3] - s[71:74])),
                root_height_error_m=float(height),
                root_tilt_error=float(tilt),
                max_abs_action=float(np.max(abs(action))),
            )
        )
        if height > 0.25 or tilt > 0.8:
            reason = "robot_failure"
            break
    if reason == "reference_end" and limit < len(states) - 1:
        reason = "step_cap"
    report = dict(
        engine="Isaac Lab / PhysX",
        device=str(device),
        checkpoint_sha256=c["checkpoint_sha256"],
        checkpoint_step=c["step"],
        clip=c["clip"]["clip_id"],
        executed_steps=len(rows),
        expected_steps=len(states) - 1,
        stop_reason=reason,
        dynamic_object=True,
        object_in_policy=False,
        robot_mass=actual_mass,
        friction={"ground": 1.0, "robot": 0.6, "object": o["friction"], "combine": "max"},
        actuator="explicit PD evaluated each physics step; exported force limits",
        physics_dt=dt,
        control_dt=c["control_dt"],
        limitations=[
            "Self collisions disabled; MuJoCo collision-pair exclusions are not reproduced",
            "Capsules imported as cylinder plus sphere compound",
            "PhysX convex cooking and solver/contact laws differ",
            "No contact-coverage success gate or hardware transition test",
            "Uses simulator root odometry",
        ],
        mean_errors={
            k: float(np.mean([r[k] for r in rows]))
            for k in [
                "root_position_error_m",
                "object_position_error_m",
                "object_orientation_error_deg",
                "joint_rmse_rad",
            ]
        }
        if rows
        else {},
        trajectory=rows,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x") as f:
        json.dump(report, f, indent=2)
    print(
        "CONTACT_ISAAC_RESULT "
        + json.dumps({k: v for k, v in report.items() if k != "trajectory"}),
        flush=True,
    )


try:
    main()
finally:
    app.close(wait_for_replicator=False, skip_cleanup=True)

"""Convert frozen MuJoCo geometry/inertias into a local URDF for PhysX import."""

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import trimesh


def xyz(v):
    return " ".join(f"{x:.12g}" for x in v)


def rot(q):
    return Rotation.from_quat(np.asarray(q)[[1, 2, 3, 0]])


def origin(parent, pos, quat):
    ET.SubElement(parent, "origin", xyz=xyz(pos), rpy=xyz(rot(quat).as_euler("xyz")))


def effort_limit(model, joint):
    actuators = np.flatnonzero(model.actuator_trnid[:, 0] == joint)
    if len(actuators) != 1:
        raise ValueError("Expected one actuator per joint")
    a = int(actuators[0])
    limits = []
    if model.actuator_forcelimited[a]:
        limits.append(float(np.max(np.abs(model.actuator_forcerange[a]))))
    if model.jnt_actfrclimited[joint]:
        limits.append(float(np.max(np.abs(model.jnt_actfrcrange[joint]))))
    if not limits or min(limits) <= 0:
        raise ValueError("Missing positive motor torque limit")
    return min(limits)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--artifacts",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "deploy_artifacts/hoi_contact_pose_sub7_largebox_016",
    )
    a = p.parse_args()
    out = a.artifacts / "isaac"
    out.mkdir(exist_ok=True)
    c = json.loads((a.artifacts / "deploy_config.json").read_text())
    m = mujoco.MjModel.from_binary_path(str(a.artifacts / "scene.mjb"))
    robot = ET.Element("robot", name="contact_g1")
    bodies = [i for i in range(1, m.nbody) if m.body(i).name.startswith("robot/")]

    def name(i):
        return m.body(i).name.removeprefix("robot/")

    for b in bodies:
        link = ET.SubElement(robot, "link", name=name(b))
        inertial = ET.SubElement(link, "inertial")
        origin(inertial, m.body_ipos[b], m.body_iquat[b])
        ET.SubElement(inertial, "mass", value=str(m.body_mass[b]))
        d = m.body_inertia[b]
        ET.SubElement(
            inertial,
            "inertia",
            ixx=str(d[0]),
            iyy=str(d[1]),
            izz=str(d[2]),
            ixy="0",
            ixz="0",
            iyz="0",
        )
        for g in np.flatnonzero(m.geom_bodyid == b):
            types = []
            if m.geom_rgba[g, 3] > 0:
                types.append("visual")
            if m.geom_contype[g] or m.geom_conaffinity[g]:
                types.append("collision")
            typ = int(m.geom_type[g])
            sz = m.geom_size[g]
            shapes = [("shape", m.geom_pos[g])]
            if typ == mujoco.mjtGeom.mjGEOM_CAPSULE:
                axis = rot(m.geom_quat[g]).apply([0, 0, sz[1]])
                shapes += [("cap", m.geom_pos[g] + axis), ("cap", m.geom_pos[g] - axis)]
            for kind in types:
                for shape, pos in shapes:
                    item = ET.SubElement(link, kind)
                    origin(item, pos, m.geom_quat[g])
                    geo = ET.SubElement(item, "geometry")
                    if shape == "cap" or typ == mujoco.mjtGeom.mjGEOM_SPHERE:
                        ET.SubElement(geo, "sphere", radius=str(sz[0]))
                    elif typ in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER):
                        ET.SubElement(geo, "cylinder", radius=str(sz[0]), length=str(2 * sz[1]))
                    elif typ == mujoco.mjtGeom.mjGEOM_BOX:
                        ET.SubElement(geo, "box", size=xyz(2 * sz))
                    elif typ == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
                        path = (out / f"ellipsoid_{g}.obj").resolve()
                        mesh = trimesh.creation.icosphere(subdivisions=3)
                        mesh.vertices *= sz
                        mesh.export(path)
                        ET.SubElement(geo, "mesh", filename=str(path))
                    elif typ == mujoco.mjtGeom.mjGEOM_MESH:
                        mid = m.geom_dataid[g]
                        v = m.mesh_vert[
                            m.mesh_vertadr[mid] : m.mesh_vertadr[mid] + m.mesh_vertnum[mid]
                        ]
                        f = m.mesh_face[
                            m.mesh_faceadr[mid] : m.mesh_faceadr[mid] + m.mesh_facenum[mid]
                        ]
                        path = (out / f"mesh_{mid}.obj").resolve()
                        trimesh.Trimesh(v, f, process=False).export(path)
                        ET.SubElement(geo, "mesh", filename=str(path))
                    else:
                        raise ValueError(f"Unsupported geom {typ}")
        parent = m.body_parentid[b]
        if parent not in bodies:
            continue
        n = int(m.body_jntnum[b])
        j = int(m.body_jntadr[b])
        if n not in (0, 1):
            raise ValueError("Multiple joints per body unsupported")
        joint = ET.SubElement(
            robot,
            "joint",
            name=m.joint(j).name.removeprefix("robot/") if n else f"fixed_{b}",
            type="revolute" if n else "fixed",
        )
        ET.SubElement(joint, "parent", link=name(parent))
        ET.SubElement(joint, "child", link=name(b))
        origin(joint, m.body_pos[b], m.body_quat[b])
        if n:
            assert np.allclose(m.jnt_pos[j], 0), "Nonzero joint anchor needs conversion"
            assert np.allclose(m.qpos0[m.jnt_qposadr[j]], 0), "Nonzero reference needs conversion"
            ET.SubElement(joint, "axis", xyz=xyz(m.jnt_axis[j]))
            effort = effort_limit(m, j)
            ET.SubElement(
                joint,
                "limit",
                lower=str(m.jnt_range[j, 0]),
                upper=str(m.jnt_range[j, 1]),
                effort=str(effort),
                velocity="100",
            )
            v = m.jnt_dofadr[j]
            ET.SubElement(
                joint,
                "dynamics",
                damping=str(m.dof_damping[v]),
                friction=str(m.dof_frictionloss[v]),
            )
    ET.indent(robot)
    ET.ElementTree(robot).write(out / "robot.urdf", encoding="utf-8", xml_declaration=True)
    ids = np.array(c["actuator_ids"])
    vs = np.array(c["joint_vadr"])
    js = m.actuator_trnid[ids, 0]
    params = dict(
        kp=m.actuator_gainprm[ids, 0].tolist(),
        kd=(-m.actuator_biasprm[ids, 2]).tolist(),
        armature=m.dof_armature[vs].tolist(),
        effort=[effort_limit(m, j) for j in js],
        robot_mass=float(m.body_mass[bodies].sum()),
    )
    (out / "physics.json").write_text(json.dumps(params, indent=2))
    # Use MuJoCo's compiled object mesh transforms: scale is already applied.
    obj = m.body("object/body").id
    parts = {}
    for g in np.flatnonzero(m.geom_bodyid == obj):
        if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            raise ValueError("Expected object mesh")
        mid = m.geom_dataid[g]
        v = m.mesh_vert[m.mesh_vertadr[mid] : m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
        v = rot(m.geom_quat[g]).apply(v) + m.geom_pos[g]
        f = m.mesh_face[m.mesh_faceadr[mid] : m.mesh_faceadr[mid] + m.mesh_facenum[mid]]
        prefix = f"collision_{g}" if m.geom_contype[g] else f"visual_{g}"
        parts[prefix + "_vertices"] = v
        parts[prefix + "_faces"] = f
    np.savez_compressed(out / "object_meshes.npz", **parts)
    (out / "object_physics.json").write_text(
        json.dumps(
            dict(
                mass=float(m.body_mass[obj]),
                com=m.body_ipos[obj].tolist(),
                inertia=m.body_inertia[obj].tolist(),
                principal_axes=m.body_iquat[obj].tolist(),
                friction=float(m.geom_friction[m.body_geomadr[obj] + 1, 0]),
            ),
            indent=2,
        )
    )
    print(out)


if __name__ == "__main__":
    main()

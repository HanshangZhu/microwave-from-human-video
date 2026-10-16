"""The exact geometry of a primitive-only URDF (boxes and cylinders, revolute joints off
the root link), posed at given joint angles, in the root link's frame. Registration and
colour baking use it as the known shape of the scanned object.
"""
import xml.etree.ElementTree as ET

import numpy as np
import trimesh


def rpy_matrix(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def origin_T(el):
    T = np.eye(4)
    o = el.find("origin") if el is not None else None
    if o is not None:
        T[:3, 3] = [float(x) for x in o.get("xyz", "0 0 0").split()]
        T[:3, :3] = rpy_matrix(*[float(x) for x in o.get("rpy", "0 0 0").split()])
    return T


def axis_angle(axis, q):
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    T = np.eye(4)
    T[:3, :3] = np.eye(3) + np.sin(q) * K + (1 - np.cos(q)) * K @ K
    return T


class Model:
    def __init__(self, urdf):
        root = ET.parse(urdf).getroot()
        self.visuals = {}                       # link -> [dict]
        for link in root.findall("link"):
            vs = []
            for v in link.findall("visual"):
                g = v.find("geometry")[0]
                c = v.find("material/color")
                vs.append(dict(name=v.get("name"), T=origin_T(v), kind=g.tag,
                               size=[float(x) for x in g.get("size", "0 0 0").split()],
                               radius=float(g.get("radius", 0)), length=float(g.get("length", 0)),
                               rgba=[float(x) for x in c.get("rgba").split()] if c is not None
                               else [0.7, 0.7, 0.7, 1]))
            self.visuals[link.get("name")] = vs
        self.joints = {}                        # child link -> (name, T_parent_joint, axis)
        for j in root.findall("joint"):
            self.joints[j.find("child").get("link")] = (
                j.get("name"), origin_T(j), [float(x) for x in j.find("axis").get("xyz").split()])
        children = set(self.joints)
        self.root = [l for l in self.visuals if l not in children][0]

    def link_T(self, link, q=None):
        """Pose of `link` in the root frame; q maps joint name -> angle (default 0)."""
        if link == self.root:
            return np.eye(4)
        name, T, axis = self.joints[link]
        return T @ axis_angle(axis, (q or {}).get(name, 0.0))

    def primitive(self, v):
        if v["kind"] == "box":
            return trimesh.creation.box(extents=v["size"])
        return trimesh.creation.cylinder(radius=v["radius"], height=v["length"], sections=32)

    def mesh(self, q=None, links=None):
        """All visuals (or those of `links`) as one mesh in the root frame."""
        parts = []
        for link, vs in self.visuals.items():
            if links is not None and link not in links:
                continue
            L = self.link_T(link, q)
            for v in vs:
                parts.append(self.primitive(v).apply_transform(L @ v["T"]))
        return trimesh.util.concatenate(parts)

#!/usr/bin/env python
"""Build the kitchen microwave as an articulated URDF: a fixed body with an open
cavity, a turntable plate, and one door on a vertical hinge at the front-left
corner with a bar handle near its free edge.

Every dimension below is tagged with where it came from:
  measured   tape measure, 2026-10-14
  photo      scaled from the front photographs against the measured 46 cm width
  guess      not measured yet -- replace when it is

Frame (metres, z up, origin on the countertop below the body's front-left corner):
  +x  to the right along the front face
  +y  into the microwave (front face of the body at y = 0, door in front of it)
  +z  up
The door joint is `door_hinge`; q = 0 closed, q > 0 opens it outward to the left.

  python assets/microwave/build_microwave.py        # writes microwave.urdf + parameters.json
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

P = {   # value [m or rad or kg], source
    "body_width":          (0.460, "measured"),
    "body_depth":          (0.279, "measured: 31-32 cm overall incl. the 3.6 cm door, without the back bump; "
                                   "single depth frames give 30-31.6 cm"),
    "body_height":         (0.270, "measured: 27.5-28 cm counter to top, minus the feet; depth frames 27.9 cm"),
    "feet_height":         (0.008, "measured"),
    "cavity_width":        (0.290, "measured"),
    "cavity_depth":        (0.290, "measured at the edges and corners; reaches past the body into a bump "
                                   "on the back"),
    "cavity_recess_depth": (0.315, "measured, centre of the back wall"),
    "cavity_recess_inset": (0.020, "guess from the door-open photo, recess edge to cavity edge"),
    "back_shell":          (0.005, "guess, sheet thickness of the back bump"),
    "cavity_height":       (0.200, "measured"),
    "cavity_left_offset":  (0.030, "measured, cavity to the hinge-side outer edge"),
    "cavity_floor_above_body_bottom": (0.028, "photo (door-open view), scaled to the 27 cm body"),
    "door_width":          (0.356, "measured, 14 in"),
    "door_thickness":      (0.036, "measured"),
    "door_max_angle":      (1.75,  "guess, ~100 deg"),
    "door_mass":           (1.0,   "guess"),
    "hinge_damping":       (0.05,  "guess"),
    "hinge_friction":      (0.05,  "guess, N m: a light hinge, as the real one (do not stiffen it to "
                                   "hide the arm brushing the door)"),
    "handle_x":            (0.343, "photo, bar centre from the hinge-side edge (~1.3 cm in from the door's free edge)"),
    "handle_length":       (0.202, "measured"),
    "handle_diameter":     (0.009, "measured"),
    "handle_offset":       (0.026, "measured, door face to bar axis"),
    "handle_post_spacing": (0.160, "measured, between the two door-to-bar posts"),
    "handle_post_radius":  (0.005, "measured"),
    "turntable_diameter":  (0.245, "guess"),
    "turntable_top_above_cavity_floor": (0.005, "measured ('turntable height above cavity floor')"),
}

# Control panel, right of the door, its face flush with the door front. Positions are
# the photo pixels mapped through the same front-face homography the texture decal
# uses (front_photo.jpg, front corners TL 210,614 TR 3784,572 BR 3568,2390 BL 455,2404
# px), so the parts sit on their own image. x from the hinge-side edge, z from the
# countertop, metres, +-2 mm. The knob is placed by its base ring: its chrome face stands 2.2 cm proud and shows parallax.
# Nothing here moves: the knob and buttons are fixed shapes, not joints.
CONTROLS = {
    "display":   dict(x=(0.3747, 0.4548), z=(0.1951, 0.2423)),      # dark glass window
    "digit_0":   dict(x=(0.4273, 0.4387), z=(0.2123, 0.2254)),      # green "0", 7-segment
    "colon":     dict(x=0.4121, z=(0.2161, 0.2219)),
    "buttons":   dict(r=0.0053, out=0.004, at=[(0.3895, 0.1693), (0.4134, 0.1691),
                                               (0.4374, 0.1693), (0.4374, 0.1384)]),
    "knob":      dict(at=(0.4162, 0.0825), r=0.020, out=0.022),       # one chrome piece
    "start_stop": dict(r=0.0045, out=0.003, at=[(0.3894, 0.0440, "0.95 0.55 0.25 1"),
                                                 (0.4370, 0.0440, "0.40 0.72 0.58 1")]),
}


def v(k):
    return P[k][0]


def box(name, lo, hi, rgba, collide=True):
    size = [hi[i] - lo[i] for i in range(3)]
    ctr = [(hi[i] + lo[i]) / 2 for i in range(3)]
    g = f'<geometry><box size="{size[0]:.4f} {size[1]:.4f} {size[2]:.4f}"/></geometry>'
    o = f'<origin xyz="{ctr[0]:.4f} {ctr[1]:.4f} {ctr[2]:.4f}"/>'
    s = f'  <visual name="{name}">{o}{g}<material name="{name}_m"><color rgba="{rgba}"/></material></visual>\n'
    if collide:
        s += f'  <collision name="{name}">{o}{g}</collision>\n'
    return s


def cyl(name, ctr, radius, length, rgba, rpy="0 0 0", collide=True):
    g = f'<geometry><cylinder radius="{radius:.4f}" length="{length:.4f}"/></geometry>'
    o = f'<origin xyz="{ctr[0]:.4f} {ctr[1]:.4f} {ctr[2]:.4f}" rpy="{rpy}"/>'
    s = f'  <visual name="{name}">{o}{g}<material name="{name}_m"><color rgba="{rgba}"/></material></visual>\n'
    if collide:
        s += f'  <collision name="{name}">{o}{g}</collision>\n'
    return s


def seven_seg_zero(name, x, z, y, rgba):
    """A 7-segment '0' (its four outer strokes) as thin boxes spanning y = (y_lo, y_hi)."""
    (xa, xb), (za, zb), st = x, z, 0.0015                 # st: stroke width
    segs = [((xa, za), (xb, za + st)), ((xa, zb - st), (xb, zb)),          # bottom, top
            ((xa, za), (xa + st, zb)), ((xb - st, za), (xb, zb))]          # left, right
    return "".join(box(f"{name}_{i}", (p[0], y[0], p[1]), (q[0], y[1], q[1]), rgba, collide=False)
                   for i, (p, q) in enumerate(segs))


def control_panel(t, dw, W, f, H, steel, dark):
    C = CONTROLS
    face = -t                                                # panel front, flush with the door
    green, chrome = "0.25 0.95 0.35 1", "0.88 0.88 0.9 1"
    out = [box("control_panel", (dw, -t, f), (W, 0, f + H), steel)]
    (xa, xb), (za, zb) = C["display"]["x"], C["display"]["z"]
    out.append(box("display", (xa, face - 0.001, za), (xb, face, zb), dark, collide=False))
    out.append(seven_seg_zero("digit_0", C["digit_0"]["x"], C["digit_0"]["z"],
                              (face - 0.0016, face - 0.0008), green))   # proud of the glass
    for i, zc in enumerate(C["colon"]["z"]):
        x = C["colon"]["x"]
        out.append(box(f"colon_{i}", (x - 0.0008, face - 0.0016, zc - 0.0008),
                       (x + 0.0008, face - 0.0008, zc + 0.0008), green, collide=False))
    b = C["buttons"]
    for i, (x, z) in enumerate(b["at"]):
        out.append(cyl(f"button_{i}", (x, face - b["out"] / 2, z), b["r"], b["out"], chrome,
                       rpy="1.5708 0 0"))
    k = C["knob"]
    (x, z) = k["at"]
    out.append(cyl("knob", (x, face - k["out"] / 2, z), k["r"], k["out"], chrome, rpy="1.5708 0 0"))
    ss = C["start_stop"]
    for i, (x, z, rgba) in enumerate(ss["at"]):
        out.append(cyl(f"start_stop_{i}", (x, face - ss["out"] / 2, z), ss["r"], ss["out"], rgba,
                       rpy="1.5708 0 0"))
    return "".join(out)


def main():
    W, D, H, f = v("body_width"), v("body_depth"), v("body_height"), v("feet_height")
    cw, cd, ch = v("cavity_width"), v("cavity_depth"), v("cavity_height")
    x0, z0 = v("cavity_left_offset"), f + v("cavity_floor_above_body_bottom")
    x1, z1 = x0 + cw, z0 + ch
    steel, dark, glass = "0.72 0.72 0.74 1", "0.08 0.08 0.09 1", "0.85 0.9 0.92 1"
    cr, ci, w = v("cavity_recess_depth"), v("cavity_recess_inset"), v("back_shell")
    assert x1 < W and z1 < f + H and D < cd < cr, "cavity does not fit the body and its back bump"

    body = "".join([
        box("feet", (0.02, 0.02, 0), (W - 0.02, D - 0.02, f), dark),
        box("floor", (0, 0, f), (W, D, z0), steel),
        box("roof", (0, 0, z1), (W, D, f + H), steel),
        box("left_wall", (0, 0, z0), (x0, D, z1), steel),
        box("right_block", (x1, 0, z0), (W, D, z1), steel),       # control panel side
        # the cavity runs past the body's back face (y = D) into a bump: a shell around
        # the cavity out to cd, with a deeper recess in the middle of its back out to cr
        box("bump_top", (x0 - w, D, z1), (x1 + w, cd + w, z1 + w), steel),
        box("bump_bottom", (x0 - w, D, z0 - w), (x1 + w, cd + w, z0), steel),
        box("bump_left", (x0 - w, D, z0), (x0, cd + w, z1), steel),
        box("bump_right", (x1, D, z0), (x1 + w, cd + w, z1), steel),
        box("bump_back_left", (x0, cd, z0), (x0 + ci, cd + w, z1), steel),
        box("bump_back_right", (x1 - ci, cd, z0), (x1, cd + w, z1), steel),
        box("bump_back_top", (x0 + ci, cd, z1 - ci), (x1 - ci, cd + w, z1), steel),
        box("bump_back_bottom", (x0 + ci, cd, z0), (x1 - ci, cd + w, z0 + ci), steel),
        box("recess_top", (x0 + ci - w, cd + w, z1 - ci), (x1 - ci + w, cr + w, z1 - ci + w), steel),
        box("recess_bottom", (x0 + ci - w, cd + w, z0 + ci - w), (x1 - ci + w, cr + w, z0 + ci), steel),
        box("recess_left", (x0 + ci - w, cd + w, z0 + ci), (x0 + ci, cr + w, z1 - ci), steel),
        box("recess_right", (x1 - ci, cd + w, z0 + ci), (x1 - ci + w, cr + w, z1 - ci), steel),
        box("recess_back", (x0 + ci, cr, z0 + ci), (x1 - ci, cr + w, z1 - ci), steel),
        cyl("turntable", ((x0 + x1) / 2, cd / 2, z0 + v("turntable_top_above_cavity_floor") / 2),
            v("turntable_diameter") / 2, v("turntable_top_above_cavity_floor"), glass),
    ])

    t, dw, hx = v("door_thickness"), v("door_width"), v("handle_x")
    body += control_panel(t, dw, W, f, H, steel, dark)
    ho, hd, hl = v("handle_offset"), v("handle_diameter"), v("handle_length")
    ps, pr = v("handle_post_spacing"), v("handle_post_radius")
    zc = f + H / 2
    door = "".join([
        box("door_panel", (0, -t, f), (dw, 0, f + H), steel),
        box("door_window", (x0, -t - 0.001, z0), (x1, -t, z1), dark, collide=False),
        cyl("handle_bar", (hx, -t - ho, zc), hd / 2, hl, steel),
        cyl("handle_post_top", (hx, -t - ho / 2, zc + ps / 2), pr, ho, steel, rpy="1.5708 0 0"),
        cyl("handle_post_bottom", (hx, -t - ho / 2, zc - ps / 2), pr, ho, steel, rpy="1.5708 0 0"),
    ])
    m = v("door_mass")
    ixx = m * (H**2 + t**2) / 12
    iyy = m * (dw**2 + H**2) / 12
    izz = m * (dw**2 + t**2) / 12
    urdf = f"""<?xml version="1.0"?>
<!-- generated by build_microwave.py; edit the parameters there, not this file -->
<robot name="microwave">
 <link name="body">
  <inertial><origin xyz="{W/2:.3f} {D/2:.3f} {f+H/2:.3f}"/><mass value="12"/>
   <inertia ixx="0.2" iyy="0.3" izz="0.3" ixy="0" ixz="0" iyz="0"/></inertial>
{body} </link>
 <link name="door">
  <inertial><origin xyz="{dw/2:.4f} {-t/2:.4f} {zc:.4f}"/><mass value="{m}"/>
   <inertia ixx="{ixx:.5f}" iyy="{iyy:.5f}" izz="{izz:.5f}" ixy="0" ixz="0" iyz="0"/></inertial>
{door} </link>
 <joint name="door_hinge" type="revolute">
  <parent link="body"/><child link="door"/>
  <origin xyz="0 0 0"/><axis xyz="0 0 -1"/>
  <limit lower="0" upper="{v('door_max_angle'):.3f}" effort="50" velocity="3"/>
  <dynamics damping="{v('hinge_damping')}" friction="{v('hinge_friction')}"/>
 </joint>
</robot>
"""
    open(os.path.join(HERE, "microwave.urdf"), "w").write(urdf)
    derived = {
        "cavity_x_range": [x0, x1], "cavity_y_range": [0, cd], "cavity_z_range": [z0, z1],
        "cavity_recess_y": cr, "overall_depth_incl_door_and_bump": t + cr + w,
        "turntable_top_z": z0 + v("turntable_top_above_cavity_floor"),
        "turntable_centre_xy": [(x0 + x1) / 2, cd / 2],
        "handle_bar_centre_closed": [hx, -t - v("handle_offset"), zc],
    }
    json.dump({"units": "m, rad, kg", "frame": __doc__.split("Frame")[1].split("  python")[0].strip(),
               "parameters": {k: {"value": a, "source": b} for k, (a, b) in P.items()},
               "derived": derived}, open(os.path.join(HERE, "parameters.json"), "w"), indent=1)
    print("wrote microwave.urdf;", json.dumps(derived))


if __name__ == "__main__":
    main()

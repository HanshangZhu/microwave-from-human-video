# Metric Heinz Snap Pot asset

The deliverables and measured results are in
[`../../assets/heinz_snap_pot/REPORT.md`](../../assets/heinz_snap_pot/REPORT.md).
The final mesh is a rigid proxy anchored to the user's approximate measurements,
with pose and rounding checked against raw metric depth. It is not ruler-certified.

The selected input is
`Gitignore-but-important/recon/beans_fuse_K/object.ply`, with the 40 RGB
frames in `Gitignore-but-important/recon/beans/input/`. Raw inputs are preserved.
Approximate inputs live in `assets/heinz_snap_pot/measurements_approx.json`:
lid 95 x 74 mm; upper body 80 x 55 mm; base 68 x 48 mm; height 63–64 mm;
top assembly 2.5 mm total. The model height is 63.5 mm. The latter thickness is
the whole cap, comprising 2.35 mm rim plastic proxy and assumed 0.15 mm film.
When editing thicknesses, keep `top_assembly_thickness_mm` equal to
`rim_thickness_mm + lid_thickness_mm`; the generator rejects inconsistent inputs.
Dimensions and radii live in `assets/heinz_snap_pot/parameters.json`; values with
`_mm` suffixes are converted to metres exactly once by the Blender generator.
Editing this JSON and rerunning the build changes the parametric model.
The Blender scene also stores all these values as named scene custom properties;
edit the JSON and rerun the script to regenerate geometry (properties alone are
metadata, not live geometry drivers).

The underside photo is preserved at `assets/heinz_snap_pot/references/bottom_34084.jpg`.
It supports a central recess and rounded perimeter, replacing the flat-base
assumption. Estimated parameters: recess depth 1 mm, outer round height/inset
3 mm, panel 56 x 36 mm with 5 mm corners, recess-wall bevel 1 mm. The measured
68 x 48 mm base outline is defined at z=3 mm; estimated contact outline is
62 x 42 mm at z=0. These new dimensions are not calibrated from the photograph.
The 3 mm fillet has 24 uniform arc intervals and is actual mesh geometry.
`smooth_curved_normals` enables analytic per-loop normals on the body and bottom
curves, with split normals preserving the lid edges. The same-camera comparison
is `rounding_before_after.png`; `renders/bottom_edge_detail.png` shows a close-up.
The single closed exterior has 35 collision pieces: upper body, flange/cap,
base bridge, and 32 radial support sectors preserving the recess. Each OBJ
`o` group is a separate closed convex piece. Angular sectors and horizontal
layers have disjoint interiors. `convex_hulls.py` canonicalizes their hulls with
double-precision SciPy Qhull; this avoids float32 BMesh near-coplanar slivers.

Run commands from the repository root. Tested with Blender **4.5.9 LTS** and
Python environments listed in `assets/heinz_snap_pot/runtime.json`.

```bash
# Optional: repeat the selected fit using the approximate user measurements.
# This overwrites parameters, so preserve any manual edits first.
python recon/snap_pot/refine_measured.py \
  --cloud Gitignore-but-important/recon/beans_fuse_K/object.ply \
  --measurements assets/heinz_snap_pot/measurements_approx.json \
  --out assets/heinz_snap_pot/parameters.json

# Prepare real-photo palette and aligned, unfiltered cloud for Blender.
python recon/snap_pot/prepare.py \
  --config assets/heinz_snap_pot/parameters.json \
  --photos Gitignore-but-important/recon/beans/input

# Headless build, GLB/OBJ export, .blend, and four paired camera renders.
# The generator calls python3 with NumPy/SciPy for hull triangulation.
# Set SNAP_POT_HULL_PYTHON=/path/to/python if using another environment.
blender --background --threads 8 --python-exit-code 1 \
  --python recon/snap_pot/build_blender.py -- \
  --config assets/heinz_snap_pot/parameters.json \
  --out assets/heinz_snap_pot

# Run in a Python environment containing Open3D and the numerical dependencies.
python recon/snap_pot/verify.py --asset assets/heinz_snap_pot --round 1

# Run in an environment containing ManiSkill 3 and SAPIEN.
python recon/snap_pot/smoke_maniskill.py --asset assets/heinz_snap_pot

# Assemble comparison sheet, residual map and this capture's detailed report.
python recon/snap_pot/report.py --asset assets/heinz_snap_pot
```

The current 3 mm verification is retained under `verification_rounds/`.
The earlier 2 mm revision's three rounds are in `bottom_2mm_verification_rounds/`.
The source-selection comparisons are historical flat-base results under
`measurement_verification_rounds/` and `measurement_candidates/`. Each used
identical nominal measurements with a different unscaled source cloud; candidate
2 (`K`) remains selected, with its rigid alignment preserved in this revision.
The earlier cloud-only asset and its code/reports are preserved in
`history/point_cloud_only_v1.zip`; the measurement-based flat-bottom revision is
in `history/measurement_flat_base_v2.zip`, and the prior 2 mm underside is in
`history/bottom_photo_2mm_v3.zip`. Repeating `verify.py --round 1` replaces that round's
report. Copy the asset folder before experimenting if preserving the supplied
history is important. `selection.json` records the selected source and three
comparison metrics; update it if selecting a new source. `fit.py` remains a
diagnostic unconstrained cloud-only estimator; do not use it to overwrite the
measurement-anchored asset unless intentionally discarding those dimensions.

Dependencies outside Blender: `numpy`, `scipy`, `trimesh`, `open3d`, `Pillow`,
`matplotlib`. Blender uses its bundled Python API and invokes external Python
with NumPy/SciPy for hull triangulation. `fit.py` and `prepare.py` need only
NumPy/SciPy/trimesh/Pillow.
On this host, verification used `~/miniforge3/envs/3dgrut/bin/python` and the
simulation smoke test used `~/miniforge3/envs/maniskill/bin/python`, both with
`PYTHONNOUSERSITE=1`; `MPLCONFIGDIR=/tmp/snap-pot-mpl` avoids shared cache writes.
The downloaded Blender binary is `/tmp/blender-4.5.9-linux-x64/blender`, from
the [official Blender archive](https://download.blender.org/release/Blender4.5/),
with its published SHA256 verified before execution. `/tmp` may be cleaned later.

To load into an existing ManiSkill scene:

```python
import sapien
from recon.snap_pot.maniskill_asset import build_snap_pot

# Supply measured values or explicitly chosen task assumptions.
pot = build_snap_pot(
    scene,
    "assets/heinz_snap_pot",
    mass_kg=measured_mass_kg,
    friction=task_friction,
    initial_pose=sapien.Pose(p=[0.0, 0.0, 0.001]),
)
```

The loader uses metre coordinates with no rotation or scale correction, loads
all OBJ convex pieces with `decomposition="none"`, and enforces the supplied
mass. Inertia is explicitly a uniform filled-envelope proxy, so PhysX convex
cooking cannot silently change the requested mass. Adjust physical properties
for the real filled/empty pot before relying on grasp dynamics.

The revised lid exceeds the Panda's 80 mm opening in its 95 mm long direction.
The short lid direction has only 6 mm nominal total margin, versus 25 mm for the
55 mm body short direction immediately below the lid. No grasp trial was run.

Verification returns a nonzero status for a structural mesh failure. A metric
failure or absent rulers is recorded as `NOT_FULLY_VERIFIED`, while retaining
the assets and detailed diagnostic files. Read `verification.json`; process exit
success alone does not certify object accuracy.

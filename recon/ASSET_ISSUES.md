# Microwave digital asset: open issues (updated 2026-10-14, 18:45)

State of the scan -> asset pipeline after the first real scan (`scans/microwave`,
shots `closed` + `open_90`, microwave on a dining table, scanned all the way round).
Data: `Gitignore-but-important/laptop/microwave/`, processing in
`Gitignore-but-important/recon/microwave/`.

## What works (tested)

| Step | Script | Evidence |
|---|---|---|
| frames + aligned depth | `capture_to_r2s.py` | 40 sharpest frames per shot |
| Pi3X poses + TSDF of the measured depth | `fuse_depth.py --cond K` | per-view scale spread 2.0 % / 2.8 % |
| model registration + door angle | `register_model.py` | synthetic: < 0.1 deg, < 1 mm, door 0/85 deg exact |
| colour bake from RGB frames | `bake_colors.py` | synthetic: re-render vs frames median 1/255, p95 2/255; occluder not baked |
| per-frame pose against the model | `frame_poses.py` | real: rmse 1.3-3 mm per frame, 39/40 and 34/40 frames kept |
| synthetic tests | `tests/synth_scan.py`, `tests/check_bake.py` | known-answer scan incl. occluder |

## Open issues

1. **Model dimensions disagree with the sensor.** Single depth frames (no fusion, no
   Pi3X) of the top surface and the table:

   | | model | single-frame depth |
   |---|---|---|
   | width | 460 mm | 445-454 mm (ok) |
   | total height, table to top | 268 mm (26 + 0.8) | **279 mm** (11 frames, p10-p90 274-280) |
   | total depth incl. door | 346 mm | **~300-316 mm** |

   The 346 mm comes from *my* assumption that the measured 31 cm excludes the door; the
   sensor says 31 cm is the whole depth. Then the measured cavity depth (29 cm) plus
   the door (3.6 cm) does not fit either. **Re-measure with a tape: total height
   (counter to top), total depth (door front to back), cavity depth (front frame to
   back wall).** Still estimated: door max angle, handle x, turntable diameter.

2. **Fused cloud of a box scanned all the way round is wrong.** Pi3X takes some back
   views for front views; the fusion grows a door on the back as well. The two shots
   disagree by 9 mm median (24 mm p90) in the model frame. Fused clouds should not be
   used for the microwave's geometry or colour.

3. **Front/back ambiguity in registration.** The microwave is nearly a box; only the
   3.6 cm door step, handle, knob and panel tell front from back. First run: flipped
   180 deg (yaw search was +-30 deg around the mean camera, which is meaningless for a
   full orbit) -> fixed with a full 360 deg search. After adding the stand-on-the-
   counter constraint, `closed` flipped again (overlay: display drawn on the left in
   a front view) -- with the wrong dimensions (issue 1) the flipped fit scores as well
   as the right one. Per-frame refinement is local and inherits the flip.
   **Fix: correct dimensions first; then also score front/back explicitly (handle
   and knob are 2-6 cm proud of the front) and always look at the overlay.**

4. **Pi3X shot poses are cm-level off.** Per-frame registration moved frames by
   1-7 cm and 1-5 deg from the shot-level pose. Use per-frame poses for anything
   metric (colour bake now; hand trajectories in the demos later).

5. **Reflective surfaces.** Door glass, display, steel interior: depth holes, purple
   speckle (IR pattern), mirror ghosts -- 8.4 % of the scan points near the microwave
   lie > 1.5 cm inside its solid parts. Plan: geometry of the microwave from the model
   only; the scan only for the environment (table, walls), with points inside or
   behind the model removed; colour from RGB via `bake_colors.py`, whose visibility
   is ray-cast against the model, not the depth.

6. **Colour bake not yet run on real data** -- blocked on 1 and 3.

## Next capture (environment, wider) and the 3 task videos

- Put the microwave where the tasks will happen and **do not move it** between the
  environment scan and the task videos -- then one registration serves all of them.
- Environment scan: an arc in front (no need to walk round the back), several
  heights, include the counter and its surroundings out to ~1.5 m; slow.
- Task videos: start each with 2-3 s holding still, looking at the microwave front
  with the counter in view (anchors the first frame's pose); keep the microwave in
  view as much as possible; one hand.

## Update 18:45

Resolved:
- **Dimensions (1):** re-measured 27.5-28 cm counter to top, 31-32 cm overall depth
  incl. the door, cavity 29 cm at the edges and 31.5 cm in the middle -- the cavity
  runs into a bump on the back, now modelled. Matches the single-frame depth.
- **Flip (3):** with the right dimensions both shots register the right way round
  (back bump and front details line up in the overlays); per-frame registration
  keeps 40/40 and 39/40 frames at 2.0-2.2 mm rmse.
- **Colour (6):** baked from per-frame poses (`frame_poses.py` -> `bake_colors.py`).
  The front is a decal from a straight-on phone photo (`front_photo.jpg`, 0.4 mm
  texels, colour-matched to the scan); the handle and knob are masked out of it
  (`make_front_mask.py`, they stand proud and show parallax) and the 3D controls are
  placed at their photo positions through the same homography. Output:
  `assets/microwave/microwave_textured.urdf` + `textured/{body,door}.glb`.

Open:
- **Kitchen environment scan** (`scans/kitchen`, a room sweep, 60 frames): fused
  (2.2 M points, per-view scale spread 6.1 %, worse than the object scans), the
  microwave located with `register_model.py --seed` (a box around it in frame 0) --
  right way round but 3-5 cm off in the overlays. The sweep sees the microwave only
  in a few frames, from 1-1.8 m, mostly one side face; per-frame ICP then slides along
  that face (corrections 3-24 cm, inconsistent) and is not used. Anchor it with a
  short clip that looks at the microwave front from 0.5-0.7 m without moving it (the
  task videos' 2-3 s still start can serve).

## Update 2026-10-16 20:10: the pot's lid texture

Spotted by eye on the final video: the real Snap Pot's rim is white plastic (the film's
orange border sits inside it, `Gitignore-but-important/recon/beans/input/0010.png`), but the
lid texture painted the whole rim orange -- the flange outside the film included. Fixed in
`recon/snap_pot/prepare.py` (white flange / orange film border 6 mm / teal centre; geometry
unchanged) and `assets/heinz_snap_pot/{visual.glb,lid_texture.png}`; the old texture is in
`history/brown_rim_v3/`. Every policy was trained and, until this point, evaluated with the
orange rim; the README's final numbers are re-evaluated with the white one and say so.
The replay and solver GIFs were regenerated with the corrected pot the same evening; every image in the README now shows the white rim.

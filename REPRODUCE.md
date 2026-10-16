# Reproducing the results

Two routes. **Quick** (about 40 minutes on one GPU): download the released checkpoint and
demonstrations and re-run the 128-start evaluation that gives the headline number.
**Full** (about a day): from the captures to the policy. Everything runs from the repo root
with `PYTHONNOUSERSITE=1`; `G=Gitignore-but-important/sim`, `K=Gitignore-but-important/recon/kitchen`.

## Hardware and environments

Used here: a desktop with an RTX 5090 (training, rendering, evaluation; 32 CPU cores for
the evaluation pool); a laptop with a RealSense D435i (capture); a DGX Spark and a Jetson
Thor for extra runs (on the Thor add `--tasks-per-child 4` to evaluations). Conda
environments:

| env | used for | key packages |
|---|---|---|
| `maniskill` | `sim/`, `recon/replay_take.py`, `recon/door_track.py` | mani_skill 3.0.1, sapien 3.0.3, torch 2.11, scipy |
| `3dgrut` | Pi3X, depth fusion, registration, colour bake, `recon/sweep_dense.py` | torch 2.11, open3d 0.19, [Pi3](https://github.com/yyfz/Pi3) (`PI3_ROOT`) |
| `sam3d` | the Gaussian splat: `recon/gs_kitchen.py`, `gs_inpaint.py`, `gs_server.py` | torch 2.7, [gsplat](https://github.com/nerfstudio-project/gsplat) 1.5.3, open3d |
| `handtrack` | `recon/hand_track.py` | mediapipe 1.1 |
| `realsense` / `rscap` | `capture/`, `recon/capture_to_r2s.py` | pyrealsense2 2.58, opencv |

The splat renderer runs in its own process (`sim/gs_client.py` starts `recon/gs_server.py`
in the `sam3d` env; set `SPLAT_PYTHON` if that interpreter is elsewhere) because gsplat and
ManiSkill need different torch versions.

## Data and checkpoints

[Release v1.0](https://github.com/HanshangZhu/microwave-from-human-video/releases/tag/v1.0),
`SHA256SUMS.txt` alongside:

| asset | size | what | unpack to |
|---|---|---|---|
| `robot_demos_solve.tar` | 148 MB | the 96 + 109 solver demonstrations (`solve.h5`: states and actions; summaries, logs) | `Gitignore-but-important/sim/` (creates `gen/`, `gen_dart02/`, `gen_dart04/`, ...) |
| `policy_v6_long.pt` | 104 MB | the submitted camera policy (EMA weights, normalisation, config) | `Gitignore-but-important/sim/vis_v6_long/dit_vis.pt` |
| `policy_v6_long_whiterim_ft.pt` | 104 MB | the same, fine-tuned 12 k steps on the corrected pot | `Gitignore-but-important/sim/vis_v6_long_wr/dit_vis.pt` |

Not published, because they are recordings of my home: the three RGB-D takes, the room
sweep, the per-take keyframes and the room splat. For these, write to zcabhz4@ucl.ac.uk.
Without them the quick route below reproduces the grey-floor number; the splatted-room
number needs the splat checkpoint (on request), the full route needs the captures.

## Quick route: evaluate the released policy (~40 min)

```bash
mkdir -p Gitignore-but-important/sim/vis_v6_long
tar -xf robot_demos_solve.tar -C Gitignore-but-important/sim/      # the evaluation reads gen/summary.json for the start region
cp policy_v6_long.pt Gitignore-but-important/sim/vis_v6_long/dit_vis.pt
export PYTHONNOUSERSITE=1; G=Gitignore-but-important/sim
python sim/dit_vis.py eval --ckpt $G/vis_v6_long/dit_vis.pt --out $G/vis_v6_long/eval128 --n 128 --jobs 6 --videos 2
```

Expected: `rates: {... 'success': '74%'} (n=128)` with the pot asset in this repo (white
rim); 83 % with the orange-rim texture in `assets/heinz_snap_pot/history/brown_rim_v3/`
(copy its `visual.glb` and `lid_texture.png` over the asset's). Run to run, the same 128
starts move by about +-5 points with the sampling noise (`NOISE_SEED_OFFSET=1 python ...`);
`--seed 2` draws 128 new starts. `eval.json` holds every episode; `video_00/`, `video_01/`
the first two as videos. 6 workers take ~10 GB of GPU memory and ~25-30 minutes.

With the splat checkpoint (`$K/gs_dense_po/ckpt.pt`, on request):

```bash
python sim/gs_cams.py --out $K/gs/cams_front.npz
conda run -n sam3d python recon/gs_kitchen.py render --keep-all --ckpt $K/gs_dense_po/ckpt.pt \
    --cams $K/gs/cams_front.npz --out $K/gs_dense_po/bgall_front.npz
python sim/dit_vis.py eval --ckpt $G/vis_v6_long/dit_vis.pt --out $G/vis_v6_long/eval128_gs_both --n 128 --jobs 6 \
    --real-bg $K/gs_dense_po/bgall_front.npz --splat-wrist $K/gs_dense_po/ckpt.pt       # expected 68 %
```

## Full route: from the captures

```bash
# 1. capture (laptop, env realsense) -- capture/README.md
python capture/scan.py env --out scans/microwave
python capture/record.py --out captures/microwave01 --count 3 --duration 40 --reset 15

# 2. microwave asset: measured URDF, then the scanned colours (env 3dgrut)
python assets/microwave/build_microwave.py && python assets/microwave/check_microwave.py
bash recon/scan_to_textured.sh <scan dir> assets/microwave/microwave.urdf <work dir> closed open_90

# 3. takes -> keyframes and the replay video (recon/DEMOS.md; env 3dgrut, hand_track in handtrack, replay in maniskill)
python recon/take_anchor.py captures/microwave01 take_000 --out takes/take_000
python recon/register_model.py takes/take_000 takes/take_000/run --urdf assets/microwave/microwave.urdf --seed "0 300 60 800 400"
python recon/door_track.py captures/microwave01 take_000 --anchor takes/take_000
python recon/pot_track.py  captures/microwave01 take_000 --anchor takes/take_000
python recon/hand_track.py captures/microwave01 take_000 --anchor takes/take_000
python recon/replay_take.py captures/microwave01 take_000 --anchor takes/take_000

# 4. robot demonstrations (env maniskill; CPU physics, a few minutes per take, tens of minutes per batch)
python sim/solve_microwave.py --take take_000 --out $G/take_000
python sim/generate.py --n 60 --out $G/gen --jobs 10
python sim/generate.py --n 80 --seed 2 --dart-noise 0.02 --jobs 14 --videos 0 --out $G/gen_dart02
python sim/generate.py --n 40 --seed 3 --dart-noise 0.04 --jobs 14 --videos 0 --out $G/gen_dart04

# 5. camera images, two random scenes per demo (~15 min per scene on the 5090 with --jobs 6)
for v in 0 1; do python sim/render_views.py --data $G/gen,$G/gen_dart02,$G/gen_dart04 --jobs 6 --dr doorman --variant $v; done

# 6. the camera policy (~2.5 h on the 5090) and its evaluations
python sim/dit_vis.py train --data $G/gen,$G/gen_dart02,$G/gen_dart04 --out $G/vis_v6_long --epochs 240 \
    --encoder resnet18 --views views_drm0,views_drm1 --ema --erase 0.3 --color-aug --save-every 80
python sim/dit_vis.py eval --ckpt $G/vis_v6_long/dit_vis.pt --out $G/vis_v6_long/eval128 --n 128 --jobs 6

# 7. the room as a Gaussian splat (env 3dgrut for sweep_dense, sam3d for the rest), then the policy in it
python recon/sweep_dense.py Gitignore-but-important/laptop/kitchen closed $K/closed --out $K/closed_dense --batch 16 --anchors 24
python recon/gs_inpaint.py --lama big-lama.pt --sweep $K/closed_dense --out $K/gs_inpaint_dense --dilate 18   # LaMa: github.com/enesmsahin/simple-lama-inpainting
python recon/gs_kitchen.py train --out $K/gs_dense_po --steps 20000 --sweep $K/closed_dense --inpainted $K/gs_inpaint_dense --pose-opt --holdout 8
python sim/gs_cams.py --out $K/gs/cams_front.npz && python sim/gs_cams.py --demo --out $K/gs/cams_demo.npz
python recon/gs_kitchen.py render --keep-all --ckpt $K/gs_dense_po/ckpt.pt --cams $K/gs/cams_front.npz --out $K/gs_dense_po/bgall_front.npz
python recon/gs_kitchen.py render --keep-all --ckpt $K/gs_dense_po/ckpt.pt --cams $K/gs/cams_demo.npz --out $K/gs_dense_po/bgall_demo.npz
python sim/dit_vis.py eval --ckpt $G/vis_v6_long/dit_vis.pt --out $G/vis_v6_long/eval128_gs_both --n 128 --jobs 6 \
    --real-bg $K/gs_dense_po/bgall_front.npz --splat-wrist $K/gs_dense_po/ckpt.pt
python sim/gs_demo.py --ckpt $G/vis_v6_long/dit_vis.pt --front $K/gs_dense_po/bgall_front.npz --splat-wrist $K/gs_dense_po/ckpt.pt \
    --demo $K/gs_dense_po/bgall_demo.npz --cams $K/gs/cams_demo.npz --out <dir>        # the video's frames
```

The G1 run: `python sim/g1_spike/g1_solve.py --take take_000 --out <dir> --video`, then
`sim/g1_spike/replay_splat.py` to re-film it in the splat. Every other experiment in
`sim/EXPERIMENTS.md` states its command next to its numbers; the demoted scripts are under
`archive/` (indexed in `archive/README.md`).

Each script's docstring states what it does, why, and its flags. Rollouts are not
bit-identical across machines (x86 vs aarch64 PhysX); compare paired on one machine.

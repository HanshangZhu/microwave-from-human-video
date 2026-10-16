# archive/ -- demoted, kept as evidence

Code that produced numbers in `sim/EXPERIMENTS.md` but is no longer on the path to the
submitted result. Nothing here is referenced by the live pipeline; each folder is runnable
from the repo root as its docstrings say. Paths inside `sim/EXPERIMENTS.md` still read
`sim/<name>.py` for these; the mapping is below.

| folder | was | numbers | why demoted |
|---|---|---|---|
| `2026-10-13-human-press/` | day 1 | README "What did not work" | the task (a cap press) carried no information a human could add; superseded by the microwave task |
| `state-policy/` | `sim/bc.py`, `dit.py`, `dagger.py`, `rl_cpu.py`, `rl_residual.py`, `probe_close.py` | EXPERIMENTS 2-6, 8 | privileged-state policies, DAgger and residual RL: the test bed the camera policy (`sim/dit_vis.py`) outgrew; RL and DAgger were negative |
| `mesh-backdrop/` | `recon/kitchen_backdrop.py`, `backdrop_colours.py` | EXPERIMENTS 7d (the 81 % "scanned room") | the fused-depth mesh of the room, superseded by the Gaussian splat (`recon/gs_kitchen.py`) |
| `newton_spike/` | `sim/newton_spike/` | EXPERIMENTS 10 | NVIDIA Newton physics check; not adopted |
| `lingbot-va-zeroshot/` | `sim/va_zeroshot.py` | EXPERIMENTS 8b | LingBot-VA zero-shot on the task; suggestive only |

Intermediate artefacts that were never code -- the pot asset's measurement rounds, candidate
meshes and render sheets (41 MB), two exploratory G1 stance scripts -- are kept outside the
public tree (`Gitignore-but-important/assets_provenance/`, available on request).

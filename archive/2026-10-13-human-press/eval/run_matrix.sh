#!/bin/bash
# The experiment: do the BF16 time-grid bug and the 1.57x speedup behave the same
# on a model post-trained on recorded human presses as they do on the released one?
#
#   arms   fp32          precision ceiling
#          bf16bug       upstream server with LINGBOT_BF16_TIME_GRID=1, 126 ms
#          bf16          upstream server, grid fixed, 126 ms  <- bug A/B against bf16bug
#          fast          our fast path, grid still BF16, 79 ms
#          leang         our fast path, grid fixed, 79 ms     <- bug A/B against fast,
#                                                                speed A/B against bf16
#
# The fix now lives in the source tree, so `bf16` is upstream WITH it; the
# September 48% was measured with LINGBOT_BF16_TIME_GRID=1 on a different seed
# set. bf16bug restores the old behaviour on these seeds, which turns "48% then
# vs 75% now" into a controlled A/B.
#   models base | hp_direct   (the second exists only because of the recorded presses)
#   tasks  click_bell
#
# An earlier version also evaluated on `click_bell_ood`, the bell placed on the
# half of the table the task never samples and which the recorded presses do
# cover. It was dropped: the base policy already scores 100 / 98 / 100 % there
# against 75 / 57 / 70 % in the sampled region, so that half is simply easier to
# reach and there is no headroom for anything to show. "Never sampled" and
# "hard" are independent, which this got wrong. The numbers are kept and
# reported as an observation about the benchmark, not as evidence about the data.
#
# One server at a time. Two policy servers plus two SAPIEN renderers do not fit
# in 32 GB: the first attempt ran bf16 and fast together and every episode came
# back `success: null, error: RuntimeError('cannot create buffer')` -- the
# renderer failing to allocate a camera buffer, which reads as 0% success and is
# not a result at all. Cells resume, so this is safe to re-run.
#
#   bash eval/run_matrix.sh <run dir>
set -u
RUN=${1:?usage: run_matrix.sh <run dir>}
ACC=/home/hanszhu/Research/VLA/lingbot-vla-v2-accel
WS=/home/hanszhu/Research/VLA/Humanoid-Internship-Challenge
BASE_CKPT=$ACC/models/lingbot-vla-v2-6b-robotwin/checkpoints/global_step_50000/hf_ckpt
HP_CKPT=$WS/Gitignore-but-important/train_out/direct/checkpoints/global_step_990/hf_ckpt
N_BF16=60
N_FP32=30

done_count() {   # episodes with a real verdict (null = the episode errored)
  python3 - "$1" <<'PY' 2>/dev/null || echo 0
import json,sys,os
f=sys.argv[1]
print(sum(1 for l in open(f) if json.loads(l).get("success") is not None) if os.path.exists(f) else 0)
PY
}

cell() {   # cell <model> <ckpt> <arm> <task> <n>
  local model=$1 ckpt=$2 arm=$3 task=$4 n=$5
  local out=$RUN/${model}__${task}
  local eps=$out/$arm/$task/episodes.jsonl
  local have; have=$(done_count "$eps")
  if [ "$have" -ge "$n" ]; then
    echo "[matrix] skip $model/$arm/$task - already $have/$n"
    return
  fi
  mkdir -p "$out"
  echo "[matrix] $model / $arm / $task / $have -> $n seeds  $(date +%T)"
  VLA2_CKPT=$ckpt CLIENT_EXTRA="--seed-start 0" \
    "$ACC/closed_loop/run_arm.sh" "$arm" "$out" "$n" "$task"
}

for model in base hp_direct; do
  ckpt=$BASE_CKPT; [ "$model" = hp_direct ] && ckpt=$HP_CKPT
  for task in click_bell; do
    for arm in bf16 fast leang; do
      cell "$model" "$ckpt" "$arm" "$task" $N_BF16
    done
    # same server as bf16, old grid restored; separate output dir
    eps=$RUN/${model}__${task}/bf16bug/$task/episodes.jsonl
    if [ "$(done_count "$eps")" -lt $N_BF16 ]; then
      echo "[matrix] $model / bf16bug / $task  $(date +%T)"
      tmp=$RUN/_bug; mkdir -p "$tmp"
      [ -d "$RUN/${model}__${task}/bf16bug" ] && mv "$RUN/${model}__${task}/bf16bug" "$tmp/bf16"
      VLA2_CKPT=$ckpt CLIENT_EXTRA="--seed-start 0" LINGBOT_BF16_TIME_GRID=1 \
        "$ACC/closed_loop/run_arm.sh" bf16 "$tmp" $N_BF16 "$task"
      mkdir -p "$RUN/${model}__${task}"
      rm -rf "$RUN/${model}__${task}/bf16bug"
      mv "$tmp/bf16" "$RUN/${model}__${task}/bf16bug"; rmdir "$tmp" 2>/dev/null
    else
      echo "[matrix] skip $model/bf16bug/$task"
    fi
  done
done

for model in base hp_direct; do          # fp32 last: slowest per call, and the
  ckpt=$BASE_CKPT                        # FP32 closed loop is deterministic, so
  [ "$model" = hp_direct ] && ckpt=$HP_CKPT   # it needs fewer seeds
  for task in click_bell; do
    cell "$model" "$ckpt" fp32 "$task" $N_FP32
  done
done
echo "[matrix] finished $(date +%T)"

# VLA Manipulation — ACT on LIBERO

ACT (Action Chunking Transformer) policy trained on the LIBERO-Object benchmark,
with a closed-loop success-rate evaluation in MuJoCo.

**Trained weights:** [rishi264/act-libero-alphabet-soup](https://huggingface.co/rishi264/act-libero-alphabet-soup)
on the Hugging Face Hub. Load them with the preprocessor and postprocessor, not on their
own — the normalization lives in those pipelines, and without them the policy silently
does nothing useful.

## Task
Pick up the alphabet soup and place it in the basket.

## Results

Success rate over 50 episodes, one per LIBERO initial state, 600-step cap:

| Policy | Success | Mean steps |
|---|---|---|
| **20k steps + temporal ensembling** | **47/50 = 94%** | 145.8 |
| 20k steps (execute full 50-action chunk) | 45/50 = 90% | 149.4 |
| 10k steps | 45/50 = 90% | 142.5 |
| 5k steps | 43/50 = 86% | 174.5 |
| 20k steps, re-plan every 10 actions | 42/50 = 84% | 169.9 |
| *previous checkpoint (`model_final/`)* | *0/50 = 0%* | — |

For reference the expert demonstrations average 156 steps, so the policy is not
merely succeeding but doing so at roughly demonstration speed.

Reproduce the table with `python summarize_results.py`; the raw per-episode
records are in `results/`.

## Three bugs stood between 0% and 94%

The previous version of this project reported a training loss of 0.31 but had
never measured a success rate; its single recorded rollout failed. Measuring it
properly (0/50) and fixing these took it to 94%:

1. **The action chunks were fake.** `train_act.py` built its 50-step target with
   `action.unsqueeze(1).expand(-1, 50, -1)` — the current action repeated 50
   times. ACT never learned to plan a trajectory. The fix is `delta_timestamps`,
   which makes the dataset return the true next 50 actions plus the
   `action_is_pad` mask for the end of an episode.

2. **Normalization was silently dropped.** In current LeRobot, `ACTPolicy` does
   no normalization itself — a preprocessor mean/std normalizes inputs and a
   postprocessor un-normalizes the predicted actions. The old code passed
   `dataset_stats=` to the `ACTPolicy` constructor, where it landed in `**kwargs`
   and was ignored, and inference never built the pipelines at all. The net was
   fed unnormalized inputs and its normalized outputs went to the robot as if
   they were joint deltas. Checkpoints now save their processors next to the
   weights.

3. **The image statistics were wrong.** LeRobot wrote the std of each frame's
   *mean* pixel (0.0019) instead of the std across pixels (0.16). Dividing by it
   pushed normalized pixels to about ±300 and the vision backbone learned
   nothing — the l1 loss sat flat at ~0.70, barely better than predicting the
   mean action. `fix_image_stats.py` rewrites `meta/stats.json` with true
   per-pixel values, and the converter now calls it automatically. This alone is
   the difference between a policy that never improves and one that reaches 86%
   in 5k steps.

Two other things worth knowing:

- Training images were being upscaled 128→256 while the evaluator rendered
  natively at 256, so the policy saw blurrier images in training than at test
  time. Everything now runs at LIBERO's native 128.
- `check_obs_alignment.py` settles the vertical-flip question by driving the sim
  to a demo's recorded state and diffing the rendered frames against the stored
  ones. No flip is correct (as-is error 39 vs 53 flipped), and proprioception
  matches to 0.005.

## Note on the loss number

The old README's "final loss 0.31" is not comparable to this run's 0.17. That
0.31 was fitting a trivial target — one action copied 50 times — so a low number
there meant very little. This run's loss is against real 50-step trajectories,
and it is the success rate, not the loss, that says whether the policy works.

## Stack
- Policy: ACT (LeRobot 0.5.2)
- Benchmark: LIBERO-Object, 50 expert demonstrations, 7,808 frames
- Simulator: MuJoCo 3.9 via robosuite 1.4.1 / LIBERO
- Training: 20,000 steps, batch 8, lr 1e-4, ~75 min on an RTX 3060 laptop GPU

## Files
| File | Purpose |
|---|---|
| `Dockerfile` / `run_docker.sh` | reproducible environment and its mounts |
| `convert_libero_to_lerobot.py` | LIBERO HDF5 → LeRobot dataset (fixes stats at the end) |
| `fix_image_stats.py` | rewrites image stats with true per-pixel mean/std |
| `train_act.py` | training with real action chunks and normalization |
| `eval_policy.py` | multi-episode closed-loop success rate |
| `policy_runner.py` | shared checkpoint loading + observation building |
| `visualize.py` / `run_viz.sh` | live MuJoCo window: demo replay or policy rollout |
| `check_obs_alignment.py` | verifies rendered observations match training data |
| `summarize_results.py` | prints the results table |

## Running it

```bash
docker build -t vla-libero:latest .
./run_docker.sh python /workspace/proj/convert_libero_to_lerobot.py
./run_docker.sh python /workspace/proj/train_act.py --steps 20000
./run_docker.sh python /workspace/proj/eval_policy.py \
    --model-dir /workspace/proj/outputs/act_chunked/final \
    --episodes 50 --temporal-ensemble 0.01
./run_viz.sh policy --model-dir /workspace/proj/outputs/act_chunked/final   # live window
```

LIBERO and LeRobot are mounted from `/home/rishi/LIBERO` and `/home/rishi/lerobot`;
the converted dataset lands in `/home/rishi/lerobot_data/alphabet_soup`.

To skip training and evaluate the published checkpoint instead, pass
`--model-dir rishi264/act-libero-alphabet-soup` to `eval_policy.py`.

## Limitations
- One task, one object; no generalization claims.
- Not language-conditioned — vision + proprioception only. This is ACT on a
  single task, not a generalist VLA.
- Simulation only; no real-robot transfer.
- Evaluated on the 50 LIBERO initial states, one episode each; no repeated seeds
  per state, so the ±3% band around 94% is not resolved.

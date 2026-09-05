"""Measure closed-loop success rate of an ACT checkpoint on the LIBERO task.

Runs N episodes from LIBERO's own held-out initial states and reports
successes / trials -- the number the project never had. Headless (EGL), so it
can run without a display.

  python eval_policy.py --model-dir /workspace/proj/outputs/act_chunked/final --episodes 50
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from libero_config_seed import seed_libero_config
seed_libero_config()

from libero.libero import benchmark, get_libero_path
from libero.libero.envs.env_wrapper import ControlEnv

from policy_runner import load_policy, predict_action

TASK_SUITE = "libero_object"
TASK_MATCH = "alphabet_soup"
IMG = 128  # native LIBERO resolution; overridden per-checkpoint below


def make_env(img=IMG):
    bm = benchmark.get_benchmark_dict()[TASK_SUITE]()
    names = bm.get_task_names()
    task_id = next(i for i, n in enumerate(names) if TASK_MATCH in n)
    task = bm.get_task(task_id)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = ControlEnv(
        bddl_file_name=bddl,
        has_renderer=False,
        has_offscreen_renderer=True,
        camera_heights=img, camera_widths=img,
        control_freq=20,
    )
    return env, bm, task_id, task


def load_init_states(bm, task_id):
    # LIBERO's get_task_init_states uses torch.load with weights_only=True,
    # which rejects these NumPy-pickled files.
    import torch
    task = bm.get_task(task_id)
    path = os.path.join(get_libero_path("init_states"), task.problem_folder, task.init_states_file)
    return torch.load(path, weights_only=False)


def run_episode(env, policy, pre, post, init_state, max_steps, dummy_steps, flip_images):
    env.reset()
    obs = env.set_init_state(init_state)
    # LIBERO's standard eval settles the physics before handing control to the
    # policy -- the object is still dropping into place on the first few frames.
    for _ in range(dummy_steps):
        obs, _, _, _ = env.step([0.0] * 6 + [-1.0])
    policy.reset()

    for t in range(max_steps):
        action = predict_action(policy, pre, post, obs, flip_images)
        obs, reward, done, info = env.step(action)
        if env.check_success():
            return True, t + 1
    return False, max_steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--dataset-root", default="/workspace/data/alphabet_soup",
                    help="only needed for checkpoints saved without processors")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--dummy-steps", type=int, default=10)
    ap.add_argument("--n-action-steps", type=int, default=None,
                    help="actions executed per inference (default: the checkpoint's chunk_size)")
    ap.add_argument("--temporal-ensemble", type=float, default=None,
                    help="enable temporal ensembling with this coefficient (e.g. 0.01)")
    ap.add_argument("--flip-images", action="store_true",
                    help="vertically flip camera images before the policy sees them")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None, help="write results JSON here")
    ap.add_argument("--label", default=None, help="name for this run in the JSON")
    args = ap.parse_args()

    print(f"[eval] MUJOCO_GL={os.environ.get('MUJOCO_GL')}")
    policy, pre, post = load_policy(
        args.model_dir, dataset_root=args.dataset_root, device=args.device,
        n_action_steps=args.n_action_steps, temporal_ensemble_coeff=args.temporal_ensemble,
    )
    img = policy.config.input_features["observation.images.image"].shape[-1]
    print(f"[eval] rendering at {img}x{img} to match the checkpoint")

    env, bm, task_id, task = make_env(img)
    print(f"[eval] task: {task.name}")

    init_states = load_init_states(bm, task_id)
    n_avail = len(init_states)
    print(f"[eval] {n_avail} initial states available; running {args.episodes} episodes")
    print(f"[eval] chunk_size={policy.config.chunk_size} "
          f"n_action_steps={policy.config.n_action_steps} "
          f"temporal_ensemble_coeff={policy.config.temporal_ensemble_coeff} "
          f"flip_images={args.flip_images}")

    results = []
    t_start = time.time()
    for ep in range(args.episodes):
        success, steps = run_episode(
            env, policy, pre, post, init_states[ep % n_avail],
            args.max_steps, args.dummy_steps, args.flip_images,
        )
        results.append({"episode": ep, "init_state": ep % n_avail,
                        "success": bool(success), "steps": int(steps)})
        n_succ = sum(r["success"] for r in results)
        print(f"[eval] ep {ep:3d} | {'SUCCESS' if success else 'fail   '} "
              f"in {steps:4d} steps | running {n_succ}/{ep + 1} = {n_succ / (ep + 1):.1%}",
              flush=True)

    env.close()
    n_succ = sum(r["success"] for r in results)
    rate = n_succ / len(results)
    succ_steps = [r["steps"] for r in results if r["success"]]
    elapsed = time.time() - t_start

    print("\n" + "=" * 60)
    print(f"model:        {args.model_dir}")
    print(f"success rate: {n_succ}/{len(results)} = {rate:.1%}")
    if succ_steps:
        print(f"mean steps to success: {np.mean(succ_steps):.1f}")
    print(f"elapsed: {elapsed / 60:.1f} min")
    print("=" * 60)

    if args.out:
        summary = {
            "label": args.label or os.path.basename(args.model_dir.rstrip("/")),
            "model_dir": args.model_dir,
            "episodes": len(results),
            "successes": n_succ,
            "success_rate": rate,
            "mean_steps_to_success": float(np.mean(succ_steps)) if succ_steps else None,
            "max_steps": args.max_steps,
            "dummy_steps": args.dummy_steps,
            "n_action_steps": policy.config.n_action_steps,
            "temporal_ensemble_coeff": policy.config.temporal_ensemble_coeff,
            "flip_images": args.flip_images,
            "elapsed_min": elapsed / 60,
            "results": results,
        }
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

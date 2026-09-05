"""
Visualize the alphabet-soup LIBERO task in MuJoCo.

  --mode demo    replay a recorded expert demonstration
  --mode policy  let the trained ACT model drive the arm

Rendering is on-screen (a live MuJoCo window) by default; set MUJOCO_GL=osmesa
and --no-window to run headless for debugging.
"""
import os, sys, argparse, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from libero_config_seed import seed_libero_config
seed_libero_config()

from libero.libero import benchmark, get_libero_path
from libero.libero.envs.env_wrapper import ControlEnv

TASK_SUITE = "libero_object"
TASK_MATCH = "alphabet_soup"
IMG = 128  # must match the training resolution (native LIBERO)


def make_env(window):
    bm = benchmark.get_benchmark_dict()[TASK_SUITE]()
    names = bm.get_task_names()
    task_id = next(i for i, n in enumerate(names) if TASK_MATCH in n)
    task = bm.get_task(task_id)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = ControlEnv(
        bddl_file_name=bddl,
        has_renderer=window,
        has_offscreen_renderer=True,
        camera_heights=IMG, camera_widths=IMG,
        control_freq=20,
    )
    env.seed(0)
    return env, bm, task_id, task


def render(env, window):
    if window:
        env.env.render()


def hold_window(env, window, seconds):
    """Keep the live window open and responsive for `seconds` after a run."""
    if not (window and seconds > 0):
        return
    print(f"[render] holding window open for {seconds}s ...")
    t0 = time.time()
    while time.time() - t0 < seconds:
        env.env.render()
        time.sleep(1 / 30)


def run_demo(env, bm, task_id, window, max_steps, fps, hold):
    import h5py
    ds_dir = get_libero_path("datasets")
    hdf5 = os.path.join(ds_dir, bm.get_task_demonstration(task_id))
    print(f"[demo] reading {hdf5}")
    assert os.path.exists(hdf5), f"demo file not found: {hdf5}"
    f = h5py.File(hdf5, "r")
    demo = f["data"]["demo_0"]
    states = demo["states"][:]
    actions = demo["actions"][:]
    print(f"[demo] {len(actions)} steps")
    env.reset()
    env.set_init_state(states[0])
    hold_window(env, window, 2)  # brief pause so the window is easy to spot
    n = min(len(actions), max_steps)
    for t in range(n):
        env.step(actions[t])
        render(env, window)
        if window:
            time.sleep(1 / fps)
    print("[demo] success:", env.check_success())
    hold_window(env, window, hold)
    f.close()


def run_policy(env, bm, task_id, window, max_steps, model_dir, fps, hold,
               dataset_root, flip_images, n_action_steps, temporal_ensemble,
               dummy_steps):
    import torch
    from policy_runner import load_policy, predict_action
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[policy] loading {model_dir} on {device}")
    policy, pre, post = load_policy(
        model_dir, dataset_root=dataset_root, device=device,
        n_action_steps=n_action_steps, temporal_ensemble_coeff=temporal_ensemble,
    )

    # Load init states directly: LIBERO's helper uses torch.load with the new
    # weights_only=True default, which rejects the NumPy-pickled .pruned_init file.
    task = bm.get_task(task_id)
    init_path = os.path.join(get_libero_path("init_states"),
                             task.problem_folder, task.init_states_file)
    init_states = torch.load(init_path, weights_only=False)
    env.reset()
    obs = env.set_init_state(init_states[0])
    # Let the scene settle before the policy takes over, as LIBERO's eval does.
    for _ in range(dummy_steps):
        obs, _, _, _ = env.step([0.0] * 6 + [-1.0])
    policy.reset()
    hold_window(env, window, 2)  # brief pause so the window is easy to spot
    success = False
    for t in range(max_steps):
        action = predict_action(policy, pre, post, obs, flip_images)
        obs, reward, done, info = env.step(action)
        render(env, window)
        if window:
            time.sleep(1 / fps)
        if env.check_success():
            success = True
            print(f"[policy] SUCCESS at step {t}")
            break
    print("[policy] success:", success)
    hold_window(env, window, hold)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["demo", "policy"], required=True)
    ap.add_argument("--no-window", action="store_true", help="headless (no live window)")
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--model-dir", default="/workspace/proj/model_final")
    ap.add_argument("--fps", type=float, default=30, help="playback speed (live window)")
    ap.add_argument("--hold", type=float, default=5, help="seconds to keep window open at end")
    ap.add_argument("--dataset-root", default="/workspace/data/alphabet_soup",
                    help="only needed for checkpoints saved without processors")
    ap.add_argument("--flip-images", action="store_true")
    ap.add_argument("--n-action-steps", type=int, default=None)
    ap.add_argument("--temporal-ensemble", type=float, default=None)
    ap.add_argument("--dummy-steps", type=int, default=10)
    args = ap.parse_args()
    window = not args.no_window
    print(f"[init] mode={args.mode} window={window} MUJOCO_GL={os.environ.get('MUJOCO_GL')}")
    env, bm, task_id, task = make_env(window)
    print(f"[init] task: {task.name} | instruction: {env.language_instruction}")
    if args.mode == "demo":
        run_demo(env, bm, task_id, window, args.max_steps, args.fps, args.hold)
    else:
        run_policy(env, bm, task_id, window, args.max_steps, args.model_dir, args.fps,
                   args.hold, args.dataset_root, args.flip_images, args.n_action_steps,
                   args.temporal_ensemble, args.dummy_steps)
    env.close()
    print("DONE")


if __name__ == "__main__":
    main()

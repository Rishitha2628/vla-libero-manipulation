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
import robosuite.utils.transform_utils as T

TASK_SUITE = "libero_object"
TASK_MATCH = "alphabet_soup"
IMG = 256


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


def build_obs(obs, device):
    import torch
    # Match training exactly: convert_libero_to_lerobot.py used the raw hdf5
    # agentview_rgb / eye_in_hand_rgb with NO vertical flip, so feed them as-is.
    ag = obs["agentview_image"]
    eih = obs["robot0_eye_in_hand_image"]
    def to_chw(img):
        x = torch.from_numpy(np.ascontiguousarray(img)).float().permute(2, 0, 1) / 255.0
        return x.unsqueeze(0).to(device)
    state = np.concatenate([
        obs["robot0_eef_pos"],
        T.quat2axisangle(obs["robot0_eef_quat"]),
        obs["robot0_gripper_qpos"],
    ]).astype(np.float32)
    st = torch.from_numpy(state).unsqueeze(0).to(device)
    return {
        "observation.images.image": to_chw(ag),
        "observation.images.image2": to_chw(eih),
        "observation.state": st,
    }


def run_policy(env, bm, task_id, window, max_steps, model_dir, fps, hold):
    import torch
    from lerobot.policies.act.modeling_act import ACTPolicy
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[policy] loading {model_dir} on {device}")
    policy = ACTPolicy.from_pretrained(model_dir)
    policy.to(device); policy.eval(); policy.reset()

    # Load init states directly: LIBERO's helper uses torch.load with the new
    # weights_only=True default, which rejects the NumPy-pickled .pruned_init file.
    task = bm.get_task(task_id)
    init_path = os.path.join(get_libero_path("init_states"),
                             task.problem_folder, task.init_states_file)
    init_states = torch.load(init_path, weights_only=False)
    env.reset()
    obs = env.set_init_state(init_states[0])
    hold_window(env, window, 2)  # brief pause so the window is easy to spot
    success = False
    for t in range(max_steps):
        batch = build_obs(obs, device)
        with torch.inference_mode():
            action = policy.select_action(batch)
        action = action.squeeze(0).cpu().numpy().astype(np.float32)
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
    args = ap.parse_args()
    window = not args.no_window
    print(f"[init] mode={args.mode} window={window} MUJOCO_GL={os.environ.get('MUJOCO_GL')}")
    env, bm, task_id, task = make_env(window)
    print(f"[init] task: {task.name} | instruction: {env.language_instruction}")
    if args.mode == "demo":
        run_demo(env, bm, task_id, window, args.max_steps, args.fps, args.hold)
    else:
        run_policy(env, bm, task_id, window, args.max_steps, args.model_dir, args.fps, args.hold)
    env.close()
    print("DONE")


if __name__ == "__main__":
    main()

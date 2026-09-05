"""Verify that the observations we feed the policy match the recorded training data.

Sets the simulator to a demo's exact recorded state, renders the cameras, and
compares against that demo's stored frames -- as-is and vertically flipped. The
orientation with the far lower error is the one inference must use. A large
error for *both* would mean something else (camera name, resolution) is off.
"""
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from libero_config_seed import seed_libero_config
seed_libero_config()

from eval_policy import make_env, IMG

HDF5 = ("/workspace/LIBERO/libero/datasets/libero_object/"
        "pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5")


def compare(name, rendered, recorded):
    a = rendered.astype(np.float32)
    b = recorded.astype(np.float32)
    as_is = np.abs(a - b).mean()
    flipped = np.abs(a[::-1] - b).mean()
    print(f"  {name:12s} shape={rendered.shape} "
          f"mean|err| as-is={as_is:7.2f}  flipped={flipped:7.2f}  "
          f"-> {'FLIP NEEDED' if flipped < as_is else 'no flip'}")
    return as_is, flipped


def main():
    env, bm, task_id, task = make_env()
    f = h5py.File(HDF5, "r")
    demo = f["data"]["demo_0"]

    env.reset()
    obs = env.set_init_state(demo["states"][0])

    print(f"[check] rendering at {IMG}x{IMG}; recorded frames are "
          f"{demo['obs']['agentview_rgb'].shape[1]}x{demo['obs']['agentview_rgb'].shape[2]}")
    print("[check] frame 0, straight after set_init_state:")
    compare("agentview", obs["agentview_image"], demo["obs"]["agentview_rgb"][0])
    compare("eye_in_hand", obs["robot0_eye_in_hand_image"], demo["obs"]["eye_in_hand_rgb"][0])

    # Replay a few recorded actions and re-check, so the verdict does not rest
    # on a single frame.
    for t in range(20):
        obs, _, _, _ = env.step(demo["actions"][t])
    print("[check] after replaying 20 recorded actions:")
    compare("agentview", obs["agentview_image"], demo["obs"]["agentview_rgb"][20])
    compare("eye_in_hand", obs["robot0_eye_in_hand_image"], demo["obs"]["eye_in_hand_rgb"][20])

    # Proprioception should line up too.
    import robosuite.utils.transform_utils as T
    state = np.concatenate([obs["robot0_eef_pos"], T.quat2axisangle(obs["robot0_eef_quat"]),
                            obs["robot0_gripper_qpos"]])
    recorded = np.concatenate([demo["obs"]["ee_pos"][20], demo["obs"]["ee_ori"][20],
                               demo["obs"]["gripper_states"][20]])
    print(f"[check] state rendered={np.round(state, 4)}")
    print(f"[check] state recorded={np.round(recorded, 4)}")
    print(f"[check] state mean|err|={np.abs(state - recorded).mean():.5f}")

    f.close()
    env.close()


if __name__ == "__main__":
    main()

import h5py
import numpy as np
import shutil
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import cv2

HDF5_PATH = "/workspace/LIBERO/libero/datasets/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5"
TASK_NAME = "pick_up_the_alphabet_soup_and_place_it_in_the_basket"
REPO_ID = "local/libero_alphabet_soup"
OUTPUT_DIR = Path("/workspace/data/alphabet_soup")
FPS = 30
IMG_SIZE = 128  # native LIBERO resolution; upscaling to 256 only blurred the data

f = h5py.File(HDF5_PATH, 'r')
demo0 = f['data']['demo_0']
ee_pos = demo0['obs']['ee_pos'][0]
ee_ori = demo0['obs']['ee_ori'][0]
gripper = demo0['obs']['gripper_states'][0]
state_dim = len(ee_pos) + len(ee_ori) + len(gripper)
print(f"state_dim: {state_dim}")
f.close()

# LeRobotDataset.create insists the directory does not exist yet.
if OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)

dataset = LeRobotDataset.create(
    repo_id=REPO_ID,
    fps=FPS,
    root=OUTPUT_DIR,
    features={
        "observation.images.image": {"dtype": "image", "shape": (IMG_SIZE, IMG_SIZE, 3), "names": ["height", "width", "channel"]},
        "observation.images.image2": {"dtype": "image", "shape": (IMG_SIZE, IMG_SIZE, 3), "names": ["height", "width", "channel"]},
        "observation.state": {"dtype": "float32", "shape": (state_dim,), "names": [f"s{i}" for i in range(state_dim)]},
        "action": {"dtype": "float32", "shape": (7,), "names": ["dx","dy","dz","dr","dp","dyw","gripper"]},
    },
    image_writer_threads=4,
)

f = h5py.File(HDF5_PATH, 'r')
for demo_key in list(f['data'].keys()):
    demo = f['data'][demo_key]
    actions = demo['actions'][:]
    agentview = demo['obs']['agentview_rgb'][:]
    eye_in_hand = demo['obs']['eye_in_hand_rgb'][:]
    ee_pos = demo['obs']['ee_pos'][:]
    ee_ori = demo['obs']['ee_ori'][:]
    gripper = demo['obs']['gripper_states'][:]
    for t in range(len(actions)):
        state = np.concatenate([ee_pos[t], ee_ori[t], gripper[t]])
        frame = {
            "observation.images.image": cv2.resize(agentview[t], (IMG_SIZE, IMG_SIZE)),
            "observation.images.image2": cv2.resize(eye_in_hand[t], (IMG_SIZE, IMG_SIZE)),
            "observation.state": state.astype(np.float32),
            "action": actions[t].astype(np.float32),
            "task": TASK_NAME,
        }
        dataset.add_frame(frame)
    dataset.save_episode()
    print(f"Saved {demo_key}")

f.close()
dataset.finalize()

# LeRobot's image stats come out as the std of each frame's mean pixel rather
# than the std across pixels, which makes MEAN_STD normalization explode.
# Rewrite them with true per-pixel values before anything trains on this.
import subprocess, sys as _sys
subprocess.run([_sys.executable, str(Path(__file__).parent / "fix_image_stats.py"),
                "--root", str(OUTPUT_DIR)], check=True)
print("Done!")

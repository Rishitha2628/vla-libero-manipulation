import os, sys
# Pre-seed LIBERO config so first import doesn't call input() and hang.
cfg_dir = os.path.expanduser("~/.libero")
os.makedirs(cfg_dir, exist_ok=True)
libero_pkg = "/workspace/LIBERO/libero/libero"
import yaml
with open(os.path.join(cfg_dir, "config.yaml"), "w") as f:
    yaml.dump({
        "benchmark_root": libero_pkg,
        "bddl_files": os.path.join(libero_pkg, "bddl_files"),
        "init_states": os.path.join(libero_pkg, "init_files"),
        "datasets": os.path.join(libero_pkg, "../datasets"),
        "assets": os.path.join(libero_pkg, "assets"),
    }, f)

import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

bm = benchmark.get_benchmark_dict()
print("benchmarks:", list(bm.keys()))
suite = bm["libero_object"]()
names = suite.get_task_names()
print("num tasks:", len(names))
# find the alphabet soup task
idx = next(i for i, n in enumerate(names) if "alphabet_soup" in n)
task = suite.get_task(idx)
print("TASK:", idx, task.name, "| bddl:", task.bddl_file)

bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
print("bddl path exists:", os.path.exists(bddl_file))

env = OffScreenRenderEnv(bddl_file_name=bddl_file, camera_heights=256, camera_widths=256)
env.seed(0)
obs = env.reset()
print("obs keys:", sorted(obs.keys()))
for k in obs:
    v = obs[k]
    if hasattr(v, "shape"):
        print(f"  {k}: {v.shape} {v.dtype}")
# step with a zero action (7-dim)
action = np.zeros(7, dtype=np.float32)
obs, reward, done, info = env.step(action)
print("stepped OK | reward:", reward, "done:", done)
img = obs.get("agentview_image")
print("agentview_image:", None if img is None else (img.shape, img.dtype, "min", int(img.min()), "max", int(img.max())))
env.close()
print("VALIDATE_OK")

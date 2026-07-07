#!/bin/bash
# Runs INSIDE the lerobot-env container. Mounts: /workspace/LIBERO, /workspace/lerobot
set -e
echo "===== python ====="; python --version
echo "===== install lerobot (editable, local checkout) ====="
pip install -e /workspace/lerobot 2>&1 | tail -5 || echo "LEROBOT_INSTALL_FAILED"
echo "===== install LIBERO runtime deps ====="
# loosened pins; robosuite 1.4 is the version LIBERO's env code targets
pip install "robosuite==1.4.1" bddl easydict transforms3d "gym==0.25.2" hydra-core matplotlib 2>&1 | tail -8 || echo "DEPS_INSTALL_FAILED"
echo "===== install LIBERO (no deps) ====="
pip install -e /workspace/LIBERO --no-deps 2>&1 | tail -5 || echo "LIBERO_INSTALL_FAILED"
echo "===== probe imports ====="
python - <<'PY' 2>&1 | tail -40
import numpy, mujoco
print("numpy", numpy.__version__)
print("mujoco", mujoco.__version__)
try:
    import robosuite; print("robosuite", robosuite.__version__)
except Exception as e:
    print("ROBOSUITE_IMPORT_ERR:", repr(e))
try:
    import bddl; print("bddl ok")
except Exception as e:
    print("BDDL_IMPORT_ERR:", repr(e))
try:
    from libero.libero import benchmark, get_libero_path
    print("LIBERO import ok")
    bm = benchmark.get_benchmark_dict()
    print("benchmarks:", list(bm.keys())[:6])
except Exception as e:
    import traceback; traceback.print_exc()
    print("LIBERO_IMPORT_ERR:", repr(e))
try:
    import lerobot
    from lerobot.policies.act.modeling_act import ACTPolicy
    print("lerobot + ACT ok")
except Exception as e:
    print("LEROBOT_IMPORT_ERR:", repr(e))
PY
echo "===== DONE ====="

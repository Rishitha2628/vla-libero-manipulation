#!/bin/bash
# Launch the LIBERO alphabet-soup visualization in a live MuJoCo window.
#   ./run_viz.sh demo            replay an expert demonstration
#   ./run_viz.sh policy          let the trained ACT model drive
#   ./run_viz.sh demo --no-window  (headless, for debugging)
# Any extra args are passed through to visualize.py.
set -e
MODE="${1:-demo}"; shift || true

# Allow the container to talk to the local X server.
xhost +local:root >/dev/null 2>&1 || true

docker run --rm --gpus all \
  -e DISPLAY="$DISPLAY" \
  -e MUJOCO_GL=glfw \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  --device /dev/dri \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /home/rishi/LIBERO:/workspace/LIBERO \
  -v /home/rishi/lerobot:/workspace/lerobot \
  -v /home/rishi/vla-libero-project:/workspace/proj \
  -v /home/rishi/vla-libero-project/macros_private.py:/usr/local/lib/python3.12/dist-packages/robosuite/macros_private.py:ro \
  vla-libero:latest python /workspace/proj/visualize.py --mode "$MODE" "$@"

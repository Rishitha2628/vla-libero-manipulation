#!/bin/bash
# Run any command inside the vla-libero image with the project's mounts.
# Headless: uses EGL off-screen rendering on the NVIDIA GPU (no X needed).
#   ./run_docker.sh python /workspace/proj/eval_policy.py --model-dir ...
set -e
mkdir -p /home/rishi/lerobot_data
docker run --rm --gpus all \
  -e MUJOCO_GL="${MUJOCO_GL:-egl}" \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e PYTHONPATH=/workspace/lerobot/src:/workspace/LIBERO:/workspace/proj \
  -e HF_LEROBOT_HOME=/workspace/hf_home \
  -v /home/rishi/LIBERO:/workspace/LIBERO \
  -v /home/rishi/lerobot:/workspace/lerobot \
  -v /home/rishi/vla-libero-project:/workspace/proj \
  -v /home/rishi/lerobot_data:/workspace/data \
  -v /home/rishi/hf_home:/workspace/hf_home \
  --shm-size=8g \
  -w /workspace/proj \
  vla-libero:latest "$@"

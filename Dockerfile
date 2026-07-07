# Self-contained image for training + visualizing the ACT / LIBERO alphabet-soup
# policy. Builds from a public CUDA base, so it works on any machine in one
# command (no pre-built local base image required):
#
#   docker build -t vla-libero:latest .
#
# Replaces the former two-stage split (Dockerfile.lerobot -> Dockerfile.vla).
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# --- Python 3.12 + pip ------------------------------------------------------
RUN apt-get update && apt-get install -y \
    software-properties-common curl git \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.12 python3.12-dev python3.12-venv \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12 \
    && rm -rf /var/lib/apt/lists/*
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/local/bin/pip3.12 1

# --- Graphics / OpenGL system libs for MuJoCo + robosuite rendering ----------
# (covers on-screen GLFW windows and off-screen EGL/OSMesa, plus X11 clients).
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libosmesa6 libegl1 libgles2 \
    libglew-dev libglfw3 libglfw3-dev \
    libsm6 libxext6 libxrender1 libx11-6 libxcursor1 \
    libxinerama1 libxi6 libxrandr2 libxfixes3 \
    patchelf ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# --- Python deps (proven combo on py3.12) -----------------------------------
# lerobot 0.5.1 pulls a GPU-enabled torch; the local lerobot 0.5.2 source is
# mounted on PYTHONPATH at runtime so the exact trained-against code wins.
RUN pip install --no-cache-dir \
    "lerobot==0.5.1" \
    "robosuite==1.4.1" \
    bddl easydict transforms3d "gym==0.25.2" \
    hydra-core matplotlib imageio imageio-ffmpeg h5py \
    && rm -rf /root/.cache/pip

# Sources (lerobot, LIBERO) are mounted at runtime; make them importable.
ENV PYTHONPATH=/workspace/lerobot/src:/workspace/LIBERO
# Default to on-screen GLFW rendering; override to egl/osmesa for headless.
ENV MUJOCO_GL=glfw

WORKDIR /workspace

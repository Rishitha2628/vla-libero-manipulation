# VLA Manipulation — ACT on LIBERO

Fine-tuned ACT (Action Chunking Transformer) policy on the LIBERO object manipulation benchmark.

## Task
Pick up the alphabet soup and place it in the basket.

## Results
- Training steps: 5000
- Final loss: 0.31 (down from 6.19)
- Dataset: 50 expert demonstrations

## Stack
- Policy: ACT (LeRobot)
- Benchmark: LIBERO-Object
- Simulator: MuJoCo (via LIBERO)
- Framework: HuggingFace LeRobot

## Files
- `train_act.py` — training script
- `convert_libero_to_lerobot.py` — dataset conversion from LIBERO HDF5 to LeRobot format
- `model_final/` — trained model checkpoint
- `Dockerfile.lerobot` — reproducible environment

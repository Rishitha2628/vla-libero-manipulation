"""Train an ACT policy on the LIBERO alphabet-soup demonstrations.

Run inside the vla-libero container (see run_train.sh), which mounts the local
LIBERO and lerobot checkouts and this project at /workspace/proj.

Two things here differ from the original May training run, and both were bugs:

1. `delta_timestamps` makes each sample's `action` the *true* next `chunk_size`
   trajectory. The old script instead did `action.unsqueeze(1).expand(-1, 50, -1)`,
   which repeats the current action 50 times -- so ACT never learned to plan and
   was effectively a single-step reactive policy.
2. The batch is pushed through the policy's preprocessor, which mean/std
   normalizes images, state and action. Normalization lives in the processor
   pipeline in current lerobot, not inside ACTPolicy, so the old
   `ACTPolicy(config, dataset_stats=...)` call silently dropped it into **kwargs.
   The pre/post processors are saved next to the weights so inference can undo
   the normalization -- without them a checkpoint is unusable.
"""
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.processor_act import make_act_pre_post_processors
from lerobot.configs.policies import PolicyFeature, FeatureType
from lerobot.configs.types import NormalizationMode

REPO_ID = "local/libero_alphabet_soup"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=Path, default=Path("/workspace/data/alphabet_soup"))
    p.add_argument("--output-dir", type=Path, default=Path("/workspace/proj/outputs/act_chunked"))
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--chunk-size", type=int, default=50)
    p.add_argument("--save-every", type=int, default=2500)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = ACTConfig(
        input_features={
            "observation.images.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 128, 128)),
            "observation.images.image2": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 128, 128)),
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(8,)),
        },
        output_features={
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
        },
        chunk_size=args.chunk_size,
        n_action_steps=args.chunk_size,
        normalization_mapping={
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        },
        device=args.device,
        # Record the LR we actually train with; the old config.json said 1e-5
        # while the script used 1e-4.
        optimizer_lr=args.lr,
        optimizer_lr_backbone=args.lr,
    )

    # Ask the dataset for the real next-chunk_size actions. config.action_delta_indices
    # is range(chunk_size); dividing by fps turns those frame offsets into the
    # timestamps LeRobotDataset expects.
    meta = LeRobotDatasetMetadata(REPO_ID, root=args.dataset_root)
    delta_timestamps = {"action": [i / meta.fps for i in config.action_delta_indices]}
    print(f"fps={meta.fps} frames={meta.total_frames} episodes={meta.total_episodes}")
    print(f"action delta_timestamps: {len(delta_timestamps['action'])} steps "
          f"({delta_timestamps['action'][0]:.3f}s .. {delta_timestamps['action'][-1]:.3f}s)")

    dataset = LeRobotDataset(REPO_ID, root=args.dataset_root, delta_timestamps=delta_timestamps)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    policy = ACTPolicy(config).to(args.device)
    policy.train()

    # Normalization now lives here, not in the policy.
    preprocessor, postprocessor = make_act_pre_post_processors(
        config, dataset_stats=dataset.meta.stats
    )

    optimizer = torch.optim.AdamW(
        policy.get_optim_params(), lr=args.lr, weight_decay=config.optimizer_weight_decay
    )

    def save(tag):
        d = args.output_dir / tag
        policy.save_pretrained(d)
        preprocessor.save_pretrained(d)
        postprocessor.save_pretrained(d)
        print(f"saved {d}")

    print(f"training {args.steps} steps, batch {args.batch_size}, lr {args.lr}")
    log = open(args.output_dir / "train_log.csv", "w")
    log.write("step,loss,l1_loss,kld_loss\n")

    step = 0
    data_iter = iter(dataloader)
    running = []
    while step < args.steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        # Normalizes and moves to device. `action` arrives as (B, chunk_size, 7)
        # and `action_is_pad` as (B, chunk_size), both straight from the dataset.
        batch = preprocessor(batch)

        loss, info = policy.forward(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running.append(loss.item())
        step += 1

        if step % 100 == 0:
            avg = sum(running[-100:]) / len(running[-100:])
            print(f"Step {step}/{args.steps} | Loss: {avg:.4f} | "
                  f"l1: {info.get('l1_loss', float('nan')):.4f} | "
                  f"kld: {info.get('kld_loss', float('nan')):.4f}", flush=True)
            log.write(f"{step},{avg:.6f},{info.get('l1_loss', '')},{info.get('kld_loss', '')}\n")
            log.flush()

        if step % args.save_every == 0:
            save(f"checkpoint_{step}")

    save("final")
    log.close()
    print("Training complete!")


if __name__ == "__main__":
    main()

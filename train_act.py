import sys
sys.path.insert(0, '/workspace/LIBERO')
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.configs.policies import PolicyFeature, FeatureType
from lerobot.configs.types import NormalizationMode

DATASET_ROOT = Path('/workspace/lerobot_dataset')
OUTPUT_DIR = Path('/workspace/outputs/act_finetune')
BATCH_SIZE = 8
STEPS = 5000
LR = 1e-4
SAVE_EVERY = 1000
DEVICE = 'cuda'

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading dataset...")
dataset = LeRobotDataset('local/libero_alphabet_soup', root=DATASET_ROOT)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)

print("Building ACT policy...")
config = ACTConfig(
    input_features={
        "observation.images.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
        "observation.images.image2": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(8,)),
    },
    output_features={
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
    },
    chunk_size=50,
    n_action_steps=50,
    normalization_mapping={
        "VISUAL": NormalizationMode.MEAN_STD,
        "STATE": NormalizationMode.MEAN_STD,
        "ACTION": NormalizationMode.MEAN_STD,
    },
)
policy = ACTPolicy(config, dataset_stats=dataset.meta.stats)
policy = policy.to(DEVICE)
policy.train()

optimizer = torch.optim.AdamW(policy.parameters(), lr=LR)

print("Starting training...")
step = 0
data_iter = iter(dataloader)
losses = []

while step < STEPS:
    try:
        batch = next(data_iter)
    except StopIteration:
        data_iter = iter(dataloader)
        batch = next(data_iter)

    batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    optimizer.zero_grad()
    batch["action"] = batch["action"].unsqueeze(1).expand(-1, 50, -1) if batch["action"].ndim == 2 else batch["action"]
    batch["action_is_pad"] = torch.zeros(batch["action"].shape[:2], dtype=torch.bool, device=DEVICE)
    loss, info = policy.forward(batch)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    step += 1

    if step % 100 == 0:
        avg_loss = sum(losses[-100:]) / len(losses[-100:])
        print(f"Step {step}/{STEPS} | Loss: {avg_loss:.4f}")

    if step % SAVE_EVERY == 0:
        policy.save_pretrained(OUTPUT_DIR / f'checkpoint_{step}')
        print(f"Saved checkpoint at step {step}")

policy.save_pretrained(OUTPUT_DIR / 'final')
print("Training complete!")

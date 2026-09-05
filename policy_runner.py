"""Shared policy loading + observation building for LIBERO rollouts.

Used by both eval_policy.py (batch success-rate eval) and visualize.py (live window).

The important part is `load_policy`: in current lerobot, ACTPolicy does no
normalization itself -- a preprocessor mean/std normalizes the inputs and a
postprocessor un-normalizes the predicted actions back into the robot's units.
Running a checkpoint without them feeds the net unnormalized inputs and sends
normalized outputs straight to the robot, which is silently, catastrophically
wrong rather than an error.
"""
import os

import numpy as np
import torch

POLICY_PREPROCESSOR = "policy_preprocessor"
POLICY_POSTPROCESSOR = "policy_postprocessor"


def load_policy(model_dir, dataset_root=None, device="cuda", n_action_steps=None,
                temporal_ensemble_coeff=None):
    """Load an ACT checkpoint together with its normalization pipelines.

    Checkpoints saved by the current train_act.py carry their own processor
    configs. Older ones (model_final/) do not, so the stats are rebuilt from the
    dataset -- they are a deterministic function of the same 50 demos.
    """
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.act.processor_act import make_act_pre_post_processors
    from lerobot.processor import PolicyProcessorPipeline
    from lerobot.processor.converters import (
        batch_to_transition, transition_to_batch,
        policy_action_to_transition, transition_to_policy_action,
    )

    model_dir = str(model_dir)
    policy = ACTPolicy.from_pretrained(model_dir)
    config = policy.config
    config.device = device

    # Executing fewer than chunk_size actions per inference makes the rollout
    # more closed-loop; temporal ensembling smooths across overlapping chunks.
    if temporal_ensemble_coeff is not None:
        from lerobot.policies.act.modeling_act import ACTTemporalEnsembler
        config.temporal_ensemble_coeff = temporal_ensemble_coeff
        config.n_action_steps = 1
        # ACTPolicy only builds the ensembler in __init__, and the checkpoint was
        # saved without it, so attach one now.
        policy.temporal_ensembler = ACTTemporalEnsembler(temporal_ensemble_coeff,
                                                         config.chunk_size)
        policy.reset()
    elif n_action_steps is not None:
        config.n_action_steps = n_action_steps

    policy.to(device)
    policy.eval()

    has_saved = os.path.exists(os.path.join(model_dir, f"{POLICY_PREPROCESSOR}.json"))
    if has_saved:
        print(f"[policy] loading saved processors from {model_dir}")
        pre = PolicyProcessorPipeline.from_pretrained(
            pretrained_model_name_or_path=model_dir,
            config_filename=f"{POLICY_PREPROCESSOR}.json",
            to_transition=batch_to_transition,
            to_output=transition_to_batch,
        )
        post = PolicyProcessorPipeline.from_pretrained(
            pretrained_model_name_or_path=model_dir,
            config_filename=f"{POLICY_POSTPROCESSOR}.json",
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        )
    else:
        if dataset_root is None:
            raise SystemExit(
                f"{model_dir} has no saved processors and no --dataset-root was given; "
                "cannot reconstruct normalization stats."
            )
        print(f"[policy] no saved processors in {model_dir}; rebuilding stats from {dataset_root}")
        from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
        meta = LeRobotDatasetMetadata("local/libero_alphabet_soup", root=dataset_root)
        pre, post = make_act_pre_post_processors(config, dataset_stats=meta.stats)

    # The device the pipelines were saved with may not be this machine's.
    for pipeline in (pre, post):
        for step in getattr(pipeline, "steps", []):
            if hasattr(step, "device") and getattr(step, "device", None) not in (None, "cpu"):
                step.device = device
    return policy, pre, post


def build_obs(obs, flip_images=False):
    """Rebuild the training-time observation from a robosuite obs dict.

    Training data came from LIBERO's hdf5 `agentview_rgb` / `eye_in_hand_rgb`
    resized to 256, with the state as [ee_pos(3), ee_ori axis-angle(3),
    gripper_qpos(2)]. `flip_images` exists because robosuite's on-line camera
    images and LIBERO's recorded ones can differ by a vertical flip; it is the
    first preprocessing mismatch to test if success rates look wrong.
    """
    import robosuite.utils.transform_utils as T

    def to_chw(img):
        if flip_images:
            img = img[::-1]
        x = torch.from_numpy(np.ascontiguousarray(img)).float().permute(2, 0, 1) / 255.0
        return x

    state = np.concatenate([
        obs["robot0_eef_pos"],
        T.quat2axisangle(obs["robot0_eef_quat"]),
        obs["robot0_gripper_qpos"],
    ]).astype(np.float32)

    return {
        "observation.images.image": to_chw(obs["agentview_image"]),
        "observation.images.image2": to_chw(obs["robot0_eye_in_hand_image"]),
        "observation.state": torch.from_numpy(state),
    }


def predict_action(policy, pre, post, obs, flip_images=False):
    """One observation in, one robot-space action out."""
    batch = pre(build_obs(obs, flip_images))
    with torch.inference_mode():
        action = policy.select_action(batch)
    action = post(action)
    return action.squeeze(0).cpu().numpy().astype(np.float32)

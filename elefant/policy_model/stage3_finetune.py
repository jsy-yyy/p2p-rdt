import logging
import json

import lightning as pl
import torch
import os
import wandb
import fsspec
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from elefant.data import (
    ActionLabelVideoProtoDataset,
    ActionLabelVideoProtoDatasetConfig,
    UniversalAutoregressiveActionMapping,
    DummyDataset,
    DummyDatasetConfig,
    StructuredAction,
)
from elefant.text_tokenizer.config import TextTokenizerConfig
from elefant.data.rand_augment import BatchRandAugment
from elefant.data.lerobot_latent_dataset import (
    MultiLatentLeRobotDataset,
    infer_text_embedding_shape_from_dataset,
)
from elefant.policy_model.config import LightningPolicyConfig
from elefant.policy_model.model_free import ModelFreePolicy
from lightning.pytorch.callbacks import ModelCheckpoint
from elefant.lightning import AsyncCheckpointIO
from lightning.pytorch.utilities import grad_norm
from typing import List, Optional
from elefant.torch import ELEFANT_WANDB_DIR
from elefant.policy_model.kv_cache import KVCacheState
import pydantic_yaml
from elefant.torch import (
    eager_assert,
    cross_entropy_to_perplexity,
    _sample_from_logits_gpu,
)
from elefant.torch import count_model_parameters
from elefant.data.action_mapping import UniversalAutoregressiveActionMapping
from elefant.metrics import LossMetric
from elefant.policy_model.config import DatasetConfig, ValidationDatasetConfig
from rdt.model import RDT
from lightning.fabric.utilities.cloud_io import get_filesystem


def _metric_to_float(metric_value: torch.Tensor | float) -> float:
    if isinstance(metric_value, torch.Tensor):
        return float(metric_value.detach().float().cpu().item())
    return float(metric_value)


def _validate_resume_checkpoint_compatibility(
    checkpoint_path: str, config: LightningPolicyConfig
) -> None:
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint.get('state_dict', {})
    if not state_dict:
        logging.warning('Checkpoint %s has no state_dict; skipping action-dimension compatibility check.', checkpoint_path)
        return

    expected_n_actions = config.stage3_finetune.training_dataset.get_robot_action_dim()
    action_pos_tokens = state_dict.get('bc_transformer.action_pos_tokens')
    if action_pos_tokens is not None:
        checkpoint_n_actions = int(action_pos_tokens.shape[1])
        if checkpoint_n_actions != expected_n_actions:
            raise ValueError(
                'Resume checkpoint action dimension mismatch: '
                f'checkpoint has {checkpoint_n_actions} action tokens but config expects {expected_n_actions}. '
                'This usually means you are trying to resume a checkpoint trained with a different robot_action_dim.'
            )

    x_pos_emb = state_dict.get('rdt_policy_head.x_pos_emb')
    if x_pos_emb is not None:
        expected_x_pos_tokens = expected_n_actions + config.policy_model.rdt.num_register_tokens
        checkpoint_x_pos_tokens = int(x_pos_emb.shape[1])
        if checkpoint_x_pos_tokens != expected_x_pos_tokens:
            raise ValueError(
                'Resume checkpoint RDT position embedding mismatch: '
                f'checkpoint has {checkpoint_x_pos_tokens} x_pos tokens but config expects {expected_x_pos_tokens}. '
                'This usually means you are trying to resume a checkpoint trained with a different robot_action_dim.'
            )


def _resolve_resume_checkpoint_path(config: LightningPolicyConfig) -> Optional[str]:
    checkpoint_path = config.stage3_finetune.init.stage3_model_path
    if not checkpoint_path:
        return None

    filesystem = get_filesystem(checkpoint_path)
    if filesystem.exists(checkpoint_path) and not filesystem.isdir(checkpoint_path):
        return checkpoint_path

    if filesystem.isdir(checkpoint_path):
        checkpoint_candidates = []
        for entry in filesystem.ls(checkpoint_path, detail=True):
            candidate = entry["name"] if isinstance(entry, dict) else entry
            candidate = str(candidate)
            if candidate.endswith(".ckpt"):
                checkpoint_candidates.append(candidate)

        if not checkpoint_candidates:
            raise FileNotFoundError(
                f"No .ckpt files found in checkpoint directory: {checkpoint_path}"
            )

        return sorted(checkpoint_candidates)[-1]

    raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint_path}")


def upload_model_config(checkpoint_path: str, config):
    """Save the model config to the checkpoint path."""
    # Save the model config to the checkpoint path.
    logging.info(f"Uploading model config to {checkpoint_path}/model_config.yaml")
    with fsspec.open(checkpoint_path + "/model_config.yaml", "wb") as f:
        f.write(pydantic_yaml.to_yaml_str(config).encode())


def upload_action_mapping(checkpoint_path: str, action_mapping):
    """Upload the action specification to the checkpoint path."""
    logging.info(f"Uploading action mapping to {checkpoint_path}/action_mapping.json")
    with fsspec.open(checkpoint_path + "/action_mapping.json", "w") as f:
        if hasattr(action_mapping, "serialize"):
            f.write(action_mapping.serialize())
        else:
            json.dump(action_mapping, f)


def _sample_from_distribution(
    logits: torch.Tensor,
    unif_rand: torch.Tensor,
) -> torch.Tensor:
    eager_assert(unif_rand.ndim, 0)
    probs = torch.softmax(logits, dim=-1)
    cdf = torch.cumsum(probs, dim=-1)
    cmp = cdf >= unif_rand.unsqueeze(-1)
    return cmp.float().argmax(dim=-1)


def _sync_text_embedding_shape_with_dataset(config: LightningPolicyConfig):
    inferred = infer_text_embedding_shape_from_dataset(
        config.stage3_finetune.training_dataset
    )
    if inferred is None:
        return

    inferred_shape, sample_path = inferred
    current_shape = list(config.shared.text_tokenizer_config.text_embedding_shape)
    if current_shape != inferred_shape:
        logging.warning(
            'Overriding text_embedding_shape from %s to %s based on %s.',
            current_shape,
            inferred_shape,
            sample_path,
        )
    else:
        logging.info(
            'Using text_embedding_shape %s inferred from %s.',
            inferred_shape,
            sample_path,
        )

    config.shared.text_tokenizer_config.text_embedding_shape = inferred_shape
    config.stage3_finetune.training_dataset.text_embedding_shape = inferred_shape
    for val_dataset in config.stage3_finetune.validation_datasets:
        val_dataset.text_embedding_shape = inferred_shape


def _sync_sequence_length_with_dataset(config: LightningPolicyConfig):
    target_seq_len = config.shared.n_seq_timesteps
    training_dataset = config.stage3_finetune.training_dataset
    if training_dataset.n_seq_timesteps != target_seq_len:
        logging.warning(
            'Overriding training dataset n_seq_timesteps from %s to %s to match shared.n_seq_timesteps.',
            training_dataset.n_seq_timesteps,
            target_seq_len,
        )
        training_dataset.n_seq_timesteps = target_seq_len

    for val_dataset in config.stage3_finetune.validation_datasets:
        if val_dataset.n_seq_timesteps != target_seq_len:
            logging.warning(
                'Overriding validation dataset `%s` n_seq_timesteps from %s to %s to match shared.n_seq_timesteps.',
                val_dataset.validation_name,
                val_dataset.n_seq_timesteps,
                target_seq_len,
            )
            val_dataset.n_seq_timesteps = target_seq_len


class PolicyModelTrainer(ModelFreePolicy):
    def __init__(
        self,
        config: LightningPolicyConfig,
        stage_name: str,
        inference_mode: bool = False,
    ):
        super().__init__(
            config=config, stage_name=stage_name, inference_mode=inference_mode
        )

        self._init_action_mapping()
        self._already_frozen = False
        self._already_unfrozen = False
        self.generate_dummy_text_embed = False
        self.lb_loss_weight = (
            self.config.policy_model.sparse_moe.lb_loss_weight
            if self.config.policy_model.model_type == "sparse_moe"
            else 0
        )
        self.z_loss_weight = self.config.policy_model.z_loss_weight
        if self.config.policy_model.model_type == "sparse_moe":
            self.compile_mode = torch.compile(fullgraph=True)
        else:
            # default compilation mode is without max autotune which gets
            # edited when initializing stage 1/3 with max-autotunes
            self.compile_mode = torch.compile()
        self.rz_loss_weight = (
            self.config.policy_model.sparse_moe.rz_loss_weight
            if self.config.policy_model.model_type == "sparse_moe"
            else 0
        )
        self.num_of_experts = (
            self.config.policy_model.sparse_moe.num_experts
            if self.config.policy_model.model_type == "sparse_moe"
            else 1
        )

        self._init_metrics()
        self.top_p = config.policy_model.top_p

    def setup(self, stage):
        self._init_rand_augment()

    def _init_rand_augment(self):
        ra_cfg = self.config.stage3_finetune.training_dataset.rand_augmentation
        frac = ra_cfg.fraction_augmented
        auglist = ra_cfg.augmentations
        assert frac == 0.0 or (auglist and len(auglist) > 0), (
            "When frac > 0, auglist must be provided and non-empty"
        )
        if frac > 0.0:
            self.rand_augment = BatchRandAugment(augmentations=auglist)
            self.augment_fraction = frac
        else:
            self.rand_augment = None
            self.augment_fraction = 0.0

    def _keyboard_mouse_action_sampler(
        self,
        action_token: torch.Tensor,
        action_idx: int,
        sampled_actions: StructuredAction,
        sampling_temperature: float = 1.0,
        unif_rand: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raise RuntimeError(
            "The keyboard/mouse autoregressive sampler is not available in the LeRobot + RDT path. Use online_full_predict() with continuous RobotAction inputs instead."
        )

    def on_validation_epoch_start(self):
        if not self._validation_metrics and self.trainer.val_dataloaders:
            val_set_names = list(self.trainer.val_dataloaders.keys())
            for val_set_name in val_set_names:
                metrics = {
                    "loss": LossMetric().to(self.device),
                    "robot_action_mse": LossMetric().to(self.device),
                    "lb_loss": LossMetric().to(self.device),
                    "rz_loss": LossMetric().to(self.device),
                }
                for i in range(self.num_of_experts):
                    metrics[f"expert_{i}_capacity"] = LossMetric().to(self.device)
                self._validation_metrics[val_set_name] = metrics

    def _init_metrics(self):
        self._training_loss_metric = LossMetric()
        self._training_ratio_unlabeled_metric = LossMetric()
        self._training_robot_action_mse_metric = LossMetric()
        self._training_lb_loss_metric = LossMetric()
        self._training_rz_loss_metric = LossMetric()

        for i in range(self.num_of_experts):
            setattr(self, f"_training_expert_{i}_capacity_metric", LossMetric())

        self._validation_metrics = {}

    def _init_action_mapping(self):
        self.n_actions = self.config.stage3_finetune.training_dataset.get_robot_action_dim()
        self.embedding_std = 0.1
        action_embed_dim = self.config.policy_model.action_decoder.embed_dim
        self.rdt_hidden_size = self.config.policy_model.rdt.hidden_size

        def _init_linear(layer: nn.Linear):
            torch.nn.init.normal_(layer.weight, mean=0.0, std=self.embedding_std)
            if layer.bias is not None:
                torch.nn.init.zeros_(layer.bias)

        self.robot_action_in_proj = nn.Linear(
            1,
            action_embed_dim,
            dtype=torch.bfloat16,
        )
        _init_linear(self.robot_action_in_proj)

        if action_embed_dim == self.rdt_hidden_size:
            self.policy_action_to_rdt_proj = nn.Identity()
        else:
            self.policy_action_to_rdt_proj = nn.Linear(
                action_embed_dim,
                self.rdt_hidden_size,
                dtype=torch.bfloat16,
            )
            _init_linear(self.policy_action_to_rdt_proj)

        self.policy_summary_to_rdt_state_proj = nn.Linear(
            self.config.policy_model.transformer_dim,
            self.rdt_hidden_size,
            dtype=torch.bfloat16,
        )
        _init_linear(self.policy_summary_to_rdt_state_proj)

        self.text_condition_to_rdt_proj = nn.Linear(
            self._get_text_embedding_dim(),
            self.rdt_hidden_size,
            dtype=torch.bfloat16,
        )
        _init_linear(self.text_condition_to_rdt_proj)

        rdt_config = self.config.policy_model.rdt
        self.rdt_policy_head = RDT(
            horizon=self.n_actions,
            output_size=1,
            config={
                "hidden_size": self.rdt_hidden_size,
                "num_heads": rdt_config.num_heads,
                "num_kv_heads": rdt_config.num_kv_heads,
                "depth": rdt_config.depth,
                "norm_eps": rdt_config.norm_eps,
                "multiple_of": rdt_config.multiple_of,
                "ffn_dim_multiplier": rdt_config.ffn_dim_multiplier,
                "use_flash_attn": rdt_config.use_flash_attn,
                "num_register_tokens": rdt_config.num_register_tokens,
                "action_dim": 1,
            },
            x_pos_emb_config=[
                ("action", self.n_actions),
                ("register", rdt_config.num_register_tokens),
            ],
            lang_pos_emb_config=[],
            max_lang_len=0,
            img_pos_emb_config=None,
            max_img_len=0,
            act_pos_emb_config=[("text", self.text_token_size)],
            max_act_len=self.text_token_size,
            dtype=torch.bfloat16,
        )

    def configure_model(self):
        pass

    def online_kv_cache_predict(
        self,
        frame: torch.Tensor,
        idx: torch.Tensor,
        kv_cache_state: List[KVCacheState],
        unif_rand: Optional[torch.Tensor] = None,
        compile: bool = True,
        sampling_temperature: float = 1.0,
        text_tokens_embed: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, List[KVCacheState]]:
        raise NotImplementedError(
            "KV-cache autoregressive inference still depends on the old discrete action decoder and is not supported in the LeRobot + RDT path. Use online_full_predict() for continuous RobotAction prediction."
        )

    def _prepare_text_tokens_embed_for_inference(
        self,
        batch_size: int,
        n_steps: int,
        device: torch.device,
        text_tokens_embed: Optional[torch.Tensor],
    ) -> torch.Tensor:
        text_dim = self._get_text_embedding_dim()
        if text_tokens_embed is None:
            return torch.zeros(
                batch_size,
                n_steps,
                self.text_token_size,
                text_dim,
                device=device,
                dtype=torch.float32,
            )

        text_tokens_embed = torch.as_tensor(text_tokens_embed, device=device)
        if text_tokens_embed.ndim == 2:
            text_tokens_embed = text_tokens_embed.unsqueeze(0).unsqueeze(0)
        elif text_tokens_embed.ndim == 3:
            if text_tokens_embed.shape[0] == n_steps:
                text_tokens_embed = text_tokens_embed.unsqueeze(0)
            elif text_tokens_embed.shape[0] == batch_size:
                text_tokens_embed = text_tokens_embed.unsqueeze(1)
            else:
                raise ValueError(
                    f"Unsupported 3D text embedding shape {tuple(text_tokens_embed.shape)} for batch_size={batch_size}, n_steps={n_steps}."
                )
        elif text_tokens_embed.ndim != 4:
            raise ValueError(
                f"Expected text embeddings with 2, 3, or 4 dims, got {text_tokens_embed.ndim}."
            )

        if text_tokens_embed.shape[0] == 1 and batch_size != 1:
            text_tokens_embed = text_tokens_embed.expand(batch_size, -1, -1, -1)
        if text_tokens_embed.shape[1] == 1 and n_steps != 1:
            text_tokens_embed = text_tokens_embed.expand(-1, n_steps, -1, -1)

        eager_assert(
            text_tokens_embed.shape,
            (batch_size, n_steps, self.text_token_size, text_dim),
        )
        return text_tokens_embed.float()

    def online_full_predict_actions(
        self,
        frames: torch.Tensor,
        actions: torch.Tensor,
        text_tokens_embed: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        frames = self._normalize_frames(frames)
        B, T = frames.shape[0], frames.shape[1]
        text_tokens_embed = self._prepare_text_tokens_embed_for_inference(
            batch_size=B,
            n_steps=T,
            device=frames.device,
            text_tokens_embed=text_tokens_embed,
        )
        action_embeddings_in = self.action_in_to_tokens(actions)
        action_out_embeddings, action_out_tokens, *_ = self.transformer_forward_function(
            frames, action_embeddings_in, text_tokens_embed
        )
        action_preds = self.action_tokens_to_actions(
            action_out_embeddings,
            action_out_tokens,
            text_tokens_embed,
        )
        eager_assert(action_preds.shape, (B, T, self.n_actions))
        return action_preds

    def online_full_predict_logits(
        self,
        frames: torch.Tensor,
        actions: torch.Tensor,
        text_tokens_embed: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.online_full_predict_actions(
            frames=frames,
            actions=actions,
            text_tokens_embed=text_tokens_embed,
        )

    def online_full_predict(
        self,
        frames: torch.Tensor,
        actions: torch.Tensor,
        kv_cache_state: List[KVCacheState] = None,
        sampling_temperature: float = 1.0,
        text_tokens_embed: Optional[torch.Tensor] = None,
        compile: bool = True,
    ) -> torch.Tensor:
        del kv_cache_state, sampling_temperature

        @(torch.compile(fullgraph=True) if compile else lambda f: f)
        def _predict(frames, actions, text_tokens_embed):
            return self.online_full_predict_actions(
                frames=frames,
                actions=actions,
                text_tokens_embed=text_tokens_embed,
            )

        with torch.inference_mode():
            return _predict(frames, actions, text_tokens_embed)

    def action_in_to_tokens(
        self, action_in: torch.Tensor, idx: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Project continuous robot actions into action tokens."""
        B, T, D = action_in.shape
        eager_assert(action_in.shape, (B, T, self.n_actions))

        if idx is None:
            action_subset = action_in
        else:
            action_subset = action_in[idx]

        action_embedding = self.robot_action_in_proj(
            action_subset.unsqueeze(-1).to(torch.bfloat16)
        )
        eager_assert(
            action_embedding.shape,
            (
                B if idx is None else len(idx),
                T,
                self.n_actions,
                self.config.policy_model.action_decoder.embed_dim,
            ),
        )
        return action_embedding

    def action_tokens_to_actions(
        self,
        action_out_embeddings: torch.Tensor,
        action_out_tokens: torch.Tensor,
        text_tokens_embed: torch.Tensor,
    ) -> torch.Tensor:
        B, T, N, _ = action_out_embeddings.shape
        eager_assert(N, self.n_actions)
        eager_assert(
            action_out_tokens.shape,
            (B, T, self.config.policy_model.transformer_dim),
        )
        eager_assert(
            text_tokens_embed.shape,
            (
                B,
                T,
                self.text_token_size,
                self._get_text_embedding_dim(),
            ),
        )

        action_tokens_for_rdt = self.policy_action_to_rdt_proj(
            action_out_embeddings.to(torch.bfloat16)
        )
        eager_assert(
            action_tokens_for_rdt.shape,
            (B, T, self.n_actions, self.rdt_hidden_size),
        )

        state_tokens = self.policy_summary_to_rdt_state_proj(
            action_out_tokens.to(torch.bfloat16)
        ).reshape(B * T, 1, self.rdt_hidden_size)
        eager_assert(state_tokens.shape, (B * T, 1, self.rdt_hidden_size))

        text_condition = self.text_condition_to_rdt_proj(
            text_tokens_embed.to(torch.bfloat16)
        )
        eager_assert(
            text_condition.shape,
            (B, T, self.text_token_size, self.rdt_hidden_size),
        )
        text_condition = text_condition.reshape(
            B * T, self.text_token_size, self.rdt_hidden_size
        )
        text_condition_mask = text_tokens_embed.abs().sum(dim=-1).reshape(
            B * T, self.text_token_size
        ) > 0

        action_preds = self.rdt_policy_head(
            x=action_tokens_for_rdt.reshape(
                B * T, self.n_actions, self.rdt_hidden_size
            ),
            t=torch.zeros(B * T, device=action_out_embeddings.device, dtype=torch.long),
            act_c=text_condition,
            state_c=state_tokens,
            act_mask=text_condition_mask,
            decode_output=True,
        ).squeeze(-1)
        eager_assert(action_preds.shape, (B * T, self.n_actions))
        action_preds = action_preds.reshape(B, T, self.n_actions)
        eager_assert(action_preds.shape, (B, T, self.n_actions))
        return action_preds

    def action_out_tokens_to_logits(
        self, action_out_tokens: torch.Tensor
    ) -> torch.Tensor:
        raise RuntimeError(
            "Discrete action logits are not available in the LeRobot + RDT training path. Use action_tokens_to_actions() instead."
        )

    def on_before_optimizer_step(self, optimizer):
        # inspect (unscaled) gradients here
        if self.global_step % 100 == 0:
            self.log_dict(grad_norm(self, norm_type=2))

    def init_from_stage2_model(self, stage2_model):
        super().copy_weights(stage2_model)

    def _calculate_z_loss(self, action_logits, masked_labels):
        key_z_loss = (
            (action_logits.keys.view(-1, self.n_keyboard_choices).logsumexp(-1).pow(2))
            * (masked_labels.keys.view(-1) != -100)
        ).mean()
        mouse_button_z_loss = (
            (
                action_logits.mouse_buttons.view(-1, self.n_mouse_button_choices)
                .logsumexp(-1)
                .pow(2)
            )
            * (masked_labels.mouse_buttons.view(-1) != -100)
        ).mean()
        mouse_delta_x_z_loss = (
            (
                action_logits.mouse_delta_x.view(-1, self.n_mouse_x_bins)
                .logsumexp(-1)
                .pow(2)
            )
            * (masked_labels.mouse_delta_x.view(-1) != -100)
        ).mean()
        mouse_delta_y_z_loss = (
            (
                action_logits.mouse_delta_y.view(-1, self.n_mouse_y_bins)
                .logsumexp(-1)
                .pow(2)
            )
            * (masked_labels.mouse_delta_y.view(-1) != -100)
        ).mean()
        return (
            key_z_loss,
            mouse_button_z_loss,
            mouse_delta_x_z_loss,
            mouse_delta_y_z_loss,
        )

    def _calculate_loss(self, batch, actions_in, masked_labels, text_tokens_embed):
        """Calculate the regression loss for continuous robot actions."""
        frames = self._normalize_frames(batch.frames)
        batch_size = batch.frames.shape[0]
        T = batch.frames.shape[1]
        action_embeddings_in = self.action_in_to_tokens(actions_in)
        eager_assert(
            action_embeddings_in.shape,
            (
                batch_size,
                T,
                self.n_actions,
                self.config.policy_model.action_decoder.embed_dim,
            ),
        )
        action_out_embeddings, action_out_tokens, auxiliary_losses, auxiliary_outputs = (
            self.transformer_forward_function(
                frames, action_embeddings_in, text_tokens_embed
            )
        )
        eager_assert(
            action_out_embeddings.shape,
            (
                batch_size,
                T,
                self.n_actions,
                self.config.policy_model.action_decoder.embed_dim,
            ),
        )

        action_preds = self.action_tokens_to_actions(
            action_out_embeddings,
            action_out_tokens,
            text_tokens_embed,
        )
        action_targets = masked_labels["targets"]
        action_mask = masked_labels["mask"].to(action_preds.dtype)
        squared_error = (action_preds - action_targets).pow(2)
        denom = action_mask.sum().clamp_min(1.0)
        robot_action_mse = (squared_error * action_mask).sum() / denom

        lb_loss = auxiliary_losses.get(
            "lb_loss", torch.tensor(0.0, device=frames.device)
        )
        rz_loss = auxiliary_losses.get(
            "rz_loss", torch.tensor(0.0, device=frames.device)
        )
        losses = {
            "robot_action_mse": robot_action_mse,
            "lb_loss": lb_loss,
            "rz_loss": rz_loss,
        }
        loss = (
            robot_action_mse
            + lb_loss * self.lb_loss_weight
            + rz_loss * self.rz_loss_weight
        )
        return loss, robot_action_mse, losses, auxiliary_outputs

    def _create_target_and_masked_labels(self, batch):
        batch_size = batch.frames.shape[0]
        T = batch.frames.shape[1]
        user_action_mask = batch.user_action_mask
        system_action_mask = batch.system_action_mask
        valid_frame_mask = batch.valid_frame_mask
        eager_assert(user_action_mask.shape, (batch_size, T))
        eager_assert(valid_frame_mask.shape, (batch_size, T))
        eager_assert(system_action_mask.shape, (batch_size, T))

        effective_mask = self._compute_effective_mask(
            user_action_mask, valid_frame_mask, system_action_mask
        )
        actions_in = batch.action_annotations.float()
        action_mask = effective_mask.unsqueeze(-1).expand_as(actions_in)
        masked_labels = {
            "targets": batch.action_annotations.float(),
            "mask": action_mask,
        }
        ratio_unlabeled = torch.zeros(
            (), dtype=torch.float32, device=user_action_mask.device
        )
        return actions_in, masked_labels, ratio_unlabeled

    def _apply_augmentations(self, batch):
        """Apply random augmentations to frames on GPU"""
        if self.rand_augment is None or self.augment_fraction == 0.0:
            return batch

        should_augment = torch.rand(1) < self.augment_fraction

        def _augment_fn(frames):
            frames = self.rand_augment(frames)
            return frames

        def _no_augment_fn(frames):
            return frames

        frames = batch.frames
        if should_augment:
            frames = _augment_fn(frames)

        # TODO: would be nice to have this compiled and use torch.cond
        # frames = torch.cond(should_augment, _augment_fn, _no_augment_fn, (frames,))
        return batch._replace(frames=frames)

    def _validate_batch_sequence_length(self, batch):
        expected_seq_len = self.config.shared.n_seq_timesteps
        lengths = {
            "frames": batch.frames.shape[1],
            "action_annotations": batch.action_annotations.shape[1],
            "user_action_mask": batch.user_action_mask.shape[1],
            "system_action_mask": batch.system_action_mask.shape[1],
            "text_embeddings": batch.text_embeddings.shape[1],
        }
        if batch.valid_frame_mask is not None:
            lengths["valid_frame_mask"] = batch.valid_frame_mask.shape[1]

        mismatched = {
            name: length
            for name, length in lengths.items()
            if length != expected_seq_len
        }
        if mismatched:
            raise ValueError(
                f"Batch sequence length must match shared.n_seq_timesteps={expected_seq_len}, got {mismatched}."
            )

    def training_step(self, batch, batch_idx):
        if self.trainer.global_step == 0:
            logging.info(
                f"First training step starting (compilation may take awhile). rank={self.trainer.global_rank}"
            )

        batch = self._apply_augmentations(batch)
        self._validate_batch_sequence_length(batch)
        text_tokens_embed = batch.text_embeddings

        @self.compile_mode
        def compiled_training_step(batch):
            with torch.no_grad():
                actions_in, masked_labels, ratio_unlabeled = (
                    self._create_target_and_masked_labels(batch)
                )
            loss, robot_action_mse, losses, auxiliary_outputs = self._calculate_loss(
                batch, actions_in, masked_labels, text_tokens_embed
            )

            auxiliary_outputs["ratio_unlabeled"] = ratio_unlabeled
            return loss, losses, robot_action_mse, auxiliary_outputs

        loss, losses, robot_action_mse, auxiliary_outputs = compiled_training_step(
            batch
        )

        if self.trainer.global_step == 0:
            logging.info(
                f"First training step completed. rank={self.trainer.global_rank}"
            )

        self._training_loss_metric.update(loss)
        self._training_ratio_unlabeled_metric.update(
            auxiliary_outputs["ratio_unlabeled"]
        )
        self._training_robot_action_mse_metric.update(robot_action_mse)
        self._training_lb_loss_metric.update(losses["lb_loss"])
        self._training_rz_loss_metric.update(losses["rz_loss"])

        for i in range(self.num_of_experts):
            metric = getattr(self, f"_training_expert_{i}_capacity_metric")
            metric.update(auxiliary_outputs["num_tokens_per_expert"][i])

        if self.trainer.global_step % 50 == 0:
            training_loss = self._training_loss_metric.compute()
            training_robot_action_mse = self._training_robot_action_mse_metric.compute()
            training_lb_loss = self._training_lb_loss_metric.compute()
            training_rz_loss = self._training_rz_loss_metric.compute()
            training_ratio_unlabeled = (
                self._training_ratio_unlabeled_metric.compute()
            )

            if self.trainer.is_global_zero:
                logging.info(
                    "global_step=%s training_loss=%.6f training_robot_action_mse=%.6f training_lb_loss=%.6f training_rz_loss=%.6f training_ratio_unlabeled=%.6f",
                    self.trainer.global_step,
                    _metric_to_float(training_loss),
                    _metric_to_float(training_robot_action_mse),
                    _metric_to_float(training_lb_loss),
                    _metric_to_float(training_rz_loss),
                    _metric_to_float(training_ratio_unlabeled),
                )

            self.log(
                "training_loss",
                training_loss,
                sync_dist=True,
                add_dataloader_idx=False,
                on_step=True,
            )
            self.log(
                "training_robot_action_mse",
                training_robot_action_mse,
                sync_dist=True,
                add_dataloader_idx=False,
                on_step=True,
            )
            self.log(
                "training_lb_loss",
                training_lb_loss,
                sync_dist=True,
                add_dataloader_idx=False,
                on_step=True,
            )
            self.log(
                "training_rz_loss",
                training_rz_loss,
                sync_dist=True,
                add_dataloader_idx=False,
                on_step=True,
            )
            self.log(
                "training_ratio_unlabeled",
                training_ratio_unlabeled,
                sync_dist=True,
                add_dataloader_idx=False,
                on_step=True,
            )
            for i in range(self.num_of_experts):
                metric = getattr(self, f"_training_expert_{i}_capacity_metric")
                self.log(
                    f"training_expert_{i}_capacity",
                    metric.compute(),
                    sync_dist=True,
                    add_dataloader_idx=False,
                    on_step=True,
                )
                metric.reset()
            self._training_loss_metric.reset()
            self._training_robot_action_mse_metric.reset()
            self._training_lb_loss_metric.reset()
            self._training_rz_loss_metric.reset()
            self._training_ratio_unlabeled_metric.reset()

        B, T = batch.frames.shape[0], batch.frames.shape[1]
        n_global_training_frames = (
            self.trainer.global_step
            * self.trainer.accumulate_grad_batches
            * self.trainer.num_devices
            * T
            * B
        )
        self.log(
            "n_global_training_frames",
            n_global_training_frames,
            add_dataloader_idx=False,
            on_step=True,
        )

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        self._validate_batch_sequence_length(batch)
        text_tokens_embed = batch.text_embeddings

        @self.compile_mode
        def _compiled_validation_step(batch, text_tokens_embed):
            actions_in, masked_labels, _ = self._create_target_and_masked_labels(batch)
            loss, robot_action_mse, losses, auxiliary_outputs = self._calculate_loss(
                batch, actions_in, masked_labels, text_tokens_embed
            )
            return loss, robot_action_mse, losses, auxiliary_outputs

        loss, robot_action_mse, losses, auxiliary_outputs = _compiled_validation_step(
            batch, text_tokens_embed
        )
        val_set_name = list(self.trainer.val_dataloaders.keys())[dataloader_idx]
        val_metrics = self._validation_metrics[val_set_name]
        val_metrics["loss"].update(loss)
        val_metrics["robot_action_mse"].update(robot_action_mse)
        val_metrics["lb_loss"].update(losses["lb_loss"])
        val_metrics["rz_loss"].update(losses["rz_loss"])
        for i in range(self.num_of_experts):
            val_metrics[f"expert_{i}_capacity"].update(
                auxiliary_outputs["num_tokens_per_expert"][i]
            )

    def on_validation_epoch_end(self):
        """Compute and log all validation metrics at the end of validation epoch"""
        for val_set_name, metrics in self._validation_metrics.items():
            # Compute, log, and reset all metrics for this validation set
            for metric_name, metric in metrics.items():
                self.log(
                    f"{val_set_name}_validation_{metric_name}",
                    metric.compute(),
                    sync_dist=True,
                    add_dataloader_idx=False,
                    on_step=False,
                    on_epoch=True,
                )
                metric.reset()

    def configure_optimizers(self):
        raise NotImplementedError("Not implemented for base class.")

    def _get_text_embedding_dim(self):
        return self.config.shared.text_tokenizer_config.text_embedding_shape[-1]

    def _get_text_tokenizer_name(self):
        return self.config.shared.text_tokenizer_config.text_tokenizer_name


class Stage3LabelledBCLightning(PolicyModelTrainer):
    def __init__(
        self,
        config: LightningPolicyConfig,
        inference_mode: bool = False,
    ):
        super().__init__(
            config,
            stage_name="stage3_finetune",
            inference_mode=inference_mode,
        )
        if self.config.policy_model.model_type == "sparse_moe":
            self.compile_mode = torch.compile(fullgraph=True)
        else:
            self.compile_mode = torch.compile(fullgraph=True, mode="max-autotune")

        self.transformer_forward_function = self.bc_transformer.forward

    def _get_transformer_mask_fn(self):
        # Use the default, causal mask.
        return None

    def configure_optimizers(self):
        assert not self.inference_mode
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.config.stage3_finetune.optim.learning_rate,
            betas=(
                self.config.stage3_finetune.optim.beta_1,
                self.config.stage3_finetune.optim.beta_2,
            ),
            weight_decay=self.config.stage3_finetune.optim.weight_decay,
            fused=True,
        )
        return [optimizer]

    def on_train_batch_start(self, batch, batch_idx):
        global_step = self.trainer.global_step
        # This will get called multiple times if gradient accumulation is used.
        if (
            global_step == 0
            and self.config.stage3_finetune.freeze_transformer_layers_for_steps > 0
            and not self._already_frozen
        ):
            logging.warning("Freezing transformer layers.")
            for param in self.bc_transformer.parameters():
                param.requires_grad = False
            for param in self.image_tokenizer.parameters():
                param.requires_grad = False
            self._already_frozen = True
        elif (
            global_step
            == self.config.stage3_finetune.freeze_transformer_layers_for_steps
            and not self._already_unfrozen
        ):
            logging.warning("Unfreezing transformer layers.")
            for param in self.bc_transformer.parameters():
                param.requires_grad = True
            for param in self.image_tokenizer.parameters():
                param.requires_grad = True
            self._already_unfrozen = True

    def _compute_effective_mask(
        self, user_action_mask, valid_frame_mask, system_action_mask
    ):
        return valid_frame_mask & (user_action_mask)


def _init_stage3_model(config: LightningPolicyConfig) -> Stage3LabelledBCLightning:
    logging.warning("Initializing stage3 model with random weights.")
    assert config.stage3_finetune.init.stage2_model_path is None, (
        "stage2_model_path is not allowed when initializing with random weights"
    )
    model = Stage3LabelledBCLightning(config)

    return model


class SupervisedDataModule(pl.LightningDataModule):
    def __init__(
        self,
        cfg: LightningPolicyConfig,
        training_dataset_cfg: DatasetConfig,
        validation_dataset_cfgs: List[ValidationDatasetConfig],
        stage_name: str,
        **kwargs,
    ):
        super().__init__()
        self.cfg = cfg
        self.training_dataset_cfg = training_dataset_cfg
        self.validation_dataset_cfgs = validation_dataset_cfgs
        self.stage_name = stage_name
        self._setup_completed = False

        assert self.cfg.stage3_finetune.accumulate_grad_batches == 1, (
            "accumulate_grad_batches can deadlock with multiple GPUs"
        )
        self.text_tokenizer_config = TextTokenizerConfig(
            text_tokenizer_name=self.cfg.shared.text_tokenizer_config.text_tokenizer_name,
            text_embedding_shape=self.cfg.shared.text_tokenizer_config.text_embedding_shape,
            text_annotation_model_version=self.cfg.shared.text_tokenizer_config.text_annotation_model_version,
        )

    def _make_dataloader(self, dataset, dataset_cfg: DatasetConfig, shuffle: bool):
        dataloader_kwargs = dict(
            dataset=dataset,
            batch_size=dataset_cfg.batch_size,
            shuffle=shuffle,
            num_workers=dataset_cfg.dataset_worker_num_workers_per_gpu,
            pin_memory=True,
            drop_last=True,
            persistent_workers=dataset_cfg.dataset_worker_num_workers_per_gpu > 0,
        )
        if dataset_cfg.dataset_worker_num_workers_per_gpu > 0:
            dataloader_kwargs["prefetch_factor"] = (
                dataset_cfg.dataset_worker_prefetch_factor
            )
        return DataLoader(**dataloader_kwargs)

    def _init_train_dataset(self):
        return MultiLatentLeRobotDataset(self.training_dataset_cfg)

    def _init_dummy_dataset(self):
        return DummyDataset(
            DummyDatasetConfig(
                frame_height=self.cfg.shared.frame_height,
                frame_width=self.cfg.shared.frame_width,
                T=self.cfg.shared.n_seq_timesteps,
                action_mapping=self.cfg.stage3_finetune.action_mapping,
            ),
        )

    def setup(self, stage: str):
        self.global_rank = getattr(self.trainer, "global_rank", 0)
        self.world_size = getattr(self.trainer, "world_size", 1)
        logging.info(
            f"Setting up datasets. global_rank: {self.global_rank}, world_size: {self.world_size}, stage {stage}"
        )
        if self._setup_completed:
            logging.info(
                f"Setup already completed. global_rank: {self.global_rank}, world_size: {self.world_size}, stage {stage}"
            )
            return
        self._setup_completed = True

        self.train_dataset = self._init_train_dataset()
        self._train_dataloader = self._make_dataloader(
            self.train_dataset,
            self.training_dataset_cfg,
            shuffle=self.training_dataset_cfg.shuffle,
        )

        self.validation_datasets = {}
        for i, validation_dataset_cfg in enumerate(self.validation_dataset_cfgs):
            validation_dataset = MultiLatentLeRobotDataset(validation_dataset_cfg)
            self.validation_datasets[validation_dataset_cfg.validation_name] = (
                validation_dataset
            )

        self._val_dataloaders = {
            k: self._make_dataloader(d, self.validation_dataset_cfgs[i], shuffle=False)
            for i, (k, d) in enumerate(self.validation_datasets.items())
        }

    def train_dataloader(self):
        return self._train_dataloader

    def val_dataloader(self):
        return self._val_dataloaders

    def get_action_mapping(self):
        return {
            "action_type": "robot_continuous",
            "robot_action_dim": self.training_dataset_cfg.get_robot_action_dim(),
        }


class Stage3DataModule(SupervisedDataModule):
    def __init__(self, cfg: LightningPolicyConfig):
        super().__init__(
            cfg=cfg,
            training_dataset_cfg=cfg.stage3_finetune.training_dataset,
            validation_dataset_cfgs=cfg.stage3_finetune.validation_datasets,
            stage_name="stage3_finetune",
        )

    def _should_drop_chunks_with_only_system_actions(self):
        return True


def train_stage3_finetune(config: LightningPolicyConfig):
    _sync_sequence_length_with_dataset(config)
    _sync_text_embedding_shape_with_dataset(config)
    resume_ckpt_path = _resolve_resume_checkpoint_path(config)
    if resume_ckpt_path is not None:
        _validate_resume_checkpoint_compatibility(resume_ckpt_path, config)
    datamodule = Stage3DataModule(config)
    # This is for start_experiment.py type jobs
    run_id = getattr(config.wandb, "run_id", None) or os.environ.get("WANDB_RUN_ID")
    if not run_id:
        run_id = wandb.util.generate_id()

    os.environ["WANDB_RUN_ID"] = run_id
    config.wandb.run_id = run_id

    wandb_logger = pl.pytorch.loggers.WandbLogger(
        entity="elefantai",
        project=config.wandb.project,
        name=config.wandb.exp_name + "_stage3_finetune",
        version=run_id,
        id=run_id,
        log_model=False,
        save_code=False,
        save_dir=ELEFANT_WANDB_DIR,
        config=config.model_dump(),
        group=config.wandb.exp_name,
        job_type="train",
        mode="online" if config.wandb.enabled else "disabled",
    )

    checkpoint_path = f"{config.shared.output_path}/stage3_finetune"
    upload_model_config(checkpoint_path, config)
    upload_action_mapping(checkpoint_path, datamodule.get_action_mapping())

    async_checkpointer = AsyncCheckpointIO()
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_path,
        every_n_train_steps=config.stage3_finetune.save_every_n_steps,
        filename="checkpoint-{step:08d}",
        enable_version_counter=False,
        save_top_k=-1,
    )

    if torch.cuda.device_count() > 1:
        logging.info(f"Using DDP strategy with {torch.cuda.device_count()} GPUs")
        strategy = pl.pytorch.strategies.DDPStrategy(find_unused_parameters=True)
    else:
        logging.info("Using SingleDeviceStrategy for single GPU.")
        # strategy = pl.pytorch.strategies.SingleDeviceStrategy(accelerator="auto")
        # Setting strategy with single GPU explicitly seems to error.
        # https://github.com/Lightning-AI/pytorch-lightning/issues/18902
        strategy = "auto"

    trainer = pl.Trainer(
        plugins=[async_checkpointer],
        callbacks=[checkpoint_callback],
        accelerator="auto",
        # For debugging it can be useful to set devices to 1.
        # for simpler stack traces etc.
        devices="auto",
        max_steps=config.stage3_finetune.n_training_steps,
        logger=wandb_logger,
        # We multiply by accumulate_grad_batches to get the number of steps between validation steps in "real" steps.
        val_check_interval=config.stage3_finetune.validation_step_interval
        * config.stage3_finetune.accumulate_grad_batches,
        limit_val_batches=config.stage3_finetune.n_validation_steps,
        check_val_every_n_epoch=None,
        precision=config.shared.precision,
        accumulate_grad_batches=config.stage3_finetune.accumulate_grad_batches,
        fast_dev_run=config.shared.fast_dev_run,
        strategy=strategy,
        # We already run validation before training starts.
        num_sanity_val_steps=0,
        # profiler="simple",
    )

    # Initialize model on the correct device using PyTorch Lightning's device management
    # Disable if using FSDP or DeepSpeed.
    # https://lightning.ai/docs/pytorch/stable/advanced/model_init.html
    with trainer.init_module():
        model = _init_stage3_model(config)

    total_params, expert_params = count_model_parameters(model)
    logging.info(
        f"Total parameters: {total_params}, Expert parameters: {expert_params}"
    )

    if resume_ckpt_path is not None:
        logging.info("Resuming stage3 training from checkpoint: %s", resume_ckpt_path)

    trainer.fit(model, datamodule, ckpt_path=resume_ckpt_path)

    wandb_logger.experiment.finish()
    return async_checkpointer.get_final_checkpoint()

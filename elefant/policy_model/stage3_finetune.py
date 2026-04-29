import contextlib
import logging
import json
import shutil
import subprocess
import sys

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
from elefant.text_tokenizer.factory import get_text_tokenizer
from elefant.data.rand_augment import BatchRandAugment
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
from elefant.policy_model.flow_matching import FlowMatchScheduler, sample_timestep_id
from rdt.model import RDT
from lightning.fabric.utilities.cloud_io import get_filesystem


def _load_lerobot_dataset_support():
    try:
        from elefant.data.lerobot_latent_dataset import (
            MultiLatentLeRobotDataset,
            infer_text_embedding_shape_from_dataset,
        )
    except ModuleNotFoundError as exc:
        if exc.name and (exc.name == 'lerobot' or exc.name.startswith('lerobot.')):
            raise ModuleNotFoundError(
                'LeRobot dataset support requires the `lerobot` package. '
                'Inference paths like the RobotWin websocket server can run '
                'without it, but stage3 dataset loading and training cannot.'
            ) from exc
        raise

    return MultiLatentLeRobotDataset, infer_text_embedding_shape_from_dataset


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

    legacy_rdt_keys = (
        'policy_summary_to_rdt_action_tokens_proj.',
        'policy_summary_to_rdt_state_proj.',
        'text_condition_to_rdt_proj.',
    )
    if any(any(key.startswith(prefix) for prefix in legacy_rdt_keys) for key in state_dict):
        raise ValueError(
            'Resume checkpoint RDT conditioning mismatch: the checkpoint uses the '
            'legacy direct-regression RDT head, but the current code expects the '
            'StreamVGGT-style denoising RDT head.'
        )
    if 'rdt_action_queries' in state_dict:
        raise ValueError(
            'Resume checkpoint RDT x-source mismatch: learned query based RDT checkpoints '
            'are not compatible with the current noisy-action denoising head.'
        )

    expected_n_actions = config.stage3_finetune.training_dataset.get_robot_action_dim()
    expected_seq_len = config.shared.n_seq_timesteps
    expected_transformer_action_tokens = 1

    action_pos_tokens = state_dict.get('bc_transformer.action_pos_tokens')
    if action_pos_tokens is not None:
        checkpoint_n_actions = int(action_pos_tokens.shape[1])
        if checkpoint_n_actions != expected_transformer_action_tokens:
            raise ValueError(
                'Resume checkpoint action-token mismatch: '
                f'checkpoint has {checkpoint_n_actions} transformer action tokens but config expects {expected_transformer_action_tokens}. '
                'This usually means you are trying to resume a checkpoint trained with a different action-tokenization scheme.'
            )

    x_pos_emb = state_dict.get('rdt_policy_head.x_pos_emb')
    if x_pos_emb is not None:
        expected_x_pos_tokens = expected_seq_len + config.policy_model.rdt.num_register_tokens
        checkpoint_x_pos_tokens = int(x_pos_emb.shape[1])
        if checkpoint_x_pos_tokens != expected_x_pos_tokens:
            raise ValueError(
                'Resume checkpoint RDT position embedding mismatch: '
                f'checkpoint has {checkpoint_x_pos_tokens} x_pos tokens but config expects {expected_x_pos_tokens}. '
                'This usually means you are trying to resume a checkpoint trained with a different sequence length.'
            )

    act_pos_emb = state_dict.get('rdt_policy_head.act_pos_emb')
    if act_pos_emb is None:
        raise ValueError(
            'Resume checkpoint RDT conditioning mismatch: checkpoint is missing '
            'rdt_policy_head.act_pos_emb required by the current action-token-conditioned RDT head.'
        )
    checkpoint_action_condition_tokens = int(act_pos_emb.shape[1])
    if checkpoint_action_condition_tokens != expected_seq_len:
        raise ValueError(
            'Resume checkpoint RDT action-condition length mismatch: '
            f'checkpoint has {checkpoint_action_condition_tokens} action condition tokens '
            f'but config expects {expected_seq_len}.'
        )

    action_embedder = state_dict.get('rdt_policy_head.action_embedder.weight')
    if action_embedder is not None and int(action_embedder.shape[1]) != expected_n_actions:
        raise ValueError(
            'Resume checkpoint RDT action embedder mismatch: '
            f'checkpoint action_dim={int(action_embedder.shape[1])} but config expects {expected_n_actions}.'
        )

    checkpoint_has_image_condition = state_dict.get('rdt_policy_head.img_pos_emb') is not None
    expected_image_condition = config.policy_model.rdt.use_image_condition
    if checkpoint_has_image_condition != expected_image_condition:
        raise ValueError(
            'Resume checkpoint RDT image-conditioning mismatch: '
            f'checkpoint image_condition={checkpoint_has_image_condition} but '
            f'config expects image_condition={expected_image_condition}.'
        )

def _resolve_optional_checkpoint_path(checkpoint_path: Optional[str]) -> Optional[str]:
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


def _resolve_resume_checkpoint_path(config: LightningPolicyConfig) -> Optional[str]:
    return _resolve_optional_checkpoint_path(config.stage3_finetune.init.stage3_model_path)


def _resolve_origin_init_checkpoint_path(
    config: LightningPolicyConfig,
) -> Optional[str]:
    return _resolve_optional_checkpoint_path(config.stage3_finetune.init.origin_model_path)


_ORIGIN_INIT_SKIP_PREFIXES = (
    "bc_transformer.action_decoder.",
    "key_action_embedding.",
    "mouse_button_embedding.",
    "mouse_delta_x_embedding.",
    "mouse_delta_y_embedding.",
    "keyboard_out_logits.",
    "mouse_button_out_logits.",
    "mouse_delta_x_out_logits.",
    "mouse_delta_y_out_logits.",
)

_ORIGIN_INIT_SKIP_KEYS = {
    "bc_transformer.action_pos_tokens",
}

_ORIGIN_TOKENIZER_PREFIX_REMAPS = (
    ("bc_transformer.image_tokenizer.", "bc_transformer.image_tokenizer.base_tokenizer."),
    ("image_tokenizer.", "image_tokenizer.base_tokenizer."),
)


def _should_skip_origin_init_key(key: str) -> bool:
    return key in _ORIGIN_INIT_SKIP_KEYS or key.startswith(_ORIGIN_INIT_SKIP_PREFIXES)



def _remap_origin_init_key(
    key: str, config: LightningPolicyConfig
) -> str:
    if config.stage3_finetune.training_dataset.multi_view_image_mode != "token_concat":
        return key

    for source_prefix, target_prefix in _ORIGIN_TOKENIZER_PREFIX_REMAPS:
        if key.startswith(source_prefix):
            return target_prefix + key[len(source_prefix) :]
    return key



def _reshape_tensor_with_repeat(
    source_tensor: torch.Tensor, target_shape: tuple[int, ...]
) -> torch.Tensor:
    flat_source = source_tensor.reshape(-1)
    target_numel = 1
    for dim in target_shape:
        target_numel *= dim

    if flat_source.numel() == 0:
        raise ValueError("Cannot adapt an empty tensor to a non-empty target shape.")

    if flat_source.numel() == target_numel:
        return flat_source.reshape(target_shape)

    repeats = (target_numel + flat_source.numel() - 1) // flat_source.numel()
    flat_source = flat_source.repeat(repeats)[:target_numel]
    return flat_source.reshape(target_shape)



def _adapt_origin_tensor_to_target(
    source_key: str,
    source_tensor: torch.Tensor,
    target_key: str,
    target_tensor: torch.Tensor,
) -> tuple[torch.Tensor, str]:
    if tuple(source_tensor.shape) == tuple(target_tensor.shape):
        return source_tensor.to(dtype=target_tensor.dtype), "direct"

    if (
        source_key.endswith("img_pos_tokens")
        and target_key.endswith("img_pos_tokens")
        and source_tensor.ndim == 3
        and target_tensor.ndim == 3
        and source_tensor.shape[0] == target_tensor.shape[0]
        and source_tensor.shape[2] == target_tensor.shape[2]
        and target_tensor.shape[1] % source_tensor.shape[1] == 0
    ):
        repeat_factor = target_tensor.shape[1] // source_tensor.shape[1]
        return (
            source_tensor.repeat(1, repeat_factor, 1).to(dtype=target_tensor.dtype),
            f"repeat_img_tokens_x{repeat_factor}",
        )

    raise ValueError(
        "Cannot adapt origin checkpoint tensor "
        f"{source_key} with shape {tuple(source_tensor.shape)} to "
        f"{target_key} with shape {tuple(target_tensor.shape)}. "
        "Only exact-shape loads and explicit whitelist adaptations are allowed."
    )


def _load_origin_checkpoint_state_dict(checkpoint_path: str) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("state_dict")
        if isinstance(state_dict, dict):
            return state_dict
        if all(isinstance(key, str) for key in checkpoint):
            return checkpoint
    raise ValueError(
        f"Unsupported origin checkpoint format at {checkpoint_path}; expected a Lightning checkpoint or a plain state_dict."
    )



def _maybe_initialize_from_origin_checkpoint(
    model: "Stage3LabelledBCLightning",
    config: LightningPolicyConfig,
) -> None:
    origin_checkpoint_path = _resolve_origin_init_checkpoint_path(config)
    if origin_checkpoint_path is None:
        return

    if config.stage3_finetune.init.stage3_model_path:
        logging.warning(
            "Skipping origin partial initialization because stage3_model_path is set; resume checkpoint loading takes precedence."
        )
        return

    logging.warning(
        "Initializing compatible weights from origin checkpoint: %s",
        origin_checkpoint_path,
    )
    origin_state_dict = _load_origin_checkpoint_state_dict(origin_checkpoint_path)
    target_state_dict = model.state_dict()
    loadable_state_dict = {}

    counts = {
        "loaded": 0,
        "adapted": 0,
        "filtered": 0,
        "missing": 0,
        "remapped": 0,
    }
    adapted_examples: list[str] = []
    filtered_examples: list[str] = []
    missing_examples: list[str] = []

    for source_key, source_tensor in origin_state_dict.items():
        if _should_skip_origin_init_key(source_key):
            counts["filtered"] += 1
            if len(filtered_examples) < 8:
                filtered_examples.append(source_key)
            continue

        target_key = _remap_origin_init_key(source_key, config)
        if target_key != source_key:
            counts["remapped"] += 1

        target_tensor = target_state_dict.get(target_key)
        if target_tensor is None:
            counts["missing"] += 1
            if len(missing_examples) < 8:
                missing_examples.append(f"{source_key} -> {target_key}")
            continue

        adapted_tensor, adapt_mode = _adapt_origin_tensor_to_target(
            source_key=source_key,
            source_tensor=source_tensor,
            target_key=target_key,
            target_tensor=target_tensor,
        )
        loadable_state_dict[target_key] = adapted_tensor
        counts["loaded"] += 1
        if adapt_mode != "direct":
            counts["adapted"] += 1
            if len(adapted_examples) < 8:
                adapted_examples.append(
                    f"{source_key} -> {target_key} [{tuple(source_tensor.shape)} -> {tuple(target_tensor.shape)}; {adapt_mode}]"
                )

    if not loadable_state_dict:
        logging.warning(
            "Origin checkpoint %s did not provide any compatible weights after filtering/remapping.",
            origin_checkpoint_path,
        )
        return

    missing_keys, unexpected_keys = model.load_state_dict(loadable_state_dict, strict=False)
    logging.warning(
        "Origin partial init summary: loaded=%s adapted=%s remapped=%s filtered=%s missing_target=%s model_missing_after_load=%s unexpected_after_load=%s",
        counts["loaded"],
        counts["adapted"],
        counts["remapped"],
        counts["filtered"],
        counts["missing"],
        len(missing_keys),
        len(unexpected_keys),
    )
    if adapted_examples:
        logging.info("Origin init adapted examples: %s", adapted_examples)
    if filtered_examples:
        logging.info("Origin init filtered examples: %s", filtered_examples)
    if missing_examples:
        logging.info("Origin init missing-target examples: %s", missing_examples)


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
    prefer_raw_text = getattr(
        config.stage3_finetune.training_dataset,
        "prefer_raw_text_for_text_embeddings",
        True,
    )
    if prefer_raw_text:
        tokenizer_name = getattr(
            config.shared.text_tokenizer_config, "text_tokenizer_name", None
        )
        if tokenizer_name is not None:
            try:
                text_tokenizer = get_text_tokenizer(config.shared.text_tokenizer_config)
            except Exception as exc:
                logging.warning(
                    'Failed to initialize configured text tokenizer `%s` for raw-text shape inference; falling back to dataset text_emb shape inference. Error: %s',
                    tokenizer_name,
                    exc,
                )
            else:
                inferred_shape = [
                    text_tokenizer.get_n_text_tokens(),
                    text_tokenizer.get_text_embed_dim(),
                ]
                current_shape = list(config.shared.text_tokenizer_config.text_embedding_shape)
                if current_shape != inferred_shape:
                    logging.warning(
                        'Overriding text_embedding_shape from %s to %s based on configured raw-text tokenizer `%s`.',
                        current_shape,
                        inferred_shape,
                        tokenizer_name,
                    )
                else:
                    logging.info(
                        'Using text_embedding_shape %s from configured raw-text tokenizer `%s`.',
                        inferred_shape,
                        tokenizer_name,
                    )
                config.shared.text_tokenizer_config.text_embedding_shape = inferred_shape
                config.stage3_finetune.training_dataset.text_embedding_shape = inferred_shape
                for val_dataset in config.stage3_finetune.validation_datasets:
                    val_dataset.text_embedding_shape = inferred_shape
                return

    _, infer_text_embedding_shape_from_dataset = _load_lerobot_dataset_support()
    inferred = infer_text_embedding_shape_from_dataset(
        config.stage3_finetune.training_dataset
    )
    if inferred is None:
        return

    inferred_shape, sample_path = inferred
    current_shape = list(config.shared.text_tokenizer_config.text_embedding_shape)
    tokenizer_name = getattr(
        config.shared.text_tokenizer_config, 'text_tokenizer_name', None
    )
    tokenizer_model_name_or_path = getattr(
        config.shared.text_tokenizer_config, 'model_name_or_path', None
    )
    if current_shape != inferred_shape:
        logging.warning(
            'Overriding text_embedding_shape from %s to %s based on %s.',
            current_shape,
            inferred_shape,
            sample_path,
        )
        if tokenizer_name is not None:
            logging.warning(
                'Configured text tokenizer `%s` (%s) does not match dataset text embedding shape %s from %s; training will follow the dataset embeddings. Regenerate dataset text_emb/empty_emb if you want to switch tokenizer end-to-end.',
                tokenizer_name,
                tokenizer_model_name_or_path or 'default-model-source',
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

        self._force_stable_rdt_training = (
            stage_name == "stage3_finetune"
            and "bf16" in str(self.config.shared.precision).lower()
        )
        self._rdt_param_dtype = self._get_rdt_param_dtype()
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
        torch_compile_enabled = bool(
            getattr(self.config.shared, "enable_torch_compile", True)
        )
        if self._force_stable_rdt_training and torch_compile_enabled:
            logging.warning(
                "Disabling torch.compile for stage3 bf16 RDT training to avoid "
                "non-finite action predictions during resumed multi-GPU runs."
            )
            torch_compile_enabled = False

        if torch_compile_enabled:
            if self.config.policy_model.model_type == "sparse_moe":
                self.compile_mode = torch.compile(fullgraph=True)
            else:
                # default compilation mode is without max autotune which gets
                # edited when initializing stage 1/3 with max-autotunes
                self.compile_mode = torch.compile()
        else:
            self.compile_mode = lambda fn: fn
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
        self._init_rdt_schedulers()
        if self._force_stable_rdt_training and self._rdt_param_dtype == torch.float32:
            logging.warning(
                "Keeping stage3 RDT trainable weights in float32 and running "
                "its denoising path without autocast for better bf16 resume "
                "stability."
            )

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

    def _init_rdt_schedulers(self):
        rdt_cfg = self.config.policy_model.rdt
        scheduler_kwargs = dict(
            num_inference_steps=rdt_cfg.num_inference_steps,
            num_train_timesteps=rdt_cfg.num_train_timesteps,
            shift=rdt_cfg.flow_match_shift,
            sigma_max=rdt_cfg.sigma_max,
            sigma_min=rdt_cfg.sigma_min,
            extra_one_step=rdt_cfg.extra_one_step,
        )
        self.rdt_train_scheduler = FlowMatchScheduler(**scheduler_kwargs)
        self.rdt_train_scheduler.set_timesteps(
            rdt_cfg.num_train_timesteps,
            training=True,
        )
        self.rdt_inference_scheduler = FlowMatchScheduler(**scheduler_kwargs)
        self.rdt_inference_scheduler.set_timesteps(rdt_cfg.num_inference_steps)

    def _get_rdt_param_dtype(self) -> torch.dtype:
        # Keep trainable RDT parameters in fp32 during training so the optimizer
        # maintains stable master weights even when Lightning runs bf16 autocast.
        if not self.inference_mode:
            return torch.float32
        precision = str(self.config.shared.precision).lower()
        if "bf16" in precision:
            return torch.bfloat16
        return torch.float32

    def _rdt_forward_context(self, device: torch.device):
        if not self._force_stable_rdt_training:
            return contextlib.nullcontext()
        return torch.autocast(device_type=device.type, enabled=False)

    def _init_action_mapping(self):
        self.n_actions = self.config.stage3_finetune.training_dataset.get_robot_action_dim()
        self.embedding_std = 0.1
        self.rdt_hidden_size = self.config.policy_model.rdt.hidden_size
        self.use_state_condition = self.config.policy_model.rdt.use_state_condition
        self.use_image_condition = self.config.policy_model.rdt.use_image_condition
        self.rdt_horizon = self.config.shared.n_seq_timesteps
        self.rdt_action_condition_len = self.config.shared.n_seq_timesteps
        self.rdt_image_condition_tokens_per_step = (
            self.image_tokenizer.get_n_img_tokens()
        )
        rdt_param_dtype = self._rdt_param_dtype

        def _init_linear(layer: nn.Linear):
            torch.nn.init.normal_(layer.weight, mean=0.0, std=self.embedding_std)
            if layer.bias is not None:
                torch.nn.init.zeros_(layer.bias)

        self.robot_action_in_proj = nn.Linear(
            self.n_actions,
            self.config.policy_model.transformer_dim,
            dtype=rdt_param_dtype,
        )
        _init_linear(self.robot_action_in_proj)

        if self.config.policy_model.transformer_dim == self.rdt_hidden_size:
            self.policy_action_condition_to_rdt_proj = nn.Identity()
        else:
            self.policy_action_condition_to_rdt_proj = nn.Linear(
                self.config.policy_model.transformer_dim,
                self.rdt_hidden_size,
                dtype=rdt_param_dtype,
            )
            _init_linear(self.policy_action_condition_to_rdt_proj)

        if self.config.policy_model.transformer_dim == self.rdt_hidden_size:
            self.image_condition_to_rdt_proj = nn.Identity()
        else:
            self.image_condition_to_rdt_proj = nn.Linear(
                self.config.policy_model.transformer_dim,
                self.rdt_hidden_size,
                dtype=rdt_param_dtype,
            )
            _init_linear(self.image_condition_to_rdt_proj)

        img_pos_emb_config = None
        max_img_len = 0
        if self.use_image_condition:
            img_pos_emb_config = [
                (
                    "image",
                    (
                        self.config.shared.n_seq_timesteps,
                        self.rdt_image_condition_tokens_per_step,
                    ),
                )
            ]
            max_img_len = (
                self.config.shared.n_seq_timesteps
                * self.rdt_image_condition_tokens_per_step
            )

        self.null_state_condition = nn.Parameter(
            torch.zeros(1, 1, self.n_actions, dtype=rdt_param_dtype)
        )
        torch.nn.init.normal_(
            self.null_state_condition, mean=0.0, std=self.embedding_std
        )

        rdt_config = self.config.policy_model.rdt
        use_flash_attn = bool(rdt_config.use_flash_attn)
        if self._force_stable_rdt_training and use_flash_attn:
            logging.warning(
                "Disabling RDT flash attention for stage3 bf16 training because "
                "this configuration has produced non-finite action predictions."
            )
            use_flash_attn = False

        self.rdt_policy_head = RDT(
            horizon=self.rdt_horizon,
            output_size=self.n_actions,
            config={
                "hidden_size": self.rdt_hidden_size,
                "num_heads": rdt_config.num_heads,
                "num_kv_heads": rdt_config.num_kv_heads,
                "depth": rdt_config.depth,
                "norm_eps": rdt_config.norm_eps,
                "multiple_of": rdt_config.multiple_of,
                "ffn_dim_multiplier": rdt_config.ffn_dim_multiplier,
                "use_flash_attn": use_flash_attn,
                "num_register_tokens": rdt_config.num_register_tokens,
                "action_dim": self.n_actions,
            },
            x_pos_emb_config=[
                ("action", self.rdt_horizon),
                ("register", rdt_config.num_register_tokens),
            ],
            lang_pos_emb_config=[],
            max_lang_len=0,
            img_pos_emb_config=img_pos_emb_config,
            max_img_len=max_img_len,
            act_pos_emb_config=[("action", self.rdt_action_condition_len)],
            max_act_len=self.rdt_action_condition_len,
            dtype=rdt_param_dtype,
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
            "KV-cache autoregressive inference is no longer supported. Use online_full_predict() for continuous RobotAction prediction."
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
        initial_sample: Optional[torch.Tensor] = None,
        compile: bool = True,
    ) -> torch.Tensor:
        valid_step_mask = self._get_rdt_valid_step_mask(frames)
        frames = self._normalize_frames(frames)
        B, T = frames.shape[0], frames.shape[1]
        text_tokens_embed = self._prepare_text_tokens_embed_for_inference(
            batch_size=B,
            n_steps=T,
            device=frames.device,
            text_tokens_embed=text_tokens_embed,
        )
        action_embeddings_in = self.action_in_to_tokens(actions)
        (
            _,
            action_out_tokens,
            image_tokens,
            *_ ,
        ) = self.transformer_forward_function(
            frames, action_embeddings_in, text_tokens_embed
        )
        action_preds = self.action_tokens_to_actions(
            action_out_tokens=action_out_tokens,
            image_tokens=image_tokens,
            action_shape_source=actions,
            state_actions=actions,
            valid_step_mask=valid_step_mask,
            initial_sample=initial_sample,
            compile=compile,
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
            compile=False,
        )

    def online_full_predict(
        self,
        frames: torch.Tensor,
        actions: torch.Tensor,
        kv_cache_state: List[KVCacheState] = None,
        sampling_temperature: float = 1.0,
        text_tokens_embed: Optional[torch.Tensor] = None,
        initial_sample: Optional[torch.Tensor] = None,
        compile: bool = True,
    ) -> torch.Tensor:
        del kv_cache_state, sampling_temperature
        with torch.inference_mode():
            return self.online_full_predict_actions(
                frames=frames,
                actions=actions,
                text_tokens_embed=text_tokens_embed,
                initial_sample=initial_sample,
                compile=compile,
            )

    def action_in_to_tokens(
        self, action_in: torch.Tensor, idx: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Project a full continuous robot action vector into one action token per step."""
        B, T, D = action_in.shape
        eager_assert(action_in.shape, (B, T, self.n_actions))

        if idx is None:
            action_subset = action_in
        else:
            action_subset = action_in[idx]

        action_proj_dtype = self.robot_action_in_proj.weight.dtype
        action_embedding = self.robot_action_in_proj(
            action_subset.to(action_proj_dtype)
        ).unsqueeze(2)
        eager_assert(
            action_embedding.shape,
            (
                B if idx is None else len(idx),
                T,
                self.transformer_n_action_tokens,
                self.config.policy_model.transformer_dim,
            ),
        )
        return action_embedding

    def _get_rdt_valid_step_mask(
        self,
        frames: torch.Tensor,
        valid_step_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if valid_step_mask is not None:
            return valid_step_mask
        inferred_mask = frames.abs().sum(dim=tuple(range(2, frames.ndim))) > 0
        if inferred_mask.any():
            return inferred_mask
        return torch.ones(
            frames.shape[0],
            frames.shape[1],
            dtype=torch.bool,
            device=frames.device,
        )

    def _maybe_add_noise_to_action_condition(
        self, action_condition: torch.Tensor
    ) -> torch.Tensor:
        noise_std = float(self.config.policy_model.rdt.action_condition_noise_std)
        if not self.training or noise_std <= 0.0:
            return action_condition

        action_condition_fp32 = action_condition.float()
        # Scale the noise by each step token RMS so one fixed std stays usable
        # across checkpoints and training stages.
        token_rms = action_condition_fp32.pow(2).mean(dim=-1, keepdim=True)
        token_rms = token_rms.add(1e-6).sqrt()
        noise = torch.randn_like(action_condition_fp32) * (token_rms * noise_std)
        return (action_condition_fp32 + noise).to(dtype=action_condition.dtype)

    def _build_rdt_conditions(
        self,
        action_out_tokens: torch.Tensor,
        image_tokens: torch.Tensor,
        state_actions: torch.Tensor,
        valid_step_mask: torch.Tensor,
        condition_step_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
        eager_assert(
            action_out_tokens.shape,
            (state_actions.shape[0], self.rdt_action_condition_len, self.config.policy_model.transformer_dim),
        )
        eager_assert(valid_step_mask.shape, (state_actions.shape[0], self.rdt_action_condition_len))
        if condition_step_mask is None:
            condition_step_mask = valid_step_mask
        else:
            eager_assert(
                condition_step_mask.shape,
                (state_actions.shape[0], self.rdt_action_condition_len),
            )
            condition_step_mask = condition_step_mask & valid_step_mask

        action_condition_dtype = getattr(
            getattr(self.policy_action_condition_to_rdt_proj, "weight", None),
            "dtype",
            action_out_tokens.dtype,
        )
        action_condition = self.policy_action_condition_to_rdt_proj(
            action_out_tokens.to(action_condition_dtype)
        )
        eager_assert(
            action_condition.shape,
            (
                state_actions.shape[0],
                self.rdt_action_condition_len,
                self.rdt_hidden_size,
            ),
        )
        action_condition = self._maybe_add_noise_to_action_condition(action_condition)

        image_condition = None
        image_condition_mask = None
        if self.use_image_condition:
            image_condition_dtype = getattr(
                getattr(self.image_condition_to_rdt_proj, "weight", None),
                "dtype",
                image_tokens.dtype,
            )
            eager_assert(
                image_tokens.shape,
                (
                    state_actions.shape[0],
                    self.rdt_action_condition_len,
                    self.rdt_image_condition_tokens_per_step,
                    self.config.policy_model.transformer_dim,
                ),
            )
            image_condition = self.image_condition_to_rdt_proj(
                image_tokens.to(image_condition_dtype)
            ).reshape(
                state_actions.shape[0],
                self.rdt_action_condition_len * self.rdt_image_condition_tokens_per_step,
                self.rdt_hidden_size,
            )
            image_condition_mask = condition_step_mask.unsqueeze(-1).expand(
                state_actions.shape[0],
                self.rdt_action_condition_len,
                self.rdt_image_condition_tokens_per_step,
            ).reshape(
                state_actions.shape[0],
                self.rdt_action_condition_len * self.rdt_image_condition_tokens_per_step,
            )

        if self.use_state_condition:
            # Align with StreamVGGT: state_c comes from the first clean action in the
            # action sequence being modeled by RDT.
            state_condition = state_actions[:, :1, :].to(self.null_state_condition.dtype)
        else:
            state_condition = self.null_state_condition.expand(
                state_actions.shape[0], -1, -1
            )
        eager_assert(state_condition.shape, (state_actions.shape[0], 1, self.n_actions))
        return (
            state_condition,
            action_condition,
            image_condition,
            condition_step_mask,
            image_condition_mask,
        )

    def _sample_noisy_action_sequence(
        self,
        clean_actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = clean_actions.shape[0]
        timestep_ids = sample_timestep_id(
            batch_size=batch_size,
            num_train_timesteps=self.rdt_train_scheduler.num_train_timesteps,
        ).to(clean_actions.device)
        timesteps = self.rdt_train_scheduler.timesteps.to(clean_actions.device)[timestep_ids]
        sigmas = self.rdt_train_scheduler.sigmas.to(clean_actions.device)[timestep_ids].view(
            batch_size, 1, 1
        )
        noise = torch.randn_like(clean_actions)
        noisy_actions = (1 - sigmas) * clean_actions + sigmas * noise
        targets = self.rdt_train_scheduler.training_target(clean_actions, noise, timesteps)
        weights = self.rdt_train_scheduler.training_weight(timesteps).view(batch_size, 1, 1)
        return noisy_actions, targets, timesteps, weights

    def _predict_rdt_denoising_target(
        self,
        noisy_actions: torch.Tensor,
        timesteps: torch.Tensor,
        action_out_tokens: torch.Tensor,
        image_tokens: torch.Tensor,
        state_actions: torch.Tensor,
        valid_step_mask: torch.Tensor,
        condition_step_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        rdt_input_dtype = self.rdt_policy_head.action_embedder.weight.dtype
        with self._rdt_forward_context(noisy_actions.device):
            (
                state_condition,
                action_condition,
                image_condition,
                action_condition_mask,
                image_condition_mask,
            ) = self._build_rdt_conditions(
                action_out_tokens=action_out_tokens,
                image_tokens=image_tokens,
                state_actions=state_actions,
                valid_step_mask=valid_step_mask,
                condition_step_mask=condition_step_mask,
            )
            action_preds = self.rdt_policy_head(
                x=noisy_actions.to(rdt_input_dtype),
                t=timesteps,
                img_c=image_condition,
                act_c=action_condition,
                state_c=state_condition,
                img_mask=image_condition_mask,
                act_mask=action_condition_mask,
                embed_input=True,
                decode_output=True,
            )
        eager_assert(action_preds.shape, noisy_actions.shape)
        return action_preds.float()

    def _denoise_action_sequence(
        self,
        action_out_tokens: torch.Tensor,
        image_tokens: torch.Tensor,
        action_shape_source: torch.Tensor,
        state_actions: torch.Tensor,
        valid_step_mask: torch.Tensor,
        initial_sample: Optional[torch.Tensor] = None,
        compile: bool = True,
    ) -> torch.Tensor:
        del compile
        if initial_sample is None:
            sample = torch.randn_like(action_shape_source)
        else:
            eager_assert(initial_sample.shape, action_shape_source.shape)
            sample = initial_sample.to(
                device=action_shape_source.device,
                dtype=action_shape_source.dtype,
            ).clone()
        scheduler = self.rdt_inference_scheduler
        timesteps = scheduler.timesteps.to(action_shape_source.device)
        for step_idx, timestep in enumerate(timesteps):
            timestep_batch = torch.full(
                (action_shape_source.shape[0],),
                float(timestep.item()),
                dtype=torch.float32,
                device=action_shape_source.device,
            )
            model_output = self._predict_rdt_denoising_target(
                noisy_actions=sample,
                timesteps=timestep_batch,
                action_out_tokens=action_out_tokens,
                image_tokens=image_tokens,
                state_actions=state_actions,
                valid_step_mask=valid_step_mask,
            )
            sample = scheduler.step(
                model_output=model_output,
                timestep=timestep_batch,
                sample=sample,
                to_final=(step_idx + 1 == len(timesteps)),
            )
        eager_assert(sample.shape, action_shape_source.shape)
        return sample

    def action_tokens_to_actions(
        self,
        action_out_tokens: torch.Tensor,
        image_tokens: torch.Tensor,
        action_shape_source: torch.Tensor,
        state_actions: Optional[torch.Tensor] = None,
        valid_step_mask: Optional[torch.Tensor] = None,
        initial_sample: Optional[torch.Tensor] = None,
        compile: bool = True,
    ) -> torch.Tensor:
        if state_actions is None:
            state_actions = action_shape_source
        if valid_step_mask is None:
            valid_step_mask = torch.ones(
                action_shape_source.shape[0],
                action_shape_source.shape[1],
                dtype=torch.bool,
                device=action_shape_source.device,
            )
        return self._denoise_action_sequence(
            action_out_tokens=action_out_tokens,
            image_tokens=image_tokens,
            action_shape_source=action_shape_source,
            state_actions=state_actions,
            valid_step_mask=valid_step_mask,
            initial_sample=initial_sample,
            compile=compile,
        )

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

    def on_fit_start(self):
        if not self._force_stable_rdt_training:
            return
        converted_tensors = 0
        converted_params = 0
        for optimizer in self.trainer.optimizers:
            for param, state in optimizer.state.items():
                if not isinstance(param, torch.Tensor) or param.dtype != torch.float32:
                    continue
                param_converted = False
                for key, value in list(state.items()):
                    if torch.is_tensor(value) and value.is_floating_point() and value.dtype != torch.float32:
                        state[key] = value.float()
                        converted_tensors += 1
                        param_converted = True
                if param_converted:
                    converted_params += 1
        if converted_tensors > 0:
            logging.warning(
                "Upcast %s resumed optimizer-state tensors across %s fp32 parameters "
                "to improve stage3 bf16 resume stability.",
                converted_tensors,
                converted_params,
            )

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
        """Calculate the denoising loss for the StreamVGGT-style RDT head."""
        frames = self._normalize_frames(batch.frames)
        batch_size = batch.frames.shape[0]
        T = batch.frames.shape[1]
        valid_step_mask = self._get_rdt_valid_step_mask(
            batch.frames,
            valid_step_mask=batch.valid_frame_mask,
        )
        action_embeddings_in = self.action_in_to_tokens(actions_in)
        eager_assert(
            action_embeddings_in.shape,
            (
                batch_size,
                T,
                self.transformer_n_action_tokens,
                self.config.policy_model.transformer_dim,
            ),
        )
        (
            _,
            action_out_tokens,
            image_tokens,
            auxiliary_losses,
            auxiliary_outputs,
        ) = self.transformer_forward_function(
            frames, action_embeddings_in, text_tokens_embed
        )
        eager_assert(
            action_out_tokens.shape,
            (batch_size, T, self.config.policy_model.transformer_dim),
        )

        noisy_actions, action_targets, timesteps, action_loss_weights = (
            self._sample_noisy_action_sequence(actions_in)
        )
        # Align RDT conditioning visibility with the supervised action steps.
        condition_step_mask = masked_labels["mask"].bool().any(dim=-1)
        action_preds = self._predict_rdt_denoising_target(
            noisy_actions=noisy_actions,
            timesteps=timesteps,
            action_out_tokens=action_out_tokens,
            image_tokens=image_tokens,
            state_actions=actions_in,
            valid_step_mask=valid_step_mask,
            condition_step_mask=condition_step_mask,
        )
        action_mask_bool = masked_labels["mask"].bool()
        masked_action_preds = torch.where(
            action_mask_bool,
            action_preds,
            torch.zeros_like(action_preds),
        )
        masked_action_targets = torch.where(
            action_mask_bool,
            action_targets,
            torch.zeros_like(action_targets),
        )
        auxiliary_outputs["nonfinite_action_pred_count"] = (
            ~torch.isfinite(masked_action_preds)
        ).sum()
        auxiliary_outputs["nonfinite_action_target_count"] = (
            ~torch.isfinite(masked_action_targets)
        ).sum()
        action_mask = action_mask_bool.to(action_preds.dtype)
        squared_error = (masked_action_preds - masked_action_targets).pow(2) * action_loss_weights
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
        if batch.action_mask is not None and batch.action_mask.shape == actions_in.shape:
            action_mask = batch.action_mask.bool() & effective_mask.unsqueeze(-1)
        else:
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
        if batch.action_mask is not None:
            lengths["action_mask"] = batch.action_mask.shape[1]
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
        self._current_batch_idx = batch_idx
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

        nonfinite_action_pred_count = int(
            auxiliary_outputs["nonfinite_action_pred_count"].detach().cpu().item()
        )
        nonfinite_action_target_count = int(
            auxiliary_outputs["nonfinite_action_target_count"].detach().cpu().item()
        )
        if nonfinite_action_pred_count > 0:
            raise RuntimeError(
                "Non-finite action predictions on valid/masked-in positions: "
                f"count={nonfinite_action_pred_count}, global_step={self.trainer.global_step}, batch_idx={batch_idx}."
            )
        if nonfinite_action_target_count > 0:
            raise RuntimeError(
                "Non-finite action targets on valid/masked-in positions: "
                f"count={nonfinite_action_target_count}, global_step={self.trainer.global_step}, batch_idx={batch_idx}."
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
        if self._force_stable_rdt_training:
            self.compile_mode = lambda fn: fn
            logging.warning(
                "Keeping torch.compile disabled for stage3 bf16 RDT training "
                "stability."
            )
        elif self.config.shared.enable_torch_compile:
            if self.config.policy_model.model_type == "sparse_moe":
                self.compile_mode = torch.compile(fullgraph=True)
            else:
                self.compile_mode = torch.compile(fullgraph=True, mode="max-autotune")
        else:
            self.compile_mode = lambda fn: fn
            logging.warning(
                "torch.compile is disabled by config.shared.enable_torch_compile; using eager execution for stage3 training."
            )

        self.transformer_forward_function = self.bc_transformer.forward

    def _get_transformer_mask_fn(self):
        # Use the default, causal mask.
        return None

    def configure_optimizers(self):
        assert not self.inference_mode
        use_fused_adamw = not self._force_stable_rdt_training
        if not use_fused_adamw:
            logging.warning(
                "Disabling fused AdamW for stage3 bf16 stability so Lightning gradient clipping can run."
            )
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.config.stage3_finetune.optim.learning_rate,
            betas=(
                self.config.stage3_finetune.optim.beta_1,
                self.config.stage3_finetune.optim.beta_2,
            ),
            weight_decay=self.config.stage3_finetune.optim.weight_decay,
            fused=use_fused_adamw,
        )
        return [optimizer]

    def on_train_batch_start(self, batch, batch_idx):
        global_step = self.trainer.global_step
        freeze_steps = self.config.stage3_finetune.freeze_transformer_layers_for_steps
        # This will get called multiple times if gradient accumulation is used.
        if (
            global_step == 0
            and freeze_steps > 0
            and not self._already_frozen
        ):
            logging.warning("Freezing transformer layers.")
            for param in self.bc_transformer.parameters():
                param.requires_grad = False
            for param in self.image_tokenizer.parameters():
                param.requires_grad = False
            self._already_frozen = True
        elif (
            freeze_steps > 0
            and global_step == freeze_steps
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
    _maybe_initialize_from_origin_checkpoint(model, config)

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
            model_name_or_path=self.cfg.shared.text_tokenizer_config.model_name_or_path,
            max_position_embeddings=self.cfg.shared.text_tokenizer_config.max_position_embeddings,
            text_embedding_shape=self.cfg.shared.text_tokenizer_config.text_embedding_shape,
            text_annotation_model_version=self.cfg.shared.text_tokenizer_config.text_annotation_model_version,
        )
        self.training_dataset_cfg.text_tokenizer_config = self.text_tokenizer_config
        for validation_dataset_cfg in self.validation_dataset_cfgs:
            validation_dataset_cfg.text_tokenizer_config = self.text_tokenizer_config

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
        MultiLatentLeRobotDataset, _ = _load_lerobot_dataset_support()
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
        MultiLatentLeRobotDataset, _ = _load_lerobot_dataset_support()
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


def _probe_requested_gpu_indices(requested_indices: list[int]) -> tuple[list[int], dict[int, str]]:
    probe_code = """
import torch

torch.cuda.init()
torch.cuda.set_device(0)
x = torch.empty(1, device='cuda')
print('probe_ok', x.device)
"""
    usable_indices: list[int] = []
    failed_indices: dict[int, str] = {}
    for device_idx in requested_indices:
        env = os.environ.copy()
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["CUDA_VISIBLE_DEVICES"] = str(device_idx)
        env["ELEFANT_STAGE3_GPU_PROBE"] = "1"
        result = subprocess.run(
            [sys.executable, "-c", probe_code],
            capture_output=True,
            text=True,
            env=env,
        )
        if result.returncode == 0:
            usable_indices.append(device_idx)
            continue
        stderr = (result.stderr or result.stdout or "probe failed").strip().splitlines()
        failed_indices[device_idx] = stderr[-1] if stderr else "probe failed"
    return usable_indices, failed_indices


def _query_gpu_inventory_via_nvidia_smi() -> tuple[list[dict], set[str]] | None:
    if shutil.which("nvidia-smi") is None:
        return None

    try:
        query = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        compute_query = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        logging.warning(
            "Failed to query GPUs via nvidia-smi (%s); leaving Lightning GPU selection on auto.",
            exc,
        )
        return None

    busy_gpu_uuids = {
        row.strip()
        for row in compute_query.stdout.splitlines()
        if row.strip() and not row.lower().startswith("no running")
    }
    gpu_rows = [row.strip() for row in query.stdout.splitlines() if row.strip()]
    inventory = []
    for row in gpu_rows:
        index_str, gpu_uuid, free_mb_str, total_mb_str = [part.strip() for part in row.split(",")]
        inventory.append(
            {
                "index": int(index_str),
                "uuid": gpu_uuid,
                "free_mb": int(free_mb_str),
                "total_mb": int(total_mb_str),
            }
        )
    return inventory, busy_gpu_uuids


def _select_training_gpus() -> tuple[str, int | str]:
    inventory_result = _query_gpu_inventory_via_nvidia_smi()

    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        visible_devices = [
            part.strip()
            for part in os.environ["CUDA_VISIBLE_DEVICES"].split(",")
            if part.strip()
        ]
        requested_indices = [int(device) for device in visible_devices]
        if inventory_result is not None:
            inventory, busy_gpu_uuids = inventory_result
            inventory_by_index = {item["index"]: item for item in inventory}
            requested_stats = []
            for device_idx in requested_indices:
                item = inventory_by_index.get(device_idx)
                if item is None:
                    requested_stats.append(f"gpu{device_idx}: unknown")
                    continue
                free_gib = item["free_mb"] / 1024.0
                total_gib = item["total_mb"] / 1024.0
                free_ratio = item["free_mb"] / max(item["total_mb"], 1)
                has_compute_process = item["uuid"] in busy_gpu_uuids
                status = "busy" if has_compute_process else "idle"
                requested_stats.append(
                    f"gpu{device_idx}: {status}, free={free_gib:.1f}GiB/{total_gib:.1f}GiB ({free_ratio:.0%})"
                )
            logging.info(
                "Requested CUDA_VISIBLE_DEVICES inventory: %s",
                ", ".join(requested_stats),
            )

        usable_indices, failed_indices = _probe_requested_gpu_indices(requested_indices)
        if usable_indices and len(usable_indices) != len(requested_indices):
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, usable_indices))
            logging.warning(
                "Requested CUDA_VISIBLE_DEVICES=%s contains GPUs that failed a live CUDA probe; using subset %s. Failures: %s",
                ",".join(map(str, requested_indices)),
                usable_indices,
                failed_indices,
            )
            visible_devices = [str(idx) for idx in usable_indices]
        elif not usable_indices:
            raise RuntimeError(
                "None of the requested CUDA_VISIBLE_DEVICES passed a live CUDA probe. "
                f"Requested={requested_indices}, failures={failed_indices}"
            )

        logging.info(
            "Respecting existing CUDA_VISIBLE_DEVICES=%s for stage3 training.",
            os.environ["CUDA_VISIBLE_DEVICES"],
        )
        return "gpu", max(len(visible_devices), 1)

    if inventory_result is None:
        logging.warning(
            "nvidia-smi is unavailable; leaving Lightning GPU selection on auto."
        )
        return "auto", "auto"

    inventory, busy_gpu_uuids = inventory_result
    if not inventory:
        return "auto", "auto"

    free_gpu_indices: list[int] = []
    gpu_stats: list[str] = []
    for item in inventory:
        device_idx = item["index"]
        free_mb = item["free_mb"]
        total_mb = item["total_mb"]
        free_gib = free_mb / 1024.0
        total_gib = total_mb / 1024.0
        free_ratio = free_mb / max(total_mb, 1)
        has_compute_process = item["uuid"] in busy_gpu_uuids
        status = "busy" if has_compute_process else "idle"
        gpu_stats.append(
            f"gpu{device_idx}: {status}, free={free_gib:.1f}GiB/{total_gib:.1f}GiB ({free_ratio:.0%})"
        )
        if (not has_compute_process) and free_gib >= 8.0 and free_ratio >= 0.5:
            free_gpu_indices.append(device_idx)

    if gpu_stats:
        logging.info(
            "Stage3 GPU availability before trainer creation: %s",
            ", ".join(gpu_stats),
        )

    if not free_gpu_indices:
        logging.warning(
            "No clearly free GPUs detected; leaving Lightning GPU selection on auto."
        )
        return "auto", "auto"

    if len(free_gpu_indices) == len(inventory):
        return "gpu", len(free_gpu_indices)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, free_gpu_indices))
    logging.warning(
        "Auto-selected currently available GPUs %s for stage3 training and exported CUDA_VISIBLE_DEVICES=%s.",
        free_gpu_indices,
        os.environ["CUDA_VISIBLE_DEVICES"],
    )
    return "gpu", len(free_gpu_indices)


def _init_stage3_model_for_trainer(
    trainer: pl.Trainer,
    config: LightningPolicyConfig,
):
    try:
        with trainer.init_module():
            return _init_stage3_model(config)
    except torch.AcceleratorError as exc:
        logging.warning(
            "trainer.init_module() hit a CUDA startup error (%s); falling back to CPU model construction before Lightning moves the model to devices.",
            exc,
        )
        return _init_stage3_model(config)


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

    accelerator, devices = _select_training_gpus()
    selected_gpu_count = devices if isinstance(devices, int) else 1

    if accelerator == "gpu" and selected_gpu_count > 1:
        logging.info(f"Using DDP strategy with {selected_gpu_count} GPUs")
        freeze_steps = config.stage3_finetune.freeze_transformer_layers_for_steps
        if freeze_steps == 0:
            logging.info(
                "Using static-graph DDP for stage3 training: this model uses a fixed "
                "parameter graph, and disabling unused-parameter discovery avoids "
                "NCCL bucket/broadcast mismatches under torch.compile."
            )
            strategy = pl.pytorch.strategies.DDPStrategy(
                find_unused_parameters=False,
                static_graph=True,
                gradient_as_bucket_view=True,
            )
        else:
            logging.info(
                "Keeping dynamic DDP because freeze_transformer_layers_for_steps=%s "
                "changes parameter participation during training.",
                freeze_steps,
            )
            strategy = pl.pytorch.strategies.DDPStrategy(
                find_unused_parameters=True,
            )
    else:
        logging.info("Using SingleDeviceStrategy for single GPU.")
        # strategy = pl.pytorch.strategies.SingleDeviceStrategy(accelerator="auto")
        # Setting strategy with single GPU explicitly seems to error.
        # https://github.com/Lightning-AI/pytorch-lightning/issues/18902
        strategy = "auto"

    trainer = pl.Trainer(
        plugins=[async_checkpointer],
        callbacks=[checkpoint_callback],
        accelerator=accelerator,
        # For debugging it can be useful to set devices to 1.
        # for simpler stack traces etc.
        devices=devices,
        max_steps=config.stage3_finetune.n_training_steps,
        logger=wandb_logger,
        # We multiply by accumulate_grad_batches to get the number of steps between validation steps in "real" steps.
        val_check_interval=config.stage3_finetune.validation_step_interval
        * config.stage3_finetune.accumulate_grad_batches,
        limit_val_batches=config.stage3_finetune.n_validation_steps,
        check_val_every_n_epoch=None,
        precision=config.shared.precision,
        accumulate_grad_batches=config.stage3_finetune.accumulate_grad_batches,
        gradient_clip_algorithm="norm",
        gradient_clip_val=1.0 if "bf16" in str(config.shared.precision).lower() else 0.0,
        fast_dev_run=config.shared.fast_dev_run,
        strategy=strategy,
        # We already run validation before training starts.
        num_sanity_val_steps=0,
        # profiler="simple",
    )

    # Prefer Lightning-managed device-aware init, but fall back to plain CPU
    # construction if CUDA is temporarily unavailable during startup.
    model = _init_stage3_model_for_trainer(trainer, config)

    total_params, expert_params = count_model_parameters(model)
    logging.info(
        f"Total parameters: {total_params}, Expert parameters: {expert_params}"
    )

    if resume_ckpt_path is not None:
        logging.info("Resuming stage3 training from checkpoint: %s", resume_ckpt_path)

    trainer.fit(model, datamodule, ckpt_path=resume_ckpt_path)

    wandb_logger.experiment.finish()
    return async_checkpointer.get_final_checkpoint()

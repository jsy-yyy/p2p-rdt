from __future__ import annotations

import argparse
import logging
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import lightning as pl
import numpy as np
import torch

from elefant.config import load_config
from elefant.policy_model.config import LightningPolicyConfig
from elefant.policy_model.stage3_finetune import (
    Stage3LabelledBCLightning,
    _validate_resume_checkpoint_compatibility,
    count_model_parameters,
)
from elefant.text_tokenizer.factory import get_text_tokenizer

from .common import (
    RobotActionAdapter,
    RobotTwinObservationAdapter,
    compose_compact_relative_action,
    resolve_checkpoint_path,
)
from .websocket_policy_server import WebsocketPolicyServer

LOGGER = logging.getLogger(__name__)


def _init_text_tokenizer(config: LightningPolicyConfig):
    text_tokenizer_cfg = getattr(config.shared, "text_tokenizer_config", None)
    if text_tokenizer_cfg is None:
        return None

    try:
        return get_text_tokenizer(text_tokenizer_cfg)
    except Exception as exc:
        LOGGER.warning(
            "Failed to initialize text tokenizer %s; falling back to zero text embeddings and ignoring prompts. Error: %s",
            getattr(text_tokenizer_cfg, "text_tokenizer_name", None),
            exc,
        )
        return None


def _resolve_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def _resolve_dtype(config: LightningPolicyConfig, device: torch.device) -> torch.dtype:
    precision = str(config.shared.precision).lower()
    if device.type == "cuda" and "bf16" in precision:
        return torch.bfloat16
    return torch.float32


def _build_trainer(config: LightningPolicyConfig, device: torch.device) -> pl.Trainer:
    if device.type == "cuda":
        return pl.Trainer(
            precision=config.shared.precision,
            accelerator="gpu",
            devices=[device.index if device.index is not None else 0],
            logger=False,
            enable_checkpointing=False,
        )
    return pl.Trainer(
        precision="32-true",
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
    )


def _checkpoint_map_location(device: torch.device) -> str:
    if device.type == "cuda":
        return "cpu"
    return str(device)


def _reset_transformer_decode_mask_caches(module: torch.nn.Module) -> None:
    for submodule in module.modules():
        if hasattr(submodule, "_cached_decode_mask"):
            submodule._cached_decode_mask = None
        if hasattr(submodule, "stable_start"):
            submodule.stable_start = None


@dataclass
class RobotTwinSequenceInferenceState:
    model: Stage3LabelledBCLightning
    config: LightningPolicyConfig
    device: torch.device
    compile: bool
    text_tokenizer: Any
    action_adapter: RobotActionAdapter

    def __post_init__(self) -> None:
        self.seq_len = int(self.config.shared.n_seq_timesteps)
        self.action_dim = int(
            self.config.stage3_finetune.training_dataset.get_robot_action_dim()
        )
        self.text_tokens_embed = None
        self.current_prompt = ""
        self.frame_history: torch.Tensor | None = None
        self.pose_history = np.zeros((self.seq_len, 16), dtype=np.float32)
        self.anchor_pose_16d = np.zeros(16, dtype=np.float32)
        self.action_history = torch.zeros(
            1,
            self.seq_len,
            self.action_dim,
            device=self.device,
            dtype=torch.float32,
        )
        self.n_prior_frames = 0

    def reset(
        self,
        prompt: str | None = None,
        text_tokens_embed: torch.Tensor | np.ndarray | None = None,
    ) -> None:
        self.current_prompt = prompt or ""
        self.text_tokens_embed = self._resolve_text_tokens_embed(
            prompt=self.current_prompt,
            text_tokens_embed=text_tokens_embed,
        )
        self.frame_history = None
        self.pose_history.fill(0.0)
        self.anchor_pose_16d.fill(0.0)
        self.action_history.zero_()
        self.n_prior_frames = 0

    def _encode_prompt(self, prompt: str):
        if self.text_tokenizer is None:
            return None
        tokenized = self.text_tokenizer.tokenize(prompt or "")
        with torch.inference_mode():
            return self.text_tokenizer(**tokenized)

    def _coerce_text_tokens_embed(self, text_tokens_embed):
        if text_tokens_embed is None:
            return None

        text_tokens_embed = torch.as_tensor(
            text_tokens_embed,
            device=self.device,
            dtype=torch.float32,
        )
        if text_tokens_embed.ndim == 1:
            text_tokens_embed = text_tokens_embed.unsqueeze(0)
        return text_tokens_embed

    def _resolve_text_tokens_embed(
        self,
        prompt: str | None = None,
        text_tokens_embed: torch.Tensor | np.ndarray | None = None,
    ):
        if text_tokens_embed is not None:
            return self._coerce_text_tokens_embed(text_tokens_embed)
        return self._encode_prompt(prompt or "")

    def _ensure_frame_history(self, frame: torch.Tensor) -> None:
        expected_shape = (1, self.seq_len, *frame.shape)
        if self.frame_history is None or tuple(self.frame_history.shape) != expected_shape:
            self.frame_history = torch.zeros(
                expected_shape,
                device=self.device,
                dtype=torch.uint8,
            )

    def _make_identity_relative_action(self, current_pose_16d: np.ndarray) -> np.ndarray:
        current_pose_16d = np.asarray(current_pose_16d, dtype=np.float32).reshape(16)
        compact_action = np.zeros(16, dtype=np.float32)
        compact_action[3:7] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        compact_action[7] = current_pose_16d[7]
        compact_action[11:15] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        compact_action[15] = current_pose_16d[15]
        full_action = self.action_adapter.compact_to_full_action(compact_action)
        return self.action_adapter.normalize(full_action).astype(np.float32)

    def _rebase_action_history_after_roll(self) -> None:
        if self.seq_len <= 1:
            return
        reference_action = self.action_history[0, 0].detach().cpu().numpy()
        history_to_rebase = self.action_history[0, : self.seq_len - 1].detach().cpu().numpy()
        rebased_history = self.action_adapter.rebase_normalized_actions(
            history_to_rebase,
            reference_action,
        )
        self.action_history[0, : self.seq_len - 1] = torch.from_numpy(rebased_history).to(
            device=self.device,
            dtype=torch.float32,
        )
        self.action_history[0, -1].zero_()

    def step(
        self,
        frame: torch.Tensor,
        current_pose_16d: np.ndarray,
        prompt: str | None = None,
        text_tokens_embed: torch.Tensor | np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        current_pose_16d = np.asarray(current_pose_16d, dtype=np.float32).reshape(16)
        if text_tokens_embed is not None:
            self.current_prompt = prompt or self.current_prompt
            self.text_tokens_embed = self._coerce_text_tokens_embed(text_tokens_embed)
        elif prompt is not None and prompt != self.current_prompt:
            self.current_prompt = prompt
            self.text_tokens_embed = self._encode_prompt(prompt)

        self._ensure_frame_history(frame)
        if self.n_prior_frames < self.seq_len:
            frame_index = self.n_prior_frames
            self.frame_history[0, frame_index] = frame
            self.pose_history[frame_index] = current_pose_16d
            self.n_prior_frames += 1
        else:
            self.frame_history = torch.roll(self.frame_history, shifts=-1, dims=1)
            self.action_history = torch.roll(self.action_history, shifts=-1, dims=1)
            self.pose_history = np.roll(self.pose_history, shift=-1, axis=0)
            self.frame_history[0, -1] = frame
            self.pose_history[-1] = current_pose_16d
            self.action_history[0, -1].zero_()
            self._rebase_action_history_after_roll()
            frame_index = self.seq_len - 1

        self.anchor_pose_16d = self.pose_history[0].copy()

        if self.n_prior_frames == 1:
            normalized_action = self._make_identity_relative_action(current_pose_16d)
            self.action_history[0, frame_index] = torch.from_numpy(normalized_action).to(
                device=self.device,
                dtype=torch.float32,
            )
            return normalized_action, self.anchor_pose_16d.copy()

        with torch.inference_mode():
            try:
                action_preds = self.model.online_full_predict(
                    frames=self.frame_history,
                    actions=self.action_history,
                    text_tokens_embed=self.text_tokens_embed,
                    compile=self.compile,
                )
            except torch.OutOfMemoryError:
                if not self.compile or self.device.type != "cuda":
                    raise
                LOGGER.warning(
                    "CUDA OOM during compiled inference; clearing cache and retrying once with compile disabled."
                )
                torch.cuda.empty_cache()
                self.compile = False
                action_preds = self.model.online_full_predict(
                    frames=self.frame_history,
                    actions=self.action_history,
                    text_tokens_embed=self.text_tokens_embed,
                    compile=False,
                )
        action = action_preds[0, frame_index].detach().to(torch.float32)
        self.action_history[0, frame_index] = action
        return action.cpu().numpy(), self.anchor_pose_16d.copy()


class OpenP2PRobotTwinPolicy:
    def __init__(
        self,
        config: LightningPolicyConfig,
        checkpoint_path: str,
        device: torch.device,
        compile_model: bool,
    ) -> None:
        self.config = config
        self.device = device
        self.dtype = _resolve_dtype(config, device)
        self.checkpoint_path = checkpoint_path
        checkpoint_map_location = _checkpoint_map_location(device)
        init_context = nullcontext()

        LOGGER.info("Loading checkpoint from %s", checkpoint_path)
        _validate_resume_checkpoint_compatibility(checkpoint_path, config)
        if checkpoint_map_location == "cpu" and device.type == "cuda":
            LOGGER.info(
                "Staging checkpoint on CPU before moving the model to %s to avoid CUDA load-time OOM.",
                device,
            )
        with init_context:
            self.model = Stage3LabelledBCLightning.load_from_checkpoint(
                checkpoint_path,
                config=config,
                inference_mode=True,
                map_location=checkpoint_map_location,
            )
        if device.type == "cuda":
            torch.cuda.empty_cache()
        try:
            self.model = self.model.to(device=device, dtype=self.dtype).eval()
            if hasattr(self.model, "bc_transformer") and hasattr(self.model.bc_transformer, "block_mask_to_device"):
                self.model.bc_transformer.block_mask_to_device(device)
            _reset_transformer_decode_mask_caches(self.model)
        except torch.OutOfMemoryError as exc:
            if device.type != "cuda":
                raise
            torch.cuda.empty_cache()
            raise RuntimeError(
                f"Checkpoint loaded from CPU staging, but moving the model to {device} still ran out of CUDA memory. "
                "This usually means the model does not fit on the selected GPU for inference."
            ) from exc

        total_params, expert_params = count_model_parameters(self.model)
        LOGGER.info(
            "Loaded model with total_params=%s expert_params=%s device=%s dtype=%s",
            total_params,
            expert_params,
            device,
            self.dtype,
        )

        self.text_tokenizer = _init_text_tokenizer(config)
        self.observation_adapter = RobotTwinObservationAdapter(
            config.stage3_finetune.training_dataset
        )
        self.action_adapter = RobotActionAdapter(config.stage3_finetune.training_dataset)
        self.state = RobotTwinSequenceInferenceState(
            model=self.model,
            config=config,
            device=device,
            compile=compile_model and device.type == "cuda",
            text_tokenizer=self.text_tokenizer,
            action_adapter=self.action_adapter,
        )
        self.state.reset("")

    def get_metadata(self) -> dict[str, Any]:
        return {
            "checkpoint_path": self.checkpoint_path,
            "device": str(self.device),
            "dtype": str(self.dtype),
            "sequence_length": int(self.config.shared.n_seq_timesteps),
            "robot_action_dim": int(
                self.config.stage3_finetune.training_dataset.get_robot_action_dim()
            ),
            "used_action_channel_ids": list(
                self.config.stage3_finetune.training_dataset.used_action_channel_ids
            ),
            "text_embedding_shape": list(
                self.config.shared.text_tokenizer_config.text_embedding_shape
            ),
            "text_tokenizer_available": self.text_tokenizer is not None,
            **self.observation_adapter.describe(),
        }

    def warmup(self) -> None:
        dummy_obs = {
            key: np.zeros(
                (
                    self.observation_adapter.image_height,
                    self.observation_adapter.image_width,
                    3,
                ),
                dtype=np.uint8,
            )
            for key in self.observation_adapter.used_video_keys
        }
        dummy_pose = np.array(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0] * 2,
            dtype=np.float32,
        )
        self.state.reset("warmup")
        _ = self.infer({"obs": dummy_obs, "prompt": "warmup", "ee_pose_16d": dummy_pose})
        self.state.reset("")

    def infer(self, request: dict[str, Any]) -> dict[str, Any]:
        text_tokens_embed = request.get("text_emb")
        if text_tokens_embed is None:
            text_tokens_embed = request.get("text_embeddings")

        if request.get("reset"):
            self.state.reset(request.get("prompt"), text_tokens_embed=text_tokens_embed)
            return {
                "status": "ok",
                "prompt": self.state.current_prompt,
                "used_text_embedding": text_tokens_embed is not None,
            }

        obs = request.get("obs")
        if obs is None:
            raise ValueError("Request must provide `obs` or set `reset=True`.")

        current_pose_16d = request.get("ee_pose_16d")
        if current_pose_16d is None:
            raise ValueError("Request must provide `ee_pose_16d` for RoboTwin evaluation.")

        prompt = request.get("prompt")
        frame = self.observation_adapter.prepare_frame(obs).to(
            self.device,
            non_blocking=self.device.type == "cuda",
        )
        normalized_action, anchor_pose_16d = self.state.step(
            frame,
            current_pose_16d=current_pose_16d,
            prompt=prompt,
            text_tokens_embed=text_tokens_embed,
        )
        full_action = self.action_adapter.denormalize(normalized_action)
        compact_action = self.action_adapter.to_compact_execution_action(full_action)
        absolute_compact_action = compose_compact_relative_action(
            compact_action,
            anchor_pose_16d,
        )
        return {
            "action": full_action.astype(np.float32),
            "action_16d": compact_action.astype(np.float32),
            "action_16d_absolute": absolute_compact_action.astype(np.float32),
            "action_reference_16d": anchor_pose_16d.astype(np.float32),
            "normalized_action": normalized_action.astype(np.float32),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve an Open P2P RoboTwin policy over websocket.")
    parser.add_argument("--config", required=True, help="Path to model_config.yaml")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path. Defaults to latest checkpoint next to model_config.yaml.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=29056)
    parser.add_argument("--device", default=None, help="Torch device, for example cuda:0.")
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile during online inference.")
    parser.add_argument("--skip-warmup", action="store_true", help="Skip one dummy warmup request before serving.")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    torch.set_float32_matmul_precision("high")

    config = load_config(args.config, LightningPolicyConfig)
    checkpoint_path = resolve_checkpoint_path(args.config, args.checkpoint)
    device = _resolve_device(args.device)
    policy = OpenP2PRobotTwinPolicy(
        config=config,
        checkpoint_path=checkpoint_path,
        device=device,
        compile_model=not args.no_compile,
    )
    if not args.skip_warmup:
        LOGGER.info("Running one warmup request before serving.")
        policy.warmup()

    metadata = policy.get_metadata()
    LOGGER.info("Starting websocket server on %s:%s", args.host, args.port)
    WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        metadata=metadata,
    ).serve_forever()


if __name__ == "__main__":
    main()

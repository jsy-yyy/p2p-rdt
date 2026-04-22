from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R
import torch
import torch.nn.functional as F


def normalize_obs_cam_keys(obs_cam_keys: list[str]) -> list[str]:
    return [
        key if key.startswith("observation.images.") else f"observation.images.{key}"
        for key in obs_cam_keys
    ]


def resolve_checkpoint_path(config_path: str, checkpoint_path: str | None) -> str:
    if checkpoint_path:
        path = Path(checkpoint_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return str(path)

    config_file = Path(config_path).expanduser().resolve()
    checkpoint_candidates = sorted(config_file.parent.glob("checkpoint-step=*.ckpt"))
    if not checkpoint_candidates:
        raise FileNotFoundError(
            f"No checkpoint-step=*.ckpt found next to model config: {config_file}"
        )
    return str(checkpoint_candidates[-1])


def _expand_norm_stat(
    stat_values: list[float],
    robot_action_dim: int,
    used_action_channel_ids: list[int],
    default_value: float,
    stat_name: str,
) -> np.ndarray:
    values = np.asarray(stat_values, dtype=np.float32)
    if values.ndim != 1:
        raise ValueError(f"{stat_name} must be 1D, got shape {values.shape}.")

    if values.size == robot_action_dim:
        return values

    if used_action_channel_ids and values.size == len(used_action_channel_ids):
        expanded = np.full(robot_action_dim, default_value, dtype=np.float32)
        for idx, action_id in enumerate(used_action_channel_ids):
            expanded[action_id] = values[idx]
        return expanded

    raise ValueError(
        f"{stat_name} must have length {robot_action_dim} or {len(used_action_channel_ids)}, got {values.size}."
    )


def _normalize_quaternion(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32).reshape(4)
    if not np.all(np.isfinite(quat)):
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    norm = np.linalg.norm(quat)
    if not np.isfinite(norm) or norm < 1e-8:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return quat / norm


def sanitize_pose_8d(
    pose_8d: np.ndarray,
    fallback_pose_8d: np.ndarray | None = None,
) -> tuple[np.ndarray, bool]:
    pose_8d = np.asarray(pose_8d, dtype=np.float32).reshape(8).copy()
    if fallback_pose_8d is None:
        fallback_pose_8d = np.zeros(8, dtype=np.float32)
        fallback_pose_8d[3:7] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    else:
        fallback_pose_8d = np.asarray(fallback_pose_8d, dtype=np.float32).reshape(8)

    changed = False

    if not np.all(np.isfinite(pose_8d[:3])):
        pose_8d[:3] = fallback_pose_8d[:3]
        changed = True

    quat = pose_8d[3:7]
    quat_is_invalid = (not np.all(np.isfinite(quat))) or float(
        np.linalg.norm(np.nan_to_num(quat, nan=0.0, posinf=0.0, neginf=0.0))
    ) < 1e-8
    if quat_is_invalid:
        pose_8d[3:7] = _normalize_quaternion(fallback_pose_8d[3:7])
        changed = True
    else:
        normalized_quat = _normalize_quaternion(quat)
        if not np.allclose(normalized_quat, quat, atol=1e-5, rtol=1e-5):
            pose_8d[3:7] = normalized_quat
            changed = True

    if not np.all(np.isfinite(pose_8d[7:8])):
        pose_8d[7:8] = fallback_pose_8d[7:8]
        changed = True

    return pose_8d.astype(np.float32), changed


def sanitize_compact_pose_16d(
    pose_16d: np.ndarray,
    fallback_pose_16d: np.ndarray | None = None,
) -> tuple[np.ndarray, bool]:
    pose_16d = np.asarray(pose_16d, dtype=np.float32).reshape(16)
    if fallback_pose_16d is None:
        fallback_pose_16d = np.zeros(16, dtype=np.float32)
        fallback_pose_16d[3:7] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        fallback_pose_16d[11:15] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    else:
        fallback_pose_16d = np.asarray(fallback_pose_16d, dtype=np.float32).reshape(16)

    left, left_changed = sanitize_pose_8d(pose_16d[:8], fallback_pose_16d[:8])
    right, right_changed = sanitize_pose_8d(pose_16d[8:], fallback_pose_16d[8:])
    return np.concatenate([left, right]).astype(np.float32), left_changed or right_changed


def compose_relative_pose(
    relative_pose: np.ndarray,
    anchor_pose: np.ndarray,
) -> np.ndarray:
    relative_pose = np.asarray(relative_pose, dtype=np.float32).reshape(8)
    anchor_pose = np.asarray(anchor_pose, dtype=np.float32).reshape(8)
    relative_rotation = R.from_quat(_normalize_quaternion(relative_pose[3:7])[None])
    anchor_rotation = R.from_quat(_normalize_quaternion(anchor_pose[3:7])[None])
    absolute_rotation = (anchor_rotation * relative_rotation).as_quat().reshape(-1)
    absolute_rotation = _normalize_quaternion(absolute_rotation)
    absolute_translation = relative_pose[:3] + anchor_pose[:3]
    return np.concatenate(
        [absolute_translation, absolute_rotation, relative_pose[7:8]]
    ).astype(np.float32)


def absolute_to_relative_pose(
    absolute_pose: np.ndarray,
    anchor_pose: np.ndarray,
) -> np.ndarray:
    absolute_pose = np.asarray(absolute_pose, dtype=np.float32).reshape(8)
    anchor_pose = np.asarray(anchor_pose, dtype=np.float32).reshape(8)
    absolute_rotation = R.from_quat(_normalize_quaternion(absolute_pose[3:7])[None])
    anchor_rotation = R.from_quat(_normalize_quaternion(anchor_pose[3:7])[None])
    relative_rotation = (anchor_rotation.inv() * absolute_rotation).as_quat().reshape(-1)
    relative_rotation = _normalize_quaternion(relative_rotation)
    relative_translation = absolute_pose[:3] - anchor_pose[:3]
    return np.concatenate(
        [relative_translation, relative_rotation, absolute_pose[7:8]]
    ).astype(np.float32)


def rebase_relative_pose(
    relative_pose: np.ndarray,
    new_anchor_relative_pose: np.ndarray,
) -> np.ndarray:
    relative_pose = np.asarray(relative_pose, dtype=np.float32).reshape(8)
    new_anchor_relative_pose = np.asarray(
        new_anchor_relative_pose, dtype=np.float32
    ).reshape(8)
    relative_rotation = R.from_quat(_normalize_quaternion(relative_pose[3:7])[None])
    anchor_rotation = R.from_quat(
        _normalize_quaternion(new_anchor_relative_pose[3:7])[None]
    )
    rebased_rotation = (anchor_rotation.inv() * relative_rotation).as_quat().reshape(-1)
    rebased_rotation = _normalize_quaternion(rebased_rotation)
    rebased_translation = relative_pose[:3] - new_anchor_relative_pose[:3]
    return np.concatenate(
        [rebased_translation, rebased_rotation, relative_pose[7:8]]
    ).astype(np.float32)


def compose_compact_relative_action(
    action_16d: np.ndarray,
    anchor_pose_16d: np.ndarray,
) -> np.ndarray:
    action_16d = np.asarray(action_16d, dtype=np.float32).reshape(16)
    anchor_pose_16d = np.asarray(anchor_pose_16d, dtype=np.float32).reshape(16)
    left = compose_relative_pose(action_16d[:8], anchor_pose_16d[:8])
    right = compose_relative_pose(action_16d[8:], anchor_pose_16d[8:])
    return np.concatenate([left, right]).astype(np.float32)


def absolute_to_relative_compact_action(
    action_16d: np.ndarray,
    anchor_pose_16d: np.ndarray,
) -> np.ndarray:
    action_16d = np.asarray(action_16d, dtype=np.float32).reshape(16)
    anchor_pose_16d = np.asarray(anchor_pose_16d, dtype=np.float32).reshape(16)
    left = absolute_to_relative_pose(action_16d[:8], anchor_pose_16d[:8])
    right = absolute_to_relative_pose(action_16d[8:], anchor_pose_16d[8:])
    return np.concatenate([left, right]).astype(np.float32)


def rebase_compact_relative_action(
    action_16d: np.ndarray,
    new_anchor_action_16d: np.ndarray,
) -> np.ndarray:
    action_16d = np.asarray(action_16d, dtype=np.float32).reshape(16)
    new_anchor_action_16d = np.asarray(new_anchor_action_16d, dtype=np.float32).reshape(
        16
    )
    left = rebase_relative_pose(action_16d[:8], new_anchor_action_16d[:8])
    right = rebase_relative_pose(action_16d[8:], new_anchor_action_16d[8:])
    return np.concatenate([left, right]).astype(np.float32)


def _slerp_quaternion(
    start_quat: np.ndarray,
    end_quat: np.ndarray,
    alpha: float,
) -> np.ndarray:
    start_quat = _normalize_quaternion(start_quat)
    end_quat = _normalize_quaternion(end_quat)
    dot = float(np.dot(start_quat, end_quat))
    if dot < 0.0:
        end_quat = -end_quat
        dot = -dot

    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        blended = (1.0 - alpha) * start_quat + alpha * end_quat
        return _normalize_quaternion(blended)

    theta_0 = float(np.arccos(dot))
    sin_theta_0 = float(np.sin(theta_0))
    theta = theta_0 * alpha
    sin_theta = float(np.sin(theta))
    s0 = float(np.sin(theta_0 - theta) / sin_theta_0)
    s1 = float(sin_theta / sin_theta_0)
    return _normalize_quaternion((s0 * start_quat) + (s1 * end_quat))


def _blend_absolute_pose(
    previous_pose: np.ndarray,
    current_pose: np.ndarray,
    alpha: float,
) -> np.ndarray:
    previous_pose = np.asarray(previous_pose, dtype=np.float32).reshape(8)
    current_pose = np.asarray(current_pose, dtype=np.float32).reshape(8)
    blended_translation = ((1.0 - alpha) * previous_pose[:3]) + (alpha * current_pose[:3])
    blended_rotation = _slerp_quaternion(previous_pose[3:7], current_pose[3:7], alpha)
    blended_gripper = ((1.0 - alpha) * previous_pose[7:8]) + (alpha * current_pose[7:8])
    return np.concatenate(
        [blended_translation, blended_rotation, blended_gripper]
    ).astype(np.float32)


def smooth_absolute_compact_action(
    previous_action_16d: np.ndarray,
    current_action_16d: np.ndarray,
    alpha: float,
) -> np.ndarray:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    previous_action_16d = np.asarray(previous_action_16d, dtype=np.float32).reshape(16)
    current_action_16d = np.asarray(current_action_16d, dtype=np.float32).reshape(16)
    if alpha <= 0.0:
        return previous_action_16d.copy()
    if alpha >= 1.0:
        return current_action_16d.copy()

    left = _blend_absolute_pose(previous_action_16d[:8], current_action_16d[:8], alpha)
    right = _blend_absolute_pose(previous_action_16d[8:], current_action_16d[8:], alpha)
    return np.concatenate([left, right]).astype(np.float32)


class RobotActionAdapter:
    """Translate model outputs back into RoboTwin execution actions."""

    def __init__(self, dataset_config: Any) -> None:
        self.robot_action_dim = int(dataset_config.get_robot_action_dim())
        self.used_action_channel_ids = list(
            getattr(dataset_config, "used_action_channel_ids", [])
        )
        if not self.used_action_channel_ids:
            raise ValueError(
                "training_dataset.used_action_channel_ids is required for RoboTwin evaluation."
            )

        norm_stat = getattr(dataset_config, "norm_stat", {}) or {}
        self.q01 = _expand_norm_stat(
            norm_stat.get("q01", [0.0] * self.robot_action_dim),
            self.robot_action_dim,
            self.used_action_channel_ids,
            default_value=0.0,
            stat_name="norm_stat.q01",
        )
        self.q99 = _expand_norm_stat(
            norm_stat.get("q99", [1.0] * self.robot_action_dim),
            self.robot_action_dim,
            self.used_action_channel_ids,
            default_value=1.0,
            stat_name="norm_stat.q99",
        )

    def normalize(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32)
        scale = self.q99 - self.q01 + 1e-6
        return ((action - self.q01) / scale) * 2.0 - 1.0

    def denormalize(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.size != self.robot_action_dim:
            raise ValueError(
                f"Expected normalized action dim {self.robot_action_dim}, got {action.size}."
            )
        scale = self.q99 - self.q01 + 1e-6
        return ((action + 1.0) * 0.5) * scale + self.q01

    def compact_to_full_action(
        self,
        compact_action: np.ndarray,
        template_action: np.ndarray | None = None,
    ) -> np.ndarray:
        compact_action = np.asarray(compact_action, dtype=np.float32).reshape(-1)
        if compact_action.size != len(self.used_action_channel_ids):
            raise ValueError(
                f"Expected compact action dim {len(self.used_action_channel_ids)}, got {compact_action.size}."
            )

        if template_action is None:
            full_action = np.zeros(self.robot_action_dim, dtype=np.float32)
        else:
            full_action = np.asarray(template_action, dtype=np.float32).reshape(-1).copy()
            if full_action.size != self.robot_action_dim:
                raise ValueError(
                    f"Expected template action dim {self.robot_action_dim}, got {full_action.size}."
                )
        full_action[self.used_action_channel_ids] = compact_action
        return full_action

    def to_compact_execution_action(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        compact = action[self.used_action_channel_ids]
        if compact.size != len(self.used_action_channel_ids):
            raise ValueError("Failed to compact action to used channels.")
        return compact

    def rebase_normalized_actions(
        self,
        normalized_actions: np.ndarray,
        new_anchor_action: np.ndarray,
    ) -> np.ndarray:
        normalized_actions = np.asarray(normalized_actions, dtype=np.float32)
        squeeze = normalized_actions.ndim == 1
        if squeeze:
            normalized_actions = normalized_actions[None, :]

        new_anchor_action = np.asarray(new_anchor_action, dtype=np.float32).reshape(-1)
        if normalized_actions.shape[-1] != self.robot_action_dim:
            raise ValueError(
                f"Expected normalized action dim {self.robot_action_dim}, got {normalized_actions.shape[-1]}."
            )
        if new_anchor_action.size != self.robot_action_dim:
            raise ValueError(
                f"Expected new anchor action dim {self.robot_action_dim}, got {new_anchor_action.size}."
            )

        new_anchor_full = self.denormalize(new_anchor_action)
        new_anchor_compact = self.to_compact_execution_action(new_anchor_full)
        rebased_actions = []
        for normalized_action in normalized_actions:
            full_action = self.denormalize(normalized_action)
            compact_action = self.to_compact_execution_action(full_action)
            rebased_compact = rebase_compact_relative_action(
                compact_action,
                new_anchor_compact,
            )
            rebased_full = self.compact_to_full_action(
                rebased_compact,
                template_action=full_action,
            )
            rebased_actions.append(self.normalize(rebased_full))

        rebased = np.stack(rebased_actions, axis=0)
        if squeeze:
            return rebased[0]
        return rebased

    def rebase_normalized_actions_to_observed_anchor(
        self,
        normalized_actions: np.ndarray,
        previous_anchor_pose_16d: np.ndarray,
        new_anchor_pose_16d: np.ndarray,
    ) -> np.ndarray:
        new_anchor_compact = absolute_to_relative_compact_action(
            action_16d=new_anchor_pose_16d,
            anchor_pose_16d=previous_anchor_pose_16d,
        )
        new_anchor_full = self.compact_to_full_action(new_anchor_compact)
        new_anchor_normalized = self.normalize(new_anchor_full)
        return self.rebase_normalized_actions(
            normalized_actions=normalized_actions,
            new_anchor_action=new_anchor_normalized,
        )


class RobotTwinObservationAdapter:
    """Mirror the RoboTwin image preprocessing used during training."""

    def __init__(self, dataset_config: Any) -> None:
        self.used_video_keys = normalize_obs_cam_keys(list(dataset_config.obs_cam_keys))
        if not self.used_video_keys:
            raise ValueError("training_dataset.obs_cam_keys must not be empty.")

        self.image_height = int(dataset_config.image_height)
        self.image_width = int(dataset_config.image_width)
        self.multi_view_image_mode = getattr(
            dataset_config, "multi_view_image_mode", "vertical"
        )
        if self.multi_view_image_mode == "frame":
            raise NotImplementedError(
                "multi_view_image_mode='frame' is not supported by this online RoboTwin evaluator yet."
            )

    def describe(self) -> dict[str, Any]:
        return {
            "obs_cam_keys": self.used_video_keys,
            "image_height": self.image_height,
            "image_width": self.image_width,
            "multi_view_image_mode": self.multi_view_image_mode,
        }

    def prepare_frame(self, obs: dict[str, Any]) -> torch.Tensor:
        image_list = []
        for key in self.used_video_keys:
            if key not in obs:
                raise KeyError(
                    f"Observation is missing `{key}`. Available keys: {sorted(obs.keys())}"
                )
            image_list.append(self._resize_and_pad(self._to_chw_uint8(obs[key])))

        if self.multi_view_image_mode == "vertical":
            frame = torch.cat(image_list, dim=1)
        elif self.multi_view_image_mode == "first":
            frame = image_list[0]
        elif self.multi_view_image_mode == "token_concat":
            frame = torch.stack(image_list, dim=0)
        else:
            raise ValueError(
                f"Unsupported multi_view_image_mode: {self.multi_view_image_mode}"
            )
        return frame.contiguous()

    def _to_chw_uint8(self, image: Any) -> torch.Tensor:
        tensor = torch.as_tensor(image)
        if tensor.ndim != 3:
            raise ValueError(f"Expected image rank 3, got shape {tuple(tensor.shape)}.")
        if tensor.shape[0] == 3:
            chw = tensor
        else:
            chw = tensor.permute(2, 0, 1)
        if chw.dtype != torch.uint8:
            chw = chw.to(torch.float32).round().clamp(0, 255).to(torch.uint8)
        return chw

    def _resize_and_pad(self, image: torch.Tensor) -> torch.Tensor:
        _, height, width = image.shape
        scale = min(self.image_height / height, self.image_width / width)
        new_height = int(round(height * scale))
        new_width = int(round(width * scale))
        resized = F.interpolate(
            image.unsqueeze(0).float(),
            size=(new_height, new_width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        pad_h = self.image_height - new_height
        pad_w = self.image_width - new_width
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        padded = F.pad(
            resized,
            (pad_left, pad_right, pad_top, pad_bottom),
            mode="constant",
            value=0,
        )
        return padded.round().clamp(0, 255).to(torch.uint8)

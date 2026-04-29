# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import json
import numpy as np
from pathlib import Path
from collections.abc import Callable
import os
from tqdm import tqdm
from multiprocessing import get_context
from functools import partial
import torch
from einops import rearrange
from torch.utils.data import DataLoader
from scipy.spatial.transform import Rotation as R
import torch.nn.functional as F
import pyarrow.parquet as pq

import logging as logger
from elefant.data.action_label_video_proto_dataset import ActionLabelVideoDatasetItem
from elefant.text_tokenizer.config import TextTokenizerConfig
from elefant.text_tokenizer.factory import get_text_tokenizer


def _normalize_obs_cam_keys(obs_cam_keys):
    return [
        key if key.startswith('observation.images.') else f'observation.images.{key}'
        for key in obs_cam_keys
    ]


def _pool_text_embedding_tokens(text_emb: torch.Tensor) -> torch.Tensor:
    text_emb = torch.as_tensor(text_emb)
    if text_emb.ndim == 2 and text_emb.shape[0] > 1:
        return text_emb.mean(dim=0, keepdim=True)
    if text_emb.ndim == 3 and text_emb.shape[-2] > 1:
        return text_emb.mean(dim=-2, keepdim=True)
    return text_emb


def _infer_text_embedding_shape_from_value(text_emb) -> list[int]:
    text_emb = _pool_text_embedding_tokens(text_emb)
    if text_emb.ndim == 1:
        return [1, int(text_emb.shape[0])]
    if text_emb.ndim == 2:
        return [int(text_emb.shape[0]), int(text_emb.shape[1])]
    if text_emb.ndim >= 3:
        return [int(text_emb.shape[-2]), int(text_emb.shape[-1])]
    raise ValueError(
        f"Unsupported text embedding rank {text_emb.ndim} for shape inference."
    )


def infer_text_embedding_shape_from_dataset(config):
    dataset_path = getattr(config, 'dataset_path', '')
    if not dataset_path or not os.path.exists(dataset_path):
        logger.warning(
            'dataset_path `%s` does not exist; skipping text embedding shape inference.',
            dataset_path,
        )
        return None

    used_video_keys = _normalize_obs_cam_keys(getattr(config, 'obs_cam_keys', []))
    if not used_video_keys:
        logger.warning(
            'obs_cam_keys is empty; skipping text embedding shape inference for `%s`.',
            dataset_path,
        )
        return None

    repo_list = recursive_find_file(dataset_path, 'info.json')
    repo_list = [v.split('/meta/info.json')[0] for v in repo_list]
    if getattr(config, 'single_task', None):
        repo_list = [
            v for v in repo_list
            if Path(v).name.startswith(f"{config.single_task}-")
        ]

    for repo_path in repo_list:
        latent_root = Path(repo_path) / 'latents'
        if not latent_root.exists():
            continue
        for video_key in used_video_keys:
            for latent_file in latent_root.glob(f"chunk-*/{video_key}/*.pth"):
                latent_data = torch.load(latent_file, weights_only=False)
                text_emb = latent_data.get('text_emb', None)
                if text_emb is None:
                    continue
                return _infer_text_embedding_shape_from_value(text_emb), str(latent_file)

    logger.warning(
        'Unable to infer text embedding shape from `%s`; no latent file with text_emb was found.',
        dataset_path,
    )
    return None


def _load_jsonl_records(path: Path) -> list[dict]:
    records = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def recursive_find_file(directory, filename='info.json'):
    result = []
    try:
        for root, dirs, files in os.walk(directory):
            if filename in files:
                full_path = os.path.join(root, filename)
                result.append(full_path)
    except PermissionError:
        print(f"Error: can not access {directory}")
    except Exception as e:
        print(f"Error: {e}")
    return result

_TEXT_TOKENIZER_CACHE = {}


def _get_text_tokenizer_cache_key(config: TextTokenizerConfig) -> tuple[str | None, str | None, int | None]:
    return (
        config.text_tokenizer_name,
        config.model_name_or_path,
        config.max_position_embeddings,
    )


def _get_shared_text_tokenizer(config: TextTokenizerConfig):
    cache_key = _get_text_tokenizer_cache_key(config)
    tokenizer = _TEXT_TOKENIZER_CACHE.get(cache_key)
    if tokenizer is None:
        tokenizer = get_text_tokenizer(config)
        _TEXT_TOKENIZER_CACHE[cache_key] = tokenizer
    return tokenizer


def construct_lerobot(
    repo_id,
    config,
):
    return LatentLeRobotDataset(
        repo_id=repo_id,
        config=config,
    )

def construct_lerobot_multi_processor(config,
                                      num_init_worker=128,
                                      ):
    datasets_out_lst = []
    construct_func = partial(
        construct_lerobot,
        config=config,
    )
    repo_list = recursive_find_file(config.dataset_path, 'info.json')
    repo_list = [v.split('/meta/info.json')[0] for v in repo_list]
    if getattr(config, 'single_task', None):
        repo_list = [
            v for v in repo_list
            if Path(v).name.startswith(f"{config.single_task}-")
        ]
        logger.info(f"Found {len(repo_list)} repositories with info.json in {config.dataset_path} for task {config.single_task}.")
    # repo_list = repo_list[:2]
    mp_start_method = getattr(config, 'dataset_mp_start_method', 'spawn')
    pool_context = get_context(mp_start_method)
    with pool_context.Pool(num_init_worker) as pool:
        datasets_out_lst = pool.map(construct_func, repo_list)
    # for repo in repo_list:
    #     datasets_out_lst.append(construct_func(repo))
                
    return datasets_out_lst

def get_relative_pose(pose):
    if torch.is_tensor(pose):
        pose = pose.detach().cpu().numpy()
    
    rot = R.from_quat(pose[:, 3:7])
    first_rot = R.from_quat(np.tile(pose[:1, 3:7], (pose.shape[0], 1)))
    trans = pose[:, :3]
    relative_trans = trans - trans[0:1]

    relative_rot = first_rot.inv() * rot
    relative_quat = relative_rot.as_quat()

    relative_pose = np.concatenate([relative_trans, relative_quat], axis=1)
    return torch.from_numpy(relative_pose)


def _resolve_action_dim(config) -> int:
    if getattr(config, 'action_dim', None) is not None:
        return int(config.action_dim)
    if getattr(config, 'robot_action_dim', None) is not None:
        return int(config.robot_action_dim)
    inverse_ids = getattr(config, 'inverse_used_action_channel_ids', [])
    if inverse_ids:
        return len(inverse_ids)
    used_ids = getattr(config, 'used_action_channel_ids', [])
    if used_ids:
        return max(used_ids) + 1
    raise ValueError('Unable to infer action_dim from config.')


def _resolve_inverse_used_action_channel_ids(config) -> list[int]:
    used_ids = list(getattr(config, 'used_action_channel_ids', []))
    if used_ids:
        action_dim = _resolve_action_dim(config)
        inverse_ids = [len(used_ids)] * action_dim
        for i, j in enumerate(used_ids):
            inverse_ids[j] = i
        return inverse_ids

    inverse_ids = list(getattr(config, 'inverse_used_action_channel_ids', []))
    if inverse_ids:
        return inverse_ids

    raise ValueError('Config must provide either used_action_channel_ids or inverse_used_action_channel_ids.')


class MultiLatentLeRobotDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        config,
        num_init_worker=None,
    ):
        if num_init_worker is None:
            num_init_worker = getattr(config, 'dataset_init_worker', 8)
        self._datasets = construct_lerobot_multi_processor(config, 
                                                           num_init_worker, 
                                                           )
        self.item_id_to_dataset_id, self.acc_dset_num = (
            self._get_item_id_to_dataset_id()
        )

    def __len__(
        self,
    ):
        return sum(len(v) for v in self._datasets)

    def _get_item_id_to_dataset_id(self):
        item_id_to_dataset_id = {}
        acc_dset_num = {}
        acc_nums = [0]
        id = 0
        for dset_id, dset in enumerate(self._datasets):
            acc_nums.append(acc_nums[-1] + len(dset))
            for _ in range(len(dset)):
                item_id_to_dataset_id[id] = dset_id
                id += 1
        for did in range(len(self._datasets)):
            acc_dset_num[did] = acc_nums[did]
        return item_id_to_dataset_id, acc_dset_num

    def __getitem__(self, idx) -> dict:
        assert idx < len(self)
        cur_dset = self._datasets[self.item_id_to_dataset_id[idx]]
        local_idx = idx - self.acc_dset_num[self.item_id_to_dataset_id[idx]]
        return cur_dset[local_idx]

class LatentLeRobotDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        repo_id,
        config=None,
    ):
        self.repo_id = str(repo_id)
        self.root = Path(repo_id)
        self.image_transforms = None
        self.delta_timestamps = None
        self.episodes = None
        self.tolerance_s = 1e-4
        self.revision = "v2.1"
        self.video_backend = 'pyav'
        self.delta_indices = None
        self.batch_encoding_size = 1
        self.episodes_since_last_encoding = 0
        self.image_writer = None
        self.episode_buffer = None
        if not self.root.exists():
            raise FileNotFoundError(f'LeRobot repo path does not exist: {self.root}')

        self.meta_root = self.root / 'meta'
        self.meta_info = json.loads((self.meta_root / 'info.json').read_text())
        self.meta_episodes = _load_jsonl_records(self.meta_root / 'episodes.jsonl')
        self.episode_chunk_by_index = self._build_episode_chunk_index()

        self.latent_path = self.root / 'latents'
        self.image_path = self.root / 'videos'
        empty_emb_path = getattr(config, 'empty_emb_path', '')
        if empty_emb_path and os.path.exists(empty_emb_path):
            self.empty_emb = torch.load(empty_emb_path, weights_only=False)
        else:
            self.empty_emb = None
            logger.warning(
                'empty_emb_path is not set or does not exist for %s; falling back to zeros_like(text_emb) when CFG text dropout is applied.',
                repo_id,
            )
        self.config = config
        self.cfg_prob = config.cfg_prob
        self.text_tokenizer_config = getattr(config, 'text_tokenizer_config', None)
        self.prefer_raw_text_for_text_embeddings = bool(
            getattr(config, 'prefer_raw_text_for_text_embeddings', True)
        )
        self._text_embedding_cache = {}
        self._empty_text_embedding = None
        self._raw_text_warning_emitted = False
        self.used_video_keys = _normalize_obs_cam_keys(config.obs_cam_keys)
        self.used_action_channel_ids = list(getattr(config, 'used_action_channel_ids', []))
        self._empty_emb_shape_warning_emitted = False
        logger.info('Using observation keys %s for repo %s', self.used_video_keys, repo_id)
        self.image_height, self.image_width = config.image_height, config.image_width
        self.inverse_used_action_channel_ids = _resolve_inverse_used_action_channel_ids(config)
        self.robot_action_dim = _resolve_action_dim(config)
        if len(self.inverse_used_action_channel_ids) != self.robot_action_dim:
            raise ValueError(
                f'inverse_used_action_channel_ids length {len(self.inverse_used_action_channel_ids)} does not match robot_action_dim {self.robot_action_dim}.'
            )
        self.q01 = np.array(config.norm_stat['q01'], dtype='float')[None]
        self.q99 = np.array(config.norm_stat['q99'], dtype='float')[None]
        self.parse_meta()

    def _build_episode_chunk_index(self) -> dict[int, int]:
        episode_chunk_by_index = {}
        data_root = self.root / 'data'
        for parquet_path in data_root.glob('chunk-*/episode_*.parquet'):
            chunk_name = parquet_path.parent.name
            episode_name = parquet_path.stem
            try:
                chunk_index = int(chunk_name.split('-')[-1])
                episode_index = int(episode_name.split('_')[-1])
            except ValueError:
                continue
            episode_chunk_by_index[episode_index] = chunk_index
        return episode_chunk_by_index

    def _get_episode_chunk(self, episode_index: int) -> int:
        if episode_index not in self.episode_chunk_by_index:
            raise KeyError(f'Missing chunk index for episode {episode_index} in {self.root}')
        return self.episode_chunk_by_index[episode_index]

    def _get_episode_action_path(self, episode_index: int) -> Path:
        episode_chunk = self._get_episode_chunk(episode_index)
        return self.root / 'data' / f'chunk-{episode_chunk:03d}' / f'episode_{episode_index:06d}.parquet'

    def parse_meta(self):
        out = []
        for value in self.meta_episodes:
            episode_index = value["episode_index"]
            tasks = value["tasks"]
            action_config = value["action_config"]
            for acfg in action_config:
                cur_meta = {
                    "episode_index": episode_index,
                    "tasks": tasks,
                }
                cur_meta.update(acfg)

                check_statu = self._check_meta(
                    cur_meta["start_frame"],
                    cur_meta["end_frame"],
                    cur_meta["episode_index"],
                )

                if check_statu:
                    out.append(cur_meta)
        self.new_metas = out

    def _check_meta(self, start_frame, end_frame, episode_index):
        episode_chunk = self._get_episode_chunk(episode_index)
        latent_path = Path(self.latent_path) / f"chunk-{episode_chunk:03d}"
        for key in self.used_video_keys:
            cur_path = latent_path / key
            latent_file = (
                cur_path / f"episode_{episode_index:06d}_{start_frame}_{end_frame}.pth"
            )
            if not os.path.exists(latent_file):
                return False
        return True

    def _get_range_hf_data(self, start_frame, end_frame, episode_index):
        action_path = self._get_episode_action_path(episode_index)
        action_table = pq.read_table(action_path, columns=['action']).slice(
            start_frame, end_frame - start_frame
        )
        action_values = np.asarray(action_table.column('action').to_pylist(), dtype=np.float32)
        return {'action': torch.from_numpy(action_values)}

    def _flatten_latent_dict(self, latent_dict):
        out = {}
        for key, value in latent_dict.items():
            for inner_key, inner_value in value.items():
                new_key = f"{key}.{inner_key}"
                out[new_key] = inner_value
        return out

    def _get_range_latent_data(self, start_frame, end_frame, episode_index):
        episode_chunk = self._get_episode_chunk(episode_index)
        latent_path = Path(self.latent_path) / f"chunk-{episode_chunk:03d}"
        out = {}
        for key in self.used_video_keys:
            cur_path = latent_path / key
            latent_file = (
                cur_path / f"episode_{episode_index:06d}_{start_frame}_{end_frame}.pth"
            )
            assert os.path.exists(latent_file)
            latent_data = torch.load(latent_file, weights_only=False)
            out[key] = latent_data
        
        return self._flatten_latent_dict(out)

    def _get_range_image_data(self, start_frame, end_frame, episode_index):
        episode_chunk = self._get_episode_chunk(episode_index)
        image_path = Path(self.image_path) / f"chunk-{episode_chunk:03d}"
        latent_path = Path(self.latent_path) / f"chunk-{episode_chunk:03d}"
        out = {}
        for key in self.used_video_keys:
            cur_path = image_path / key
            image_file = (
                cur_path / f"episode_{episode_index:06d}.mp4"
            )
            assert os.path.exists(image_file)
            # Load image_data from mp4 file as a tensor of shape [F, H, W, C].
            try:
                import av
                with av.open(str(image_file)) as container:
                    frames = [
                        frame.to_ndarray(format='rgb24')
                        for frame in container.decode(video=0)
                    ]
            except Exception as e:
                raise RuntimeError(
                    f"Failed to decode video {image_file}. "
                    "OpenCV and PyAV both failed."
                ) from e

            image_data = torch.from_numpy(np.stack(frames))
            image_data = image_data[start_frame:end_frame]
            num_frames, height, width, channels = image_data.shape

            if num_frames == 0:
                raise RuntimeError(
                    f"Empty frame slice [{start_frame}:{end_frame}] for {image_file}. "
                    "Check action segment boundaries versus available frames."
                )
            
            cur_path = latent_path / key
            latent_file = (
                cur_path / f"episode_{episode_index:06d}_{start_frame}_{end_frame}.pth"
            )
            assert os.path.exists(latent_file)
            latent_data = torch.load(latent_file, weights_only=False)

            out[key] = {
                "video": image_data.reshape(num_frames * height * width, channels),
                "video_num_frames": num_frames,
                "video_height": height,
                "video_width": width,
                "frame_ids": torch.arange(start_frame, end_frame, dtype=torch.long),
                "text_emb": latent_data.get("text_emb", None)
            }
        return self._flatten_latent_dict(out)
    
        
    def _get_text_tokenizer(self):
        if self.text_tokenizer_config is None:
            return None
        tokenizer_name = getattr(self.text_tokenizer_config, 'text_tokenizer_name', None)
        if tokenizer_name is None:
            return None
        return _get_shared_text_tokenizer(self.text_tokenizer_config)

    def _resolve_raw_text(self, meta: dict) -> str | None:
        action_text = str(meta.get('action_text', '') or '').strip()
        if action_text:
            return action_text

        for task in meta.get('tasks', []) or []:
            task_text = str(task or '').strip()
            if task_text:
                return task_text
        return None

    def _encode_raw_text_embedding(self, text: str) -> torch.Tensor | None:
        tokenizer = self._get_text_tokenizer()
        if tokenizer is None:
            return None

        cached = self._text_embedding_cache.get(text)
        if cached is not None:
            return cached

        tokenized = tokenizer.tokenize(text)
        with torch.inference_mode():
            text_embedding = tokenizer(**tokenized)
        if text_embedding.ndim != 4:
            raise ValueError(
                f'Expected tokenizer output with 4 dims, got {text_embedding.ndim} for text `{text}`.'
            )
        text_embedding = text_embedding.squeeze(0).squeeze(0).cpu().float().contiguous()
        self._text_embedding_cache[text] = text_embedding
        return text_embedding

    def _expand_text_embedding_like(
        self, source_embedding, target_embedding: torch.Tensor
    ) -> torch.Tensor | None:
        if source_embedding is None:
            return None

        source_embedding = torch.as_tensor(source_embedding)
        target_embedding = torch.as_tensor(target_embedding)
        if tuple(source_embedding.shape) == tuple(target_embedding.shape):
            return source_embedding.to(dtype=target_embedding.dtype)

        if (
            source_embedding.ndim + 1 == target_embedding.ndim
            and tuple(source_embedding.shape) == tuple(target_embedding.shape[1:])
        ):
            return source_embedding.unsqueeze(0).expand(*target_embedding.shape).contiguous().to(
                dtype=target_embedding.dtype
            )

        if (
            source_embedding.ndim == target_embedding.ndim
            and source_embedding.shape[0] == 1
            and tuple(source_embedding.shape[1:]) == tuple(target_embedding.shape[1:])
        ):
            return source_embedding.expand(*target_embedding.shape).contiguous().to(
                dtype=target_embedding.dtype
            )

        return None

    def _get_empty_text_embedding(self, template_text_embedding: torch.Tensor) -> torch.Tensor | None:
        if self._empty_text_embedding is None:
            self._empty_text_embedding = self._encode_raw_text_embedding('')
        return self._expand_text_embedding_like(
            self._empty_text_embedding,
            template_text_embedding,
        )

    def _get_selected_text_embedding(self, meta: dict, data_dict: dict) -> torch.Tensor:
        raw_text = self._resolve_raw_text(meta)
        if self.prefer_raw_text_for_text_embeddings and raw_text:
            try:
                raw_text_embedding = self._encode_raw_text_embedding(raw_text)
            except Exception as exc:
                if not self._raw_text_warning_emitted:
                    logger.warning(
                        'Failed to encode raw text `%s` for repo %s; falling back to precomputed text_emb. Error: %s',
                        raw_text,
                        self.repo_id,
                        exc,
                    )
                    self._raw_text_warning_emitted = True
            else:
                if raw_text_embedding is not None:
                    return self._get_cfg_text_embedding(raw_text_embedding)

        fallback_text_embedding = data_dict.get(f"{self.used_video_keys[0]}.text_emb")
        if fallback_text_embedding is None:
            if raw_text:
                raw_text_embedding = self._encode_raw_text_embedding(raw_text)
                if raw_text_embedding is not None:
                    return self._get_cfg_text_embedding(raw_text_embedding)
            raise ValueError(
                f'No text embedding source found for repo {self.repo_id}, episode {meta.get("episode_index")}.',
            )

        return self._get_cfg_text_embedding(fallback_text_embedding)

    def _cat_video_latents(self,
                           data_dict,
                           text_embedding=None
                           ):
        latent_lst = []
        for key in self.used_video_keys:
            latent= data_dict[f"{key}.latent"]
            latent_num_frames = data_dict[f"{key}.latent_num_frames"]
            latent_height = data_dict[f"{key}.latent_height"]
            latent_width = data_dict[f"{key}.latent_width"]
            latent = rearrange(latent, 
                                 '(f h w) c -> f h w c', 
                                 f=latent_num_frames, 
                                 h=latent_height, 
                                 w=latent_width)
            latent_lst.append(latent)
        wrist_latent = torch.cat(latent_lst[1:], dim=2)
        cat_latent = torch.cat([wrist_latent, latent_lst[0]], dim=1)

        if text_embedding is None:
            text_embedding = data_dict.get(f"{self.used_video_keys[0]}.text_emb")
        text_emb = self._get_cfg_text_embedding(text_embedding)

        out_dict = dict(
            latents = cat_latent,
            text_emb = text_emb,
        )
        print(f"lactent shape: {cat_latent.shape}")
        return out_dict

    def _merge_multi_view_images(self, image_lst):
        mode = getattr(self.config, 'multi_view_image_mode', 'vertical')
        if mode == 'vertical':
            return torch.cat(image_lst, dim=2)
        if mode == 'frame':
            return rearrange(torch.stack(image_lst, dim=1), 'f v c h w -> (f v) c h w')
        if mode == 'first':
            return image_lst[0]
        if mode == 'token_concat':
            return torch.stack(image_lst, dim=1)
        raise ValueError(
            f"Unsupported multi_view_image_mode `{mode}`. Expected one of ['vertical', 'frame', 'first', 'token_concat']."
        )

    def _align_actions_with_multi_view_mode(self, actions, actions_mask):
        if getattr(self.config, 'multi_view_image_mode', 'vertical') != 'frame':
            return actions, actions_mask

        num_views = len(self.used_video_keys)
        actions = actions.repeat_interleave(num_views, dim=1)
        actions_mask = actions_mask.repeat_interleave(num_views, dim=1)
        return actions, actions_mask
    
    def _cat_video_images(self,
                          data_dict,
                          text_embedding=None):
        image_lst = []
        for key in self.used_video_keys:
            image = data_dict[f"{key}.video"]
            image_num_frames = data_dict[f"{key}.video_num_frames"]
            image_height = data_dict[f"{key}.video_height"]
            image_width = data_dict[f"{key}.video_width"]
            image = rearrange(image,
                                 '(f h w) c -> f c h w ',
                                 f=image_num_frames,
                                 h=image_height,
                                 w=image_width)
            image = image[::self.config.image_frame_stride]

            h = image.shape[2]
            w = image.shape[3]
            scale = min(self.image_height / h, self.image_width / w)
            new_h = int(round(h * scale))
            new_w = int(round(w * scale))

            image = F.interpolate(
                image.float(),
                size=(new_h, new_w),
                mode='bilinear',
                align_corners=False,
            )

            pad_h = self.image_height - new_h
            pad_w = self.image_width - new_w
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            image = F.pad(image, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
            image = image.round().clamp(0, 255).to(torch.uint8)
            image_lst.append(image)

        cat_image = self._merge_multi_view_images(image_lst)
        if text_embedding is None:
            text_embedding = data_dict.get(f"{self.used_video_keys[0]}.text_emb")
        text_emb = self._get_cfg_text_embedding(text_embedding)

        out_dict = dict(
            frames=cat_image,
            text_embeddings=text_emb,
        )
        return out_dict

    def _get_cfg_text_embedding(self, text_emb):
        if torch.rand(1).item() >= self.cfg_prob:
            return text_emb

        matched_empty_embedding = self._expand_text_embedding_like(self.empty_emb, text_emb)
        if matched_empty_embedding is not None:
            return matched_empty_embedding

        matched_empty_text_embedding = self._get_empty_text_embedding(text_emb)
        if matched_empty_text_embedding is not None:
            return matched_empty_text_embedding

        if self.empty_emb is None:
            return torch.zeros_like(text_emb)

        if not self._empty_emb_shape_warning_emitted:
            logger.warning(
                'empty_emb shape %s does not match text_emb shape %s for repo %s; falling back to tokenizer empty-text embedding or zeros_like(text_emb).',
                tuple(torch.as_tensor(self.empty_emb).shape),
                tuple(text_emb.shape),
                self.repo_id,
            )
            self._empty_emb_shape_warning_emitted = True
        return torch.zeros_like(text_emb)

    def _flatten_action_annotations(self, actions: torch.Tensor) -> torch.Tensor:
        return rearrange(actions, "c f n one -> f (c n one)").float()

    def _flatten_action_mask(self, actions_mask: torch.Tensor) -> torch.Tensor:
        return rearrange(actions_mask, "c f n one -> f (c n one)").bool()

    def _frame_mask_from_actions(self, actions_mask: torch.Tensor) -> torch.Tensor:
        return self._flatten_action_mask(actions_mask).any(dim=1)

    def _format_text_embeddings(
        self, text_embeddings: torch.Tensor, num_frames: int
    ) -> torch.Tensor:
        text_embeddings = _pool_text_embedding_tokens(text_embeddings)

        if text_embeddings.ndim == 1:
            text_embeddings = text_embeddings.unsqueeze(0)

        if text_embeddings.ndim == 3:
            if text_embeddings.shape[0] == num_frames:
                return text_embeddings.float()
            if text_embeddings.shape[0] == 1:
                text_embeddings = text_embeddings.squeeze(0)
            else:
                raise ValueError(
                    f"Unexpected text embedding shape {tuple(text_embeddings.shape)} for {num_frames} frames."
                )

        if text_embeddings.ndim != 2:
            raise ValueError(
                f"Expected text embeddings with 1, 2, or 3 dims, got {text_embeddings.ndim}."
            )

        return (
            text_embeddings.unsqueeze(0)
            .expand(num_frames, -1, -1)
            .contiguous()
            .float()
        )

    def _get_target_seq_len(self) -> int | None:
        target_seq_len = getattr(self.config, "n_seq_timesteps", None)
        if target_seq_len is None:
            return None
        target_seq_len = int(target_seq_len)
        if target_seq_len <= 0:
            raise ValueError(f"n_seq_timesteps must be positive, got {target_seq_len}.")
        return target_seq_len

    def _select_time_slice(self, num_frames: int, target_seq_len: int) -> slice:
        if num_frames <= target_seq_len:
            return slice(0, num_frames)

        max_start = num_frames - target_seq_len
        if getattr(self.config, "shuffle", False) and max_start > 0:
            start = int(torch.randint(0, max_start + 1, ()).item())
        else:
            start = 0
        return slice(start, start + target_seq_len)

    def _pad_time_dim(
        self,
        tensor: torch.Tensor,
        target_seq_len: int,
        pad_mode: str,
    ) -> torch.Tensor:
        current_seq_len = tensor.shape[0]
        if current_seq_len >= target_seq_len:
            return tensor

        pad_len = target_seq_len - current_seq_len
        pad_shape = (pad_len, *tensor.shape[1:])

        if pad_mode == "repeat_last" and current_seq_len > 0:
            pad = tensor[-1:].expand(*pad_shape).clone()
        elif pad_mode == "zeros":
            pad = torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)
        else:
            raise ValueError(f"Unsupported pad_mode `{pad_mode}`.")

        return torch.cat([tensor, pad], dim=0)

    def _normalize_quaternion_np(self, quat: np.ndarray) -> np.ndarray:
        quat = np.asarray(quat, dtype=np.float32).reshape(4)
        norm = np.linalg.norm(quat)
        if norm < 1e-8:
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        return quat / norm

    def _rebase_compact_action_sequence(
        self,
        compact_actions: np.ndarray,
        reference_action: np.ndarray,
    ) -> np.ndarray:
        compact_actions = np.asarray(compact_actions, dtype=np.float32)
        reference_action = np.asarray(reference_action, dtype=np.float32).reshape(16)
        rebased = compact_actions.copy()

        left_reference_rot = R.from_quat(
            self._normalize_quaternion_np(reference_action[3:7])[None]
        )
        right_reference_rot = R.from_quat(
            self._normalize_quaternion_np(reference_action[11:15])[None]
        )
        for idx, compact_action in enumerate(compact_actions):
            left_rot = R.from_quat(self._normalize_quaternion_np(compact_action[3:7])[None])
            right_rot = R.from_quat(
                self._normalize_quaternion_np(compact_action[11:15])[None]
            )
            rebased[idx, :3] = compact_action[:3] - reference_action[:3]
            rebased[idx, 3:7] = (
                left_reference_rot.inv() * left_rot
            ).as_quat().reshape(-1)
            rebased[idx, 7] = compact_action[7]
            rebased[idx, 8:11] = compact_action[8:11] - reference_action[8:11]
            rebased[idx, 11:15] = (
                right_reference_rot.inv() * right_rot
            ).as_quat().reshape(-1)
            rebased[idx, 15] = compact_action[15]
        return rebased

    def _rebase_item_actions_to_slice_start(
        self, item: ActionLabelVideoDatasetItem
    ) -> ActionLabelVideoDatasetItem:
        if not self.used_action_channel_ids:
            return item

        valid_mask = item.user_action_mask.bool()
        if item.valid_frame_mask is not None:
            valid_mask = valid_mask & item.valid_frame_mask.bool()
        valid_indices = torch.nonzero(valid_mask, as_tuple=False).flatten()
        if valid_indices.numel() == 0:
            return item

        first_valid_idx = int(valid_indices[0].item())
        action_annotations = item.action_annotations.detach().cpu().numpy().astype(np.float32, copy=True)
        used_ids = np.asarray(self.used_action_channel_ids, dtype=np.int64)
        q01 = self.q01.reshape(-1)[used_ids].astype(np.float32)
        q99 = self.q99.reshape(-1)[used_ids].astype(np.float32)
        scale = q99 - q01 + 1e-6

        compact_actions = ((action_annotations[:, used_ids] + 1.0) * 0.5) * scale + q01
        rebased_compact_actions = compact_actions.copy()
        rebased_compact_actions[first_valid_idx:] = self._rebase_compact_action_sequence(
            compact_actions[first_valid_idx:],
            compact_actions[first_valid_idx],
        )
        action_annotations[:, used_ids] = ((rebased_compact_actions - q01) / scale) * 2.0 - 1.0
        return item._replace(
            action_annotations=torch.from_numpy(action_annotations).float()
        )

    def _ensure_fixed_sequence_length(
        self, item: ActionLabelVideoDatasetItem
    ) -> ActionLabelVideoDatasetItem:
        target_seq_len = self._get_target_seq_len()
        if target_seq_len is None:
            return item

        time_slice = self._select_time_slice(item.frames.shape[0], target_seq_len)
        valid_frame_mask = item.valid_frame_mask
        if valid_frame_mask is None:
            valid_frame_mask = torch.ones(item.frames.shape[0], dtype=torch.bool)

        sliced_item = ActionLabelVideoDatasetItem(
            frames=item.frames[time_slice].contiguous(),
            action_annotations=item.action_annotations[time_slice].contiguous(),
            env_subenv_encoding=item.env_subenv_encoding[time_slice].contiguous(),
            user_action_mask=item.user_action_mask[time_slice].contiguous(),
            text_embeddings=item.text_embeddings[time_slice].contiguous(),
            system_action_mask=item.system_action_mask[time_slice].contiguous(),
            valid_frame_mask=valid_frame_mask[time_slice].contiguous(),
            action_mask=item.action_mask[time_slice].contiguous(),
        )

        fixed_item = ActionLabelVideoDatasetItem(
            frames=self._pad_time_dim(
                sliced_item.frames, target_seq_len, pad_mode="repeat_last"
            ),
            action_annotations=self._pad_time_dim(
                sliced_item.action_annotations, target_seq_len, pad_mode="zeros"
            ),
            env_subenv_encoding=self._pad_time_dim(
                sliced_item.env_subenv_encoding,
                target_seq_len,
                pad_mode="repeat_last",
            ),
            user_action_mask=self._pad_time_dim(
                sliced_item.user_action_mask, target_seq_len, pad_mode="zeros"
            ),
            text_embeddings=self._pad_time_dim(
                sliced_item.text_embeddings, target_seq_len, pad_mode="repeat_last"
            ),
            system_action_mask=self._pad_time_dim(
                sliced_item.system_action_mask, target_seq_len, pad_mode="zeros"
            ),
            valid_frame_mask=self._pad_time_dim(
                sliced_item.valid_frame_mask, target_seq_len, pad_mode="zeros"
            ),
            action_mask=self._pad_time_dim(
                sliced_item.action_mask, target_seq_len, pad_mode="zeros"
            ),
        )
        return self._rebase_item_actions_to_slice_start(fixed_item)

    def _action_post_process(self, local_start_frame, local_end_frame, image_frame_ids, action):
        act_shift = int(image_frame_ids[0] - local_start_frame)
        frame_stride = image_frame_ids[1] - image_frame_ids[0]
        action = action[act_shift:]
        left_action = get_relative_pose(action[:, :7])
        right_action = get_relative_pose(action[:, 8:15])
        action = np.concatenate([left_action, action[:, 7:8], right_action, action[:, 15:16]], axis=1)

        action_mask = np.ones_like(action, dtype='bool')
        pad_steps = frame_stride * self.config.image_frame_stride
        action = np.pad(
            action,
            pad_width=((pad_steps, 0), (0, 0)),
            mode='constant',
            constant_values=0,
        )
        action_mask = np.pad(
            action_mask,
            pad_width=((pad_steps, 0), (0, 0)),
            mode='constant',
            constant_values=0,
        )

        image_frame_num = (len(image_frame_ids) - 1) // self.config.image_frame_stride + 1
        required_action_num = image_frame_num * frame_stride * self.config.image_frame_stride

        action = action[:required_action_num]
        action_mask = action_mask[:required_action_num]
        assert action.shape[0] == required_action_num
        assert action_mask.shape == action.shape

        action_paded = np.pad(action, ((0, 0), (0, 1)), mode='constant', constant_values=0)
        action_mask_padded = np.pad(action_mask, ((0, 0), (0, 1)), mode='constant', constant_values=0)

        action_aligned = action_paded[:, self.inverse_used_action_channel_ids]
        action_mask_aligned = action_mask_padded[:, self.inverse_used_action_channel_ids]
        action_aligned = (action_aligned - self.q01) / (
                self.q99 - self.q01 + 1e-6) * 2. - 1.
        action_aligned = rearrange(action_aligned, "(f n) c -> c f n 1", f=image_frame_num)
        action_mask_aligned = rearrange(action_mask_aligned, "(f n) c -> c f n 1", f=image_frame_num)
        action_aligned *= action_mask_aligned
        return torch.from_numpy(action_aligned).float(), torch.from_numpy(action_mask_aligned).bool()

    def __getitem__(self, idx) -> ActionLabelVideoDatasetItem:
        """Return a sample in the shared training dataset format."""
        idx = idx % len(self.new_metas)
        cur_meta = self.new_metas[idx]
        episode_index = cur_meta["episode_index"]
        start_frame = cur_meta["start_frame"]
        end_frame = cur_meta["end_frame"]
        local_start_frame = start_frame
        local_end_frame = end_frame

        ori_data_dict = self._get_range_image_data(start_frame, end_frame, episode_index)

        image_frame_ids = ori_data_dict[f"{self.used_video_keys[0]}.frame_ids"]

        hf_data_frames = self._get_range_hf_data(start_frame, end_frame, episode_index)
        ori_data_dict.update(hf_data_frames)
        selected_text_embedding = self._get_selected_text_embedding(cur_meta, ori_data_dict)
        out_dict = self._cat_video_images(
            ori_data_dict,
            text_embedding=selected_text_embedding,
        )

        actions, actions_mask = self._action_post_process(
            local_start_frame,
            local_end_frame,
            image_frame_ids,
            ori_data_dict['action'],
        )
        actions, actions_mask = self._align_actions_with_multi_view_mode(
            actions,
            actions_mask,
        )

        frames = out_dict["frames"]
        num_frames = frames.shape[0]
        action_annotations = self._flatten_action_annotations(actions)
        action_mask = self._flatten_action_mask(actions_mask)
        user_action_mask = self._frame_mask_from_actions(actions_mask)

        assert num_frames == action_annotations.shape[0], (
            f"Number of frames in images and actions must be the same, but got {num_frames} and {action_annotations.shape[0]}"
        )

        item = ActionLabelVideoDatasetItem(
            frames=frames,
            action_annotations=action_annotations,
            env_subenv_encoding=torch.zeros(num_frames, dtype=torch.long),
            user_action_mask=user_action_mask,
            text_embeddings=self._format_text_embeddings(
                out_dict["text_embeddings"], num_frames
            ),
            system_action_mask=torch.zeros(num_frames, dtype=torch.bool),
            valid_frame_mask=torch.ones(num_frames, dtype=torch.bool),
            action_mask=action_mask,
        )
        return self._ensure_fixed_sequence_length(item)

    def __len__(self):
        return len(self.new_metas)

if __name__ == '__main__':
    from wan_va.configs import VA_CONFIGS
    from tqdm import tqdm
    dset = MultiLatentLeRobotDataset(
        VA_CONFIGS['robotwin_train']
    )
    for key, value in dset[0]._asdict().items():
        if isinstance(value, torch.Tensor):
            print(f'{key}: {value.shape} tensor')
        elif isinstance(value, np.ndarray):
            print(f'{key}: {value.shape} np')
        else:
            print(f'{key}: {value}')
    print(len(dset))
    dloader = DataLoader(
            dset,
            batch_size=1,
            shuffle=True,
            num_workers=32,
        )
    max_l = 0
    action_list = []
    for data in tqdm(dloader):
        _, F, H, W = data.frames.shape[1:]
        max_l = max(max_l, F * H * W)
        action_list.append(data.action_annotations.flatten(0, 1))
    action_all = torch.cat(action_list, dim=0)
    print(max_l)
    print(action_all.shape, action_all.mean(dim=0), action_all.min(dim=0)[0], action_all.max(dim=0)[0])
    

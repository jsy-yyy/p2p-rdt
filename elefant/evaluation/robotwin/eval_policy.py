from __future__ import annotations

import argparse
import importlib
import importlib.util
import types
import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R

from .websocket_client_policy import WebsocketClientPolicy

LOGGER = logging.getLogger(__name__)


def _raise_missing_sapien_dependency(exc: ModuleNotFoundError) -> None:
    raise ModuleNotFoundError(
        "RoboTwin evaluation requires `sapien` in the Python environment. "
        "The current launcher should use the RoboTwin interpreter, for example: "
        "ROBOTWIN_PYTHON=/home/zmz/miniconda3/envs/RoboTwin/bin/python bash scripts/eval_robotwin.sh adjust_bottle demo_clean"
    ) from exc


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        LOGGER.warning("Invalid integer env %s=%r; using default %s.", name, value, default)
        return default
    return max(parsed, 1)


def _configure_robowin_rendering() -> None:
    disable_oidn = _env_flag("OPEN_P2P_ROBOTWIN_DISABLE_OIDN", default=True)
    force_raster = _env_flag("OPEN_P2P_ROBOTWIN_FORCE_RASTER", default=False)
    if not disable_oidn and not force_raster:
        return

    try:
        import sapien
    except ModuleNotFoundError as exc:
        if exc.name == "sapien":
            _raise_missing_sapien_dependency(exc)
        raise

    render = sapien.render

    if disable_oidn:
        original_set_ray_tracing_denoiser = render.set_ray_tracing_denoiser

        def _set_ray_tracing_denoiser(name: str):
            if str(name).lower() == "oidn":
                LOGGER.info(
                    "Replacing RoboTwin ray-tracing denoiser 'oidn' with 'none' to avoid OIDN renderer crashes."
                )
                try:
                    return original_set_ray_tracing_denoiser("none")
                except Exception:
                    LOGGER.warning(
                        "Failed to switch RoboTwin denoiser to 'none'; skipping denoiser setup instead."
                    )
                    return None
            return original_set_ray_tracing_denoiser(name)

        render.set_ray_tracing_denoiser = _set_ray_tracing_denoiser

    if force_raster:
        original_set_camera_shader_dir = render.set_camera_shader_dir
        original_set_ray_tracing_samples_per_pixel = render.set_ray_tracing_samples_per_pixel
        original_set_ray_tracing_path_depth = render.set_ray_tracing_path_depth
        original_set_ray_tracing_denoiser = render.set_ray_tracing_denoiser

        def _set_camera_shader_dir(shader_dir: str):
            if str(shader_dir).lower() == "rt":
                LOGGER.info(
                    "Replacing RoboTwin camera shader dir 'rt' with 'default' to force raster rendering."
                )
                return original_set_camera_shader_dir("default")
            return original_set_camera_shader_dir(shader_dir)

        def _skip_ray_tracing_samples_per_pixel(*args, **kwargs):
            LOGGER.info("Skipping RoboTwin ray-tracing samples-per-pixel configuration in raster mode.")
            return None

        def _skip_ray_tracing_path_depth(*args, **kwargs):
            LOGGER.info("Skipping RoboTwin ray-tracing path-depth configuration in raster mode.")
            return None

        def _skip_ray_tracing_denoiser(*args, **kwargs):
            LOGGER.info("Skipping RoboTwin ray-tracing denoiser configuration in raster mode.")
            return None

        render.set_camera_shader_dir = _set_camera_shader_dir
        render.set_ray_tracing_samples_per_pixel = _skip_ray_tracing_samples_per_pixel
        render.set_ray_tracing_path_depth = _skip_ray_tracing_path_depth
        render.set_ray_tracing_denoiser = _skip_ray_tracing_denoiser


def _bootstrap_robowin(robowin_root: str) -> Path:
    root = Path(robowin_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"RoboTwin root not found: {root}")

    utils_root = root / "description" / "utils"
    for path in (root, utils_root):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

    os.chdir(root)
    return root


def _coerce_mplib_planner_type(planner_type: str | None) -> str:
    if planner_type in {"mplib_RRT", "mplib_screw"}:
        return planner_type
    LOGGER.warning(
        "Unsupported RoboTwin planner_type=%r for mplib fallback; forcing 'mplib_RRT'.",
        planner_type,
    )
    return "mplib_RRT"


def _install_robowin_planner_fallback() -> None:
    if "envs.robot.robot" in sys.modules and "envs.robot.planner" in sys.modules:
        return

    envs_root = Path.cwd() / "envs"
    robot_root = envs_root / "robot"
    planner_path = robot_root / "planner.py"
    robot_path = robot_root / "robot.py"

    def _load_module(module_name: str, file_path: Path):
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load module {module_name} from {file_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    planner_module = _load_module("envs.robot.planner", planner_path)
    if hasattr(planner_module, "CuroboPlanner"):
        return

    class _UnavailableCuroboPlanner:
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "CuroboPlanner is unavailable in the current RoboTwin environment."
            )

    planner_module.CuroboPlanner = _UnavailableCuroboPlanner

    robot_pkg = types.ModuleType("envs.robot")
    robot_pkg.__path__ = [str(robot_root)]
    robot_pkg.__package__ = "envs.robot"
    robot_pkg.planner = planner_module
    sys.modules["envs.robot"] = robot_pkg

    robot_module = _load_module("envs.robot.robot", robot_path)

    class _FallbackBatchPlanner:
        def __init__(
            self,
            urdf_path,
            srdf_path,
            move_group,
            robot_origion_pose,
            robot_entity,
            planner_type="mplib_RRT",
            scene=None,
        ):
            # Avoid RoboTwin's scene-aware SapienPlanner path here: with the
            # current mplib/SAPIEN stack it can fail during planning setup with
            # an internal `result` unbound error. The plain mplib planner is
            # less feature-rich but is stable enough for evaluation fallback.
            planner_type = _coerce_mplib_planner_type(planner_type)
            self._planner = planner_module.MplibPlanner(
                urdf_path,
                srdf_path,
                move_group,
                robot_origion_pose,
                robot_entity,
                planner_type,
                None,
            )

        def plan_path(self, curr_joint_pos, target_gripper_pose, constraint_pose=None, arms_tag=None):
            del constraint_pose
            return self._planner.plan_path(curr_joint_pos, target_gripper_pose, arms_tag=arms_tag, log=False)

        def plan_batch(self, curr_joint_pos, target_gripper_pose_list, constraint_pose=None, arms_tag=None):
            del constraint_pose
            statuses = []
            positions = []
            velocities = []
            joint_dim = len(curr_joint_pos)
            for target_pose in target_gripper_pose_list:
                result = self._planner.plan_path(curr_joint_pos, target_pose, arms_tag=arms_tag, log=False)
                if result.get("status") == "Success":
                    statuses.append("Success")
                    positions.append(np.array(result["position"]))
                    velocities.append(np.array(result.get("velocity", np.zeros_like(result["position"]))))
                else:
                    statuses.append("Failure")
                    positions.append(np.zeros((0, joint_dim), dtype=np.float32))
                    velocities.append(np.zeros((0, joint_dim), dtype=np.float32))
            return {
                "status": np.array(statuses, dtype=object),
                "position": np.array(positions, dtype=object),
                "velocity": np.array(velocities, dtype=object),
            }

        def plan_grippers(self, now_val, target_val):
            return self._planner.plan_grippers(now_val, target_val)

        def update_point_cloud(self, world_pcd, resolution=0.02):
            if hasattr(self._planner, "update_point_cloud"):
                return self._planner.update_point_cloud(world_pcd, resolution=resolution)
            return None

    def _fallback_set_planner(self, scene=None):
        LOGGER.warning(
            "CuroboPlanner is unavailable; falling back to MplibPlanner for RoboTwin evaluation."
        )
        self.communication_flag = False
        self.left_planner = _FallbackBatchPlanner(
            self.left_urdf_path,
            self.left_srdf_path,
            self.left_move_group,
            self.left_entity_origion_pose,
            self.left_entity,
            self.left_planner_type,
            scene,
        )
        self.right_planner = _FallbackBatchPlanner(
            self.right_urdf_path,
            self.right_srdf_path,
            self.right_move_group,
            self.right_entity_origion_pose,
            self.right_entity,
            self.right_planner_type,
            scene,
        )

        if self.need_topp:
            self.left_mplib_planner = planner_module.MplibPlanner(
                self.left_urdf_path,
                self.left_srdf_path,
                self.left_move_group,
                self.left_entity_origion_pose,
                self.left_entity,
                _coerce_mplib_planner_type(self.left_planner_type),
                None,
            )
            self.right_mplib_planner = planner_module.MplibPlanner(
                self.right_urdf_path,
                self.right_srdf_path,
                self.right_move_group,
                self.right_entity_origion_pose,
                self.right_entity,
                _coerce_mplib_planner_type(self.right_planner_type),
                None,
            )

    robot_module.CuroboPlanner = _UnavailableCuroboPlanner
    robot_module.Robot.set_planner = _fallback_set_planner
    robot_pkg.robot = robot_module
    robot_pkg.Robot = robot_module.Robot



def _class_decorator(task_name: str):
    try:
        _install_robowin_planner_fallback()
        envs_module = importlib.import_module(f"envs.{task_name}")
    except ModuleNotFoundError as exc:
        if exc.name == "sapien":
            _raise_missing_sapien_dependency(exc)
        raise
    try:
        env_class = getattr(envs_module, task_name)
    except AttributeError as exc:
        raise SystemExit(f"No task named `{task_name}`.") from exc
    return env_class()


def _get_camera_config(root: Path, camera_type: str) -> dict[str, Any]:
    config_path = root / "task_config" / "_camera_config.yml"
    with config_path.open("r", encoding="utf-8") as file:
        configs = yaml.load(file.read(), Loader=yaml.FullLoader)
    if camera_type not in configs:
        raise KeyError(f"Camera `{camera_type}` is not defined in {config_path}")
    return configs[camera_type]


def _extract_rgb_frame(camera_data: Any) -> np.ndarray | None:
    if not isinstance(camera_data, dict) or "rgb" not in camera_data:
        return None

    frame = np.asarray(camera_data["rgb"])
    if frame.ndim != 3 or frame.shape[-1] != 3:
        return None
    if frame.dtype != np.uint8:
        frame = np.clip(np.rint(frame), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(frame)


def _collect_eval_video_frames(observation: dict[str, Any]) -> list[np.ndarray]:
    observation_block = observation.get("observation")
    if not isinstance(observation_block, dict):
        raise KeyError("Observation does not contain an `observation` camera block.")

    preferred_names = ("head_camera", "left_camera", "right_camera")
    ordered_names: list[str] = []
    seen_names: set[str] = set()

    for name in preferred_names:
        if name in observation_block:
            ordered_names.append(name)
            seen_names.add(name)

    for name in sorted(observation_block):
        if name not in seen_names:
            ordered_names.append(name)

    frames: list[np.ndarray] = []
    for name in ordered_names:
        frame = _extract_rgb_frame(observation_block.get(name))
        if frame is not None:
            frames.append(frame)

    if not frames:
        raise KeyError(
            f"No RGB camera views found in observation keys: {sorted(observation_block.keys())}"
        )
    return frames


def _pad_frame_to_height(frame: np.ndarray, target_height: int) -> np.ndarray:
    height = int(frame.shape[0])
    if height == target_height:
        return frame

    pad_total = target_height - height
    if pad_total < 0:
        raise ValueError(
            f"Cannot pad frame with height {height} to smaller target height {target_height}."
        )

    pad_top = pad_total // 2
    pad_bottom = pad_total - pad_top
    return np.pad(
        frame,
        ((pad_top, pad_bottom), (0, 0), (0, 0)),
        mode="constant",
        constant_values=0,
    )


def _compose_eval_video_frame(observation: dict[str, Any]) -> np.ndarray:
    frames = _collect_eval_video_frames(observation)
    target_height = max(int(frame.shape[0]) for frame in frames)
    padded_frames = [_pad_frame_to_height(frame, target_height) for frame in frames]

    if len(padded_frames) == 1:
        return np.ascontiguousarray(padded_frames[0])

    separator = np.zeros((target_height, 2, 3), dtype=np.uint8)
    stitched_frames: list[np.ndarray] = []
    for index, frame in enumerate(padded_frames):
        if index:
            stitched_frames.append(separator)
        stitched_frames.append(frame)
    return np.ascontiguousarray(np.concatenate(stitched_frames, axis=1))


def _start_eval_video_ffmpeg(output_path: Path, frame: np.ndarray) -> subprocess.Popen:
    height, width = frame.shape[:2]
    return subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            f"{width}x{height}",
            "-framerate",
            "10",
            "-i",
            "-",
            "-pix_fmt",
            "yuv420p",
            "-vcodec",
            "libopenh264",
            "-crf",
            "23",
            str(output_path),
        ],
        stdin=subprocess.PIPE,
    )


def _write_eval_video_frame(ffmpeg: subprocess.Popen, frame: np.ndarray) -> None:
    if ffmpeg.stdin is None:
        raise RuntimeError("Eval video ffmpeg process has no stdin pipe.")
    ffmpeg.stdin.write(np.ascontiguousarray(frame).tobytes())


def _close_eval_video_ffmpeg(ffmpeg: subprocess.Popen | None) -> None:
    if ffmpeg is None:
        return

    if ffmpeg.stdin is not None and not ffmpeg.stdin.closed:
        ffmpeg.stdin.close()

    try:
        return_code = ffmpeg.wait(timeout=10)
    except subprocess.TimeoutExpired:
        ffmpeg.kill()
        return_code = ffmpeg.wait(timeout=5)

    if return_code != 0:
        LOGGER.warning("Eval video ffmpeg exited with non-zero return code %s.", return_code)


def _snapshot_task_env_observation(task_env: Any) -> dict[str, Any]:
    task_env.cameras.update_picture()
    observation_block = task_env.cameras.get_config()
    rgb = task_env.cameras.get_rgb()
    for camera_name, camera_rgb in rgb.items():
        camera_entry = observation_block.setdefault(camera_name, {})
        if isinstance(camera_entry, dict):
            camera_entry.update(camera_rgb)
        else:
            observation_block[camera_name] = dict(camera_rgb)
    return {"observation": observation_block}


def _estimate_endpose_motion(previous_pose_16d: np.ndarray, current_pose_16d: np.ndarray) -> float:
    previous_pose_16d = np.asarray(previous_pose_16d, dtype=np.float32).reshape(16)
    current_pose_16d = np.asarray(current_pose_16d, dtype=np.float32).reshape(16)
    left_translation = float(np.linalg.norm(current_pose_16d[:3] - previous_pose_16d[:3]))
    right_translation = float(np.linalg.norm(current_pose_16d[8:11] - previous_pose_16d[8:11]))
    left_gripper = float(abs(current_pose_16d[7] - previous_pose_16d[7]))
    right_gripper = float(abs(current_pose_16d[15] - previous_pose_16d[15]))
    return max(left_translation, right_translation, left_gripper, right_gripper)


class _EvalVideoRecorder:
    def __init__(self, output_path: Path, action_render_stride: int) -> None:
        self.output_path = output_path
        self.action_render_stride = max(int(action_render_stride), 1)
        self.ffmpeg: subprocess.Popen | None = None
        self.disabled = False
        self._action_active = False
        self._action_render_count = 0
        self._restore_env: Any = None

    def install_on_env(self, task_env: Any) -> None:
        original_update_render = task_env._update_render

        def _wrapped_update_render(*args, **kwargs):
            result = original_update_render(*args, **kwargs)
            if self._action_active and not self.disabled:
                self._action_render_count += 1
                if self._action_render_count % self.action_render_stride == 0:
                    self.record_observation(_snapshot_task_env_observation(task_env))
            return result

        task_env._update_render = _wrapped_update_render
        self._restore_env = lambda: setattr(task_env, "_update_render", original_update_render)

    def uninstall_from_env(self) -> None:
        if self._restore_env is not None:
            self._restore_env()
            self._restore_env = None

    def begin_action(self) -> None:
        self._action_active = True
        self._action_render_count = 0

    def end_action(self) -> None:
        self._action_active = False

    def record_observation(self, observation: dict[str, Any]) -> None:
        if self.disabled:
            return

        try:
            frame = _compose_eval_video_frame(observation)
            if self.ffmpeg is None:
                self.ffmpeg = _start_eval_video_ffmpeg(self.output_path, frame)
            _write_eval_video_frame(self.ffmpeg, frame)
        except Exception as exc:
            LOGGER.warning(
                "Disabling eval video recording for %s after frame export failure: %s",
                self.output_path,
                exc,
            )
            self.disabled = True
            _close_eval_video_ffmpeg(self.ffmpeg)
            self.ffmpeg = None

    def close(self) -> None:
        self.end_action()
        self.uninstall_from_env()
        _close_eval_video_ffmpeg(self.ffmpeg)
        self.ffmpeg = None


def _get_embodiment_config(robot_file: str) -> dict[str, Any]:
    with open(Path(robot_file) / "config.yml", "r", encoding="utf-8") as file:
        return yaml.load(file.read(), Loader=yaml.FullLoader)


def _format_obs(observation: dict[str, Any], prompt: str) -> dict[str, Any]:
    return {
        "observation.images.cam_high": observation["observation"]["head_camera"]["rgb"],
        "observation.images.cam_left_wrist": observation["observation"]["left_camera"]["rgb"],
        "observation.images.cam_right_wrist": observation["observation"]["right_camera"]["rgb"],
        "observation.state": observation["joint_action"]["vector"],
        "task": prompt,
    }


def _extract_current_pose_16d(observation: dict[str, Any]) -> np.ndarray:
    return np.array(
        observation["endpose"]["left_endpose"]
        + [observation["endpose"]["left_gripper"]]
        + observation["endpose"]["right_endpose"]
        + [observation["endpose"]["right_gripper"]],
        dtype=np.float32,
    )


def _build_fallback_episode_info(task_name: str, task_env: Any) -> dict[str, Any] | None:
    if task_name == "adjust_bottle":
        model_id = getattr(task_env, "model_id", None)
        qpose_tag = getattr(task_env, "qpose_tag", None)
        if model_id is None or qpose_tag is None:
            return None
        arm_tag = "right" if int(qpose_tag) == 1 else "left"
        return {
            "info": {
                "{A}": f"001_bottle/base{int(model_id)}",
                "{a}": arm_tag,
            }
        }
    return None


def _choose_instruction(
    task_name: str,
    episode_info: dict[str, Any] | None,
    instruction_type: str,
    max_descriptions: int,
) -> str:
    if episode_info is not None:
        from description.utils.generate_episode_instructions import generate_episode_descriptions

        results = generate_episode_descriptions(task_name, [episode_info["info"]], max_descriptions)
        if results:
            candidates = results[0].get(instruction_type) or []
            if candidates:
                return str(np.random.choice(candidates))
    return task_name.replace("_", " ")


def _compose_relative_pose(relative_pose: np.ndarray, initial_pose: np.ndarray) -> np.ndarray:
    relative_rotation = R.from_quat(relative_pose[3:7][None])
    initial_rotation = R.from_quat(initial_pose[3:7][None])
    absolute_rotation = (initial_rotation * relative_rotation).as_quat().reshape(-1)
    absolute_rotation = absolute_rotation / np.linalg.norm(absolute_rotation)
    absolute_translation = relative_pose[:3] + initial_pose[:3]
    return np.concatenate([absolute_translation, absolute_rotation, relative_pose[7:8]])


def _compact_action_to_absolute_ee(action_16d: np.ndarray, initial_pose_16d: np.ndarray) -> np.ndarray:
    left = _compose_relative_pose(action_16d[:8], initial_pose_16d[:8])
    right = _compose_relative_pose(action_16d[8:], initial_pose_16d[8:])
    return np.concatenate([left, right]).astype(np.float32)


def _write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def _build_eval_args(root: Path, user_cfg: dict[str, Any]) -> dict[str, Any]:
    task_name = user_cfg["task_name"]
    task_config = user_cfg["task_config"]
    task_config_path = root / "task_config" / f"{task_config}.yml"
    with task_config_path.open("r", encoding="utf-8") as file:
        args = yaml.load(file.read(), Loader=yaml.FullLoader)

    args["task_name"] = task_name
    args["task_config"] = task_config
    args["ckpt_setting"] = user_cfg["ckpt_setting"]
    args["policy_name"] = user_cfg["policy_name"]

    embodiment_type = args.get("embodiment")
    with open(root / "task_config" / "_embodiment_config.yml", "r", encoding="utf-8") as file:
        embodiment_types = yaml.load(file.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(name: str) -> str:
        robot_file = embodiment_types[name]["file_path"]
        if robot_file is None:
            raise ValueError(f"Embodiment `{name}` has no file_path configured.")
        return robot_file

    with open(root / "task_config" / "_camera_config.yml", "r", encoding="utf-8") as file:
        camera_cfg = yaml.load(file.read(), Loader=yaml.FullLoader)

    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = camera_cfg[head_camera_type]["h"]
    args["head_camera_w"] = camera_cfg[head_camera_type]["w"]

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise ValueError("embodiment items should have length 1 or 3")

    args["left_embodiment_config"] = _get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = _get_embodiment_config(args["right_robot_file"])
    return args


def _eval_remote_policy(user_cfg: dict[str, Any]) -> None:
    robowin_root = _bootstrap_robowin(user_cfg["robowin_root"])
    _configure_robowin_rendering()

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    args = _build_eval_args(robowin_root, user_cfg)
    task_name = args["task_name"]

    task_env = _class_decorator(task_name)
    model = WebsocketClientPolicy(host=user_cfg["host"], port=user_cfg["port"])

    left_arm_dim = len(args["left_embodiment_config"]["arm_joints_name"][0])
    right_arm_dim = len(args["right_embodiment_config"]["arm_joints_name"][1])
    LOGGER.info("Loaded RoboTwin task=%s left_arm_dim=%s right_arm_dim=%s", task_name, left_arm_dim, right_arm_dim)

    save_dir = (
        Path(user_cfg["save_root"]).expanduser().resolve()
        / task_name
        / user_cfg["policy_name"]
        / args["task_config"]
        / args["ckpt_setting"]
        / current_time
    )
    save_dir.mkdir(parents=True, exist_ok=True)

    st_seed = 100000 * (1 + int(user_cfg["seed"]))
    task_env.suc = 0
    task_env.test_num = 0
    succ_seed = 0
    now_seed = st_seed
    now_id = 0
    clear_cache_freq = int(args["clear_cache_freq"])
    args["eval_mode"] = True

    while succ_seed < int(user_cfg["test_num"]):
        render_freq = args["render_freq"]
        args["render_freq"] = 0

        episode_info = None
        demo_failure: Exception | None = None
        demo_ready = False
        try:
            task_env.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
            episode_info = task_env.play_once()
            demo_ready = bool(task_env.plan_success and task_env.check_success())
        except Exception as exc:
            demo_failure = exc
            if exc.__class__.__name__ == "UnStableError":
                task_env.close_env()
                now_seed += 1
                args["render_freq"] = render_freq
                continue

        fallback_episode_info = _build_fallback_episode_info(task_name, task_env)
        task_env.close_env()
        args["render_freq"] = render_freq

        if not demo_ready:
            if fallback_episode_info is None:
                now_seed += 1
                if demo_failure is not None:
                    LOGGER.warning("Skipping unstable seed=%s because setup failed: %s", now_seed - 1, demo_failure)
                continue
            episode_info = fallback_episode_info
            if demo_failure is not None:
                LOGGER.warning(
                    "Expert demo failed for seed=%s (%s); using fallback task metadata for instruction generation.",
                    now_seed,
                    demo_failure,
                )
            else:
                LOGGER.warning(
                    "Expert demo failed validation for seed=%s; using fallback task metadata for instruction generation.",
                    now_seed,
                )

        succ_seed += 1
        task_env.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
        instruction = _choose_instruction(
            task_name,
            episode_info,
            str(user_cfg["instruction_type"]),
            int(user_cfg["test_num"]),
        )
        task_env.set_instruction(instruction=instruction)
        model.reset(prompt=instruction)

        eval_video_path = save_dir / f"episode{task_env.test_num}.mp4" if args["eval_video_log"] else None
        eval_video_recorder = (
            _EvalVideoRecorder(
                output_path=eval_video_path,
                action_render_stride=_env_int("OPEN_P2P_ROBOTWIN_EVAL_VIDEO_ACTION_RENDER_STRIDE", 10),
            )
            if eval_video_path is not None
            else None
        )

        if eval_video_recorder is not None:
            eval_video_recorder.install_on_env(task_env)

        observation = task_env.get_obs()
        stalled_action_streak = 0
        succ = False
        try:
            while task_env.take_action_cnt < task_env.step_lim:
                if eval_video_recorder is not None:
                    eval_video_recorder.record_observation(observation)

                current_pose_16d = _extract_current_pose_16d(observation)
                payload = _format_obs(observation, instruction)
                result = model.infer({"obs": payload, "ee_pose_16d": current_pose_16d})
                action_16d = np.asarray(result["action_16d"], dtype=np.float32).reshape(-1)
                if action_16d.size != 16:
                    raise ValueError(f"Expected 16-D compact action, got shape {action_16d.shape}.")

                absolute_action_16d = result.get("action_16d_absolute")
                if absolute_action_16d is not None:
                    ee_action = np.asarray(absolute_action_16d, dtype=np.float32).reshape(-1)
                else:
                    reference_pose = np.asarray(
                        result.get("action_reference_16d", current_pose_16d),
                        dtype=np.float32,
                    ).reshape(-1)
                    ee_action = _compact_action_to_absolute_ee(action_16d, reference_pose)
                if ee_action.size != 16:
                    raise ValueError(f"Expected 16-D absolute ee action, got shape {ee_action.shape}.")

                if eval_video_recorder is not None:
                    eval_video_recorder.begin_action()
                try:
                    task_env.take_action(ee_action, action_type="ee")
                finally:
                    if eval_video_recorder is not None:
                        eval_video_recorder.end_action()

                observation = task_env.get_obs()
                next_pose_16d = _extract_current_pose_16d(observation)
                motion_delta = _estimate_endpose_motion(current_pose_16d, next_pose_16d)
                if motion_delta < 1e-4 and not task_env.eval_success:
                    stalled_action_streak += 1
                    if stalled_action_streak == 10 or stalled_action_streak % 25 == 0:
                        LOGGER.warning(
                            "Detected %s consecutive near-static eval actions at episode=%s step=%s (motion_delta=%.6f). "
                            "This usually means the planner failed silently or the policy saturated to an almost identical target pose.",
                            stalled_action_streak,
                            task_env.test_num,
                            task_env.take_action_cnt,
                            motion_delta,
                        )
                else:
                    stalled_action_streak = 0

                if task_env.eval_success:
                    succ = True
                    if eval_video_recorder is not None:
                        eval_video_recorder.record_observation(observation)
                    break
        finally:
            if eval_video_recorder is not None:
                eval_video_recorder.close()

        if succ:
            task_env.suc += 1
            LOGGER.info("Episode %s succeeded at seed=%s", task_env.test_num, now_seed)
        else:
            LOGGER.info("Episode %s failed at seed=%s", task_env.test_num, now_seed)

        now_id += 1
        task_env.close_env(clear_cache=((succ_seed + 1) % clear_cache_freq == 0))
        if task_env.render_freq:
            task_env.viewer.close()

        task_env.test_num += 1
        metrics = {
            "timestamp": current_time,
            "task_name": task_name,
            "task_config": args["task_config"],
            "policy_name": user_cfg["policy_name"],
            "ckpt_setting": args["ckpt_setting"],
            "instruction_type": user_cfg["instruction_type"],
            "success_num": float(task_env.suc),
            "total_num": float(task_env.test_num),
            "success_rate": float(task_env.suc / max(task_env.test_num, 1)),
            "latest_seed": int(now_seed),
            "server_host": user_cfg["host"],
            "server_port": int(user_cfg["port"]),
        }
        _write_json(metrics, save_dir / "res.json")
        with (save_dir / "_result.txt").open("w", encoding="utf-8") as file:
            file.write(f"Timestamp: {current_time}\n\n")
            file.write(f"Instruction Type: {user_cfg['instruction_type']}\n\n")
            file.write(json.dumps(metrics, indent=2, ensure_ascii=False))

        LOGGER.info(
            "%s | %s | %s | %s => %.1f%% (%s/%s), current seed=%s",
            task_name,
            user_cfg["policy_name"],
            args["task_config"],
            args["ckpt_setting"],
            round(task_env.suc / task_env.test_num * 100, 1),
            task_env.suc,
            task_env.test_num,
            now_seed,
        )
        now_seed += 1


def parse_args_and_config() -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="Evaluate the remote Open P2P RoboTwin policy.")
    parser.add_argument("--config", required=True, help="Path to the evaluation yaml config.")
    parser.add_argument("--overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if args.overrides:
        if len(args.overrides) % 2 != 0:
            raise ValueError("Overrides must be provided as --key value pairs.")
        for index in range(0, len(args.overrides), 2):
            key = args.overrides[index].lstrip("-")
            value = args.overrides[index + 1]
            try:
                value = eval(value)
            except Exception:
                pass
            config[key] = value

    return config


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cfg = parse_args_and_config()
    _eval_remote_policy(cfg)


if __name__ == "__main__":
    main()

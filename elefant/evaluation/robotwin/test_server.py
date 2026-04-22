from types import SimpleNamespace

import numpy as np
import torch

from elefant.evaluation.robotwin.server import RobotTwinSequenceInferenceState


def _make_pose(offset: float) -> np.ndarray:
    pose = np.zeros(16, dtype=np.float32)
    pose[0] = offset
    pose[3:7] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    pose[8] = offset + 10.0
    pose[11:15] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return pose


class _DummyDatasetConfig:
    def get_robot_action_dim(self) -> int:
        return 16


class _DummyModel:
    def online_full_predict(
        self,
        frames: torch.Tensor,
        actions: torch.Tensor,
        text_tokens_embed=None,
        initial_sample=None,
        compile: bool = True,
    ) -> torch.Tensor:
        del frames, text_tokens_embed, initial_sample, compile
        return torch.zeros_like(actions)


def _make_state() -> RobotTwinSequenceInferenceState:
    config = SimpleNamespace(
        shared=SimpleNamespace(n_seq_timesteps=3),
        stage3_finetune=SimpleNamespace(training_dataset=_DummyDatasetConfig()),
    )
    state = RobotTwinSequenceInferenceState(
        model=_DummyModel(),
        config=config,
        device=torch.device("cpu"),
        compile=False,
        text_tokenizer=None,
        action_adapter=SimpleNamespace(),
        legacy_inference=False,
    )
    state.frame_history = torch.zeros((1, 3, 3, 2, 2), dtype=torch.uint8)
    state.pose_history[:] = np.stack(
        [_make_pose(1.0), _make_pose(2.0), _make_pose(3.0)],
        axis=0,
    )
    state.n_prior_frames = state.seq_len
    return state


def test_step_uses_pre_roll_anchor_for_rebase():
    state = _make_state()
    captured: dict[str, np.ndarray] = {}

    def _capture_rebase(previous_anchor_pose_16d, new_anchor_pose_16d):
        captured["previous"] = previous_anchor_pose_16d.copy()
        captured["new"] = new_anchor_pose_16d.copy()

    def _capture_warm_start(previous_anchor_pose_16d, new_anchor_pose_16d):
        captured["warm_previous"] = previous_anchor_pose_16d.copy()
        captured["warm_new"] = new_anchor_pose_16d.copy()

    state._rebase_action_history_after_roll = _capture_rebase
    state._roll_warm_start_predictions = _capture_warm_start
    state._stabilize_normalized_action = lambda action, anchor_pose_16d, current_pose_16d: np.zeros(
        state.action_dim, dtype=np.float32
    )

    _, anchor_pose_16d = state.step(
        frame=torch.zeros((3, 2, 2), dtype=torch.uint8),
        current_pose_16d=_make_pose(4.0),
    )

    np.testing.assert_allclose(captured["previous"], _make_pose(1.0))
    np.testing.assert_allclose(captured["new"], _make_pose(2.0))
    np.testing.assert_allclose(captured["warm_previous"], _make_pose(1.0))
    np.testing.assert_allclose(captured["warm_new"], _make_pose(2.0))
    np.testing.assert_allclose(anchor_pose_16d, _make_pose(2.0))
    np.testing.assert_allclose(state.pose_history[0], _make_pose(2.0))
    np.testing.assert_allclose(state.pose_history[-1], _make_pose(4.0))

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from habitat_baselines.config.default_structured_configs import AuxLossConfig
from habitat_baselines.rl.ppo.policy import Net
from hydra.core.config_store import ConfigStore
from dataclasses import dataclass

# 使用延迟导入避免循环依赖
def get_baseline_registry():
    from habitat_baselines.common.baseline_registry import baseline_registry
    return baseline_registry

@dataclass
class PeopleCountingLossConfig(AuxLossConfig):
    """People Counting predictive coding loss"""
    max_human_num: int = 6
    loss_scale: float = 0.1

@dataclass
class GuessHumanPositionLossConfig(AuxLossConfig):
    """Guess Human Position predictive coding loss"""
    max_human_num: int = 6
    position_dim: int = 2
    loss_scale: float = 0.1

@dataclass
class FutureTrajectoryPredictionLossConfig(AuxLossConfig):
    """Future Trajectory predictive coding loss"""
    max_human_num: int = 6
    future_step: int = 4
    loss_scale: float = 0.1

# 获取 baseline_registry
baseline_registry = get_baseline_registry()

@baseline_registry.register_auxiliary_loss(name="people_counting")
class PeopleCounting(nn.Module):
    """People Counting task: Estimate the number of people in the scene."""
    def __init__(self, action_space: gym.spaces.Box, net: Net, max_human_num: int = 6, loss_scale: float = 0.1):
        super().__init__()
        self.max_human_num = max_human_num
        self.loss_scale = loss_scale
        hidden_size = net.output_size

        # LSTM for temporal processing
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)

        # Attention mechanism
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=4, batch_first=True)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, max_human_num + 1)
        )

        # CrossEntropy loss
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, aux_loss_state, batch):
        scene_features = aux_loss_state["rnn_output"]  # (batch_size, hidden_size)
        lstm_output, _ = self.lstm(scene_features.unsqueeze(1))  # Add sequence dim
        attn_output, _ = self.attention(lstm_output, lstm_output, lstm_output)
        logits = self.classifier(attn_output.mean(dim=1))  # Mean pooling
        logits = torch.clamp(logits, min=-10, max=10)

        target = batch["observations"]["human_num_sensor"].squeeze(-1).long()
        ori_loss = self.loss_fn(logits, target)
        sigmoid_loss = torch.sigmoid(ori_loss)
        loss = self.loss_scale * sigmoid_loss
        return dict(loss=loss)

@baseline_registry.register_auxiliary_loss(name="guess_human_position")
class GuessHumanPosition(nn.Module):
    """Predict human positions relative to the agent."""
    def __init__(self, action_space: gym.spaces.Box, net: Net, max_human_num: int = 6, position_dim: int = 2, loss_scale: float = 0.1):
        super().__init__()
        self.loss_scale = loss_scale
        hidden_size = net.output_size
        self.position_dim = position_dim
        self.max_human_num = max_human_num

        self.lstm = nn.LSTM(input_size=hidden_size + 1, hidden_size=hidden_size, batch_first=True)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=4, batch_first=True)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, max_human_num * position_dim)
        )
        self.loss_fn = nn.MSELoss(reduction="none")

    def forward(self, aux_loss_state, batch):
        scene_features = aux_loss_state["rnn_output"]
        human_num_features = batch["observations"]["human_num_sensor"].float()
        features = torch.cat((scene_features, human_num_features), dim=-1)

        lstm_output, _ = self.lstm(features)
        attn_output, _ = self.attention(lstm_output, lstm_output, lstm_output)
        positions_pred = self.classifier(attn_output.mean(dim=1)).view(-1, self.max_human_num, self.position_dim)

        positions_gt = batch["observations"]["oracle_humanoid_future_trajectory"][:, :, 0, :]
        positions_gt_agent0 = batch["observations"]["localization_sensor"][:, [0, 2]]
        positions_gt_relative = positions_gt - positions_gt_agent0.unsqueeze(1).repeat(1, 6, 1)

        mask = (positions_gt != -100.0).all(dim=-1).unsqueeze(-1)
        loss_per_position = self.loss_fn(positions_pred, positions_gt_relative)
        masked_loss = loss_per_position * mask

        if mask.sum() < 1:
            loss = torch.norm(loss_per_position) / 1e5
        else:
            loss = masked_loss.sum() / mask.sum()

        sigmoid_loss = torch.sigmoid(loss)
        return dict(loss=sigmoid_loss * self.loss_scale)

@baseline_registry.register_auxiliary_loss(name="future_trajectory_prediction")
class FutureTrajectoryPrediction(nn.Module):
    """Predict future trajectories for humans."""
    def __init__(self, action_space: gym.spaces.Box, net: Net, max_human_num: int = 6, position_dim: int = 2, loss_scale: float = 0.1, future_step: int = 4):
        super().__init__()
        self.max_human_num = max_human_num
        self.position_dim = position_dim
        self.future_step = future_step
        self.loss_scale = loss_scale
        hidden_size = net.output_size

        self.lstm = nn.LSTM(input_size=hidden_size + 1 + max_human_num * position_dim, hidden_size=hidden_size, num_layers=2, bidirectional=True, batch_first=True)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size * 2, num_heads=4, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, max_human_num * future_step * position_dim)
        )
        self.loss_fn = nn.MSELoss(reduction="none")

    def forward(self, aux_loss_state, batch):
        scene_features = aux_loss_state["rnn_output"]
        batch_size = scene_features.size(0)
        human_num_features = batch["observations"]["human_num_sensor"].float()
        position_features = batch["observations"]["oracle_humanoid_future_trajectory"][:, :, 0, :].reshape(batch_size, -1)
        features = torch.cat((scene_features, human_num_features, position_features), dim=-1)

        lstm_output, _ = self.lstm(features.unsqueeze(1))
        attn_output, _ = self.attention(lstm_output, lstm_output, lstm_output)
        positions_pred = self.classifier(attn_output.mean(dim=1)).view(batch_size, self.max_human_num, self.future_step, self.position_dim)

        positions_gt = batch["observations"]["oracle_humanoid_future_trajectory"][:, :, -self.future_step:, :]
        positions_gt_agent0 = batch["observations"]["localization_sensor"][:, [0, 2]]
        positions_gt_relative = positions_gt - positions_gt_agent0.unsqueeze(1).unsqueeze(2).repeat(1, self.max_human_num, self.future_step, 1)

        mask = (positions_gt != -100.0).all(dim=-1).unsqueeze(-1)
        loss_per_position = self.loss_fn(positions_pred, positions_gt_relative)
        masked_loss = loss_per_position * mask

        if mask.sum() < 1:
            loss = torch.norm(loss_per_position) / 1e5
        else:
            loss = masked_loss.sum() / mask.sum()

        sigmoid_loss = torch.sigmoid(loss)
        return dict(loss=sigmoid_loss * self.loss_scale)

# Register auxiliary loss configurations
cs = ConfigStore.instance()

cs.store(
    package="habitat_baselines.rl.auxiliary_losses.people_counting",
    group="habitat_baselines/rl/auxiliary_losses",
    name="people_counting",
    node=PeopleCountingLossConfig,
)

cs.store(
    package="habitat_baselines.rl.auxiliary_losses.guess_human_position",
    group="habitat_baselines/rl/auxiliary_losses",
    name="guess_human_position",
    node=GuessHumanPositionLossConfig,
)

cs.store(
    package="habitat_baselines.rl.auxiliary_losses.future_trajectory_prediction",
    group="habitat_baselines/rl/auxiliary_losses",
    name="future_trajectory_prediction",
    node=FutureTrajectoryPredictionLossConfig,
)

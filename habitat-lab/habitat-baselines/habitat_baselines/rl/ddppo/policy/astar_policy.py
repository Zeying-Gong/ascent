# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os.path as osp
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import gym.spaces as spaces
import numpy as np
import torch
import torch.nn as nn

from habitat.core.spaces import ActionSpace
from habitat.utils.geometry_utils import (
    quaternion_from_coeff,
    quaternion_rotate_vector,
)
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.ppo.policy import Policy, PolicyActionData
from habitat_baselines.utils.common import get_num_actions
from habitat.tasks.utils import cartesian_to_polar

@baseline_registry.register_policy
class ASTARPolicy(nn.Module, Policy):
    def __init__(
        self,
        config,
        full_config,
        observation_space: spaces.Space,
        action_space: ActionSpace,
        orig_action_space: ActionSpace,
        num_envs: int,
        aux_loss_config,
        agent_name: Optional[str],
    ):
        Policy.__init__(self, action_space)
        nn.Module.__init__(self)

        self._num_envs: int = num_envs
        self._recurrent_hidden_size = (
            full_config.habitat_baselines.rl.ppo.hidden_size
        )
        self._device = None
        self.action_distribution_type = "categorical"
        self.turn_threshold = np.deg2rad(15.0)
        self.repeated_turn_count = [0] * self._num_envs
        self.angle_diff = [0] * self._num_envs
        self.max_repeated_turns = 3
        self.angle_threshold_for_forced_forward = np.deg2rad(5.0)

    @classmethod
    def from_config(
        cls,
        config,
        observation_space,
        action_space,
        orig_action_space,
        agent_name=None,
        **kwargs,
    ):
        if agent_name is None:
            if len(config.habitat.simulator.agents_order) > 1:
                raise ValueError(
                    "If there is more than an agent, you need to specify the agent name"
                )
            else:
                agent_name = config.habitat.simulator.agents_order[0]
        return cls(
            config=config.habitat_baselines.rl.policy[agent_name],
            full_config=config,
            observation_space=observation_space,
            action_space=action_space,
            orig_action_space=orig_action_space,
            num_envs=config.habitat_baselines.num_environments,
            aux_loss_config=config.habitat_baselines.rl.auxiliary_losses,
            agent_name=agent_name,
        )
    
    def to(self, device):
        self._device = device
        return super().to(device)
    
    @property
    def hidden_state_shape(self):
        return (
            self.num_recurrent_layers,
            self.recurrent_hidden_size,
        )

    @property
    def hidden_state_shape_lens(self):
        return [self.recurrent_hidden_size]

    @property
    def recurrent_hidden_size(self) -> int:
        return self._recurrent_hidden_size

    @property
    def num_recurrent_layers(self):
        return (0)

    @property
    def should_load_agent_state(self):
        return False
    
    @property
    def policy_action_space(self):
        """
        Fetches the policy action space for learning. If we are learning the HL
        policy, it will return its custom action space for learning.
        """
        return super().policy_action_space
    
    @property
    def policy_action_space_shape_lens(self):
        return [self._action_space]
    
    def parameters(self):
        return iter([nn.Parameter(torch.zeros((1,), device=self._device))])
    
    def get_value(self, observations, rnn_hidden_states, prev_actions, masks):
        # We assign a value of 0. This is needed so that we can concatenate values in multiagent
        # policies
        return torch.zeros(rnn_hidden_states.shape[0], 1).to(
            rnn_hidden_states.device
        )

    def calculate_angle_to_target(self, agent_position, agent_yaw, target_position):
        delta_y = target_position[0] - agent_position[0]
        delta_x = target_position[2] - agent_position[2]
        
        target_angle = torch.atan2( delta_y, - delta_x) # torch.atan2( -delta_y,delta_x)
        
        angle_diff = target_angle - agent_yaw
        return angle_diff

    def normalize_angle(self,angle):
        angle = (angle + torch.pi) % (2 * torch.pi) - torch.pi
        return angle

    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic=False,
        **kwargs,
    ):
        
        batch_size = masks.shape[0]
        
        actions = torch.zeros(
            size=prev_actions.shape,
            device=masks.device,
            dtype=torch.int64,
        )
        for i in range(batch_size):
            if 'oracle_shortest_path_follower_sensor' in observations and observations['oracle_shortest_path_follower_sensor'][i] is not None:
                actions[i] = observations['oracle_shortest_path_follower_sensor'][i].item()
                # print(f'Env:{i}, action:{actions[i]}')
            if actions[i] == 0: # prevent wrong judge from follower
                if 'distance_to_goal_sensor' in observations and observations['distance_to_goal_sensor'][i] <= 0.1:
                    actions[i] = 0
                else:
                    current_point = observations['oracle_shortest_path_sensor'][i, 0, :]
                    try:
                        # 尝试获取下一个点
                        next_point = observations['oracle_shortest_path_sensor'][i, 1, :]
                    except IndexError as e:
                        print(f"IndexError for oracle_shortest_path_sensor at index {i}: {e}")
                        print(f"Observation shape: {observations['oracle_shortest_path_sensor'].shape}")
                        actions[i] = 0
                        continue
                    agent_yaw = observations['localization_sensor'][i, -1]
                    self.angle_diff[i] = self.calculate_angle_to_target(current_point, agent_yaw, next_point)
                    if abs(self.angle_diff[i]) < self.turn_threshold or abs(self.angle_diff[i]) > 2 * torch.pi - self.turn_threshold:
                        actions[i] = 1  
                        self.repeated_turn_count[i] = 0 
                    elif (self.angle_diff[i] > -2 * torch.pi + self.turn_threshold and self.angle_diff[i] < - torch.pi) or (self.angle_diff[i] > self.turn_threshold and self.angle_diff[i] < torch.pi):
                        if prev_actions[i] == 2: # 3:
                            self.repeated_turn_count[i] += 1
                        else:
                            self.repeated_turn_count[i] = 0
                        actions[i] = 3 # 2  # TURN_RIGHT  # TURN_LEFT
                    else:  
                        if prev_actions[i] == 3: # 2:
                            self.repeated_turn_count[i] += 1
                        else:
                            self.repeated_turn_count[i] = 0
                        actions[i] = 2 # 3   # TURN_LEFT # TURN_RIGHT 

                    if self.repeated_turn_count[i] >= self.max_repeated_turns:
                        actions[i] = 1  
                        self.repeated_turn_count[i] = 0  
            
                # print(f"agent_yaw:{agent_yaw}, target_angle:{self.angle_diff[i]+ agent_yaw}, action:{actions[i]}")
                # print(f"current_point:{current_point}, next_point:{next_point}")

        # This will update the prev action
        use_action = actions

        return PolicyActionData(
            take_actions=actions,
            actions=use_action,
            rnn_hidden_states=rnn_hidden_states,
        )
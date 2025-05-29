from typing import Any, Dict, List, Optional, Tuple

import gym.spaces as spaces
import numpy as np
import torch
import torch.nn as nn

from habitat.core.spaces import ActionSpace
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.ppo.policy import Policy, PolicyActionData
from habitat_baselines.utils.common import get_num_actions


@baseline_registry.register_policy
class FixedPolicy(nn.Module, Policy):
    """
    :only apply one action, on my case, that is oracle random walk for humanoids.
    """

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
        self.action_distribution_type = "gaussian"

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
        log_info: List[Dict[str, Any]] = [{} for _ in range(batch_size)]
        # Initialize empty action set based on the overall action space.
        actions = torch.zeros(
            (batch_size, get_num_actions(self._action_space)),
            device=masks.device,
        )
        
        # This will update the prev action
        use_action = actions

        return PolicyActionData(
            take_actions=actions,
            actions=use_action,
            rnn_hidden_states=rnn_hidden_states,
        )
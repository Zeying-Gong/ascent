from habitat import logger
from habitat_baselines import PPOTrainer
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)
from omegaconf import DictConfig


class DummyAgent:
    def __init__(self, actor_critic):
        self.actor_critic = actor_critic

    def load_state_dict(self, *args, **kwargs):
        pass

@baseline_registry.register_trainer(name="nonlearned_policy")
class NonLearnedTrainer(PPOTrainer):
    agent: DummyAgent = None

    def _setup_actor_critic_agent(self, ppo_cfg: DictConfig) -> None:
        r"""Sets up actor critic and agent for PPO.

        Args:
            ppo_cfg: config node with relevant params

        Returns:
            None
        """
        logger.add_filehandler(self.config.habitat_baselines.log_file)

        policy = baseline_registry.get_policy(
            self.config.habitat_baselines.rl.policy.name
        )
        observation_space = self.obs_space
        self.obs_transforms = get_active_obs_transforms(self.config)
        observation_space = apply_obs_transforms_obs_space(
            observation_space, self.obs_transforms
        )

        self.actor_critic = policy.from_config(
            self.config,
            observation_space,
            self.policy_action_space,
            orig_action_space=self.orig_policy_action_space,
        )
        self.obs_space = observation_space
        self.agent = DummyAgent(self.actor_critic)

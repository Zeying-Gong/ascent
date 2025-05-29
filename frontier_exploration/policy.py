import torch
from gym import spaces
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.ppo import Policy

try:
    from habitat_baselines.rl.ppo.policy import PolicyActionData

    POLICY_ACTION_DATA = True
except:
    POLICY_ACTION_DATA = False


from frontier_exploration.base_explorer import BaseExplorer
from frontier_exploration.objnav_explorer import GreedyObjNavExplorer, ObjNavExplorer


@baseline_registry.register_policy
class FrontierExplorationPolicy(Policy):
    def __init__(self, *args, **kwargs):
        super().__init__()

    @property
    def should_load_agent_state(self):
        return False

    @classmethod
    def from_config(
        cls,
        config: "DictConfig",
        observation_space: spaces.Dict,
        action_space,
        **kwargs,
    ):
        return cls()

    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic=False,
    ):
        # Convert obs to torch.long
        if BaseExplorer.cls_uuid in observations:
            sensor_uuid = BaseExplorer.cls_uuid
        elif ObjNavExplorer.cls_uuid in observations:
            sensor_uuid = ObjNavExplorer.cls_uuid
        elif GreedyObjNavExplorer.cls_uuid in observations:
            sensor_uuid = GreedyObjNavExplorer.cls_uuid
        elif "teacher_label" in observations:
            sensor_uuid = "teacher_label"
        else:
            raise RuntimeError("FrontierExplorationPolicy needs an exploration sensor")
        action = observations[sensor_uuid].type(torch.long)
        if POLICY_ACTION_DATA:
            return PolicyActionData(actions=action, rnn_hidden_states=rnn_hidden_states)
        return None, action, None, rnn_hidden_states

    # used in ppo_trainer.py eval:

    def to(self, *args, **kwargs):
        return

    def eval(self):
        return

    def parameters(self):
        yield torch.zeros(1)

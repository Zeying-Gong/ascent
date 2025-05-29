from omegaconf import DictConfig
from habitat.core.registry import registry
import os
import habitat_sim
from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
from habitat.core.simulator import Observations
@registry.register_simulator(name="Sim-v2")
class HabitatSim_v2(HabitatSim):
    r"""Simulator wrapper over habitat-sim, with 3rd view locobot rendering

    habitat-sim repo: https://github.com/facebookresearch/habitat-sim

    Args:
        config: configuration for initializing the simulator.
    """

    def __init__(self, config: DictConfig) -> None:
        super().__init__(config)
        obj_templates_mgr = self.get_object_template_manager()
        locobot_template_id = obj_templates_mgr.load_configs(
            str(os.path.join("data/", "objects/locobot_merged")) # data_path = "data/" defaultly
        )[0]
        rigid_obj_mgr = self.get_rigid_object_manager()
        locobot = rigid_obj_mgr.add_object_by_template_id(
            locobot_template_id, self.agents[0].scene_node
        )
        # set the agent's body to kinematic since we will be updating position manually
        locobot.motion_type = habitat_sim.physics.MotionType.KINEMATIC

    def reset(self) -> Observations:
        sim_obs = super().reset()
        obj_templates_mgr = self.get_object_template_manager()
        locobot_template_id = obj_templates_mgr.load_configs(
            str(os.path.join("data/", "objects/locobot_merged")) # data_path = "data/" defaultly
        )[0]
        rigid_obj_mgr = self.get_rigid_object_manager()
        locobot = rigid_obj_mgr.add_object_by_template_id(
            locobot_template_id, self.agents[0].scene_node
        )
        # set the agent's body to kinematic since we will be updating position manually
        locobot.motion_type = habitat_sim.physics.MotionType.KINEMATIC
        if self._update_agents_state():
            sim_obs = self.get_sensor_observations()

        self._prev_sim_obs = sim_obs
        if self.config.enable_batch_renderer:
            self.add_keyframe_to_observations(sim_obs)
            return sim_obs
        else:
            return self._sensor_suite.get_observations(sim_obs)
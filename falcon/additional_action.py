# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.
import os
import csv
from gym import spaces
from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
from habitat.core.logging import logger
# from habitat_baselines.common.baseline_registry import baseline_registry
from habitat.config.default_structured_configs import (
    ActionConfig,
)

from hydra.core.config_store import ConfigStore
from typing import Dict, Optional, List, Tuple
from habitat.core.spaces import ActionSpace, EmptySpace, Space
from dataclasses import dataclass, field
from habitat.core.registry import registry
from habitat.tasks.rearrange.actions.actions import (
    BaseVelAction
)
from habitat.tasks.rearrange.rearrange_sim import RearrangeSim
from habitat.tasks.utils import get_angle
import magnum as mn
from habitat.datasets.rearrange.navmesh_utils import get_largest_island_index
import habitat_sim
from habitat.datasets.rearrange.navmesh_utils import SimpleVelocityControlEnv
from habitat.tasks.rearrange.social_nav.utils import (
    robot_human_vec_dot_product,
)
from habitat.tasks.rearrange.actions.actions import HumanoidJointAction
from habitat.tasks.rearrange.utils import place_agent_at_dist_from_pos
from habitat.articulated_agent_controllers import HumanoidRearrangeController

play_i = 0

@registry.register_task_action
class DiscreteStopAction(BaseVelAction):
    def __init__(self, *args, config, sim: RearrangeSim, **kwargs):
        super().__init__(*args, config=config, sim=sim, **kwargs)
        self._checkpoint = self._config.get("leg_animation_checkpoint")
        self._use_range = self._config.get("use_range")
        assert os.path.exists(self._checkpoint) == 1
        self._leg_data = {}  # type: ignore
        kwargs['task'].is_stop_called = False
        
    @property
    def action_space(self):
        return EmptySpace()
    
    def reset(self, *args, **kwargs):
        kwargs['task'].is_stop_called = False  # type: ignore

    def step(self, *args, **kwargs):
        kwargs['task'].is_stop_called = True  # type: ignore
        # kwargs['task'].should_end = True  # type: ignore
        self.base_vel_ctrl.linear_velocity = mn.Vector3(0.0, 0, 0)
        self.base_vel_ctrl.angular_velocity = mn.Vector3(0, 0.0, 0)

@registry.register_task_action
class DiscreteMoveForwardAction(BaseVelAction):
    def __init__(self, *args, config, sim: RearrangeSim, **kwargs):
        super().__init__(*args, config=config, sim=sim, **kwargs)
        self._checkpoint = self._config.get("leg_animation_checkpoint")
        self._use_range = self._config.get("use_range")
        assert os.path.exists(self._checkpoint) == 1
        self._leg_data = {}  # type: ignore
        self._load_animation()

        self._play_length_data = len(self._leg_data)
        self._play_i_perframe = self._config.get("play_i_perframe")
        self.lin_vel = config['lin_speed']
        self.ang_vel = config['ang_speed']
    def _load_animation(self):
        first_row = True
        time_i = 0
        with open(self._checkpoint, newline="") as csvfile:
            spamreader = csv.reader(csvfile, delimiter=" ", quotechar="|")
            for row in spamreader:
                if not first_row:
                    if (
                        time_i >= self._use_range[0]
                        and time_i < self._use_range[1]
                    ):
                        joint_angs = row[0].split(",")[1:13]
                        joint_angs = [float(i) for i in joint_angs]
                        self._leg_data[
                            time_i - self._use_range[0]
                        ] = joint_angs
                    time_i += 1
                first_row = False
    @property
    def action_space(self):
        return EmptySpace()
    
    def step(self, *args, **kwargs):
        global play_i
        lin_vel = self.lin_vel
        ang_vel = self.ang_vel

        self.base_vel_ctrl.linear_velocity = mn.Vector3(lin_vel, 0, 0)
        self.base_vel_ctrl.angular_velocity = mn.Vector3(0, ang_vel, 0)

        if lin_vel != 0.0 or ang_vel != 0.0:
            self.update_base(fix_leg=False)
            cur_i = int(play_i % self._play_length_data)
            self.cur_articulated_agent.leg_joint_pos = self._leg_data[cur_i]
            play_i += self._play_i_perframe
        else:
            play_i = 0
            # Fix the leg joints
            self.cur_articulated_agent.leg_joint_pos = (
                self.cur_articulated_agent.params.leg_init_params
            )

@registry.register_task_action
class DiscreteTurnLeftAction(BaseVelAction):
    def __init__(self, *args, config, sim: RearrangeSim, **kwargs):
        super().__init__(*args, config=config, sim=sim, **kwargs)
        self.lin_vel = config['lin_speed']
        self.ang_vel = config['ang_speed']
    @property
    def action_space(self):
        return EmptySpace()
    
    def step(self, *args, **kwargs):
        lin_vel = self.lin_vel
        ang_vel = self.ang_vel

        self.base_vel_ctrl.linear_velocity = mn.Vector3(lin_vel, 0, 0)
        self.base_vel_ctrl.angular_velocity = mn.Vector3(0, ang_vel, 0)
        self.update_base(fix_leg=False)

@registry.register_task_action
class DiscreteTurnRightAction(BaseVelAction):
    def __init__(self, *args, config, sim: RearrangeSim, **kwargs):
        super().__init__(*args, config=config, sim=sim, **kwargs)
        self.lin_vel = config['lin_speed']
        self.ang_vel = config['ang_speed']
    @property
    def action_space(self):
        return EmptySpace()
    
    def step(self, *args, **kwargs):
        lin_vel = self.lin_vel
        ang_vel = self.ang_vel

        self.base_vel_ctrl.linear_velocity = mn.Vector3(lin_vel, 0, 0)
        self.base_vel_ctrl.angular_velocity = mn.Vector3(0, ang_vel, 0)
        self.update_base(fix_leg=False)

# for human

@registry.register_task_action
class OracleNavAction_wopddl(BaseVelAction, HumanoidJointAction):
    """
    An action that will convert the index of an entity (in the sense of
    `PddlEntity`) to navigate to and convert this to base/humanoid joint control to move the
    robot to the closest navigable position to that entity. The entity index is
    the index into the list of all available entities in the current scene. The
    config flag motion_type indicates whether the low level action will be a base_velocity or
    a joint control.
    """

    def __init__(self, *args, task, **kwargs):
        config = kwargs["config"]
        self.motion_type = config.motion_control
        if self.motion_type == "base_velocity":
            BaseVelAction.__init__(self, *args, **kwargs)

        elif self.motion_type == "human_joints":
            HumanoidJointAction.__init__(self, *args, **kwargs)
            self.humanoid_controller = self.spec_inst_humanoid_controller( # self.lazy_inst_humanoid_controller(
                task, config
            )

        else:
            raise ValueError("Unrecognized motion type for oracle nav action")

        self._task = task
        if hasattr(task,"pddl_problem"):
            self._poss_entities = (
                self._task.pddl_problem.get_ordered_entities_list()
            )
        else:
            self._poss_entities = None
        self._prev_ep_id = None
        self.skill_done = False
        self._targets = {}

    @staticmethod
    def _compute_turn(rel, turn_vel, robot_forward):
        is_left = np.cross(robot_forward, rel) > 0
        if is_left:
            vel = [0, -turn_vel]
        else:
            vel = [0, turn_vel]
        return vel

    def lazy_inst_humanoid_controller(self, task, config):
        # Lazy instantiation of humanoid controller
        # We assign the task with the humanoid controller, so that multiple actions can
        # use it.

        if (
            not hasattr(task, "humanoid_controller")
            or task.humanoid_controller is None
        ):
            # Initialize humanoid controller
            agent_name = self._sim.habitat_config.agents_order[
                self._agent_index
            ]
            walk_pose_path = self._sim.habitat_config.agents[
                agent_name
            ].motion_data_path

            humanoid_controller = HumanoidRearrangeController(walk_pose_path)
            humanoid_controller.set_framerate_for_linspeed(
                config["lin_speed"], config["ang_speed"], self._sim.ctrl_freq
            )
            task.humanoid_controller = humanoid_controller
        return task.humanoid_controller
    
    def spec_inst_humanoid_controller(self, task, config):
        # Instantiation of humanoid controller for specific agent
        # Follow the lazy version, but tell each humanoid agent

        # Initialize humanoid controller
        agent_name = self._sim.habitat_config.agents_order[
            self._agent_index
        ]
        walk_pose_path = self._sim.habitat_config.agents[
            agent_name
        ].motion_data_path

        humanoid_controller = HumanoidRearrangeController(walk_pose_path)
        humanoid_controller.set_framerate_for_linspeed(
            config["lin_speed"], config["ang_speed"], self._sim.ctrl_freq
        )

        # 动态设置 task.{agent_name}_humanoid_controller
        exec("task.{0}_humanoid_controller = humanoid_controller".format(agent_name))

        return getattr(task, "{0}_humanoid_controller".format(agent_name))

    def _update_controller_to_navmesh(self):
        base_offset = self.cur_articulated_agent.params.base_offset
        prev_query_pos = self.cur_articulated_agent.base_pos
        target_query_pos = (
            self.humanoid_controller.obj_transform_base.translation
            + base_offset
        )

        filtered_query_pos = self._sim.step_filter(
            prev_query_pos, target_query_pos, self._agent_index # 1 for agent_1
        )
        fixup = filtered_query_pos - target_query_pos
        self.humanoid_controller.obj_transform_base.translation += fixup

    @property
    def action_space(self):
        return spaces.Dict(
            {
                self._action_arg_prefix
                + "oracle_nav_action": spaces.Box(
                    shape=(1,),
                    low=np.finfo(np.float32).min,
                    high=np.finfo(np.float32).max,
                    dtype=np.float32,
                )
            }
        )

    def reset(self, *args, **kwargs):
        super().reset(*args, **kwargs)
        if self._task._episode_id != self._prev_ep_id:
            self._targets = {}
            self._prev_ep_id = self._task._episode_id
            self.skill_done = False

    def _get_target_for_idx(self, nav_to_target_idx: int):
        if nav_to_target_idx not in self._targets:
            nav_to_obj = self._poss_entities[nav_to_target_idx]
            obj_pos = self._task.pddl_problem.sim_info.get_entity_pos(
                nav_to_obj
            )
            start_pos, _, _ = place_agent_at_dist_from_pos(
                np.array(obj_pos),
                0.0,
                self._config.spawn_max_dist_to_obj,
                self._sim,
                self._config.num_spawn_attempts,
                True,
                self.cur_articulated_agent,
            )
            if self.motion_type == "human_joints":
                self.humanoid_controller.reset(
                    self.cur_articulated_agent.base_transformation
                )
            self._targets[nav_to_target_idx] = (
                np.array(start_pos),
                np.array(obj_pos),
            )
        return self._targets[nav_to_target_idx]

    def _path_to_point(self, point):
        """
        Obtain path to reach the coordinate point. If agent_pos is not given
        the path starts at the agent base pos, otherwise it starts at the agent_pos
        value
        :param point: Vector3 indicating the target point
        """
        agent_pos = self.cur_articulated_agent.base_pos

        path = habitat_sim.ShortestPath()
        path.requested_start = agent_pos
        path.requested_end = point
        found_path = self._sim.pathfinder.find_path(path)
        if not found_path:
            return [agent_pos, point]
        return path.points

    def step(self, *args, **kwargs):
        self.skill_done = False
        nav_to_target_idx = kwargs[
            self._action_arg_prefix + "oracle_nav_action"
        ]

        if nav_to_target_idx <= 0 or self._poss_entities == None or nav_to_target_idx > len(
            self._poss_entities
        ):
            return
        nav_to_target_idx = int(nav_to_target_idx[0]) - 1

        final_nav_targ, obj_targ_pos = self._get_target_for_idx(
            nav_to_target_idx
        )
        base_T = self.cur_articulated_agent.base_transformation
        curr_path_points = self._path_to_point(final_nav_targ)
        robot_pos = np.array(self.cur_articulated_agent.base_pos)

        if curr_path_points is None:
            raise Exception
        else:
            # Compute distance and angle to target
            cur_nav_targ = curr_path_points[1]
            forward = np.array([1.0, 0, 0])
            robot_forward = np.array(base_T.transform_vector(forward))

            # Compute relative target.
            rel_targ = cur_nav_targ - robot_pos

            # Compute heading angle (2D calculation)
            robot_forward = robot_forward[[0, 2]]
            rel_targ = rel_targ[[0, 2]]
            rel_pos = (obj_targ_pos - robot_pos)[[0, 2]]

            angle_to_target = get_angle(robot_forward, rel_targ)
            angle_to_obj = get_angle(robot_forward, rel_pos)

            dist_to_final_nav_targ = np.linalg.norm(
                (final_nav_targ - robot_pos)[[0, 2]]
            )
            at_goal = (
                dist_to_final_nav_targ < self._config.dist_thresh
                and angle_to_obj < self._config.turn_thresh
            )

            if self.motion_type == "base_velocity":
                if not at_goal:
                    if dist_to_final_nav_targ < self._config.dist_thresh:
                        # Look at the object
                        vel = OracleNavAction_wopddl._compute_turn(
                            rel_pos, self._config.turn_velocity, robot_forward
                        )
                    elif angle_to_target < self._config.turn_thresh:
                        # Move towards the target
                        vel = [self._config.forward_velocity, 0]
                    else:
                        # Look at the target waypoint.
                        vel = OracleNavAction_wopddl._compute_turn(
                            rel_targ, self._config.turn_velocity, robot_forward
                        )
                else:
                    vel = [0, 0]
                    self.skill_done = True
                kwargs[f"{self._action_arg_prefix}base_vel"] = np.array(vel)
                BaseVelAction.step(self, *args, **kwargs)
                return

            elif self.motion_type == "human_joints":
                # Update the humanoid base
                self.humanoid_controller.obj_transform_base = base_T
                if not at_goal:
                    if dist_to_final_nav_targ < self._config.dist_thresh:
                        # Look at the object
                        self.humanoid_controller.calculate_turn_pose(
                            mn.Vector3([rel_pos[0], 0.0, rel_pos[1]])
                        )
                    else:
                        # Move towards the target
                        self.humanoid_controller.calculate_walk_pose(
                            mn.Vector3([rel_targ[0], 0.0, rel_targ[1]])
                        )
                else:
                    self.humanoid_controller.calculate_stop_pose()
                    self.skill_done = True

                base_action = self.humanoid_controller.get_pose()
                kwargs[
                    f"{self._action_arg_prefix}human_joints_trans"
                ] = base_action

                HumanoidJointAction.step(self, *args, **kwargs)
                return
            else:
                raise ValueError(
                    "Unrecognized motion type for oracle nav action"
                )
            
@registry.register_task_action
class OracleNavObstacleAction(OracleNavAction_wopddl):
    def __init__(self, *args, task, **kwargs):
        OracleNavAction_wopddl.__init__(self, *args, task=task, **kwargs)
        self.old_human_pos_list = None
        self.rand_human_speed_scale = np.random.uniform(0.8, 1.2)

    def update_rel_targ_obstacle(
        self, rel_targ, new_human_pos, old_human_pos=None
    ):
        if old_human_pos is None or len(old_human_pos) == 0:
            human_velocity_scale = 0.0
        else:
            # take the norm of the distance between old and new human position
            human_velocity_scale = (
                np.linalg.norm(new_human_pos - old_human_pos) / 0.25
            )  # 0.25 is a magic number
            # set a minimum value for the human velocity scale
            human_velocity_scale = max(human_velocity_scale, 0.1)

        std = 8.0
        # scale the amplitude by the human velocity
        amp = 8.0 * human_velocity_scale

        # Get the position of the other agents
        other_agent_rel_pos, other_agent_dist = [], []
        curr_agent_T = np.array(
            self.cur_articulated_agent.base_transformation.translation
        )[[0, 2]]

        other_agent_rel_pos.append(rel_targ[None, :])
        other_agent_dist.append(0.0)  # dummy value
        rel_pos = new_human_pos - curr_agent_T
        dist_pos = np.linalg.norm(rel_pos, ord=2, axis=-1) # np.linalg.norm(rel_pos)
        # normalized relative vector
        rel_pos = rel_pos / dist_pos[:, np.newaxis]
        # dist_pos = np.squeeze(dist_pos)
        other_agent_dist.extend(dist_pos)
        other_agent_rel_pos.append(-rel_pos) # -rel_pos[None, :]

        rel_pos = np.concatenate(other_agent_rel_pos)
        rel_dist = np.array(other_agent_dist)
        weight = amp * np.exp(-(rel_dist**2) / std)
        weight[0] = 1.0
        # TODO: explore softmax?
        weight_norm = weight[:, None] / weight.sum()
        # weighted sum of the old target position and
        # relative position that avoids human
        final_rel_pos = (rel_pos * weight_norm).sum(0)
        return final_rel_pos

    @property
    def action_space(self):
        return spaces.Dict(
            {
                self._action_arg_prefix
                + "oracle_nav_obstacle_action": spaces.Box(
                    shape=(1,),
                    low=np.finfo(np.float32).min,
                    high=np.finfo(np.float32).max,
                    dtype=np.float32,
                )
            }
        )

    def step(self, *args, **kwargs):
        self.skill_done = False
        nav_to_target_coord = kwargs.get(
            self._action_arg_prefix + "oracle_nav_obstacle_action"
        )
        if nav_to_target_coord is None or np.linalg.norm(nav_to_target_coord) == 0:
            return None
        self.humanoid_controller.reset(
                self.cur_articulated_agent.base_transformation
            )

        base_T = self.cur_articulated_agent.base_transformation
        curr_path_points = self._path_to_point(nav_to_target_coord)
        robot_pos = np.array(self.cur_articulated_agent.base_pos)

        if curr_path_points is None:
            raise Exception
        else:
            # Compute distance and angle to target
            if len(curr_path_points) == 1:
                curr_path_points += curr_path_points
            cur_nav_targ = curr_path_points[1]
            forward = np.array([1.0, 0, 0])
            robot_forward = np.array(base_T.transform_vector(forward))

            # Compute relative target.
            rel_targ = cur_nav_targ - robot_pos

            # Compute heading angle (2D calculation)
            robot_forward = robot_forward[[0, 2]]
            rel_targ = rel_targ[[0, 2]]
            rel_pos = (nav_to_target_coord - robot_pos)[[0, 2]]

            # NEW: We will update the rel_targ position to avoid the humanoid
            # rel_targ is the next position that the agent wants to walk to
            # old_rel_targ = rel_targ.copy()
            new_human_pos_list = []
            nearby_human_idx = []
            old_rel_targ = rel_targ
            if self.human_num > 0:
                # This is very specific to SIRo. Careful merging
                for agent_index in range(1, self._sim.num_articulated_agents):
                    new_human_pos = np.array(
                        self._sim.get_agent_data(
                            agent_index
                        ).articulated_agent.base_transformation.translation
                    )
                    new_human_pos_list.append(new_human_pos)
                    if self._agent_index != agent_index: # agent_index is actual index, dog is zero, humanoid are 1-8
                        distance = self._sim.geodesic_distance(robot_pos, new_human_pos)
                        if distance < 2.0 and robot_human_vec_dot_product(robot_pos, new_human_pos,  base_T) > 0.5:
                            nearby_human_idx.append(agent_index-1) # human_idx are humanoid index, 0-7, = agent_index - 1
                if self.old_human_pos_list is not None and len(nearby_human_idx) > 0:
                    new_human_pos_array = np.array(new_human_pos_list)
                    old_human_pos_array = np.array(self.old_human_pos_list)
                    rel_targ = self.update_rel_targ_obstacle(
                        rel_targ, new_human_pos_array[nearby_human_idx][:, [0, 2]], old_human_pos_array[nearby_human_idx][:, [0, 2]]
                    )
                self.old_human_pos_list = new_human_pos_list 
            # NEW: If avoiding the human makes us change dir, we will
            # go backwards at times to avoid rotating
                
            dot_prod_rel_targ = (rel_targ * old_rel_targ).sum()
            did_change_dir = dot_prod_rel_targ < 0

            angle_to_target = get_angle(robot_forward, rel_targ) # next goal
            angle_to_obj = get_angle(robot_forward, rel_pos) # final goal

            dist_to_final_nav_targ = np.linalg.norm(
                (nav_to_target_coord - robot_pos)[[0, 2]]
            )
            at_goal = (
                dist_to_final_nav_targ < self._config.dist_thresh
                and angle_to_obj < self._config.turn_thresh
            ) or dist_to_final_nav_targ < self._config.dist_thresh / 10.0
            if self.motion_type == "base_velocity":
                if not at_goal:
                    if dist_to_final_nav_targ < self._config.dist_thresh:
                        # Look at the object
                        vel = OracleNavAction_wopddl._compute_turn(
                            rel_pos, self._config.turn_velocity, robot_forward
                        )
                    elif angle_to_target < self._config.turn_thresh:
                        # Move towards the target
                        vel = [self._config.forward_velocity, 0]
                    else:
                        # Look at the target waypoint.
                        if did_change_dir:
                            if (np.pi - angle_to_target) < self._config.turn_thresh:
                                # Move towards the target
                                vel = [-self._config.forward_velocity, 0]
                            else:
                                vel = OracleNavAction_wopddl._compute_turn(
                                    -rel_targ,
                                    self._config.turn_velocity,
                                    robot_forward,
                                )
                        else:
                            vel = OracleNavAction_wopddl._compute_turn(
                                rel_targ, self._config.turn_velocity, robot_forward
                            )
                else:
                    vel = [0, 0]
                    self.skill_done = True
                kwargs[f"{self._action_arg_prefix}base_vel"] = np.array(vel)
                return BaseVelAction.step(
                    self, *args, **kwargs
                )
            elif self.motion_type == "human_joints":
                self.humanoid_controller.obj_transform_base = base_T
                if not at_goal:
                    if dist_to_final_nav_targ < self._config.dist_thresh:
                        # Look at the object
                        self.humanoid_controller.calculate_turn_pose(
                            mn.Vector3([rel_pos[0], 0.0, rel_pos[1]])
                        )
                        try:
                            if  'human_velocity_measure' in kwargs['task'].measurements.measures and self._agent_index <= self.human_num:
                                kwargs["task"].measurements.measures["human_velocity_measure"].velo_coff[self._agent_index - 1][0] = 0
                                kwargs["task"].measurements.measures["human_velocity_measure"].velo_coff[self._agent_index - 1][1] = 1
                        except Exception:
                            pass
                    elif angle_to_target < self._config.turn_thresh:
                        # Move towards the target
                        if self._config["lin_speed"] == 0:
                            distance_multiplier = 0.0
                        else:
                            distance_multiplier = 1.0  * self.rand_human_speed_scale
                        self.humanoid_controller.calculate_walk_pose(
                            mn.Vector3([rel_targ[0], 0.0, rel_targ[1]]),
                            distance_multiplier
                            )
                        try:
                            if  'human_velocity_measure' in kwargs['task'].measurements.measures and self._agent_index <= self.human_num:
                                kwargs["task"].measurements.measures["human_velocity_measure"].velo_coff[self._agent_index - 1][0] = self.rand_human_speed_scale 
                                kwargs["task"].measurements.measures["human_velocity_measure"].velo_coff[self._agent_index - 1][1] = 0
                        except Exception:
                            pass

                    else:
                        # Look at the target waypoint.
                        if did_change_dir:
                            if (np.pi - angle_to_target) < self._config.turn_thresh:
                                # Move towards the target
                                if self._config["lin_speed"] == 0:
                                    distance_multiplier = 0.0
                                else:
                                    distance_multiplier = 1.0
                                self.humanoid_controller.calculate_walk_pose(
                                mn.Vector3([-rel_targ[0], 0.0, -rel_targ[1]]),
                                distance_multiplier
                                )
                                try:
                                    if  'human_velocity_measure' in kwargs['task'].measurements.measures and self._agent_index <= self.human_num:
                                        kwargs["task"].measurements.measures["human_velocity_measure"].velo_coff[self._agent_index - 1][0] = - self.rand_human_speed_scale
                                        kwargs["task"].measurements.measures["human_velocity_measure"].velo_coff[self._agent_index - 1][1] = 0
                                except Exception:
                                    pass
                            else:
                                self.humanoid_controller.calculate_turn_pose(
                                    mn.Vector3([-rel_targ[0], 0.0, -rel_targ[1]])
                                    )
                                try:
                                    if  'human_velocity_measure' in kwargs['task'].measurements.measures and self._agent_index <= self.human_num:
                                        kwargs["task"].measurements.measures["human_velocity_measure"].velo_coff[self._agent_index - 1][0] = 0
                                        kwargs["task"].measurements.measures["human_velocity_measure"].velo_coff[self._agent_index - 1][1] = -1
                                except Exception:
                                    pass
                        else:
                            self.humanoid_controller.calculate_turn_pose( # turn
                                mn.Vector3([rel_targ[0], 0.0, rel_targ[1]])
                                )
                            try:
                                if  'human_velocity_measure' in kwargs['task'].measurements.measures and self._agent_index <= self.human_num:
                                    kwargs["task"].measurements.measures["human_velocity_measure"].velo_coff[self._agent_index - 1][0] = 0
                                    kwargs["task"].measurements.measures["human_velocity_measure"].velo_coff[self._agent_index - 1][1] = 1
                            except Exception:
                                pass
                else:
                    self.humanoid_controller.calculate_stop_pose()
                    self.skill_done = True
                    try:
                        if  'human_velocity_measure' in kwargs['task'].measurements.measures and self._agent_index <= self.human_num:
                            kwargs["task"].measurements.measures["human_velocity_measure"].velo_coff[self._agent_index - 1][0] = 0
                            kwargs["task"].measurements.measures["human_velocity_measure"].velo_coff[self._agent_index - 1][1] = 0
                    except Exception:
                        pass
                self._update_controller_to_navmesh()
                base_action = self.humanoid_controller.get_pose()
                kwargs[
                    f"{self._action_arg_prefix}human_joints_trans"
                ] = base_action

                return HumanoidJointAction.step(self, *args, **kwargs)
            else:
                raise ValueError(
                    "Unrecognized motion type for oracle nav obstacle action"
                )
            
@registry.register_task_action
class OracleNavRandCoordAction_Obstacle(OracleNavObstacleAction):  # type: ignore # (OracleNavObstacleAction)
    """
    Oracle Nav RandCoord Action. Selects a random position in the scene and navigates
    there until reaching. When the arg is 1, does replanning.
    """

    def __init__(self, *args, task, **kwargs):
        super().__init__(*args, task=task, **kwargs)
        self._config = kwargs["config"]
        self.human_num = 0
        self.num_goals = 2
        self.current_goal_idx = 0
        self.goals = [np.array([0, 0, 0], dtype=np.float32) for _ in range(self.num_goals)]
        self._largest_indoor_island_idx = get_largest_island_index(
            self._sim.pathfinder, self._sim, allow_outdoor=False # True
        )

    
    @property
    def action_space(self):
        return spaces.Dict(
            {
                self._action_arg_prefix
                + "oracle_nav_randcoord_action_obstacle": spaces.Box(
                    shape=(1,),
                    low=np.finfo(np.float32).min,
                    high=np.finfo(np.float32).max,
                    dtype=np.float32,
                )
            }
        )

    def _add_n_coord_nav_goals(self,n=3):
        max_tries = 10

        for i in range(n):
            temp_coord_nav = self._sim.pathfinder.get_random_navigable_point(
                max_tries,
                island_index=self._largest_indoor_island_idx,
            )
            while len(self.goals) >= 1 and np.linalg.norm(temp_coord_nav - self.goals[-1],ord=2,axis=-1) < 3:
                temp_coord_nav = self._sim.pathfinder.get_random_navigable_point(
                max_tries,
                island_index=self._largest_indoor_island_idx,)
            if np.linalg.norm(temp_coord_nav - self._task.nav_goal_pos,ord=2,axis=-1) < 1 and i == n-1:
                temp_coord_nav = self._sim.pathfinder.get_random_navigable_point(
                max_tries,
                island_index=self._largest_indoor_island_idx,
            )
            self.goals[i] = temp_coord_nav

    def reset(self, *args, **kwargs):
        self.human_num = kwargs['task']._human_num
        super().reset(*args, **kwargs)
        if self._task._episode_id != self._prev_ep_id:
            self._prev_ep_id = self._task._episode_id
        self.skill_done = False
        self.coord_nav = None
        # my added 
        self.current_goal_idx = 0
        self._largest_indoor_island_idx = get_largest_island_index(
            self._sim.pathfinder, self._sim, allow_outdoor=False # True
        )
        if self._agent_index <= self.human_num:
            if self._task._use_episode_start_goal:
                for i in range(self.num_goals):
                    attribute_name = f"human_{self._agent_index - 1}_waypoint_{i+1}_position"
                    if attribute_name in kwargs["episode"].info:
                        self.goals[i] = kwargs["episode"].info[attribute_name]
                    else:
                        self.goals = None
            else:
                self._add_n_coord_nav_goals(self.num_goals)

    def _find_path_given_start_end(self, start, end):
        """Helper function to find the path given starting and end locations"""
        path = habitat_sim.ShortestPath()
        path.requested_start = start
        path.requested_end = end
        found_path = self._sim.pathfinder.find_path(path)
        if not found_path:
            return [start, end]
        return path.points

    def _reach_human(self, robot_pos, human_pos, base_T):
        """Check if the agent reaches the human or not"""
        facing = (
            robot_human_vec_dot_product(robot_pos, human_pos, base_T) > 0.5
        )

        # Use geodesic distance here
        dis = self._sim.geodesic_distance(robot_pos, human_pos)

        return dis <= 2.0 and facing

    def _compute_robot_to_human_min_step(
        self, robot_trans, human_pos, human_pos_list
    ):
        """The function to compute the minimum step to reach the goal"""
        _vel_scale = self._config.lin_speed

        # Copy the robot transformation
        base_T = mn.Matrix4(robot_trans)

        vc = SimpleVelocityControlEnv()

        # Compute the step taken to reach the human
        robot_pos = np.array(base_T.translation)
        robot_pos[1] = human_pos[1]
        step_taken = 0
        while (
            not self._reach_human(robot_pos, human_pos, base_T)
            and step_taken <= 1500
        ):
            path_points = self._find_path_given_start_end(robot_pos, human_pos)
            cur_nav_targ = path_points[1]
            obj_targ_pos = path_points[1]
            forward = np.array([1.0, 0, 0])
            robot_forward = np.array(base_T.transform_vector(forward))

            # Compute relative target.
            rel_targ = cur_nav_targ - robot_pos
            rel_pos = (obj_targ_pos - robot_pos)[[0, 2]]

            # Compute heading angle (2D calculation)
            robot_forward = robot_forward[[0, 2]]
            rel_targ = rel_targ[[0, 2]]
            angle_to_target = get_angle(robot_forward, rel_targ)
            dist_to_final_nav_targ = np.linalg.norm(
                (human_pos - robot_pos)[[0, 2]]
            )

            if dist_to_final_nav_targ < self._config.dist_thresh:
                # Look at the object
                vel = OracleNavAction_wopddl._compute_turn(
                    rel_pos,
                    self._config.turn_velocity * _vel_scale,
                    robot_forward,
                )
            elif angle_to_target < self._config.turn_thresh:
                # Move towards the target
                vel = [self._config.forward_velocity * _vel_scale, 0]
            else:
                # Look at the target waypoint.
                vel = OracleNavAction_wopddl._compute_turn(
                    rel_targ,
                    self._config.turn_velocity * _vel_scale,
                    robot_forward,
                )

            # Update the robot's info
            base_T = vc.act(base_T, vel)
            robot_pos = np.array(base_T.translation)
            step_taken += 1

            robot_pos[1] = human_pos[1]
        return step_taken

    def _get_target_for_coord(self, obj_pos):
        start_pos = obj_pos
        if self.motion_type == "human_joints":
            self.humanoid_controller.reset(
                self.cur_articulated_agent.base_transformation
            )
        return (start_pos, np.array(obj_pos))

    def step(self, *args, **kwargs):
        self.skill_done = False

        if self.coord_nav is None and self.goals is not None:
            if self.current_goal_idx < len(self.goals):
                self.coord_nav = self.goals[self.current_goal_idx]
                self.current_goal_idx += 1

        kwargs[
            self._action_arg_prefix + "oracle_nav_obstacle_action"
        ] = self.coord_nav

        ret_val = super().step(*args, **kwargs)

        if self.skill_done:
            self.coord_nav = None

        # If the robot is nearby, the human starts to walk, otherwise, the human
        # just stops there and waits for robot to find it
        if self._config.human_stop_and_walk_to_robot_distance_threshold != -1:
            assert (
                len(self._sim.agents_mgr) == 2
            ), "Does not support more than two agents when you want human to stop and walk based on the distance to the robot"
            robot_id = int(1 - self._agent_index)
            robot_pos = self._sim.get_agent_data(
                robot_id
            ).articulated_agent.base_pos
            human_pos = self.cur_articulated_agent.base_pos
            dis = self._sim.geodesic_distance(robot_pos, human_pos)
            # The human needs to stop and wait for robot to come if the distance is too large
            if (
                dis
                > self._config.human_stop_and_walk_to_robot_distance_threshold
            ):
                self.humanoid_controller.set_framerate_for_linspeed(
                    0.0, 0.0, self._sim.ctrl_freq
                )
            # The human needs to walk otherwise
            else:
                speed = np.random.uniform(
                    self._config.lin_speed / 5.0, self._config.lin_speed
                )
                lin_speed = speed
                ang_speed = speed
                self.humanoid_controller.set_framerate_for_linspeed(
                    lin_speed, ang_speed, self._sim.ctrl_freq
                )

        try:
            if  'human_future_trajectory' in kwargs['task'].measurements.measures:
                kwargs["task"].measurements.measures["human_future_trajectory"].target_dict[self._agent_index - 1] = self.goals[self.current_goal_idx-1:] # .copy()
        except Exception:
            pass
        return ret_val

@dataclass
class DiscreteStopActionConfig(ActionConfig):
    type: str = "DiscreteStopAction"
    lin_speed: float = 0.0
    ang_speed: float = 0.0 
    allow_back: bool = False
    value: int = 1
    allow_dyn_slide: bool = False # True
    leg_animation_checkpoint: str = (
        "data/robots/spot_data/spot_walking_trajectory.csv"
    )
    play_i_perframe: int = 5
    use_range: Optional[List[int]] = field(default_factory=lambda: [107, 863])

@dataclass
class DiscreteMoveForwardActionConfig(ActionConfig):
    type: str = "DiscreteMoveForwardAction"
    lin_speed: float = 10.0
    ang_speed: float = 0.0 
    allow_back: bool = False
    value: int = 1
    allow_dyn_slide: bool = False # True
    leg_animation_checkpoint: str = (
        "data/robots/spot_data/spot_walking_trajectory.csv"
    )
    play_i_perframe: int = 5
    use_range: Optional[List[int]] = field(default_factory=lambda: [107, 863])

@dataclass
class DiscreteTurnLeftActionConfig(ActionConfig):
    type: str = "DiscreteTurnLeftAction"
    lin_speed: float = 0.0
    ang_speed: float = 10.0 
    allow_back: bool = False
    value: int = 1
    allow_dyn_slide: bool = False # True
    leg_animation_checkpoint: str = (
        "data/robots/spot_data/spot_walking_trajectory.csv"
    )
    play_i_perframe: int = 5
    use_range: Optional[List[int]] = field(default_factory=lambda: [107, 863])

@dataclass
class DiscreteTurnRightActionConfig(ActionConfig):
    type: str = "DiscreteTurnRightAction"
    lin_speed: float = 0.0
    ang_speed: float = -10.0 
    allow_back: bool = False
    value: int = 1
    allow_dyn_slide: bool = False # True
    leg_animation_checkpoint: str = (
        "data/robots/spot_data/spot_walking_trajectory.csv"
    )
    play_i_perframe: int = 5
    use_range: Optional[List[int]] = field(default_factory=lambda: [107, 863])

@dataclass
class OracleNavActionWOPDDLConfig(ActionConfig):
    """
    Rearrangement Only, Oracle navigation action.
    This action takes as input a discrete ID which refers to an object in the
    PDDL domain. The oracle navigation controller then computes the actions to
    navigate to that desired object.
    """

    type: str = "OracleNavAction_wopddl"
    # Whether the motion is in the form of base_velocity or human_joints
    motion_control: str = "base_velocity"
    num_joints: int = 17
    turn_velocity: float = 1.0
    forward_velocity: float = 1.0
    turn_thresh: float = 0.1
    dist_thresh: float = 0.2
    lin_speed: float = 10.0
    ang_speed: float = 10.0
    allow_dyn_slide: bool = True
    allow_back: bool = True
    spawn_max_dist_to_obj: float = 2.0
    num_spawn_attempts: int = 200
    # For social nav training only. It controls the distance threshold
    # between the robot and the human and decide if the human wants to walk or not
    human_stop_and_walk_to_robot_distance_threshold: float = -1.0

cs = ConfigStore.instance()

cs.store( ##
    package="habitat.task.actions.discrete_stop",
    group="habitat/task/actions",
    name="discrete_stop",
    node=DiscreteStopActionConfig,
)

cs.store( ##
    package="habitat.task.actions.discrete_move_forward",
    group="habitat/task/actions",
    name="discrete_move_forward",
    node=DiscreteMoveForwardActionConfig,
)

cs.store( ##
    package="habitat.task.actions.discrete_turn_left",
    group="habitat/task/actions",
    name="discrete_turn_left",
    node=DiscreteTurnLeftActionConfig,
)
cs.store( ##
    package="habitat.task.actions.discrete_turn_right",
    group="habitat/task/actions",
    name="discrete_turn_right",
    node=DiscreteTurnRightActionConfig,
)
cs.store(
    package="habitat.task.actions.oracle_nav_action",
    group="habitat/task/actions",
    name="oracle_nav_action",
    node=OracleNavActionWOPDDLConfig,
)
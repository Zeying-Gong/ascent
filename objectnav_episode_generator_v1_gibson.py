import gzip
import json
import os
import copy
from typing import Any, Dict, List, Optional
import pickle
from habitat.core.registry import registry
from habitat.core.simulator import AgentState, ShortestPathPoint
from habitat.core.utils import DatasetFloatJSONEncoder
from habitat.datasets.pointnav.pointnav_dataset import (
    CONTENT_SCENES_PATH_FIELD,
    DEFAULT_SCENE_PATH_PREFIX,
    PointNavDatasetV1,
)
from habitat.core.dataset import ALL_SCENES_MASK, Dataset
# from habitat.tasks.nav.nav import (
#     NavigationEpisode,
#     NavigationGoal,
#     ShortestPathPoint,
# )
from habitat.tasks.nav.object_nav_task import ObjectGoalNavEpisode
from habitat.utils.geometry_utils import (
    quaternion_from_coeff,
    quaternion_rotate_vector,
)
from habitat.datasets.rearrange.navmesh_utils import get_largest_island_index
from habitat_sim.nav import NavMeshSettings
from omegaconf import DictConfig, OmegaConf
import math
import habitat_sim
import magnum as mn
import warnings
from habitat.tasks.rearrange.rearrange_sim_v2 import RearrangeSim_v2
warnings.filterwarnings('ignore')
from habitat_sim.utils.settings import make_cfg
from matplotlib import pyplot as plt
from habitat_sim.utils import viz_utils as vut
import numpy as np
from habitat.articulated_agents.robots import FetchRobot
from habitat.config.default import get_agent_config
from habitat.config.default_structured_configs import (
    ThirdRGBSensorConfig, HeadRGBSensorConfig, HeadPanopticSensorConfig,
    SimulatorConfig, HabitatSimV0Config, AgentConfig)
from habitat.config.default import get_agent_config
from habitat_sim.physics import JointMotorSettings, MotionType
from habitat.utils.visualizations import maps
from habitat_sim.utils import common as utils
from habitat.tasks.rearrange.utils import (    add_perf_timing_func,
    get_rigid_aabb,
    make_render_only,
    rearrange_collision,
    rearrange_logger,
)
import pandas as pd
from habitat.datasets.pointnav.pointnav_generator import is_compatible_episode
CONTENT_SCENES_PATH_FIELD = "content_scenes_path"
DEFAULT_SCENE_PATH_PREFIX = "data/scene_datasets/"

split="val"
current_working_directory = os.getcwd()
folder_path = os.path.join(current_working_directory, "data/datasets/objectnav/gibson/v1.1", split, "content")
output_dir = os.path.join(current_working_directory, "data/datasets/objectnav/gibson/multi_agent_filtered_v1", split, "content")
output_map = os.path.join(current_working_directory, "data/datasets/objectnav/gibson/multi_agent_filtered_v1", split, "map")
scenes_dir = 'data/scene_datasets/'
# scene_dataset_config = "data/scene_datasets/gibson/gibson_basis.scene_dataset_config.json"

def quaternion_to_rad_angle(source_rotation):
    rad_angle = 2 * np.arctan2(np.sqrt(source_rotation[1]**2 + source_rotation[2]**2 + source_rotation[3]**2), source_rotation[0])
    return rad_angle

def init_rearrange_sim(cfg):
    sim = RearrangeSim_v2(cfg)
    sim.agents_mgr.on_new_scene()
    return sim

def save_to_excel(scene_data, temp_file):
    df = pd.DataFrame(scene_data)
    df.to_excel(temp_file, index=False)

# def display_map(output_map, topdown_map, scene_count, scene_id, episode_id, split, key_points=None):
#     os.makedirs(output_map, exist_ok=True)
#     file_name = f"gibson_scene{str(scene_count)}_{scene_id}_{episode_id}_map.png"
#     file_path = os.path.join(output_map, file_name)

#     plt.figure(figsize=(12, 8))
#     ax = plt.subplot(1, 1, 1)
#     ax.axis("off")
#     plt.imshow(topdown_map)

#     if key_points is not None:
#         colors = ['blue', 'orange', 'purple']
#         for idx, point in enumerate(key_points):
#             color = colors[idx] if idx < len(colors) else 'purple'
#             plt.plot(point[0], point[1], marker="o", markersize=10, alpha=0.8, color=color)

#     plt.savefig(file_path, bbox_inches='tight')
#     plt.close()

def display_map(output_map, topdown_map, scene_count, scene_id, episode_id, split, 
                robot_start=None, goal_positions=None, human_start_positions=None):
    os.makedirs(output_map, exist_ok=True)
    
    file_name = f"gibson_scene{str(scene_count)}_{scene_id}_{episode_id}_map.png"
    file_path = os.path.join(output_map, file_name)

    plt.figure(figsize=(12, 8))
    ax = plt.subplot(1, 1, 1)
    ax.axis("off")
    plt.imshow(topdown_map)

    # Plot robot start position (in blue)
    if robot_start is not None:
        for point in robot_start:
            plt.plot(point[0], point[1], marker="o", markersize=10, alpha=0.8, color="blue")

    # Plot goal positions (in orange)
    if goal_positions is not None:
        for point in goal_positions:
            plt.plot(point[0], point[1], marker="o", markersize=10, alpha=0.8, color="orange")

    # Plot human start positions (in purple)
    if human_start_positions is not None:
        for point in human_start_positions:
            plt.plot(point[0], point[1], marker="o", markersize=10, alpha=0.8, color="purple")

    # Save the map
    plt.savefig(file_path, bbox_inches='tight')
    plt.close()

def get_coll_forces_for_agent0(sim, articulated_agent_id):
    # grasp_mgr = self._sim.get_agent_data(articulated_agent_id).grasp_mgr
    articulated_agent = sim.get_agent_data(
        articulated_agent_id
    ).articulated_agent
    # snapped_obj = grasp_mgr.snap_idx
    articulated_agent_id = articulated_agent.sim_obj.object_id
    contact_points = sim.get_physics_contact_points()

    def get_max_force(contact_points, check_id):
        match_contacts = [
            x
            for x in contact_points
            if (check_id in [x.object_id_a, x.object_id_b])
            and (x.object_id_a != x.object_id_b)
        ]

        max_force = 0
        if len(match_contacts) > 0:
            max_force = max([abs(x.normal_force) for x in match_contacts])

        return max_force
    
    max_articulated_agent_force = get_max_force(
        contact_points, articulated_agent_id
    )
    return max_articulated_agent_force
def convert_points_to_topdown(pathfinder, points, meters_per_pixel):
    points_topdown = []
    bounds = pathfinder.get_bounds()
    for point in points:
        # convert 3D x,z to topdown x,y
        px = (point[0] - bounds[0][0]) / meters_per_pixel
        py = (point[2] - bounds[0][2]) / meters_per_pixel
        points_topdown.append(np.array([px, py]))
    return points_topdown

def _is_on_same_floor(
    sim, height, ref_floor_height=None, ceiling_height=2.0
):
    if ref_floor_height is None:
        ref_floor_height = sim.get_agent_data(0).articulated_agent.base_pos[1] # sim.get_agent_data(0).state.position[1]
        ref_floor_height_rounded = round(ref_floor_height, 1)
        height_rounded = round(height, 1)
    return ref_floor_height_rounded <= height_rounded < ref_floor_height_rounded + ceiling_height
    # return ref_floor_height <= height < ref_floor_height + ceiling_height
    
def _filter_agent_position(prev_pose_agents, start_pos):
    start_pos = np.array(start_pos)
    start_pos_2d = start_pos[[0, 2]]
    prev_pos_2d = [
        [prev_pose_agent[0], prev_pose_agent[2]]
        for prev_pose_agent in prev_pose_agents
    ]
    distances = np.array(
        [
            np.linalg.norm(start_pos_2d - prev_pos_2d_i)
            for prev_pos_2d_i in prev_pos_2d
        ]
    )
    return np.any(distances < 2.0)
    
def add_human(sim, deserialized, episode_count, agent_0_pos, start_island_index, human_num):
    prev_pose_agents = [agent_0_pos]
    all_human_start = []
    attempt_i_num = 100
    # Initialize an empty list to store valid instances
    valid_instances = []

    # Get all the categories
    categories = list(deserialized['goals_by_category'].keys())

    # Traverse each category and each instance
    for category in categories:
        for instance in deserialized['goals_by_category'][category]:
            # Get the position of the instance
            position = instance['position']
            
            # Check if the instance's island_id matches the start_island_index
            island_id = sim.pathfinder.get_island(position)
            if island_id == start_island_index:
                # If it matches, add the instance to the valid_instances list
                valid_instances.append(instance)

    if len(valid_instances) < 2:
        print("Not enough valid instances found that match the start_island_index.")
        return -1, all_human_start

    for human_idx in range(human_num):
        for attempt_i in range(attempt_i_num):
            human_1_pos = sim.pathfinder.get_random_navigable_point(max_tries=10, island_index=start_island_index) # 
            if not np.isnan(human_1_pos[0]):
                human_1_pos = human_1_pos.tolist()
            else:
                continue
            if _filter_agent_position(prev_pose_agents,human_1_pos):
                continue
            elif not _is_on_same_floor(sim, human_1_pos[1]):
                continue
            else:
                human_1_rot = np.random.uniform(0, 2 * np.pi)
                break
        if attempt_i == attempt_i_num-1:
            return -1, all_human_start
        else:
            prev_pose_agents.append(human_1_pos)

        for attempt_i in range(attempt_i_num):        
            # Try to place the human 2nd and 3rd point with semantic object goals.
            selected_instances = np.random.choice(valid_instances, 2, replace=False)
            
            # Get the position of each instance
            objectgoal_position1 = selected_instances[0]['position']
            objectgoal_position2 = selected_instances[1]['position']

            human_2_pos = sim.pathfinder.get_random_navigable_point_near(
                    circle_center=objectgoal_position1, radius=2.0, island_index=start_island_index
                )
            # human_2_pos, suc = safe_snap_point(sim, human_2_pos)
            if not np.isnan(human_2_pos[0]):
                human_2_pos = human_2_pos.tolist()
                # human_2_rot = np.random.uniform(0, 2 * np.pi)
            else:
                continue
            human_3_pos = sim.pathfinder.get_random_navigable_point_near(
                    circle_center=objectgoal_position2, radius=2.0, island_index=start_island_index
                )
            # human_3_pos, suc = safe_snap_point(sim, human_3_pos)
            if not np.isnan(human_3_pos[0]):
                human_3_pos = human_3_pos.tolist()
                # human_3_rot = np.random.uniform(0, 2 * np.pi)
                break # get 2nd and 3rd points 
            else:
                continue

        if attempt_i == attempt_i_num-1:
            return -1, all_human_start

        if sim.pathfinder.is_navigable(human_2_pos) and sim.pathfinder.is_navigable(human_3_pos):
            deserialized['episodes'][episode_count]['info'][f'human_{human_idx}_waypoint_0_position'] = human_1_pos
            deserialized['episodes'][episode_count]['info'][f'human_{human_idx}_waypoint_0_rotation'] = human_1_rot
            deserialized['episodes'][episode_count]['info'][f'human_{human_idx}_waypoint_1_position'] = human_2_pos
            # deserialized['episodes'][episode_count]['info'][f'human_{human_idx}_waypoint_1_rotation'] = human_2_rot
            deserialized['episodes'][episode_count]['info'][f'human_{human_idx}_waypoint_2_position'] = human_3_pos
            # deserialized['episodes'][episode_count]['info'][f'human_{human_idx}_waypoint_2_rotation'] = human_3_rot
            all_human_start.append(human_1_pos)
        else:
            return -1, all_human_start
    deserialized['episodes'][episode_count]['info']['human_num'] = human_num
    return human_num, all_human_start

def safe_snap_point(sim, pos):
    """
    Returns the 3D coordinates corresponding to a point belonging
    to the biggest navmesh island in the scene and closest to pos.
    When that point returns NaN, computes a navigable point at increasing
    distances to it.
    """
    new_pos = sim.pathfinder.snap_point(
        pos # , island_idx
    )

    max_iter = 100
    offset_distance = 1.5
    distance_per_iter = 0.5
    num_sample_points = 1000

    regen_i = 0
    while np.isnan(new_pos[0]) and regen_i < max_iter:
        # Increase the search radius
        new_pos = sim.pathfinder.get_random_navigable_point_near(
            pos,
            offset_distance + regen_i * distance_per_iter,
            num_sample_points,
            # island_index=island_idx,
        )
        regen_i += 1
    return new_pos, not np.isnan(new_pos[0])

# main function
os.makedirs(output_dir, exist_ok=True)
all_scene_episode_data = [] 

# Loop through each scene
for scene_count, json_str in enumerate(os.listdir(folder_path)):
    total_scene_num = len(os.listdir(folder_path))
    print(f"Processing {scene_count+1} - {total_scene_num} scene")
    episode_json_str = os.path.join(folder_path, json_str)
    with gzip.open(episode_json_str, "rt") as f:
        deserialized = json.loads(f.read())

    scene_id = json_str.split('.')[0]
    final_output_dir = os.path.join(output_dir, f"{scene_id}.json.gz")

    # Initialize simulation only once per scene
    main_agent_config = AgentConfig()
    urdf_path = "data/humanoids/humanoid_data/female_2/female_2.urdf"
    main_agent_config.articulated_agent_urdf = urdf_path
    main_agent_config.articulated_agent_type = "KinematicHumanoid"
    main_agent_config.motion_data_path = "data/humanoids/humanoid_data/female_2/female_2_motion_data_smplx.pkl"
    main_agent_config.height = 1.5
    main_agent_config.radius = 0.3
    agent_dict = {"main_agent": main_agent_config}     
    sim_cfg = SimulatorConfig(type="RearrangeSim-v2")
    sim_cfg.habitat_sim_v0.enable_hbao = True
    sim_cfg.habitat_sim_v0.enable_physics = True
    try:
        scene_id_dir = deserialized['episodes'][0]['scene_id']
    except IndexError as e:
        print(f"Scene {scene_id} seems wrong")
        continue
    sim_cfg.scene = os.path.join(scenes_dir, scene_id_dir)
    # sim_cfg.scene_dataset = scene_dataset_config
    sim_cfg.agents = agent_dict
    cfg = OmegaConf.create(sim_cfg)
    cfg.agents_order = list(cfg.agents.keys())
    sim = init_rearrange_sim(cfg)
    if not sim.pathfinder.is_loaded:
        print("Pathfinder not initialized, aborting.")
        continue

    # Set up the navmesh
    navmesh_settings = NavMeshSettings()
    navmesh_settings.set_defaults()
    navmesh_settings.agent_radius = cfg.agents.main_agent.radius
    navmesh_settings.agent_height = cfg.agents.main_agent.height
    navmesh_settings.include_static_objects = True
    navmesh_settings.agent_max_climb = cfg.agents.main_agent.max_climb
    navmesh_settings.agent_max_slope = cfg.agents.main_agent.max_slope
    sim.recompute_navmesh(sim.pathfinder, navmesh_settings)

    episode_total_num = len(deserialized["episodes"])
    deserialized_filtered = copy.deepcopy(deserialized)
    deserialized_filtered['episodes'] = []
    total_navigable_area = sim.pathfinder.navigable_area
    actual_largest_island_index = get_largest_island_index(sim.pathfinder, sim, allow_outdoor=False)
    actual_largest_island_area = sim.pathfinder.island_area(actual_largest_island_index)
    filter_id_list = []
    scene_episode_data = []

    # Outer loop to try reducing add_human_num if success_episode < 10
    success_episode = 0  # Initialize success_episode count for the scene

    for episode_count, episode in enumerate(deserialized["episodes"]):
        # deserialized_filtered['episodes'] = []  # Clear episodes for this attempt
        amount_flag = 0
        add_human_amount = 6  # Start with the maximum number of humans

        episode = ObjectGoalNavEpisode(**episode)
        episode.episode_id = str(episode_count)

        is_same_floor = None
        ori_is_available = False
        is_compatible = False

        agent_pos = episode.start_position
        start_island_index = sim.pathfinder.get_island(agent_pos)
        start_island_area = sim.pathfinder.island_area(start_island_index)
        is_start_on_largest_island = (start_island_index == actual_largest_island_index)
        # Iterate through each episode
        # for episode_count, episode in enumerate(deserialized["episodes"]):
        while add_human_amount >= 1:

            # Initial add_human_num based on the start_island_area
            if amount_flag == 0:
                amount_flag = 1
                if start_island_area < 20:
                    add_human_amount = 1
                elif start_island_area < 40:
                    add_human_amount = 2
                elif start_island_area < 60:
                    add_human_amount = 3
                elif start_island_area < 80:
                    add_human_amount = 4
                elif start_island_area < 100:
                    add_human_amount = 5
                else:
                    add_human_amount = 6

            add_human_num, all_human_start = add_human(sim, deserialized, episode_count, agent_pos, start_island_index, add_human_amount)
            if add_human_num != -1:
                if split != 'train':
                    top_down_map = maps.get_topdown_map(sim.pathfinder, height=agent_pos[1], meters_per_pixel=0.025)
                    recolor_map = np.array([[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8)
                    top_down_map = recolor_map[top_down_map]

                    goal_objects = deserialized["goals_by_category"][episode.goals_key]
                    goal_objects_pos = [goal['position'] for goal in goal_objects]

                    # Convert different types of points separately
                    start_position = convert_points_to_topdown(sim.pathfinder, [deserialized["episodes"][episode_count]["start_position"]], 0.025)
                    goal_positions = convert_points_to_topdown(sim.pathfinder, goal_objects_pos, 0.025)
                    human_start_positions = convert_points_to_topdown(sim.pathfinder, all_human_start, 0.025)

                    display_map(output_map, top_down_map, scene_count, scene_id, success_episode, split, 
                                robot_start=start_position, 
                                goal_positions=goal_positions, 
                                human_start_positions=human_start_positions)
                success_episode += 1
                deserialized_filtered['episodes'].append(deserialized["episodes"][episode_count])
                print(f"Successful finish episodes {success_episode} of scene {scene_id}, there are {add_human_amount} humans in this episode, {success_episode}/{episode_total_num}.")
                break
            else:
                add_human_amount -= 1
                print(f"Retrying scene {scene_id} with {add_human_amount} humans due to insufficient success episodes.")
                # Store episode data
        scene_episode_data.append({
            "dataset": "gibson_multi_agent_objectnav",  # seems wrong
            "split": split,
            "scene": scene_id,
            "episode_id": str(episode_count),
            "is_same_floor": is_same_floor,
            "total_navigable_area": total_navigable_area,
            "actual_largest_island_area": actual_largest_island_area,
            "start_island_area": start_island_area,
            "is_start_on_largest_island": is_start_on_largest_island,
            "add_human_num": add_human_num,
        })

        
    sim.close(destroy=True)
    
    all_scene_episode_data.extend(scene_episode_data) 
    save_to_excel(all_scene_episode_data, os.path.join(output_map, "scene_episode_data.xlsx"))
    if len(deserialized_filtered['episodes']) > 0:
        with gzip.open(final_output_dir, "wt") as f:
            json.dump(deserialized_filtered, f)
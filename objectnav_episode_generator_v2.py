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

from habitat.tasks.rearrange.articulated_agent_manager_v2 import ( # use a new manager for adapt navigation version Habitat
    ArticulatedAgentData,
)

CONTENT_SCENES_PATH_FIELD = "content_scenes_path"
DEFAULT_SCENE_PATH_PREFIX = "data/scene_datasets/"

split="val"
current_working_directory = os.getcwd()
folder_path = os.path.join(current_working_directory, "data/datasets/objectnav/hm3d/v1", split, "content")
output_dir = os.path.join(current_working_directory, "data/datasets/objectnav/hm3d/multi_agent_filtered_v2", split, "content")
output_map = os.path.join(current_working_directory, "data/datasets/objectnav/hm3d/multi_agent_filtered_v2", split, "map")
scenes_dir = 'data/scene_datasets/'
scene_dataset_config = "data/scene_datasets/hm3d/hm3d_basis.scene_dataset_config.json"

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

def display_map(output_map, topdown_map, scene_count, scene_id, episode_id, split, 
                robot_start=None, goal_positions=None, human_start_positions=None):
    os.makedirs(output_map, exist_ok=True)
    
    file_name = f"hm3d_scene{str(scene_count)}_{scene_id}_{episode_id}_map.png"
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
        if isinstance(sim.agents_mgr._all_agent_data[0], ArticulatedAgentData):
            ref_floor_height = sim.get_agent_data(0).articulated_agent.base_pos[1] # 
        else:
            ref_floor_height = sim.get_agent_data(0).state.position[1]
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
    return np.any(distances < 3.0) # 3.0

def extract_category_name(category):
    """
    Extract the simple category name (e.g., 'tv_monitor', 'sofa') from a string like 'cvZr5TUy5C5.basis.glb_tv_monitor'.
    """
    if 'glb_' in category:
        category = category.split('glb_')[-1]  # Extract everything after 'glb_'
    elif '.' in category:
        category = category.split('.')[-1]  # Remove prefix before the first dot
    return category

def initialize_scene_islands_by_category(sim, deserialized):
    """
    Initialize and create a dictionary mapping categories to their islands and objects.
    """
    islands_by_category = {}

    for category, instances in deserialized['goals_by_category'].items():
        simple_category = extract_category_name(category)
        if simple_category not in islands_by_category:
            islands_by_category[simple_category] = {}
        for instance in instances:
            island_id = sim.pathfinder.get_island(instance['position'])
            if island_id not in islands_by_category[simple_category]:
                islands_by_category[simple_category][island_id] = []
            islands_by_category[simple_category][island_id].append(instance)

    # For each category, sort the islands by the number of objects (descending)
    for category in islands_by_category:
        islands_by_category[category] = sorted(
            islands_by_category[category].items(), key=lambda x: len(x[1]), reverse=True
        )

    return islands_by_category

def get_filtered_islands_for_humans(islands_by_category, goal_objects_category):
    """
    Filter out the robot's target category and merge remaining categories.
    Optimized to reduce unnecessary loops and operations.
    """
    combined_islands = {}

    for category, islands in islands_by_category.items():
        if category == goal_objects_category:
            continue  # Skip the robot's target category
        for island_id, instances in islands:
            # Add instances to the combined dictionary
            if island_id not in combined_islands:
                combined_islands[island_id] = []
            combined_islands[island_id].extend(instances)

    # Filter islands with at least two valid objects
    filtered_islands = {
        island_id: instances
        for island_id, instances in combined_islands.items() if len(instances) >= 2
    }

    # Sort filtered islands by the number of objects (descending)
    return sorted(filtered_islands.items(), key=lambda x: len(x[1]), reverse=True)

def add_human(sim, deserialized, episode_count, agent_0_pos, sorted_islands, human_num, max_retries=100):
    prev_pose_agents = [agent_0_pos]
    all_human_starts = []
    # Step 2: Generate paths for each human
    for human_idx in range(human_num):
        island_found = False
        for island_id, goal_instances in sorted_islands:
            for attempt in range(max_retries):
                # Generate the starting position
                start_pos = sim.pathfinder.get_random_navigable_point(max_tries=10, island_index=island_id)
                if np.isnan(start_pos[0]):
                    continue  # Failed to generate valid position
                
                start_pos = start_pos.tolist()
                if _filter_agent_position(prev_pose_agents,start_pos):
                    continue  # Too close to existing positions
                start_rot = np.random.uniform(0, 2 * np.pi)
                # Select two goal objects with sufficient distance
                goal_pair = None
                for _ in range(max_retries):
                    selected_goals = np.random.choice(goal_instances, 2, replace=False)
                    goal1_pos, goal2_pos = selected_goals[0]['position'], selected_goals[1]['position']
                    # if  np.linalg.norm(goal1_pos - goal2_pos) >= 3.0:
                    goal_pair = (goal1_pos, goal2_pos)
                    break
                
                if not goal_pair:
                    continue  # Failed to find suitable goal pair
                
                # Generate goal points near the objects
                goal1_pos_near = sim.pathfinder.get_random_navigable_point_near(
                    circle_center=goal_pair[0], radius=2.0, island_index=island_id
                )
                goal2_pos_near = sim.pathfinder.get_random_navigable_point_near(
                    circle_center=goal_pair[1], radius=2.0, island_index=island_id
                )

                if np.isnan(goal1_pos_near) or np.isnan(goal2_pos_near):
                    print("nan of either goal of human")
                    continue  # Failed to generate valid goal positions

                path_1 = habitat_sim.ShortestPath()
                path_1.requested_start = start_pos
                path_1.requested_end = goal1_pos_near
                found_path_1 = sim.pathfinder.find_path(path_1)
                
                path_2 = habitat_sim.ShortestPath()
                path_2.requested_start = goal1_pos_near
                path_2.requested_end = goal2_pos_near
                found_path_2 = sim.pathfinder.find_path(path_2)
            
                if not found_path_1 or not found_path_2:
                    continue

                # All points successfully generated
                island_found = True
                goal1_pos_near, goal2_pos_near = goal1_pos_near.tolist(), goal2_pos_near.tolist()
                prev_pose_agents.append(start_pos)

                deserialized['episodes'][episode_count]['info'][f'human_{human_idx}_waypoint_0_position'] = start_pos
                deserialized['episodes'][episode_count]['info'][f'human_{human_idx}_waypoint_0_rotation'] = start_rot
                deserialized['episodes'][episode_count]['info'][f'human_{human_idx}_waypoint_1_position'] = goal1_pos_near
                deserialized['episodes'][episode_count]['info'][f'human_{human_idx}_waypoint_2_position'] = goal2_pos_near

                all_human_starts.append(start_pos)
                break  # Successfully generated for this human
            
            if island_found:
                break
        else:
            print(f"Failed to generate paths for human {human_idx}. Moving to next island.")
            return -1, all_human_starts
    deserialized['episodes'][episode_count]['info']['human_num'] = human_num
    return human_num, all_human_starts

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
    sim_cfg.scene_dataset = scene_dataset_config
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

    islands_by_category  = initialize_scene_islands_by_category(sim, deserialized)
    total_navigable_area = sim.pathfinder.navigable_area
    # actual_largest_island_index = get_largest_island_index(sim.pathfinder, sim, allow_outdoor=False)
    # actual_largest_island_area = sim.pathfinder.island_area(actual_largest_island_index)
    scene_episode_data = []

    # Outer loop to try reducing add_human_num if success_episode < 10
    success_episode = 0  # Initialize success_episode count for the scene

    for episode_count, episode in enumerate(deserialized["episodes"]):
        # deserialized_filtered['episodes'] = []  # Clear episodes for this attempt
        add_human_amount = 6  # Start with the maximum number of humans

        episode = ObjectGoalNavEpisode(**episode)
        episode.episode_id = str(episode_count)

        is_same_floor = None
        ori_is_available = False
        is_compatible = False

        agent_pos = episode.start_position
        goal_objects_category = episode.object_category
        goal_objects = deserialized["goals_by_category"][episode.goals_key]
        goal_objects_pos = [goal['position'] for goal in goal_objects]
        filtered_islands = get_filtered_islands_for_humans(islands_by_category, goal_objects_category)
        while add_human_amount >= 1:

            # Initial add_human_num based on the start_island_area
            if total_navigable_area < 20:
                add_human_amount = 1
            elif total_navigable_area < 40:
                add_human_amount = 2
            elif total_navigable_area < 60:
                add_human_amount = 3
            elif total_navigable_area < 80:
                add_human_amount = 4
            elif total_navigable_area < 100:
                add_human_amount = 5
            else:
                add_human_amount = 6

            add_human_num, all_human_start = add_human(sim, deserialized, episode_count, agent_pos, filtered_islands, add_human_amount)
            if add_human_num != -1:
                if split != 'train':
                    top_down_map = maps.get_topdown_map(sim.pathfinder, height=agent_pos[1], meters_per_pixel=0.025)
                    recolor_map = np.array([[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8)
                    top_down_map = recolor_map[top_down_map]

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
            "dataset": "hm3d_multi_agent_objectnav",  # seems wrong
            "split": split,
            "scene": scene_id,
            "episode_id": str(episode_count),
            "is_same_floor": is_same_floor,
            "total_navigable_area": total_navigable_area,
            # "actual_largest_island_area": actual_largest_island_area,
            # "start_island_area": start_island_area,
            # "is_start_on_largest_island": is_start_on_largest_island,
            "add_human_num": add_human_num,
        })

        
    sim.close(destroy=True)
    
    all_scene_episode_data.extend(scene_episode_data) 
    save_to_excel(all_scene_episode_data, os.path.join(output_map, "scene_episode_data.xlsx"))
    if len(deserialized_filtered['episodes']) > 0:
        with gzip.open(final_output_dir, "wt") as f:
            json.dump(deserialized_filtered, f)
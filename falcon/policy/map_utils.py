import numpy as np
import os
import cv2
from vlfm.mapping.obstacle_map import ObstacleMap, filter_points_by_height
from vlfm.mapping.value_map import ValueMap
from habitat_baselines.common.tensor_dict import TensorDict
from vlfm.mapping.base_map import BaseMap
from vlfm.mapping.object_point_cloud_map import ObjectPointCloudMap, too_offset, get_random_subarray, open3d_dbscan_filtering 
from PIL import Image
from vlfm.utils.geometry_utils import get_fov, within_fov_cone
from torch.nn import functional as F
from torch.autograd import Variable as V
from vlfm.utils.img_utils import fill_small_holes
from vlfm.utils.geometry_utils import extract_yaw, get_point_cloud, transform_points
from frontier_exploration.utils.fog_of_war import reveal_fog_of_war
from typing import Dict
from collections import deque
from falcon.policy.data_utils import direct_mapping, reference_rooms, STAIR_CLASS_ID

def filter_points_by_height_below_ground_0(points: np.ndarray) -> np.ndarray:
    data = points[(points[:, 2] < 0)] # 0.2 是机器人的max_climb
    return data

def clear_connected_region(map_array, start_y, start_x):
    rows, cols = map_array.shape
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 上、下、左、右四个方向

    # 初始化队列进行 BFS
    queue = deque([(start_y, start_x)])
    map_array[start_y, start_x] = False  # 将起始点标记为 False

    while queue:
        y, x = queue.popleft()
        
        # 遍历四个方向
        for dy, dx in directions:
            ny, nx = y + dy, x + dx
            if 0 <= ny < rows and 0 <= nx < cols and map_array[ny, nx]:  # 在地图范围内且为 True
                map_array[ny, nx] = False  # 设置为 False
                queue.append((ny, nx))  # 将该点加入队列继续搜索

class ObstacleMapUpdater(ObstacleMap):

    def __init__(
        self,
        min_height: float,
        max_height: float,
        agent_radius: float,
        area_thresh: float = 3.0,  # square meters
        hole_area_thresh: int = 100000,  # square pixels
        size: int = 1000,
        pixels_per_meter: int = 20,
    ):
        super().__init__(min_height,max_height,agent_radius,area_thresh,hole_area_thresh,size,pixels_per_meter)

        self._map_dtype = np.dtype(bool)
        self._frontiers_px = np.array([])
        self.frontiers = np.array([])
        self.radius_padding_color = (100, 100, 100)

        self._map_size = size
        self.explored_area = np.zeros((size, size), dtype=bool)
        self._map = np.zeros((size, size), dtype=bool)
        self._navigable_map = np.zeros((size, size), dtype=bool)

        self._up_stair_map = np.zeros((size, size), dtype=bool)  # for upstairs
        self._down_stair_map = np.zeros((size, size), dtype=bool)  # for downstairs
        self._disabled_stair_map = np.zeros((size, size), dtype=bool)  # for disabled stairs
        self._min_height = min_height
        self._max_height = max_height
        self._agent_radius = agent_radius
        self._area_thresh_in_pixels = area_thresh * (self.pixels_per_meter**2)
        self._hole_area_thresh = hole_area_thresh
        kernel_size = self.pixels_per_meter * agent_radius * 2 
        kernel_size = int(kernel_size) + (int(kernel_size) % 2 == 0)

        self._navigable_kernel = np.ones((kernel_size, kernel_size), np.uint8)

        self._has_up_stair = False
        self._has_down_stair = False
        self._done_initializing = False
        self._this_floor_explored = False

        self._up_stair_frontiers_px = np.array([])
        self._up_stair_frontiers = np.array([])
        self._down_stair_frontiers_px = np.array([])
        self._down_stair_frontiers = np.array([])

        self._up_stair_start = np.array([])
        self._up_stair_end = np.array([])
        self._down_stair_start = np.array([])
        self._down_stair_end = np.array([])

        self._carrot_goal_px = np.array([])
        self._explored_up_stair = False
        self._explored_down_stair = False

        self.stair_boundary = np.zeros((size, size), dtype=bool)
        self.stair_boundary_goal = np.zeros((size, size), dtype=bool)
        self._floor_num_steps = 0
        self._disabled_frontiers = set()
        self._disabled_frontiers_px =  np.array([], dtype=np.float64).reshape(0, 2) # np.array([])
        self._climb_stair_paused_step = 0
        self._disable_end = False
        self._look_for_downstair_flag = False
        self._potential_stair_centroid_px = np.array([])
        self._potential_stair_centroid = np.array([])
        self._reinitialize_flag = False
        self._tight_search_thresh = True
        self._best_frontier_selection_count = {}

        self.previous_frontiers = []  # 存储之前已经可视化过的 frontiers 的索引
        self.frontier_visualization_info = {}  # 存储每个 frontier 对应的 步数
        self._each_step_rgb = {} # 存储每一步对应的rgb, 仅供debug
        self._finish_first_explore = False
        self._neighbor_search = False

        ## stair_status
        self._climb_stair_over = True 
        self._reach_stair = False 
        self._reach_stair_centroid = False
        self._stair_frontier = None
        
    def reset(self) -> None:
        super().reset()
        # initialize class variable
        self._map_dtype = np.dtype(bool)
        self._frontiers_px = np.array([])
        self.frontiers = np.array([])
        self.radius_padding_color = (100, 100, 100)
        self._navigable_map.fill(0)
        self._up_stair_map.fill(0) # for upstairs_map
        self._down_stair_map.fill(0) # for downstairs_map
        self._disabled_stair_map.fill(0) # True for not possible for stair
        self.explored_area.fill(0)
        self.stair_boundary.fill(0)
        self.stair_boundary_goal.fill(0)
        self._frontiers_px = np.array([])
        self.frontiers = np.array([])
        self._has_up_stair = False
        self._has_down_stair = False
        self._explored_up_stair = False
        self._explored_down_stair = False
        self._done_initializing = False
        self._up_stair_frontiers_px = np.array([])
        self._up_stair_frontiers = np.array([])
        self._down_stair_frontiers_px = np.array([])
        self._down_stair_frontiers = np.array([])
        self._up_stair_start = np.array([])
        self._up_stair_end = np.array([])
        self._down_stair_start = np.array([])
        self._down_stair_end = np.array([])
        self._carrot_goal_px = np.array([])
        self._floor_num_steps = 0      
        self._disabled_frontiers = set()
        self._disabled_frontiers_px =  np.array([], dtype=np.float64).reshape(0, 2) # np.array([])
        self._climb_stair_paused_step = 0
        self._disable_end = False
        self._look_for_downstair_flag = False
        self._potential_stair_centroid_px = np.array([])
        self._potential_stair_centroid = np.array([])
        self._reinitialize_flag = False
        self._tight_search_thresh = True
        self._best_frontier_selection_count = {}
        self.previous_frontiers = []  # 存储之前已经可视化过的 frontiers 的索引
        self.frontier_visualization_info = {}  # 存储每个 frontier 对应的 RGB 图以及箭头标记
        self._each_step_rgb = {}
        self._each_step_rgb_phash = {}
        self._finish_first_explore = False
        self._neighbor_search = False
    
    def is_robot_in_stair_map_fast(self, robot_px:np.ndarray, stair_map: np.ndarray):
        """
        高效判断以机器人质心为圆心、指定半径的圆是否覆盖 stair_map 中值为 1 的点。

        Args:
            stair_map (np.ndarray): 地图的 _stair_map。
            robot_xy_2d (np.ndarray): 机器人质心在相机坐标系下的 (x, y) 坐标。
            agent_radius (float): 机器人在相机坐标系中的半径。
            obstacle_map: 包含坐标转换功能和地图信息的对象。

        Returns:
            bool: 如果范围内有值为 1,则返回 True,否则返回 False。
        """
        x, y = robot_px[0, 0], robot_px[0, 1]

        # 转换半径到地图坐标系
        radius_px = self._agent_radius * self.pixels_per_meter

        # 获取地图边界
        rows, cols = stair_map.shape
        x_min = max(0, int(x - radius_px))
        x_max = min(cols - 1, int(x + radius_px))
        y_min = max(0, int(y - radius_px))
        y_max = min(rows - 1, int(y + radius_px))

        # 提取感兴趣的子矩阵
        sub_matrix = stair_map[y_min:y_max + 1, x_min:x_max + 1]

        # 创建圆形掩码
        # y_indices, x_indices = np.ogrid[0:sub_matrix.shape[0], 0:sub_matrix.shape[1]]
        y_indices, x_indices = np.ogrid[y_min:y_max + 1, x_min:x_max + 1]
        mask = (y_indices - y) ** 2 + (x_indices - x) ** 2 <= radius_px ** 2

        # 获取sub_matrix中为 True 的坐标
        if np.any(sub_matrix[mask]):  # 在圆形区域内有值为True的元素
            # 找出sub_matrix中值为 True 的位置
            true_coords_in_sub_matrix = np.column_stack(np.where(sub_matrix))  # 获取相对于sub_matrix的坐标

            # 通过mask过滤,只留下圆形区域内为 True 的坐标
            true_coords_filtered = true_coords_in_sub_matrix[mask[true_coords_in_sub_matrix[:, 0], true_coords_in_sub_matrix[:, 1]]]

            # 将相对坐标转换为 stair_map 中的坐标
            true_coords_in_stair_map = true_coords_filtered + [y_min, x_min]
            
            return True, true_coords_in_stair_map
        else:
            return False, None

    def visualize(self) -> np.ndarray:
        # 影响画图
        temp_disabled_frontiers = np.atleast_2d(np.array(list(self._disabled_frontiers)))
        if len(temp_disabled_frontiers[0]) > 0:
            self._disabled_frontiers_px = self._xy_to_px(temp_disabled_frontiers)
        """Visualizes the map."""
        vis_img = np.ones((*self._map.shape[:2], 3), dtype=np.uint8) * 255
        # Draw explored area in light green
        vis_img[self.explored_area == 1] = (200, 255, 200)
        # Draw unnavigable areas in gray
        vis_img[self._navigable_map == 0] = self.radius_padding_color
        # Draw obstacles in black
        vis_img[self._map == 1] = (0, 0, 0)
        # Draw detected upstair area in purple
        vis_img[self._up_stair_map == 1] = (128,0,128)
        # Draw detected downstair area in orange
        vis_img[self._down_stair_map == 1] = (139, 26, 26)
        
        for carrot in self._carrot_goal_px:
            cv2.circle(vis_img, tuple([int(i) for i in carrot]), 5, (42, 42, 165), 2) # 红色空心圆
        # vis_img[self._temp_down_stair_map == 1] = (139, 69, 19)
        # Draw frontiers in blue (200, 0, 0), 似乎是bgr，不是rgb
        if len(self._down_stair_end) > 0:
            cv2.circle(vis_img, tuple([int(i) for i in self._down_stair_end]), 5, (0, 255, 255), 2) # 黄色空心圆
        if len(self._up_stair_end) > 0:
            cv2.circle(vis_img, tuple([int(i) for i in self._up_stair_end]), 5, (0, 255, 255), 2) # 黄色空心圆
        if len(self._down_stair_start) > 0:
            cv2.circle(vis_img, tuple([int(i) for i in self._down_stair_start]), 5, (101, 96, 127), 2) # 粉色空心圆
        if len(self._up_stair_start) > 0:
            cv2.circle(vis_img, tuple([int(i) for i in self._up_stair_start]), 5, (101, 96, 127), 2) # 粉色空心圆
        for frontier in self._frontiers_px:
            temp = np.array([int(i) for i in frontier])
            if temp not in self._disabled_frontiers_px:
                cv2.circle(vis_img, tuple(temp), 5, (200, 0, 0), 2) # 蓝色空心圆
        # Draw stair frontiers in orange (100, 0, 0)
        for up_stair_frontier in self._up_stair_frontiers_px:
            cv2.circle(vis_img, tuple([int(i) for i in up_stair_frontier]), 5, (255, 128, 0), 2) # 淡蓝色空心圆
        for down_stair_frontier in self._down_stair_frontiers_px:
            cv2.circle(vis_img, tuple([int(i) for i in down_stair_frontier]), 5, (19, 69, 139), 2) # 暗橙色空心圆
                    
        for potential_downstair in self._potential_stair_centroid_px:
            cv2.circle(vis_img, tuple([int(i) for i in potential_downstair]), 5, (128, 69, 128), 2) # 紫色空心圆

        vis_img = cv2.flip(vis_img, 0)

        if len(self._camera_positions) > 0:
            self._traj_vis.draw_trajectory(
                vis_img,
                self._camera_positions,
                self._last_camera_yaw,
            )

        return vis_img
    def visualize_and_save_frontiers(self, save_dir="debug/20250124/test_frontier_rgb/v7"):
        """
        Visualizes frontiers on the RGB images using the stored information in self._each_step_rgb
        and self.frontier_visualization_info, and saves the images to the specified directory.

        Args:
            save_dir (str): Directory to save the visualized images.
        """
        # Ensure the save directory exists
        os.makedirs(save_dir, exist_ok=True)

        for floor_num_steps, rgb_image in self._each_step_rgb.items():
            # Get the RGB image for the current floor step
            # Get the frontier visualization information for the current floor step
            for frontier, info in self.frontier_visualization_info.items():
                visualized_rgb = rgb_image.copy()
                if info['floor_num_steps'] == floor_num_steps:
                    arrow_end_pixel = info['arrow_end_pixel']
                    arrow_start_pixel = (visualized_rgb.shape[1] // 2, int(visualized_rgb.shape[0] * 0.9))  # Adjusted start point

                    # Draw a solid ellipse at the start point (robot position)
                    center = (arrow_start_pixel[0], arrow_start_pixel[1])
                    axes = (5, 5)  # Major and minor axes lengths
                    cv2.ellipse(visualized_rgb, center, axes, 0, 0, 360, (0, 255, 0), -1)

                    # Draw the arrow on the RGB image
                    cv2.arrowedLine(
                        visualized_rgb,
                        arrow_start_pixel,
                        arrow_end_pixel,
                        color=(0, 0, 255),  # Red color for the arrow
                        thickness=4,
                        tipLength=0.08
                    )
                    
                    # Save the visualized image
                    filename = f"floor_{floor_num_steps}_x_{int(frontier[0])}_y_{int(frontier[1])}.png"
                    save_path = os.path.join(save_dir, filename)
                    cv2.imwrite(save_path, visualized_rgb)
                    print(f"Saved: {save_path}")
    def update_map_with_stair(
        self,
        depth: np.ndarray,
        tf_camera_to_episodic: np.ndarray,
        min_depth: float,
        max_depth: float,
        fx: float,
        fy: float,
        topdown_fov: float,
        stair_mask: np.ndarray, # only a (480,640) mask, also for multiple stairs
        seg_mask: np.ndarray,
        agent_pitch_angle: int,
        search_stair_over: bool,
        reach_stair: bool,
        climb_stair_flag: int,
        explore: bool = True,
        update_obstacles: bool = True,
    ) -> None:
        """
        Adds all obstacles from the current view to the map. Also updates the area
        that the robot has explored so far.

        Args:
            depth (np.ndarray): The depth image to use for updating the object map. It
                is normalized to the range [0, 1] and has a shape of (height, width).

            tf_camera_to_episodic (np.ndarray): The transformation matrix from the
                camera to the episodic coordinate frame.
            min_depth (float): The minimum depth value (in meters) of the depth image.
            max_depth (float): The maximum depth value (in meters) of the depth image.
            fx (float): The focal length of the camera in the x direction.
            fy (float): The focal length of the camera in the y direction.
            topdown_fov (float): The field of view of the depth camera projected onto
                the topdown map.
            explore (bool): Whether to update the explored area.
            update_obstacles (bool): Whether to update the obstacle map.
        """

        if update_obstacles:
            if self._hole_area_thresh == -1:
                filled_depth = depth.copy()
                filled_depth[depth == 0] = 1.0
            else:
                filled_depth = fill_small_holes(depth, self._hole_area_thresh)
                         
            scaled_depth = filled_depth * (max_depth - min_depth) + min_depth


            # for stair
            ## upstair or look down to find downstair
            if np.any(stair_mask) > 0 and np.sum(seg_mask == STAIR_CLASS_ID) > 20: # STAIR_CLASS_ID in seg_mask
                stair_map = (seg_mask == STAIR_CLASS_ID)
                fusion_stair_mask = stair_mask & stair_map
                if np.any(fusion_stair_mask) > 0: # 检测出楼梯
                # fusion_stair_mask = stair_mask
                    stair_depth = np.full_like(depth, max_depth)
                    scaled_depth_stair = scaled_depth.copy()
                    stair_depth[fusion_stair_mask] = scaled_depth_stair[fusion_stair_mask]
                    # scaled_depth_stair[fusion_stair_mask] = max_depth
                    # 在楼梯地图中标记楼梯位置
                    # stair_xy_points = stair_clouds["stair"][:, :2]  # easy to false positive
                    stair_cloud_camera_frame = get_point_cloud(stair_depth, fusion_stair_mask, fx, fy)
                    stair_cloud_episodic_frame = transform_points(tf_camera_to_episodic, stair_cloud_camera_frame)
                    stair_xy_points = stair_cloud_episodic_frame[:, :2]
                    stair_pixel_points = self._xy_to_px(stair_xy_points)
                    if agent_pitch_angle >= 0 and climb_stair_flag != 2: # 有可能是reverse_climb_stair
                    # 遍历每个 stair_pixel_points 点，进行标记和清除
                        for x, y in stair_pixel_points:
                            # 在 _stair_map 上标记为确定的楼梯
                            if 0 <= x < self._up_stair_map.shape[1] and 0 <= y < self._up_stair_map.shape[0] and self._up_stair_map[y, x] == 0:
                                self._up_stair_map[y, x] = 1
                        self._map[self._up_stair_map == 1] = 1
                    elif agent_pitch_angle < 0 and climb_stair_flag != 1: # 有可能是reverse_climb_stair
                        for x, y in stair_pixel_points:
                            # 在 _stair_map 上标记为确定的楼梯
                            if 0 <= x < self._down_stair_map.shape[1] and 0 <= y < self._down_stair_map.shape[0] and self._down_stair_map[y, x] == 0:
                                self._down_stair_map[y, x] = 1 
                        self._map[self._down_stair_map == 1] = 1 # 不可通行范围大一点，减少探索

            ## normal to look for downstair
            ## 反转深度，但发现对短楼梯不好使 
            if agent_pitch_angle <= 0 and reach_stair == False: # 靠近楼梯的时候也要找，不然楼梯间的时候下楼误以为上楼了
                filled_depth_for_stair = fill_small_holes(depth, self._hole_area_thresh)
                inverted_depth_for_stair = max_depth - filled_depth_for_stair * (max_depth - min_depth)
                inverted_mask = inverted_depth_for_stair < 2 # inverted_depth_for_stair < 2 # 3 <= true depth value < max_depth 
                inverted_point_cloud_camera_frame = get_point_cloud(inverted_depth_for_stair, inverted_mask, fx, fy)
                inverted_point_cloud_episodic_frame = transform_points(tf_camera_to_episodic, inverted_point_cloud_camera_frame)
                # below_ground_obstacle_cloud = filter_points_by_height_below_ground(inverted_point_cloud_episodic_frame)
                below_ground_obstacle_cloud_0 = filter_points_by_height_below_ground_0(inverted_point_cloud_episodic_frame)
                below_ground_xy_points = below_ground_obstacle_cloud_0[:, :2] # below_ground_obstacle_cloud[:, :2]
                # 获取需要赋值的点的像素坐标
                below_ground_pixel_points = self._xy_to_px(below_ground_xy_points)
                self._down_stair_map[below_ground_pixel_points[:, 1], below_ground_pixel_points[:, 0]] = 1
                
            # 不爬楼梯的时候标注
            if search_stair_over == True: # reach_stair == False:
                mask = scaled_depth < max_depth
                point_cloud_camera_frame = get_point_cloud(scaled_depth, mask, fx, fy)
                point_cloud_episodic_frame = transform_points(tf_camera_to_episodic, point_cloud_camera_frame)
                obstacle_cloud = filter_points_by_height(point_cloud_episodic_frame, self._min_height, self._max_height)

                xy_points = obstacle_cloud[:, :2]
                pixel_points = self._xy_to_px(xy_points)

                self._map[pixel_points[:, 1], pixel_points[:, 0]] = 1
            
            self._up_stair_map = self._up_stair_map & (~self._disabled_stair_map)
            self._down_stair_map = self._down_stair_map & (~self._disabled_stair_map)
            
            stair_dilated_mask = (self._up_stair_map == 1) | (self._down_stair_map == 1)
            # stair_dilated_mask = self._up_stair_map == 1 | self._down_stair_map == 1 # ((self._map == 1) & (self._up_stair_map == 1)) | ((self._map == 1) & (self._down_stair_map == 1))
            self._map[stair_dilated_mask] = 0

            dilated_map = cv2.dilate(
                self._map.astype(np.uint8),
                self._navigable_kernel,
                iterations=1
            )
            dilated_map[stair_dilated_mask] = 1
            self._map[stair_dilated_mask] = 1
            # 不让楼梯膨胀
            self._navigable_map = 1 - dilated_map.astype(bool)

        if not explore:
            return

        # Update the explored area
        agent_xy_location = tf_camera_to_episodic[:2, 3]
        agent_pixel_location = self._xy_to_px(agent_xy_location.reshape(1, 2))[0]
        new_explored_area = reveal_fog_of_war(
            top_down_map=self._navigable_map.astype(np.uint8),
            current_fog_of_war_mask=np.zeros_like(self._map, dtype=np.uint8),
            current_point=agent_pixel_location[::-1],
            current_angle=-extract_yaw(tf_camera_to_episodic),
            fov=np.rad2deg(topdown_fov),
            max_line_len=max_depth * self.pixels_per_meter,
        ) # 传入的是一个假设全都是False的值，如果中间有障碍物，最终只会将观察者视野中未被障碍物遮挡的部分标记为 True
        new_explored_area = cv2.dilate(new_explored_area, np.ones((3, 3), np.uint8), iterations=1)

        self.explored_area[new_explored_area > 0] = 1
        self.explored_area[self._navigable_map == 0] = 0

        # Compute frontier locations
        self._frontiers_px = self._get_frontiers() # _get_frontiers()
        if len(self._frontiers_px) == 0: 
            self.frontiers = np.array([])
        else:
            self.frontiers = self._px_to_xy(self._frontiers_px)

        # Compute stair frontier
            
        if np.sum(self._down_stair_map == 1) > 20:
            self._down_stair_map = cv2.morphologyEx(self._down_stair_map.astype(np.uint8) , cv2.MORPH_CLOSE, self._navigable_kernel,) # 一条细线做先膨胀后腐蚀操作

            # 应该剔除小的，不连通的区域
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(self._down_stair_map, connectivity=8)
            min_area_threshold = 10  # 设定最小面积阈值为 10（即小于 10 个像素的区域被视为小连通域）
            filtered_map = np.zeros_like(self._down_stair_map)
            max_area = 0
            max_label = 1
            for i in range(1, num_labels):  # 从1开始，0是背景
                area = stats[i, cv2.CC_STAT_AREA]
                if area >= min_area_threshold:
                    filtered_map[labels == i] = 1  # 保留面积大于阈值的区域
                    # 更新最大面积区域的标签
                    if area > max_area:
                        max_area = area
                        max_label = i

            self._down_stair_map = filtered_map
            self._down_stair_frontiers_px = np.array([[centroids[max_label][0], centroids[max_label][1]]])

            self._down_stair_frontiers = self._px_to_xy(self._down_stair_frontiers_px)
            self._has_down_stair = True
            self._look_for_downstair_flag = False
            self._potential_stair_centroid_px = np.array([])
            self._potential_stair_centroid = np.array([])
        else:
            # self._down_stair_frontiers_px = np.array([])  # 没有楼梯区域时
            # self._has_down_stair = False
            if np.sum(self._down_stair_map == 1) > 0:
                # self._down_stair_map = cv2.morphologyEx(self._down_stair_map.astype(np.uint8) , cv2.MORPH_CLOSE, self._navigable_kernel,) # 一条细线做先膨胀后腐蚀操作

                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(self._down_stair_map.astype(np.uint8), connectivity=8)
                max_area = 0
                max_label = 1
                # 逐个找最大区域的质心，直到有向下楼梯
                for i in range(1, num_labels):  # 从1开始，0是背景
                    area = stats[i, cv2.CC_STAT_AREA]
                    if area > max_area:
                        max_area = area
                        max_label = i
                self._potential_stair_centroid_px = np.array([[centroids[max_label][0], centroids[max_label][1]]])
                self._potential_stair_centroid = self._px_to_xy(self._potential_stair_centroid_px)
                # self._down_stair_map = filtered_map
                # self._down_stair_frontiers_px = np.array([[centroids[max_label][0], centroids[max_label][1]]])
                # self._down_stair_frontiers = self._px_to_xy(self._down_stair_frontiers_px)
                # self._has_down_stair = True
                # self._look_for_downstair_flag = False

        if np.sum(self._up_stair_map == 1) > 20:
            self._up_stair_map = cv2.morphologyEx(self._up_stair_map.astype(np.uint8) , cv2.MORPH_CLOSE, self._navigable_kernel,) # 一条细线做先膨胀后腐蚀操作

            # 应该剔除小的，不连通的区域
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(self._up_stair_map, connectivity=8)
            min_area_threshold = 10  # 设定最小面积阈值为 10（即小于 10 个像素的区域被视为小连通域）
            filtered_map = np.zeros_like(self._up_stair_map)
            max_area = 0
            max_label = 1
            for i in range(1, num_labels):  # 从1开始，0是背景
                area = stats[i, cv2.CC_STAT_AREA]
                if area >= min_area_threshold:
                    filtered_map[labels == i] = 1  # 保留面积大于阈值的区域
                    # 更新最大面积区域的标签
                    if area > max_area:
                        max_area = area
                        max_label = i

            self._up_stair_map = filtered_map
            self._up_stair_frontiers_px = np.array([[centroids[max_label][0], centroids[max_label][1]]])

            self._up_stair_frontiers = self._px_to_xy(self._up_stair_frontiers_px)
            self._has_up_stair = True
        else:
            self._up_stair_frontiers_px = np.array([])  # 没有楼梯区域时
            self._has_up_stair = False

        if len(self._down_stair_frontiers) == 0 and np.sum(self._down_stair_map) > 0:
            # 标识，提示agent往这边导航
            self._look_for_downstair_flag = True 
    def project_frontiers_to_rgb_hush(self, rgb: np.ndarray) -> dict: 
        # , robot_xy: np.ndarray, min_arrow_length: float = 4.0, max_arrow_length: float = 10.0
        """
        Projects the frontiers from the map to the corresponding positions in the RGB image,
        and visualizes them on the RGB image.

        Args:
            rgb (np.ndarray): The RGB image (H x W x 3).
            robot_xy (np.ndarray): The robot's position in the map coordinates (x, y).
            min_arrow_length (float): The minimum length of the arrow in meters. Default is 4.0 meter.
            max_arrow_length (float): The maximum length of the arrow in meters. Default is 10.0 meter.

        Returns:
            dict: A dictionary containing the visualized RGB images with frontiers marked for each new frontier.
        """
        if len(self.frontiers) == 0:
            return {}  # No frontiers to project

        new_frontiers = [f for f in self.frontiers if f.tolist() not in self.previous_frontiers]
        if len(new_frontiers) == 0:
            return {}
        
        self.previous_frontiers.extend([f.tolist() for f in new_frontiers])

        visualized_rgb_ori = rgb.copy()
        self._each_step_rgb[self._floor_num_steps] = visualized_rgb_ori

        for frontier in new_frontiers:

            visualization_info = {
                'floor_num_steps': self._floor_num_steps,
            }
            self.frontier_visualization_info[tuple(frontier)] = visualization_info

    def extract_frontiers_with_image(self, frontier):
        """
        Visualizes frontiers on the RGB images using the stored information in self._each_step_rgb
        and self.frontier_visualization_info. Draws a blue circle with index at the end of a line.
        """
        
        # 获取相关信息
        floor_num_steps = self.frontier_visualization_info[tuple(frontier)]['floor_num_steps']
        visualized_rgb = self._each_step_rgb[floor_num_steps].copy()
        return floor_num_steps, visualized_rgb
    
class ValueMapUpdater(ValueMap):
    def _update_value_map(self, observations_cache, text_prompt, target_object, itm) -> None:
        # 收集所有环境的 RGB 图像和文本
        all_rgb = []
        all_texts = []
        
        # 提取当前环境的 RGB 图像
        rgb = observations_cache["value_map_rgbd"][0][0]
        all_rgb.append(rgb)
        
        # 准备当前环境的文本
        text = text_prompt.replace("target_object", target_object.replace("|", "/"))
        all_texts.append(text)
        
        # 批量计算所有环境的余弦相似度
        all_cosines = itm.cosine(all_rgb, all_texts)
        
        # 更新每个环境的 value map 和保存结果
        # 更新 value map
        self.update_map(
            np.array([all_cosines]),  # 将余弦相似度包装为数组
            observations_cache["value_map_rgbd"][0][1],
            observations_cache["value_map_rgbd"][0][2],
            observations_cache["value_map_rgbd"][0][3],
            observations_cache["value_map_rgbd"][0][4],
            observations_cache["value_map_rgbd"][0][5]
        )
        
        # 更新 agent 轨迹
        self.update_agent_traj(
            observations_cache["robot_xy"],
            observations_cache["robot_heading"],
        )
        
        # 保存 BLIP 余弦相似度
        self._blip_cosine = all_cosines

    def reset(self) -> None:
        super().reset()
        self._value_map.fill(0)
        
        # initialize class variable
        self._confidence_masks = {}
        self._camera_positions = []
        self._last_camera_yaw = 0.0
        self._min_confidence = 0.25
        self._decision_threshold = 0.35

class ObjectMapUpdater(ObjectPointCloudMap,BaseMap):
    def __init__(self, erosion_size: float, size: int = 1000) -> None:
        ObjectPointCloudMap.__init__(self,erosion_size=erosion_size)
        BaseMap.__init__(self,size=size)
        self._map = np.zeros((size, size), dtype=bool)
        self._disabled_object_map = np.zeros((size, size), dtype=bool)  # for disabled objects
        self.clouds = {}
        self.use_dbscan = True
        self.stair_clouds: Dict[str, np.ndarray] = {}
        self.visualization = np.ones((*self._map.shape[:2], 3), dtype=np.uint8) * 255  # 创建白色背景的图像
        self.each_step_objects = {}
        self.each_step_rooms = {}
        self.this_floor_rooms = set()
        self.this_floor_objects = set()

    def reset(self) -> None:
        ObjectPointCloudMap.reset(self)
        BaseMap.reset(self)
        self._map.fill(0)
        self._disabled_object_map.fill(0)
        self.use_dbscan = True
        self.clouds = {}
        self.last_target_coord = None
        self.stair_clouds = {}

    def update_map(
        self,
        object_name: str,
        depth_img: np.ndarray,
        object_mask: np.ndarray,
        tf_camera_to_episodic: np.ndarray,
        min_depth: float,
        max_depth: float,
        fx: float,
        fy: float,
    ) -> None:
        """Updates the object map with the latest information from the agent."""
        local_cloud = self._extract_object_cloud(depth_img, object_mask, min_depth, max_depth, fx, fy)
        if len(local_cloud) == 0:
            return

        # For second-class, bad detections that are too offset or out of range, we
        # assign a random number to the last column of its point cloud that can later
        # be used to identify which points came from the same detection.
        if too_offset(object_mask):
            within_range = np.ones_like(local_cloud[:, 0]) * np.random.rand()
        else:
            # Mark all points of local_cloud whose distance from the camera is too far
            # as being out of range
            within_range = (local_cloud[:, 0] <= max_depth * 0.95) * 1.0  # 5% margin
            # All values of 1 in within_range will be considered within range, and all
            # values of 0 will be considered out of range; these 0s need to be
            # assigned with a random number so that they can be identified later.
            within_range = within_range.astype(np.float32)
            within_range[within_range == 0] = np.random.rand()
        global_cloud = transform_points(tf_camera_to_episodic, local_cloud)
        global_cloud = np.concatenate((global_cloud, within_range[:, None]), axis=1)


        # Populate topdown map with obstacle locations
        xy_points = global_cloud[:, :2]
        pixel_points = self._xy_to_px(xy_points)
        valid_points_mask = ~self._disabled_object_map[pixel_points[:, 1], pixel_points[:, 0]]

        # Apply the mask to filter out disabled points
        global_cloud = global_cloud[valid_points_mask]
        if len(global_cloud) == 0:
            return  # If global_cloud is empty, return early to avoid further processing
            
        # Populate topdown map with obstacle locations
        xy_points = global_cloud[:, :2]
        pixel_points = self._xy_to_px(xy_points)
        self._map[pixel_points[:, 1], pixel_points[:, 0]] = 1
        self._map = self._map & (~self._disabled_object_map)

        curr_position = tf_camera_to_episodic[:3, 3]
        closest_point = self._get_closest_point(global_cloud, curr_position)
        dist = np.linalg.norm(closest_point[:3] - curr_position)
        if dist <= 0.5: # 1.0
            # Object is too close to trust as a valid object
            return

        if object_name in self.clouds:
            self.clouds[object_name] = np.concatenate((self.clouds[object_name], global_cloud), axis=0)
        else:
            self.clouds[object_name] = global_cloud
    def visualize(self) -> np.ndarray:  
        """Visualizes the map."""
        vis_img = np.ones((*self._map.shape[:2], 3), dtype=np.uint8) * 255
        # Draw goal in red
        vis_img[self._map == 1] = (0, 0, 128)
        vis_img = cv2.flip(vis_img, 0)
        if len(self._camera_positions) > 0:
            self._traj_vis.draw_trajectory(
                vis_img,
                self._camera_positions,
                self._last_camera_yaw,
            )
        return vis_img

    def _extract_object_cloud(
        self,
        depth: np.ndarray,
        object_mask: np.ndarray,
        min_depth: float,
        max_depth: float,
        fx: float,
        fy: float,
    ) -> np.ndarray:
        final_mask = object_mask * 255
        final_mask = cv2.erode(final_mask, None, iterations=self._erosion_size)  # type: ignore

        valid_depth = depth.copy()
        # valid_depth[valid_depth == 0] = 1  # set all holes (0) to just be far (1) 
        # 这样会让本来很近的点映射到远处，原来的代码是有问题的
        valid_mask = (valid_depth > 0) & final_mask
        
        valid_depth = valid_depth * (max_depth - min_depth) + min_depth
        cloud = get_point_cloud(valid_depth, valid_mask, fx, fy) # final_mask
        cloud = get_random_subarray(cloud, 5000)
        if self.use_dbscan:
            cloud = open3d_dbscan_filtering(cloud)

        return cloud
    def update_explored(self, tf_camera_to_episodic: np.ndarray, max_depth: float, cone_fov: float) -> None:
        """
        This method will remove all point clouds in self.clouds that were originally
        detected to be out-of-range, but are now within range. This is just a heuristic
        that suppresses ephemeral false positives that we now confirm are not actually
        target objects.

        Args:
            tf_camera_to_episodic: The transform from the camera to the episode frame.
            max_depth: The maximum distance from the camera that we consider to be
                within range.
            cone_fov: The field of view of the camera.
        """
        camera_coordinates = tf_camera_to_episodic[:3, 3]
        camera_yaw = extract_yaw(tf_camera_to_episodic)

        for obj in self.clouds:
            within_range = within_fov_cone(
                camera_coordinates,
                camera_yaw,
                cone_fov,
                max_depth * 0.5,
                self.clouds[obj],
            )
            range_ids = set(within_range[..., -1].tolist())
            for range_id in range_ids:
                if range_id == 1:
                    # Detection was originally within range
                    continue
                # Remove all points from self.clouds[obj] that have the same range_id
                self.clouds[obj] = self.clouds[obj][self.clouds[obj][..., -1] != range_id]
                
        # 在方法末尾检查并删除所有空的点云数组
        for obj in list(self.clouds.keys()):
            if self.clouds[obj].size == 0:
                del self.clouds[obj]

import pandas as pd
import numpy as np
import logging
import torch
import json
import networkx as nx

PROMPT_SEPARATOR = "|"
STAIR_CLASS_ID = 17  # MPCAT40中 楼梯的类别编号是 16 + 1

# 关键修改：调整列表项生成方式，确保缩进正确
INDENT_L1 = " " * 4
INDENT_L2 = " " * 8

## knowledge graph
with open('knowledge_graph.json', 'r') as f:
    knowledge_graph = nx.node_link_graph(json.load(f))

reference_rooms = [
            "bathroom", "bedroom", "dining_room", "garage", "hall",
            "kitchen", "laundry_room", "living_room", "office", "rec_room"
        ]
# 直接映射表
direct_mapping = {
    # Bathroom 相关
    "bathroom": "bathroom",
    "shower": "bathroom",
    "jacuzzi/indoor": "bathroom",

    # Bedroom 相关
    "bedroom": "bedroom",
    "bedchamber": "bedroom",
    "dorm_room": "bedroom",
    "hotel_room": "bedroom",
    "childs_room": "bedroom",

    # Dining Room 相关
    "dining_room": "dining_room",
    "dining_hall": "dining_room",
    "banquet_hall": "dining_room",
    "restaurant": "dining_room",
    "cafeteria": "dining_room",

    # Garage 相关
    "garage/indoor": "garage",
    "garage/outdoor": "garage",
    "parking_garage/indoor": "garage",
    "parking_garage/outdoor": "garage",

    # Hall 相关
    "entrance_hall": "hall",
    "lobby": "hall",
    "corridor": "hall",
    "mezzanine": "hall",

    # Kitchen 相关
    "kitchen": "kitchen",
    "restaurant_kitchen": "kitchen",

    # Laundry Room 相关
    "laundry_room": "laundry_room",
    "laundromat": "laundry_room",

    # Living Room 相关
    "living_room": "living_room",
    "home_theater": "living_room",
    "television_room": "living_room",

    # Office 相关
    "office": "office",
    "office_cubicles": "office",
    "conference_room": "office",
    "home_office": "office",
    "computer_room": "office",

    # Rec Room 相关
    "recreation_room": "rec_room",
    "playroom": "rec_room",
    "amusement_arcade": "rec_room",
    "gymnasium/indoor": "rec_room",
    "arcade": "rec_room",
}
# 加载知识图谱

def load_floor_probabilities(file_path):
    """
    加载楼层和物体分布概率表格。

    Parameters:
    file_path (str): 表格文件路径。

    Returns:
    pd.DataFrame: 包含物体分布概率的表格。
    """
    try:
        df = pd.read_excel(file_path)
        return df
    except Exception as e:
        logging.error(f"Failed to load floor probabilities: {e}")
        return None
floor_probabilities_df = load_floor_probabilities("falcon/floor_object_possibility.xlsx")

class TorchActionIDs_plook:
    STOP = torch.tensor([[0]], dtype=torch.long)
    MOVE_FORWARD = torch.tensor([[1]], dtype=torch.long)
    TURN_LEFT = torch.tensor([[2]], dtype=torch.long)
    TURN_RIGHT = torch.tensor([[3]], dtype=torch.long)
    LOOK_UP = torch.tensor([[4]], dtype=torch.long)
    LOOK_DOWN = torch.tensor([[5]], dtype=torch.long)

def xyz_yaw_pitch_roll_to_tf_matrix(xyz: np.ndarray, yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Converts a given position and yaw, pitch, roll angles to a 4x4 transformation matrix.

    Args:
        xyz (np.ndarray): A 3D vector representing the position.
        yaw (float): The yaw angle in radians (rotation around Z-axis).
        pitch (float): The pitch angle in radians (rotation around Y-axis).
        roll (float): The roll angle in radians (rotation around X-axis).

    Returns:
        np.ndarray: A 4x4 transformation matrix.
    """
    x, y, z = xyz
    
    # Rotation matrices for yaw, pitch, roll
    R_yaw = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1],
    ])
    R_pitch = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)],
    ])
    R_roll = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)],
    ])
    
    # Combined rotation matrix
    R = R_yaw @ R_pitch @ R_roll
    
    # Construct 4x4 transformation matrix
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = R  # Rotation
    transformation_matrix[:3, 3] = [x, y, z]  # Translation

    return transformation_matrix

def check_stairs_in_upper_50_percent(mask):
    """
    检查在图像的上方30%区域是否有STAIR_CLASS_ID的标记
    参数：
    - mask: 布尔值数组，表示各像素是否属于STARR_CLASS_ID
    
    返回：
    - 如果上方30%区域有True，则返回True，否则返回False
    """
    # 获取图像的高度
    height = mask.shape[0]
    
    # 计算上方50%的区域的高度范围
    upper_50_height = int(height * 0.5)
    
    # 获取上方50%的区域的掩码
    upper_50_mask = mask[:upper_50_height, :]
    
    print(f"Stair upper 50% points: {np.sum(upper_50_mask)}")
    # 检查该区域内是否有True
    if np.sum(upper_50_mask) > 50:  # 如果上方50%区域内有True
        return True
    return False

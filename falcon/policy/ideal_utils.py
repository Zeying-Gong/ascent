import os
from typing import Dict, List

class IdealUtils:
    def extract_scene_with_target(self, scene_id: str, target_object: str = "") -> Dict:
        if target_object == "":
            return {
                "scene_id": None,
                "target_ids": None,
            }
        # 构造语义标注文件的路径
        semantic_file_path = self._get_semantic_file_path(scene_id)

        # 读取语义标注文件并提取目标物体的 ID
        target_ids = self._parse_semantic_file(semantic_file_path, target_object)

        # 返回包含场景 ID 和目标物体 ID 的字典
        return {
            "scene_id": scene_id,
            "target_ids": target_ids
        }


    def _get_semantic_file_path(self, scene_id: str) -> str:
        # 从 scene_id 构造语义标注文件的路径
        scene_dir = os.path.dirname(scene_id)
        scene_name = os.path.basename(scene_id).replace(".basis.glb", "")
        semantic_file_path = os.path.join(scene_dir, f"{scene_name}.semantic.txt")
        return semantic_file_path

    def _parse_semantic_file(self, semantic_file_path: str, target_object: str) -> List[int]:
        target_ids = []
        with open(semantic_file_path, "r") as f:
            for line in f:
                # 跳过注释行
                if line.startswith("#"):
                    continue
                
                # 解析每一行
                parts = line.strip().split(",")
                if len(parts) >= 3:
                    id = int(parts[0])
                    object_name = parts[2].strip('"')
                    # 检查目标物体名称是否在映射表中
                    if target_object in self.object_mapping:
                        mapped_object_name = self.object_mapping[target_object]
                    else:
                        mapped_object_name = target_object
                    
                    if object_name == mapped_object_name:
                        target_ids.append(id)
        return target_ids
import json
import logging
from falcon.policy.data_utils import INDENT_L1, INDENT_L2, floor_probabilities_df, reference_rooms

class MLLMUtils:
    def llm_analyze_single_floor(self, env, target_object_category, frontier_index_list):
        """
        Analyze the environment using the Large Language Model (LLM) to determine the best frontier to explore.

        Parameters:
        env (str): The current environment identifier.
        target_object_category (str): The category of the target object to find.
        frontier_identifiers (list): A list of frontier identifiers (e.g., ["A", "B", "C", "P"]).
        exploration_status (str): A binary string representing the exploration status of each floor.

        Returns:
        str: The identifier of the frontier that is most likely to lead to the target object.
        """
    
        # else, continue to explore on this floor
        prompt = self._prepare_single_floor_prompt(target_object_category, env)

        # Get the visualization of the current environment
        # image = reorient_rescale_map(self._object_map[env].visualization)

        # Analyze the environment using the VLM
        print(f"## Single-floor Prompt:\n{prompt}")
        response = self._llm.chat(prompt)
        
        # Extract the frontier identifier from the response
        if response == "-1":
            temp_frontier_index = 0
        else:
            # Parse the JSON response
            try:
                cleaned_response = response.replace("\n", "").replace("\r", "")
                response_dict = json.loads(cleaned_response)
            except json.JSONDecodeError:
                logging.warning(f"Failed to parse JSON response: {response}")
                temp_frontier_index = 0
            else:
                # Extract Index
                index = response_dict.get("Index", "N/A")
                if index == "N/A":
                    logging.warning("Index not found in response")
                    temp_frontier_index = 0
                else:
                    # Extract Reason
                    reason = response_dict.get("Reason", "N/A")
                    
                    # Form the response string
                    if reason != "N/A":
                        response_str = f"Area Index: {index}. Reason: {reason}"
                        self.vlm_response[env] = "## Single-floor Prompt:\n" + response_str
                        print(f"## Single-floor Response:\n{response_str}")
                    else:
                        print(f"Index: {index}")
                    
                    # Convert index to integer and validate
                    try:
                        index_int = int(index)
                    except ValueError:
                        logging.warning(f"Index is not a valid integer: {index}")
                        temp_frontier_index = 0
                    else:
                        # Check if index is within valid range
                        if 1 <= index_int <= len(frontier_index_list):
                            temp_frontier_index = index_int - 1  # Convert to 0-based index
                        else:
                            logging.warning(f"Index ({index_int}) is out of valid range: 1 to {len(frontier_index_list)}")
                            temp_frontier_index = 0
        
        return frontier_index_list[temp_frontier_index]

    def get_room_probabilities(self, target_object_category: str):
        """
        获取目标对象类别在各个房间类型的概率。
        
        :param target_object_category: 目标对象类别
        :return: 房间类型概率字典
        """
        # 定义一个映射表，用于扩展某些目标对象类别的查询范围
        synonym_mapping = {
            "couch": ["sofa"],
            "sofa": ["couch"],
            # 可以根据需要添加更多映射关系
        }

        # 获取目标对象类别及其同义词
        target_categories = [target_object_category] + synonym_mapping.get(target_object_category, [])

        # 如果目标对象类别及其同义词都不在知识图谱中，直接返回空字典
        if not any(category in self.knowledge_graph for category in target_categories):
            return {}

        room_probabilities = {}
        for room in reference_rooms:
            for category in target_categories:
                if self.knowledge_graph.has_edge(category, room):
                    room_probabilities[room] = round(self.knowledge_graph[category][room]['weight'] * 100, 1)
                    break  # 找到一个有效类别后，不再检查其他类别
            else:
                room_probabilities[room] = 0.0
        return room_probabilities

    def get_floor_probabilities(self, df, target_object_category, floor_num):
        """
        获取当前楼层和场景的物体分布概率。

        Parameters:
        df (pd.DataFrame): 包含物体分布概率的表格。
        target_object_category (str): 目标物体类别。
        floor_num (int): 总楼层数。

        Returns:
        dict: 所有相关楼层的物体分布概率。
        """
        if df is None:
            return None

        # 初始化概率字典
        probabilities = {}

        # 如果检测到的楼层数超出了表格的范围，展示所有已知多楼层场景的概率
        if floor_num > 4:  # 假设表格最多支持 4 层
            logging.warning(f"Floor number {floor_num} exceeds the maximum supported floor number (4). Showing probabilities for maximum multi-floor scenarios.")
            floor_num = 4
            for floor in range(1, 5):
                column_name = f"train_floor{floor_num}_{floor}"
                if column_name in df.columns:
                    prob = df.set_index("category").at[target_object_category, column_name]
                    probabilities[floor] = prob
                else:
                    logging.warning(f"Column {column_name} not found in the probability table.")
                    probabilities[floor] = 0.0  # 默认概率为 0
        else:
            # 展示当前检测到的楼层数的概率
            for floor in range(1, floor_num + 1):
                column_name = f"train_floor{floor_num}_{floor}"
                if column_name in df.columns:
                    prob = df.set_index("category").at[target_object_category, column_name]
                    probabilities[floor] = prob
                else:
                    logging.warning(f"Column {column_name} not found in the probability table.")
                    probabilities[floor] = 0.0  # 默认概率为 0

        return probabilities

    def _prepare_single_floor_prompt(self, target_object_category, env):
        """
        Prepare the prompt for the LLM in a single-floor scenario.
        """

        area_descriptions = []
        self.frontier_rgb_list[env] = []
        for i, step in enumerate(self.frontier_step_list[env]):
            try:
                room = self._object_map[env].each_step_rooms[step] or "unknown room"
                objects = self._object_map[env].each_step_objects[step] or "no visible objects"
                if isinstance(objects, list):
                    objects = ", ".join(objects)
                self.frontier_rgb_list[env].append(self._obstacle_map[env]._each_step_rgb[step])
                area_description = {
                    "area_id": i + 1,
                    "room": room,
                    "objects": objects
                }
                area_descriptions.append(area_description)
            except (IndexError, KeyError) as e:
                logging.warning(f"Error accessing room or objects for step {step}: {e}")
                continue
        # 获取房间-对象关联概率
        room_probabilities = self.get_room_probabilities(target_object_category)
        sorted_rooms = sorted(
            room_probabilities.items(), 
            key=lambda x: (-x[1], x[0])  # 按概率降序排列
        )
        probability_strings = [
            f'{INDENT_L2}"{room.capitalize()}": {prob:.1f}%'
            for room, prob in sorted_rooms
        ]
        prob_entries = ',\n'.join(probability_strings)

        # 生成带缩进的列表项
        formatted_area_descriptions = [
            f'{INDENT_L2}"Area {desc["area_id"]}": "a {desc["room"].replace("_", " ")} containing objects: {desc["objects"]}"'
            for desc in area_descriptions
        ]
        area_entries = ',\n'.join(formatted_area_descriptions)

        # 构建示例输入（手动控制缩进）
        example_input = (
            'Example Input:\n'
            '{\n'
            f'{INDENT_L1}"Goal": "toilet",\n'
            f'{INDENT_L1}"Prior Probabilities between Room Type and Goal Object": [\n'
            f'{INDENT_L2}"Bathroom": 90.0%,\n'
            f'{INDENT_L2}"Bedroom": 10.0%,\n'
            f'{INDENT_L1}],\n'
            f'{INDENT_L1}"Area Descriptions": [\n'
            f'{INDENT_L2}"Area 1": "a bathroom containing objects: shower, towel",\n'
            f'{INDENT_L2}"Area 2": "a bedroom containing objects: bed, nightstand",\n'
            f'{INDENT_L2}"Area 3": "a garage containing objects: car",\n'
            f'{INDENT_L1}]\n'
            '}'
        ).strip()
        # 构建实际输入（避免使用dedent）
        actual_input = (
            'Now answer question:\n'
            'Input:\n'
            '{\n'
            f'{INDENT_L1}"Goal": "{target_object_category}",\n'
            f'{INDENT_L1}"Prior Probabilities between Room Type and Goal Object": [\n'
            f'{prob_entries}\n'
            f'{INDENT_L1}],\n'
            f'{INDENT_L1}"Area Descriptions": [\n'
            f'{area_entries}\n'
            f'{INDENT_L1}]\n'
            '}'
        ).strip()
        prompt = "\n".join([
            "You need to select the optimal area based on prior probabilistic data and environmental context.",
            "You need to answer the question in the following JSON format:",
            example_input,
            'Example Response:\n{"Index": "1", "Reason": "Shower and towel in Bathroom indicate toilet location, with high probability (90.0%)."}',
            actual_input
        ])
        # 最终prompt组装保持不变...
        return prompt
    
    def _prepare_multiple_floor_prompt(self, target_object_category, env):
        """
        多楼层决策提示生成（兼容单楼层风格）
        """
        # =============== 基础数据准备 ===============
        current_floor = self._cur_floor_index[env] + 1 # 从1开始
        total_floors = self.floor_num[env]
        floor_probs = self.get_floor_probabilities(floor_probabilities_df, target_object_category, total_floors)
        floor_probability_strings = [
            f'{INDENT_L2}"Floor {floor}": {prob:.1f}%'
            for floor, prob in floor_probs.items()
        ]
        floor_prob_entries = ',\n'.join(floor_probability_strings) 
        room_probabilities = self.get_room_probabilities(target_object_category)
        sorted_rooms = sorted(
            room_probabilities.items(), 
            key=lambda x: (-x[1], x[0])  # 按概率降序排列
        )
        probability_strings = [
            f'{INDENT_L2}"{room.capitalize()}": {prob:.1f}%'
            for room, prob in sorted_rooms
        ]
        prob_entries = ',\n'.join(probability_strings)
        # =============== 楼层特征描述 ===============
        floor_descriptions = []
        for floor in range(total_floors):
            try:
                # 获取楼层特征
                rooms = self._object_map_list[env][floor].this_floor_rooms or {"unknown rooms"}
                objects = self._object_map_list[env][floor].this_floor_objects or {"unknown objects"}
                # 将 set 转换为字符串（以逗号分隔）
                rooms_str = ", ".join(rooms)
                objects_str = ", ".join(objects)
                floor_description = {
                    "floor_id": floor + 1,
                    "status": 'Current floor' if floor + 1 == current_floor else 'Other floor',
                    # "have_explored": str(self._obstacle_map_list[env][floor]._done_initializing),
                    "fully_explored": self._obstacle_map_list[env][floor]._this_floor_explored,
                    "room": rooms_str,
                    "objects": objects_str,
                }
                floor_descriptions.append(floor_description)
            except Exception as e:
                logging.error(f"Error describing floor {floor}: {e}")
                continue

        # 生成带缩进的列表项（合并条件判断）
        formatted_floor_descriptions = [
            f'{INDENT_L2}"Floor {desc["floor_id"]}": "{desc["status"]}. There are room types: {desc["room"]}, containing objects: {desc["objects"]}'
            + ('. You do not need to explore this floor again"' if desc.get("fully_explored", False) else '"')
            for desc in floor_descriptions
        ]

        floor_entries = ',\n'.join(formatted_floor_descriptions)
        example_input = (
            'Example Input:\n'
            '{\n'
            f'{INDENT_L1}"Goal": "bed",\n'
            f'{INDENT_L1}"Prior Probabilities between Floor and Goal Object": [\n'
            f'{INDENT_L2}"Floor 1": 10.0%,\n'
            f'{INDENT_L2}"Floor 2": 10.0%,\n'
            f'{INDENT_L2}"Floor 3": 80.0%,\n'
            f'{INDENT_L1}],\n'
            f'{INDENT_L1}"Prior Probabilities between Room Type and Goal Object": [\n'
            f'{INDENT_L2}"Bedroom": 80.0%,\n'
            f'{INDENT_L2}"Living room": 15.0%,\n'
            f'{INDENT_L2}"Bathroom": 5.0%,\n'
            f'{INDENT_L1}],\n'
            f'{INDENT_L1}"Floor Descriptions": [\n'
            f'{INDENT_L2}"Floor 1": "Current floor. There are room types: hall, living room, containing objects: tv, sofa",\n'
            f'{INDENT_L2}"Floor 2": "Other floor. There are room types: bathroom containing objects: shower, towel. You do not need to explore this floor again",\n'
            f'{INDENT_L2}"Floor 3": "Other floor. There are room types: unknown rooms containing objects: unknown objects",\n'
            f'{INDENT_L1}]\n'
            '}'
        ).strip()

        actual_input = (
            'Now answer question:\n'
            'Input:\n'
            '{\n'
            f'{INDENT_L1}"Goal": "{target_object_category}",\n'
            f'{INDENT_L1}"Prior Probabilities between Floor and Goal Object": [\n'
            f'{floor_prob_entries}\n'
            f'{INDENT_L1}],\n'
            f'{INDENT_L1}"Prior Probabilities between Room Type and Goal Object": [\n'
            f'{prob_entries}\n'
            f'{INDENT_L1}],\n'
            f'{INDENT_L1}"Floor Descriptions": [\n'
            f'{floor_entries}\n'
            f'{INDENT_L1}]\n'
            '}'
        ).strip()

        # =============== 组合完整提示 ===============
        prompt =  "\n".join([
            "You need to select the optimal floor based on prior probabilistic data and environmental context.",
            "You need to answer the question in the following JSON format:",
            example_input,
            'Example Response:\n{"Index": "3", "Reason": "The bedroom is most likely to be on the Floor 3, and the room types and object types on the Floor 1 and Floor 2 are not directly related to the target object bed, especially it do not need to explore Floor 2 again."}',
            actual_input
        ])
        
        return prompt

    def _extract_multiple_floor_decision(self, multi_floor_response, env) -> int:
        """
        从LLM响应中提取多楼层决策
        
        参数:
            multi_floor_response (str): LLM的原始响应文本
            current_floor (int): 当前楼层索引 (0-based)
            total_floors (int): 总楼层数
            
        返回:
            int: 楼层决策 0/1/2，解析失败返回0
        """
        # 防御性输入检查
        try:
            # 解析 LLM 的回复
            cleaned_response = multi_floor_response.replace("\n", "").replace("\r", "")
            response_dict = json.loads(cleaned_response)
            target_floor_index = int(response_dict.get("Index", -1))
            current_floor = self._cur_floor_index[env] + 1  # 当前楼层（从1开始）
            reason = response_dict.get("Reason", "N/A")
            # Form the response string
            if reason != "N/A":
                response_str = f"Floor Index: {target_floor_index}. Reason: {reason}"
                self.vlm_response[env] = "## Multi-floor Prompt:\n" + response_str
                print(f"## Multi-floor Response:\n{response_str}")
            # 检查目标楼层是否合理
            if target_floor_index <= 0 or target_floor_index > self.floor_num[env]:
                logging.warning("Invalid floor index from LLM response. Returning current floor.")
                return current_floor  # 返回当前楼层

            return target_floor_index  # 返回目标楼层

        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse LLM response: {e}")
        except Exception as e:
            logging.error(f"Error extracting floor decision: {e}")

        # 如果解析失败或异常，返回当前楼层
        return self._cur_floor_index[env] + 1  # 当前楼层（从1开始）
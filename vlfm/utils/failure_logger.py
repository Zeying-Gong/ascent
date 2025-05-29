# failure_logger.py
import json
from collections import defaultdict

# 用于存储每个失败原因的统计数据和每次失败的详细信息
failure_stats = defaultdict(int)  # 统计每个 failure_cause 出现的次数
failure_records = []  # 保存每次失败的详细记录

def save_failure_data(filename="failure_data.json"):
    """Save failure statistics and detailed records to a file."""
    output_data = {
        "failure_stats": dict(failure_stats),
        "failure_records": failure_records
    }
    
    with open(filename, "w") as f:
        json.dump(output_data, f, indent=4)

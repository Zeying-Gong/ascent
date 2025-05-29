import numpy as np
from PIL import Image
from typing import Any, Optional, List

import numpy as np

class BLIP2ITMClient:
    def __init__(self, port: int = 12182):
        self.url = f"http://localhost:{port}/blip2itm"

    def cosine(self, images: List[np.ndarray], texts: List[str]) -> List[float]:
        # 模拟发送请求到服务端
        print(f"Sending {len(images)} images and {len(texts)} texts to server...")
        # 这里假设 send_request 是一个模拟函数，实际需要替换为真实的 HTTP 请求
        response = self._mock_send_request(images, texts, method="cosine")
        return response

    def _mock_send_request(self, images: List[np.ndarray], texts: List[str], method: str) -> List[float]:
        # 模拟服务端返回的响应
        print(f"Mocking server response for method: {method}")
        # 返回一个随机的相似度列表
        return [np.random.random() for _ in range(len(images))]

def create_dummy_image(width: int = 224, height: int = 224) -> np.ndarray:
    """创建一个随机的 dummy 图像"""
    return np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)

if __name__ == "__main__":
    # 初始化客户端
    client = BLIP2ITMClient(port=12182)

    # 创建测试数据
    images = [create_dummy_image() for _ in range(3)]  # 3 张随机图像
    texts = ["a cat sitting on a mat", "a dog playing in the park", "a sunset over the mountains"]  # 3 条文本

    # 调用客户端方法
    cosine_scores = client.cosine(images, texts)

    # 打印结果
    print("Cosine Similarity Scores:")
    for i, score in enumerate(cosine_scores):
        print(f"Image {i + 1} and Text {i + 1}: {score:.4f}")
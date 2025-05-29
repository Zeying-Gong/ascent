import numpy as np
import torch
import torchvision.transforms.functional as F

from vlfm.vlm.server_wrapper import ServerMixin, host_model, send_request, str_to_image

import supervision as sv
from PIL import Image

import argparse
import numpy as np
from typing import List
from vlfm.vlm.server_wrapper import ServerMixin, host_model, send_request_vlm, str_to_image
from vlfm.vlm.recognize_anything.ram.models import ram_plus
from vlfm.vlm.recognize_anything.ram import inference_ram as inference
from vlfm.vlm.recognize_anything.ram import get_transform

RAM_CHECKPOINT_PATH = "data/ram_plus_swin_large_14m.pth" # "./ram_swin_large_14m.pth"

class RAM:
    def __init__(
        self,
        config_path: str = RAM_CHECKPOINT_PATH,
        image_size: str = 384,
        device: torch.device = torch.device("cuda"),
    ):
        '''
        * The Recognize Anything Plus Model (RAM++)
        '''
        self.transform = get_transform(image_size=image_size)

        #######load model
        self.model = ram_plus(pretrained=config_path,
                                image_size=384,
                                vit='swin_l')
        self.model.eval()
        self.device = device
        self.model = self.model.to(self.device)

        # image = transform(Image.open(args.image)).unsqueeze(0).to(device)

        # res = inference(image, model)

    def predict(self, images: List[np.ndarray]) -> List[str]:
        """
        Perform batch inference on a list of images.
        """
        # Convert images to PIL and apply transformation
        batch_images = [Image.fromarray(img.astype('uint8')).convert("RGB") for img in images]
        batch_transformed = torch.stack([self.transform(img).to(self.device) for img in batch_images])
        print(f"Shape of Ram input: {batch_transformed.shape}")
        # Perform batch inference
        with torch.inference_mode():
            results = self.model.generate_tag(batch_transformed) # generate_tag支持并行，inference不支持
            print(f"Result: {results}")
        # Convert results to list of strings
        return [result for result in results[0]]


class RAMClient:
    def __init__(self, port: int = 15185):
        self.url = f"http://localhost:{port}/ram"

    def predict(self, images: List[np.ndarray]) -> str:
        response = send_request_vlm(self.url, images=images)
        return response


if __name__ == "__main__":

    # True Use
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=15185)
    args = parser.parse_args()

    print("Loading model...")

    class RAMServer(ServerMixin, RAM):

        def process_payload(self, payload: dict) -> dict:
            images = [str_to_image(img) for img in payload["images"]]
            response = self.predict(images)
            return {"response": response}
        
    ram = RAMServer()
    print("Model loaded!")
    print(f"Hosting on port {args.port}...")
    host_model(ram, name="ram", port=args.port)


    # import cv2
    # # 初始化 RAM 模型
    # ram_model = RAM(config_path=RAM_CHECKPOINT_PATH, image_size=384, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # # 准备一批测试图像
    # image_paths = [
    #     "debug/20250316/01.png",
    #     "debug/20250316/02.png",
    #     "debug/20250316/03.png"
    # ]

    # test_images = []
    # for path in image_paths:
    #     # 使用 OpenCV 读取图片
    #     image = cv2.imread(path)  # 读取图片
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # OpenCV 默认读取为 BGR，需要转换为 RGB
    #     image = cv2.resize(image, (384, 384))  # 调整图片大小以匹配模型的输入尺寸
    #     image = np.array(image, dtype=np.uint8)  # 确保是 NumPy 数组
    #     test_images.append(image)

    # # 调用 predict 方法进行预测
    # results = ram_model.predict(test_images)

    # # 打印预测结果
    # print("Predicted results:")
    # for i, result in enumerate(results):
    #     print(f"Image {i + 1}: {result}")
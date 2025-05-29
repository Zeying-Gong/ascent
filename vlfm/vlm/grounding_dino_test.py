from typing import Optional, List

import numpy as np
import torch
import torchvision.transforms.functional as F

from vlfm.vlm.detections import ObjectDetections

from vlfm.vlm.server_wrapper import ServerMixin, host_model, send_request_vlm, str_to_image

try:
    from groundingdino.util.inference_batch import load_model, predict ## 
except ModuleNotFoundError:
    print("Could not import groundingdino. This is OK if you are only using the client.")

import groundingdino.datasets.transforms as T
from PIL import Image

GROUNDING_DINO_CONFIG = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_WEIGHTS = "data/groundingdino_swint_ogc.pth"
CLASSES = "chair . person . dog ."  # Default classes. Can be overridden at inference.


class GroundingDINO_MF:
    def __init__(
        self,
        config_path: str = GROUNDING_DINO_CONFIG,
        weights_path: str = GROUNDING_DINO_WEIGHTS,
        caption: str = CLASSES,
        box_threshold: float = 0.15, # 0.35,
        text_threshold: float = 0.10, # 0.25,
        device: torch.device = torch.device("cuda"),
    ):
        self.model = load_model(model_config_path=config_path, model_checkpoint_path=weights_path).to(device)
        self.caption = caption
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.device = device

    def predict(self, images: List[np.ndarray], captions: Optional[List[str]] = None, box_threshold: float = 0.15, text_threshold: float = 0.10) -> ObjectDetections: # 35,25
        """
        This function makes predictions on an input image tensor or numpy array using a
        pretrained model.

        Arguments:
            image (np.ndarray): An image in the form of a numpy array.
            caption (Optional[str]): A string containing the possible classes
                separated by periods. If not provided, the default classes will be used.

        Returns:
            ObjectDetections: An instance of the ObjectDetections class containing the
                object detections.
        """
        # Convert image to tensor and normalize from 0-255 to 0-1
        # image_tensor = F.to_tensor(image)
        # image_transformed = F.normalize(image_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image_sources = [Image.fromarray(image.astype('uint8')).convert("RGB") for image in images]
        image_transformed_batch = torch.stack([transform(image, None)[0] for image in image_sources]).to(self.device)
        
        # 批量推理
        with torch.inference_mode():
            # 直接输入整个批次到模型
            boxes_batch, logits_batch, phrases_batch = predict(
                model=self.model,
                images=image_transformed_batch,  # 直接输入批次
                captions=captions,  # 输入文本描述列表
                box_threshold=box_threshold,
                text_threshold=text_threshold,
            )

        # 解析结果
        detections_list = []
        for i, (boxes, logits, phrases) in enumerate(zip(boxes_batch, logits_batch, phrases_batch)):
            detections = ObjectDetections(boxes, logits, phrases, image_source=image_sources[i])

            # 过滤检测结果
            caption = captions[i]
            if caption.endswith(" ."):
                classes = caption[:-len(" .")].split(" .") # 
            else:  # 可能只有一个类别
                classes = caption
            detections.filter_by_class(classes)

            detections_list.append(detections.to_json())

        return detections_list


class GroundingDINOClient_MF:
    def __init__(self, port: int = 12184):
        self.url = f"http://localhost:{port}/gdino"

    def predict(self, images: List[np.ndarray], captions: Optional[List[str]] = "", box_threshold: float = 0.15, text_threshold: float = 0.10) -> ObjectDetections:
        responses = send_request_vlm(self.url, images=images, captions=captions, box_threshold=box_threshold, text_threshold=text_threshold)
        detections_list = []
        for	idx, response in enumerate(responses):
            detections = ObjectDetections.from_json(response, image_source=images[idx])
            detections_list.append(detections)
        return detections_list


if __name__ == "__main__":
    # Test usage
    # image_paths = [
    #     "debug/20250316/01.png",
    #     "debug/20250316/02.png",
    #     "debug/20250316/03.png"
    # ]

    # # 加载图像
    # images = [np.array(Image.open(path).convert("RGB")) for path in image_paths]

    # # 初始化 GroundingDINO_MF
    # grounding_dino = GroundingDINO_MF()

    # # 批量推理
    # captions = ["a photo of a cat", "a photo of a dog", "a photo of a car"]  # 每张图片的文本描述
    # detections_list = grounding_dino.predict(images, captions=captions)

    # # 打印结果
    # for i, detections in enumerate(detections_list):
    #     print(f"Image {i + 1} detections:")
    #     print(f"  Boxes: {detections.boxes}")
    #     print(f"  Logits: {detections.logits}")
    #     print(f"  Phrases: {detections.phrases}")
    #     print("-" * 50)

    # True Use
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=12184)
    args = parser.parse_args()

    print("Loading model...")

    class GroundingDINOServer_MF(ServerMixin, GroundingDINO_MF):
        def process_payload(self, payload: dict) -> dict:
            images = [str_to_image(img) for img in payload["images"]]
            return self.predict(images=images, captions=payload["captions"]) # .to_json()

    gdino = GroundingDINOServer_MF()
    print("Model loaded!")
    print(f"Hosting on port {args.port}...")
    host_model(gdino, name="gdino", port=args.port)
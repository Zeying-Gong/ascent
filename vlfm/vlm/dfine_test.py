# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import sys
from typing import List, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from vlfm.vlm.coco_classes import COCO_CLASSES
from vlfm.vlm.detections import ObjectDetections
from vlfm.vlm.server_wrapper import ServerMixin, host_model, send_request_vlm, str_to_image
from PIL import Image
sys.path.insert(0, "D-FINE/")
from src.core import YAMLConfig
sys.path.pop(0)


class DFine(nn.Module):
    def __init__(self, config_path: str, checkpoint_path: str, device: str = "cpu"):
        super().__init__()
        self.device = torch.device(device)
        self._load_model(config_path, checkpoint_path)
        self.model.eval()
        # Warmup
        if self.device.type != "cpu":
            dummy_img = torch.rand(1, 3, 640, 640).to(self.device)
            with torch.no_grad():
                self.model(dummy_img)

    def _load_model(self, config_path: str, checkpoint_path: str):
        """Loads and configures the D-FINE model."""
        cfg = YAMLConfig(config_path)
        # Handle special cases in config
        if "HGNetv2" in cfg.yaml_cfg:
            cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state = checkpoint.get("ema", {}).get("module", checkpoint.get("model"))
        
        # Build and convert model to deploy mode
        cfg.model.load_state_dict(state)
        self.model = cfg.model.deploy().to(self.device)
        self.postprocessor = cfg.postprocessor.deploy()

    def predict(
        self,
        images: List[np.ndarray],  # 修改为列表，支持批量输入
        conf_thres: float = 0.4,
    ) -> List[ObjectDetections]:
        """
        Performs object detection on the input images.

        Args:
            images: List of input images in RGB format as numpy arrays
            conf_thres: Confidence threshold for detection filtering
            classes: Optional list of classes to filter by (not implemented)
        """
        # Convert numpy arrays to PIL Images and preprocess
        pil_images = [Image.fromarray(image).convert("RGB") for image in images]
        orig_sizes = torch.tensor([image.shape[:2][::-1] for image in images]).to(self.device)  # (batch_size, 2) (w, h)

        # Preprocess
        transform = T.Compose([
            T.Resize((640, 640)),
            T.ToTensor(),
        ])
        img_tensors = torch.stack([transform(pil_image).to(self.device) for pil_image in pil_images])  # (batch_size, 3, 640, 640)

        # Inference
        with torch.no_grad():
            outputs = self.model(img_tensors)
            batch_labels, batch_boxes, batch_scores = self.postprocessor(outputs, orig_sizes)

        # Process each image's detections
        batch_detections = []
        for i in range(len(images)):
            labels, boxes, scores = batch_labels[i], batch_boxes[i], batch_scores[i]

            # Filter by confidence threshold
            mask = scores > conf_thres
            labels = labels[mask].cpu().numpy()
            boxes = boxes[mask].cpu().numpy()
            scores = scores[mask].cpu().numpy()

            # Convert to normalized coordinates [x1, y1, x2, y2] format
            h, w = images[i].shape[:2]
            normalized_boxes = boxes.copy()
            normalized_boxes[:, [0, 2]] /= w
            normalized_boxes[:, [1, 3]] /= h

            phrases = [COCO_CLASSES[int(idx)] for idx in labels]
            detections = ObjectDetections(
                boxes=normalized_boxes,
                logits=scores,
                phrases=phrases,
                image_source=images[i],
                fmt="xyxy",
            )
            batch_detections.append(detections.to_json())

        return batch_detections


class DFineClient:
    def __init__(self, port: int = 12184):
        self.url = f"http://localhost:{port}/dfine"

    def predict(self, images: List[np.ndarray]) -> ObjectDetections:
        responses = send_request_vlm(self.url, images=images)
        detections_list = []
        for	idx, response in enumerate(responses):
            detections = ObjectDetections.from_json(response, image_source=images[idx])
            detections_list.append(detections)
        return detections_list

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=12184) 
    parser.add_argument("--config", type=str, default="D-FINE/configs/dfine/objects365/dfine_hgnetv2_x_obj2coco.yml")
    parser.add_argument("--checkpoint", type=str, default="D-FINE/models/dfine_x_obj2coco.pth")
    args = parser.parse_args()

    print("Loading model...")

    class DFineServer(ServerMixin, DFine):
        def __init__(self, config: str, checkpoint: str):
            super().__init__(config, checkpoint, device="cuda" if torch.cuda.is_available() else "cpu")
        def process_payload(self, payload: dict) -> dict:
            images = [str_to_image(img) for img in payload["images"]]
            return self.predict(images) # .to_json()

    dfine = DFineServer(args.config, args.checkpoint)
    print("Model loaded!")
    print(f"Hosting on port {args.port}...")
    host_model(dfine, name="dfine", port=args.port)

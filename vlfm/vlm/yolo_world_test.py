
import os.path as osp

import torch
from mmengine.config import Config
from mmengine.dataset import Compose
from mmdet.apis import init_detector
from mmdet.utils import get_test_pipeline_cfg

from typing import Optional, List

import numpy as np
# import torchvision.transforms.functional as F

from vlfm.vlm.detections import ObjectDetections
from vlfm.vlm.server_wrapper import ServerMixin, host_model, send_request_vlm, str_to_image

from PIL import Image

import os

# 假设您设置了一个环境变量 YOLO_WORLD_DIR
YOLO_WORLD_DIR = os.getenv("YOLO_WORLD_DIR", "/home/zeyingg/github/habitat-lab-vlfm/YOLO-World")

YOLO_WORLD_CONFIG = os.path.join(YOLO_WORLD_DIR,    "configs/pretrain/yolo_world_v2_l_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py") # "configs/pretrain/yolo_world_v2_x_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_1280ft_lvis_minival.py") #
YOLO_WORLD_WEIGHTS = os.path.join(YOLO_WORLD_DIR,   "weights/yolo_world_v2_l_obj365v1_goldg_cc3mv2_pretrain-2f3a4a22.pth") # "weights/yolo_world_v2_x_obj365v1_goldg_cc3mlite_pretrain_1280ft-14996a36.pth") # 

CLASSES = [["chair", " "]]  # Default classes. Can be overridden at inference.

class YoloWorld_MF:
    def __init__(
        self,
        config_path: str = YOLO_WORLD_CONFIG,
        checkpoint_path: str = YOLO_WORLD_WEIGHTS,
        work_dir: str = './work_dirs',
        device: str = 'cuda:0',
        score_thr: float = 0.05,
        max_dets: int = 100,
    ):
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.work_dir = work_dir
        self.device = device
        self.score_thr = score_thr
        self.max_dets = max_dets
        self.caption = CLASSES
        # Load configuration
        cfg = Config.fromfile(self.config_path)
        cfg.work_dir = osp.join(self.work_dir)
        cfg.load_from = self.checkpoint_path

        # Initialize the model
        self.model = init_detector(cfg, checkpoint=self.checkpoint_path, device=self.device)

        # Prepare test pipeline
        test_pipeline_cfg = get_test_pipeline_cfg(cfg=cfg)
        test_pipeline_cfg[0].type = 'mmdet.LoadImageFromNDArray'
        self.test_pipeline = Compose(test_pipeline_cfg)

    # def preprocess(self, image: np.ndarray, size: tuple = (640, 640)):
    #     h, w = image.shape[:2]
    #     max_size = max(h, w)
    #     scale_factor = size[0] / max_size
    #     pad_h = (max_size - h) // 2
    #     pad_w = (max_size - w) // 2
    #     pad_image = np.zeros((max_size, max_size, 3), dtype=image.dtype)
    #     pad_image[pad_h:h + pad_h, pad_w:w + pad_w] = image
    #     image = cv2.resize(pad_image, size,
    #                     interpolation=cv2.INTER_LINEAR).astype('float32')
    #     image /= 255.0
    #     image = image[None]
    #     return image #, scale_factor, (pad_h, pad_w)
    
    def predict(
        self,
        images: List[np.ndarray],  # 改为支持批量输入
        captions: Optional[List[List[List[str]]]] = None,  # 改为支持批量描述
        score_thr: float = 0.05,
    ) -> List[ObjectDetections]:  # 返回检测结果列表
        """
        Perform object detection on a batch of input images.

        Args:
            images (List[np.ndarray]): List of input images as numpy arrays (H x W x C) in BGR format.
            captions (Optional[List[List[str]]]): List of texts corresponding to classes for each image.

        Returns:
            List[ObjectDetections]: List of detection results for each image.
        """
        if captions is None:
            captions_to_use = [self.caption] * len(images)  # 如果没有提供描述，使用默认描述
        else:
            captions_to_use = captions

        # 批量预处理
        data_batch = []
        for i, (image, caption) in enumerate(zip(images, captions_to_use)):
            data_info = dict(img=image, img_id=i, texts=caption)
            data_info = self.test_pipeline(data_info)  # 预处理
            data_batch.append(data_info)

        # 将数据堆叠成一个批次
        inputs = torch.stack([data['inputs'] for data in data_batch])  # 堆叠图像张量
        data_samples = [data['data_samples'] for data in data_batch]  # 数据样本列表

        # 构建批次数据
        batch_data = dict(inputs=inputs, data_samples=data_samples)

        # 批量推理
        with torch.no_grad():
            outputs = self.model.test_step(batch_data)  # 批量推理

        # 批量后处理
        detections_list = []
        for i, output in enumerate(outputs):
            pred_instances = output.pred_instances

            # 应用分数阈值
            mask = pred_instances.scores.float() > score_thr
            pred_instances = pred_instances[mask]

            # 应用最大检测数限制
            if len(pred_instances.scores) > self.max_dets:
                indices = pred_instances.scores.float().topk(self.max_dets)[1]
                pred_instances = pred_instances[indices]

            # 转换为 numpy 数组
            pred_instances = pred_instances.cpu().numpy()
            boxes = pred_instances['bboxes']
            labels = pred_instances['labels']
            logits = pred_instances['scores']
            phrases = [captions_to_use[i][x][0] for x in labels]  # 获取对应的类别描述

            # 创建 ObjectDetections 对象
            detections = ObjectDetections(boxes, logits, phrases, image_source=images[i], fmt="xyxy")
            detections_list.append(detections.to_json())

        return detections_list

class YoloWorldClient_MF:
    def __init__(self, port: int = 13184):
        self.url = f"http://localhost:{port}/yolo_world"

    def predict(self, images: List[np.ndarray], captions: Optional[List[List[str]]] = [[[" "]]], score_thr: float = 0.05) -> ObjectDetections:
        responses = send_request_vlm(self.url, images=images, captions=captions, score_thr=score_thr)
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
    # yolo_world = YoloWorld_MF()

    # # 批量推理
    # captions = [[["table"],[" "]], [["couch"],[" "]], [["bed"],[" "]]]  # 每张图片的文本描述
    # detections_list = yolo_world.predict(images, captions=captions)

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
    parser.add_argument("--port", type=int, default=13184)
    args = parser.parse_args()

    print("Loading model...")

    class YoloWorldServer(ServerMixin, YoloWorld_MF):
        def process_payload(self, payload: dict) -> dict:
            images = [str_to_image(img) for img in payload["images"]]
            return self.predict(images=images, captions=payload["captions"])

    yolo_world = YoloWorldServer()
    print("Model loaded!")
    print(f"Hosting on port {args.port}...")
    host_model(yolo_world, name="yolo_world", port=args.port)
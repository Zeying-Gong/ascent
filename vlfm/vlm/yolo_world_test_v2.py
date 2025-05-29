
import os.path as osp
import cv2
import torch
from mmengine.config import Config
from mmengine.dataset import Compose
from mmdet.apis import init_detector
from mmdet.utils import get_test_pipeline_cfg

from typing import Optional, List

import numpy as np
import torchvision.transforms.functional as F

from vlfm.vlm.detections import ObjectDetections
from vlfm.vlm.server_wrapper import ServerMixin, host_model, send_request, str_to_image

from PIL import Image

import os

# 假设您设置了一个环境变量 YOLO_WORLD_DIR
YOLO_WORLD_DIR = os.getenv("YOLO_WORLD_DIR", "/home/zeyingg/github/habitat-lab-vlfm/YOLO-World")

YOLO_WORLD_CONFIG = os.path.join(YOLO_WORLD_DIR,  "configs/pretrain/yolo_world_v2_l_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py") # "configs/pretrain/yolo_world_v2_x_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_1280ft_lvis_minival.py") #
YOLO_WORLD_WEIGHTS = os.path.join(YOLO_WORLD_DIR,   "weights/yolo_world_v2_l_obj365v1_goldg_cc3mv2_pretrain-2f3a4a22.pth") # "weights/yolo_world_v2_x_obj365v1_goldg_cc3mlite_pretrain_1280ft-14996a36.pth") #

CLASSES = ["chair"]  # Default classes. Can be overridden at inference.

def inference(model, image_path, texts, test_pipeline, score_thr=0.3, max_dets=100):
    # 读取原始图像
    image = cv2.imread(image_path)
    original_image = image.copy()  # 保留原始图像用于绘制
    image = image[:, :, [2, 1, 0]]  # BGR转RGB
    data_info = dict(img=image, img_id=0, texts=texts)
    data_info = test_pipeline(data_info)
    data_batch = dict(inputs=data_info['inputs'].unsqueeze(0),
                      data_samples=[data_info['data_samples']])
    with torch.no_grad():
        output = model.test_step(data_batch)[0]
    pred_instances = output.pred_instances
    # 置信度阈值筛选
    pred_instances = pred_instances[pred_instances.scores.float() > score_thr]
    # 最大检测数限制
    if len(pred_instances.scores) > max_dets:
        indices = pred_instances.scores.float().topk(max_dets)[1]
        pred_instances = pred_instances[indices]

    pred_instances = pred_instances.cpu().numpy()
    boxes = pred_instances['bboxes']
    labels = pred_instances['labels']
    scores = pred_instances['scores']
    label_texts = [texts[x][0] for x in labels]
    return boxes, labels, label_texts, scores, original_image

def draw_and_save(image, boxes, labels, label_texts, scores, save_path):
    for box, label, label_text, score in zip(boxes, labels, label_texts, scores):
        x1, y1, x2, y2 = box.astype(int)
        # 绘制边界框
        cv2.rectangle(image, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
        # 构建标签文本
        text = f"{label_text}: {score:.2f}"
        # 获取文本尺寸
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        # 绘制文本背景
        cv2.rectangle(image, (x1, y1 - text_height - baseline), (x1 + text_width, y1), (0, 255, 0), -1)
        # 绘制文本
        cv2.putText(image, text, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    # 保存图像
    cv2.imwrite(save_path, image)
    print(f"Processed image saved to: {save_path}")

if __name__ == "__main__":
    config_file = YOLO_WORLD_CONFIG
    checkpoint = YOLO_WORLD_WEIGHTS

    cfg = Config.fromfile(config_file)
    cfg.work_dir = osp.join('./work_dirs')
    # 初始化模型
    cfg.load_from = checkpoint
    model = init_detector(cfg, checkpoint=checkpoint, device='cuda:0')
    test_pipeline_cfg = get_test_pipeline_cfg(cfg=cfg)
    test_pipeline_cfg[0].type = 'mmdet.LoadImageFromNDArray'
    test_pipeline = Compose(test_pipeline_cfg)

    texts = [['stair'], ['person'], ['tv'], [" "]]
    image_path = "debug/20241219/stair.png" # "YOLO-World/demo/sample_images/bus.jpg"
    print(f"Starting to detect: {image_path}")
    results = inference(model, image_path, texts, test_pipeline, score_thr=0.1)
    boxes, labels, label_texts, scores, original_image = results

    format_str = [
        f"obj-{idx}: {box}, label-{lbl}, class-{lbl_text}, score-{score}"
        for idx, (box, lbl, lbl_text, score) in enumerate(zip(boxes, labels, label_texts, scores))
    ]
    print("Detecting results:")
    for q in format_str:
        print(q)
    
    # 指定保存路径
    save_path = "debug/20241219/l_v3_stair_processed.jpg" # "/home/zeyingg/github/habitat-lab-vlfm/debug/20241219/bus_detected_v2.jpg"
    # 绘制检测结果并保存图像
    draw_and_save(original_image, boxes, labels, label_texts, scores, save_path)

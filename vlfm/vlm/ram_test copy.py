from typing import Optional

import numpy as np
import torch
import torchvision.transforms.functional as F

from vlfm.vlm.detections import ObjectDetections

from vlfm.vlm.server_wrapper import ServerMixin, host_model, send_request, str_to_image

try:
    from groundingdino.util.inference import load_model, predict
except ModuleNotFoundError:
    print("Could not import groundingdino. This is OK if you are only using the client.")

import groundingdino.datasets.transforms as T

import supervision as sv
from PIL import Image

import cv2
import argparse
import numpy as np
import random

from ram.models import ram_plus
from ram import inference_ram as inference
from ram import get_transform

GROUNDING_DINO_CONFIG = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_WEIGHTS = "data/groundingdino_swint_ogc.pth"
CLASSES = "chair . person . dog ."  # Default classes. Can be overridden at inference.

RAM_CHECKPOINT_PATH = "./ram_plus_swin_large_14m.pth" # "./ram_swin_large_14m.pth"

BOX_THRESHOLD = 0.2
TEXT_THRESHOLD = 0.2
IOU_THRESHOLD = 0.5

class GroundingDINO_MF:
    def __init__(
        self,
        config_path: str = GROUNDING_DINO_CONFIG,
        weights_path: str = GROUNDING_DINO_WEIGHTS,
        caption: str = CLASSES,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
        device: torch.device = torch.device("cuda"),
    ):
        self.model = load_model(model_config_path=config_path, model_checkpoint_path=weights_path).to(device)
        self.caption = caption
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

    def predict(self, image: np.ndarray, caption: Optional[str] = None, box_threshold: float = 0.35, text_threshold: float = 0.25) -> ObjectDetections:
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
        image_source = Image.fromarray(image.astype('uint8')).convert("RGB")
        image_transformed, _ = transform(image_source, None)
        
        if caption is None:
            caption_to_use = self.caption
        else:
            caption_to_use = caption
        print("Caption:", caption_to_use)
        with torch.inference_mode():
            boxes, logits, phrases = predict(
                model=self.model,
                image=image_transformed,
                caption=caption_to_use,
                box_threshold=box_threshold, # self.box_threshold,
                text_threshold=text_threshold, # self.text_threshold,
            )
        detections = ObjectDetections(boxes, logits, phrases, image_source=image_source)

        # Remove detections whose class names do not exactly match the provided classes
        # classes = caption_to_use[: -len(" .")].split(" . ")

        if caption_to_use.endswith(" ."):
            classes = caption_to_use[:-len(" .")].split(" . ")
        else: # maybe just one category
            classes = caption_to_use
        detections.filter_by_class(classes)

        return detections


class GroundingDINOClient_MF:
    def __init__(self, port: int = 12181):
        self.url = f"http://localhost:{port}/gdino"

    def predict(self, image_numpy: np.ndarray, caption: Optional[str] = "", box_threshold: float = 0.35, text_threshold: float = 0.25) -> ObjectDetections:
        response = send_request(self.url, image=image_numpy, caption=caption, box_threshold=box_threshold, text_threshold=text_threshold)
        detections = ObjectDetections.from_json(response, image_source=image_numpy)

        return detections


if __name__ == "__main__":
    # Test usage
    # import cv2
    # from groundingdino.util.inference import load_model, load_image, predict, annotate
    # image_path = "debug/20241204/stair_dino_test/stair.png"
    # caption = "stair"
    # image_source, image = load_image(image_path)
    # model = load_model(GROUNDING_DINO_CONFIG, GROUNDING_DINO_WEIGHTS)
    # boxes, logits, phrases = predict(
    #     model=model, 
    #     image=image, 
    #     caption=caption, 
    #     box_threshold=0.35, 
    #     text_threshold=0.25
    # )
    # annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
    # # 显示结果图像
    # result_image_path = "debug/20241204/stair_dino_test/detected_stair_v3.png"
    # cv2.imwrite(result_image_path, annotated_frame)  # 保存带检测框的图像
    # print(f"Detection results saved to {result_image_path}.")

    # Test usage v2
    # import cv2
    # from groundingdino.util.inference import load_model, load_image, predict, annotate
    # loaded_img = np.load("image.npy")
    # caption = "stair"
    # gdino = GroundingDINO_MF()
    # detections = gdino.predict(image=loaded_img, caption="stair")
    # annotated_frame = annotate(image_source=loaded_img, boxes=detections.boxes, logits=detections.logits, phrases=detections.phrases)
    # # 显示结果图像
    # result_image_path = "debug/20241204/stair_dino_test/detected_stair_v3.png"
    # cv2.imwrite(result_image_path, annotated_frame)  # 保存带检测框的图像
    # print(f"Detection results saved to {result_image_path}.")

    # True Use
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=12181)
    args = parser.parse_args()

    print("Loading model...")

    class GroundingDINOServer(ServerMixin, GroundingDINO_MF):
        def process_payload(self, payload: dict) -> dict:
            image = str_to_image(payload["image"])
            return self.predict(image, caption=payload["caption"]).to_json()

    gdino = GroundingDINOServer()
    print("Model loaded!")
    print(f"Hosting on port {args.port}...")
    host_model(gdino, name="gdino", port=args.port)

'''
 * The Recognize Anything Plus Model (RAM++)
'''



parser = argparse.ArgumentParser(
    description='Tag2Text inferece for tagging and captioning')
parser.add_argument('--image',
                    metavar='DIR',
                    help='path to dataset',
                    default='images/demo/demo1.jpg')
parser.add_argument('--pretrained',
                    metavar='DIR',
                    help='path to pretrained model',
                    default='pretrained/ram_plus_swin_large_14m.pth')
parser.add_argument('--image-size',
                    default=384,
                    type=int,
                    metavar='N',
                    help='input image size (default: 448)')


if __name__ == "__main__":

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = get_transform(image_size=args.image_size)

    #######load model
    model = ram_plus(pretrained=args.pretrained,
                             image_size=args.image_size,
                             vit='swin_l')
    model.eval()

    model = model.to(device)

    image = transform(Image.open(args.image)).unsqueeze(0).to(device)

    res = inference(image, model)
    print("Image Tags: ", res[0])
    print("图像标签: ", res[1])

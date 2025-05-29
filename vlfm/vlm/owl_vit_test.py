from typing import Optional
import numpy as np
import torch
from PIL import Image
from vlfm.vlm.server_wrapper import ServerMixin, host_model, send_request, str_to_image
from transformers import Owlv2Processor, Owlv2ForObjectDetection
import torchvision.transforms.functional as F
import cv2
from vlfm.vlm.detections import ObjectDetections
class OwlV2_MF:
    def __init__(
        self,
        model_name: str = "google/owlv2-base-patch16-ensemble",
        caption: str = "tv",  # Default class for object detection, can be overridden at inference
        box_threshold: float = 0.9,
        # nms_threshold: float = 0.3,
        device: torch.device = torch.device("cuda"),
    ):
        # 加载模型和处理器
        self.processor = Owlv2Processor.from_pretrained(model_name)
        self.detector = Owlv2ForObjectDetection.from_pretrained(model_name)
        self.caption = caption
        self.box_threshold = box_threshold
        # self.nms_threshold = nms_threshold
        self.device = device
        self.detector.to(device)
        
    def predict(self, image: np.ndarray, caption: Optional[str] = None, box_threshold: float = 0.9):
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
        # 转换图像为 PIL 格式
        image_pil = Image.fromarray(image.astype('uint8')).convert("RGB")
        
        # 处理输入图像和文本
        if caption is None:
            caption = self.caption
        if caption is None:
            caption_to_use = self.caption
        else:
            caption_to_use = caption
        # inputs = self.processor(text=caption_to_use, images=image_pil, return_tensors="pt")
        # outputs = self.detector(**inputs)
        inputs = self.processor(text=caption_to_use, images=image_pil, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}  # 确保 inputs 在正确的设备上
        outputs = self.detector(**inputs)

        # 获取目标尺寸
        target_sizes = torch.Tensor([image_pil.size[::-1]]).to(self.device)  # PIL Image size 是 (宽, 高)，需要反转为 (高, 宽)
        
        # 处理检测结果
        predictions = self.processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=box_threshold) # ,nms_threshold=self.nms_threshold
        labels_tensor = predictions[0]["labels"]
        labels_cpu = labels_tensor.cpu().numpy()
        # 提取预测结果
        # boxes = predictions[0]["boxes"]
        # scores = predictions[0]["scores"]
        # labels = predictions[0]["labels"]
        detections = ObjectDetections(predictions[0]["boxes"], predictions[0]["scores"], [caption[idx] for idx in labels_cpu], image_source=image_pil)
        # if caption_to_use.endswith(" ."):
        #     classes = caption_to_use[:-len(" .")].split(" . ")
        # else: # maybe just one category
        #     classes = caption_to_use
        # detections.filter_by_class_id(classes)

        return detections

class OwlV2Client_MF:
    def __init__(self, port: int = 13186):
        self.url = f"http://localhost:{port}/owlv2"

    def predict(self, image: np.ndarray, caption: Optional[str] = "", box_threshold: float = 0.9):
        response = send_request(self.url, image=image, caption=caption, box_threshold=box_threshold)
        detections = ObjectDetections.from_json(response, image_source=image)
        return detections

if __name__ == "__main__":
    # 启动服务
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=13186)
    args = parser.parse_args()

    print("Loading model...")

    class OwlV2Server(ServerMixin, OwlV2_MF):
        def process_payload(self, payload: dict) -> dict:
            image = str_to_image(payload["image"])
            # boxes, scores, labels = self.predict(image, caption=payload["caption"])
            # 这里返回预测结果
            return self.predict(image, caption=payload["caption"]).to_json()

    owl_v2 = OwlV2Server()
    print("Model loaded!")
    print(f"Hosting on port {args.port}...")
    host_model(owl_v2, name="owlv2", port=args.port)

# if __name__ == "__main__":
#     import argparse
#     import sys

#     parser = argparse.ArgumentParser(description="调试OwlV2_MF模型")
#     parser.add_argument("--image_path", type=str, default="/home/zeyingg/github/habitat-lab-vlfm/debug/20250116/chair_sofa.png", help="待检测的图像路径")
#     # parser.add_argument("--caption", type=str, default="chair", help="检测类别，多个类别用 ' . ' 分隔")
#     parser.add_argument("--box_threshold", type=float, default=0.1, help="置信度阈值")
#     parser.add_argument("--port", type=int, default=13186, help="服务器端口（如果需要启动服务）")
#     # parser.add_argument("--debug_server", action='store_true', help="是否启动服务器模式")
#     args = parser.parse_args()

    # if args.debug_server:
    #     # 启动服务模式
    #     print("Loading model...")
        
    #     class OwlV2Server(ServerMixin, OwlV2_MF):
    #         def process_payload(self, payload: dict) -> dict:
    #             image = str_to_image(payload["image"])
    #             return self.predict(image, caption=payload.get("caption", self.caption), box_threshold=payload.get("box_threshold", self.box_threshold)).to_json()

    #     owl_v2 = OwlV2Server()
    #     print("Model loaded!")
    #     print(f"Hosting on port {args.port}...")
    #     host_model(owl_v2, name="owlv2", port=args.port)
    # else:
        # 调试模式
    # caption = ["chair", "sofa", "stair"]
    # print("初始化OwlV2_MF模型...")
    # model = OwlV2_MF(caption=caption, box_threshold=args.box_threshold)
    # print("模型加载完成。")

    # print(f"加载图像: {args.image_path}")
    # try:
    #     image = Image.open(args.image_path).convert("RGB")
    #     image_np = np.array(image)
    # except Exception as e:
    #     print(f"无法加载图像: {e}")
    #     sys.exit(1)

    # print("进行预测...")
    # detections = model.predict(image_np, caption=caption, box_threshold=args.box_threshold)
    # print("预测完成。")

    # # 输出检测结果
    # print("检测结果:")
    # for idx, (box, score, label) in enumerate(zip(detections.boxes, detections.logits, detections.phrases)):
    #     print(f"对象 {idx+1}:")
    #     print(f"  边界框: {box.tolist()}")
    #     print(f"  置信度: {score:.4f}")
    #     print(f"  类别: {label}")

    # # 遍历所有检测框，绘制框和标签
    # for box, score, label in zip(detections.boxes, detections.logits, detections.phrases):
    #     x_min, y_min, x_max, y_max = map(int, box)  # 转换为整数
    #     # 绘制边界框
    #     cv2.rectangle(image_np, (x_min, y_min), (x_max, y_max), color=(0, 255, 0), thickness=2)
    #     # 绘制标签和分数
    #     cv2.putText(image_np, f"{label}: {score:.2f}", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # # 将 numpy 数组转换为 PIL 图像
    # result_image = Image.fromarray(image_np) # Image.fromarray(image_np)

    # # 保存图像到指定文件路径
    # # save_path = input("请输入保存路径和文件名（例如: ./output_result.jpg）: ").strip()
    # save_path = "/home/zeyingg/github/habitat-lab-vlfm/debug/20250116/chair_sofa_result.png"
    # result_image.save(save_path)  # 使用 PIL 保存图像

    # print(f"可视化结果已保存至 {save_path}")
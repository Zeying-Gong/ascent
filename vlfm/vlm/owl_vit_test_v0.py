from transformers import Owlv2Processor, Owlv2ForObjectDetection # OwlViTProcessor, OwlViTForObjectDetection
from PIL import Image
import cv2
import numpy as np
import torch

# 加载处理器和模型
# processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
# detector = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")  # 这一步可能需要一些时间来下载和加载模型

processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
detector = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")

# 读取并转换图像
path = "/home/zeyingg/github/habitat-lab-vlfm/debug/20241218/no_tv.png"
image = Image.open(path)
image = image.convert("RGB")  # 确保图像为 RGB 模式

# 定义类别
classes = ["tv"]

# 处理输入
inputs = processor(text=classes, images=image, return_tensors="pt")
outputs = detector(**inputs)

# 获取图像的目标尺寸
target_sizes = torch.Tensor([image.size[::-1]])  # PIL Image size 是 (宽, 高)，需要反转为 (高, 宽)

# 处理检测结果
predictions = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.1)

# 将 PIL Image 转换为 NumPy 数组
image_np = np.array(image)

# 将 RGB 转换为 BGR，因为 OpenCV 使用 BGR
image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

# 提取预测结果
boxes = predictions[0]["boxes"]
scores = predictions[0]["scores"]
labels = predictions[0]["labels"]

# 绘制边界框和标签
for box, score, label in zip(boxes, scores, labels):
    # 绘制矩形框
    cv2.rectangle(
        image_np,
        (int(box[0]), int(box[1])),
        (int(box[2]), int(box[3])),
        (255, 0, 0),  # 红色框
        2  # 框的厚度
    )
    
    # 准备标签文本，包含类别名称和置信度分数
    label_text = f"{classes[label]}: {score:.2f}"
    
    # 获取文本大小以便在绘制时调整背景
    (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    
    # 绘制标签背景矩形（黑色背景）
    cv2.rectangle(
        image_np,
        (int(box[0]), int(box[1] - text_height - baseline)),
        (int(box[0] + text_width), int(box[1])),
        (0, 0, 0),  # 黑色背景
        thickness=cv2.FILLED
    )
    
    # 绘制标签文本（白色文字）
    cv2.putText(
        image_np,
        label_text,
        (int(box[0]), int(box[1] - baseline)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),  # 白色文字
        1
    )

# 定义保存路径
output_path = "/home/zeyingg/github/habitat-lab-vlfm/debug/20241218/stair_processed.png"

# 保存处理后的图像
cv2.imwrite(output_path, image_np)

print(f"Processed image saved to {output_path}")

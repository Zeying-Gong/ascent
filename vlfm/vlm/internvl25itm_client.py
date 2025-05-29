# import argparse
# import numpy as np
# from PIL import Image
# from vlfm.vlm.internvl25itm import INTERNVL2_5ITMClient  # 假设这是正确的导入路径

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--port", type=int, default=14181)
#     args = parser.parse_args()

#     # 设置图像路径和问题
#     image_path = "/home/zeyingg/github/habitat-lab-vlfm/debug/20250116/chair_sofa.png"
#     question = "<image>\nPlease describe the image shortly."
    
#     # 加载图像
#     image = Image.open(image_path)
#     image = np.array(image)  # 将图像转换为 numpy 数组
    
#     # 创建 INTERNVL2_5ITMClient 实例并进行聊天
#     client = INTERNVL2_5ITMClient(port=args.port)
    
#     try:
#         # 调用模型的 chat 方法并打印响应
#         response = client.chat(image, question)
#         print("Model response:", response)
#     except Exception as e:
#         print(f"Error occurred: {e}")

# if __name__ == "__main__":
#     main()

# export CUDA_VISIBLE_DEVICES=0,1,3

import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
from vlfm.vlm.server_wrapper import ServerMixin, host_model, send_request, str_to_image, send_request_vlm
import math
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def load_image(image_file, input_size=448, max_num=12):
        image = Image.open(image_file).convert('RGB')
        transform = build_transform(input_size=input_size)
        images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values

def load_image_array(image_array, input_size=448, max_num=12, device='cuda:0'):
    """
    Load and preprocess an image from a numpy.array instead of a file.
    
    Args:
        image_array (numpy.array): Input image in numpy.array format.
        input_size (int): Desired input size for the image.
        max_num (int): Maximum number of images to process.
        device (str): The device where the tensor should be loaded (e.g., 'cuda:0', 'cpu').
    """
    # Ensure the input is a numpy array
    if not isinstance(image_array, np.ndarray):
        raise ValueError("Input must be a numpy.array.")
    
    # Convert numpy array to PIL Image
    image = Image.fromarray(image_array).convert('RGB')
    
    # Build transformation pipeline
    transform = build_transform(input_size=input_size)
    
    # Dynamic preprocessing (assuming this function exists and works with PIL images)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    
    # Apply transformation to each image
    pixel_values = [transform(image) for image in images]
    
    # Stack into a tensor and move to the specified device
    pixel_values = torch.stack(pixel_values).to(device)
    
    # Convert to bfloat16 for consistency with the model's precision
    pixel_values = pixel_values.to(torch.bfloat16)
    
    return pixel_values

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    # print(f"image_size: {image_size}, type: {type(image_size)}")  # 打印调试信息
    for ratio in target_ratios:
        # print(f"ratio: {ratio}, type: {type(ratio)}")  # 打印调试信息
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * float(ratio[0]) * float(ratio[1]):
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def split_model(model_name):
    device_map = {}
    world_size = torch.cuda.device_count()
    num_layers = {
        'InternVL2_5-1B': 24, 'InternVL2_5-2B': 24, 'InternVL2_5-4B': 36, 'InternVL2_5-8B': 32,
        'InternVL2_5-26B': 48, 'InternVL2_5-38B': 64, 'InternVL2_5-78B': 80}[model_name]
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

class INTERNVL2_5ITM:
    """InternVL2.5 Image-Text Matching model using transformers."""

    def __init__(self, model_name: str = "/home/zeyingg/github/habitat-lab-vlfm/InternVL/InternVL2_5-2B", device: str = None) -> None: # OpenGVLab/InternVL2_5-2B
        """
        初始化INTERNVL2.5模型。

        :param model_name: 模型名称或路径
        :param device: 设备（cuda 或 cpu）
        """
        # 设置设备（默认使用 CUDA 或 CPU）
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        
        # 加载模型和分词器
        print(f"Loading model {model_name}...")
        # device_map = split_model('InternVL2_5-2B')# -MPO
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,  # 使用 bfloat16 精度
            low_cpu_mem_usage=True,      # 减少 CPU 内存使用
            use_flash_attn=True,         # 启用 Flash Attention
            trust_remote_code=True,      # 信任远程代码
            # device_map=device_map
        ).eval().to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
        # print("Model and tokenizer loaded!")

        self.generation_config = dict(max_new_tokens=1024, do_sample=True)

    def chat(self, image_list: list, txt: str) -> str:
        device = next(self.model.parameters()).device
        pixel_value_list = []
        for image_array in image_list:
            pixel_value = load_image_array(image_array=image_array, device=device)
            pixel_value_list.append(pixel_value)
        pixel_values_tensor = torch.cat(pixel_value_list, dim=0)
        # Generate the response using the model
        response = self.model.chat(self.tokenizer, pixel_values_tensor, txt, self.generation_config,
                               history=None, return_history=False)
        return response

if __name__ == "__main__":
    import os
    import cv2
    # 初始化模型
    model = INTERNVL2_5ITM()

    # 读取npy文件
    npy_file_path = 'debug/20250124/test_vlm/frontier_rgb_list.npy'
    try:
        frontier_rgb_array = np.load(npy_file_path)  # 加载npy文件
        print(f"Loaded {len(frontier_rgb_array)} images from {npy_file_path}")
    except Exception as e:
        print(f"Failed to load npy file: {e}")

    # 将npy文件中的数据转换为图像数组列表
    # 假设 frontier_rgb_array 是一个形状为 (N, H, W, C) 的 NumPy 数组
    image_array_list = [frontier_rgb_array[i] for i in range(len(frontier_rgb_array))]

    # 调用chat函数
    txt = "Describe the scene in the images."  # 示例文本
    try:
        response = model.chat(image_array_list[:2], txt)
        print("Response from model:", response)
    except Exception as e:
        print(f"Failed to get response from model: {e}")
    # 输出目录
    # save_dir = 'debug/20250125/test_vlm/v1'
    # os.makedirs(save_dir, exist_ok=True)

    # # 遍历并保存图片
    # for i, img in enumerate(frontier_rgb_array):

    #     filename = f"image_{i + 1}.jpg"
    #     save_path = os.path.join(save_dir, filename)
    #     cv2.imwrite(save_path, img)  # 使用cv2.imwrite保存为jpg
    #     print(f"Saved: {save_path}")

    # instance = INTERNVL2_5ITM()
    # image_path_1 = "/home/zeyingg/github/habitat-lab-vlfm/debug/20250125/test_vlm/v1/image_1.jpg"
    # image_path_2 = "/home/zeyingg/github/habitat-lab-vlfm/debug/20250125/test_vlm/v1/image_2.jpg"
    # # 加载图像
    # # set the max number of tiles in `max_num`
    # pixel_values_1 = load_image(image_path_1, max_num=12).to(torch.bfloat16).cuda()
    # pixel_values_2 = load_image(image_path_2, max_num=12).to(torch.bfloat16).cuda()
    # pixel_values = torch.cat((pixel_values_1, pixel_values_2), dim=0)
    # generation_config = dict(max_new_tokens=1024, do_sample=True)

    # prompt = (
    #     f"As an AI assistant with advanced spatial reasoning capabilities, your task is to navigate an indoor environment in search of a target object. "
    #     f"You will be given a set of frontier images, each corresponding to different areas and directions in the environment. "
    #     f"Each frontier image is labeled with a number indicating the direction, and a red line in the image shows the robot's potential movement direction. "
    #     f"Your goal is to analyze these frontier images and determine which area is most likely to contain the target object. "
    #     f"Target Object Category: \"toilet\" "
    #     f"Frontier Images: <image> "
    #     f"Your output should be in the following format: "
    #     f"[Frontier Identifier]: The most promising frontier to explore (a number between 1 and the number of frontiers). "
    #     f"Reason: A concise explanation (20 words or less) based on the images and target object category. "
    #     f"Ensure that your rationale is concise and directly tied to the visual cues provided by the frontier images, including the direction of movement. "
    #     f"Output only one frontier as the most promising to explore. Do not provide multiple answers. "
    #     f"Example Input: the number of frontiers: 3. "
    #     f"Example Output: 1. Reason: Based on the target object category and the images, Frontier 1 is the most promising to explore first. "
    #     f"Another Example Input: the number of frontiers: 2. "
    #     f"Example Output: 2. Reason: Frontier 2 appears to have a bathroom-like setup, making it the most likely location for a toilet. "
    #     f"True Input: the number of frontiers: 2. "
    #     f"True Output: "
    # )

    # response = instance.model.chat(instance.tokenizer, pixel_values, prompt, generation_config)
    # print(f'User: {prompt}\nAssistant: {response}')

    
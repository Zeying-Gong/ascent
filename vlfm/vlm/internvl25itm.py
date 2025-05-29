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
import random
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def set_seed(seed: int):
    """
    固定随机种子，确保结果可复现。
    
    :param seed: 随机种子值
    """
    torch.manual_seed(seed)       # 设置 PyTorch 的随机种子
    torch.cuda.manual_seed(seed)  # 设置 CUDA 的随机种子
    torch.cuda.manual_seed_all(seed)  # 如果使用多 GPU，设置所有 GPU 的随机种子
    random.seed(seed)             # 设置 Python 随机模块的种子
    np.random.seed(seed)          # 设置 NumPy 的随机种子
    torch.backends.cudnn.deterministic = True  # 确保 CUDA 卷积操作是确定性的
    torch.backends.cudnn.benchmark = False     # 关闭 CUDA 的自动优化

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

    def __init__(self, model_name: str = "/home/zeyingg/github/habitat-lab-vlfm/InternVL/InternVL2_5-8B-MPO", device: str = None, seed: int = 2025) -> None: # OpenGVLab/InternVL2_5-2B 78B-MPO
        """
        初始化INTERNVL2.5模型。

        :param model_name: 模型名称或路径
        :param device: 设备（cuda 或 cpu）
        """
        set_seed(seed)

        # 设置设备（默认使用 CUDA 或 CPU）
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        
        # 加载模型和分词器
        print(f"Loading model {model_name}...")
        device_map = split_model('InternVL2_5-8B') # InternVL2_5-2B
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,  # 使用 bfloat16 精度
            low_cpu_mem_usage=True,      # 减少 CPU 内存使用
            use_flash_attn=False,        # 启用 Flash Attention
            trust_remote_code=True,      # 信任远程代码
            device_map=device_map,
        ).eval().to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
        print(f"Model max sequence length: {self.tokenizer.model_max_length}")
        print("Model and tokenizer loaded!")

        # 修改生成配置，确保结果稳定
        self.generation_config = dict(
            max_new_tokens=1024,  # 最大生成 token 数
            do_sample=False,      # 关闭随机采样
            temperature=0,        # 温度设置为 0，确保确定性输出
            top_p=1.0,            # top-p 采样设置为 1.0
            top_k=50,              # top-k 采样设置为 50
        )

    def chat(self, image_list: list, txt: str) -> str:
        device = next(self.model.parameters()).device
        if image_list is None:
            response = self.model.chat(self.tokenizer, None, txt, self.generation_config,
                        history=None, return_history=False)
        else:
            pixel_value_list = []
            for image_array in image_list:
                pixel_value = load_image_array(image_array=image_array, device=device)
                pixel_value_list.append(pixel_value)
            pixel_values_tensor = torch.cat(pixel_value_list, dim=0)
            response = self.model.chat(self.tokenizer, pixel_values_tensor, txt, self.generation_config,
                                history=None, return_history=False)
        
        return response
    
class INTERNVL2_5ITMClient:
    def __init__(self, port: int = 15181):
        self.url = f"http://localhost:{port}/internvl2_5itm"

    def chat(self, image_list: list, txt: str) -> float:
        # response = send_request_vlm(self.url, timeout=5,image=image_list, txt=txt)
        # return response["response"]
        try:
            response = send_request_vlm(self.url, timeout=5, image=image_list, txt=txt)
            return response["response"]
        except Exception as e:  # 捕获所有异常
            # print(f"Request failed: {e}")
            return "-1"  # 返回默认值

if __name__ == "__main__":
    
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=15181)
    args = parser.parse_args()

    print("Loading model...")

    class INTERNVL2_5ITMServer(ServerMixin, INTERNVL2_5ITM):
        def process_payload(self, payload: dict) -> dict:
            # image = str_to_image(payload["image"])
            if payload["image"] is not None:
                images = [str_to_image(img_str) for img_str in payload["image"]]
            else:
                images = None
            return {"response": self.chat(images, payload["txt"])} # image

    internvl2_5 = INTERNVL2_5ITMServer()
    print("Model loaded!")
    print(f"Hosting on port {args.port}...")
    host_model(internvl2_5, name="internvl2_5itm", port=args.port)

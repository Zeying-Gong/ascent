# export CUDA_VISIBLE_DEVICES=0,1,3

import numpy as np
import torch
from modelscope import AutoModelForCausalLM, AutoTokenizer
from vlfm.vlm.server_wrapper import ServerMixin, host_model, send_request, send_request_vlm
import random


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

class QWen2_5ITM:
    """QWen2_5 Image-Text Matching model using transformers."""

    def __init__(self, model_name: str = "/home/zeyingg/github/habitat-lab-vlfm/Qwen/Qwen2.5-7b", device: str = None, seed: int = 2025) -> None: 
        """
        初始化QWen2.5模型。

        :param model_name: 模型名称或路径
        :param device: 设备（cuda 或 cpu）
        """
        set_seed(seed)

        # 设置设备（默认使用 CUDA 或 CPU）
        if device is None:
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.device = device

        # 加载模型和分词器
        print(f"Loading model {model_name}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,  # 使用 bfloat16 精度
            device_map={"": self.device},          # 自动分配到可用设备
            trust_remote_code=True,      # 信任远程代码
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.tokenizer.device = self.device  # 确保 tokenizer 知晓设备
        print(f"Model max sequence length: {self.tokenizer.model_max_length}")
        print("Model and tokenizer loaded!")
        # 关键修改4：验证设备一致性
        print(f"Model device: {next(self.model.parameters()).device}")
        print(f"Target device: {self.device}")
        # 修改生成配置，确保结果稳定
        self.generation_config = dict(
            max_new_tokens=512,  # 最大生成 token 数
            do_sample=False,      # 关闭随机采样
            temperature=0,        # 温度设置为 0，确保确定性输出
            top_p=1.0,            # top-p 采样设置为 1.0
            top_k=50,             # top-k 采样设置为 50
        )

    def chat(self, txt: str) -> str:
        # 构造输入消息
        messages = [
            {"role": "system", "content": "You are an AI assistant with advanced spatial reasoning capabilities. Your task is to choose the optimal option to find the target object."},
            {"role": "user", "content": txt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # 准备模型输入
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        # 关键修改6：添加同步点
        torch.cuda.synchronize(self.device)
        # 生成文本
        generated_ids = self.model.generate(
            **model_inputs,
            # max_new_tokens=512,
            **self.generation_config
        )

        # 提取生成的输出
        input_ids = model_inputs.input_ids[0]
        output_ids = generated_ids[0][len(input_ids):]
        response = self.tokenizer.decode(output_ids, skip_special_tokens=True)

        return response
    
class Qwen2_5ITMClient:
    def __init__(self, port: int = 13181):
        self.url = f"http://localhost:{port}/qwen2_5itm"

    def chat(self, txt: str) -> float:
        # response = send_request_vlm(self.url, timeout=5,image=image_list, txt=txt)
        # return response["response"]
        try:
            response = send_request_vlm(self.url, timeout=20, txt=txt)
            return response["response"]
        except Exception as e:  # 捕获所有异常
            print(f"Request failed: {e}")
            return "-1"  # 返回默认值

if __name__ == "__main__":
    # model = QWen2_5ITM()
    # response = model.chat("Give me a short introduction to large language models.")
    # print("Response:", response)
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=13181)
    args = parser.parse_args()

    print("Loading model...")

    class Qwen2_5ITMServer(ServerMixin, QWen2_5ITM):
        def process_payload(self, payload: dict) -> dict:
            return {"response": self.chat(payload["txt"])}

    qwen2_5 = Qwen2_5ITMServer()
    print("Model loaded!")
    print(f"Hosting on port {args.port}...")
    host_model(qwen2_5, name="qwen2_5itm", port=args.port)

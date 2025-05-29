# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import Any, Optional

import numpy as np
import torch
from PIL import Image

from transformers import AutoModel, CLIPImageProcessor
from transformers import AutoTokenizer

from vlfm.vlm.server_wrapper import ServerMixin, host_model, send_request, str_to_image


class InternVLITM:

    def __init__(
        self,
        model_path: str = "/home/zeyingg/github/habitat-lab-vlfm/InternVL/InternVL-14B-224px", # "/home/zeyingg/github/habitat-lab-vlfm/InternVL/InternViT-300M-448px-V2_5", # 
        device: Optional[Any] = "cpu", # Optional[Any] = None,
    ) -> None:
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

        self.device = device
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map='cpu').eval() # auto

        self.image_processor = CLIPImageProcessor.from_pretrained(model_path,trust_remote_code=True,device_map='cpu')

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, use_fast=False, add_eos_token=True,trust_remote_code=True,device_map='cpu')
        self.tokenizer.pad_token_id = 0  # set pad_token_id to 0
    def cosine(self, image: np.ndarray, txt: str) -> float:
        """
        Compute the cosine similarity between the image and the prompt.

        Args:
            image (numpy.ndarray): The input image as a numpy array.
            txt (str): The text to compare the image to.

        Returns:
            float: The cosine similarity between the image and the prompt.
        """
        pil_img = [Image.fromarray(image)]
        txt = 'summarize:' + txt
        pixel_value = self.image_processor(images=pil_img, return_tensors='pt').pixel_values
        pixel_value = pixel_value.to(torch.bfloat16)#.cuda()
        input_id = self.tokenizer([txt], return_tensors='pt', max_length=80,
                            truncation=True, padding='max_length').input_ids#.cuda()
        image_features = self.model.encode_image(pixel_value, mode = 'InternVL-C') # G
        text_features = self.model.encode_text(input_id)
        text_features = text_features.to(image_features.device)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        print("shape:",image_features.shape, text_features.shape)
        cosine = torch.nn.functional.cosine_similarity(image_features, text_features).item()
        return cosine


class InternVLITMClient:
    def __init__(self, port: int = 14182):
        self.url = f"http://localhost:{port}/internvlitm"

    def cosine(self, image: np.ndarray, txt: str) -> float:
        response = send_request(self.url, image=image, txt=txt)
        return float(response["response"])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=14182)
    args = parser.parse_args()

    print("Loading model...")

    class InternVLITMServer(ServerMixin, InternVLITM):
        def process_payload(self, payload: dict) -> dict:
            image = str_to_image(payload["image"])
            return {"response": self.cosine(image, payload["txt"])}

    blip = InternVLITMServer()
    print("Model loaded!")
    print(f"Hosting on port {args.port}...")
    host_model(blip, name="internvlitm", port=args.port)

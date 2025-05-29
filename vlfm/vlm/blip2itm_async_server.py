# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import Any, Optional, List

import numpy as np
import torch
from PIL import Image

from vlfm.vlm.server_wrapper import ServerMixin, host_model, send_request_vlm, str_to_image
try:
  from lavis.models import load_model_and_preprocess
except ModuleNotFoundError:
    print("Could not import lavis. This is OK if you are only using the client.")

class AsyncBLIP2ITM:
    """BLIP 2 Image-Text Matching model."""

    def __init__(
        self,
        name: str = "blip2_image_text_matching",
        model_type: str = "pretrain",
        device: Optional[Any] = None,
    ) -> None:
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

        self.model, self.vis_processors, self.text_processors = load_model_and_preprocess(
            name=name,
            model_type=model_type,
            is_eval=True,
            device=device,
        )
        self.device = device

    def cosine(self, images: List[np.ndarray], texts: List[str]) -> List[float]:
        """
        Compute the cosine similarity between the images and the prompts.

        Args:
            images (List[numpy.ndarray]): The input images as a list of numpy arrays.
            texts (List[str]): The texts to compare the images to.

        Returns:
            List[float]: The cosine similarities between the images and the prompts.
        """
        pil_images = [Image.fromarray(image) for image in images]
        img_batch = torch.stack([self.vis_processors["eval"](img) for img in pil_images]).to(self.device)
        txt_batch = [self.text_processors["eval"](txt) for txt in texts]

        with torch.inference_mode():
            cosine_scores = self.model({"image": img_batch, "text_input": txt_batch}, match_head="itc").cpu().numpy()

        return cosine_scores.tolist()

class AsyncBLIP2ITMClient:
    def __init__(self, port: int = 12182):
        self.url = f"http://localhost:{port}/blip2itm"

    def cosine(self, images: List[np.ndarray], texts: List[str]) -> List[float]:
        response = send_request_vlm(self.url, images=images, texts=texts, method="cosine")
        return [float(score[0]) for score in response["response"]]

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=12182)
    args = parser.parse_args()

    print("Loading model...")

    class AsyncBLIP2ITMServer(ServerMixin, AsyncBLIP2ITM):
        def process_payload(self, payload: dict) -> dict:
            images = [str_to_image(img) for img in payload["images"]]
            response = self.cosine(images, payload["texts"])
            return {"response": response}

    blip = AsyncBLIP2ITMServer()
    print("Model loaded!")
    print(f"Hosting on port {args.port}...")
    host_model(blip, name="blip2itm", port=args.port)
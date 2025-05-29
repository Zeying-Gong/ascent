# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import base64
import os
import random
import socket
import time
from typing import Any, Dict

import cv2
import numpy as np
import requests
from flask import Flask, jsonify, request
from PIL import Image
import io 

class ServerMixin:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def process_payload(self, payload: dict) -> dict:
        raise NotImplementedError


def host_model(model: Any, name: str, port: int = 5000) -> None:
    """
    Hosts a model as a REST API using Flask.
    """
    app = Flask(__name__)

    @app.route(f"/{name}", methods=["POST"])
    def process_request() -> Dict[str, Any]:
        payload = request.json
        return jsonify(model.process_payload(payload))

    app.run(host="localhost", port=port)


def bool_arr_to_str(arr: np.ndarray) -> str:
    """Converts a boolean array to a string."""
    packed_str = base64.b64encode(arr.tobytes()).decode()
    return packed_str


def str_to_bool_arr(s: str, shape: tuple) -> np.ndarray:
    """Converts a string to a boolean array."""
    # Convert the string back into bytes using base64 decoding
    bytes_ = base64.b64decode(s)

    # Convert bytes to np.uint8 array
    bytes_array = np.frombuffer(bytes_, dtype=np.uint8)

    # Reshape the data back into a boolean array
    unpacked = bytes_array.reshape(shape)
    return unpacked


def image_to_str(img_np: np.ndarray, quality: float = 90.0) -> str:
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    retval, buffer = cv2.imencode(".jpg", img_np, encode_param)
    img_str = base64.b64encode(buffer).decode("utf-8")
    return img_str

def image_to_str_pillow(img_np: np.ndarray, format: str = "JPEG", quality: int = 90) -> str:
    """
    使用 Pillow 将图像转换为 base64 编码的字符串。

    Args:
        img_np (np.ndarray): 输入的图像数组。
        format (str): 图像格式（默认 "JPEG"）。
        quality (int): 编码质量（0-100，默认 90）。

    Returns:
        str: base64 编码的字符串。
    """
    # 将 numpy 数组转换为 Pillow 图像
    img_pil = Image.fromarray(img_np)
    
    # 将图像保存到字节流
    buffer = io.BytesIO()
    img_pil.save(buffer, format=format, quality=quality)
    
    # 将字节流转换为 base64 字符串
    img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return img_str

def str_to_image(img_str: str) -> np.ndarray:
    img_bytes = base64.b64decode(img_str)
    img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img_np = cv2.imdecode(img_arr, cv2.IMREAD_ANYCOLOR)
    return img_np


def send_request(url: str, **kwargs: Any) -> dict:
    response = {}
    for attempt in range(10):
        try:
            response = _send_request(url, **kwargs)
            break
        except Exception as e:
            if attempt == 9:
                print(e)
                exit()
            else:
                print(f"Error: {e}. Retrying in 20-30 seconds...")
                time.sleep(20 + random.random() * 10)

    return response


def _send_request(url: str, **kwargs: Any) -> dict:
    lockfiles_dir = "lockfiles"
    if not os.path.exists(lockfiles_dir):
        os.makedirs(lockfiles_dir)
    filename = url.replace("/", "_").replace(":", "_") + ".lock"
    filename = filename.replace("localhost", socket.gethostname())
    filename = os.path.join(lockfiles_dir, filename)
    try:
        while True:
            # Use a while loop to wait until this filename does not exist
            while os.path.exists(filename):
                # If the file exists, wait 50ms and try again
                time.sleep(0.05)

                try:
                    # If the file was last modified more than 120 seconds ago, delete it
                    if time.time() - os.path.getmtime(filename) > 120:
                        os.remove(filename)
                except FileNotFoundError:
                    pass

            rand_str = str(random.randint(0, 1000000))

            with open(filename, "w") as f:
                f.write(rand_str)
            time.sleep(0.05)
            try:
                with open(filename, "r") as f:
                    if f.read() == rand_str:
                        break
            except FileNotFoundError:
                pass

        # Create a payload dict which is a clone of kwargs but all np.array values are
        # converted to strings
        payload = {}
        for k, v in kwargs.items():
            if isinstance(v, np.ndarray):
                payload[k] = image_to_str(v, quality=kwargs.get("quality", 90))
            else:
                payload[k] = v

        # Set the headers
        headers = {"Content-Type": "application/json"}

        start_time = time.time()
        while True:
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=1)
                if resp.status_code == 200:
                    result = resp.json()
                    break
                else:
                    raise Exception("Request failed")
            except (
                requests.exceptions.Timeout,
                requests.exceptions.RequestException,
            ) as e:
                print(e)
                if time.time() - start_time > 20:
                    raise Exception("Request timed out after 20 seconds")

        try:
            # Delete the lock file
            os.remove(filename)
        except FileNotFoundError:
            pass

    except Exception as e:
        try:
            # Delete the lock file
            os.remove(filename)
        except FileNotFoundError:
            pass
        raise e

    return result

def send_request_vlm(url: str, timeout: int = 20, **kwargs: Any) -> dict:
    """
    Sends a request to the given URL with retries.
    
    :param url: The endpoint to send the request to.
    :param timeout: Maximum time (in seconds) allowed for a request to complete.
    :param kwargs: Additional arguments to be passed to _send_request.
    :return: Response as a dictionary.
    """
    response = {}
    for attempt in range(10):
        try:
            response = _send_request_vlm(url, timeout=timeout, **kwargs)
            break
        except Exception as e:
            if attempt == 2:
                # print(e)
                # exit()
                error_message = f"Error: {e}. All retries failed."
                # response = {"error": "Request failed after multiple retries"} 
                raise Exception(error_message)  # 抛出自定义异常
            else:
                print(f"Error: {e}. Retrying in 5-10 seconds...")
                time.sleep(5 + random.random() * 5)

    return response


def _send_request_vlm(url: str, timeout: int, **kwargs: Any) -> dict:
    """
    Internal function to send a request with locking and retry logic.

    :param url: The endpoint to send the request to.
    :param timeout: Maximum time (in seconds) allowed for a request to complete.
    :param kwargs: Payload and additional options.
    :return: Response as a dictionary.
    """
    lockfiles_dir = "lockfiles"
    if not os.path.exists(lockfiles_dir):
        os.makedirs(lockfiles_dir)
    filename = url.replace("/", "_").replace(":", "_") + ".lock"
    filename = filename.replace("localhost", socket.gethostname())
    filename = os.path.join(lockfiles_dir, filename)
    try:
        while True:
            while os.path.exists(filename):
                time.sleep(0.05)
                try:
                    if time.time() - os.path.getmtime(filename) > 120:
                        os.remove(filename)
                except FileNotFoundError:
                    pass

            rand_str = str(random.randint(0, 1000000))

            with open(filename, "w") as f:
                f.write(rand_str)
            time.sleep(0.05)
            try:
                with open(filename, "r") as f:
                    if f.read() == rand_str:
                        break
            except FileNotFoundError:
                pass

        payload = {}
        for k, v in kwargs.items():
            if isinstance(v, np.ndarray):
                payload[k] = image_to_str(v, quality=kwargs.get("quality", 90))
            elif isinstance(v, list) and all(isinstance(i, np.ndarray) for i in v):
                payload[k] = [image_to_str(img, quality=kwargs.get("quality", 90)) for img in v]
            else:
                payload[k] = v

        headers = {"Content-Type": "application/json"}

        start_time = time.time()
        while True:
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
                if resp.status_code == 200:
                    result = resp.json()
                    break
                else:
                    raise Exception("Request failed with status code: " + str(resp.status_code))
            except (requests.exceptions.Timeout, requests.exceptions.RequestException) as e:
                print(e)
                if time.time() - start_time > timeout:
                    raise Exception(f"Request timed out after {timeout} seconds")

        try:
            os.remove(filename)
        except FileNotFoundError:
            pass

    except Exception as e:
        try:
            os.remove(filename)
        except FileNotFoundError:
            pass
        raise e

    return result

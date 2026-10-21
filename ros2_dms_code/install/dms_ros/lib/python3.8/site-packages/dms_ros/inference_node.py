#!/usr/bin/env python3
"""Inference node.

Uses the same model call as validate.py (the validated path):
two images (full frame + face crop), per-model training prompt,
greedy decoding, strict parser.
"""
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import gc
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import torch
import yaml
from PIL import Image as PILImage
from transformers import AutoProcessor, AutoModelForVision2Seq

from dms_interfaces.msg import FramePair
from dms_ros.common import image_to_rgb, parse_prediction


class InferenceNode(Node):

    def __init__(self):
        super().__init__('inference_node')

        self.declare_parameter('models_dir', '')
        self.declare_parameter('active_model', '')
        self.declare_parameter('prompts_file', '')
        self.declare_parameter('max_new_tokens', 12)
        self.declare_parameter('longest_edge', 512)
        self.declare_parameter('do_sample', False)
        self.declare_parameter('queue_size', 1)

        models_dir = self.get_parameter('models_dir').value
        self.active_model = self.get_parameter('active_model').value
        prompts_file = self.get_parameter('prompts_file').value
        self.max_new_tokens = int(self.get_parameter('max_new_tokens').value)
        self.longest_edge = int(self.get_parameter('longest_edge').value)
        self.do_sample = bool(self.get_parameter('do_sample').value)
        queue_size = self.get_parameter('queue_size').value

        if not models_dir or not self.active_model:
            raise ValueError('models_dir / active_model not set')
        if not prompts_file or not os.path.isfile(prompts_file):
            raise FileNotFoundError(f'prompts_file not found: {prompts_file}')

        with open(prompts_file, 'r') as f:
            prompts = yaml.safe_load(f).get('model_prompts', {})
        if self.active_model not in prompts:
            raise KeyError(
                f"No prompt for '{self.active_model}' in {prompts_file}. "
                f"Available: {list(prompts)}")
        self.instruction = str(prompts[self.active_model]).strip()

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        dtype = torch.float16 if self.device.type == 'cuda' else torch.float32

        model_dir = os.path.join(models_dir, self.active_model)
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(f'Model dir not found: {model_dir}')

        self.get_logger().info(
            f'Loading {self.active_model} on {self.device} ({dtype})...')
        self.processor = AutoProcessor.from_pretrained(
            model_dir, local_files_only=True)
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_dir, torch_dtype=dtype,
            low_cpu_mem_usage=True, local_files_only=True)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.model.config.use_cache = True
        self.get_logger().info('Model loaded')

        self.state_pub = self.create_publisher(String, '/dms/state', 10)
        self.metric_pub = self.create_publisher(
            String, '/benchmark/raw_metrics', 50)
        self.sub = self.create_subscription(
            FramePair, '/dms/frame_pair', self.on_pair, queue_size)

    def _run_model(self, full_crop, face_crop):
        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "image"},
                {"type": "text", "text": self.instruction},
            ],
        }]
        prompt = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False)
        inputs = self.processor(
            text=prompt,
            images=[[PILImage.fromarray(full_crop).convert("RGB"),
                     PILImage.fromarray(face_crop).convert("RGB")]],
            return_tensors="pt",
            size={"longest_edge": self.longest_edge},
            max_image_size={"longest_edge": self.longest_edge},
        )
        inputs = {k: v.to(self.device) if torch.is_tensor(v) else v
                  for k, v in inputs.items()}

        pv = inputs["pixel_values"]
        if pv.ndim != 5 or pv.shape[1] != 2:
            raise RuntimeError(
                f"Expected 2 images per sample, got {tuple(pv.shape)}")

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens,
                do_sample=self.do_sample)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - started) * 1000

        n_in = inputs["input_ids"].shape[-1]
        raw = self.processor.batch_decode(
            generated[:, n_in:], skip_special_tokens=True)[0].strip()
        return raw, latency_ms

    def on_pair(self, msg):
        if msg.skip:
            self.get_logger().info(
                f'Frame #{msg.seq} -> NORMAL '
                f'(mesh skip: {msg.gate_reason}, no model call)')
            state = String()
            state.data = 'NORMAL'
            self.state_pub.publish(state)
            return
        try:
            full_crop = image_to_rgb(msg.full_crop)
            face_crop = image_to_rgb(msg.face_crop)
            raw, latency_ms = self._run_model(full_crop, face_crop)
            label, _exact = parse_prediction(raw)
            parser_failed = (label == "INVALID")

            why = f', mesh: {msg.gate_reason}' if msg.mesh_enabled else ''
            self.get_logger().info(
                f'Frame #{msg.seq} -> {label} '
                f'({latency_ms:.0f} ms, raw="{raw}"{why})')

            state = String()
            state.data = label
            self.state_pub.publish(state)

            safe_raw = raw.replace('|', '/').replace('\n', ' ')
            metric = String()
            metric.data = (f"{self.active_model}|{int(latency_ms)}|{label}|"
                           f"{safe_raw}|{'1' if parser_failed else '0'}")
            self.metric_pub.publish(metric)

            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
        except Exception as e:
            self.get_logger().error(f'Inference failed: {e}')


def main(args=None):
    rclpy.init(args=args)
    try:
        rclpy.spin(InferenceNode())
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()

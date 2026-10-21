#!/usr/bin/env python3
"""Preprocess node.

/dms/camera/raw (bgr8)  ->  enhance_frame
    mesh off: always send the frame to the model
    mesh on : the mesh gate decides invoke / skip
              invoke -> resize_full_frame + extract_face_crop -> /dms/frame_pair
              skip   -> /dms/frame_pair with skip=True and no images

The crops are made exactly like create_dual_crops() in validate.py.
"""
import os
import time
from pathlib import Path

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from dms_interfaces.msg import FramePair

from dms_ros import dms_preprocess as lgp
from dms_ros.common import image_to_bgr, fill_image

NAN = float('nan')


class PreprocessNode(Node):

    def __init__(self):
        super().__init__('preprocess_node')
        self.declare_parameter('queue_size', 1)
        self.declare_parameter('debug_dir', '/home/jetson/live_debug')
        self.declare_parameter('mesh_enabled', False)
        self.declare_parameter('scheduled_interval_sec', 10.0)
        self.declare_parameter('face_landmarker_path', '')

        queue_size = self.get_parameter('queue_size').value
        self.debug_dir = self.get_parameter('debug_dir').value
        self.mesh_enabled = bool(self.get_parameter('mesh_enabled').value)
        self.scheduled_interval = float(
            self.get_parameter('scheduled_interval_sec').value)
        self.last_invoke_t = -1e9
        self.extractor = None
        self.gate = None

        if self.debug_dir:
            os.makedirs(self.debug_dir, exist_ok=True)
            self.get_logger().info(f'Saving debug images to {self.debug_dir}')

        if lgp._get_haar_face_detector().empty() or \
                lgp._get_haar_profile_detector().empty():
            raise RuntimeError(
                'Haar face cascades failed to load. '
                'Face crops would be wrong.')
        self.get_logger().info(f'Cascades: {lgp._cascade_dir()}')

        if self.mesh_enabled:
            self._start_mesh()

        self.seq = 0
        self.pub = self.create_publisher(
            FramePair, '/dms/frame_pair', queue_size)
        self.sub = self.create_subscription(
            Image, '/dms/camera/raw', self.on_image, queue_size)
        self.get_logger().info(
            f"Preprocess node ready. Mesh {'ON' if self.mesh_enabled else 'OFF'}")

    def _start_mesh(self):
        try:
            from dms_ros.mesh_landmark_extractor import (
                DEFAULT_MODEL_PATH, MeshLandmarkExtractor)
            from dms_ros.mesh_gate import MeshGate
            configured = self.get_parameter('face_landmarker_path').value
            model_path = Path(configured) if configured else DEFAULT_MODEL_PATH
            self.extractor = MeshLandmarkExtractor(model_path=model_path)
            self.gate = MeshGate()
            self.get_logger().info(
                f'Mesh gate: ear_closed={self.gate.ear_closed}, '
                f'yaw_away={self.gate.yaw_away}, '
                f'scheduled every {self.scheduled_interval}s')
        except Exception as e:
            self.get_logger().error(
                f'Mesh could not start ({e}). Running with mesh OFF.')
            self.mesh_enabled = False
            self.extractor = None
            self.gate = None

    def on_image(self, msg):
        started = time.perf_counter()
        try:
            raw_bgr = image_to_bgr(msg)
            enhanced_rgb, quality_ok = lgp.enhance_frame(raw_bgr)
            if enhanced_rgb is None:
                self.get_logger().warn('enhance_frame failed, frame dropped')
                return

            out = FramePair()
            out.header = msg.header
            out.seq = self.seq
            out.mesh_enabled = self.mesh_enabled
            out.skip = False
            out.gate_reason = 'mesh_disabled'
            out.mesh_ms = 0.0
            out.ear = NAN
            out.yaw = NAN

            if self.mesh_enabled:
                t0 = time.perf_counter()
                landmarks = self.extractor.extract(
                    enhanced_rgb).to_landmarks_dict()
                out.mesh_ms = float((time.perf_counter() - t0) * 1000.0)
                now = time.monotonic()
                decision = self.gate.evaluate(
                    landmarks,
                    True if quality_ok is None else bool(quality_ok),
                    now, self.last_invoke_t, self.scheduled_interval)
                out.skip = decision.decision == 'skip'
                out.gate_reason = decision.reason
                out.ear = decision.ear
                out.yaw = decision.yaw
                if not out.skip:
                    self.last_invoke_t = now

            full_crop = face_crop = None
            if not out.skip:
                full_crop = lgp.resize_full_frame(enhanced_rgb)
                face_crop = lgp.extract_face_crop(enhanced_rgb, landmarks=None)
                fill_image(out.full_crop, full_crop, 'rgb8')
                fill_image(out.face_crop, face_crop, 'rgb8')

            self._save_debug(out, raw_bgr, enhanced_rgb, full_crop, face_crop)
        except Exception as e:
            self.get_logger().error(f'Preprocess failed: {e}')
            return

        self.seq += 1
        out.preprocess_ms = (time.perf_counter() - started) * 1000.0
        self.pub.publish(out)
        self.get_logger().debug(
            f'Frame #{out.seq} {"skip" if out.skip else "invoke"} '
            f'({out.gate_reason}) in {out.preprocess_ms:.0f} ms')

    def _save_debug(self, out, raw_bgr, enhanced_rgb, full_crop, face_crop):
        if not self.debug_dir:
            return
        try:
            if out.skip:
                tag = f'{out.seq:05d}_skip_{out.gate_reason}'
            else:
                found = lgp.detect_face_bbox_haar(enhanced_rgb) is not None
                tag = f'{out.seq:05d}_face{"YES" if found else "NO"}'
            base = os.path.join(self.debug_dir, tag)
            cv2.imwrite(base + '_raw.jpg', raw_bgr)
            if full_crop is not None:
                cv2.imwrite(base + '_full.jpg',
                            cv2.cvtColor(full_crop, cv2.COLOR_RGB2BGR))
                cv2.imwrite(base + '_face.jpg',
                            cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR))
        except Exception as e:
            self.get_logger().warn(f'Debug save failed: {e}')

    def destroy_node(self):
        if self.extractor is not None:
            self.extractor.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = PreprocessNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

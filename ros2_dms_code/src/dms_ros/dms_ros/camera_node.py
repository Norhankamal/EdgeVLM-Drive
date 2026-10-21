#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2


class CameraNode(Node):

    def __init__(self):
        super().__init__('camera_node')

        # ── Parameters (no defaults → forced from YAML) ───────────
        self.declare_parameter('camera_index',                -1)
        self.declare_parameter('width',                        0)
        self.declare_parameter('height',                       0)
        self.declare_parameter('frame_sampling_interval_sec', 0.0)
        self.declare_parameter('queue_size',                   1)
        
        idx      = self.get_parameter('camera_index').value
        w        = self.get_parameter('width').value
        h        = self.get_parameter('height').value
        interval = self.get_parameter('frame_sampling_interval_sec').value
        queue_size = self.get_parameter('queue_size').value
        # ── Validate ──────────────────────────────────────────────
        if idx < 0:
            raise ValueError(
                'camera_index not set! Check dms_params.yaml')
        if w <= 0:
            raise ValueError(
                'width not set! Check dms_params.yaml')
        if h <= 0:
            raise ValueError(
                'height not set! Check dms_params.yaml')
        if interval <= 0.0:
            raise ValueError(
                'frame_sampling_interval_sec not set! '
                'Check dms_params.yaml')

        # ── Open Camera ───────────────────────────────────────────
        self.cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            self.get_logger().error(
                f'Cannot open camera index={idx}')
            raise RuntimeError('Camera not found')

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

        self.get_logger().info(
            f'✅ Camera ready '
            f'{w}x{h} interval={interval}s')

        # ── Publisher ─────────────────────────────────────────────
        self.pub = self.create_publisher(
            Image, '/dms/camera/raw', queue_size)

        # ── Timer ─────────────────────────────────────────────────
        self.timer = self.create_timer(interval, self.tick)

    def tick(self):
        self.cap.grab()
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn('Frame read failed')
            return

        msg             = Image()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera_frame'
        msg.height          = frame.shape[0]
        msg.width           = frame.shape[1]
        msg.encoding        = 'bgr8'
        msg.is_bigendian    = False
        msg.step            = frame.shape[1] * 3
        msg.data            = frame.tobytes()

        self.pub.publish(msg)
        self.get_logger().debug(
            f'📸 Frame → /dms/camera/raw')

    def destroy_node(self):
        if self.cap.isOpened():
            self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    try:
        rclpy.spin(CameraNode())
    except KeyboardInterrupt:
        pass
    except (RuntimeError, ValueError) as e:
        print(f'Camera error: {e}')
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()

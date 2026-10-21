"""Small helpers shared by the nodes (no ROS or torch needed)."""
import numpy as np

LABELS = ["NORMAL", "DROWSY", "LOOKING_AWAY", "PHONE_USAGE"]


def image_to_bgr(msg):
    """sensor_msgs/Image (bgr8 or rgb8) -> HxWx3 uint8 BGR array."""
    arr = np.frombuffer(msg.data, dtype=np.uint8)
    arr = arr.reshape((msg.height, msg.step // 3, 3))[:, :msg.width, :]
    if msg.encoding == "rgb8":
        arr = arr[:, :, ::-1]
    elif msg.encoding != "bgr8":
        raise ValueError(f"Unsupported encoding: {msg.encoding}")
    return np.ascontiguousarray(arr)


def image_to_rgb(msg):
    """sensor_msgs/Image (rgb8) -> HxWx3 uint8 RGB array."""
    if msg.encoding != "rgb8":
        raise ValueError(f"Expected rgb8, got {msg.encoding}")
    arr = np.frombuffer(msg.data, dtype=np.uint8)
    arr = arr.reshape((msg.height, msg.step // 3, 3))[:, :msg.width, :]
    return np.ascontiguousarray(arr)


def fill_image(msg, array, encoding):
    """Fill a sensor_msgs/Image from a HxWx3 uint8 array."""
    array = np.ascontiguousarray(array, dtype=np.uint8)
    msg.height = int(array.shape[0])
    msg.width = int(array.shape[1])
    msg.encoding = encoding
    msg.is_bigendian = False
    msg.step = int(array.shape[1]) * 3
    msg.data = array.tobytes()
    return msg


def parse_prediction(raw_text):
    """Same strict parser as validate.py. Returns (label, exact)."""
    normalized = str(raw_text).strip().upper()
    exact = normalized in LABELS
    cleaned = normalized.strip(" .,;:!?'\"")
    return (cleaned if cleaned in LABELS else "INVALID"), exact

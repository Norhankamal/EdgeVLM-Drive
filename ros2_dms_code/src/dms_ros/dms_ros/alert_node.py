#!/usr/bin/env python3
"""Alert node: reacts whenever /dms/state is DROWSY.

Actions on every DROWSY detection:
  1. WARN message in the terminal
  2. terminal bell (beeps in the SSH terminal if the terminal allows it)
  3. beep sound through the Jetson audio (HDMI or USB speaker)
  4. publishes "DROWSY" on /dms/alert
  5. optional buzzer/LED on a GPIO pin (buzzer_pin > 0)
"""
import math
import os
import shutil
import struct
import subprocess
import tempfile
import threading
import time
import wave

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


def make_beep_wav(path, seconds, freq=1000, rate=22050):
    frames = int(seconds * rate)
    fade = int(0.01 * rate)
    with wave.open(path, 'wb') as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(rate)
        data = bytearray()
        for i in range(frames):
            gain = min(1.0, i / fade, (frames - i) / fade)
            value = int(20000 * gain * math.sin(2 * math.pi * freq * i / rate))
            data += struct.pack('<h', value)
        f.writeframes(bytes(data))


class AlertNode(Node):

    def __init__(self):
        super().__init__('alert_node')
        self.declare_parameter('buzzer_pin', 0)       # 0 = no buzzer
        self.declare_parameter('beep_sec', 2.0)
        self.declare_parameter('cooldown_sec', 0.0)   # 0 = alert every time
        self.declare_parameter('sound_enabled', True)
        self.declare_parameter('audio_device', '')    # '' = default, or e.g. plughw:0,3
        self.declare_parameter('terminal_bell', True)

        self.pin = int(self.get_parameter('buzzer_pin').value)
        self.beep_sec = float(self.get_parameter('beep_sec').value)
        self.cooldown = float(self.get_parameter('cooldown_sec').value)
        self.last_alert = 0.0
        self.gpio = None
        self.sound_enabled = bool(self.get_parameter('sound_enabled').value)
        self.audio_device = self.get_parameter('audio_device').value
        self.terminal_bell = bool(self.get_parameter('terminal_bell').value)
        self.proc = None
        self.wav_path = os.path.join(tempfile.gettempdir(), 'dms_alert.wav')

        if self.sound_enabled:
            if shutil.which('aplay') is None:
                self.get_logger().warn(
                    'aplay not found (sudo apt install alsa-utils), sound disabled')
                self.sound_enabled = False
            else:
                make_beep_wav(self.wav_path, self.beep_sec)
                self.get_logger().info(
                    f'Alert sound ready (device="{self.audio_device or "default"}")')

        if self.pin > 0:
            try:
                import Jetson.GPIO as GPIO
                GPIO.setmode(GPIO.BOARD)
                GPIO.setup(self.pin, GPIO.OUT, initial=GPIO.LOW)
                self.gpio = GPIO
                self.get_logger().info(f'Buzzer on board pin {self.pin}')
            except Exception as e:
                self.get_logger().warn(
                    f'GPIO unavailable ({e}), buzzer disabled')

        self.pub = self.create_publisher(String, '/dms/alert', 10)
        self.sub = self.create_subscription(
            String, '/dms/state', self.on_state, 10)
        self.get_logger().info('Alert node ready (trigger: DROWSY)')

    def on_state(self, msg):
        if msg.data != 'DROWSY':
            return
        now = time.monotonic()
        if self.cooldown > 0 and (now - self.last_alert) < self.cooldown:
            return
        self.last_alert = now
        bell = '\a' if self.terminal_bell else ''
        self.get_logger().warn(f'*** ALERT: DRIVER DROWSY ***{bell}')
        out = String()
        out.data = 'DROWSY'
        self.pub.publish(out)
        if self.sound_enabled:
            self._play_sound()
        if self.gpio is not None:
            threading.Thread(target=self._beep, daemon=True).start()

    def _play_sound(self):
        if self.proc is not None:
            if self.proc.poll() is None:
                return
            if self.proc.returncode != 0:
                err = self.proc.stderr.read().decode(errors='ignore').strip()
                self.get_logger().warn(f'aplay failed: {err}')
        cmd = ['aplay', '-q']
        if self.audio_device:
            cmd += ['-D', self.audio_device]
        cmd.append(self.wav_path)
        self.proc = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    def _beep(self):
        self.gpio.output(self.pin, self.gpio.HIGH)
        time.sleep(self.beep_sec)
        self.gpio.output(self.pin, self.gpio.LOW)

    def destroy_node(self):
        if self.gpio is not None:
            self.gpio.output(self.pin, self.gpio.LOW)
            self.gpio.cleanup(self.pin)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    try:
        rclpy.spin(AlertNode())
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()

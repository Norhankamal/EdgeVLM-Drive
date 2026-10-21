#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

import csv
import os
import re
import time
import threading
import queue
from collections import defaultdict


def get_ram_mb_fallback():
    try:
        with open('/proc/meminfo') as f:
            content = f.read()
        total = int(re.search(
            r'MemTotal:\s+(\d+)', content).group(1)) // 1024
        avail = int(re.search(
            r'MemAvailable:\s+(\d+)', content).group(1)) // 1024
        return total - avail
    except Exception:
        return 0


class PostBenchmarkNode(Node):
    """
    Post-inference benchmark node.

    Records:
    1. Inference latency (avg/min/max/std)
    2. Most frequent predicted state
    3. State distribution (human readable)
    4. Peak RAM via jtop or /proc/meminfo
    5. VLM invocation frequency
    6. Parser failure rate
    """

    def __init__(self):
        super().__init__('post_benchmark_node')

        # ── Parameters (from YAML only) ───────────────────────────
        self.declare_parameter(
            'output_csv_path', '')
        self.declare_parameter(
            'flush_interval_sec', 0.0)
        self.declare_parameter(
            'ram_monitor_interval_sec', 0.0)

        self.csv_path   = self.get_parameter(
            'output_csv_path').value
        flush_sec       = self.get_parameter(
            'flush_interval_sec').value
        self.ram_interval = self.get_parameter(
            'ram_monitor_interval_sec').value

        # ── Validate ──────────────────────────────────────────────
        if not self.csv_path:
            raise ValueError(
                'output_csv_path not set! '
                'Check dms_params.yaml')
        if flush_sec <= 0.0:
            raise ValueError(
                'flush_interval_sec not set! '
                'Check dms_params.yaml')
        if self.ram_interval <= 0.0:
            raise ValueError(
                'ram_monitor_interval_sec not set! '
                'Check dms_params.yaml')

        # ── Thread-safe storage ───────────────────────────────────
        self._lock           = threading.Lock()
        self.model_name      = "unknown"
        self.latencies       = []
        self.label_counts    = defaultdict(int)
        self.parser_failures = 0
        self.total_inferences = 0
        self.timestamps      = []
        self.peak_ram_mb     = 0
        self.current_ram_mb  = 0

        # ── Non-blocking queue ────────────────────────────────────
        self._metric_queue = queue.Queue()

        # ── Background threads ────────────────────────────────────
        # Thread 1: Process incoming metrics
        self._metric_worker = threading.Thread(
            target=self._process_metric_queue,
            name="metric_worker",
            daemon=True
        )
        self._metric_worker.start()

        # Thread 2: Monitor RAM
        self._ram_worker = threading.Thread(
            target=self._monitor_ram,
            name="ram_monitor",
            daemon=True
        )
        self._ram_worker.start()

        # ── ROS subscriber ────────────────────────────────────────
        self.sub = self.create_subscription(
            String,
            '/benchmark/raw_metrics',
            self.on_metric,
            50
        )

        # ── Timer: flush CSV periodically ─────────────────────────
        self.timer = self.create_timer(
            flush_sec, self.flush_summary)

        self._init_csv()
        self.get_logger().info(
            f'✅ Post benchmark node ready '
            f'CSV={self.csv_path} '
            f'flush={flush_sec}s')

    # ── CSV initialization ────────────────────────────────────────

    def _init_csv(self):
        os.makedirs(
            os.path.dirname(self.csv_path)
            if os.path.dirname(self.csv_path) else '.',
            exist_ok=True
        )
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp',
                    'model',
                    'total_inferences',
                    'avg_latency_ms',
                    'min_latency_ms',
                    'max_latency_ms',
                    'std_latency_ms',
                    'most_frequent_state',
                    'state_distribution',
                    'peak_ram_mb',
                    'current_ram_mb',
                    'invocations_per_minute',
                    'parser_failure_rate_pct',
                    'parser_failures',
                ])

    # ── ROS callback: NON-BLOCKING ────────────────────────────────

    def on_metric(self, msg):
        """
        Called every time inference_node publishes metric.
        Only puts data in queue → returns immediately.
        Never blocks inference pipeline!
        """
        self._metric_queue.put(msg.data)

    # ── Background Thread 1: Process metric queue ─────────────────

    def _process_metric_queue(self):
        """Runs in background thread"""
        while True:
            try:
                data = self._metric_queue.get(timeout=1.0)
                self._parse_and_store(data)
            except queue.Empty:
                continue
            except Exception:
                pass

    def _parse_and_store(self, data: str):
        """
        Parse metric message and store values.
        Format: model|latency_ms|label|raw|parser_failed
        """
        parts = data.split('|')
        if len(parts) < 5:
            return

        model         = parts[0]
        try:
            latency   = int(parts[1])
        except ValueError:
            return
        label         = parts[2]
        parser_failed = parts[4] == '1'

        with self._lock:
            self.model_name = model
            self.latencies.append(latency)
            self.label_counts[label] += 1
            self.total_inferences += 1
            self.timestamps.append(time.time())
            if parser_failed:
                self.parser_failures += 1

    # ── Background Thread 2: Monitor RAM ─────────────────────────

    def _monitor_ram(self):
        """Continuously monitors RAM using jtop or fallback"""
        jtop_available = False

        try:
            from jtop import jtop
            self.get_logger().info(
                'Using jtop for RAM monitoring')
            jtop_available = True

            with jtop() as jetson:
                while True:
                    try:
                        ram_used = jetson.stats.get('RAM', 0)
                        with self._lock:
                            self.current_ram_mb = ram_used
                            if ram_used > self.peak_ram_mb:
                                self.peak_ram_mb = ram_used
                    except Exception:
                        pass
                    time.sleep(self.ram_interval)

        except Exception:
            self.get_logger().warn(
                'jtop not available → using /proc/meminfo')

        if not jtop_available:
            while True:
                ram_used = get_ram_mb_fallback()
                with self._lock:
                    self.current_ram_mb = ram_used
                    if ram_used > self.peak_ram_mb:
                        self.peak_ram_mb = ram_used
                time.sleep(self.ram_interval)

    # ── Frequency Calculation ─────────────────────────────────────

    def _calculate_frequency(self) -> float:
        """Calculate VLM invocation frequency per minute"""
        if not self.timestamps:
            return 0.0
        now    = time.time()
        window = 60.0
        recent = [t for t in self.timestamps
                  if now - t <= window]
        return len(recent) * (60.0 / window)

    # ── CSV Flush ─────────────────────────────────────────────────

    def flush_summary(self):
        """
        Called by ROS timer every flush_interval_sec.
        Launches CSV write in background thread.
        Never blocks ROS spin loop!
        """
        t = threading.Thread(
            target=self._write_csv,
            name="csv_writer",
            daemon=True
        )
        t.start()

    def _write_csv(self):
        """Compute stats and write to CSV"""
        with self._lock:
            if not self.latencies:
                return

            n   = len(self.latencies)
            avg = sum(self.latencies) / n
            mn  = min(self.latencies)
            mx  = max(self.latencies)
            std = (sum(
                (x - avg)**2 for x in self.latencies
            ) / n) ** 0.5

            # Parser failure rate
            failure_rate = (
                self.parser_failures / n * 100
            ) if n > 0 else 0.0

            # Invocation frequency
            freq = self._calculate_frequency()

            # ── Most frequent state ───────────────────────────────
            most_frequent = max(
                self.label_counts,
                key=self.label_counts.get
            ) if self.label_counts else 'UNKNOWN'

            # ── Human readable state distribution ─────────────────
            # e.g. "NORMAL:5(50%) | DROWSY:3(30%) | ..."
            total_labels = sum(self.label_counts.values())
            distribution = " | ".join([
                f"{state}:{self.label_counts.get(state, 0)}"
                f"({self.label_counts.get(state, 0) / total_labels * 100:.0f}%)"
                for state in [
                    'NORMAL',
                    'DROWSY',
                    'LOOKING_AWAY',
                    'PHONE_USAGE'
                ]
            ]) if total_labels > 0 else "no data"

            peak_ram    = self.peak_ram_mb
            current_ram = self.current_ram_mb
            model       = self.model_name
            failures    = self.parser_failures

        # Write to CSV (outside lock)
        try:
            with open(self.csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    time.strftime('%Y-%m-%d %H:%M:%S'),
                    model,
                    n,
                    f'{avg:.0f}',
                    f'{mn:.0f}',
                    f'{mx:.0f}',
                    f'{std:.0f}',
                    most_frequent,
                    distribution,
                    f'{peak_ram:.0f}',
                    f'{current_ram:.0f}',
                    f'{freq:.2f}',
                    f'{failure_rate:.1f}',
                    failures,
                ])

            self.get_logger().debug(
                f'💾 [{model}] '
                f'n={n} '
                f'avg={avg:.0f}ms '
                f'state={most_frequent} '
                f'dist=[{distribution}]')

        except Exception as e:
            self.get_logger().error(
                f'CSV write failed: {e}')


def main(args=None):
    rclpy.init(args=args)
    try:
        rclpy.spin(PostBenchmarkNode())
    except KeyboardInterrupt:
        pass
    except ValueError as e:
        print(f'Post benchmark error: {e}')
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()

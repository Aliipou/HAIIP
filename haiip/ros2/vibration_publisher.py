"""
ROS2 Vibration Publisher Node
==============================
Publishes haiip_msgs/VibrationReading at 50 Hz.
Falls back to sensor_msgs/Imu if haiip_msgs is not built yet.
Runs standalone (no ROS2) via StandaloneVibrationPublisher.

ROS2 usage:
    ros2 run haiip vibration_publisher \\
        --ros-args -p machine_id:=pump-01 -p hz:=50 -p fault_mode:=false

Standalone:
    python -m haiip.ros2.vibration_publisher
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# ROS2 import — degrade gracefully
# ---------------------------------------------------------------------------
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

    # Prefer custom message; fall back to Imu if haiip_msgs not built
    try:
        from haiip_msgs.msg import VibrationReading as _VibMsg
        _MSG_TYPE = "haiip_msgs"
    except ImportError:
        from sensor_msgs.msg import Imu as _VibMsg          # type: ignore
        _MSG_TYPE = "sensor_msgs"

    _ROS2 = True
except ImportError:
    _ROS2 = False
    _MSG_TYPE = "standalone"


# ---------------------------------------------------------------------------
# Signal model
# ---------------------------------------------------------------------------

@dataclass
class MachineState:
    machine_id:          str   = "pump-01"
    rotational_speed_rpm: float = 1450.0
    torque_nm:           float = 22.0
    fault_mode:          bool  = False
    _t: float = field(default=0.0, init=False, repr=False)

    def next_sample(self, dt: float = 0.02) -> dict:
        self._t += dt
        omega    = 2 * math.pi * self.rotational_speed_rpm / 60
        base     = 0.15 * math.sin(omega * self._t)

        if self.fault_mode:
            # Bearing outer-race defect harmonic
            bpfo = 3.5 * omega
            base += random.gauss(0.8, 0.1) * math.sin(bpfo * self._t)

        n = lambda s: random.gauss(0, s)
        vib_x = base + n(0.02)
        vib_y = 0.6 * base + n(0.015)
        vib_z = 0.3 * base + n(0.01)
        return {
            "machine_id":       self.machine_id,
            "vib_x":            vib_x,
            "vib_y":            vib_y,
            "vib_z":            vib_z,
            "vib_rms":          (vib_x**2 + vib_y**2 + vib_z**2) ** 0.5,
            "rotational_speed": self.rotational_speed_rpm + n(5),
            "torque":           self.torque_nm + n(0.5),
            "fault_injected":   self.fault_mode,
            "timestamp":        time.time(),
        }


# ---------------------------------------------------------------------------
# ROS2 Node
# ---------------------------------------------------------------------------

if _ROS2:
    _SENSOR_QOS = QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        history=HistoryPolicy.KEEP_LAST,
        depth=10,
    )

    class VibrationPublisherNode(Node):
        def __init__(self) -> None:
            super().__init__("haiip_vibration_publisher")
            self.declare_parameter("machine_id",  "pump-01")
            self.declare_parameter("hz",          50)
            self.declare_parameter("fault_mode",  False)

            machine_id = self.get_parameter("machine_id").value
            hz         = self.get_parameter("hz").value
            fault_mode = self.get_parameter("fault_mode").value

            self._state = MachineState(machine_id=machine_id, fault_mode=fault_mode)
            topic       = f"/haiip/vibration/{machine_id}"

            self._pub   = self.create_publisher(_VibMsg, topic, _SENSOR_QOS)
            self._timer = self.create_timer(1.0 / hz, self._publish)

            self.get_logger().info(
                f"VibrationPublisher  machine={machine_id}  topic={topic}  "
                f"hz={hz}  msg={_MSG_TYPE}  fault={fault_mode}"
            )

        def _publish(self) -> None:
            s   = self._state.next_sample()
            msg = _VibMsg()
            msg.header.stamp    = self.get_clock().now().to_msg()
            msg.header.frame_id = s["machine_id"]

            if _MSG_TYPE == "haiip_msgs":
                msg.machine_id        = s["machine_id"]
                msg.vib_x             = s["vib_x"]
                msg.vib_y             = s["vib_y"]
                msg.vib_z             = s["vib_z"]
                msg.vib_rms           = s["vib_rms"]
                msg.rotational_speed  = s["rotational_speed"]
                msg.torque            = s["torque"]
                msg.fault_injected    = s["fault_injected"]
            else:
                # sensor_msgs/Imu fallback — pack into linear_acceleration + covariance
                msg.linear_acceleration.x = s["vib_x"]
                msg.linear_acceleration.y = s["vib_y"]
                msg.linear_acceleration.z = s["vib_z"]
                cov = list(msg.linear_acceleration_covariance)
                cov[0] = s["rotational_speed"]
                cov[1] = s["torque"]
                cov[2] = s["vib_rms"]
                msg.linear_acceleration_covariance = cov

            self._pub.publish(msg)


# ---------------------------------------------------------------------------
# Standalone publisher (asyncio, no ROS2)
# ---------------------------------------------------------------------------

class StandaloneVibrationPublisher:
    def __init__(
        self,
        machine_id: str  = "pump-01",
        hz:         int  = 50,
        fault_mode: bool = False,
        on_sample         = None,
    ) -> None:
        self._state    = MachineState(machine_id=machine_id, fault_mode=fault_mode)
        self._hz       = hz
        self._callback = on_sample or self._print

    @staticmethod
    def _print(s: dict) -> None:
        print(
            f"[VIB] {s['machine_id']}  "
            f"x={s['vib_x']:+.3f}  y={s['vib_y']:+.3f}  z={s['vib_z']:+.3f}  "
            f"rms={s['vib_rms']:.3f}  rpm={s['rotational_speed']:.0f}"
        )

    def run(self) -> None:
        import asyncio
        asyncio.run(self._loop())

    async def _loop(self) -> None:
        import asyncio
        interval = 1.0 / self._hz
        print(f"[StandalonePublisher] machine={self._state.machine_id}  "
              f"hz={self._hz}  fault={self._state.fault_mode}")
        try:
            while True:
                sample = self._state.next_sample(dt=interval)
                self._callback(sample)
                await asyncio.sleep(interval)
        except (KeyboardInterrupt, asyncio.CancelledError):
            pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    if _ROS2:
        rclpy.init()
        node = VibrationPublisherNode()
        try:
            rclpy.spin(node)
        finally:
            node.destroy_node()
            rclpy.shutdown()
    else:
        print("[haiip.ros2] rclpy not found — standalone mode.")
        StandaloneVibrationPublisher(fault_mode=False).run()


if __name__ == "__main__":
    main()

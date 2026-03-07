"""
Human Override Interface  —  EU AI Act Article 14
==================================================
Publishes haiip_msgs/HumanOverride to /haiip/override/{machine_id}.

Valid commands: STOP | SLOW_DOWN | MONITOR | NOMINAL | RELEASE

ROS2:
    python -m haiip.ros2.human_override --machine pump-01 --command STOP
    ros2 topic pub --once /haiip/override/pump-01 haiip_msgs/HumanOverride \\
        "{machine_id: pump-01, command: STOP, reason: smoke}"

Standalone (interactive console during pipeline run):
    s=STOP  d=SLOW_DOWN  m=MONITOR  n=NOMINAL  r=RELEASE  q=quit
"""

from __future__ import annotations

import asyncio
import time

VALID = {"STOP", "SLOW_DOWN", "MONITOR", "NOMINAL", "RELEASE"}

try:
    import rclpy
    from rclpy.node import Node

    try:
        from haiip_msgs.msg import HumanOverride as HumanOverrideMsg

        _MSG = "haiip_msgs"
    except ImportError:
        import json

        from std_msgs.msg import String as HumanOverrideMsg  # type: ignore

        _MSG = "std_msgs"
    _ROS2 = True
except ImportError:
    _ROS2 = False


def make_override(
    machine_id: str, command: str, reason: str = "", operator_id: str = "operator"
) -> dict:
    if command not in VALID:
        raise ValueError(f"Invalid command '{command}'. Valid: {VALID}")
    return {
        "machine_id": machine_id,
        "command": command,
        "reason": reason or f"Operator: {command}",
        "operator_id": operator_id,
        "timestamp": time.time(),
    }


if _ROS2:

    class HumanOverridePublisher(Node):
        """One-shot node — publishes one override and exits."""

        def __init__(self, machine_id: str, command: str, reason: str = "") -> None:
            super().__init__("haiip_human_override")
            override = make_override(machine_id, command, reason)
            pub = self.create_publisher(HumanOverrideMsg, f"/haiip/override/{machine_id}", 10)
            self._done = False

            def _publish():
                if self._done:
                    return
                if _MSG == "haiip_msgs":
                    msg = HumanOverrideMsg()
                    msg.header.stamp = self.get_clock().now().to_msg()
                    msg.header.frame_id = machine_id
                    msg.machine_id = override["machine_id"]
                    msg.command = override["command"]
                    msg.reason = override["reason"]
                    msg.operator_id = override["operator_id"]
                else:
                    msg = HumanOverrideMsg()
                    msg.data = json.dumps(override)

                pub.publish(msg)
                self.get_logger().warn(
                    f"Override: machine={machine_id}  cmd={command}  reason={reason}"
                )
                self._done = True

            self.create_timer(0.3, _publish)


class StandaloneHumanOverride:
    """Interactive console that feeds override dicts into an asyncio.Queue."""

    KEYS = {
        "s": "STOP",
        "d": "SLOW_DOWN",
        "m": "MONITOR",
        "n": "NOMINAL",
        "r": "RELEASE",
    }

    def __init__(self, machine_id: str, queue: asyncio.Queue) -> None:
        self._machine_id = machine_id
        self._queue = queue

    async def run(self) -> None:
        print(
            "\n[HumanOverride] EU AI Act Art.14 compliant console\n"
            "  s=STOP  d=SLOW_DOWN  m=MONITOR  n=NOMINAL  r=RELEASE  q=quit\n"
        )
        loop = asyncio.get_event_loop()
        while True:
            key = await loop.run_in_executor(None, lambda: input("Override> ").strip().lower())
            if key == "q":
                return
            cmd = self.KEYS.get(key)
            if not cmd:
                print(f"  Unknown '{key}'. Try: {list(self.KEYS)}")
                continue
            ov = make_override(self._machine_id, cmd)
            await self._queue.put(ov)
            print(f"  -> {cmd} queued")


def main() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--machine", default="pump-01")
    p.add_argument("--command", default="STOP", choices=list(VALID))
    p.add_argument("--reason", default="")
    args = p.parse_args()

    if _ROS2:
        rclpy.init()
        node = HumanOverridePublisher(args.machine, args.command, args.reason)
        try:
            rclpy.spin_once(node, timeout_sec=2.0)
        finally:
            node.destroy_node()
            rclpy.shutdown()
    else:
        import json

        print(json.dumps(make_override(args.machine, args.command, args.reason), indent=2))


if __name__ == "__main__":
    main()

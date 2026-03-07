"""
ROS2 Action Node
=================
Subscribes to /haiip/decision/{machine_id} and /haiip/override/{machine_id},
publishes haiip_msgs/MachineCommand to /haiip/command/{machine_id}.

Action → Command mapping:
    repair_now → STOP
    schedule   → SLOW_DOWN
    monitor    → MONITOR
    ignore     → NOMINAL

Human override always wins. Override auto-expires after TTL.
"""

from __future__ import annotations

import asyncio
import json
import time

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

    try:
        from haiip_msgs.msg import EconomicDecision, HumanOverride, MachineCommand

        _MSG = "haiip_msgs"
    except ImportError:
        from std_msgs.msg import String as EconomicDecision  # type: ignore
        from std_msgs.msg import String as HumanOverride  # type: ignore
        from std_msgs.msg import String as MachineCommand  # type: ignore

        _MSG = "std_msgs"
    _ROS2 = True
except ImportError:
    _ROS2 = False


ACTION_TO_CMD = {
    "repair_now": "STOP",
    "schedule": "SLOW_DOWN",
    "monitor": "MONITOR",
    "ignore": "NOMINAL",
}

_COLORS = {
    "STOP": "\033[91m",
    "SLOW_DOWN": "\033[93m",
    "MONITOR": "\033[94m",
    "NOMINAL": "\033[92m",
}
_RST = "\033[0m"


def make_command(decision: dict, override: dict | None, ttl: float = 600.0) -> dict:
    if override and (time.time() - override.get("timestamp", 0)) < ttl:
        cmd = override["command"] if override["command"] != "RELEASE" else "NOMINAL"
        remaining = max(0.0, ttl - (time.time() - override["timestamp"]))
        return {
            "machine_id": decision["machine_id"],
            "command": cmd,
            "source": "human",
            "reason": override.get("reason", "operator override"),
            "override": True,
            "override_ttl_remaining": round(remaining, 1),
        }
    cmd = ACTION_TO_CMD.get(decision.get("action", "ignore"), "NOMINAL")
    return {
        "machine_id": decision["machine_id"],
        "command": cmd,
        "source": "ai",
        "reason": (
            f"{decision.get('action', '?')}  "
            f"EUR {decision.get('net_benefit_eur', 0):,.0f}  "
            f"review={'yes' if decision.get('requires_human_review') else 'no'}"
        ),
        "override": False,
        "override_ttl_remaining": 0.0,
    }


def print_command(cmd: dict) -> None:
    color = _COLORS.get(cmd["command"], "")
    src = "[HUMAN]" if cmd["override"] else "[AI]   "
    print(
        f"{color}[Action] {cmd['machine_id']}  >> {cmd['command']:10s}  "
        f"{src}  {cmd['reason']}{_RST}"
    )


if _ROS2:
    _QOS = QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        history=HistoryPolicy.KEEP_LAST,
        depth=10,
    )

    class ActionNode(Node):
        def __init__(self) -> None:
            super().__init__("haiip_action_node")
            self.declare_parameter("machine_ids", ["pump-01"])
            self.declare_parameter("override_ttl_sec", 600.0)

            self._ttl = self.get_parameter("override_ttl_sec").value
            self._overrides: dict[str, dict | None] = {}
            self._pubs: dict[str, object] = {}

            for mid in list(self.get_parameter("machine_ids").value):
                self._overrides[mid] = None
                self._pubs[mid] = self.create_publisher(
                    MachineCommand, f"/haiip/command/{mid}", _QOS
                )
                self.create_subscription(
                    EconomicDecision,
                    f"/haiip/decision/{mid}",
                    lambda msg, m=mid: self._on_decision(msg, m),
                    _QOS,
                )
                self.create_subscription(
                    HumanOverride,
                    f"/haiip/override/{mid}",
                    lambda msg, m=mid: self._on_override(msg, m),
                    _QOS,
                )
            self.get_logger().info(f"ActionNode ready  ttl={self._ttl}s  msg_type={_MSG}")

        def _on_decision(self, msg, machine_id: str) -> None:
            if _MSG == "haiip_msgs":
                dec = {
                    "machine_id": msg.machine_id,
                    "action": msg.action,
                    "net_benefit_eur": msg.net_benefit_eur,
                    "requires_human_review": msg.requires_human_review,
                    "explanation": msg.explanation,
                }
            else:
                dec = json.loads(msg.data)

            cmd = make_command(dec, self._overrides.get(machine_id), self._ttl)
            print_command(cmd)
            self._publish_cmd(machine_id, cmd)

        def _on_override(self, msg, machine_id: str) -> None:
            if _MSG == "haiip_msgs":
                ov = {
                    "machine_id": msg.machine_id,
                    "command": msg.command,
                    "reason": msg.reason,
                    "timestamp": time.time(),
                }
            else:
                ov = {**json.loads(msg.data), "timestamp": time.time()}

            if ov["command"] == "RELEASE":
                self._overrides[machine_id] = None
                self.get_logger().info(f"Override released for {machine_id} — AI resumed")
            else:
                self._overrides[machine_id] = ov
                self.get_logger().warn(
                    f"Override SET  {machine_id}  cmd={ov['command']}  reason={ov['reason']}"
                )

        def _publish_cmd(self, machine_id: str, cmd: dict) -> None:
            if _MSG == "haiip_msgs":
                out = MachineCommand()
                out.header.stamp = self.get_clock().now().to_msg()
                out.header.frame_id = machine_id
                out.machine_id = cmd["machine_id"]
                out.command = cmd["command"]
                out.source = cmd["source"]
                out.reason = cmd["reason"]
                out.override = cmd["override"]
                out.override_ttl_remaining = cmd["override_ttl_remaining"]
            else:
                out = MachineCommand()
                out.data = json.dumps(cmd)
            self._pubs[machine_id].publish(out)


async def action_coroutine(
    decision_queue: asyncio.Queue,
    command_queue: asyncio.Queue,
    override_queue: asyncio.Queue,
    ttl: float = 600.0,
) -> None:
    overrides: dict[str, dict | None] = {}
    while True:
        while not override_queue.empty():
            ov = override_queue.get_nowait()
            mid = ov["machine_id"]
            if ov.get("command") == "RELEASE":
                overrides[mid] = None
            else:
                overrides[mid] = {**ov, "timestamp": time.time()}

        dec = await decision_queue.get()
        mid = dec["machine_id"]
        cmd = make_command(dec, overrides.get(mid), ttl)
        print_command(cmd)
        await command_queue.put(cmd)


def main() -> None:
    if _ROS2:
        rclpy.init()
        node = ActionNode()
        try:
            rclpy.spin(node)
        finally:
            node.destroy_node()
            rclpy.shutdown()
    else:
        print("[ActionNode] rclpy not found. Use pipeline.py for standalone mode.")


if __name__ == "__main__":
    main()

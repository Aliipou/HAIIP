"""
ROS2 Economic Decision Node
============================
Subscribes to /haiip/ai/{machine_id} (AIPrediction),
runs EconomicDecisionEngine in-process (< 1 ms),
publishes haiip_msgs/EconomicDecision to /haiip/decision/{machine_id}.
"""

from __future__ import annotations

import asyncio
import json
import time

from haiip.core.economic_ai import CostProfile, EconomicDecisionEngine

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

    try:
        from haiip_msgs.msg import AIPrediction
        from haiip_msgs.msg import EconomicDecision as EconDec

        _MSG = "haiip_msgs"
    except ImportError:
        from std_msgs.msg import String as AIPrediction  # type: ignore
        from std_msgs.msg import String as EconDec

        _MSG = "std_msgs"
    _ROS2 = True
except ImportError:
    _ROS2 = False


NORDIC_SME = CostProfile(
    production_rate_eur_hr=500.0,
    downtime_hours_avg=8.0,
    labour_rate_eur_hr=85.0,
    labour_hours_avg=4.0,
    parts_cost_eur=250.0,
    opportunity_cost_eur=150.0,
    safety_factor=1.5,
    urgency_factor=0.7,
)


def run_economic(ai: dict) -> dict:
    engine = EconomicDecisionEngine(cost_profile=NORDIC_SME)
    decision = engine.decide(
        anomaly_score=ai.get("anomaly_score", 0.0),
        failure_probability=ai.get("failure_probability", 0.0),
        confidence=ai.get("confidence", 0.8),
        machine_id=ai.get("machine_id"),
    )
    return {
        "machine_id": ai["machine_id"],
        "action": decision.action.value,
        "net_benefit_eur": round(decision.net_benefit, 2),
        "expected_cost_wait": round(decision.expected_cost_wait, 2),
        "expected_cost_action": round(decision.expected_cost_action, 2),
        "confidence": round(decision.confidence, 4),
        "requires_human_review": decision.requires_human_review,
        "explanation": decision.explanation,
        "timestamp": time.time(),
    }


if _ROS2:
    _QOS = QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        history=HistoryPolicy.KEEP_LAST,
        depth=10,
    )

    class EconomicNode(Node):
        def __init__(self) -> None:
            super().__init__("haiip_economic_node")
            self.declare_parameter("machine_ids", ["pump-01"])

            self._pubs: dict[str, object] = {}
            for mid in list(self.get_parameter("machine_ids").value):
                self._pubs[mid] = self.create_publisher(EconDec, f"/haiip/decision/{mid}", _QOS)
                self.create_subscription(
                    AIPrediction,
                    f"/haiip/ai/{mid}",
                    lambda msg, m=mid: self._on_ai(msg, m),
                    _QOS,
                )
            self.get_logger().info(f"EconomicNode ready  msg_type={_MSG}")

        def _on_ai(self, msg, machine_id: str) -> None:
            if _MSG == "haiip_msgs":
                ai = {
                    "machine_id": msg.machine_id,
                    "anomaly_score": msg.anomaly_score,
                    "failure_probability": msg.failure_probability,
                    "confidence": msg.confidence,
                }
            else:
                ai = json.loads(msg.data)

            dec = run_economic(ai)
            self.get_logger().info(
                f"[Economic] {machine_id}  {dec['action'].upper():12s}  "
                f"EUR {dec['net_benefit_eur']:,.0f}  "
                f"review={dec['requires_human_review']}"
            )

            if _MSG == "haiip_msgs":
                out = EconDec()
                out.header.stamp = self.get_clock().now().to_msg()
                out.header.frame_id = machine_id
                out.machine_id = dec["machine_id"]
                out.action = dec["action"]
                out.net_benefit_eur = dec["net_benefit_eur"]
                out.expected_cost_wait = dec["expected_cost_wait"]
                out.expected_cost_action = dec["expected_cost_action"]
                out.confidence = dec["confidence"]
                out.requires_human_review = dec["requires_human_review"]
                out.explanation = dec["explanation"]
            else:
                out = EconDec()
                out.data = json.dumps(dec)

            self._pubs[machine_id].publish(out)


async def economic_coroutine(
    ai_queue: asyncio.Queue,
    decision_queue: asyncio.Queue,
) -> None:
    while True:
        ai = await ai_queue.get()
        dec = run_economic(ai)
        await decision_queue.put(dec)


def main() -> None:
    if _ROS2:
        rclpy.init()
        node = EconomicNode()
        try:
            rclpy.spin(node)
        finally:
            node.destroy_node()
            rclpy.shutdown()
    else:
        print("[EconomicNode] rclpy not found. Use pipeline.py for standalone mode.")


if __name__ == "__main__":
    main()

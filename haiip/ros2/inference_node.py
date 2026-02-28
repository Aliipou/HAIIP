"""
ROS2 AI Inference Node
=======================
Subscribes to /haiip/vibration/{machine_id},
calls HAIIP /api/v1/predict (throttled),
publishes haiip_msgs/AIPrediction to /haiip/ai/{machine_id}.
"""

from __future__ import annotations

import asyncio
import json
import time

import httpx

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    try:
        from haiip_msgs.msg import VibrationReading, AIPrediction as AIPred
        _MSG = "haiip_msgs"
    except ImportError:
        from sensor_msgs.msg import Imu as VibrationReading          # type: ignore
        from std_msgs.msg import String as AIPred                    # type: ignore
        _MSG = "std_msgs"
    _ROS2 = True
except ImportError:
    _ROS2 = False


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _sample_from_haiip_msg(msg, msg_type: str) -> dict:
    if msg_type == "haiip_msgs":
        return {
            "machine_id":       msg.machine_id,
            "vib_x":            msg.vib_x,
            "vib_y":            msg.vib_y,
            "vib_z":            msg.vib_z,
            "vib_rms":          msg.vib_rms,
            "rotational_speed": msg.rotational_speed,
            "torque":           msg.torque,
        }
    else:  # sensor_msgs/Imu fallback
        cov = list(msg.linear_acceleration_covariance)
        rms = (msg.linear_acceleration.x**2 +
               msg.linear_acceleration.y**2 +
               msg.linear_acceleration.z**2) ** 0.5
        return {
            "machine_id":       msg.header.frame_id,
            "vib_x":            msg.linear_acceleration.x,
            "vib_y":            msg.linear_acceleration.y,
            "vib_z":            msg.linear_acceleration.z,
            "vib_rms":          rms,
            "rotational_speed": cov[0],
            "torque":           cov[1],
        }


def _build_sensor_reading(sample: dict) -> dict:
    """Convert vibration sample to HAIIP SensorReading schema."""
    return {
        "machine_id":          sample["machine_id"],
        "air_temperature":     22.0,
        "process_temperature": 22.0 + sample["vib_rms"] * 1.5,
        "rotational_speed":    sample.get("rotational_speed", 1450.0),
        "torque":              sample.get("torque", 22.0),
        "tool_wear":           0.0,
        "extra_features": {
            "vib_x":   round(sample["vib_x"], 5),
            "vib_y":   round(sample["vib_y"], 5),
            "vib_z":   round(sample["vib_z"], 5),
            "vib_rms": round(sample["vib_rms"], 5),
            "source":  "ros2",
        },
    }


async def call_api(sample: dict, client: httpx.AsyncClient, token: str, api_url: str) -> dict | None:
    try:
        resp = await client.post(
            f"{api_url}/api/v1/predict",
            json=_build_sensor_reading(sample),
            headers={"Authorization": f"Bearer {token}"},
            timeout=8,
        )
        resp.raise_for_status()
        pred = resp.json().get("data", {})
        return {
            "machine_id":          sample["machine_id"],
            "label":               pred.get("prediction_label", "UNKNOWN"),
            "confidence":          pred.get("confidence", 0.0),
            "anomaly_score":       pred.get("anomaly_score") or 0.0,
            "failure_probability": pred.get("confidence", 0.0),
            "requires_human_review": pred.get("confidence", 1.0) < 0.35,
            "offline":             False,
        }
    except Exception as e:
        print(f"[InferenceNode] API error: {e}")
        return None


# ---------------------------------------------------------------------------
# ROS2 Node
# ---------------------------------------------------------------------------

if _ROS2:
    _QOS = QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        history=HistoryPolicy.KEEP_LAST,
        depth=10,
    )

    class InferenceNode(Node):
        def __init__(self) -> None:
            super().__init__("haiip_inference_node")
            self.declare_parameter("machine_ids",    ["pump-01"])
            self.declare_parameter("api_url",        "http://localhost:8000")
            self.declare_parameter("sample_every_n", 50)
            self.declare_parameter("tenant_slug",    "demo-sme")
            self.declare_parameter("email",          "admin@haiip.ai")
            self.declare_parameter("password",       "Demo1234!")

            self._api_url = self.get_parameter("api_url").value
            self._every_n = self.get_parameter("sample_every_n").value
            self._counts: dict[str, int] = {}
            self._token: str | None = None
            self._pubs:  dict[str, object] = {}
            self._client = httpx.AsyncClient()

            cfg = {
                "tenant_slug": self.get_parameter("tenant_slug").value,
                "email":       self.get_parameter("email").value,
                "password":    self.get_parameter("password").value,
            }

            for mid in list(self.get_parameter("machine_ids").value):
                self._counts[mid] = 0
                self._pubs[mid]   = self.create_publisher(AIPred, f"/haiip/ai/{mid}", _QOS)
                self.create_subscription(
                    VibrationReading, f"/haiip/vibration/{mid}",
                    lambda msg, m=mid: self._on_vib(msg, m), _QOS,
                )

            # Login once at startup
            self.create_timer(0.1, lambda: self._login(cfg))
            self.get_logger().info(f"InferenceNode ready  msg_type={_MSG}")

        def _login(self, cfg: dict) -> None:
            loop = asyncio.get_event_loop()
            try:
                resp = loop.run_until_complete(
                    self._client.post(
                        f"{self._api_url}/api/v1/auth/login", json=cfg, timeout=10
                    )
                )
                self._token = resp.json()["access_token"]
                self.get_logger().info("InferenceNode: authenticated")
            except Exception as e:
                self.get_logger().warn(f"InferenceNode login failed: {e}")

        def _on_vib(self, msg, machine_id: str) -> None:
            self._counts[machine_id] += 1
            if self._counts[machine_id] % self._every_n != 0:
                return
            if not self._token:
                return

            sample = _sample_from_haiip_msg(msg, _MSG)
            loop   = asyncio.get_event_loop()
            result = loop.run_until_complete(
                call_api(sample, self._client, self._token, self._api_url)
            )
            if not result:
                return

            if _MSG == "haiip_msgs":
                out = AIPred()
                out.header.stamp    = self.get_clock().now().to_msg()
                out.header.frame_id = machine_id
                out.machine_id      = result["machine_id"]
                out.label           = result["label"]
                out.confidence      = result["confidence"]
                out.anomaly_score   = result["anomaly_score"]
                out.failure_probability  = result["failure_probability"]
                out.requires_human_review = result["requires_human_review"]
                out.offline         = result["offline"]
            else:
                out = AIPred()
                out.data = json.dumps(result)

            self._pubs[machine_id].publish(out)


# ---------------------------------------------------------------------------
# Standalone coroutine
# ---------------------------------------------------------------------------

async def inference_coroutine(
    vib_queue: asyncio.Queue,
    ai_queue:  asyncio.Queue,
    api_url:   str   = "http://localhost:8000",
    tenant:    str   = "demo-sme",
    email:     str   = "admin@haiip.ai",
    password:  str   = "Demo1234!",
    every_n:   int   = 50,
) -> None:
    token = None
    count = 0
    async with httpx.AsyncClient() as client:
        try:
            r = await client.post(
                f"{api_url}/api/v1/auth/login",
                json={"tenant_slug": tenant, "email": email, "password": password},
                timeout=10,
            )
            token = r.json()["access_token"]
        except Exception as e:
            print(f"[InferenceNode] login failed: {e}")

        while True:
            sample = await vib_queue.get()
            count += 1
            if count % every_n != 0:
                continue
            result = await call_api(sample, client, token or "", api_url)
            if result:
                await ai_queue.put(result)


def main() -> None:
    if _ROS2:
        rclpy.init()
        node = InferenceNode()
        try:
            rclpy.spin(node)
        finally:
            node.destroy_node()
            rclpy.shutdown()
    else:
        print("[InferenceNode] rclpy not found. Use pipeline.py for standalone mode.")


if __name__ == "__main__":
    main()

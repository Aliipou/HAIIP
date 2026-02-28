"""
HAIIP Standalone Closed-Loop Pipeline  (no ROS2 required)
==========================================================

Full dataflow using asyncio Queues:
    VibrationPublisher -> InferenceNode -> EconomicNode -> ActionNode
                                                               ^
                                                        HumanOverride console

Usage:
    python -m haiip.ros2.pipeline --no-api          # offline, synthetic scores
    python -m haiip.ros2.pipeline                   # live HAIIP API
    python -m haiip.ros2.pipeline --fault           # fault injection
    python -m haiip.ros2.pipeline --machine fan-01 --hz 25
"""

from __future__ import annotations

import argparse
import asyncio
import random
import time


# ---------------------------------------------------------------------------
# Offline inference (no API — synthetic scores from vibration magnitude)
# ---------------------------------------------------------------------------

async def _offline_infer(vib_q: asyncio.Queue, ai_q: asyncio.Queue, every_n: int = 50) -> None:
    count = 0
    while True:
        sample = await vib_q.get()
        count += 1
        if count % every_n != 0:
            continue
        rms   = sample["vib_rms"]
        score = min(1.0, rms * 2.5 + random.gauss(0, 0.05))
        await ai_q.put({
            "machine_id":          sample["machine_id"],
            "label":               "ANOMALY" if score > 0.5 else "NORMAL",
            "confidence":          round(random.uniform(0.75, 0.97), 3),
            "anomaly_score":       round(max(0.0, score), 4),
            "failure_probability": round(max(0.0, score * 0.85), 4),
            "requires_human_review": score > 0.75,
            "offline":             True,
        })


# ---------------------------------------------------------------------------
# Command sink — final output (in production: → OPC-UA / PLC register)
# ---------------------------------------------------------------------------

async def _command_sink(cmd_q: asyncio.Queue) -> None:
    while True:
        cmd = await cmd_q.get()
        if cmd["command"] in ("STOP", "SLOW_DOWN"):
            print(f"\n  [!] ACTUATION  {cmd['machine_id']} -> {cmd['command']}"
                  f"  source={cmd['source']}  override={cmd['override']}\n")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

async def run_pipeline(
    machine_id: str  = "pump-01",
    hz:         int  = 50,
    fault:      bool = False,
    use_api:    bool = True,
) -> None:
    from haiip.ros2.vibration_publisher import StandaloneVibrationPublisher
    from haiip.ros2.inference_node      import inference_coroutine
    from haiip.ros2.economic_node       import economic_coroutine
    from haiip.ros2.action_node         import action_coroutine
    from haiip.ros2.human_override      import StandaloneHumanOverride

    vib_q      : asyncio.Queue = asyncio.Queue(maxsize=500)
    ai_q       : asyncio.Queue = asyncio.Queue(maxsize=100)
    decision_q : asyncio.Queue = asyncio.Queue(maxsize=100)
    command_q  : asyncio.Queue = asyncio.Queue(maxsize=100)
    override_q : asyncio.Queue = asyncio.Queue(maxsize=10)

    pub = StandaloneVibrationPublisher(
        machine_id=machine_id, hz=hz, fault_mode=fault,
        on_sample=vib_q.put_nowait,
    )

    infer = (
        inference_coroutine(vib_q, ai_q, every_n=hz)
        if use_api
        else _offline_infer(vib_q, ai_q, every_n=hz)
    )

    print(
        f"\n{'='*58}\n"
        f"  HAIIP Closed-Loop Pipeline\n"
        f"  machine={machine_id}  hz={hz}  fault={fault}  api={'on' if use_api else 'off'}\n"
        f"{'='*58}\n"
        f"  /haiip/vibration/{machine_id}  ->  AI  ->  Economic  ->  Command\n"
        f"  Human override: s=STOP  d=SLOW_DOWN  m=MONITOR  r=RELEASE  q=quit\n"
        f"{'='*58}\n"
    )

    tasks = [
        asyncio.create_task(pub._loop(),              name="publisher"),
        asyncio.create_task(infer,                    name="inference"),
        asyncio.create_task(
            economic_coroutine(ai_q, decision_q),     name="economic"),
        asyncio.create_task(
            action_coroutine(decision_q, command_q, override_q),
                                                      name="action"),
        asyncio.create_task(_command_sink(command_q), name="sink"),
        asyncio.create_task(
            StandaloneHumanOverride(machine_id, override_q).run(),
                                                      name="override"),
    ]

    try:
        await asyncio.gather(*tasks)
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\n[Pipeline] Stopped.")
    finally:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


def main() -> None:
    p = argparse.ArgumentParser(description="HAIIP Closed-Loop Pipeline")
    p.add_argument("--machine", default="pump-01")
    p.add_argument("--hz",      type=int, default=50)
    p.add_argument("--fault",   action="store_true")
    p.add_argument("--no-api",  action="store_true")
    args = p.parse_args()
    asyncio.run(run_pipeline(
        machine_id=args.machine,
        hz=args.hz,
        fault=args.fault,
        use_api=not args.no_api,
    ))


if __name__ == "__main__":
    main()

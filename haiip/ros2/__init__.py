"""
haiip.ros2 — Human-Aligned ROS2 Closed-Loop Industrial System
==============================================================

Build (with ROS2 installed):
    colcon build --packages-select haiip_msgs haiip
    source install/setup.bash

Run full pipeline (no ROS2 needed):
    python -m haiip.ros2.pipeline --no-api        # offline
    python -m haiip.ros2.pipeline                 # live API
    python -m haiip.ros2.pipeline --fault         # fault injection

Run with ROS2:
    ros2 launch haiip haiip_closed_loop.launch.py machine_id:=pump-01

Topics
------
/haiip/vibration/{machine_id}   haiip_msgs/VibrationReading   50 Hz
/haiip/ai/{machine_id}          haiip_msgs/AIPrediction        ~1 Hz
/haiip/decision/{machine_id}    haiip_msgs/EconomicDecision    ~1 Hz
/haiip/command/{machine_id}     haiip_msgs/MachineCommand      ~1 Hz
/haiip/override/{machine_id}    haiip_msgs/HumanOverride       on demand

Closed-loop dataflow:
    VibrationPublisher
        -> InferenceNode   (calls /api/v1/predict)
        -> EconomicNode    (EconomicDecisionEngine, in-process)
        -> ActionNode      (maps decision to STOP/SLOW_DOWN/MONITOR/NOMINAL)
        ^
        HumanOverride      (EU AI Act Art. 14 — TTL auto-expiry)
"""

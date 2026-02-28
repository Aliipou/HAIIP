"""
ROS2 Python package setup for HAIIP.

Build with colcon:
    cd ~/ros2_ws
    colcon build --packages-select haiip haiip_msgs
    source install/setup.bash

Run nodes:
    ros2 run haiip vibration_publisher --ros-args -p machine_id:=pump-01
    ros2 run haiip inference_node
    ros2 run haiip economic_node
    ros2 run haiip action_node
    ros2 run haiip human_override -- --machine pump-01 --command STOP

Launch full pipeline:
    ros2 launch haiip haiip_closed_loop.launch.py machine_id:=pump-01
"""

import os
from glob import glob
from setuptools import find_packages, setup

package_name = "haiip"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        # ROS2 resource index marker — required for discovery
        (
            "share/ament_index/resource_index/packages",
            [f"resource/{package_name}"],
        ),
        # Package manifest
        (f"share/{package_name}", ["package.xml"]),
        # Launch files
        (
            os.path.join("share", package_name, "launch"),
            glob("haiip/ros2/launch/*.py"),
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="NextIndustriAI Team",
    maintainer_email="admin@haiip.ai",
    description="HAIIP — Human-Aligned Industrial Intelligence Platform (ROS2 nodes)",
    license="Proprietary",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            # ros2 run haiip vibration_publisher
            "vibration_publisher = haiip.ros2.vibration_publisher:main",
            # ros2 run haiip inference_node
            "inference_node      = haiip.ros2.inference_node:main",
            # ros2 run haiip economic_node
            "economic_node       = haiip.ros2.economic_node:main",
            # ros2 run haiip action_node
            "action_node         = haiip.ros2.action_node:main",
            # ros2 run haiip human_override
            "human_override      = haiip.ros2.human_override:main",
            # ros2 run haiip pipeline  (standalone asyncio, no ROS2 required)
            "pipeline            = haiip.ros2.pipeline:main",
        ],
    },
)

"""
ROS2 Launch — HAIIP Full Closed-Loop Pipeline
==============================================
Launches all 4 nodes with proper QoS and parameter passing.

Usage:
    ros2 launch haiip haiip_closed_loop.launch.py machine_id:=pump-01
    ros2 launch haiip haiip_closed_loop.launch.py fault_mode:=true api_url:=http://10.0.0.5:8000
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    machine_id = LaunchConfiguration("machine_id")
    api_url = LaunchConfiguration("api_url")
    hz = LaunchConfiguration("hz")
    fault_mode = LaunchConfiguration("fault_mode")

    return LaunchDescription(
        [
            DeclareLaunchArgument("machine_id", default_value="pump-01"),
            DeclareLaunchArgument("api_url", default_value="http://localhost:8000"),
            DeclareLaunchArgument("hz", default_value="50"),
            DeclareLaunchArgument("fault_mode", default_value="false"),
            Node(
                package="haiip",
                executable="vibration_publisher",
                name="vibration_publisher",
                parameters=[{"machine_id": machine_id, "hz": hz, "fault_mode": fault_mode}],
                output="screen",
            ),
            Node(
                package="haiip",
                executable="inference_node",
                name="inference_node",
                parameters=[
                    {
                        "machine_ids": [machine_id],
                        "api_url": api_url,
                        "sample_every_n": hz,
                    }
                ],
                output="screen",
            ),
            Node(
                package="haiip",
                executable="economic_node",
                name="economic_node",
                parameters=[{"machine_ids": [machine_id]}],
                output="screen",
            ),
            Node(
                package="haiip",
                executable="action_node",
                name="action_node",
                parameters=[{"machine_ids": [machine_id], "override_ttl_sec": 600.0}],
                output="screen",
            ),
        ]
    )

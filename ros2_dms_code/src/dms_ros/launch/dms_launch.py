from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    cfg = os.path.join(
        get_package_share_directory('dms_ros'),
        'config',
        'dms_params.yaml'
    )

    prompts = os.path.join(
        get_package_share_directory('dms_ros'),
        'config',
        'prompts.yaml'
    )

    return LaunchDescription([
        Node(
            package='dms_ros',
            executable='camera_node',
            name='camera_node',
            parameters=[cfg],
            output='screen'
        ),
        Node(
            package='dms_ros',
            executable='preprocess_node',
            name='preprocess_node',
            parameters=[cfg],
            output='screen'
        ),
        Node(
            package='dms_ros',
            executable='inference_node',
            name='inference_node',
            parameters=[cfg, {'prompts_file': prompts}],
            output='screen'
        ),
        Node(
            package='dms_ros',
            executable='alert_node',
            name='alert_node',
            parameters=[cfg],
            output='screen'
        ),
        Node(
            package='dms_ros',
            executable='post_benchmark_node',
            name='post_benchmark_node',
            parameters=[cfg],
            output='screen'
        ),
    ])

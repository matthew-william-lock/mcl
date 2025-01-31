import os
import launch

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch.actions import IncludeLaunchDescription
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    package_dir = get_package_share_directory('webots_ros2_epuck')

    use_rviz = LaunchConfiguration('rviz', default=False)
    use_randombounce = LaunchConfiguration('rb', default=True) # Use dualsense controller  
    synchronization = LaunchConfiguration('synchronization', default=False)
    use_sim_time = LaunchConfiguration('use_sim_time', default=True)
    world = LaunchConfiguration('world', default='epuck_world.wbt')
    mission_time = LaunchConfiguration('mission_time', default=5)
    rviz_config = os.path.join(get_package_share_directory('mcl'), 'resource', 'configs.rviz')

    webots_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(package_dir,'launch', 'robot_launch.py')
        ),
        launch_arguments={
            'synchronization': synchronization,
            'use_sim_time': use_sim_time
        }.items()
    )

    return LaunchDescription([
        webots_launch,
        ExecuteProcess(cmd=['ros2', 'bag', 'record', '-a','--exclude','\/camera.*\w*'], output='screen', log_cmd=True),
        Node(
            package='rviz2',
            executable='rviz2',
            output='log',
            arguments=['--display-config=' + rviz_config],
            parameters=[{'use_sim_time': use_sim_time}],
            condition=launch.conditions.IfCondition(use_rviz)
        ),
        Node(
            package='mcl',
            executable='random_bounce',
            output='screen',
            condition=launch.conditions.IfCondition(use_randombounce)
        ),
        Node(
            package='mcl',
            executable='mission_controller',
            output='log',
            parameters=[{'mission_time': mission_time}]
        ),
        Node(
            package='mcl',
            executable='robot_simulator_teensy',
            output='log'
        ),
        
    ])

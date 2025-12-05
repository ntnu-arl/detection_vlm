cmake_minimum_required(VERSION 3.16)
project(detection_vlm_ros)
find_package(catkin REQUIRED COMPONENTS
  rospy
)

find_package(dynamic_reconfigure REQUIRED)

catkin_python_setup()
generate_dynamic_reconfigure_options(
    cfg/DetectionVLMConfig.cfg
)
catkin_package()
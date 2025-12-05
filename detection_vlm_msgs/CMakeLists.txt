cmake_minimum_required(VERSION 3.16)
project(detection_vlm_msgs)

find_package(catkin REQUIRED message_generation std_msgs)

add_service_files(FILES SetPrompt.srv)

generate_messages(DEPENDENCIES std_msgs)

catkin_package(
  CATKIN_DEPENDS message_runtime std_msgs
  DEPENDS
  INCLUDE_DIRS
  LIBRARIES
)

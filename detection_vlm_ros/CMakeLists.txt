cmake_minimum_required(VERSION 3.16)
project(detection_vlm_ros)
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

install(PROGRAMS app/detection_vlm_node.py app/reasoning_vlm_node.py
        DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})
install(DIRECTORY config DESTINATION share/${PROJECT_NAME})
install(DIRECTORY prompts DESTINATION share/${PROJECT_NAME})

ament_python_install_package(${PROJECT_NAME})
ament_package()
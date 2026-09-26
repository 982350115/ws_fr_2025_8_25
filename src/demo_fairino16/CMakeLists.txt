cmake_minimum_required(VERSION 3.22)
project(demo_fairino16)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(moveit_visual_tools REQUIRED)
find_package(moveit_core REQUIRED)
find_package(gazebo_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV 4 REQUIRED) 
find_package(kinematics_interface REQUIRED)

add_executable(move_group src/move_group.cpp)

ament_target_dependencies(
  move_group
  rclcpp
  rclcpp_action
  moveit_ros_planning_interface
  moveit_visual_tools
  moveit_core
  tf2
  tf2_geometry_msgs
  Eigen3
)

ament_package()
install(TARGETS  move_group
  DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME}
  PATTERN "setup_assistant.launch" EXCLUDE)
install(DIRECTORY config DESTINATION share/${PROJECT_NAME})
install(FILES .setup_assistant DESTINATION share/${PROJECT_NAME})

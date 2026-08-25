cmake_minimum_required(VERSION 3.22)
project(fairino16_v6_moveit2_config)

# 1. 设置 C++ 标准 (ROS 2 建议使用 C++17)
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

# 2. 查找必要的依赖包
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(moveit_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

# 3. 定义可执行文件 (你的 C++ 代码)
# 确保你的 cpp 文件放在 src 目录下，名字叫 circle_demo.cpp
add_executable(circle_demo src/circle_demo.cpp)
add_executable(a_demo src/a_demo.cpp)

# 4. 链接依赖库
ament_target_dependencies(circle_demo
  rclcpp
  moveit_ros_planning_interface
  moveit_msgs
  geometry_msgs
)

ament_target_dependencies(a_demo
  rclcpp
  moveit_ros_planning_interface
  moveit_msgs
  geometry_msgs
)

# 5. 安装规则：将可执行文件安装到 lib 目录，以便 ros2 run 可以找到
install(TARGETS circle_demo
  DESTINATION lib/${PROJECT_NAME}
)

install(TARGETS a_demo
  DESTINATION lib/${PROJECT_NAME}
)

# 6. 安装规则：保留原有的 launch 和 config 文件夹安装规则
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME}
  PATTERN "setup_assistant.launch" EXCLUDE)
install(DIRECTORY config DESTINATION share/${PROJECT_NAME})
install(FILES .setup_assistant DESTINATION share/${PROJECT_NAME})

# 7. 结束宏
ament_package()

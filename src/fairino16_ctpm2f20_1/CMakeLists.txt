cmake_minimum_required(VERSION 3.22)
project(fairino16_ctpm2f20_1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# --------------------------------------------------
# 1. 依赖声明 (REQUIRED)
# --------------------------------------------------
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(moveit_core REQUIRED) 
find_package(moveit_visual_tools REQUIRED) 
find_package(example_interfaces REQUIRED)

# --------------------------------------------------
# 2. 包含路径
# --------------------------------------------------
include_directories(
    include
    ${moveit_ros_planning_interface_INCLUDE_DIRS}
    ${moveit_core_INCLUDE_DIRS}
    ${moveit_visual_tools_INCLUDE_DIRS}
)

# --------------------------------------------------
# 3. 可执行文件 (Executables)
# --------------------------------------------------
add_executable(simple_pick_place_5 src/simple_pick_place_5.cpp) 
add_executable(calibration_verify src/calibration_verify.cpp)
add_executable(interactive_circle_node src/interactive_circle.cpp)
add_executable(tcp_pivot_test src/tcp_pivot_test.cpp)

# --------------------------------------------------
# 4. 节点依赖关系 (ament_target_dependencies)
# --------------------------------------------------
# [修正] 为 simple_pick_place_5 显式添加 moveit_visual_tools 依赖
ament_target_dependencies(simple_pick_place_5 
    rclcpp 
    geometry_msgs 
    trajectory_msgs 
    tf2_geometry_msgs 
    moveit_ros_planning_interface 
    moveit_core 
    moveit_visual_tools
    example_interfaces)

ament_target_dependencies(calibration_verify
    rclcpp geometry_msgs 
    tf2_geometry_msgs 
    moveit_ros_planning_interface 
    moveit_core)

ament_target_dependencies(interactive_circle_node 
    rclcpp 
    geometry_msgs 
    tf2_geometry_msgs 
    moveit_ros_planning_interface 
    moveit_core
    moveit_visual_tools)

ament_target_dependencies(tcp_pivot_test
  rclcpp
  geometry_msgs
  tf2_geometry_msgs
  moveit_ros_planning_interface
  moveit_core
  moveit_visual_tools
)

# --------------------------------------------------
# 5. 节点链接库 (target_link_libraries)
# --------------------------------------------------
# [补充] 为 simple_pick_place_5 链接可视化工具库
target_link_libraries(simple_pick_place_5 
    ${moveit_ros_planning_interface_LIBRARIES} 
    ${moveit_visual_tools_LIBRARIES}
    ${moveit_core_LIBRARIES})

target_link_libraries(calibration_verify 
    ${moveit_ros_planning_interface_LIBRARIES} 
    ${moveit_core_LIBRARIES})

target_link_libraries(interactive_circle_node
    ${moveit_ros_planning_interface_LIBRARIES}
    ${moveit_visual_tools_LIBRARIES}
    ${moveit_core_LIBRARIES}
)

target_link_libraries(tcp_pivot_test
    ${moveit_ros_planning_interface_LIBRARIES}
    ${moveit_visual_tools_LIBRARIES}
    ${moveit_core_LIBRARIES}
)

# --------------------------------------------------
# 6. 安装 (Install)
# --------------------------------------------------
install(TARGETS
    simple_pick_place_5
    calibration_verify  
    interactive_circle_node 
    tcp_pivot_test
    DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch config meshes DESTINATION share/${PROJECT_NAME})

# --------------------------------------------------
# 7. 导出依赖
# --------------------------------------------------
ament_export_dependencies(
    rclcpp
    geometry_msgs
    trajectory_msgs
    tf2_geometry_msgs
    moveit_ros_planning_interface
    moveit_core
    moveit_visual_tools
    example_interfaces
)

ament_package()
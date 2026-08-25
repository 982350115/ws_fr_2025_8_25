cmake_minimum_required(VERSION 3.22)
project(fairino16_ctpm2f20)

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
# [新增] 显式查找 moveit_core，因为我们用了 trajectory_processing
find_package(moveit_core REQUIRED) 
find_package(example_interfaces REQUIRED)

# --------------------------------------------------
# 2. 包含路径
# --------------------------------------------------
# 注意：ament_target_dependencies 通常会自动处理 include 路径，
# 但保留这个以防万一，同时加入 moveit_core 的路径
include_directories(
    include
    ${moveit_ros_planning_interface_INCLUDE_DIRS}
    ${moveit_core_INCLUDE_DIRS} 
)

# --------------------------------------------------
# 3. 可执行文件 (Executables)
# --------------------------------------------------
add_executable(simple_pick_place_3 src/simple_pick_place_3.cpp) 

# --------------------------------------------------
# 4. 节点依赖关系 (ament_target_dependencies)
# --------------------------------------------------
ament_target_dependencies(simple_pick_place_3
    rclcpp
    geometry_msgs
    trajectory_msgs
    tf2_geometry_msgs
    moveit_ros_planning_interface
    moveit_core         # [重要修改] 添加 moveit_core 以支持时间参数化算法
    example_interfaces
)

# --------------------------------------------------
# 5. 节点链接库 (target_link_libraries)
# --------------------------------------------------
# 如果使用了 ament_target_dependencies，通常不需要手动 link libraries，
# 但为了兼容旧习惯和确保链接，保留并添加 moveit_core 库
target_link_libraries(simple_pick_place_3
    ${moveit_ros_planning_interface_LIBRARIES}
    ${moveit_core_LIBRARIES}
)

# --------------------------------------------------
# 6. 安装 (Install)
# --------------------------------------------------
install(TARGETS
    simple_pick_place_3
    DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
    launch
    config
    meshes
    DESTINATION share/${PROJECT_NAME}
)

# --------------------------------------------------
# 7. 导出依赖
# --------------------------------------------------
ament_export_dependencies(
    rclcpp
    geometry_msgs
    trajectory_msgs
    tf2_geometry_msgs
    moveit_ros_planning_interface
    moveit_core         # [建议] 导出它
    example_interfaces
)

ament_package()
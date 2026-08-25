#include <memory>
#include <vector>
#include <cmath>
#include <thread>
#include <chrono>

#include <rclcpp/rclcpp.hpp>
#include <moveit/move_group_interface/move_group_interface.h>

int main(int argc, char * argv[])
{
  // 1. 初始化 ROS
  rclcpp::init(argc, argv);
  
  // 创建节点
  auto const node = std::make_shared<rclcpp::Node>(
    "circle_moveit_demo",
    rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true)
  );

  auto const logger = rclcpp::get_logger("hello_moveit");

  // ==========================================
  // 【核心机制】: 开启后台线程接收数据
  // ==========================================
  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  // 分离线程，让它在后台跑 spin()，这样 getCurrentPose 才能拿到最新数据
  std::thread([&executor]() { executor.spin(); }).detach();

  // 2. 创建 MoveGroup 接口
  using moveit::planning_interface::MoveGroupInterface;
  auto move_group_interface = MoveGroupInterface(node, "fairino16_v6_group");

  // 设置速度/加速度比例 (稍微放慢一点以保证平稳)
  move_group_interface.setMaxVelocityScalingFactor(0.005);
  move_group_interface.setMaxAccelerationScalingFactor(0.005);

  // ==========================================
  // 步骤 0: 记录初始位置 (Home)
  // ==========================================
  RCLCPP_INFO(logger, "Step 0: Recording Home position...");
  
  // 稍微等待一下后台线程同步数据
  std::this_thread::sleep_for(std::chrono::milliseconds(500));

  auto current_pose_stamped = move_group_interface.getCurrentPose();
  auto home_pose = current_pose_stamped.pose;

  RCLCPP_INFO(logger, "Captured Home Pose -> X: %.3f, Y: %.3f, Z: %.3f", 
              home_pose.position.x, home_pose.position.y, home_pose.position.z);

  // 简单检查数据有效性
  if (std::abs(home_pose.position.x) < 0.001 && 
      std::abs(home_pose.position.y) < 0.001 && 
      std::abs(home_pose.position.z) < 0.001) {
       RCLCPP_WARN(logger, "Warning: Home pose is (0,0,0). Robot state might not be synced.");
  }

  // 定义圆的参数 (圆心在前上方)
  const double center_x = 0.5;
  const double center_y = 0.2;
  const double center_z = 0.5;
  const double radius = 0.1;

  // ==========================================
  // 步骤 1: 移动到圆的起始点
  // ==========================================
  geometry_msgs::msg::Pose start_pose;
  
  // 位置：圆的最右侧点
  start_pose.position.x = center_x + radius;
  start_pose.position.y = center_y;
  start_pose.position.z = center_z;

  // 【关键修改】: 强制设定姿态为“垂直向下”
  // 之前的代码继承了 Home 点的姿态，导致手腕可能处于极限角度。
  // 这里手动指定四元数 (w=0, x=1, y=0, z=0) 通常代表末端垂直向下（绕X轴转180度）。
  // 这样机械臂画水平圆时，关节最自由。
  start_pose.orientation.w = 0.0;
  start_pose.orientation.x = 1.0;
  start_pose.orientation.y = 0.0;
  start_pose.orientation.z = 0.0;

  RCLCPP_INFO(logger, "Step 1: Moving to start point (Orientation: Vertical Down)...");
  move_group_interface.setPoseTarget(start_pose);
  
  auto const [success_start, plan_start] = [&move_group_interface]{
    moveit::planning_interface::MoveGroupInterface::Plan msg;
    auto const ok = static_cast<bool>(move_group_interface.plan(msg));
    return std::make_pair(ok, msg);
  }();

  if(success_start) {
    move_group_interface.execute(plan_start);
  } else {
    RCLCPP_ERROR(logger, "Failed to reach start point! Hint: Check if the robot can reach (0.4, 0.2, 0.5) with vertical orientation.");
    rclcpp::shutdown();
    return 1;
  }

  // 等待稳定
  RCLCPP_INFO(logger, "Reached Start Point. Waiting 1 second...");
  std::this_thread::sleep_for(std::chrono::seconds(1));

  // ==========================================
  // 步骤 2: 生成圆形轨迹 (Cartesian Path)
  // ==========================================
  RCLCPP_INFO(logger, "Step 2: Computing circular path...");
  
  std::vector<geometry_msgs::msg::Pose> waypoints;
  // 生成圆周路径点
  for (double angle = 0.0; angle <= 2 * M_PI; angle += 0.05) {
      geometry_msgs::msg::Pose waypoint = start_pose;
      // 保持 orientation 不变（一直垂直向下）
      // 只改变位置
      waypoint.position.x = center_x + radius * std::cos(angle);
      waypoint.position.y = center_y + radius * std::sin(angle);
      waypoint.position.z = center_z; 
      waypoints.push_back(waypoint);
  }

  moveit_msgs::msg::RobotTrajectory trajectory;
  const double jump_threshold = 0.0; // 0.0 表示不检查跳跃（如果报错 jump threshold exceeded，可改为 5.0 试试）
  const double eef_step = 0.01;      // 插补步长 1cm

  // 计算笛卡尔路径
  double fraction = move_group_interface.computeCartesianPath(
    waypoints, eef_step, jump_threshold, trajectory);

  RCLCPP_INFO(logger, "Visualizing plan (%.2f%% achieved)", fraction * 100.0);

  if (fraction > 0.9) { 
      move_group_interface.execute(trajectory);
  } else {
      RCLCPP_ERROR(logger, "Path planning failed (fraction too low)!");
      rclcpp::shutdown();
      return 1;
  }

  RCLCPP_INFO(logger, "Finished circle. Waiting 1 second...");
  std::this_thread::sleep_for(std::chrono::seconds(1));

  // ==========================================
  // 步骤 3: 归位 (回到 Home)
  // ==========================================
  RCLCPP_INFO(logger, "Step 3: Returning to Home position...");
  move_group_interface.setPoseTarget(home_pose);
  
  auto const [success_home, plan_home] = [&move_group_interface]{
    moveit::planning_interface::MoveGroupInterface::Plan msg;
    auto const ok = static_cast<bool>(move_group_interface.plan(msg));
    return std::make_pair(ok, msg);
  }();

  if(success_home) {
    move_group_interface.execute(plan_home);
    RCLCPP_INFO(logger, "Mission Complete!");
  } else {
    RCLCPP_ERROR(logger, "Failed to return home!");
  }

  // 退出 ROS
  rclcpp::shutdown();
  return 0;
}
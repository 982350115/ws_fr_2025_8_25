#include <rclcpp/rclcpp.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <geometry_msgs/msg/pose.hpp>

int main(int argc, char** argv)
{
  // 初始化 ROS 2
  rclcpp::init(argc, argv);
  auto node = rclcpp::Node::make_shared("moveit_test_node");
  auto const logger = rclcpp::get_logger("moveit_test");

  RCLCPP_INFO(logger, "MoveIt2 test node started.");

  // 创建 MoveGroupInterface（规划组：manipulator）
  using moveit::planning_interface::MoveGroupInterface;
  MoveGroupInterface move_group_interface(node, "manipulator");

  // 设置目标位姿
  auto const target_pose = [] {
    geometry_msgs::msg::Pose msg;
    msg.orientation.w = 1.0;
    msg.position.x = 0.28;
    msg.position.y = -0.2;
    msg.position.z = 0.5;
    return msg;
  }();
  move_group_interface.setPoseTarget(target_pose);
  RCLCPP_INFO(logger, "Target pose set: [x=%.2f, y=%.2f, z=%.2f]",
              target_pose.position.x, target_pose.position.y, target_pose.position.z);

  // 开始规划
  RCLCPP_INFO(logger, "Planning trajectory...");
  auto const [success, plan] = [&move_group_interface] {
    moveit::planning_interface::MoveGroupInterface::Plan msg;
    auto const ok = static_cast<bool>(move_group_interface.plan(msg));
    return std::make_pair(ok, msg);
  }();

  // 执行规划
  if (success) {
    RCLCPP_INFO(logger, "Planning successful. Executing trajectory...");
    move_group_interface.execute(plan);
    RCLCPP_INFO(logger, "Execution finished.");
  } else {
    RCLCPP_ERROR(logger, "Planning failed!");
  }

  // 关闭节点
  rclcpp::shutdown();
  return 0;
}

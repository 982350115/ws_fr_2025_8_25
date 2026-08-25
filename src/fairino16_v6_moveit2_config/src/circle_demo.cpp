#include <rclcpp/rclcpp.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <thread> // 必须包含这个

int main(int argc, char** argv)
{
  // 1. 初始化 ROS 2
  rclcpp::init(argc, argv);
  
  // 2. 配置节点选项 (MoveIt 必需)
  rclcpp::NodeOptions node_options;
  node_options.automatically_declare_parameters_from_overrides(true);
  
  // 创建节点
  auto node = rclcpp::Node::make_shared("moveit_test_node", node_options);

  // =================================================================================
  // 核心修复点：使用多线程执行器，并扔到后台线程去跑
  // 如果没有这一步，execute() 发出指令后无法接收反馈，导致实物/仿真不动
  // =================================================================================
  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);
  std::thread spinner([&executor]() { executor.spin(); });

  // 3. 创建 MoveGroup 接口
  // 根据你的日志，你的规划组名字叫 "fairino16_v6_group"
  static const std::string PLANNING_GROUP = "fairino16_v6_group"; 
  moveit::planning_interface::MoveGroupInterface move_group(node, PLANNING_GROUP);

  // 打印一下，确认连接上了
  RCLCPP_INFO(node->get_logger(), "规划组 %s 已加载", PLANNING_GROUP.c_str());

  // 4. 设置一个随机目标 (先测试随机目标，排除是 Pose 设置错误的问题)
  move_group.setStartStateToCurrentState();
  move_group.setRandomTarget();

  // 5. 规划 (Plan)
  moveit::planning_interface::MoveGroupInterface::Plan my_plan;
  bool success = (move_group.plan(my_plan) == moveit::core::MoveItErrorCode::SUCCESS);

  if (success)
  {
    RCLCPP_INFO(node->get_logger(), "规划成功！RViz 中应该能看到橙色影子...");
    
    // 稍微停顿一下，让你看清楚影子
    std::this_thread::sleep_for(std::chrono::seconds(1));

    RCLCPP_INFO(node->get_logger(), "开始执行 (Execute)...");

    // 6. 执行 (Execute)
    // 因为上面有 spinner 线程在跑，这里才能正常工作
    auto result = move_group.execute(my_plan);

    if (result == moveit::core::MoveItErrorCode::SUCCESS)
    {
      RCLCPP_INFO(node->get_logger(), "执行成功！白色机械臂应该动了！");
    }
    else
    {
      RCLCPP_ERROR(node->get_logger(), "执行失败！错误代码: %d", result.val);
    }
  }
  else
  {
    RCLCPP_ERROR(node->get_logger(), "规划失败，请检查目标是否在工作空间内。");
  }

  // 结束
  rclcpp::shutdown();
  // 等待线程结束
  spinner.join();
  return 0;
}
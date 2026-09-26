/* cartesian_bspline_demo.cpp
 * ----------------------------------------------
 * 用 Eigen 三次 B-样条生成平面路径 → MoveIt 2 规划笛卡尔轨迹
 * 并将关节位置保存到 trajectory_positions.txt
 *
 * 依赖：
 *   rclcpp
 *   geometry_msgs
 *   moveit_ros_planning_interface
 *   eigen3 (unsupported/Eigen/Spline, SplineFitting)
 * 编译：
 *   colcon build --packages-select <your_package>
 */

#include <rclcpp/rclcpp.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <geometry_msgs/msg/pose.hpp>
#include <tf2/LinearMath/Quaternion.h>

#include <unsupported/Eigen/src/Splines/Spline.h>
#include <unsupported/Eigen/src/Splines/SplineFitting.h>
#include <moveit_visual_tools/moveit_visual_tools.h>
#include <moveit/trajectory_processing/iterative_time_parameterization.h>
#include <moveit/robot_trajectory/robot_trajectory.h>

#include <fstream>
#include <vector>
#include <thread>
#include <chrono>

// 简化日志宏
static const rclcpp::Logger LOGGER = rclcpp::get_logger("bspline_demo");

/* ------------------------------------------------------------------ */
/* 采样三次 B-样条函数：返回 geometry_msgs::msg::Pose 序列              */
/* ------------------------------------------------------------------ */
static std::vector<geometry_msgs::msg::Pose>
sampleCubicBSpline(const std::vector<Eigen::Vector2d>& control_pts,
                   int    num_samples,
                   double plane_z)
{
  using Spline2d = Eigen::Spline<double, 2>;
  constexpr int DEG = 3;  // 三次

  // 1) 控制点 -> Eigen 矩阵 (2 × N)
  Eigen::Matrix<double, 2, Eigen::Dynamic> ctrl(2, control_pts.size());
  for (size_t i = 0; i < control_pts.size(); ++i)
    ctrl.col(i) = control_pts[i];

  // 2) 均匀参数向量 u
  Eigen::RowVectorXd u(ctrl.cols());
  u.setLinSpaced(ctrl.cols(), 0.0, 1.0);

  // 3) 插值生成样条
  Spline2d spline = Eigen::SplineFitting<Spline2d>::Interpolate(ctrl, DEG, u);

  // 4) 均匀采样
  std::vector<geometry_msgs::msg::Pose> waypoints;
  waypoints.reserve(num_samples);
  tf2::Quaternion q; q.setRPY(0,0,3.14);  // 固定姿态M_PI
  
  for (int i = 0; i < num_samples; ++i)
  {
    double t = static_cast<double>(i) / (num_samples - 1);
    Eigen::Vector2d p = spline(t);

    geometry_msgs::msg::Pose pose;
    pose.position.x = p.x();
    pose.position.y = p.y();
    pose.position.z = plane_z;
    pose.orientation.x = 0;
    pose.orientation.y = 1;
    pose.orientation.z = 0;
    pose.orientation.w = 0;
    waypoints.emplace_back(pose);
  }
  return waypoints;
}

/* ------------------------------------------------------------------ */
/*                               main                                 */
/* ------------------------------------------------------------------ */
int main(int argc, char** argv)
{
  /* -------- ROS 2 初始化 -------- */
  rclcpp::init(argc, argv);
  auto node = rclcpp::Node::make_shared(
      "bspline_cartesian_demo",
      rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true));

  // 让 Node 保持自旋，保证 MoveIt CurrentStateMonitor 正常更新
  rclcpp::executors::SingleThreadedExecutor exec;
  exec.add_node(node);
  std::thread{[&exec]{ exec.spin(); }}.detach();

  /* -------- MoveIt 接口 -------- */
  moveit::planning_interface::MoveGroupInterface move_group(node, "fairino16_v6_group");
  move_group.setPlanningTime(60.0);

  /* -------- 样条参数 -------- */
  std::vector<Eigen::Vector2d> ctrl_pts {
      {0.133,0.825},
      {0.138,0.668},
      {0.39,0.403},
      {0.60,0.401},
      {0.64,0.208},
     
      // {0.60, 0.3},
      // {0.63, 0.4},
      // {0.66, 0.4},
      // {0.69, 0.3},
      // {0.66, 0.2},
      // {0.63, 0.2}
  };
  const int    NUM_SAMPLES    = 100;
  const double PLANE_Z        = 0.6;  // m
  const double EEF_STEP       = 0.0001; // m
  const double JUMP_THRESHOLD = 0.0;   // rad

  /* -------- 1) 生成样条路径点 -------- */
  auto waypoints = sampleCubicBSpline(ctrl_pts, NUM_SAMPLES, PLANE_Z);
  RCLCPP_INFO(LOGGER, "Generated %zu waypoints.", waypoints.size());

  /* ---------- 2) 先移动到起始 Pose ---------- */
  const auto& init_pose = waypoints.front();
  // init_pose.position.z =0.30;
  move_group.setPoseTarget(init_pose);
  moveit::planning_interface::MoveGroupInterface::Plan init_plan;
  if (move_group.plan(init_plan) == moveit::core::MoveItErrorCode::SUCCESS)
  {
    RCLCPP_INFO(LOGGER, "Moving to start pose...");
    move_group.execute(init_plan);
        // **修改点 1: 在移动到起始点后等待 1 秒**
    RCLCPP_INFO(LOGGER, "Finished moving to start pose. Waiting 1 seconds for stability...");
    std::this_thread::sleep_for(std::chrono::seconds(3));
  }

  else
  {
    RCLCPP_ERROR(LOGGER, "Failed to plan to start pose, abort.");
    rclcpp::shutdown();  return 1;
  }

  /* 更新起始状态为当前位置 */
  move_group.setStartStateToCurrentState();

  /* -------- 2) 笛卡尔路径规划 -------- */
  moveit_msgs::msg::RobotTrajectory trajectory;
  double fraction = move_group.computeCartesianPath(
      waypoints, EEF_STEP, JUMP_THRESHOLD, trajectory);
  RCLCPP_INFO(LOGGER, "Path completion: %.1f %%", fraction * 100.0);

  /* -------- 3) 保存关节位置到 TXT -------- */
  {
    std::ofstream ofs("trajectory_positions.txt");
    ofs << "t(s)";
    for (auto& n : trajectory.joint_trajectory.joint_names) ofs << '\t' << n;
    ofs << '\n';

    for (auto& pt : trajectory.joint_trajectory.points)
    {
      double t = pt.time_from_start.sec + 1e-9 * pt.time_from_start.nanosec;
      ofs << t;
      for (double pos : pt.positions) ofs << '\t' << pos;
      ofs << '\n';
    }
    RCLCPP_INFO(LOGGER, "Trajectory saved to trajectory_positions.txt");
  }

  /* -------- 4) 执行轨迹（完整才执行） -------- */
  namespace rvt = rviz_visual_tools;
  moveit_visual_tools::MoveItVisualTools visual_tools(node, "base_link", "move_group_tutorial",
                                                      move_group.getRobotModel());
  visual_tools.deleteAllMarkers();
  // visual_tools.publishPath(waypoints, rvt::LIME_GREEN, rvt::SMALL);
  const moveit::core::JointModelGroup* joint_model_group =
      move_group.getCurrentState()->getJointModelGroup("fairino16_v6_group");
  
  visual_tools.publishTrajectoryLine(trajectory, joint_model_group);
  visual_tools.trigger();

  RCLCPP_INFO(LOGGER, "Trajectory points: %zu", trajectory.joint_trajectory.points.size());
  if (!joint_model_group)
    RCLCPP_ERROR(LOGGER, "Joint model group is nullptr!");
  
  if (fraction > 0.99)
  {
  // ==========================================
    // 步骤 1: 将消息转换为 RobotTrajectory 对象以进行计算
    // ==========================================
    // 获取当前的 RobotModel 和 State
    auto robot_model_ptr = move_group.getRobotModel();
    auto current_state_ptr = move_group.getCurrentState();
    
    // 创建 RobotTrajectory 对象
    robot_trajectory::RobotTrajectory rt(robot_model_ptr, move_group.getName());
    
    // 用生成的笛卡尔路径填充它
    rt.setRobotTrajectoryMsg(*current_state_ptr, trajectory);

    // ==========================================
    // 步骤 2: 进行时间参数化 (这是关键！)
    // ==========================================
    // 实例化算法类 (IPTP 是最常用的)
    trajectory_processing::IterativeParabolicTimeParameterization iptp;

    // 获取你在 MoveGroup 中设置的比例因子 (或者手动指定 0.1 ~ 1.0)
    // 这一步会让 joint_limits.yaml 中的限制生效，并乘以这个比例
    double vel_scale = 0.05; // 这里的 1.0 代表使用 joint_limits.yaml 的 100%
    double acc_scale = 0.1; 
    
    // 如果你想使用 move_group 设置过的因子，也可以这样尝试获取（视版本而定可能需要手动维护变量）：
    // vel_scale = move_group.getMaxVelocityScalingFactor(); 

    // 计算时间戳、速度和加速度
    bool success = iptp.computeTimeStamps(rt, vel_scale, acc_scale);

    if (success)
    {
        RCLCPP_INFO(LOGGER, "时间参数化成功，准备执行...");

        // ==========================================
        // 步骤 3: 将处理后的轨迹放回 Plan 并执行
        // ==========================================
        moveit::planning_interface::MoveGroupInterface::Plan plan;
        
        // 将计算好速度的 rt 对象转回消息格式
        rt.getRobotTrajectoryMsg(plan.trajectory_);
        
        move_group.execute(plan);
    }
    else
    {
        RCLCPP_ERROR(LOGGER, "时间参数化失败！");
    }
  }
  // {
  //   moveit::planning_interface::MoveGroupInterface::Plan plan;
  //   plan.trajectory_ = trajectory;
  //   move_group.execute(plan);
  // }

  rclcpp::shutdown();
  return 0;
}
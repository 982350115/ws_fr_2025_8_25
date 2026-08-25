#include <rclcpp/rclcpp.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit_visual_tools/moveit_visual_tools.h>
#include <geometry_msgs/msg/pose.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <thread>
#include <cmath>

class InteractiveCircle : public rclcpp::Node {
public:
    InteractiveCircle() : Node("interactive_circle_node") {
        RCLCPP_INFO(this->get_logger(), "=== 画圆验证节点 (全交互+舒展位姿TCP向下) 启动 ===");
    }

    void init() {
        move_group_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(shared_from_this(), "manipulator");
        tcp_link_ = "gripper_center_tcp";
        move_group_->setEndEffectorLink(tcp_link_);
        
        move_group_->setMaxVelocityScalingFactor(0.05);
        move_group_->setMaxAccelerationScalingFactor(0.05);
        move_group_->setPlanningTime(10.0); 

        visual_tools_ = std::make_shared<moveit_visual_tools::MoveItVisualTools>(
            shared_from_this(), "base_link", "circle_movement_visualization", move_group_->getRobotModel());
        visual_tools_->deleteAllMarkers();
        visual_tools_->loadRemoteControl();
        visual_tools_->trigger();

        RCLCPP_INFO(this->get_logger(), "MoveIt 接口就绪。请确认 URDF 中 Z 轴已翻转。");
    }

    // --- 关节预览与执行 (用于舒展且夹爪向下的初始化) ---
    bool confirmJointStep(const std::string& step_name, const std::vector<double>& joint_positions) {
        std::cout << "\n>>> 任务: " << step_name << std::endl;
        geometry_msgs::msg::Pose current_pose = move_group_->getCurrentPose().pose;
        visual_tools_->publishText(current_pose, "1. [PLAN] " + step_name, rviz_visual_tools::YELLOW, rviz_visual_tools::XLARGE);
        visual_tools_->trigger();
        std::cout << "[交互] 点击 'Next' 预览【舒展且夹爪朝下】的规划..." << std::endl;
        visual_tools_->prompt("等待预览指令...");

        move_group_->setJointValueTarget(joint_positions);
        moveit::planning_interface::MoveGroupInterface::Plan my_plan;
        
        if (move_group_->plan(my_plan) == moveit::core::MoveItErrorCode::SUCCESS) {
            visual_tools_->publishText(current_pose, "2. [EXECUTE] " + step_name, rviz_visual_tools::GREEN, rviz_visual_tools::XLARGE);
            visual_tools_->trigger();
            std::cout << "[确认] 预览成功。点击 'Next' 物理执行位姿初始化..." << std::endl;
            visual_tools_->prompt("等待执行确认...");
            move_group_->execute(my_plan);
            return true;
        }
        RCLCPP_ERROR(this->get_logger(), "姿态初始化规划失败！");
        return false;
    }

    // --- 位姿预览与执行 (用于前往安全点和圆心) ---
    bool confirmPoseStep(const std::string& step_name, const geometry_msgs::msg::Pose& target_pose) {
        std::cout << "\n>>> 任务: " << step_name << std::endl;
        visual_tools_->publishText(target_pose, "1. [PLAN] " + step_name, rviz_visual_tools::YELLOW, rviz_visual_tools::XLARGE);
        visual_tools_->trigger();
        visual_tools_->prompt("等待预览指令...");

        move_group_->setPoseTarget(target_pose, tcp_link_);
        moveit::planning_interface::MoveGroupInterface::Plan my_plan;
        
        if (move_group_->plan(my_plan) == moveit::core::MoveItErrorCode::SUCCESS) {
            visual_tools_->publishText(target_pose, "2. [EXECUTE] " + step_name, rviz_visual_tools::GREEN, rviz_visual_tools::XLARGE);
            visual_tools_->trigger();
            visual_tools_->prompt("规划成功，等待执行确认...");
            move_group_->execute(my_plan);
            return true;
        }
        return false;
    }

    void run() {
        // [修复命名空间] 修正后的 PlanningSceneInterface
        moveit::planning_interface::PlanningSceneInterface planning_scene_interface;
        setupBaseScene(planning_scene_interface);

        // --- 动作 0: 姿态初始化 (舒展且夹爪垂直向下) ---
        // J1=0, J2=-1.57(抬大臂), J3=1.57(平小臂), J4=0, J5=1.57(弯手腕使夹爪向下), J6=0
        std::vector<double> stretch_joints = {0.0, -1.57, 1.57, 0.0, 1.57, 0.0};
        if (!confirmJointStep("初始化舒展位姿 (夹爪垂直向下)", stretch_joints)) return;

        // 统一姿态约束 (假设 URDF 已翻转 Z 轴，此处使用单位旋转)
        geometry_msgs::msg::Quaternion down_orientation;
        down_orientation.x = 0.0; down_orientation.y = 0.0; down_orientation.z = 0.0; down_orientation.w = 1.0;

        // --- 动作 1: 安全起点 ---
        geometry_msgs::msg::Pose safe_pose;
        safe_pose.position.x = -0.55; safe_pose.position.y = 0.0; safe_pose.position.z = 0.50;
        safe_pose.orientation = down_orientation;
        if (!confirmPoseStep("前往安全起点", safe_pose)) return;

        // --- 动作 2: 前往圆心 ---
        geometry_msgs::msg::Pose center_pose;
        center_pose.position.x = -0.55; center_pose.position.y = 0.0; center_pose.position.z = 0.40;
        center_pose.orientation = down_orientation;
        if (!confirmPoseStep("前往圆心点", center_pose)) return;

        // --- 动作 3: 画圆预览与执行 ---
        std::cout << "\n>>> 准备画圆预览 (点击 'Next')" << std::endl;
        visual_tools_->prompt("等待画圆预览...");
        
        std::vector<geometry_msgs::msg::Pose> waypoints;
        for (double theta = 0.0; theta <= 2 * M_PI; theta += 0.2) {
            geometry_msgs::msg::Pose wp = center_pose;
            wp.position.x = center_pose.position.x + 0.10 * std::cos(theta);
            wp.position.y = center_pose.position.y + 0.10 * std::sin(theta);
            waypoints.push_back(wp);
        }

        moveit_msgs::msg::RobotTrajectory trajectory;
        double fraction = move_group_->computeCartesianPath(waypoints, 0.01, 0.1, trajectory);
        if (fraction >= 1.0) {
            std::cout << ">>> 路径完整。点击 'Next' 物理画圆" << std::endl;
            visual_tools_->prompt("等待执行确认...");
            move_group_->execute(trajectory);
            RCLCPP_INFO(this->get_logger(), "验证任务结束。");
        }
    }

private:
    std::shared_ptr<moveit::planning_interface::MoveGroupInterface> move_group_;
    std::shared_ptr<moveit_visual_tools::MoveItVisualTools> visual_tools_;
    std::string tcp_link_;

    // [修复命名空间] 修正函数签名
    void setupBaseScene(moveit::planning_interface::PlanningSceneInterface& psi) {
        moveit_msgs::msg::CollisionObject table;
        table.id = "table";
        table.header.frame_id = "base_link";
        table.operation = table.ADD;
        shape_msgs::msg::SolidPrimitive box;
        box.type = box.BOX;
        box.dimensions = {2.0, 2.0, 0.01};
        geometry_msgs::msg::Pose p;
        p.position.z = -0.005; 
        p.orientation.w = 1.0;
        table.primitives.push_back(box);
        table.primitive_poses.push_back(p);
        psi.applyCollisionObjects({table});
    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<InteractiveCircle>();
    node->init();
    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node);
    std::thread([node]() { node->run(); }).detach();
    executor.spin();
    rclcpp::shutdown();
    return 0;
}
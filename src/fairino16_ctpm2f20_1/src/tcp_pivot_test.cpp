#include <rclcpp/rclcpp.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit_visual_tools/moveit_visual_tools.h>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <thread>

class TCPVerifier : public rclcpp::Node {
public:
    TCPVerifier() : Node("tcp_pivot_test") {
        RCLCPP_INFO(this->get_logger(), "=== TCP 验证节点已启动 ===");
    }

    // 初始化函数：在构造函数之外执行，确保 shared_from_this() 可用
    void init() {
        // 初始化 MoveGroup
        move_group_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(
            shared_from_this(), "manipulator");

        // 设置自定义的 TCP 末端 Link
        tcp_link_ = "gripper_center_tcp";
        move_group_->setEndEffectorLink(tcp_link_);

        // 初始化 Visual Tools 交互工具
        visual_tools_ = std::make_shared<moveit_visual_tools::MoveItVisualTools>(
            shared_from_this(), "base_link", "tcp_pivot_visualization", move_group_->getRobotModel());
        
        visual_tools_->deleteAllMarkers();
        visual_tools_->loadRemoteControl();
        visual_tools_->trigger();

        RCLCPP_INFO(this->get_logger(), "MoveIt 接口就绪。当前验证 TCP: %s", tcp_link_.c_str());
        RCLCPP_INFO(this->get_logger(), "请确保 RViz 中已添加 'MoveIt Visual Tools Gui' 面板。");
    }

    void run_validation() {
        // 等待 2 秒确保控制器状态同步
        rclcpp::sleep_for(std::chrono::seconds(2));

        // 记录起始点作为旋转中心
        geometry_msgs::msg::PoseStamped start_pose_stamped = move_group_->getCurrentPose(tcp_link_);
        geometry_msgs::msg::Pose base_pose = start_pose_stamped.pose;

        // 定义测试动作：X、Y、Z 轴的局部旋转
        struct TestStep { std::string name; double roll, pitch, yaw; };
        double rad = 0.26; // 约 15 度
        std::vector<TestStep> steps = {
            {"绕 X 轴旋转", rad, 0.0, 0.0},
            {"绕 X 轴反向", -rad, 0.0, 0.0},
            {"绕 Y 轴旋转", 0.0, rad, 0.0},
            {"绕 Y 轴反向", 0.0, -rad, 0.0},
            {"绕 Z 轴自旋", 0.0, 0.0, 0.5},
            {"绕 Z 轴反向", 0.0, 0.0, -0.5}
        };

        for (const auto& step : steps) {
            std::cout << "\n------------------------------------------------" << std::endl;
            std::cout << ">>> 准备验证步骤: " << step.name << std::endl;

            // --- 第一阶段：规划与预览 (PLAN) ---
            visual_tools_->publishText(base_pose, "1. [PLAN] " + step.name, rviz_visual_tools::YELLOW, rviz_visual_tools::XLARGE);
            visual_tools_->trigger();
            
            std::cout << "[交互] 请点击 RViz 面板上的 'Next' 开始【轨迹规划预览】..." << std::endl;
            visual_tools_->prompt("等待规划指令...");

            // 计算目标位姿
            tf2::Quaternion q_rot;
            q_rot.setRPY(step.roll, step.pitch, step.yaw);
            tf2::Quaternion q_orig;
            tf2::fromMsg(base_pose.orientation, q_orig);
            tf2::Quaternion q_new = q_orig * q_rot;
            q_new.normalize();

            geometry_msgs::msg::Pose target_pose = base_pose;
            target_pose.orientation = tf2::toMsg(q_new);
            move_group_->setPoseTarget(target_pose, tcp_link_);

            // 执行规划但不执行
            moveit::planning_interface::MoveGroupInterface::Plan my_plan;
            bool success = (move_group_->plan(my_plan) == moveit::core::MoveItErrorCode::SUCCESS);

            if (success) {
                // --- 第二阶段：硬件确认与执行 (EXECUTE) ---
                std::cout << "[成功] 规划完成！请在 RViz 中检查轨迹。如果没有自碰撞，执行下一步。" << std::endl;
                std::cout << "[交互] 请再次点击 'Next' 按钮，启动【物理机械臂运动】..." << std::endl;

                visual_tools_->publishText(base_pose, "2. [EXECUTE] " + step.name, rviz_visual_tools::GREEN, rviz_visual_tools::XLARGE);
                visual_tools_->trigger();

                // 第二次阻塞：等待物理执行确认
                visual_tools_->prompt("等待物理执行确认...");

                RCLCPP_INFO(this->get_logger(), "执行指令下发：物理机械臂开始动作！");
                move_group_->execute(my_plan);
                RCLCPP_INFO(this->get_logger(), "步骤动作已完成。");
            } else {
                RCLCPP_ERROR(this->get_logger(), "规划失败！可能存在自碰撞或超出机械臂限位。");
                visual_tools_->publishText(base_pose, "PLAN FAILED!", rviz_visual_tools::RED, rviz_visual_tools::XLARGE);
                visual_tools_->trigger();
            }
        }
        RCLCPP_INFO(this->get_logger(), "所有测试步骤已循环完毕。");
    }

private:
    std::shared_ptr<moveit::planning_interface::MoveGroupInterface> move_group_;
    std::shared_ptr<moveit_visual_tools::MoveItVisualTools> visual_tools_;
    std::string tcp_link_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    
    // 创建节点实例
    auto node = std::make_shared<TCPVerifier>();
    
    // 调用初始化函数 (在构造函数之外)
    node->init();
    
    // 使用多线程执行器以处理 MoveGroup 的后台回调
    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node);
    
    // 在独立线程中运行验证逻辑，避免阻塞 ROS 2 调度
    std::thread([&node]() {
        node->run_validation();
    }).detach();

    executor.spin();
    rclcpp::shutdown();
    return 0;
}
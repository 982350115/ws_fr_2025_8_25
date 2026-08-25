#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <thread>

// ==========================================
// ========== 在此处修改坐标 (用户配置) ==========
// ==========================================

// 1. 你的标定参考点坐标 (十字中心的坐标)
const double TARGET_X = -0.325;  // 米
const double TARGET_Y = -0.799;  // 米
const double TARGET_Z = 0.199;  // 米 (你要验证的高度，比如桌面上方1cm)

// 2. 安全高度 (Z轴)，防止移动过程中撞到东西
const double SAFE_HEIGHT = 0.20; 

// 3. 你的 TCP 名字
const std::string TARGET_EE_LINK = "gripper_center_tcp"; 
const std::string MOVE_GROUP_NAME = "manipulator";

// ==========================================

class CalibrationVerifier : public rclcpp::Node
{
public:
    CalibrationVerifier() : Node("calibration_verify_auto")
    {
        // 启动一个线程执行任务，不阻塞 ROS 节点的回调
        worker_thread_ = std::thread(&CalibrationVerifier::executePath, this);
    }

    ~CalibrationVerifier()
    {
        if (worker_thread_.joinable()) worker_thread_.join();
    }

private:
    std::thread worker_thread_;

    void executePath()
    {
        // 给 MoveIt 一点时间初始化
        rclcpp::sleep_for(std::chrono::seconds(2));

        auto move_group = std::make_shared<moveit::planning_interface::MoveGroupInterface>(shared_from_this(), MOVE_GROUP_NAME);
        
        move_group->setEndEffectorLink(TARGET_EE_LINK);
        move_group->setMaxVelocityScalingFactor(0.10); // 慢速安全
        move_group->setMaxAccelerationScalingFactor(0.10);

        RCLCPP_INFO(this->get_logger(), "===================================");
        RCLCPP_INFO(this->get_logger(), "   标定验证开始 (自动模式)");
        RCLCPP_INFO(this->get_logger(), "   目标 TCP: %s", TARGET_EE_LINK.c_str());
        RCLCPP_INFO(this->get_logger(), "   目标点: [%.3f, %.3f, %.3f]", TARGET_X, TARGET_Y, TARGET_Z);
        RCLCPP_INFO(this->get_logger(), "===================================");

        // --- 步骤 1: 移动到上方安全点 ---
        geometry_msgs::msg::Pose target_pose;
        target_pose.position.x = TARGET_X;
        target_pose.position.y = TARGET_Y;
        target_pose.position.z = SAFE_HEIGHT; // 先去高处

        // 设置姿态：垂直向下 (绕X轴转180度)
        tf2::Quaternion q;
        q.setRPY(M_PI, 0.0, 0.0); 
        target_pose.orientation = tf2::toMsg(q);

        RCLCPP_INFO(this->get_logger(), "1. 前往上方安全点 (Z=%.2f)...", SAFE_HEIGHT);
        move_group->setPoseTarget(target_pose);
        
        // 执行移动 1
        if (move_group->move() != moveit::core::MoveItErrorCode::SUCCESS) {
            RCLCPP_ERROR(this->get_logger(), "移动到安全点失败！请检查是否超出工作空间。");
            return;
        }

        rclcpp::sleep_for(std::chrono::seconds(1)); // 停顿一下

        // --- 步骤 2: 垂直下潜到标定点 ---
        target_pose.position.z = TARGET_Z; // 设置为目标高度

        RCLCPP_INFO(this->get_logger(), "2. 下潜到标定验证点 (Z=%.3f)...", TARGET_Z);
        move_group->setPoseTarget(target_pose);

        // 执行移动 2
        if (move_group->move() == moveit::core::MoveItErrorCode::SUCCESS) {
            RCLCPP_INFO(this->get_logger(), ">>> 到达目标！请拿针尖检查 TCP 是否对齐。");
        } else {
            RCLCPP_ERROR(this->get_logger(), "下潜失败！可能发生碰撞或奇异点。");
        }
    }
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<CalibrationVerifier>();
    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node);
    executor.spin(); // 保持节点存活
    rclcpp::shutdown();
    return 0;
}
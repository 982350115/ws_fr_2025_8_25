#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>
#include <trajectory_msgs/msg/joint_trajectory_point.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>

// MoveIt Headers
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit_msgs/msg/collision_object.hpp>
#include <moveit_msgs/msg/attached_collision_object.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>

// TF2 Headers
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2/LinearMath/Quaternion.h>

#include <thread>
#include <chrono>

using namespace std::chrono_literals;

// 定义常量
const std::string MOVE_GROUP_NAME = "manipulator";
const std::string GRIPPER_CONTROLLER_TOPIC = "/gripper_controller/joint_trajectory";
const std::string GRIPPER_JOINT_NAME = "gripper_left_joint";
const std::string TARGET_EE_LINK = "gripper_base_link"; // 末端执行器参考坐标系
const std::string WORLD_FRAME = "world";

// 抓取相关参数
const double EE_TO_GRASP_OFFSET_Z = 0.205; // 末端执行器中心到抓取点的Z轴偏移
const double PRE_GRASP_CLEARANCE = 0.15;   // 预抓取高度（相对于抓取点）
const double BALL_RADIUS = 0.025;          // 小球半径

// 初始小球位置 (0.5, 0.2, 0)
const double BALL_X = 0.5;
const double BALL_Y = 0.2;
const double BALL_Z = 0.0;

// 放置点位置 (-0.3, -0.4, 0)
const double PLACE_X = -0.3;
const double PLACE_Y = -0.4;
const double PLACE_Z = 0.0;

// 速度缩放因子
const double VELOCITY_SCALE = 0.5;
const double ACCELERATION_SCALE = 0.5;

class SimplePickPlace : public rclcpp::Node
{
public:
    SimplePickPlace() : Node("simple_pick_place")
    {
        // 创建夹爪控制发布者
        gripper_traj_pub_ = this->create_publisher<trajectory_msgs::msg::JointTrajectory>(GRIPPER_CONTROLLER_TOPIC, 10);

        // 使用单独的线程运行主要逻辑，避免阻塞 ROS 回调
        init_timer_ = this->create_wall_timer(
            100ms, std::bind(&SimplePickPlace::run, this));
    }

private:
    rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr gripper_traj_pub_;
    rclcpp::TimerBase::SharedPtr init_timer_;
    
    // MoveIt 接口指针
    std::shared_ptr<moveit::planning_interface::MoveGroupInterface> move_group_;
    std::shared_ptr<moveit::planning_interface::PlanningSceneInterface> planning_scene_interface_;

    void run()
    {
        init_timer_->cancel(); // 只运行一次

        // 初始化 MoveIt 接口
        move_group_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(shared_from_this(), MOVE_GROUP_NAME);
        planning_scene_interface_ = std::make_shared<moveit::planning_interface::PlanningSceneInterface>();

        // 设置最大速度和加速度
        move_group_->setMaxVelocityScalingFactor(VELOCITY_SCALE);
        move_group_->setMaxAccelerationScalingFactor(ACCELERATION_SCALE);

        // 初始化场景：添加桌子和小球
        setupPlanningScene();

        // 确保夹爪处于打开状态
        openGripper();
        rclcpp::sleep_for(1s);

        // 执行任务流程
        executeTask();
    }

    // 设置规划场景：添加桌子和小球
    void setupPlanningScene()
    {
        // 1. 添加桌子 (Z=0 平面)
        moveit_msgs::msg::CollisionObject table;
        table.id = "table";
        table.header.frame_id = move_group_->getPlanningFrame();
        table.operation = table.ADD;

        shape_msgs::msg::SolidPrimitive table_shape;
        table_shape.type = shape_msgs::msg::SolidPrimitive::BOX;
        table_shape.dimensions = {2.0, 2.0, 0.02}; // 2x2米，厚度2cm

        geometry_msgs::msg::Pose table_pose;
        table_pose.position.x = 0.0;
        table_pose.position.y = 0.0;
        table_pose.position.z = -0.01; // 桌面在 z=0，所以中心在 -0.01
        table_pose.orientation.w = 1.0;

        table.primitives.push_back(table_shape);
        table.primitive_poses.push_back(table_pose);

        // 2. 添加小球
        moveit_msgs::msg::CollisionObject ball;
        ball.id = "target_ball";
        ball.header.frame_id = move_group_->getPlanningFrame();
        ball.operation = ball.ADD;

        shape_msgs::msg::SolidPrimitive ball_shape;
        ball_shape.type = shape_msgs::msg::SolidPrimitive::SPHERE;
        ball_shape.dimensions.resize(1);
        ball_shape.dimensions[0] = BALL_RADIUS;

        geometry_msgs::msg::Pose ball_pose;
        ball_pose.position.x = BALL_X;
        ball_pose.position.y = BALL_Y;
        ball_pose.position.z = BALL_Z + BALL_RADIUS; // 球心高度
        ball_pose.orientation.w = 1.0;

        ball.primitives.push_back(ball_shape);
        ball.primitive_poses.push_back(ball_pose);

        // 应用到场景
        planning_scene_interface_->applyCollisionObjects({table, ball});
        RCLCPP_INFO(this->get_logger(), "Planning scene initialized.");
    }

    // 控制夹爪移动
    void sendGripperTrajectory(double position)
    {
        trajectory_msgs::msg::JointTrajectory traj;
        traj.header.stamp = this->get_clock()->now();
        traj.joint_names.push_back(GRIPPER_JOINT_NAME);

        trajectory_msgs::msg::JointTrajectoryPoint pt;
        pt.positions.push_back(position);
        pt.time_from_start = rclcpp::Duration::from_seconds(1.0); // 1秒完成动作

        traj.points.push_back(pt);
        gripper_traj_pub_->publish(traj);
    }

    void openGripper()
    {
        RCLCPP_INFO(this->get_logger(), "Opening gripper...");
        sendGripperTrajectory(0.0); // 0.0 为打开位置
    }

    void closeGripper()
    {
        RCLCPP_INFO(this->get_logger(), "Closing gripper...");
        sendGripperTrajectory(-0.009); // -0.009 为闭合位置（根据之前代码）
    }

    // 附着小球到末端执行器
    void attachBall()
    {
        moveit_msgs::msg::AttachedCollisionObject attached_object;
        attached_object.link_name = TARGET_EE_LINK;
        attached_object.object.header.frame_id = move_group_->getPlanningFrame();
        attached_object.object.id = "target_ball";
        attached_object.object.operation = attached_object.object.ADD;
        
        // 允许接触的连杆，防止误报碰撞
        attached_object.touch_links = {"gripper_left_link", "gripper_right_link", TARGET_EE_LINK};

        planning_scene_interface_->applyAttachedCollisionObject(attached_object);
        RCLCPP_INFO(this->get_logger(), "Ball attached.");
    }

    // 从末端执行器分离小球
    void detachBall(const geometry_msgs::msg::Pose& drop_pose)
    {
        // 1. 分离物体
        moveit_msgs::msg::AttachedCollisionObject detach_object;
        detach_object.object.id = "target_ball";
        detach_object.link_name = TARGET_EE_LINK;
        detach_object.object.operation = detach_object.object.REMOVE;
        planning_scene_interface_->applyAttachedCollisionObject(detach_object);

        // 2. 将物体重新添加到环境中的指定位置
        moveit_msgs::msg::CollisionObject ball;
        ball.id = "target_ball";
        ball.header.frame_id = move_group_->getPlanningFrame();
        ball.operation = ball.ADD;

        shape_msgs::msg::SolidPrimitive ball_shape;
        ball_shape.type = shape_msgs::msg::SolidPrimitive::SPHERE;
        ball_shape.dimensions.resize(1);
        ball_shape.dimensions[0] = BALL_RADIUS;

        ball.primitives.push_back(ball_shape);
        ball.primitive_poses.push_back(drop_pose);

        planning_scene_interface_->applyCollisionObjects({ball});
        RCLCPP_INFO(this->get_logger(), "Ball detached at new position.");
    }

    // 移动到目标位姿 (使用 MoveIt 规划)
    bool moveToPose(const geometry_msgs::msg::Pose& target_pose)
    {
        move_group_->setPoseTarget(target_pose);
        
        moveit::planning_interface::MoveGroupInterface::Plan my_plan;
        bool success = (move_group_->plan(my_plan) == moveit::core::MoveItErrorCode::SUCCESS);

        if (success)
        {
            move_group_->execute(my_plan);
            return true;
        }
        else
        {
            RCLCPP_ERROR(this->get_logger(), "Planning failed!");
            return false;
        }
    }

    // 笛卡尔直线运动
    bool moveCartesian(const std::vector<geometry_msgs::msg::Pose>& waypoints)
    {
        moveit_msgs::msg::RobotTrajectory trajectory;
        // 保持 jump_threshold = 1.5，用于容忍奇点，这是之前成功的关键
        const double jump_threshold = 1.5; 
        const double eef_step = 0.01;
        
        double fraction = move_group_->computeCartesianPath(waypoints, eef_step, jump_threshold, trajectory);
        
        if (fraction > 0.9) // 如果路径规划成功率超过90%
        {
            move_group_->execute(trajectory);
            return true;
        }
        else
        {
            RCLCPP_ERROR(this->get_logger(), "Cartesian path planning failed (fraction: %.2f)", fraction);
            return false;
        }
    }

    // 获取抓取位姿的四元数 (朝下)
    geometry_msgs::msg::Quaternion getGraspOrientation()
    {
        tf2::Quaternion q;
        q.setRPY(M_PI, 0.0, 0.0); // 绕X轴旋转180度，使夹爪朝下
        return tf2::toMsg(q);
    }

    // 执行完整的抓取-放置-返回流程
    void executeTask()
    {
        // --- 1. 定义关键位置 ---
        geometry_msgs::msg::Pose start_pose; // 小球初始位置对应的抓取位姿（地面上）
        start_pose.position.x = BALL_X;
        start_pose.position.y = BALL_Y;
        start_pose.position.z = BALL_Z + BALL_RADIUS + EE_TO_GRASP_OFFSET_Z; // Z轴需要加上末端执行器的长度
        start_pose.orientation = getGraspOrientation();

        geometry_msgs::msg::Pose place_pose; // 放置位置对应的位姿
        place_pose.position.x = PLACE_X;
        place_pose.position.y = PLACE_Y;
        // 保持之前成功的 Z 轴定义
        place_pose.position.z = PLACE_Z + BALL_RADIUS + EE_TO_GRASP_OFFSET_Z + 0.02; 
        place_pose.orientation = getGraspOrientation();

        // 预抓取点 (上方)
        geometry_msgs::msg::Pose pre_grasp_pose = start_pose;
        pre_grasp_pose.position.z += PRE_GRASP_CLEARANCE;

        // 预放置点 (上方)
        geometry_msgs::msg::Pose pre_place_pose = place_pose;
        pre_place_pose.position.z += PRE_GRASP_CLEARANCE;

        // --- 2. 开始流程 ---

        // A. 移动到预抓取位置
        RCLCPP_INFO(this->get_logger(), "1. Moving to Pre-Grasp position...");
        if (!moveToPose(pre_grasp_pose)) return;

        // B. 下降抓取
        RCLCPP_INFO(this->get_logger(), "2. Descending to Grasp...");
        std::vector<geometry_msgs::msg::Pose> waypoints;
        waypoints.push_back(start_pose);
        if (!moveCartesian(waypoints)) return;

        // C. 抓取小球
        RCLCPP_INFO(this->get_logger(), "3. Grasping ball...");
        closeGripper();
        rclcpp::sleep_for(1s);
        attachBall();

        // D. 上升
        RCLCPP_INFO(this->get_logger(), "4. Lifting ball...");
        waypoints.clear();
        waypoints.push_back(pre_grasp_pose);
        if (!moveCartesian(waypoints)) return;

        // E. 移动到放置点上方
        RCLCPP_INFO(this->get_logger(), "5. Moving to Place location...");
        if (!moveToPose(pre_place_pose)) return;

        // F. 下降放置
        RCLCPP_INFO(this->get_logger(), "6. Descending to Place...");
        waypoints.clear();
        waypoints.push_back(place_pose);
        if (!moveCartesian(waypoints)) return;

        // G. 放下小球
        RCLCPP_INFO(this->get_logger(), "7. Releasing ball...");
        openGripper();
        rclcpp::sleep_for(1s);
        
        // 计算放置后小球的实际位置用于 detach
        geometry_msgs::msg::Pose drop_ball_pose;
        drop_ball_pose.position.x = PLACE_X;
        drop_ball_pose.position.y = PLACE_Y;
        drop_ball_pose.position.z = PLACE_Z + BALL_RADIUS;
        drop_ball_pose.orientation.w = 1.0;
        detachBall(drop_ball_pose);

        // H. 上升离开
        RCLCPP_INFO(this->get_logger(), "8. Lifting after place...");
        waypoints.clear();
        waypoints.push_back(pre_place_pose);
        if (!moveCartesian(waypoints)) return;

        // --- 反向过程：从放置点抓回起始点 ---
        RCLCPP_INFO(this->get_logger(), "--- Starting Reverse Process ---");

        // I. 下降抓取 (在放置点)
        RCLCPP_INFO(this->get_logger(), "9. Descending to re-grasp...");
        waypoints.clear();
        waypoints.push_back(place_pose);
        if (!moveCartesian(waypoints)) return;

        // J. 抓取小球
        RCLCPP_INFO(this->get_logger(), "10. Re-grasping ball...");
        closeGripper();
        rclcpp::sleep_for(1s);
        attachBall();

        // K. 上升
        RCLCPP_INFO(this->get_logger(), "11. Lifting ball...");
        waypoints.clear();
        waypoints.push_back(pre_place_pose);
        if (!moveCartesian(waypoints)) return;

        // L. 移动回起始点上方
        RCLCPP_INFO(this->get_logger(), "12. Moving back to Start location...");
        if (!moveToPose(pre_grasp_pose)) return;

        // --- M. 仅调整步骤 13 的 Z 轴目标 ---
        geometry_msgs::msg::Pose adjusted_start_pose = start_pose;
        const double Z_ADJUSTMENT = 0.03; // 目标 Z 轴抬高 1cm
        adjusted_start_pose.position.z += Z_ADJUSTMENT;
        
        // M. 下降放置 (回原位)
        RCLCPP_INFO(this->get_logger(), "13. Descending to Start (Adjusted Z: %.3f)...", adjusted_start_pose.position.z);
        waypoints.clear();
        waypoints.push_back(adjusted_start_pose); // 使用调整后的姿态
        if (!moveCartesian(waypoints)) return;

        // N. 放下小球
        RCLCPP_INFO(this->get_logger(), "14. Releasing ball at Start...");
        openGripper();
        rclcpp::sleep_for(1s);
        
        // 计算原位小球位置
        geometry_msgs::msg::Pose original_ball_pose;
        original_ball_pose.position.x = BALL_X;
        original_ball_pose.position.y = BALL_Y;
        original_ball_pose.position.z = BALL_Z + BALL_RADIUS;
        original_ball_pose.orientation.w = 1.0;
        detachBall(original_ball_pose);

        // O. 上升离开
        RCLCPP_INFO(this->get_logger(), "15. Lifting away...");
        waypoints.clear();
        waypoints.push_back(pre_grasp_pose);
        if (!moveCartesian(waypoints)) return;

        // P. 回归原位 (Home)
        RCLCPP_INFO(this->get_logger(), "16. Returning to Home...");
        move_group_->setNamedTarget("home");
        moveit::planning_interface::MoveGroupInterface::Plan home_plan;
        if (move_group_->plan(home_plan) == moveit::core::MoveItErrorCode::SUCCESS)
        {
            move_group_->execute(home_plan);
            RCLCPP_INFO(this->get_logger(), "Task Finished Successfully!");
        }
        else
        {
            RCLCPP_ERROR(this->get_logger(), "Failed to plan to Home.");
        }
    }
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<SimplePickPlace>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
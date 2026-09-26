#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>

// MoveIt Headers
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit_msgs/msg/collision_object.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>

// 时间参数化 (ROS 2 Humble 推荐方式)
#include <moveit/trajectory_processing/iterative_time_parameterization.h>
#include <moveit/robot_trajectory/robot_trajectory.h>

// TF2 Headers
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2/LinearMath/Quaternion.h>

#include <thread>
#include <chrono>
#include <cmath> 

using namespace std::chrono_literals;

// ==========================================
// ========== 核心参数配置区域 ==========
// ==========================================
const std::string MOVE_GROUP_NAME = "manipulator";
const std::string TARGET_EE_LINK = "gripper_center_tcp"; 

// 运动速度与加速度控制 (极慢速，确保安全)
const double VELOCITY_SCALE = 0.05;      
const double ACCELERATION_SCALE = 0.05;  
const double CARTESIAN_VEL_SCALE = 0.02; 
const double CARTESIAN_ACC_SCALE = 0.02; 

// 到达目标点之前的悬停高度差 (单位: 米)
const double PRE_REACH_CLEARANCE = 0.10; 

class ReachValidator : public rclcpp::Node
{
public:
    ReachValidator() : Node("reach_validator_node")
    {
        cb_group_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

        target_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            "/target_pose", 10, 
            std::bind(&ReachValidator::targetCallback, this, std::placeholders::_1));

        // 延迟初始化 MoveIt 以确保节点完全启动
        init_timer_ = this->create_wall_timer(
            500ms, std::bind(&ReachValidator::run, this), cb_group_);
        
        target_received_ = false;
    }

private:
    rclcpp::TimerBase::SharedPtr init_timer_;
    rclcpp::CallbackGroup::SharedPtr cb_group_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr target_sub_;
    
    geometry_msgs::msg::Pose target_pose_raw_;
    bool target_received_;
    
    std::shared_ptr<moveit::planning_interface::MoveGroupInterface> move_group_;
    std::shared_ptr<moveit::planning_interface::PlanningSceneInterface> planning_scene_interface_;

    void targetCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
    {
        target_pose_raw_ = msg->pose;
        target_received_ = true;
        RCLCPP_INFO(this->get_logger(), "收到新目标点: X=%.3f, Y=%.3f, Z=%.3f", 
                    msg->pose.position.x, msg->pose.position.y, msg->pose.position.z);
    }

    void run()
    {
        init_timer_->cancel(); 
        
        // 初始化 MoveGroupInterface
        move_group_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(shared_from_this(), MOVE_GROUP_NAME);
        planning_scene_interface_ = std::make_shared<moveit::planning_interface::PlanningSceneInterface>();

        move_group_->setEndEffectorLink(TARGET_EE_LINK);
        move_group_->setMaxVelocityScalingFactor(VELOCITY_SCALE);
        move_group_->setMaxAccelerationScalingFactor(ACCELERATION_SCALE);

        setupBaseScene();
        executeTaskLoop();
    }

    // 保留桌面避障以防万一发生严重错误
    void setupBaseScene()
    {
        std::vector<moveit_msgs::msg::CollisionObject> collision_objects;
        moveit_msgs::msg::CollisionObject table;
        table.id = "table";
        table.header.frame_id = move_group_->getPlanningFrame();
        table.operation = table.ADD;
        shape_msgs::msg::SolidPrimitive table_shape;
        table_shape.type = shape_msgs::msg::SolidPrimitive::BOX;
        table_shape.dimensions = {2.0, 2.0, 0.02}; 
        geometry_msgs::msg::Pose table_pose;
        table_pose.position.z = -0.01; 
        table_pose.orientation.w = 1.0;
        table.primitives.push_back(table_shape);
        table.primitive_poses.push_back(table_pose);
        collision_objects.push_back(table);

        planning_scene_interface_->applyCollisionObjects(collision_objects);
        RCLCPP_INFO(this->get_logger(), "安全桌面碰撞体已加载。");
    }

    // 执行普通空间关节规划
    bool moveToPose(const geometry_msgs::msg::Pose& target_pose)
    {
        move_group_->setPoseTarget(target_pose); 
        moveit::planning_interface::MoveGroupInterface::Plan my_plan;
        if (move_group_->plan(my_plan) == moveit::core::MoveItErrorCode::SUCCESS) {
            move_group_->execute(my_plan); 
            return true;
        }
        RCLCPP_ERROR(this->get_logger(), "自由空间路径规划失败！");
        return false;
    }

    // 执行直线笛卡尔运动
    bool moveCartesian(const std::vector<geometry_msgs::msg::Pose>& waypoints)
    {
        moveit_msgs::msg::RobotTrajectory trajectory_msg;
        // eef_step = 0.01m (1cm)
        double fraction = move_group_->computeCartesianPath(waypoints, 0.01, 0.0, trajectory_msg);
        
        if (fraction > 0.8) { 
            auto robot_model = move_group_->getRobotModel();
            auto current_state = move_group_->getCurrentState(); 
            robot_trajectory::RobotTrajectory rt(robot_model, move_group_->getName());
            rt.setRobotTrajectoryMsg(*current_state, trajectory_msg);
            
            // 重新参数化轨迹速度 (Humble 中推荐的方式)
            trajectory_processing::IterativeParabolicTimeParameterization iptp;
            if (iptp.computeTimeStamps(rt, CARTESIAN_VEL_SCALE, CARTESIAN_ACC_SCALE)) {
                moveit::planning_interface::MoveGroupInterface::Plan plan;
                rt.getRobotTrajectoryMsg(plan.trajectory_);
                move_group_->execute(plan);
                return true;
            }
        }
        RCLCPP_ERROR(this->get_logger(), "笛卡尔直线路径规划失败，完成率仅为: %.2f%%", fraction * 100);
        return false;
    }

    void executeTaskLoop()
    {
        while (rclcpp::ok()) {
            RCLCPP_INFO(this->get_logger(), ">>> 等待点击目标点 <<<");
            while (rclcpp::ok() && !target_received_) {
                rclcpp::sleep_for(200ms);
            }
            if (!rclcpp::ok()) break;
            
            target_received_ = false; 
            executeVerificationMotion();
        }
    }

    void executeVerificationMotion()
    {
        RCLCPP_INFO(this->get_logger(), "开始执行到达动作...");

        // 1. 设置末端姿态 (强制垂直向下)
        // 假设 gripper_center_tcp 的 Z 轴是工具的伸出方向，那么 Roll=180(M_PI), Pitch=0, Yaw=0 通常代表 Z 轴垂直朝下。
        // 如果你的末端坐标系定义不同（比如 X 轴朝下），请在这里修改 RPY 值。
        tf2::Quaternion q_down;
        q_down.setRPY(M_PI, 0.0, 0.0); 

        // 2. 构造目标点 Pose
        geometry_msgs::msg::Pose final_pose;
        final_pose.position.x = target_pose_raw_.position.x;
        final_pose.position.y = target_pose_raw_.position.y;
        final_pose.position.z = target_pose_raw_.position.z;
        final_pose.orientation = tf2::toMsg(q_down);

        // 3. 构造预到达点 Pose (在目标点正上方)
        geometry_msgs::msg::Pose pre_pose = final_pose;
        pre_pose.position.z += PRE_REACH_CLEARANCE;

        // --- 运动序列 ---
        
        // 步骤 A: 以关节空间规划安全移动到目标上方
        RCLCPP_INFO(this->get_logger(), "移动至预到达点 (Z + %.2f m)...", PRE_REACH_CLEARANCE);
        if (!moveToPose(pre_pose)) return;
        
        // 停顿一小下，方便你观察偏差
        rclcpp::sleep_for(1s);

        // 步骤 B: 笛卡尔直线下降到目标点
        RCLCPP_INFO(this->get_logger(), "直线下降接触目标...");
        if (!moveCartesian({final_pose})) return;

        RCLCPP_INFO(this->get_logger(), "已到达！请测量误差。");
        
        // 可选：完成测量后可以写一个归位代码或原路返回
        // 比如直线退回到 pre_pose
        // rclcpp::sleep_for(3s);
        // moveCartesian({pre_pose});
    }
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ReachValidator>();
    
    // 使用 MultiThreadedExecutor 以确保 MoveIt 接口和 ROS 订阅回调能并发执行
    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node);
    executor.spin();
    
    rclcpp::shutdown();
    return 0;
}
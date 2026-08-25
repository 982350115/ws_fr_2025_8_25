#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <example_interfaces/srv/set_bool.hpp> 

// MoveIt Headers
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit_msgs/msg/collision_object.hpp>
#include <moveit_msgs/msg/attached_collision_object.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>

// 时间参数化
#include <moveit/trajectory_processing/iterative_time_parameterization.h>
#include <moveit/robot_trajectory/robot_trajectory.h>

// TF2 Headers
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2/LinearMath/Quaternion.h>

#include <thread>
#include <chrono>
#include <cmath> // 引入数学库计算 atan2

using namespace std::chrono_literals;

// ==========================================
// ========== 用户参数区域 ==========
// ==========================================

const std::string MOVE_GROUP_NAME = "manipulator";

// [TCP Link] 你的 URDF 中新增的 TCP Link 名字
const std::string TARGET_EE_LINK = "gripper_center_tcp"; 

const std::string GRIPPER_SERVICE_NAME = "/gripper_set_open"; 

// 下潜深度 (单位: 米)
// 0.0 表示 TCP 刚好停在视觉识别到的表面
const double GRASP_DIP = 0.000; 

// 工件(矿泉水瓶)参数
const double BOTTLE_HEIGHT = 0.17; 
const double BOTTLE_RADIUS = 0.01; 

// 放置位置
const double PLACE_X = 0.4;
const double PLACE_Y = 0.3;
const double PLACE_Z = 0.015; 

// ==========================================

const double VELOCITY_SCALE = 0.015; 
const double ACCELERATION_SCALE = 0.015;
const double CARTESIAN_VEL_SCALE = 0.015; 
const double CARTESIAN_ACC_SCALE = 0.015;
const double PRE_GRASP_CLEARANCE = 0.15; 

class SimplePickPlace : public rclcpp::Node
{
public:
    SimplePickPlace() : Node("simple_pick_place_tcp")
    {
        cb_group_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

        target_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            "/target_pose", 10, 
            std::bind(&SimplePickPlace::targetCallback, this, std::placeholders::_1));

        init_timer_ = this->create_wall_timer(
            100ms, std::bind(&SimplePickPlace::run, this), cb_group_);
        
        target_received_ = false;
    }

private:
    rclcpp::TimerBase::SharedPtr init_timer_;
    rclcpp::CallbackGroup::SharedPtr cb_group_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr target_sub_;
    
    geometry_msgs::msg::Pose target_pose_;
    bool target_received_;
    
    std::shared_ptr<moveit::planning_interface::MoveGroupInterface> move_group_;
    std::shared_ptr<moveit::planning_interface::PlanningSceneInterface> planning_scene_interface_;

    void targetCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
    {
        target_pose_ = msg->pose;
        target_received_ = true;
        RCLCPP_INFO(this->get_logger(), "收到视觉坐标(TCP目标) -> X: %.3f, Y: %.3f, Z: %.3f", 
            target_pose_.position.x, target_pose_.position.y, target_pose_.position.z);
    }

    void run()
    {
        init_timer_->cancel(); 

        move_group_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(shared_from_this(), MOVE_GROUP_NAME);
        planning_scene_interface_ = std::make_shared<moveit::planning_interface::PlanningSceneInterface>();

        // [关键设置] 显式设置末端连杆为 TCP
        move_group_->setEndEffectorLink(TARGET_EE_LINK);
        
        RCLCPP_INFO(this->get_logger(), "规划参考系: %s", move_group_->getPlanningFrame().c_str());
        RCLCPP_INFO(this->get_logger(), "末端执行器Link: %s", move_group_->getEndEffectorLink().c_str());

        move_group_->setMaxVelocityScalingFactor(VELOCITY_SCALE);
        move_group_->setMaxAccelerationScalingFactor(ACCELERATION_SCALE);

        setupBaseScene();
        controlGripper(true);
        executeTaskLoop();
    }

    // --- 夹爪驱动逻辑 ---
    bool controlGripper(bool open)
    {
        auto temp_node = rclcpp::Node::make_shared("temp_gripper_client");
        auto client = temp_node->create_client<example_interfaces::srv::SetBool>(GRIPPER_SERVICE_NAME);

        if (!client->wait_for_service(std::chrono::seconds(1))) {
            RCLCPP_ERROR(this->get_logger(), "连接夹爪服务失败！");
            return false;
        }

        auto request = std::make_shared<example_interfaces::srv::SetBool::Request>();
        request->data = open; 
        auto result_future = client->async_send_request(request);

        if (rclcpp::spin_until_future_complete(temp_node, result_future) == rclcpp::FutureReturnCode::SUCCESS) {
            auto response = result_future.get();
            if (response->success) {
                RCLCPP_INFO(this->get_logger(), "夹爪动作: %s", response->message.c_str());
                return true;
            }
        }
        return false;
    }
    void openGripper() { controlGripper(true); }
    void closeGripper() { controlGripper(false); }

    // --- 场景建模 ---
    void setupBaseScene()
    {
        moveit_msgs::msg::CollisionObject table;
        table.id = "table";
        table.header.frame_id = move_group_->getPlanningFrame();
        table.operation = table.ADD;
        shape_msgs::msg::SolidPrimitive table_shape;
        table_shape.type = shape_msgs::msg::SolidPrimitive::BOX;
        table_shape.dimensions = {2.0, 2.0, 0.02}; 
        geometry_msgs::msg::Pose table_pose;
        table_pose.position.x = 0.0; table_pose.position.y = 0.0; table_pose.position.z = -0.01; table_pose.orientation.w = 1.0;
        table.primitives.push_back(table_shape); table.primitive_poses.push_back(table_pose);

        planning_scene_interface_->applyCollisionObjects({table});
        RCLCPP_INFO(this->get_logger(), "环境初始化完成...");
    }

    void addTargetBottle(const geometry_msgs::msg::Pose& click_pose)
    {
        moveit_msgs::msg::CollisionObject bottle;
        bottle.id = "target_bottle";
        bottle.header.frame_id = move_group_->getPlanningFrame();
        bottle.operation = bottle.ADD;
        
        shape_msgs::msg::SolidPrimitive shape;
        shape.type = shape_msgs::msg::SolidPrimitive::CYLINDER;
        shape.dimensions.resize(2);
        shape.dimensions[0] = BOTTLE_HEIGHT; 
        shape.dimensions[1] = BOTTLE_RADIUS; 
        
        geometry_msgs::msg::Pose pose;
        pose.position.x = click_pose.position.x;
        pose.position.y = click_pose.position.y;
        pose.position.z = click_pose.position.z - (BOTTLE_HEIGHT / 2.0);
        pose.orientation.w = 1.0; 

        bottle.primitives.push_back(shape); 
        bottle.primitive_poses.push_back(pose);

        planning_scene_interface_->applyCollisionObjects({bottle});
    }

    void attachBottle()
    {
        moveit_msgs::msg::AttachedCollisionObject attached_object;
        attached_object.link_name = TARGET_EE_LINK; // 附着在 TCP 上
        attached_object.object.header.frame_id = move_group_->getPlanningFrame();
        attached_object.object.id = "target_bottle";
        attached_object.object.operation = attached_object.object.ADD;
        attached_object.touch_links = {"gripper_left_link", "gripper_right_link", TARGET_EE_LINK}; 
        planning_scene_interface_->applyAttachedCollisionObject(attached_object);
    }

    void detachBottle()
    {
        moveit_msgs::msg::AttachedCollisionObject detach_object;
        detach_object.object.id = "target_bottle";
        detach_object.link_name = TARGET_EE_LINK;
        detach_object.object.operation = detach_object.object.REMOVE;
        planning_scene_interface_->applyAttachedCollisionObject(detach_object);

        moveit_msgs::msg::CollisionObject remove_obj;
        remove_obj.id = "target_bottle";
        remove_obj.operation = remove_obj.REMOVE;
        planning_scene_interface_->applyCollisionObjects({remove_obj});
    }

    // --- 运动控制 ---
    bool moveToPose(const geometry_msgs::msg::Pose& target_pose)
    {
        move_group_->setPoseTarget(target_pose); 
        moveit::planning_interface::MoveGroupInterface::Plan my_plan;
        if (move_group_->plan(my_plan) == moveit::core::MoveItErrorCode::SUCCESS) {
            move_group_->execute(my_plan); return true;
        } else {
            RCLCPP_ERROR(this->get_logger(), "点对点规划失败!"); return false;
        }
    }

    bool moveCartesian(const std::vector<geometry_msgs::msg::Pose>& waypoints)
    {
        moveit_msgs::msg::RobotTrajectory trajectory_msg;
        const double jump_threshold = 0.0; 
        const double eef_step = 0.01;      
        
        double fraction = move_group_->computeCartesianPath(waypoints, eef_step, jump_threshold, trajectory_msg);
        
        if (fraction > 0.8) { 
            auto robot_model = move_group_->getRobotModel();
            auto current_state = move_group_->getCurrentState(); 
            if (!robot_model || !current_state) return false;

            robot_trajectory::RobotTrajectory rt(robot_model, move_group_->getName());
            rt.setRobotTrajectoryMsg(*current_state, trajectory_msg);

            trajectory_processing::IterativeParabolicTimeParameterization iptp;
            bool success = iptp.computeTimeStamps(rt, CARTESIAN_VEL_SCALE, CARTESIAN_ACC_SCALE);

            if (success) {
                moveit::planning_interface::MoveGroupInterface::Plan plan;
                rt.getRobotTrajectoryMsg(plan.trajectory_);
                move_group_->execute(plan);
                return true;
            }
        }
        RCLCPP_ERROR(this->get_logger(), "直线规划失败 (%.2f)", fraction);
        return false;
    }

    bool goHomeJoints()
    {
        // 你的自定义安全点
        std::vector<double> target_joints = {0.6433, -2.04713, -1.04908, -1.61334, 1.57927, 1.15};
        move_group_->setJointValueTarget(target_joints);
        
        moveit::planning_interface::MoveGroupInterface::Plan my_plan;
        if (move_group_->plan(my_plan) == moveit::core::MoveItErrorCode::SUCCESS) {
            move_group_->execute(my_plan); return true;
        }
        return false;
    }

    void executeTaskLoop()
    {
        while (rclcpp::ok()) {
            RCLCPP_INFO(this->get_logger(), ">>> 等待点击 (Waiting for /target_pose) <<<");
            while (rclcpp::ok() && !target_received_) {
                rclcpp::sleep_for(200ms);
            }
            if (!rclcpp::ok()) break;

            target_received_ = false; 
            executeSinglePickAndPlace();
            rclcpp::sleep_for(1s);
        }
    }

    // --- 核心逻辑：单次抓取 ---
    void executeSinglePickAndPlace()
    {
        RCLCPP_INFO(this->get_logger(), "启动 TCP 抓取序列...");

        addTargetBottle(target_pose_);

        // 1. 计算抓取位姿
        geometry_msgs::msg::Pose grasp_pose;
        grasp_pose.position.x = target_pose_.position.x;
        grasp_pose.position.y = target_pose_.position.y;
        grasp_pose.position.z = target_pose_.position.z - GRASP_DIP;
        
        // [修复] 姿态计算：Z轴朝上(Roll=0)，Yaw轴顺着手臂方向(atan2)
        tf2::Quaternion q;
        double grasp_yaw = std::atan2(target_pose_.position.y, target_pose_.position.x);
        q.setRPY(0.0, 0.0, grasp_yaw); 
        grasp_pose.orientation = tf2::toMsg(q);

        // 2. 计算放置位姿
        geometry_msgs::msg::Pose place_pose;
        place_pose.position.x = PLACE_X;
        place_pose.position.y = PLACE_Y;
        place_pose.position.z = 0.0 + BOTTLE_HEIGHT + 0.035; // 0.22m
        
        // [优化] 放置时也顺着放置点的方向
        tf2::Quaternion q_place;
        double place_yaw = std::atan2(PLACE_Y, PLACE_X);
        q_place.setRPY(0.0, 0.0, place_yaw);
        place_pose.orientation = tf2::toMsg(q_place);

        // 预备点
        geometry_msgs::msg::Pose pre_grasp_pose = grasp_pose;
        pre_grasp_pose.position.z += PRE_GRASP_CLEARANCE;

        geometry_msgs::msg::Pose pre_place_pose = place_pose;
        pre_place_pose.position.z += PRE_GRASP_CLEARANCE;

        // --- 动作序列 ---
        RCLCPP_INFO(this->get_logger(), "1. TCP 移动到上方...");
        if (!moveToPose(pre_grasp_pose)) return;

        RCLCPP_INFO(this->get_logger(), "2. TCP 下降...");
        std::vector<geometry_msgs::msg::Pose> waypoints;
        waypoints.push_back(grasp_pose);
        if (!moveCartesian(waypoints)) return;

        RCLCPP_INFO(this->get_logger(), "3. 闭合...");
        closeGripper();
        rclcpp::sleep_for(500ms);
        attachBottle();

        RCLCPP_INFO(this->get_logger(), "4. 抬起...");
        waypoints.clear();
        waypoints.push_back(pre_grasp_pose);
        if (!moveCartesian(waypoints)) return;

        RCLCPP_INFO(this->get_logger(), "5. 运送...");
        if (!moveToPose(pre_place_pose)) return;

        RCLCPP_INFO(this->get_logger(), "6. 下降...");
        waypoints.clear();
        waypoints.push_back(place_pose);
        if (!moveCartesian(waypoints)) return;

        RCLCPP_INFO(this->get_logger(), "7. 释放...");
        openGripper();
        rclcpp::sleep_for(500ms);
        detachBottle();

        RCLCPP_INFO(this->get_logger(), "8. 离开...");
        waypoints.clear();
        waypoints.push_back(pre_place_pose);
        moveCartesian(waypoints);

        RCLCPP_INFO(this->get_logger(), "9. 回 Home...");
        goHomeJoints();
    }
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<SimplePickPlace>();
    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node);
    executor.spin();
    rclcpp::shutdown();
    return 0;
}
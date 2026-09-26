#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
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

// 可视化工具
#include <moveit_visual_tools/moveit_visual_tools.h>

// TF2 Headers
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2/LinearMath/Quaternion.h>

#include <thread>
#include <chrono>
#include <cmath> 

using namespace std::chrono_literals;

// ==========================================
// ========== 用户参数区域 ==========
// ==========================================
const std::string MOVE_GROUP_NAME = "manipulator";
const std::string TARGET_EE_LINK = "gripper_center_tcp"; 
const std::string GRIPPER_SERVICE_NAME = "/gripper_set_open"; 

// 下潜深度 (单位: 米)
const double GRASP_DIP = 0.000; 
// 放置时比抓取点抬高的高度 (单位: 米)
const double PLACE_HEIGHT_OFFSET = 0.010;

// 工件参数
const double BOTTLE_HEIGHT = 0.17; 
const double BOTTLE_RADIUS = 0.01; 

// --- [速度参数已修改为极慢速 0.005] ---
const double VELOCITY_SCALE = 0.005;      // 关节运动速度百分比
const double ACCELERATION_SCALE = 0.005;  // 关节运动加速度百分比
const double CARTESIAN_VEL_SCALE = 0.005; // 笛卡尔运动速度百分比
const double CARTESIAN_ACC_SCALE = 0.005; // 笛卡尔运动加速度百分比

const double PRE_GRASP_CLEARANCE = 0.15; 

class SimplePickPlace : public rclcpp::Node
{
public:
    SimplePickPlace() : Node("simple_pick_place_tcp")
    {
        cb_group_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

        pick_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            "/target_pick_pose", 10, 
            std::bind(&SimplePickPlace::pickCallback, this, std::placeholders::_1));

        place_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            "/target_place_pose", 10, 
            std::bind(&SimplePickPlace::placeCallback, this, std::placeholders::_1));

        init_timer_ = this->create_wall_timer(
            100ms, std::bind(&SimplePickPlace::run, this), cb_group_);
        
        pick_received_ = false;
        place_received_ = false;
    }

private:
    rclcpp::TimerBase::SharedPtr init_timer_;
    rclcpp::CallbackGroup::SharedPtr cb_group_;
    
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr pick_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr place_sub_;
    
    geometry_msgs::msg::Pose pick_pose_raw_;
    geometry_msgs::msg::Pose place_pose_raw_;
    bool pick_received_;
    bool place_received_;
    
    std::shared_ptr<moveit::planning_interface::MoveGroupInterface> move_group_;
    std::shared_ptr<moveit::planning_interface::PlanningSceneInterface> planning_scene_interface_;
    
    std::shared_ptr<moveit_visual_tools::MoveItVisualTools> visual_tools_;

    void pickCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
    {
        pick_pose_raw_ = msg->pose;
        pick_received_ = true;
        RCLCPP_INFO(this->get_logger(), "已收到 PICK 坐标: X=%.3f, Y=%.3f, Z=%.3f", 
                    msg->pose.position.x, msg->pose.position.y, msg->pose.position.z);
    }

    void placeCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
    {
        place_pose_raw_ = msg->pose;
        place_received_ = true;
        RCLCPP_INFO(this->get_logger(), "已收到 PLACE 坐标 -> 准备执行任务");
    }

    void run()
    {
        init_timer_->cancel(); 
        move_group_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(shared_from_this(), MOVE_GROUP_NAME);
        planning_scene_interface_ = std::make_shared<moveit::planning_interface::PlanningSceneInterface>();

        // 修正初始化：传入 RobotModel
        visual_tools_ = std::make_shared<moveit_visual_tools::MoveItVisualTools>(
            shared_from_this(), "world", rviz_visual_tools::RVIZ_MARKER_TOPIC, move_group_->getRobotModel());
        visual_tools_->deleteAllMarkers();
        visual_tools_->trigger();

        move_group_->setEndEffectorLink(TARGET_EE_LINK);
        move_group_->setMaxVelocityScalingFactor(VELOCITY_SCALE);
        move_group_->setMaxAccelerationScalingFactor(ACCELERATION_SCALE);

        setupBaseScene();
        controlGripper(true);
        executeTaskLoop();
    }

    bool controlGripper(bool open)
    {
        auto temp_node = rclcpp::Node::make_shared("temp_gripper_client");
        auto client = temp_node->create_client<example_interfaces::srv::SetBool>(GRIPPER_SERVICE_NAME);
        if (!client->wait_for_service(std::chrono::seconds(1))) return false;
        auto request = std::make_shared<example_interfaces::srv::SetBool::Request>();
        request->data = open; 
        auto result_future = client->async_send_request(request);
        if (rclcpp::spin_until_future_complete(temp_node, result_future) == rclcpp::FutureReturnCode::SUCCESS) return true;
        return false;
    }

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
        table_pose.position.z = -0.01; table_pose.orientation.w = 1.0;
        table.primitives.push_back(table_shape);
        table.primitive_poses.push_back(table_pose);
        collision_objects.push_back(table);

        moveit_msgs::msg::CollisionObject camera_obs;
        camera_obs.id = "physical_camera";
        camera_obs.header.frame_id = "world"; 
        camera_obs.operation = camera_obs.ADD;
        shape_msgs::msg::SolidPrimitive cam_shape;
        cam_shape.type = shape_msgs::msg::SolidPrimitive::BOX;
        cam_shape.dimensions = {0.14, 0.10, 0.12}; 

        geometry_msgs::msg::Pose cam_pose;
        cam_pose.position.x = -0.195918;
        cam_pose.position.y = -0.579599;
        cam_pose.position.z = 0.838521;
        cam_pose.orientation.w = 1.0; 

        camera_obs.primitives.push_back(cam_shape);
        camera_obs.primitive_poses.push_back(cam_pose);
        collision_objects.push_back(camera_obs);

        planning_scene_interface_->applyCollisionObjects(collision_objects);
        RCLCPP_INFO(this->get_logger(), "已加载避障场景（包含相机障碍物）");
    }

    void addTargetBottle(const geometry_msgs::msg::Pose& click_pose)
    {
        moveit_msgs::msg::CollisionObject bottle;
        bottle.id = "target_bottle";
        bottle.header.frame_id = move_group_->getPlanningFrame();
        bottle.operation = bottle.ADD;
        shape_msgs::msg::SolidPrimitive shape;
        shape.type = shape_msgs::msg::SolidPrimitive::CYLINDER;
        shape.dimensions = {BOTTLE_HEIGHT, BOTTLE_RADIUS};
        geometry_msgs::msg::Pose pose = click_pose;
        pose.position.z = click_pose.position.z - (BOTTLE_HEIGHT / 2.0);
        pose.orientation.w = 1.0; 
        bottle.primitives.push_back(shape); bottle.primitive_poses.push_back(pose);
        planning_scene_interface_->applyCollisionObjects({bottle});
    }

    void attachBottle() {
        moveit_msgs::msg::AttachedCollisionObject att;
        att.link_name = TARGET_EE_LINK;
        att.object.header.frame_id = move_group_->getPlanningFrame();
        att.object.id = "target_bottle";
        att.object.operation = att.object.ADD;
        att.touch_links = {"gripper_left_link", "gripper_right_link", TARGET_EE_LINK}; 
        planning_scene_interface_->applyAttachedCollisionObject(att);
    }

    void detachBottle() {
        moveit_msgs::msg::AttachedCollisionObject det;
        det.object.id = "target_bottle";
        det.link_name = TARGET_EE_LINK;
        det.object.operation = det.object.REMOVE;
        planning_scene_interface_->applyAttachedCollisionObject(det);
        moveit_msgs::msg::CollisionObject rem;
        rem.id = "target_bottle"; rem.operation = rem.REMOVE;
        planning_scene_interface_->applyCollisionObjects({rem});
    }

    bool moveToPose(const geometry_msgs::msg::Pose& target_pose)
    {
        move_group_->setPoseTarget(target_pose); 
        moveit::planning_interface::MoveGroupInterface::Plan my_plan;
        if (move_group_->plan(my_plan) == moveit::core::MoveItErrorCode::SUCCESS) {
            visual_tools_->publishTrajectoryLine(
                my_plan.trajectory_, 
                move_group_->getRobotModel()->getJointModelGroup(MOVE_GROUP_NAME), 
                rviz_visual_tools::LIME_GREEN);
            visual_tools_->trigger();
            
            move_group_->execute(my_plan); return true;
        }
        return false;
    }

    bool moveCartesian(const std::vector<geometry_msgs::msg::Pose>& waypoints)
    {
        moveit_msgs::msg::RobotTrajectory trajectory_msg;
        double fraction = move_group_->computeCartesianPath(waypoints, 0.01, 0.0, trajectory_msg);
        if (fraction > 0.8) { 
            auto robot_model = move_group_->getRobotModel();
            auto current_state = move_group_->getCurrentState(); 
            robot_trajectory::RobotTrajectory rt(robot_model, move_group_->getName());
            rt.setRobotTrajectoryMsg(*current_state, trajectory_msg);
            trajectory_processing::IterativeParabolicTimeParameterization iptp;
            if (iptp.computeTimeStamps(rt, CARTESIAN_VEL_SCALE, CARTESIAN_ACC_SCALE)) {
                moveit::planning_interface::MoveGroupInterface::Plan plan;
                rt.getRobotTrajectoryMsg(plan.trajectory_);
                
                visual_tools_->publishTrajectoryLine(
                    plan.trajectory_, 
                    robot_model->getJointModelGroup(MOVE_GROUP_NAME), 
                    rviz_visual_tools::CYAN);
                visual_tools_->trigger();

                move_group_->execute(plan);
                return true;
            }
        }
        return false;
    }

    void executeTaskLoop()
    {
        while (rclcpp::ok()) {
            RCLCPP_INFO(this->get_logger(), ">>> 等待指令 <<<");
            while (rclcpp::ok() && (!pick_received_ || !place_received_)) {
                rclcpp::sleep_for(200ms);
            }
            if (!rclcpp::ok()) break;
            pick_received_ = false; place_received_ = false;
            executeSinglePickAndPlace();
            rclcpp::sleep_for(1s);
        }
    }

    void executeSinglePickAndPlace()
    {
        RCLCPP_INFO(this->get_logger(), "开始任务 [已应用标定数据]");
        addTargetBottle(pick_pose_raw_);

        geometry_msgs::msg::Pose grasp_pose;
        grasp_pose.position.x = pick_pose_raw_.position.x;
        grasp_pose.position.y = pick_pose_raw_.position.y;
        grasp_pose.position.z = pick_pose_raw_.position.z - GRASP_DIP;
        
        tf2::Quaternion q_pick;
        double pick_yaw = std::atan2(pick_pose_raw_.position.y, pick_pose_raw_.position.x);
        q_pick.setRPY(0.0, 0.0, pick_yaw); 
        grasp_pose.orientation = tf2::toMsg(q_pick);

        geometry_msgs::msg::Pose place_pose;
        place_pose.position.x = place_pose_raw_.position.x;
        place_pose.position.y = place_pose_raw_.position.y;
        place_pose.position.z = grasp_pose.position.z + PLACE_HEIGHT_OFFSET;

        tf2::Quaternion q_place;
        double place_yaw = std::atan2(place_pose_raw_.position.y, place_pose_raw_.position.x);
        q_place.setRPY(0.0, 0.0, place_yaw);
        place_pose.orientation = tf2::toMsg(q_place);

        geometry_msgs::msg::Pose pre_grasp_pose = grasp_pose;
        pre_grasp_pose.position.z += PRE_GRASP_CLEARANCE;
        geometry_msgs::msg::Pose pre_place_pose = place_pose;
        pre_place_pose.position.z += PRE_GRASP_CLEARANCE;

        if (!moveToPose(pre_grasp_pose)) return;
        if (!moveCartesian({grasp_pose})) return;
        
        auto temp_node = rclcpp::Node::make_shared("gripper_client");
        auto client = temp_node->create_client<example_interfaces::srv::SetBool>(GRIPPER_SERVICE_NAME);
        auto request = std::make_shared<example_interfaces::srv::SetBool::Request>();
        request->data = false; // Close
        client->async_send_request(request);
        
        rclcpp::sleep_for(500ms);
        attachBottle();
        
        if (!moveCartesian({pre_grasp_pose})) return;
        if (!moveToPose(pre_place_pose)) return;
        if (!moveCartesian({place_pose})) return;
        
        request->data = true; // Open
        client->async_send_request(request);
        
        rclcpp::sleep_for(500ms);
        detachBottle();
        moveCartesian({pre_place_pose});
        
        std::vector<double> home_joints = {-1.3487, -1.6333, -1.18826, -1.899669, 1.569767, 0.84196};
        move_group_->setJointValueTarget(home_joints);
        move_group_->move();
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
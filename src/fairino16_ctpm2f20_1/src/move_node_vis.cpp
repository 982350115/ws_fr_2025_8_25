#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>
#include <trajectory_msgs/msg/joint_trajectory_point.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
// #include <std_srvs/srv/set_bool.hpp> // ❌ [移除] 力控服务不再需要

// MoveIt Headers
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit_msgs/msg/collision_object.hpp>
#include <moveit_msgs/msg/attached_collision_object.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>

// TF2 Headers
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2/LinearMath/Quaternion.h>

// Gazebo Service Headers
#include <gazebo_msgs/srv/set_entity_state.hpp>
#include <gazebo_msgs/msg/entity_state.hpp>

// 可视化头文件
#include <moveit_visual_tools/moveit_visual_tools.h>
#include <thread>
#include <atomic>
#include <chrono>
#include <mutex>

using namespace std::chrono_literals;

class MoveToTargetNode : public rclcpp::Node
{
private:
    // MoveIt components
    moveit::planning_interface::MoveGroupInterfacePtr move_group_;
    moveit::planning_interface::PlanningSceneInterfacePtr planning_scene_interface_;
    std::shared_ptr<moveit_visual_tools::MoveItVisualTools> moveit_visual_tools_;

    // ROS 2 components
    rclcpp::TimerBase::SharedPtr init_timer_;
    rclcpp::TimerBase::SharedPtr gazebo_update_timer_;
    
    // ✅ [恢复] 位置控制轨迹发布者
    rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr gripper_traj_pub_;
    
    // ❌ [删除] 力控服务客户端
    // rclcpp::Client<std_srvs::srv::SetBool>::SharedPtr force_control_client_;

    rclcpp::Client<gazebo_msgs::srv::SetEntityState>::SharedPtr set_entity_state_client_;
    rclcpp::Subscription<geometry_msgs::msg::PointStamped>::SharedPtr ball_pose_sub_;

    // TF2 components
    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

    // State management
    std::atomic<bool> is_carrying_{false};
    std::atomic<bool> ball_pose_received_{false};

    geometry_msgs::msg::Pose current_ball_pose_;
    std::mutex ball_pose_mutex_;

    // --- 参数 / 常量定义 ---
    const std::string move_group_name_ = "manipulator";
    
    // ✅ [恢复] 夹爪控制话题和关节名称
    const std::string gripper_controller_topic_ = "/gripper_controller/joint_trajectory";
    const std::string gripper_joint_name_ = "gripper_left_joint"; 
    
    const std::string target_ee_link_ = "gripper_base_link";
    const std::string world_frame_ = "world";
    
    const double ee_to_grasp_offset_Z_ = 0.192;
    const std::string gazebo_model_name_ = "target_ball";

    const double ball_radius_ = 0.025;
    const geometry_msgs::msg::Pose place_pose_ = [](){
        geometry_msgs::msg::Pose p;
        p.position.x = -0.3; p.position.y = -0.4; p.position.z = 0.825;
        p.orientation.w = 1.0; return p;
    }();
    
    const double pre_grasp_clearance_ = 0.20;
    const double pre_grasp_velocity_ = 0.3;
    const double approach_velocity_ = 0.10;
    const double lift_velocity_ = 0.25;

public:
    MoveToTargetNode() : rclcpp::Node("move_to_target_node")
    {
        init_timer_ = this->create_wall_timer(
            100ms, std::bind(&MoveToTargetNode::initialize_and_start, this));
    }

private:
    void ball_pose_callback(const geometry_msgs::msg::PointStamped::SharedPtr msg)
    {
        std::lock_guard<std::mutex> lock(ball_pose_mutex_);
        current_ball_pose_.position.x = msg->point.x;
        current_ball_pose_.position.y = msg->point.y;
        current_ball_pose_.position.z = msg->point.z;
        current_ball_pose_.orientation.w = 1.0;
        
        if (!ball_pose_received_.load()) {
            ball_pose_received_ = true;
            RCLCPP_INFO(this->get_logger(), "Initial ball pose received.");
        }
    }

    void initialize_and_start()
    {
        init_timer_.reset();

        move_group_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(shared_from_this(), move_group_name_);
        planning_scene_interface_ = std::make_shared<moveit::planning_interface::PlanningSceneInterface>();

        moveit_visual_tools_ = std::make_shared<moveit_visual_tools::MoveItVisualTools>(
            shared_from_this(), 
            world_frame_,       
            rviz_visual_tools::RVIZ_MARKER_TOPIC,
            move_group_->getRobotModel());
            
        moveit_visual_tools_->deleteAllMarkers();
        moveit_visual_tools_->loadRemoteControl();
        
        tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        // ✅ [恢复] 创建轨迹发布者
        gripper_traj_pub_ = this->create_publisher<trajectory_msgs::msg::JointTrajectory>(gripper_controller_topic_, 10);
        RCLCPP_INFO(this->get_logger(), "Position Control Publisher created on: %s", gripper_controller_topic_.c_str());
        
        // ❌ [删除] 力控服务客户端
        // force_control_client_ = this->create_client<std_srvs::srv::SetBool>("toggle_force_control");

        set_entity_state_client_ = this->create_client<gazebo_msgs::srv::SetEntityState>("/set_entity_state");

        ball_pose_sub_ = this->create_subscription<geometry_msgs::msg::PointStamped>(
            "/ball_detector/ball_pose", 1,
            std::bind(&MoveToTargetNode::ball_pose_callback, this, std::placeholders::_1));

        gazebo_update_timer_ = this->create_wall_timer(
            50ms, std::bind(&MoveToTargetNode::continuousGazeboUpdate, this));

        // 初始化：打开夹爪
        openGripper();
        rclcpp::sleep_for(200ms);

        addTableAndBallToPlanningScene();

        std::thread([this](){
            rclcpp::sleep_for(1s);
            this->pickPlaceSequence();
        }).detach();
    }
    
    void continuousGazeboUpdate()
    {
        if (!is_carrying_.load()) return;
        
        if (!set_entity_state_client_->wait_for_service(0s)) return;
        
        geometry_msgs::msg::TransformStamped transformStamped;
        try {
            transformStamped = tf_buffer_->lookupTransform(world_frame_, target_ee_link_, tf2::TimePointZero);
        } catch (const tf2::TransformException &ex) { return; }
        
        geometry_msgs::msg::Pose ball_pose_in_world;
        geometry_msgs::msg::Pose gripper_pose;
        gripper_pose.position.x = transformStamped.transform.translation.x;
        gripper_pose.position.y = transformStamped.transform.translation.y;
        gripper_pose.position.z = transformStamped.transform.translation.z;
        gripper_pose.orientation = transformStamped.transform.rotation;
        
        tf2::Transform gripper_tf;
        tf2::fromMsg(gripper_pose, gripper_tf);

        tf2::Transform ball_offset_tf;
        ball_offset_tf.setIdentity();
        ball_offset_tf.setOrigin(tf2::Vector3(0.0, 0.0, ee_to_grasp_offset_Z_));

        tf2::Transform ball_world_tf = gripper_tf * ball_offset_tf;
        tf2::toMsg(ball_world_tf, ball_pose_in_world);

        auto request = std::make_shared<gazebo_msgs::srv::SetEntityState::Request>();
        request->state.name = gazebo_model_name_;
        request->state.pose = ball_pose_in_world;
        request->state.reference_frame = world_frame_;
        
        set_entity_state_client_->async_send_request(request);
    }

    void addTableAndBallToPlanningScene() 
    {
        moveit_msgs::msg::CollisionObject table;
        table.id = "work_table";
        table.header.frame_id = move_group_->getPlanningFrame();
        shape_msgs::msg::SolidPrimitive table_primitive;
        table_primitive.type = shape_msgs::msg::SolidPrimitive::BOX;
        table_primitive.dimensions = {2.0, 2.0, 0.05};
        geometry_msgs::msg::Pose table_pose;
        table_pose.orientation.w = 1.0;
        table_pose.position.z = 0.75;
        table.primitives.push_back(table_primitive);
        table.primitive_poses.push_back(table_pose);
        table.operation = table.ADD;
        planning_scene_interface_->applyCollisionObjects({table});

        moveit_msgs::msg::CollisionObject ball_co;
        ball_co.id = "target_ball";
        ball_co.header.frame_id = move_group_->getPlanningFrame();
        shape_msgs::msg::SolidPrimitive sph;
        sph.type = sph.SPHERE;
        sph.dimensions.resize(1);
        sph.dimensions[0] = ball_radius_;
        ball_co.primitives.push_back(sph);
        ball_co.primitive_poses.push_back(place_pose_);
        ball_co.operation = ball_co.ADD;
        planning_scene_interface_->applyCollisionObjects({ball_co});
    }

    void setGazeboBallPoseOnce(const geometry_msgs::msg::Pose& pose) {
        if (!set_entity_state_client_->wait_for_service(1s)) return;
        auto request = std::make_shared<gazebo_msgs::srv::SetEntityState::Request>();
        request->state.name = gazebo_model_name_;
        request->state.pose = pose;
        request->state.reference_frame = world_frame_;
        set_entity_state_client_->async_send_request(request);
    }

    // ✅ [恢复] 发送位置指令的通用函数
    void sendGripperTrajectory(double position, double time_from_start_sec = 0.5)
    {
        trajectory_msgs::msg::JointTrajectory traj;
        traj.header.stamp = this->get_clock()->now();
        traj.joint_names.push_back(gripper_joint_name_); 

        trajectory_msgs::msg::JointTrajectoryPoint pt;
        pt.positions.push_back(position);
        pt.time_from_start = rclcpp::Duration::from_seconds(time_from_start_sec);
        
        traj.points.push_back(pt);

        if (gripper_traj_pub_) {
            gripper_traj_pub_->publish(traj);
            RCLCPP_DEBUG(this->get_logger(), "Published gripper trajectory: pos=%.4f", position);
        }
    }

    // ❌ [删除] 力控服务调用函数
    // void callForceControlService(bool enable) { ... }

    // ✅ [修改] 打开夹爪 = 位置控制 (0.0)
    void openGripper()
    {
        RCLCPP_INFO(this->get_logger(), "Opening gripper (Position Control)...");
        sendGripperTrajectory(0.0, 1.0); 
        rclcpp::sleep_for(1000ms); 
    }

    // ✅ [修改] 关闭夹爪 = 位置控制 (负值)
    // 根据你的经验值，-0.009 是你之前用的数值，或者你可以尝试 -0.8
    void closeGripper()
    {
        RCLCPP_INFO(this->get_logger(), "Closing gripper (Position Control)...");
        sendGripperTrajectory(-0.009, 1.0); // 如果抓不紧，可以把这个值调得更小，例如 -0.8
        rclcpp::sleep_for(1000ms); 
    }

    bool moveLinkToPose(const geometry_msgs::msg::Pose& target_pose, const std::string& link_name, double velocity_scale, 
                         moveit::planning_interface::MoveGroupInterface::Plan& plan_out, double planning_time = 6.0)
    {
        move_group_->setEndEffectorLink(link_name);
        move_group_->setMaxVelocityScalingFactor(velocity_scale);
        move_group_->setMaxAccelerationScalingFactor(velocity_scale);
        move_group_->setPlanningTime(planning_time);
        move_group_->setPoseTarget(target_pose, link_name);
        auto res = move_group_->plan(plan_out);
        if (res != moveit::core::MoveItErrorCode::SUCCESS) {
            move_group_->clearPoseTargets();
            return false;
        }
        move_group_->clearPoseTargets();
        return true;
    }

    void attachBall()
    {
        moveit_msgs::msg::AttachedCollisionObject attached_ball;
        attached_ball.link_name = target_ee_link_;
        attached_ball.object.id = "target_ball";
        attached_ball.object.operation = moveit_msgs::msg::CollisionObject::ADD;
        attached_ball.touch_links = { target_ee_link_, "gripper_left_link", "gripper_right_link" };
        planning_scene_interface_->applyAttachedCollisionObject(attached_ball);
        is_carrying_ = true;
    }

    void detachBall()
    {
        is_carrying_ = false;
        moveit_msgs::msg::AttachedCollisionObject detached_ball;
        detached_ball.object.id = "target_ball";
        detached_ball.object.operation = moveit_msgs::msg::CollisionObject::REMOVE;
        planning_scene_interface_->applyAttachedCollisionObject(detached_ball);
        moveit_msgs::msg::CollisionObject placed_ball_co;
        placed_ball_co.id = "target_ball";
        placed_ball_co.header.frame_id = move_group_->getPlanningFrame();
        shape_msgs::msg::SolidPrimitive sph;
        sph.type = sph.SPHERE;
        sph.dimensions.resize(1);
        sph.dimensions[0] = ball_radius_;
        placed_ball_co.primitives.push_back(sph);
        placed_ball_co.primitive_poses.push_back(place_pose_);
        placed_ball_co.operation = placed_ball_co.ADD;
        planning_scene_interface_->applyCollisionObjects({placed_ball_co});
        setGazeboBallPoseOnce(place_pose_);
    }

    void pickPlaceSequence()
    {
        auto const draw_title = [this](auto text) {
            if (!moveit_visual_tools_) return;
            auto msg = Eigen::Isometry3d::Identity();
            msg.translation().z() = 1.0; 
            moveit_visual_tools_->publishText(msg, text, rviz_visual_tools::WHITE, rviz_visual_tools::XLARGE);
            moveit_visual_tools_->trigger();
        };
        auto const prompt = [this](auto text) {
            if (!moveit_visual_tools_) return;
            moveit_visual_tools_->prompt(text);
        };
        auto const draw_trajectory_tool_path = 
            [this, jmg = move_group_->getRobotModel()->getJointModelGroup(move_group_name_)](auto const trajectory) {
                if (!moveit_visual_tools_) return;
                if (!jmg) return;
                moveit_visual_tools_->publishTrajectoryLine(trajectory, jmg);
                moveit_visual_tools_->trigger();
            };

        // 1. 等待小球位置
        while (rclcpp::ok() && !ball_pose_received_.load()) { rclcpp::sleep_for(100ms); }
        if (!rclcpp::ok()) return;
        
        geometry_msgs::msg::Pose ball_pose_copy;
        {
            std::lock_guard<std::mutex> lock(ball_pose_mutex_);
            ball_pose_copy = current_ball_pose_;
        }
        
        // 更新碰撞体
        moveit_msgs::msg::CollisionObject ball_co;
        ball_co.id = "target_ball";
        ball_co.header.frame_id = move_group_->getPlanningFrame();
        shape_msgs::msg::SolidPrimitive sph;
        sph.type = sph.SPHERE;
        sph.dimensions.resize(1);
        sph.dimensions[0] = ball_radius_;
        ball_co.primitives.push_back(sph);
        ball_co.primitive_poses.push_back(ball_pose_copy);
        ball_co.operation = ball_co.ADD;
        planning_scene_interface_->applyCollisionObjects({ball_co});
        setGazeboBallPoseOnce(ball_pose_copy);
        rclcpp::sleep_for(500ms);

        // 预抓取位姿
        geometry_msgs::msg::Pose pre_grasp_pose = ball_pose_copy;
        pre_grasp_pose.position.z += pre_grasp_clearance_ + ee_to_grasp_offset_Z_;
        tf2::Quaternion q;
        q.setRPY(M_PI, 0.0, 0.0);
        pre_grasp_pose.orientation = tf2::toMsg(q);

        // 1) 移动到预抓取
        moveit::planning_interface::MoveGroupInterface::Plan pre_grasp_plan;
        prompt("Press 'Next' to pre-grasp.");
        if (!moveLinkToPose(pre_grasp_pose, target_ee_link_, pre_grasp_velocity_, pre_grasp_plan)) return;
        draw_trajectory_tool_path(pre_grasp_plan.trajectory_);
        move_group_->execute(pre_grasp_plan);
        rclcpp::sleep_for(200ms);

        // 2) 下降
        geometry_msgs::msg::Pose grasp_pose = pre_grasp_pose;
        grasp_pose.position.z = ball_pose_copy.position.z + ee_to_grasp_offset_Z_;
        std::vector<geometry_msgs::msg::Pose> waypoints = {grasp_pose};
        moveit_msgs::msg::RobotTrajectory trajectory;
        double fraction = move_group_->computeCartesianPath(waypoints, 0.01, 0.0, trajectory);
        prompt("Press 'Next' to descend.");
        
        if (fraction >= 0.9) {
            draw_trajectory_tool_path(trajectory);
            moveit::planning_interface::MoveGroupInterface::Plan cartesian_plan;
            cartesian_plan.trajectory_ = trajectory;
            move_group_->execute(cartesian_plan);
        } else {
            moveit::planning_interface::MoveGroupInterface::Plan fallback;
            moveLinkToPose(grasp_pose, target_ee_link_, approach_velocity_, fallback);
            move_group_->execute(fallback);
        }

        // --- 3) 关闭夹爪 (恢复为位置控制) ---
        closeGripper(); 
        
        // --- 4) 附着小球 ---
        prompt("Press 'Next' to attach ball.");
        attachBall();

        // --- 5) 抬起 ---
        waypoints = {pre_grasp_pose};
        fraction = move_group_->computeCartesianPath(waypoints, 0.01, 0.0, trajectory);
        prompt("Press 'Next' to LIFT.");
        if (fraction >= 0.9) {
            moveit::planning_interface::MoveGroupInterface::Plan p; p.trajectory_ = trajectory;
            move_group_->execute(p);
        } else {
             moveit::planning_interface::MoveGroupInterface::Plan fallback;
             moveLinkToPose(pre_grasp_pose, target_ee_link_, lift_velocity_, fallback);
             move_group_->execute(fallback);
        }

        // --- 6) 移动到放置点上方 ---
        geometry_msgs::msg::Pose place_above = place_pose_;
        place_above.position.z += pre_grasp_clearance_ + ee_to_grasp_offset_Z_;
        place_above.orientation = pre_grasp_pose.orientation;
        moveit::planning_interface::MoveGroupInterface::Plan place_above_plan;
        prompt("Press 'Next' to PLACE ABOVE.");
        if (moveLinkToPose(place_above, target_ee_link_, 0.35, place_above_plan)) {
            move_group_->execute(place_above_plan);
        }

        // --- 7) 下降 ---
        geometry_msgs::msg::Pose place_down = place_above;
        place_down.position.z = place_pose_.position.z + ee_to_grasp_offset_Z_;
        waypoints = {place_down};
        fraction = move_group_->computeCartesianPath(waypoints, 0.01, 0.0, trajectory);
        prompt("Press 'Next' to DESCEND.");
        if (fraction >= 0.9) {
             moveit::planning_interface::MoveGroupInterface::Plan p; p.trajectory_ = trajectory;
             move_group_->execute(p);
        } else {
             moveit::planning_interface::MoveGroupInterface::Plan fallback;
             moveLinkToPose(place_down, target_ee_link_, approach_velocity_, fallback);
             move_group_->execute(fallback);
        }

        // --- 8) 放下并打开夹爪 (恢复为位置控制) ---
        prompt("Press 'Next' to detach and OPEN.");
        detachBall();
        openGripper(); 

        // --- 9) 抬起 ---
        waypoints = {place_above};
        fraction = move_group_->computeCartesianPath(waypoints, 0.01, 0.0, trajectory);
        prompt("Press 'Next' to LIFT AWAY.");
        if (fraction >= 0.9) {
             moveit::planning_interface::MoveGroupInterface::Plan p; p.trajectory_ = trajectory;
             move_group_->execute(p);
        }

        // --- 10) 归位 ---
        prompt("Press 'Next' to HOME.");
        move_group_->setNamedTarget("home");
        moveit::planning_interface::MoveGroupInterface::Plan home_plan;
        if (move_group_->plan(home_plan) == moveit::core::MoveItErrorCode::SUCCESS) {
            move_group_->execute(home_plan);
        }
        
        planning_scene_interface_->removeCollisionObjects({"target_ball"});
        RCLCPP_INFO(this->get_logger(), "Sequence Finished.");
    }
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<MoveToTargetNode>());
    rclcpp::shutdown();
    return 0;
}
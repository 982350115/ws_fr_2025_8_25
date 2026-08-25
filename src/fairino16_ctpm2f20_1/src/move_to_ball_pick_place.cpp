#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>
#include <trajectory_msgs/msg/joint_trajectory_point.hpp>

// MoveIt Headers
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit_msgs/msg/collision_object.hpp>
#include <moveit_msgs/msg/attached_collision_object.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>

// TF2 Headers for real-time pose tracking
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2/LinearMath/Quaternion.h> 

// Gazebo Service Headers
#include <gazebo_msgs/srv/set_entity_state.hpp> 
#include <gazebo_msgs/msg/entity_state.hpp>

#include <atomic>
#include <chrono>

using namespace std::chrono_literals;

class MoveToTargetNode : public rclcpp::Node
{
private:
    // MoveIt components
    moveit::planning_interface::MoveGroupInterfacePtr move_group_;
    moveit::planning_interface::PlanningSceneInterfacePtr planning_scene_interface_;

    // ROS 2 components
    rclcpp::TimerBase::SharedPtr init_timer_;
    rclcpp::TimerBase::SharedPtr gazebo_update_timer_; // 实时 Gazebo 更新计时器
    rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr gripper_traj_pub_;
    rclcpp::Client<gazebo_msgs::srv::SetEntityState>::SharedPtr set_entity_state_client_;

    // TF2 components
    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

    // State management for continuous update
    std::atomic<bool> is_carrying_{false}; // 标志，指示是否启动实时跟随

    // --- 参数 / 常量定义 ---
    const std::string move_group_name_ = "manipulator";
    const std::string gripper_controller_topic_ = "/gripper_controller/joint_trajectory";
    const std::string gripper_joint_name_ = "gripper_left_joint";
    const std::string target_ee_link_ = "gripper_base_link"; // 夹爪中心作为MoveIt目标连杆
    const std::string world_frame_ = "world"; // 世界坐标系名称
    
    // 从 gripper_base_link 原点到小球抓取中心点的Z轴距离
    const double ee_to_grasp_offset_Z_ = 0.180; 
    const std::string gazebo_model_name_ = "target_ball"; 

    // 小球和放置点
    const double ball_radius_ = 0.025;
    const geometry_msgs::msg::Pose ball_pose_ = [](){
        geometry_msgs::msg::Pose p;
        p.position.x = 0.5; p.position.y = 0.2; p.position.z = 0.825; 
        p.orientation.w = 1.0; return p;
    }();
    const geometry_msgs::msg::Pose place_pose_ = [](){
        geometry_msgs::msg::Pose p;
        p.position.x = 0.3; p.position.y = 0.4; p.position.z = 0.825; 
        p.orientation.w = 1.0; return p;
    }();
    
    // 运动参数
    const double pre_grasp_clearance_ = 0.20; // 预抓取高度（夹爪中心上方 0.2m）
    const double pre_grasp_velocity_ = 0.3;
    const double approach_velocity_ = 0.10; // 下降时慢速
    const double lift_velocity_ = 0.25;
    // --- 参数 / 常量定义结束 ---

public:
    MoveToTargetNode() : rclcpp::Node("move_to_target_node")
    {
        // 延迟初始化 MoveIt 和 TF2
        init_timer_ = this->create_wall_timer(
            100ms, std::bind(&MoveToTargetNode::initialize_and_start, this));
    }

private:
    void initialize_and_start()
    {
        init_timer_.reset(); // 停止初始化计时器

        // 初始化 MoveIt 接口
        move_group_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(shared_from_this(), move_group_name_);
        planning_scene_interface_ = std::make_shared<moveit::planning_interface::PlanningSceneInterface>();

        // 初始化 TF2 监听
        tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        // 初始化 ROS 2 客户端和发布器
        gripper_traj_pub_ = this->create_publisher<trajectory_msgs::msg::JointTrajectory>(gripper_controller_topic_, 10);
        set_entity_state_client_ = this->create_client<gazebo_msgs::srv::SetEntityState>("/set_entity_state");
        RCLCPP_INFO(this->get_logger(), "Gazebo SetEntityState client created.");

        // 创建实时 Gazebo 更新计时器 (50ms 频率)
        gazebo_update_timer_ = this->create_wall_timer(
            50ms, std::bind(&MoveToTargetNode::continuousGazeboUpdate, this));

        openGripper();
        rclcpp::sleep_for(200ms);

        addTableAndBallToPlanningScene();

        // 启动 pick-and-place 流程
        std::thread([this](){
            rclcpp::sleep_for(1s);
            this->pickPlaceSequence();
        }).detach();
    }
    
    // 实时更新 Gazebo 中小球位姿的函数
    void continuousGazeboUpdate()
    {
        if (!is_carrying_.load()) {
            return; // 只有在 is_carrying_ 为 true 时才执行
        }
        
        // 1. 检查 Gazebo 服务是否可用
        if (!set_entity_state_client_->wait_for_service(0s)) {
            RCLCPP_WARN_ONCE(this->get_logger(), "Gazebo set_entity_state service not available. Cannot move visual model.");
            is_carrying_ = false; // 停止尝试
            return;
        }

        // 2. 获取夹爪的实时位姿 (从 TF 树中)
        geometry_msgs::msg::TransformStamped transformStamped;
        try {
            // 获取从 world 到 target_ee_link_ 的变换
            transformStamped = tf_buffer_->lookupTransform(
                world_frame_, target_ee_link_, tf2::TimePointZero);
        } catch (tf2::TransformException &ex) {
            RCLCPP_ERROR(this->get_logger(), "TF Error: Could not transform %s to %s: %s",
                         world_frame_.c_str(), target_ee_link_.c_str(), ex.what());
            return;
        }

        // 3. 计算小球中心在世界坐标系中的位姿
        geometry_msgs::msg::Pose ball_pose_in_world;
        
        // 将 TF 变换转换为 Pose 消息
        geometry_msgs::msg::Pose gripper_pose;
        gripper_pose.position.x = transformStamped.transform.translation.x;
        gripper_pose.position.y = transformStamped.transform.translation.y;
        gripper_pose.position.z = transformStamped.transform.translation.z;
        gripper_pose.orientation = transformStamped.transform.rotation;
        
        // 将 gripper_pose 转换为 TF2 姿态
        tf2::Transform gripper_tf;
        tf2::fromMsg(gripper_pose, gripper_tf);

        // 小球在夹爪坐标系中的偏移 (沿夹爪的 -Z 轴偏移 ee_to_grasp_offset_Z_)
        tf2::Transform ball_offset_tf;
        ball_offset_tf.setIdentity();
        // 设置偏移量，注意是负号，因为夹爪 Z 轴朝下，小球在原点下方
        ball_offset_tf.setOrigin(tf2::Vector3(0.0, 0.0, ee_to_grasp_offset_Z_)); 

        // 最终小球的世界位姿 = 夹爪世界位姿 * 小球相对夹爪的偏移
        tf2::Transform ball_world_tf = gripper_tf * ball_offset_tf;
        
        // 转换回 ROS 消息
        tf2::toMsg(ball_world_tf, ball_pose_in_world);

        // 4. 调用 Gazebo 服务
        auto request = std::make_shared<gazebo_msgs::srv::SetEntityState::Request>();
        request->state.name = gazebo_model_name_; 
        request->state.pose = ball_pose_in_world;
        request->state.reference_frame = world_frame_; 
        
        // 异步发送请求，不阻塞主线程
        set_entity_state_client_->async_send_request(request);
        // RCLCPP_DEBUG(this->get_logger(), "Gazebo ball pose updated."); // 启用 DEBUG 级别可查看实时更新
    }


    // 通用 MoveIt 工具函数 (保持不变)
    void addTableAndBallToPlanningScene()
    {
        // 1. 添加桌子
        moveit_msgs::msg::CollisionObject table;
        table.id = "work_table";
        table.header.frame_id = move_group_->getPlanningFrame();
        shape_msgs::msg::SolidPrimitive table_primitive;
        table_primitive.type = shape_msgs::msg::SolidPrimitive::BOX;
        table_primitive.dimensions = {2.0, 2.0, 0.05}; 
        geometry_msgs::msg::Pose table_pose;
        table_pose.orientation.w = 1.0;
        table_pose.position.z = 0.75; // 假设桌子顶面中心在 Z=0.75
        table.primitives.push_back(table_primitive);
        table.primitive_poses.push_back(table_pose);
        table.operation = table.ADD;
        planning_scene_interface_->applyCollisionObjects({table});
        RCLCPP_INFO(this->get_logger(), "Added table collision object to planning scene.");
        
        // 2. 添加小球作为碰撞物体
        moveit_msgs::msg::CollisionObject ball_co;
        ball_co.id = "target_ball";
        ball_co.header.frame_id = move_group_->getPlanningFrame();
        shape_msgs::msg::SolidPrimitive sph;
        sph.type = sph.SPHERE;
        sph.dimensions.resize(1);
        sph.dimensions[0] = ball_radius_;
        ball_co.primitives.push_back(sph);
        ball_co.primitive_poses.push_back(ball_pose_);
        ball_co.operation = ball_co.ADD;
        planning_scene_interface_->applyCollisionObjects({ball_co});
        RCLCPP_INFO(this->get_logger(), "Added target_ball collision object to planning scene.");
        
        // 3. 将 Gazebo 中的小球模型瞬移到初始抓取位置 (防止其在初始化时滚走)
        setGazeboBallPoseOnce(ball_pose_);
    }

    // 单次调用 Gazebo 服务更新模型位姿 (用于初始化和放置)
    void setGazeboBallPoseOnce(const geometry_msgs::msg::Pose& pose) {
        if (!set_entity_state_client_->wait_for_service(std::chrono::seconds(1))) {
            RCLCPP_ERROR(this->get_logger(), "Gazebo entity state service /set_entity_state not available.");
            return;
        }

        auto request = std::make_shared<gazebo_msgs::srv::SetEntityState::Request>();
        request->state.name = gazebo_model_name_; 
        request->state.pose = pose;
        request->state.reference_frame = world_frame_; 

        RCLCPP_INFO(this->get_logger(), "Calling Gazebo service to set model '%s' pose to (%.3f, %.3f, %.3f).", 
                                        gazebo_model_name_.c_str(), 
                                        pose.position.x, pose.position.y, pose.position.z);

        auto future = set_entity_state_client_->async_send_request(request);
        if (future.wait_for(5s) != std::future_status::ready) {
            RCLCPP_ERROR(this->get_logger(), "Failed to call Gazebo set_entity_state service (timeout).");
        }

    }

    void sendGripperTrajectory(double position, double time_from_start_sec = 0.5)
    {
        trajectory_msgs::msg::JointTrajectory traj;
        traj.joint_names.push_back(gripper_joint_name_);
        trajectory_msgs::msg::JointTrajectoryPoint pt;
        pt.positions.push_back(position);
        pt.time_from_start = rclcpp::Duration::from_seconds(time_from_start_sec);
        traj.points.push_back(pt);
        if (gripper_traj_pub_) { gripper_traj_pub_->publish(traj); }
    }

    void openGripper()
    {
        RCLCPP_INFO(this->get_logger(), "Opening gripper...");
        sendGripperTrajectory(0.0, 0.3); // 使用常量 0.0
        rclcpp::sleep_for(400ms);
    }

    void closeGripper()
    {
        RCLCPP_INFO(this->get_logger(), "Closing gripper...");
        sendGripperTrajectory(-0.009, 0.4); // 使用常量 -0.009
        rclcpp::sleep_for(600ms);
    }

    bool moveLinkToPose(const geometry_msgs::msg::Pose& target_pose, const std::string& link_name, double velocity_scale, double planning_time = 6.0)
    {
        move_group_->setEndEffectorLink(link_name); 
        move_group_->setMaxVelocityScalingFactor(velocity_scale);
        move_group_->setMaxAccelerationScalingFactor(velocity_scale);
        move_group_->setPlanningTime(planning_time);
        move_group_->setPoseTarget(target_pose, link_name); 
        
        moveit::planning_interface::MoveGroupInterface::Plan plan;
        auto res = move_group_->plan(plan);
        bool ok = (res == moveit::core::MoveItErrorCode::SUCCESS);
        if (!ok) {
            unsigned int error_code = static_cast<unsigned int>(res.val);
            RCLCPP_WARN_STREAM(this->get_logger(),
                "Plan to pose for link " << link_name
                << " failed. MoveIt Error Code value: " << error_code);
            move_group_->clearPoseTargets();
            return false;
        }
        RCLCPP_INFO(this->get_logger(), "Plan success, executing...");
        move_group_->execute(plan);
        move_group_->clearPoseTargets();
        rclcpp::sleep_for(200ms);
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
        RCLCPP_INFO(this->get_logger(), "Ball attached to %s in Planning Scene.", target_ee_link_.c_str());
        
        // **启动实时跟随**
        is_carrying_ = true;
        RCLCPP_INFO(this->get_logger(), "--- STARTING REAL-TIME GAZEBO FOLLOW ---");
    }

    void detachBall()
    {
        // **停止实时跟随**
        is_carrying_ = false;
        RCLCPP_INFO(this->get_logger(), "--- STOPPING REAL-TIME GAZEBO FOLLOW ---");

        // 1. 在 MoveIt 场景中解除附着
        moveit_msgs::msg::AttachedCollisionObject detached_ball;
        detached_ball.object.id = "target_ball";
        detached_ball.object.operation = moveit_msgs::msg::CollisionObject::REMOVE; 
        planning_scene_interface_->applyAttachedCollisionObject(detached_ball);
        RCLCPP_INFO(this->get_logger(), "Ball detached from gripper in Planning Scene.");

        // 2. 在 MoveIt 场景中重新添加小球到放置位置 (作为静态障碍物)
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
        RCLCPP_INFO(this->get_logger(), "Ball object re-added to Planning Scene at place position.");
        
        // 3. 将 Gazebo 中的可视化模型瞬移到放置位置 (确保最终位置准确)
        setGazeboBallPoseOnce(place_pose_);
    }

    // 执行 pick-and-place 流程（全自动）
    void pickPlaceSequence()
    {
        RCLCPP_INFO(this->get_logger(), "Starting FULL pick-and-place sequence (WITH REAL-TIME GAZEBO FOLLOW).");

        // 预抓取位姿计算
        geometry_msgs::msg::Pose pre_grasp_pose = ball_pose_;
        pre_grasp_pose.position.z += pre_grasp_clearance_ + ee_to_grasp_offset_Z_;
        tf2::Quaternion q;
        q.setRPY(M_PI, 0.0, 0.0);
        pre_grasp_pose.orientation = tf2::toMsg(q);

        RCLCPP_INFO(this->get_logger(), "1) Move to pre-grasp (C-Space Planning)");
        if (!moveLinkToPose(pre_grasp_pose, target_ee_link_, pre_grasp_velocity_)) {
            RCLCPP_ERROR(this->get_logger(), "Failed to move to pre-grasp. Aborting."); return;
        }

        // 2) 缓慢下降到抓取位姿 (笛卡尔路径)
        geometry_msgs::msg::Pose grasp_pose = pre_grasp_pose;
        grasp_pose.position.z = ball_pose_.position.z + ee_to_grasp_offset_Z_; 
        RCLCPP_INFO(this->get_logger(), "2) Approach down to grasp. Cartesian Path.");
        std::vector<geometry_msgs::msg::Pose> waypoints = {grasp_pose};
        moveit_msgs::msg::RobotTrajectory trajectory;
        double fraction = move_group_->computeCartesianPath(waypoints, 0.01, 0.0, trajectory);

        if (fraction < 0.9) {
            RCLCPP_WARN_STREAM(this->get_logger(), "Cartesian Path Failed (Only " << (fraction * 100.0) << "% achieved). Fallback to C-Space.");
            if (!moveLinkToPose(grasp_pose, target_ee_link_, approach_velocity_, 4.0)) {
                RCLCPP_ERROR(this->get_logger(), "Fallback C-Space planning failed. Aborting."); return;
            }
        } else {
            move_group_->setMaxVelocityScalingFactor(approach_velocity_);
            move_group_->setMaxAccelerationScalingFactor(approach_velocity_);
            moveit::planning_interface::MoveGroupInterface::Plan cartesian_plan;
            cartesian_plan.trajectory_ = trajectory;
            RCLCPP_INFO(this->get_logger(), "Cartesian Path success (%0.1f%%), executing...", fraction * 100.0);
            move_group_->execute(cartesian_plan);
        }

        // 3) 关闭夹爪
        closeGripper();
        RCLCPP_INFO(this->get_logger(), "3) Closed gripper.");

        // 4) 附着小球，并启动实时 Gazebo 跟随！
        attachBall();
        RCLCPP_INFO(this->get_logger(), "4) Attached ball and STARTED REAL-TIME FOLLOW.");

        // 5) 抬起到 pre-grasp 高度（带物体） (笛卡尔路径)
        RCLCPP_INFO(this->get_logger(), "5) Lift up to pre-grasp height (WITH OBJECT). Cartesian Path.");
        waypoints = {pre_grasp_pose};
        fraction = move_group_->computeCartesianPath(waypoints, 0.01, 0.0, trajectory);
        if (fraction < 0.9) {
            RCLCPP_WARN_STREAM(this->get_logger(), "Lift after grasp FAILED. Fallback to C-Space.");
            if (!moveLinkToPose(pre_grasp_pose, target_ee_link_, lift_velocity_, 6.0)) {
                RCLCPP_ERROR(this->get_logger(), "Fallback C-Space planning failed for lift. Aborting."); return;
            }
        } else {
            move_group_->setMaxVelocityScalingFactor(lift_velocity_);
            move_group_->setMaxAccelerationScalingFactor(lift_velocity_);
            moveit::planning_interface::MoveGroupInterface::Plan cartesian_plan;
            cartesian_plan.trajectory_ = trajectory;
            RCLCPP_INFO(this->get_logger(), "Cartesian Path success (%0.1f%%), executing...", fraction * 100.0);
            move_group_->execute(cartesian_plan);
        }

        // 6) 移动到放置点上方 (C-Space 规划)
        geometry_msgs::msg::Pose place_above = place_pose_;
        place_above.position.z += pre_grasp_clearance_ + ee_to_grasp_offset_Z_;
        place_above.orientation = pre_grasp_pose.orientation;
        RCLCPP_INFO(this->get_logger(), "6) Move to place-above. C-Space Planning.");
        if (!moveLinkToPose(place_above, target_ee_link_, 0.35, 10.0)) {
            RCLCPP_ERROR(this->get_logger(), "Move to place-above failed. Aborting."); return;
        }

        // 7) 下降到放置位 (笛卡尔路径)
        geometry_msgs::msg::Pose place_down = place_above;
        place_down.position.z = place_pose_.position.z + ee_to_grasp_offset_Z_;
        RCLCPP_INFO(this->get_logger(), "7) Descend to place. Cartesian Path.");
        waypoints = {place_down};
        fraction = move_group_->computeCartesianPath(waypoints, 0.01, 0.0, trajectory);
        if (fraction < 0.9) {
            RCLCPP_WARN_STREAM(this->get_logger(), "Descend to place FAILED. Fallback to C-Space.");
            if (!moveLinkToPose(place_down, target_ee_link_, approach_velocity_, 4.0)) {
                RCLCPP_ERROR(this->get_logger(), "Fallback C-Space planning failed. Aborting."); return;
            }
        } else {
            move_group_->setMaxVelocityScalingFactor(approach_velocity_);
            move_group_->setMaxAccelerationScalingFactor(approach_velocity_);
            moveit::planning_interface::MoveGroupInterface::Plan cartesian_plan;
            cartesian_plan.trajectory_ = trajectory;
            RCLCPP_INFO(this->get_logger(), "Cartesian Path success (%0.1f%%), executing...", fraction * 100.0);
            move_group_->execute(cartesian_plan);
        }

        // 8) 解除附着并打开夹爪 (放下)
        detachBall(); 
        openGripper();
        RCLCPP_INFO(this->get_logger(), "8) Detached ball, STOPPED FOLLOW, and opened gripper.");

        // 9) 抬起离开放置点 (笛卡尔路径)
        RCLCPP_INFO(this->get_logger(), "9) Lift away from place position. Cartesian Path.");
        waypoints = {place_above};
        fraction = move_group_->computeCartesianPath(waypoints, 0.01, 0.0, trajectory);
        if (fraction < 0.9) {
            RCLCPP_WARN_STREAM(this->get_logger(), "Lift after placing FAILED. Fallback to C-Space.");
            if (!moveLinkToPose(place_above, target_ee_link_, 0.35, 6.0)) {
                RCLCPP_ERROR(this->get_logger(), "Fallback C-Space planning failed.");
            }
        } else {
            move_group_->setMaxVelocityScalingFactor(0.35);
            move_group_->setMaxAccelerationScalingFactor(0.35);
            moveit::planning_interface::MoveGroupInterface::Plan cartesian_plan;
            cartesian_plan.trajectory_ = trajectory;
            RCLCPP_INFO(this->get_logger(), "Cartesian Path success (%0.1f%%), executing...", fraction * 100.0);
            move_group_->execute(cartesian_plan);
        }

        // 10) 归位
        RCLCPP_INFO(this->get_logger(), "10) Returning to safety/home pose.");
        try {
            move_group_->setNamedTarget("home");
            move_group_->setMaxVelocityScalingFactor(0.3);
            move_group_->setMaxAccelerationScalingFactor(0.3);
            moveit::planning_interface::MoveGroupInterface::Plan home_plan;
            if (move_group_->plan(home_plan) == moveit::core::MoveItErrorCode::SUCCESS) {
                move_group_->execute(home_plan);
            }
        } catch (...) {
            RCLCPP_WARN(this->get_logger(), "Named target 'home' not available or planning failed.");
        }
        
        // 最后清理 MoveIt 场景中的障碍物 (小球)
        planning_scene_interface_->removeCollisionObjects({"target_ball"});
        RCLCPP_INFO(this->get_logger(), "FULL pick-and-place sequence finished.");
    }
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<MoveToTargetNode>());
    rclcpp::shutdown();
    return 0;
}


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

// Trajectory Processing
#include <moveit/trajectory_processing/iterative_time_parameterization.h>
#include <moveit/robot_trajectory/robot_trajectory.h>

// TF2 Headers
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Vector3.h>

#include <thread>
#include <chrono>
#include <cmath> 
#include <vector>

using namespace std::chrono_literals;

// ==========================================
// ========== User Configuration ==========
// ==========================================

const std::string MOVE_GROUP_NAME = "manipulator";
const std::string TARGET_EE_LINK = "gripper_center_tcp"; 
const std::string GRIPPER_SERVICE_NAME = "/gripper_set_open"; 

// --- Hardcoded Z Heights (Meters) ---
// Adjust based on physical measurements
const double FIXED_GRASP_Z  = 0.083;  // Handle height
const double FIXED_CENTER_Z = 0.002;  // Target cup center height
const double FIXED_SPOUT_Z  = 0.083;  // Spout height

// --- Action Parameters ---
const double LIFT_HEIGHT = 0.25;            // Lift height after grasp (25cm)
const double POUR_PRE_HEIGHT_OFFSET = 0.25; // Spout height above target cup
const double POUR_DIP_HEIGHT = 0.00;        // Dip distance before pouring
const double POUR_ANGLE = 35.0;             // Pouring angle (Degrees)

// --- [NEW] Granular Speed Controls ---
// 1. Joint Planning (Approach/Retreat - Empty Hand)
const double VEL_JOINT_GLOBAL = 0.05; 
const double ACC_JOINT_GLOBAL = 0.05;

// 2. Cartesian Move (Lifting/Transferring - Holding Cup)
// Needs to be smooth but efficient
const double VEL_CART_MOVE = 0.05; 
const double ACC_CART_MOVE = 0.05;

// 3. Process Move (Pouring - Careful Operation)
// Needs to be very slow to prevent spilling
const double VEL_CART_POUR = 0.03; 
const double ACC_CART_POUR = 0.03;

class DualCupPouring : public rclcpp::Node
{
public:
    DualCupPouring() : Node("dual_cup_pouring_node")
    {
        cb_group_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

        sub_grasp_A_ = create_pose_sub("/task/cup_A_grasp", grasp_A_, flag_grasp_A_);
        sub_center_A_ = create_point_sub("/task/cup_A_center", center_A_, flag_center_A_);
        sub_pivot_A_ = create_point_sub("/task/cup_A_pivot", pivot_A_, flag_pivot_A_);
        
        sub_grasp_B_ = create_pose_sub("/task/cup_B_grasp", grasp_B_, flag_grasp_B_);
        sub_center_B_ = create_point_sub("/task/cup_B_center", center_B_, flag_center_B_);
        sub_pivot_B_ = create_point_sub("/task/cup_B_pivot", pivot_B_, flag_pivot_B_);

        init_timer_ = this->create_wall_timer(100ms, std::bind(&DualCupPouring::run, this), cb_group_);
    }

private:
    // Helper Subscriptions
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr create_pose_sub(
        std::string topic, geometry_msgs::msg::Pose& data, bool& flag) {
        return this->create_subscription<geometry_msgs::msg::PoseStamped>(
            topic, 10, [&](const geometry_msgs::msg::PoseStamped::SharedPtr msg){
                data = msg->pose; flag = true;
                RCLCPP_INFO(this->get_logger(), "Received Pose: %s", topic.c_str());
            });
    }
    rclcpp::Subscription<geometry_msgs::msg::PointStamped>::SharedPtr create_point_sub(
        std::string topic, geometry_msgs::msg::Point& data, bool& flag) {
        return this->create_subscription<geometry_msgs::msg::PointStamped>(
            topic, 10, [&](const geometry_msgs::msg::PointStamped::SharedPtr msg){
                data = msg->point; flag = true;
                RCLCPP_INFO(this->get_logger(), "Received Point: %s", topic.c_str());
            });
    }

    rclcpp::TimerBase::SharedPtr init_timer_;
    rclcpp::CallbackGroup::SharedPtr cb_group_;
    
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr sub_grasp_A_, sub_grasp_B_;
    rclcpp::Subscription<geometry_msgs::msg::PointStamped>::SharedPtr sub_center_A_, sub_center_B_;
    rclcpp::Subscription<geometry_msgs::msg::PointStamped>::SharedPtr sub_pivot_A_, sub_pivot_B_;

    geometry_msgs::msg::Pose grasp_A_, grasp_B_;
    geometry_msgs::msg::Point center_A_, center_B_;
    geometry_msgs::msg::Point pivot_A_, pivot_B_;
    
    bool flag_grasp_A_ = false, flag_center_A_ = false, flag_pivot_A_ = false;
    bool flag_grasp_B_ = false, flag_center_B_ = false, flag_pivot_B_ = false;

    std::shared_ptr<moveit::planning_interface::MoveGroupInterface> move_group_;
    std::shared_ptr<moveit::planning_interface::PlanningSceneInterface> planning_scene_interface_;

    void run()
    {
        init_timer_->cancel();
        move_group_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(shared_from_this(), MOVE_GROUP_NAME);
        planning_scene_interface_ = std::make_shared<moveit::planning_interface::PlanningSceneInterface>();

        move_group_->setEndEffectorLink(TARGET_EE_LINK);
        move_group_->setPlanningTime(10.0);
        
        setupBaseScene();
        controlGripper(true); 
        executeLoop();
    }

    void controlGripper(bool open) {
        auto temp_node = rclcpp::Node::make_shared("temp_gripper_client");
        auto client = temp_node->create_client<example_interfaces::srv::SetBool>(GRIPPER_SERVICE_NAME);
        if (!client->wait_for_service(std::chrono::seconds(1))) return;
        auto req = std::make_shared<example_interfaces::srv::SetBool::Request>();
        req->data = open;
        auto future = client->async_send_request(req);
        rclcpp::spin_until_future_complete(temp_node, future);
    }
    void openGripper() { controlGripper(true); }
    void closeGripper() { controlGripper(false); }

    void setupBaseScene() {
        moveit_msgs::msg::CollisionObject table;
        table.id = "table"; table.header.frame_id = move_group_->getPlanningFrame();
        table.operation = table.ADD;
        shape_msgs::msg::SolidPrimitive box; box.type = box.BOX; box.dimensions = {2.0, 2.0, 0.02};
        geometry_msgs::msg::Pose p; p.position.z = -0.01; p.orientation.w = 1.0;
        table.primitives.push_back(box); table.primitive_poses.push_back(p);
        planning_scene_interface_->applyCollisionObjects({table});
    }

    // --- Joint Space Movement (Approach / Retreat) ---
    bool moveToPose(const geometry_msgs::msg::Pose& pose) {
        move_group_->setPoseTarget(pose);
        // Use Global Joint Speed settings
        move_group_->setMaxVelocityScalingFactor(VEL_JOINT_GLOBAL);
        move_group_->setMaxAccelerationScalingFactor(ACC_JOINT_GLOBAL);
        return (move_group_->move() == moveit::core::MoveItErrorCode::SUCCESS);
    }

    // --- Cartesian Linear Movement (Lifting / Pouring) ---
    // Takes a custom velocity scale for fine control
    bool moveCartesian(const std::vector<geometry_msgs::msg::Pose>& waypoints, double vel_scale) {
        moveit_msgs::msg::RobotTrajectory trajectory;
        // 0.01 step size, 0.0 jump threshold (disabled for robustness)
        double fraction = move_group_->computeCartesianPath(waypoints, 0.01, 0.0, trajectory);
        
        if (fraction > 0.8) {
            auto rt = std::make_shared<robot_trajectory::RobotTrajectory>(move_group_->getRobotModel(), move_group_->getName());
            rt->setRobotTrajectoryMsg(*move_group_->getCurrentState(), trajectory);
            
            // Apply the specific velocity scale passed in (MOVE vs POUR)
            trajectory_processing::IterativeParabolicTimeParameterization iptp;
            if (iptp.computeTimeStamps(*rt, vel_scale, vel_scale)) {
                moveit::planning_interface::MoveGroupInterface::Plan plan;
                rt->getRobotTrajectoryMsg(plan.trajectory_);
                return (move_group_->execute(plan) == moveit::core::MoveItErrorCode::SUCCESS);
            }
        }
        RCLCPP_WARN(this->get_logger(), "Cartesian Path Failed Fraction: %f", fraction);
        return false;
    }

    // ===============================================
    // ============ Logic & Calculations ============
    // ===============================================

    geometry_msgs::msg::Quaternion calculateVerticalGraspOrientation(
        const geometry_msgs::msg::Pose& handle_pose, 
        const geometry_msgs::msg::Point& center_point) 
    {
        // Calculate Yaw based on 2D projection (X, Y)
        double dx = center_point.x - handle_pose.position.x;
        double dy = center_point.y - handle_pose.position.y;
        double yaw = std::atan2(dy, dx);
        
        // Z-axis points up (World Z), Yaw aligns X-axis with cup center
        tf2::Quaternion q;
        q.setRPY(0.0, 0.0, yaw); 
        return tf2::toMsg(q);
    }

    std::vector<geometry_msgs::msg::Pose> generatePivotPath(
        const geometry_msgs::msg::Pose& start_pose,
        const geometry_msgs::msg::Point& pivot_point_world,
        double angle_deg)
    {
        std::vector<geometry_msgs::msg::Pose> waypoints;
        int steps = 50; 

        tf2::Vector3 p_tcp(start_pose.position.x, start_pose.position.y, start_pose.position.z);
        tf2::Vector3 p_pivot(pivot_point_world.x, pivot_point_world.y, pivot_point_world.z);
        tf2::Vector3 radius_vec = p_tcp - p_pivot; 

        tf2::Quaternion q_start(start_pose.orientation.x, start_pose.orientation.y, start_pose.orientation.z, start_pose.orientation.w);
        tf2::Matrix3x3 m_rot(q_start);
        tf2::Vector3 rot_axis = m_rot.getColumn(1); // Local Y axis

        for(int i=1; i<=steps; ++i) {
            double theta = (angle_deg * M_PI / 180.0) * ((double)i / steps);
            
            tf2::Quaternion q_delta;
            q_delta.setRotation(rot_axis, theta);
            
            tf2::Vector3 radius_new = tf2::quatRotate(q_delta, radius_vec);
            tf2::Vector3 p_new = p_pivot + radius_new;

            tf2::Quaternion q_new = q_delta * q_start;
            q_new.normalize();

            geometry_msgs::msg::Pose wp;
            wp.position.x = p_new.x(); wp.position.y = p_new.y(); wp.position.z = p_new.z();
            wp.orientation = tf2::toMsg(q_new);
            waypoints.push_back(wp);
        }
        return waypoints;
    }

    // ===============================================
    // ============ Task Execution ============
    // ===============================================
    void executeTask()
    {
        RCLCPP_INFO(this->get_logger(), "=== Starting Task (Refined Speeds) ===");

        // 1. Data Prep (Hardcoded Z)
        geometry_msgs::msg::Pose target_grasp_pose = grasp_A_;
        target_grasp_pose.position.z = FIXED_GRASP_Z; 

        tf2::Vector3 p_grasp_vis(grasp_A_.position.x, grasp_A_.position.y, 0.0); 
        tf2::Vector3 p_pivot_vis(pivot_A_.x, pivot_A_.y, 0.0);
        tf2::Vector3 v_offset_2d = p_pivot_vis - p_grasp_vis;
        tf2::Vector3 v_tcp_to_pivot(v_offset_2d.x(), v_offset_2d.y(), FIXED_SPOUT_Z - FIXED_GRASP_Z);

        geometry_msgs::msg::Point target_center_dst = center_B_;
        target_center_dst.z = FIXED_CENTER_Z;

        target_grasp_pose.orientation = calculateVerticalGraspOrientation(target_grasp_pose, center_A_);
        
        // Pre-grasp Pose
        geometry_msgs::msg::Pose pre_grasp_pose = target_grasp_pose;
        pre_grasp_pose.position.z += 0.15; 

        // -----------------------------------------------------------
        // Execution Sequence
        // -----------------------------------------------------------
        
        // 1. GLOBAL APPROACH: Fast Joint Move
        RCLCPP_INFO(this->get_logger(), ">> 1. Approach (Joint - Fast)...");
        if(!moveToPose(pre_grasp_pose)) return;

        // 2. VERTICAL DESCENT: Moderate Linear Move
        RCLCPP_INFO(this->get_logger(), ">> 2. Descent (Cartesian - Moderate)...");
        std::vector<geometry_msgs::msg::Pose> waypoints;
        waypoints.push_back(target_grasp_pose);
        if(!moveCartesian(waypoints, VEL_CART_MOVE)) return; // Moderate speed

        closeGripper();
        rclcpp::sleep_for(1s);

        // 3. VERTICAL LIFT: Moderate Linear Move (High Lift)
        
        RCLCPP_INFO(this->get_logger(), ">> 3. Lift 25cm (Cartesian - Moderate)...");
        geometry_msgs::msg::Pose lift_pose = target_grasp_pose;
        lift_pose.position.z = FIXED_GRASP_Z + LIFT_HEIGHT; 
        
        waypoints.clear();
        waypoints.push_back(lift_pose);
        if(!moveCartesian(waypoints, VEL_CART_MOVE)) return; // Moderate speed

        // 4. TRANSFER: Moderate Linear Move (Maintain Height)
        tf2::Vector3 p_pivot_target_world(target_center_dst.x, target_center_dst.y, target_center_dst.z + POUR_PRE_HEIGHT_OFFSET);
        tf2::Vector3 p_tcp_target_world = p_pivot_target_world - v_tcp_to_pivot;

        geometry_msgs::msg::Pose high_transfer_pose = lift_pose;
        high_transfer_pose.position.x = p_tcp_target_world.x();
        high_transfer_pose.position.y = p_tcp_target_world.y();

        RCLCPP_INFO(this->get_logger(), ">> 4. Transfer (Cartesian - Moderate)...");
        waypoints.clear();
        waypoints.push_back(high_transfer_pose);
        if(!moveCartesian(waypoints, VEL_CART_MOVE)) return; // Moderate speed

        // 5. POUR ALIGN: Moderate Linear Move
        geometry_msgs::msg::Pose pour_align_pose = high_transfer_pose;
        pour_align_pose.position.z = p_tcp_target_world.z(); 

        RCLCPP_INFO(this->get_logger(), ">> 5. Align Descent (Cartesian - Moderate)...");
        waypoints.clear();
        waypoints.push_back(pour_align_pose);
        if(!moveCartesian(waypoints, VEL_CART_MOVE)) return; // Moderate speed

        // 6. DIP: Slow Linear Move (Careful)
        RCLCPP_INFO(this->get_logger(), ">> 6. Dip (Cartesian - Slow)...");
        geometry_msgs::msg::Pose pour_start_pose = pour_align_pose;
        pour_start_pose.position.z -= POUR_DIP_HEIGHT;
        
        geometry_msgs::msg::Point current_pivot_pos;
        current_pivot_pos.x = p_pivot_target_world.x();
        current_pivot_pos.y = p_pivot_target_world.y();
        current_pivot_pos.z = p_pivot_target_world.z() - POUR_DIP_HEIGHT;

        waypoints.clear();
        waypoints.push_back(pour_start_pose);
        moveCartesian(waypoints, VEL_CART_POUR); // Switch to Slow speed

        // 7. POUR: Very Slow Rotation
        RCLCPP_INFO(this->get_logger(), ">> 7. Pouring (Cartesian - Very Slow)...");
        auto pour_path = generatePivotPath(pour_start_pose, current_pivot_pos, POUR_ANGLE);
        moveCartesian(pour_path, VEL_CART_POUR); // Very Slow speed
        
        rclcpp::sleep_for(2000ms); 

        // 8. RETURN TILT: Very Slow Rotation
        RCLCPP_INFO(this->get_logger(), ">> 8. Tilt Back (Cartesian - Very Slow)...");
        std::reverse(pour_path.begin(), pour_path.end());
        moveCartesian(pour_path, VEL_CART_POUR); // Very Slow speed

        // 9. RETURN PATH: Moderate Speed
        RCLCPP_INFO(this->get_logger(), ">> 9. Return (Cartesian - Moderate)...");
        waypoints.clear();
        waypoints.push_back(pour_align_pose); 
        waypoints.push_back(high_transfer_pose); 
        moveCartesian(waypoints, VEL_CART_MOVE); // Back to Moderate speed

        // 10. GLOBAL RETURN: Fast Joint Move
        // Reset to pre-grasp using joint planner to unwind any awkward joints
        moveToPose(pre_grasp_pose); 
        
        // 11. PLACE: Moderate Linear Move
        RCLCPP_INFO(this->get_logger(), ">> 10. Place (Cartesian - Moderate)...");
        waypoints.clear();
        waypoints.push_back(target_grasp_pose);
        moveCartesian(waypoints, VEL_CART_MOVE); // Moderate speed

        openGripper();
        rclcpp::sleep_for(500ms);

        // 12. RETREAT: Fast Linear Move
        waypoints.clear();
        waypoints.push_back(pre_grasp_pose);
        moveCartesian(waypoints, VEL_CART_MOVE); // Moderate speed
    }

    void executeLoop()
    {
        while (rclcpp::ok()) {
            RCLCPP_INFO(this->get_logger(), "Waiting for 6 points...");
            while(rclcpp::ok() && !(flag_grasp_A_ && flag_center_A_ && flag_pivot_A_ &&
                                    flag_grasp_B_ && flag_center_B_ && flag_pivot_B_)) {
                rclcpp::sleep_for(500ms);
            }
            if(!rclcpp::ok()) break;

            flag_grasp_A_ = false; flag_center_A_ = false; flag_pivot_A_ = false;
            flag_grasp_B_ = false; flag_center_B_ = false; flag_pivot_B_ = false;

            executeTask();
        }
    }
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<DualCupPouring>();
    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node);
    executor.spin();
    rclcpp::shutdown();
    return 0;
}
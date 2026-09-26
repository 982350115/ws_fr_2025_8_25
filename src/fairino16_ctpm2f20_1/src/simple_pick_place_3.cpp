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

// --- 时间参数化相关头文件 ---
#include <moveit/trajectory_processing/iterative_time_parameterization.h>
#include <moveit/robot_trajectory/robot_trajectory.h>

// TF2 Headers
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2/LinearMath/Quaternion.h>

#include <thread>
#include <chrono>

using namespace std::chrono_literals;

// 定义常量
const std::string MOVE_GROUP_NAME = "manipulator";
const std::string TARGET_EE_LINK = "gripper_base_link"; 
const std::string GRIPPER_SERVICE_NAME = "/gripper_set_open"; 

// 抓取相关参数
const double EE_TO_GRASP_OFFSET_Z = 0.280;  
const double PRE_GRASP_CLEARANCE = 0.15;   
const double BALL_RADIUS = 0.025;          

// 位置定义
const double BALL_X = 0.45;
const double BALL_Y = 0.3;
const double BALL_Z = 0.0;

const double PLACE_X = 0.62;
const double PLACE_Y = 0.13;
const double PLACE_Z = 0.0;

// 全局移动速度
const double VELOCITY_SCALE = 0.05;
const double ACCELERATION_SCALE = 0.05;

// 笛卡尔直线运动的超低速因子
const double CARTESIAN_VEL_SCALE = 0.005; 
const double CARTESIAN_ACC_SCALE = 0.005;

class SimplePickPlace : public rclcpp::Node
{
public:
    SimplePickPlace() : Node("simple_pick_place_3")
    {
        // --- 修改 1: 创建一个“互斥回调组”专门给 run 任务使用 ---
        // 这样 run 任务就不会阻塞默认回调组（MoveIt 用默认组来收 joint_states）
        cb_group_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

        // --- 修改 2: 将定时器加入到这个特殊的组中 ---
        init_timer_ = this->create_wall_timer(
            100ms, std::bind(&SimplePickPlace::run, this), cb_group_);
    }

private:
    rclcpp::TimerBase::SharedPtr init_timer_;
    // --- 修改 3: 声明回调组变量 ---
    rclcpp::CallbackGroup::SharedPtr cb_group_;
    
    std::shared_ptr<moveit::planning_interface::MoveGroupInterface> move_group_;
    std::shared_ptr<moveit::planning_interface::PlanningSceneInterface> planning_scene_interface_;

    void run()
    {
        init_timer_->cancel(); 

        move_group_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(shared_from_this(), MOVE_GROUP_NAME);
        planning_scene_interface_ = std::make_shared<moveit::planning_interface::PlanningSceneInterface>();

        move_group_->setMaxVelocityScalingFactor(VELOCITY_SCALE);
        move_group_->setMaxAccelerationScalingFactor(ACCELERATION_SCALE);

        setupPlanningScene();

        controlGripper(true);
        rclcpp::sleep_for(1s);

        executeTask();
    }

    bool controlGripper(bool open)
    {
        // 夹爪控制也建议用独立的 Client，这里我们直接新建一个临时节点最稳妥
        auto temp_node = rclcpp::Node::make_shared("temp_gripper_client");
        auto client = temp_node->create_client<example_interfaces::srv::SetBool>(GRIPPER_SERVICE_NAME);

        RCLCPP_INFO(this->get_logger(), "连接夹爪服务: %s ...", GRIPPER_SERVICE_NAME.c_str());
        if (!client->wait_for_service(std::chrono::seconds(2))) {
            RCLCPP_ERROR(this->get_logger(), "连接失败！请先运行 ros2 run gripper_driver gripper_service");
            return false;
        }

        auto request = std::make_shared<example_interfaces::srv::SetBool::Request>();
        request->data = open; 
        auto result_future = client->async_send_request(request);

        if (rclcpp::spin_until_future_complete(temp_node, result_future) == rclcpp::FutureReturnCode::SUCCESS)
        {
            auto response = result_future.get();
            if (response->success) {
                RCLCPP_INFO(this->get_logger(), "夹爪动作完成: %s", response->message.c_str());
                return true;
            } else {
                RCLCPP_WARN(this->get_logger(), "夹爪报错: %s", response->message.c_str());
                return false;
            }
        }
        else {
            RCLCPP_ERROR(this->get_logger(), "服务调用超时");
            return false;
        }
    }

    void setupPlanningScene()
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

        moveit_msgs::msg::CollisionObject ball;
        ball.id = "target_ball";
        ball.header.frame_id = move_group_->getPlanningFrame();
        ball.operation = ball.ADD;
        shape_msgs::msg::SolidPrimitive ball_shape;
        ball_shape.type = shape_msgs::msg::SolidPrimitive::SPHERE;
        ball_shape.dimensions.resize(1); ball_shape.dimensions[0] = BALL_RADIUS;
        geometry_msgs::msg::Pose ball_pose;
        ball_pose.position.x = BALL_X; ball_pose.position.y = BALL_Y; ball_pose.position.z = BALL_Z + BALL_RADIUS; ball_pose.orientation.w = 1.0;
        ball.primitives.push_back(ball_shape); ball.primitive_poses.push_back(ball_pose);

        planning_scene_interface_->applyCollisionObjects({table, ball});
        RCLCPP_INFO(this->get_logger(), "场景初始化完成");
    }

    void openGripper() { controlGripper(true); }
    void closeGripper() { controlGripper(false); }

    void attachBall()
    {
        moveit_msgs::msg::AttachedCollisionObject attached_object;
        attached_object.link_name = TARGET_EE_LINK;
        attached_object.object.header.frame_id = move_group_->getPlanningFrame();
        attached_object.object.id = "target_ball";
        attached_object.object.operation = attached_object.object.ADD;
        attached_object.touch_links = {"gripper_left_link", "gripper_right_link", TARGET_EE_LINK};
        planning_scene_interface_->applyAttachedCollisionObject(attached_object);
    }

    void detachBall(const geometry_msgs::msg::Pose& drop_pose)
    {
        moveit_msgs::msg::AttachedCollisionObject detach_object;
        detach_object.object.id = "target_ball";
        detach_object.link_name = TARGET_EE_LINK;
        detach_object.object.operation = detach_object.object.REMOVE;
        planning_scene_interface_->applyAttachedCollisionObject(detach_object);

        moveit_msgs::msg::CollisionObject ball;
        ball.id = "target_ball";
        ball.header.frame_id = move_group_->getPlanningFrame();
        ball.operation = ball.ADD;
        shape_msgs::msg::SolidPrimitive ball_shape;
        ball_shape.type = shape_msgs::msg::SolidPrimitive::SPHERE;
        ball_shape.dimensions.resize(1); ball_shape.dimensions[0] = BALL_RADIUS;
        ball.primitives.push_back(ball_shape); ball.primitive_poses.push_back(drop_pose);
        planning_scene_interface_->applyCollisionObjects({ball});
    }

    bool moveToPose(const geometry_msgs::msg::Pose& target_pose)
    {
        move_group_->setPoseTarget(target_pose);
        moveit::planning_interface::MoveGroupInterface::Plan my_plan;
        if (move_group_->plan(my_plan) == moveit::core::MoveItErrorCode::SUCCESS) {
            move_group_->execute(my_plan); return true;
        } else {
            RCLCPP_ERROR(this->get_logger(), "规划失败!"); return false;
        }
    }

    bool moveCartesian(const std::vector<geometry_msgs::msg::Pose>& waypoints)
    {
        moveit_msgs::msg::RobotTrajectory trajectory_msg;
        const double jump_threshold = 0.0; 
        const double eef_step = 0.01;      
        
        double fraction = move_group_->computeCartesianPath(waypoints, eef_step, jump_threshold, trajectory_msg);
        
        if (fraction > 0.9) {
            RCLCPP_INFO(this->get_logger(), "笛卡尔路径计算成功 (%.2f%%)，正在进行时间重参数化(降速)...", fraction * 100);

            // 获取 RobotModel
            auto robot_model = move_group_->getRobotModel();
            
            // --- 这里的 getCurrentState() 现在可以工作了！---
            // 因为 run() 在单独的回调组，不阻塞默认组，所以 joint_states 可以被处理
            auto current_state = move_group_->getCurrentState(); 
            
            if (!robot_model || !current_state) {
                RCLCPP_ERROR(this->get_logger(), "无法获取 RobotModel 或 CurrentState");
                return false;
            }

            robot_trajectory::RobotTrajectory rt(robot_model, move_group_->getName());
            rt.setRobotTrajectoryMsg(*current_state, trajectory_msg);

            // 使用超低速因子
            trajectory_processing::IterativeParabolicTimeParameterization iptp;
            bool success = iptp.computeTimeStamps(rt, CARTESIAN_VEL_SCALE, CARTESIAN_ACC_SCALE);

            if (success) {
                moveit::planning_interface::MoveGroupInterface::Plan plan;
                rt.getRobotTrajectoryMsg(plan.trajectory_);
                move_group_->execute(plan);
                return true;
            } else {
                RCLCPP_ERROR(this->get_logger(), "时间参数化失败！");
                return false;
            }
        } else {
            RCLCPP_ERROR(this->get_logger(), "直线规划失败 (%.2f)", fraction);
            return false;
        }
    }

    geometry_msgs::msg::Quaternion getGraspOrientation()
    {
        tf2::Quaternion q;
        q.setRPY(M_PI, 0.0, 0.0); 
        return tf2::toMsg(q);
    }

    bool goHomeJoints()
    {
        std::vector<double> current_joints = move_group_->getCurrentJointValues();
        std::vector<double> target_joints = {
            0.71, -2.4, -0.2, -1.9, 1.53, 0.0 
        };

        if (target_joints.size() != 6) {
            RCLCPP_ERROR(this->get_logger(), "关节角数量必须是 6 个！");
            return false;
        }

        move_group_->setJointValueTarget(target_joints);

        moveit::planning_interface::MoveGroupInterface::Plan my_plan;
        if (move_group_->plan(my_plan) == moveit::core::MoveItErrorCode::SUCCESS) {
            RCLCPP_INFO(this->get_logger(), "正在移动到自定义 Home 点...");
            move_group_->execute(my_plan);
            return true;
        } else {
            RCLCPP_ERROR(this->get_logger(), "回 Home 规划失败！");
            return false;
        }
    }
    
    void executeTask()
    {
        // 你的任务逻辑保持不变
        geometry_msgs::msg::Pose start_pose; 
        start_pose.position.x = BALL_X;
        start_pose.position.y = BALL_Y;
        start_pose.position.z = BALL_Z + BALL_RADIUS + EE_TO_GRASP_OFFSET_Z; 
        start_pose.orientation = getGraspOrientation();

        geometry_msgs::msg::Pose place_pose; 
        place_pose.position.x = PLACE_X;
        place_pose.position.y = PLACE_Y;
        place_pose.position.z = PLACE_Z + BALL_RADIUS + EE_TO_GRASP_OFFSET_Z + 0.02; 
        place_pose.orientation = getGraspOrientation();

        geometry_msgs::msg::Pose pre_grasp_pose = start_pose;
        pre_grasp_pose.position.z += PRE_GRASP_CLEARANCE;

        geometry_msgs::msg::Pose pre_place_pose = place_pose;
        pre_place_pose.position.z += PRE_GRASP_CLEARANCE;

        RCLCPP_INFO(this->get_logger(), "1. 移动到预抓取位置...");
        if (!moveToPose(pre_grasp_pose)) return;

        RCLCPP_INFO(this->get_logger(), "2. 下降 (慢速)...");
        std::vector<geometry_msgs::msg::Pose> waypoints;
        waypoints.push_back(start_pose);
        if (!moveCartesian(waypoints)) return;

        RCLCPP_INFO(this->get_logger(), "3. 闭合夹爪...");
        closeGripper(); 
        attachBall();   

        RCLCPP_INFO(this->get_logger(), "4. 抬起 (慢速)...");
        waypoints.clear();
        waypoints.push_back(pre_grasp_pose);
        if (!moveCartesian(waypoints)) return;

        RCLCPP_INFO(this->get_logger(), "5. 移动到放置点上方...");
        if (!moveToPose(pre_place_pose)) return;

        RCLCPP_INFO(this->get_logger(), "6. 下降 (慢速)...");
        waypoints.clear();
        waypoints.push_back(place_pose);
        if (!moveCartesian(waypoints)) return;

        RCLCPP_INFO(this->get_logger(), "7. 张开夹爪...");
        openGripper(); 
        
        geometry_msgs::msg::Pose drop_ball_pose;
        drop_ball_pose.position.x = PLACE_X;
        drop_ball_pose.position.y = PLACE_Y;
        drop_ball_pose.position.z = PLACE_Z + BALL_RADIUS;
        drop_ball_pose.orientation.w = 1.0;
        detachBall(drop_ball_pose); 

        RCLCPP_INFO(this->get_logger(), "8. 抬起离开 (慢速)...");
        waypoints.clear();
        waypoints.push_back(pre_place_pose);
        if (!moveCartesian(waypoints)) return;

        RCLCPP_INFO(this->get_logger(), "--- 逆向流程演示 ---");

        RCLCPP_INFO(this->get_logger(), "9. 下降重抓 (慢速)...");
        waypoints.clear();
        waypoints.push_back(place_pose);
        if (!moveCartesian(waypoints)) return;

        RCLCPP_INFO(this->get_logger(), "10. 闭合夹爪...");
        closeGripper();
        attachBall();

        RCLCPP_INFO(this->get_logger(), "11. 抬起 (慢速)...");
        waypoints.clear();
        waypoints.push_back(pre_place_pose);
        if (!moveCartesian(waypoints)) return;

        RCLCPP_INFO(this->get_logger(), "12. 回到起点上方...");
        if (!moveToPose(pre_grasp_pose)) return;

        geometry_msgs::msg::Pose adjusted_start_pose = start_pose;
        adjusted_start_pose.position.z += 0.03; 
        
        RCLCPP_INFO(this->get_logger(), "13. 下降放置 (慢速)...");
        waypoints.clear();
        waypoints.push_back(adjusted_start_pose); 
        if (!moveCartesian(waypoints)) return;

        RCLCPP_INFO(this->get_logger(), "14. 张开释放...");
        openGripper();
        
        geometry_msgs::msg::Pose original_ball_pose;
        original_ball_pose.position.x = BALL_X;
        original_ball_pose.position.y = BALL_Y;
        original_ball_pose.position.z = BALL_Z + BALL_RADIUS;
        original_ball_pose.orientation.w = 1.0;
        detachBall(original_ball_pose);

        RCLCPP_INFO(this->get_logger(), "15. 抬起 (慢速)...");
        waypoints.clear();
        waypoints.push_back(pre_grasp_pose);
        if (!moveCartesian(waypoints)) return;

        RCLCPP_INFO(this->get_logger(), "16. 移动到自定义 Home 点...");
        if (goHomeJoints()) {
            RCLCPP_INFO(this->get_logger(), "任务完美结束！");
        } else {
            RCLCPP_ERROR(this->get_logger(), "最后一步回零失败！");
        }
    }
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<SimplePickPlace>();
    
    // 依然使用多线程执行器
    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node);
    executor.spin();

    rclcpp::shutdown();
    return 0;
}
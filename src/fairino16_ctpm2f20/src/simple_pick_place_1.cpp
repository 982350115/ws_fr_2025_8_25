#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>

// --- 修改 1: 引入服务头文件，移除 trajectory_msgs ---
#include <example_interfaces/srv/set_bool.hpp> 

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
// const std::string GRIPPER_CONTROLLER_TOPIC = "/gripper_controller/joint_trajectory"; // 已移除
// const std::string GRIPPER_JOINT_NAME = "gripper_left_joint"; // 已移除 (不需要C++发关节状态了)
const std::string TARGET_EE_LINK = "gripper_base_link"; 
const std::string WORLD_FRAME = "world";

// --- 修改 2: 定义服务名称 (必须和 python 节点一致) ---
const std::string GRIPPER_SERVICE_NAME = "/gripper_set_open"; 

// 抓取相关参数
const double EE_TO_GRASP_OFFSET_Z = 0.205; 
const double PRE_GRASP_CLEARANCE = 0.15;   
const double BALL_RADIUS = 0.025;          

const double BALL_X = 0.5;
const double BALL_Y = 0.2;
const double BALL_Z = 0.0;

const double PLACE_X = 0.3;
const double PLACE_Y = 0.4;
const double PLACE_Z = 0.0;

const double VELOCITY_SCALE = 0.5;
const double ACCELERATION_SCALE = 0.5;

class SimplePickPlace : public rclcpp::Node
{
public:
    SimplePickPlace() : Node("simple_pick_place")
    {
        // 移除原有的 publisher

        // 使用单独的线程运行主要逻辑
        init_timer_ = this->create_wall_timer(
            100ms, std::bind(&SimplePickPlace::run, this));
    }

private:
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

        move_group_->setMaxVelocityScalingFactor(VELOCITY_SCALE);
        move_group_->setMaxAccelerationScalingFactor(ACCELERATION_SCALE);

        setupPlanningScene();

        // 确保夹爪处于打开状态 (True = Open)
        controlGripper(true);
        rclcpp::sleep_for(1s);

        executeTask();
    }

    // --- 修改 3: 新增控制实物夹爪的核心函数 ---
    // 使用服务客户端模式，这会让实体夹爪动，且等待动作完成
    bool controlGripper(bool open)
    {
        // 创建一个临时的节点来发送请求，防止卡住主线程
        auto temp_node = rclcpp::Node::make_shared("temp_gripper_client");
        auto client = temp_node->create_client<example_interfaces::srv::SetBool>(GRIPPER_SERVICE_NAME);

        RCLCPP_INFO(this->get_logger(), "连接夹爪服务: %s ...", GRIPPER_SERVICE_NAME.c_str());
        if (!client->wait_for_service(std::chrono::seconds(2))) {
            RCLCPP_ERROR(this->get_logger(), "连接失败！请先运行 ros2 run gripper_driver gripper_service");
            return false;
        }

        auto request = std::make_shared<example_interfaces::srv::SetBool::Request>();
        request->data = open; // true=张开, false=闭合

        RCLCPP_INFO(this->get_logger(), "发送指令: %s", open ? "张开" : "闭合");
        
        // 发送异步请求
        auto result_future = client->async_send_request(request);

        // 等待结果
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
        // 1. 添加桌子 (Z=0 平面)
        moveit_msgs::msg::CollisionObject table;
        table.id = "table";
        table.header.frame_id = move_group_->getPlanningFrame();
        table.operation = table.ADD;

        shape_msgs::msg::SolidPrimitive table_shape;
        table_shape.type = shape_msgs::msg::SolidPrimitive::BOX;
        table_shape.dimensions = {2.0, 2.0, 0.02}; 

        geometry_msgs::msg::Pose table_pose;
        table_pose.position.x = 0.0;
        table_pose.position.y = 0.0;
        table_pose.position.z = -0.01; 
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
        ball_pose.position.z = BALL_Z + BALL_RADIUS; 
        ball_pose.orientation.w = 1.0;

        ball.primitives.push_back(ball_shape);
        ball.primitive_poses.push_back(ball_pose);

        planning_scene_interface_->applyCollisionObjects({table, ball});
        RCLCPP_INFO(this->get_logger(), "Planning scene initialized.");
    }

    // --- 修改 4: 重写 Open/Close 函数，调用服务 ---
    void openGripper()
    {
        controlGripper(true); 
    }

    void closeGripper()
    {
        controlGripper(false);
    }

    // 附着小球
    void attachBall()
    {
        moveit_msgs::msg::AttachedCollisionObject attached_object;
        attached_object.link_name = TARGET_EE_LINK;
        attached_object.object.header.frame_id = move_group_->getPlanningFrame();
        attached_object.object.id = "target_ball";
        attached_object.object.operation = attached_object.object.ADD;
        attached_object.touch_links = {"gripper_left_link", "gripper_right_link", TARGET_EE_LINK};

        planning_scene_interface_->applyAttachedCollisionObject(attached_object);
        RCLCPP_INFO(this->get_logger(), "Ball attached (Software Logic).");
    }

    // 分离小球
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
        ball_shape.dimensions.resize(1);
        ball_shape.dimensions[0] = BALL_RADIUS;

        ball.primitives.push_back(ball_shape);
        ball.primitive_poses.push_back(drop_pose);

        planning_scene_interface_->applyCollisionObjects({ball});
        RCLCPP_INFO(this->get_logger(), "Ball detached (Software Logic).");
    }

    bool moveToPose(const geometry_msgs::msg::Pose& target_pose)
    {
        move_group_->setPoseTarget(target_pose);
        moveit::planning_interface::MoveGroupInterface::Plan my_plan;
        if (move_group_->plan(my_plan) == moveit::core::MoveItErrorCode::SUCCESS)
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

    bool moveCartesian(const std::vector<geometry_msgs::msg::Pose>& waypoints)
    {
        moveit_msgs::msg::RobotTrajectory trajectory;
        const double jump_threshold = 1.5; 
        const double eef_step = 0.01;
        
        double fraction = move_group_->computeCartesianPath(waypoints, eef_step, jump_threshold, trajectory);
        
        if (fraction > 0.9) 
        {
            move_group_->execute(trajectory);
            return true;
        }
        else
        {
            RCLCPP_ERROR(this->get_logger(), "Cartesian path failed (%.2f)", fraction);
            return false;
        }
    }

    geometry_msgs::msg::Quaternion getGraspOrientation()
    {
        tf2::Quaternion q;
        q.setRPY(M_PI, 0.0, 0.0); 
        return tf2::toMsg(q);
    }
    
    // 任务流程 (逻辑保持不变，但 closeGripper 内部已经变了)
    void executeTask()
    {
        // 1. 定义位姿
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

        // --- 流程开始 ---

        RCLCPP_INFO(this->get_logger(), "1. 移动到预抓取位置...");
        if (!moveToPose(pre_grasp_pose)) return;

        RCLCPP_INFO(this->get_logger(), "2. 下降...");
        std::vector<geometry_msgs::msg::Pose> waypoints;
        waypoints.push_back(start_pose);
        if (!moveCartesian(waypoints)) return;

        RCLCPP_INFO(this->get_logger(), "3. 闭合夹爪 (实体动作)...");
        closeGripper(); // <--- 调用服务，等待实体到位
        attachBall();   // <--- 告诉 MoveIt 手里有东西了

        RCLCPP_INFO(this->get_logger(), "4. 抬起...");
        waypoints.clear();
        waypoints.push_back(pre_grasp_pose);
        if (!moveCartesian(waypoints)) return;

        RCLCPP_INFO(this->get_logger(), "5. 移动到放置点上方...");
        if (!moveToPose(pre_place_pose)) return;

        RCLCPP_INFO(this->get_logger(), "6. 下降...");
        waypoints.clear();
        waypoints.push_back(place_pose);
        if (!moveCartesian(waypoints)) return;

        RCLCPP_INFO(this->get_logger(), "7. 张开夹爪 (实体动作)...");
        openGripper(); // <--- 调用服务
        
        geometry_msgs::msg::Pose drop_ball_pose;
        drop_ball_pose.position.x = PLACE_X;
        drop_ball_pose.position.y = PLACE_Y;
        drop_ball_pose.position.z = PLACE_Z + BALL_RADIUS;
        drop_ball_pose.orientation.w = 1.0;
        detachBall(drop_ball_pose); // <--- 告诉 MoveIt 东西放下了

        RCLCPP_INFO(this->get_logger(), "8. 抬起离开...");
        waypoints.clear();
        waypoints.push_back(pre_place_pose);
        if (!moveCartesian(waypoints)) return;

        // --- 回归部分 ---
        RCLCPP_INFO(this->get_logger(), "--- 逆向流程演示 ---");

        RCLCPP_INFO(this->get_logger(), "9. 下降重抓...");
        waypoints.clear();
        waypoints.push_back(place_pose);
        if (!moveCartesian(waypoints)) return;

        RCLCPP_INFO(this->get_logger(), "10. 闭合夹爪...");
        closeGripper();
        attachBall();

        RCLCPP_INFO(this->get_logger(), "11. 抬起...");
        waypoints.clear();
        waypoints.push_back(pre_place_pose);
        if (!moveCartesian(waypoints)) return;

        RCLCPP_INFO(this->get_logger(), "12. 回到起点上方...");
        if (!moveToPose(pre_grasp_pose)) return;

        geometry_msgs::msg::Pose adjusted_start_pose = start_pose;
        adjusted_start_pose.position.z += 0.03; 
        
        RCLCPP_INFO(this->get_logger(), "13. 下降放置...");
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

        RCLCPP_INFO(this->get_logger(), "15. 抬起...");
        waypoints.clear();
        waypoints.push_back(pre_grasp_pose);
        if (!moveCartesian(waypoints)) return;

        RCLCPP_INFO(this->get_logger(), "16. 回家...");
        move_group_->setNamedTarget("home");
        moveit::planning_interface::MoveGroupInterface::Plan home_plan;
        if (move_group_->plan(home_plan) == moveit::core::MoveItErrorCode::SUCCESS)
        {
            move_group_->execute(home_plan);
            RCLCPP_INFO(this->get_logger(), "任务全部完成！");
        }
        else
        {
            RCLCPP_ERROR(this->get_logger(), "无法规划到Home点");
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
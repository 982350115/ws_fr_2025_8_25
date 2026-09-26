#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit_msgs/msg/collision_object.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>

// 注意：这里没有显式包含 <moveit/core/moveit_error_codes.h>

using std::placeholders::_1;

class MoveToTargetNode : public rclcpp::Node
{
public:
    MoveToTargetNode() : Node("move_to_target_node")
    {
        // 关键修复：使用定时器延迟初始化 MoveIt 接口，解决 std::bad_weak_ptr 错误
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(100),
            std::bind(&MoveToTargetNode::initialize_and_start, this));
    }

private:
    moveit::planning_interface::MoveGroupInterfacePtr move_group_;
    moveit::planning_interface::PlanningSceneInterfacePtr planning_scene_interface_;
    rclcpp::TimerBase::SharedPtr timer_; // 用于延迟初始化

    void initialize_and_start()
    {
        // 停止并销毁定时器，确保初始化只运行一次
        timer_.reset();

        // 创建 MoveGroupInterface 控制机械臂 (现在 shared_from_this() 是安全的)
        move_group_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(shared_from_this(), "manipulator");
        planning_scene_interface_ = std::make_shared<moveit::planning_interface::PlanningSceneInterface>();

        RCLCPP_INFO(this->get_logger(), "MoveGroupInterface 和 PlanningSceneInterface 已初始化。");

        // 添加桌子作为障碍物
        addTableToPlanningScene();

        // 移动机械臂到目标位姿
        moveToTargetPose(0.4, 0.3, 1.0);
    }

    void addTableToPlanningScene()
    {
        moveit_msgs::msg::CollisionObject table;
        table.id = "work_table";
        table.header.frame_id = move_group_->getPlanningFrame();

        // 定义桌子形状
        shape_msgs::msg::SolidPrimitive table_primitive;
        table_primitive.type = shape_msgs::msg::SolidPrimitive::BOX;
        table_primitive.dimensions = {2.0, 2.0, 0.05}; // 与Gazebo定义一致

        // 桌子中心位置
        geometry_msgs::msg::Pose table_pose;
        table_pose.orientation.w = 1.0;
        table_pose.position.x = 0.0;
        table_pose.position.y = 0.0;
        // 修正 Z 坐标：与 Gazebo SDF <pose>0 0 0.75... 匹配，0.75是几何体中心的高度
        table_pose.position.z = 0.75; 

        table.primitives.push_back(table_primitive);
        table.primitive_poses.push_back(table_pose);
        table.operation = table.ADD;

        // 添加到场景
        planning_scene_interface_->applyCollisionObjects({table});

        RCLCPP_INFO(this->get_logger(), "已将桌子添加到 MoveIt 场景，中心Z=0.75m。");
    }

    void moveToTargetPose(double x, double y, double z)
    {
        geometry_msgs::msg::Pose target_pose;
        target_pose.orientation.w = 1.0;
        target_pose.position.x = x;
        target_pose.position.y = y;
        target_pose.position.z = z;

        move_group_->setPoseTarget(target_pose);

        // 设置规划参数
        move_group_->setMaxVelocityScalingFactor(0.5);
        move_group_->setMaxAccelerationScalingFactor(0.5);
        move_group_->setPlanningTime(5.0);

        moveit::planning_interface::MoveGroupInterface::Plan plan;
        // 使用推荐的 moveit::core::MoveItErrorCode，尽管头文件未显式包含，但 MoveGroupInterface.h 已间接提供
        bool success = (move_group_->plan(plan) == moveit::core::MoveItErrorCode::SUCCESS);

        if (success)
        {
            RCLCPP_INFO(this->get_logger(), "规划成功，开始执行。");
            move_group_->execute(plan);
        }
        else
        {
            RCLCPP_WARN(this->get_logger(), "规划失败。");
        }
    }
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<MoveToTargetNode>()); 
    rclcpp::shutdown();
    return 0;
}


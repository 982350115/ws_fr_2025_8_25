#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>

#include <moveit/planning_scene/planning_scene.h>
#include <moveit/robot_model_loader/robot_model_loader.h>

#include <moveit/task_constructor/task.h>
#include <moveit/task_constructor/stage.h>
#include <moveit/task_constructor/container.h>  // SerialContainer
#include <moveit/task_constructor/stages/current_state.h>
#include <moveit/task_constructor/stages/connect.h>
#include <moveit/task_constructor/stages/move_to.h>
#include <moveit/task_constructor/stages/move_relative.h>
#include <moveit/task_constructor/stages/modify_planning_scene.h>

using namespace moveit::task_constructor;

static const std::string PLANNING_GROUP = "arm";
static const std::string GRIPPER_GROUP = "gripper";
static const std::string EE_LINK = "tool0";
static const std::string OBJECT_ID = "object";

class PickPlaceNode : public rclcpp::Node
{
public:
  PickPlaceNode() : Node("pick_place_mtc")
  {
    RCLCPP_INFO(get_logger(), "Starting Pick and Place (Humble API)");

    task_.reset(new Task("pick_place_task"));
    buildTask();
    runTask();
  }

private:
  std::unique_ptr<Task> task_;

  void buildTask()
  {
    task_->setProperty("group", PLANNING_GROUP);

    // ================================
    // 1. Current State
    // ================================
    auto current_state = std::make_unique<stages::CurrentState>("current state");
    task_->add(std::move(current_state));

    // ================================
    // 2. Pick container
    // ================================
    auto pick = std::make_unique<SerialContainer>("pick");
    pick->properties().configureInitFrom(Stage::PARENT);

    // --- Move to pre-grasp ---
    auto move_pre = std::make_unique<stages::MoveTo>("move to pregrasp", PLANNING_GROUP);
    move_pre->setProperty("group", PLANNING_GROUP);

    geometry_msgs::msg::PoseStamped pre_pose;
    pre_pose.header.frame_id = "base_link";
    pre_pose.pose.position.x = 0.4;
    pre_pose.pose.position.y = 0;
    pre_pose.pose.position.z = 0.3;
    pre_pose.pose.orientation.w = 1.0;

    move_pre->setIKFrame(EE_LINK);
    move_pre->setGoalPose(pre_pose.pose);

    pick->add(std::move(move_pre));

    // --- Approach ---
    auto approach = std::make_unique<stages::MoveRelative>("approach", PLANNING_GROUP);
    geometry_msgs::msg::Vector3Stamped vec;
    vec.header.frame_id = EE_LINK;
    vec.vector.z = -0.10;
    approach->setDirection(vec);
    pick->add(std::move(approach));

    // --- Attach Object ---
    auto attach = std::make_unique<stages::ModifyPlanningScene>("attach object");
    attach->attachObject(OBJECT_ID, EE_LINK);
    pick->add(std::move(attach));

    task_->add(std::move(pick));

    // ================================
    // 3. Place container
    // ================================
    auto place = std::make_unique<SerialContainer>("place");
    place->properties().configureInitFrom(Stage::PARENT);

    // --- Retreat ---
    auto retreat = std::make_unique<stages::MoveRelative>("retreat", PLANNING_GROUP);
    vec.vector.z = 0.15;
    retreat->setDirection(vec);
    place->add(std::move(retreat));

    // --- Move to place pose ---
    auto move_place = std::make_unique<stages::MoveTo>("move to place", PLANNING_GROUP);
    move_place->setIKFrame(EE_LINK);

    geometry_msgs::msg::PoseStamped place_pose;
    place_pose.header.frame_id = "base_link";
    place_pose.pose.position.x = 0.4;
    place_pose.pose.position.y = 0.2;
    place_pose.pose.position.z = 0.3;
    place_pose.pose.orientation.w = 1.0;

    move_place->setGoalPose(place_pose.pose);
    place->add(std::move(move_place));

    // --- Detach ---
    auto detach = std::make_unique<stages::ModifyPlanningScene>("detach object");
    detach->detachObject(OBJECT_ID);
    place->add(std::move(detach));

    task_->add(std::move(place));
  }

  // =======================================
  // Task execution
  // =======================================
  void runTask()
  {
    if (!task_->plan(5))
    {
      RCLCPP_ERROR(get_logger(), "Task planning failed");
      return;
    }
    RCLCPP_INFO(get_logger(), "Task planned successfully");
    task_->execute();
  }
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<PickPlaceNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}


#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <geometry_msgs/msg/wrench_stamped.hpp>
#include <std_srvs/srv/set_bool.hpp>
#include <cmath>
#include <algorithm>

using std::placeholders::_1;
using std::placeholders::_2;

class ForceControllerNode : public rclcpp::Node
{
public:
    ForceControllerNode() : Node("force_controller_node_cpp")
    {
        RCLCPP_INFO(this->get_logger(), "Force Controller Node Started (Wait for Trigger).");

        // 参数初始化
        this->declare_parameter<double>("desired_force", 2.0);
        this->declare_parameter<double>("Kp", 0.1); // 调大一点以克服静摩擦
        this->declare_parameter<double>("Ki", 0.1);
        this->declare_parameter<double>("effort_limit", 50.0);

        this->get_parameter("desired_force", desired_force_);
        this->get_parameter("Kp", Kp_);
        this->get_parameter("Ki", Ki_);
        this->get_parameter("effort_limit", effort_limit_);

        // 状态变量
        control_enabled_ = false; // 默认为关闭状态
        measured_force_z_ = 0.0;
        integral_error_ = 0.0;
        control_dt_ = 0.01; // 100Hz

        // 1. 订阅力传感器 (⚠️ 话题名称保持用户原状)
        force_subscriber_ = this->create_subscription<geometry_msgs::msg::WrenchStamped>(
            "/ft_sensor_broadcaster/wrench", 10, 
            std::bind(&ForceControllerNode::force_callback, this, _1));
        
        // 2. 发布力矩指令
        effort_publisher_ = this->create_publisher<std_msgs::msg::Float64MultiArray>(
            "/gripper_controller/commands", 10);

        // 3. 创建控制开关服务
        server_ = this->create_service<std_srvs::srv::SetBool>(
            "toggle_force_control",
            std::bind(&ForceControllerNode::handle_service, this, _1, _2));

        // 4. 定时器
        timer_ = this->create_wall_timer(
            std::chrono::duration<double>(control_dt_),
            std::bind(&ForceControllerNode::control_loop, this));
    }

private:
    double desired_force_, Kp_, Ki_, effort_limit_, control_dt_;
    double measured_force_z_, integral_error_;
    bool control_enabled_; 

    rclcpp::Subscription<geometry_msgs::msg::WrenchStamped>::SharedPtr force_subscriber_;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr effort_publisher_;
    rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr server_;
    rclcpp::TimerBase::SharedPtr timer_;

    // 力传感器回调
    void force_callback(const geometry_msgs::msg::WrenchStamped::SharedPtr msg)
    {
        // 假设 Z 轴方向是夹持力的主要测量方向
        measured_force_z_ = msg->wrench.force.z; 
    }

    // 服务回调函数：处理开启/关闭请求
    void handle_service(const std::shared_ptr<std_srvs::srv::SetBool::Request> request,
                        std::shared_ptr<std_srvs::srv::SetBool::Response> response)
    {
        control_enabled_ = request->data;
        
        if (control_enabled_) {
            // 开启时重置积分项，避免积分累积导致突变
            integral_error_ = 0.0;
            response->message = "Force Control ENABLED. Gripper Closing...";
        } else {
            response->message = "Force Control DISABLED. Gripper Relaxed.";
        }
        
        response->success = true;
        RCLCPP_INFO(this->get_logger(), "%s", response->message.c_str());
    }

    // 控制循环
    void control_loop()
    {
        double command_effort = 0.0;

        if (control_enabled_) 
        {
            // --- 恒力抓取模式 (PID) ---
            
            // 误差 = 目标力 - 实际测得力 (取绝对值，因为力传感器可能返回负值)
            double error = desired_force_ - std::abs(measured_force_z_);
            
            double p_term = Kp_ * error;
            
            // 积分项计算
            integral_error_ += error * control_dt_;
            double max_integral = effort_limit_ / (Ki_ + 1e-6);
            integral_error_ = std::clamp(integral_error_, -max_integral, max_integral);
            double i_term = Ki_ * integral_error_;
            
            command_effort = p_term + i_term;
            
            // 限制力矩范围 (只允许闭合方向的正力矩，且不超过上限)
            command_effort = std::clamp(command_effort, 0.0, effort_limit_);
        }
        else
        {
            // --- 禁用模式 (张开夹爪) ---
            // 发送一个负向力矩，让夹爪保持张开/放松
            command_effort = -2.0; 
        }

        // ✅ [关键修改] 发布指令：同时发送给左爪和右爪
        auto effort_msg = std::make_unique<std_msgs::msg::Float64MultiArray>();
        
        // 1. 为左爪 (gripper_left_joint) 推入力矩
        effort_msg->data.push_back(-command_effort); 
        
        // 2. 为右爪 (gripper_right_joint) 推入相同的力矩
        // 这确保了左右爪都能主动发力进行夹持/放松。
        effort_msg->data.push_back(command_effort); 
        
        effort_publisher_->publish(std::move(effort_msg));
        
        RCLCPP_DEBUG(this->get_logger(), "Published effort: [%.2f, %.2f], Measured Fz: %.2f", 
                     -command_effort, command_effort, measured_force_z_);
    }
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ForceControllerNode>());
    rclcpp::shutdown();
    return 0;
}
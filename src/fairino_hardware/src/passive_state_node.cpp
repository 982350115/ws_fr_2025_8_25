#include "fairino_hardware/command_server.hpp"
#include "rclcpp/rclcpp.hpp"

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);

  // Deliberately instantiate only the TCP feedback receiver. In particular,
  // this process creates no command service and no ros2_control write loop.
  auto state_node = std::make_shared<robot_recv_thread>("fairino_passive_state");
  rclcpp::spin(state_node);

  rclcpp::shutdown();
  return 0;
}

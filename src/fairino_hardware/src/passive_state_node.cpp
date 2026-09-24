#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "fairino_hardware/passive_frame_parser.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/joint_state.hpp"

using namespace std::chrono_literals;

// This node only reads 8081 feedback and publishes the six verified joint
// positions. It does not instantiate the vendor command server or motion code.
class PassiveJointStateNode : public rclcpp::Node
{
public:
  PassiveJointStateNode()
  : Node("fairino_passive_state"),
    robot_ip_(declare_parameter<std::string>("robot_ip", "192.168.58.2")),
    state_port_(declare_parameter<int>("state_port", 8081))
  {
    if (state_port_ < 1 || state_port_ > 65535) {
      throw std::invalid_argument("state_port must be in the range 1..65535");
    }
    publisher_ = create_publisher<sensor_msgs::msg::JointState>(
      "joint_states", rclcpp::SensorDataQoS());
    timer_ = create_wall_timer(10ms, [this]() {poll_feedback();});
    RCLCPP_INFO(get_logger(), "Read-only joint feedback: %s:%d", robot_ip_.c_str(), state_port_);
  }

  ~PassiveJointStateNode() override
  {
    if (socket_fd_ >= 0) {
      close(socket_fd_);
    }
  }

private:
  void connect_feedback()
  {
    const auto current = std::chrono::steady_clock::now();
    if (current < next_connect_attempt_) {
      return;
    }
    next_connect_attempt_ = current + 1s;

    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
      return;
    }
    const int flags = fcntl(fd, F_GETFL, 0);
    if (flags < 0 || fcntl(fd, F_SETFL, flags | O_NONBLOCK) < 0) {
      close(fd);
      return;
    }
    sockaddr_in endpoint{};
    endpoint.sin_family = AF_INET;
    endpoint.sin_port = htons(static_cast<std::uint16_t>(state_port_));
    if (inet_pton(AF_INET, robot_ip_.c_str(), &endpoint.sin_addr) != 1) {
      close(fd);
      throw std::invalid_argument("robot_ip must be an IPv4 address");
    }
    int result = connect(fd, reinterpret_cast<sockaddr *>(&endpoint), sizeof(endpoint));
    if (result < 0 && errno == EINPROGRESS) {
      pollfd wait_for_connect{fd, POLLOUT, 0};
      if (poll(&wait_for_connect, 1, 200) > 0) {
        int error = 0;
        socklen_t error_size = sizeof(error);
        if (getsockopt(fd, SOL_SOCKET, SO_ERROR, &error, &error_size) == 0) {
          result = error == 0 ? 0 : -1;
        }
      }
    }
    if (result != 0) {
      close(fd);
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 5000, "Waiting for feedback at %s:%d",
        robot_ip_.c_str(), state_port_);
      return;
    }
    socket_fd_ = fd;
    parser_.reset();
    RCLCPP_INFO(get_logger(), "Feedback socket connected");
  }

  void disconnect_feedback()
  {
    close(socket_fd_);
    socket_fd_ = -1;
    parser_.reset();
    RCLCPP_WARN(get_logger(), "Feedback socket disconnected; reconnecting");
  }

  void poll_feedback()
  {
    if (socket_fd_ < 0) {
      connect_feedback();
      return;
    }

    std::array<std::uint8_t, 4096> chunk{};
    for (int read_count = 0; read_count < 16; ++read_count) {
      const auto received = recv(socket_fd_, chunk.data(), chunk.size(), 0);
      if (received > 0) {
        parser_.append(chunk.data(), static_cast<std::size_t>(received));
        std::array<double, 6> joints_deg{};
        std::uint32_t frame_length = 0;
        while (parser_.next(joints_deg, frame_length)) {
          if (!warned_version_mismatch_ && frame_length != sizeof(_CTRL_STATE) - 14) {
            RCLCPP_WARN(
              get_logger(),
              "Controller frame length %u differs from local full-state layout %zu; "
              "publishing only the verified six joint positions",
              frame_length, sizeof(_CTRL_STATE) - 14);
            warned_version_mismatch_ = true;
          }
          sensor_msgs::msg::JointState joint_state;
          joint_state.header.stamp = now();
          joint_state.name = {"j1", "j2", "j3", "j4", "j5", "j6"};
          joint_state.position.reserve(6);
          for (double angle : joints_deg) {
            joint_state.position.push_back(angle * 0.017453292519943295);
          }
          publisher_->publish(joint_state);
        }
        continue;
      }
      if (received == 0 || (errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR)) {
        disconnect_feedback();
      }
      break;
    }
  }

  const std::string robot_ip_;
  const int state_port_;
  int socket_fd_ = -1;
  bool warned_version_mismatch_ = false;
  std::chrono::steady_clock::time_point next_connect_attempt_{};
  fairino_hardware::PassiveFrameParser parser_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr publisher_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<PassiveJointStateNode>());
  rclcpp::shutdown();
  return 0;
}

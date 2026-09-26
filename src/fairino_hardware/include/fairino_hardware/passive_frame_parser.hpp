#ifndef FAIRINO_HARDWARE_PASSIVE_FRAME_PARSER_HPP_
#define FAIRINO_HARDWARE_PASSIVE_FRAME_PARSER_HPP_

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "fairino_hardware/data_type_def.h"

namespace fairino_hardware
{

// The six current joint positions are in the verified prefix of the 8081 frame.
// Later fields must not be interpreted using an older controller layout.
class PassiveFrameParser
{
public:
  static constexpr std::size_t kJointOffset = offsetof(_CTRL_STATE, jt_cur_pos);
  static constexpr std::size_t kMaximumFrameSize = 65536;

  void reset() {bytes_.clear();}

  void append(const std::uint8_t * data, std::size_t size)
  {
    bytes_.insert(bytes_.end(), data, data + size);
    if (bytes_.size() > 2 * kMaximumFrameSize) {
      bytes_.erase(bytes_.begin(), bytes_.end() - kMaximumFrameSize);
    }
  }

  bool next(std::array<double, 6> & joints_deg, std::uint32_t & frame_length)
  {
    constexpr std::array<std::uint8_t, 4> head{{'/', 'f', '/', 'b'}};
    constexpr std::array<std::uint8_t, 7> tail{{'I', 'I', 'I', '/', 'b', '/', 'f'}};
    constexpr std::size_t header_size = 11;
    constexpr std::size_t minimum_size = kJointOffset + 6 * sizeof(double) + tail.size();

    for (;;) {
      auto start = std::search(bytes_.begin(), bytes_.end(), head.begin(), head.end());
      if (start == bytes_.end()) {
        if (bytes_.size() > head.size() - 1) {
          bytes_.erase(bytes_.begin(), bytes_.end() - (head.size() - 1));
        }
        return false;
      }
      bytes_.erase(bytes_.begin(), start);
      if (bytes_.size() < header_size) {
        return false;
      }

      // The controller's length includes its four length bytes and payload.
      // The seven-byte head and tail are outside that count.
      std::int32_t length = 0;
      std::memcpy(&length, bytes_.data() + 7, sizeof(length));
      if (length < static_cast<std::int32_t>(minimum_size - 14) ||
        length > static_cast<std::int32_t>(kMaximumFrameSize - 14))
      {
        bytes_.erase(bytes_.begin());
        continue;
      }
      const auto total_size = static_cast<std::size_t>(length) + 14;
      if (bytes_.size() < total_size) {
        return false;
      }
      if (!std::equal(tail.begin(), tail.end(), bytes_.begin() + total_size - tail.size())) {
        bytes_.erase(bytes_.begin());
        continue;
      }

      std::memcpy(joints_deg.data(), bytes_.data() + kJointOffset, 6 * sizeof(double));
      bytes_.erase(bytes_.begin(), bytes_.begin() + total_size);
      if (std::any_of(joints_deg.begin(), joints_deg.end(),
        [](double angle) {return !std::isfinite(angle) || std::abs(angle) > 720.0;}))
      {
        continue;
      }
      frame_length = static_cast<std::uint32_t>(length);
      return true;
    }
  }

private:
  std::vector<std::uint8_t> bytes_;
};

}  // namespace fairino_hardware

#endif  // FAIRINO_HARDWARE_PASSIVE_FRAME_PARSER_HPP_

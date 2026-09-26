#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include "gtest/gtest.h"
#include "fairino_hardware/passive_frame_parser.hpp"

namespace
{

std::vector<std::uint8_t> make_frame(std::size_t extra_bytes = 0)
{
  std::vector<std::uint8_t> frame(sizeof(_CTRL_STATE) + extra_bytes, 0);
  const char head[] = "/f/bIII";
  const char tail[] = "III/b/f";
  std::memcpy(frame.data(), head, 7);
  const auto length = static_cast<std::int32_t>(frame.size() - 14);
  std::memcpy(frame.data() + 7, &length, sizeof(length));
  const std::array<double, 6> joints{{27.8, -84.2, 121.5, -129.0, -86.3, -136.3}};
  std::memcpy(
    frame.data() + fairino_hardware::PassiveFrameParser::kJointOffset,
    joints.data(), sizeof(joints));
  std::memcpy(frame.data() + frame.size() - 7, tail, 7);
  return frame;
}

TEST(PassiveFrameParser, accepts_fragmented_extended_frame)
{
  fairino_hardware::PassiveFrameParser parser;
  auto frame = make_frame(256);
  std::array<double, 6> joints{};
  std::uint32_t length = 0;
  for (std::size_t pos = 0; pos < frame.size(); pos += 137) {
    const auto count = std::min<std::size_t>(137, frame.size() - pos);
    parser.append(frame.data() + pos, count);
    if (pos + count < frame.size()) {
      EXPECT_FALSE(parser.next(joints, length));
    }
  }
  ASSERT_TRUE(parser.next(joints, length));
  EXPECT_EQ(length, frame.size() - 14);
  EXPECT_DOUBLE_EQ(joints[0], 27.8);
  EXPECT_DOUBLE_EQ(joints[2], 121.5);
  EXPECT_FALSE(parser.next(joints, length));
}

TEST(PassiveFrameParser, resynchronizes_after_junk_and_bad_tail)
{
  fairino_hardware::PassiveFrameParser parser;
  const auto bad = make_frame();
  auto corrupted = bad;
  corrupted.back() = '?';
  const auto good = make_frame(128);
  const std::uint8_t junk[] = {0, 1, 2, 3, 4};
  parser.append(junk, sizeof(junk));
  parser.append(corrupted.data(), corrupted.size());
  parser.append(good.data(), good.size());
  std::array<double, 6> joints{};
  std::uint32_t length = 0;
  ASSERT_TRUE(parser.next(joints, length));
  EXPECT_EQ(length, good.size() - 14);
  EXPECT_FALSE(parser.next(joints, length));
}

TEST(PassiveFrameParser, rejects_impossible_length_and_nonfinite_joint)
{
  fairino_hardware::PassiveFrameParser parser;
  auto bad_length = make_frame();
  const std::int32_t huge = 100000000;
  std::memcpy(bad_length.data() + 7, &huge, sizeof(huge));
  auto bad_joint = make_frame();
  const double nan = std::numeric_limits<double>::quiet_NaN();
  std::memcpy(
    bad_joint.data() + fairino_hardware::PassiveFrameParser::kJointOffset,
    &nan, sizeof(nan));
  const auto good = make_frame();
  parser.append(bad_length.data(), bad_length.size());
  parser.append(bad_joint.data(), bad_joint.size());
  parser.append(good.data(), good.size());
  std::array<double, 6> joints{};
  std::uint32_t length = 0;
  ASSERT_TRUE(parser.next(joints, length));
  EXPECT_DOUBLE_EQ(joints[0], 27.8);
  EXPECT_FALSE(parser.next(joints, length));
}

}  // namespace

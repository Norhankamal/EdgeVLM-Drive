// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from dms_interfaces:msg/FramePair.idl
// generated code does not contain a copyright notice

#ifndef DMS_INTERFACES__MSG__DETAIL__FRAME_PAIR__BUILDER_HPP_
#define DMS_INTERFACES__MSG__DETAIL__FRAME_PAIR__BUILDER_HPP_

#include "dms_interfaces/msg/detail/frame_pair__struct.hpp"
#include <rosidl_runtime_cpp/message_initialization.hpp>
#include <algorithm>
#include <utility>


namespace dms_interfaces
{

namespace msg
{

namespace builder
{

class Init_FramePair_yaw
{
public:
  explicit Init_FramePair_yaw(::dms_interfaces::msg::FramePair & msg)
  : msg_(msg)
  {}
  ::dms_interfaces::msg::FramePair yaw(::dms_interfaces::msg::FramePair::_yaw_type arg)
  {
    msg_.yaw = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dms_interfaces::msg::FramePair msg_;
};

class Init_FramePair_ear
{
public:
  explicit Init_FramePair_ear(::dms_interfaces::msg::FramePair & msg)
  : msg_(msg)
  {}
  Init_FramePair_yaw ear(::dms_interfaces::msg::FramePair::_ear_type arg)
  {
    msg_.ear = std::move(arg);
    return Init_FramePair_yaw(msg_);
  }

private:
  ::dms_interfaces::msg::FramePair msg_;
};

class Init_FramePair_mesh_ms
{
public:
  explicit Init_FramePair_mesh_ms(::dms_interfaces::msg::FramePair & msg)
  : msg_(msg)
  {}
  Init_FramePair_ear mesh_ms(::dms_interfaces::msg::FramePair::_mesh_ms_type arg)
  {
    msg_.mesh_ms = std::move(arg);
    return Init_FramePair_ear(msg_);
  }

private:
  ::dms_interfaces::msg::FramePair msg_;
};

class Init_FramePair_gate_reason
{
public:
  explicit Init_FramePair_gate_reason(::dms_interfaces::msg::FramePair & msg)
  : msg_(msg)
  {}
  Init_FramePair_mesh_ms gate_reason(::dms_interfaces::msg::FramePair::_gate_reason_type arg)
  {
    msg_.gate_reason = std::move(arg);
    return Init_FramePair_mesh_ms(msg_);
  }

private:
  ::dms_interfaces::msg::FramePair msg_;
};

class Init_FramePair_skip
{
public:
  explicit Init_FramePair_skip(::dms_interfaces::msg::FramePair & msg)
  : msg_(msg)
  {}
  Init_FramePair_gate_reason skip(::dms_interfaces::msg::FramePair::_skip_type arg)
  {
    msg_.skip = std::move(arg);
    return Init_FramePair_gate_reason(msg_);
  }

private:
  ::dms_interfaces::msg::FramePair msg_;
};

class Init_FramePair_mesh_enabled
{
public:
  explicit Init_FramePair_mesh_enabled(::dms_interfaces::msg::FramePair & msg)
  : msg_(msg)
  {}
  Init_FramePair_skip mesh_enabled(::dms_interfaces::msg::FramePair::_mesh_enabled_type arg)
  {
    msg_.mesh_enabled = std::move(arg);
    return Init_FramePair_skip(msg_);
  }

private:
  ::dms_interfaces::msg::FramePair msg_;
};

class Init_FramePair_preprocess_ms
{
public:
  explicit Init_FramePair_preprocess_ms(::dms_interfaces::msg::FramePair & msg)
  : msg_(msg)
  {}
  Init_FramePair_mesh_enabled preprocess_ms(::dms_interfaces::msg::FramePair::_preprocess_ms_type arg)
  {
    msg_.preprocess_ms = std::move(arg);
    return Init_FramePair_mesh_enabled(msg_);
  }

private:
  ::dms_interfaces::msg::FramePair msg_;
};

class Init_FramePair_face_crop
{
public:
  explicit Init_FramePair_face_crop(::dms_interfaces::msg::FramePair & msg)
  : msg_(msg)
  {}
  Init_FramePair_preprocess_ms face_crop(::dms_interfaces::msg::FramePair::_face_crop_type arg)
  {
    msg_.face_crop = std::move(arg);
    return Init_FramePair_preprocess_ms(msg_);
  }

private:
  ::dms_interfaces::msg::FramePair msg_;
};

class Init_FramePair_full_crop
{
public:
  explicit Init_FramePair_full_crop(::dms_interfaces::msg::FramePair & msg)
  : msg_(msg)
  {}
  Init_FramePair_face_crop full_crop(::dms_interfaces::msg::FramePair::_full_crop_type arg)
  {
    msg_.full_crop = std::move(arg);
    return Init_FramePair_face_crop(msg_);
  }

private:
  ::dms_interfaces::msg::FramePair msg_;
};

class Init_FramePair_seq
{
public:
  explicit Init_FramePair_seq(::dms_interfaces::msg::FramePair & msg)
  : msg_(msg)
  {}
  Init_FramePair_full_crop seq(::dms_interfaces::msg::FramePair::_seq_type arg)
  {
    msg_.seq = std::move(arg);
    return Init_FramePair_full_crop(msg_);
  }

private:
  ::dms_interfaces::msg::FramePair msg_;
};

class Init_FramePair_header
{
public:
  Init_FramePair_header()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_FramePair_seq header(::dms_interfaces::msg::FramePair::_header_type arg)
  {
    msg_.header = std::move(arg);
    return Init_FramePair_seq(msg_);
  }

private:
  ::dms_interfaces::msg::FramePair msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::dms_interfaces::msg::FramePair>()
{
  return dms_interfaces::msg::builder::Init_FramePair_header();
}

}  // namespace dms_interfaces

#endif  // DMS_INTERFACES__MSG__DETAIL__FRAME_PAIR__BUILDER_HPP_

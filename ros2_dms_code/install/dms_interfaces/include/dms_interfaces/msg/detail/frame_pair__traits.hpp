// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from dms_interfaces:msg/FramePair.idl
// generated code does not contain a copyright notice

#ifndef DMS_INTERFACES__MSG__DETAIL__FRAME_PAIR__TRAITS_HPP_
#define DMS_INTERFACES__MSG__DETAIL__FRAME_PAIR__TRAITS_HPP_

#include "dms_interfaces/msg/detail/frame_pair__struct.hpp"
#include <rosidl_runtime_cpp/traits.hpp>
#include <stdint.h>
#include <type_traits>

// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__traits.hpp"
// Member 'full_crop'
// Member 'face_crop'
#include "sensor_msgs/msg/detail/image__traits.hpp"

namespace rosidl_generator_traits
{

template<>
inline const char * data_type<dms_interfaces::msg::FramePair>()
{
  return "dms_interfaces::msg::FramePair";
}

template<>
inline const char * name<dms_interfaces::msg::FramePair>()
{
  return "dms_interfaces/msg/FramePair";
}

template<>
struct has_fixed_size<dms_interfaces::msg::FramePair>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<dms_interfaces::msg::FramePair>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<dms_interfaces::msg::FramePair>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // DMS_INTERFACES__MSG__DETAIL__FRAME_PAIR__TRAITS_HPP_

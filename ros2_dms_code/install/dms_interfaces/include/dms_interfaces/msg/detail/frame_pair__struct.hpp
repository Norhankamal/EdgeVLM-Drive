// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from dms_interfaces:msg/FramePair.idl
// generated code does not contain a copyright notice

#ifndef DMS_INTERFACES__MSG__DETAIL__FRAME_PAIR__STRUCT_HPP_
#define DMS_INTERFACES__MSG__DETAIL__FRAME_PAIR__STRUCT_HPP_

#include <rosidl_runtime_cpp/bounded_vector.hpp>
#include <rosidl_runtime_cpp/message_initialization.hpp>
#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>


// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__struct.hpp"
// Member 'full_crop'
// Member 'face_crop'
#include "sensor_msgs/msg/detail/image__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__dms_interfaces__msg__FramePair __attribute__((deprecated))
#else
# define DEPRECATED__dms_interfaces__msg__FramePair __declspec(deprecated)
#endif

namespace dms_interfaces
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct FramePair_
{
  using Type = FramePair_<ContainerAllocator>;

  explicit FramePair_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_init),
    full_crop(_init),
    face_crop(_init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->seq = 0ul;
      this->preprocess_ms = 0.0f;
      this->mesh_enabled = false;
      this->skip = false;
      this->gate_reason = "";
      this->mesh_ms = 0.0f;
      this->ear = 0.0f;
      this->yaw = 0.0f;
    }
  }

  explicit FramePair_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_alloc, _init),
    full_crop(_alloc, _init),
    face_crop(_alloc, _init),
    gate_reason(_alloc)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->seq = 0ul;
      this->preprocess_ms = 0.0f;
      this->mesh_enabled = false;
      this->skip = false;
      this->gate_reason = "";
      this->mesh_ms = 0.0f;
      this->ear = 0.0f;
      this->yaw = 0.0f;
    }
  }

  // field types and members
  using _header_type =
    std_msgs::msg::Header_<ContainerAllocator>;
  _header_type header;
  using _seq_type =
    uint32_t;
  _seq_type seq;
  using _full_crop_type =
    sensor_msgs::msg::Image_<ContainerAllocator>;
  _full_crop_type full_crop;
  using _face_crop_type =
    sensor_msgs::msg::Image_<ContainerAllocator>;
  _face_crop_type face_crop;
  using _preprocess_ms_type =
    float;
  _preprocess_ms_type preprocess_ms;
  using _mesh_enabled_type =
    bool;
  _mesh_enabled_type mesh_enabled;
  using _skip_type =
    bool;
  _skip_type skip;
  using _gate_reason_type =
    std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other>;
  _gate_reason_type gate_reason;
  using _mesh_ms_type =
    float;
  _mesh_ms_type mesh_ms;
  using _ear_type =
    float;
  _ear_type ear;
  using _yaw_type =
    float;
  _yaw_type yaw;

  // setters for named parameter idiom
  Type & set__header(
    const std_msgs::msg::Header_<ContainerAllocator> & _arg)
  {
    this->header = _arg;
    return *this;
  }
  Type & set__seq(
    const uint32_t & _arg)
  {
    this->seq = _arg;
    return *this;
  }
  Type & set__full_crop(
    const sensor_msgs::msg::Image_<ContainerAllocator> & _arg)
  {
    this->full_crop = _arg;
    return *this;
  }
  Type & set__face_crop(
    const sensor_msgs::msg::Image_<ContainerAllocator> & _arg)
  {
    this->face_crop = _arg;
    return *this;
  }
  Type & set__preprocess_ms(
    const float & _arg)
  {
    this->preprocess_ms = _arg;
    return *this;
  }
  Type & set__mesh_enabled(
    const bool & _arg)
  {
    this->mesh_enabled = _arg;
    return *this;
  }
  Type & set__skip(
    const bool & _arg)
  {
    this->skip = _arg;
    return *this;
  }
  Type & set__gate_reason(
    const std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other> & _arg)
  {
    this->gate_reason = _arg;
    return *this;
  }
  Type & set__mesh_ms(
    const float & _arg)
  {
    this->mesh_ms = _arg;
    return *this;
  }
  Type & set__ear(
    const float & _arg)
  {
    this->ear = _arg;
    return *this;
  }
  Type & set__yaw(
    const float & _arg)
  {
    this->yaw = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    dms_interfaces::msg::FramePair_<ContainerAllocator> *;
  using ConstRawPtr =
    const dms_interfaces::msg::FramePair_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<dms_interfaces::msg::FramePair_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<dms_interfaces::msg::FramePair_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      dms_interfaces::msg::FramePair_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<dms_interfaces::msg::FramePair_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      dms_interfaces::msg::FramePair_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<dms_interfaces::msg::FramePair_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<dms_interfaces::msg::FramePair_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<dms_interfaces::msg::FramePair_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__dms_interfaces__msg__FramePair
    std::shared_ptr<dms_interfaces::msg::FramePair_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__dms_interfaces__msg__FramePair
    std::shared_ptr<dms_interfaces::msg::FramePair_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const FramePair_ & other) const
  {
    if (this->header != other.header) {
      return false;
    }
    if (this->seq != other.seq) {
      return false;
    }
    if (this->full_crop != other.full_crop) {
      return false;
    }
    if (this->face_crop != other.face_crop) {
      return false;
    }
    if (this->preprocess_ms != other.preprocess_ms) {
      return false;
    }
    if (this->mesh_enabled != other.mesh_enabled) {
      return false;
    }
    if (this->skip != other.skip) {
      return false;
    }
    if (this->gate_reason != other.gate_reason) {
      return false;
    }
    if (this->mesh_ms != other.mesh_ms) {
      return false;
    }
    if (this->ear != other.ear) {
      return false;
    }
    if (this->yaw != other.yaw) {
      return false;
    }
    return true;
  }
  bool operator!=(const FramePair_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct FramePair_

// alias to use template instance with default allocator
using FramePair =
  dms_interfaces::msg::FramePair_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace dms_interfaces

#endif  // DMS_INTERFACES__MSG__DETAIL__FRAME_PAIR__STRUCT_HPP_

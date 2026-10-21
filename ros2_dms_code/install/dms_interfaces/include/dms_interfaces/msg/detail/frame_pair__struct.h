// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from dms_interfaces:msg/FramePair.idl
// generated code does not contain a copyright notice

#ifndef DMS_INTERFACES__MSG__DETAIL__FRAME_PAIR__STRUCT_H_
#define DMS_INTERFACES__MSG__DETAIL__FRAME_PAIR__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__struct.h"
// Member 'full_crop'
// Member 'face_crop'
#include "sensor_msgs/msg/detail/image__struct.h"
// Member 'gate_reason'
#include "rosidl_runtime_c/string.h"

// Struct defined in msg/FramePair in the package dms_interfaces.
typedef struct dms_interfaces__msg__FramePair
{
  std_msgs__msg__Header header;
  uint32_t seq;
  sensor_msgs__msg__Image full_crop;
  sensor_msgs__msg__Image face_crop;
  float preprocess_ms;
  bool mesh_enabled;
  bool skip;
  rosidl_runtime_c__String gate_reason;
  float mesh_ms;
  float ear;
  float yaw;
} dms_interfaces__msg__FramePair;

// Struct for a sequence of dms_interfaces__msg__FramePair.
typedef struct dms_interfaces__msg__FramePair__Sequence
{
  dms_interfaces__msg__FramePair * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} dms_interfaces__msg__FramePair__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // DMS_INTERFACES__MSG__DETAIL__FRAME_PAIR__STRUCT_H_

// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from dms_interfaces:msg/FramePair.idl
// generated code does not contain a copyright notice
#include "dms_interfaces/msg/detail/frame_pair__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `header`
#include "std_msgs/msg/detail/header__functions.h"
// Member `full_crop`
// Member `face_crop`
#include "sensor_msgs/msg/detail/image__functions.h"
// Member `gate_reason`
#include "rosidl_runtime_c/string_functions.h"

bool
dms_interfaces__msg__FramePair__init(dms_interfaces__msg__FramePair * msg)
{
  if (!msg) {
    return false;
  }
  // header
  if (!std_msgs__msg__Header__init(&msg->header)) {
    dms_interfaces__msg__FramePair__fini(msg);
    return false;
  }
  // seq
  // full_crop
  if (!sensor_msgs__msg__Image__init(&msg->full_crop)) {
    dms_interfaces__msg__FramePair__fini(msg);
    return false;
  }
  // face_crop
  if (!sensor_msgs__msg__Image__init(&msg->face_crop)) {
    dms_interfaces__msg__FramePair__fini(msg);
    return false;
  }
  // preprocess_ms
  // mesh_enabled
  // skip
  // gate_reason
  if (!rosidl_runtime_c__String__init(&msg->gate_reason)) {
    dms_interfaces__msg__FramePair__fini(msg);
    return false;
  }
  // mesh_ms
  // ear
  // yaw
  return true;
}

void
dms_interfaces__msg__FramePair__fini(dms_interfaces__msg__FramePair * msg)
{
  if (!msg) {
    return;
  }
  // header
  std_msgs__msg__Header__fini(&msg->header);
  // seq
  // full_crop
  sensor_msgs__msg__Image__fini(&msg->full_crop);
  // face_crop
  sensor_msgs__msg__Image__fini(&msg->face_crop);
  // preprocess_ms
  // mesh_enabled
  // skip
  // gate_reason
  rosidl_runtime_c__String__fini(&msg->gate_reason);
  // mesh_ms
  // ear
  // yaw
}

bool
dms_interfaces__msg__FramePair__are_equal(const dms_interfaces__msg__FramePair * lhs, const dms_interfaces__msg__FramePair * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // header
  if (!std_msgs__msg__Header__are_equal(
      &(lhs->header), &(rhs->header)))
  {
    return false;
  }
  // seq
  if (lhs->seq != rhs->seq) {
    return false;
  }
  // full_crop
  if (!sensor_msgs__msg__Image__are_equal(
      &(lhs->full_crop), &(rhs->full_crop)))
  {
    return false;
  }
  // face_crop
  if (!sensor_msgs__msg__Image__are_equal(
      &(lhs->face_crop), &(rhs->face_crop)))
  {
    return false;
  }
  // preprocess_ms
  if (lhs->preprocess_ms != rhs->preprocess_ms) {
    return false;
  }
  // mesh_enabled
  if (lhs->mesh_enabled != rhs->mesh_enabled) {
    return false;
  }
  // skip
  if (lhs->skip != rhs->skip) {
    return false;
  }
  // gate_reason
  if (!rosidl_runtime_c__String__are_equal(
      &(lhs->gate_reason), &(rhs->gate_reason)))
  {
    return false;
  }
  // mesh_ms
  if (lhs->mesh_ms != rhs->mesh_ms) {
    return false;
  }
  // ear
  if (lhs->ear != rhs->ear) {
    return false;
  }
  // yaw
  if (lhs->yaw != rhs->yaw) {
    return false;
  }
  return true;
}

bool
dms_interfaces__msg__FramePair__copy(
  const dms_interfaces__msg__FramePair * input,
  dms_interfaces__msg__FramePair * output)
{
  if (!input || !output) {
    return false;
  }
  // header
  if (!std_msgs__msg__Header__copy(
      &(input->header), &(output->header)))
  {
    return false;
  }
  // seq
  output->seq = input->seq;
  // full_crop
  if (!sensor_msgs__msg__Image__copy(
      &(input->full_crop), &(output->full_crop)))
  {
    return false;
  }
  // face_crop
  if (!sensor_msgs__msg__Image__copy(
      &(input->face_crop), &(output->face_crop)))
  {
    return false;
  }
  // preprocess_ms
  output->preprocess_ms = input->preprocess_ms;
  // mesh_enabled
  output->mesh_enabled = input->mesh_enabled;
  // skip
  output->skip = input->skip;
  // gate_reason
  if (!rosidl_runtime_c__String__copy(
      &(input->gate_reason), &(output->gate_reason)))
  {
    return false;
  }
  // mesh_ms
  output->mesh_ms = input->mesh_ms;
  // ear
  output->ear = input->ear;
  // yaw
  output->yaw = input->yaw;
  return true;
}

dms_interfaces__msg__FramePair *
dms_interfaces__msg__FramePair__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  dms_interfaces__msg__FramePair * msg = (dms_interfaces__msg__FramePair *)allocator.allocate(sizeof(dms_interfaces__msg__FramePair), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(dms_interfaces__msg__FramePair));
  bool success = dms_interfaces__msg__FramePair__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
dms_interfaces__msg__FramePair__destroy(dms_interfaces__msg__FramePair * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    dms_interfaces__msg__FramePair__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
dms_interfaces__msg__FramePair__Sequence__init(dms_interfaces__msg__FramePair__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  dms_interfaces__msg__FramePair * data = NULL;

  if (size) {
    data = (dms_interfaces__msg__FramePair *)allocator.zero_allocate(size, sizeof(dms_interfaces__msg__FramePair), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = dms_interfaces__msg__FramePair__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        dms_interfaces__msg__FramePair__fini(&data[i - 1]);
      }
      allocator.deallocate(data, allocator.state);
      return false;
    }
  }
  array->data = data;
  array->size = size;
  array->capacity = size;
  return true;
}

void
dms_interfaces__msg__FramePair__Sequence__fini(dms_interfaces__msg__FramePair__Sequence * array)
{
  if (!array) {
    return;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();

  if (array->data) {
    // ensure that data and capacity values are consistent
    assert(array->capacity > 0);
    // finalize all array elements
    for (size_t i = 0; i < array->capacity; ++i) {
      dms_interfaces__msg__FramePair__fini(&array->data[i]);
    }
    allocator.deallocate(array->data, allocator.state);
    array->data = NULL;
    array->size = 0;
    array->capacity = 0;
  } else {
    // ensure that data, size, and capacity values are consistent
    assert(0 == array->size);
    assert(0 == array->capacity);
  }
}

dms_interfaces__msg__FramePair__Sequence *
dms_interfaces__msg__FramePair__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  dms_interfaces__msg__FramePair__Sequence * array = (dms_interfaces__msg__FramePair__Sequence *)allocator.allocate(sizeof(dms_interfaces__msg__FramePair__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = dms_interfaces__msg__FramePair__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
dms_interfaces__msg__FramePair__Sequence__destroy(dms_interfaces__msg__FramePair__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    dms_interfaces__msg__FramePair__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
dms_interfaces__msg__FramePair__Sequence__are_equal(const dms_interfaces__msg__FramePair__Sequence * lhs, const dms_interfaces__msg__FramePair__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!dms_interfaces__msg__FramePair__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
dms_interfaces__msg__FramePair__Sequence__copy(
  const dms_interfaces__msg__FramePair__Sequence * input,
  dms_interfaces__msg__FramePair__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(dms_interfaces__msg__FramePair);
    dms_interfaces__msg__FramePair * data =
      (dms_interfaces__msg__FramePair *)realloc(output->data, allocation_size);
    if (!data) {
      return false;
    }
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!dms_interfaces__msg__FramePair__init(&data[i])) {
        /* free currently allocated and return false */
        for (; i-- > output->capacity; ) {
          dms_interfaces__msg__FramePair__fini(&data[i]);
        }
        free(data);
        return false;
      }
    }
    output->data = data;
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!dms_interfaces__msg__FramePair__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}

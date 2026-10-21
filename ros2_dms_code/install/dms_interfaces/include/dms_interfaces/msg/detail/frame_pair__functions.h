// generated from rosidl_generator_c/resource/idl__functions.h.em
// with input from dms_interfaces:msg/FramePair.idl
// generated code does not contain a copyright notice

#ifndef DMS_INTERFACES__MSG__DETAIL__FRAME_PAIR__FUNCTIONS_H_
#define DMS_INTERFACES__MSG__DETAIL__FRAME_PAIR__FUNCTIONS_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stdlib.h>

#include "rosidl_runtime_c/visibility_control.h"
#include "dms_interfaces/msg/rosidl_generator_c__visibility_control.h"

#include "dms_interfaces/msg/detail/frame_pair__struct.h"

/// Initialize msg/FramePair message.
/**
 * If the init function is called twice for the same message without
 * calling fini inbetween previously allocated memory will be leaked.
 * \param[in,out] msg The previously allocated message pointer.
 * Fields without a default value will not be initialized by this function.
 * You might want to call memset(msg, 0, sizeof(
 * dms_interfaces__msg__FramePair
 * )) before or use
 * dms_interfaces__msg__FramePair__create()
 * to allocate and initialize the message.
 * \return true if initialization was successful, otherwise false
 */
ROSIDL_GENERATOR_C_PUBLIC_dms_interfaces
bool
dms_interfaces__msg__FramePair__init(dms_interfaces__msg__FramePair * msg);

/// Finalize msg/FramePair message.
/**
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_dms_interfaces
void
dms_interfaces__msg__FramePair__fini(dms_interfaces__msg__FramePair * msg);

/// Create msg/FramePair message.
/**
 * It allocates the memory for the message, sets the memory to zero, and
 * calls
 * dms_interfaces__msg__FramePair__init().
 * \return The pointer to the initialized message if successful,
 * otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_dms_interfaces
dms_interfaces__msg__FramePair *
dms_interfaces__msg__FramePair__create();

/// Destroy msg/FramePair message.
/**
 * It calls
 * dms_interfaces__msg__FramePair__fini()
 * and frees the memory of the message.
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_dms_interfaces
void
dms_interfaces__msg__FramePair__destroy(dms_interfaces__msg__FramePair * msg);

/// Check for msg/FramePair message equality.
/**
 * \param[in] lhs The message on the left hand size of the equality operator.
 * \param[in] rhs The message on the right hand size of the equality operator.
 * \return true if messages are equal, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_dms_interfaces
bool
dms_interfaces__msg__FramePair__are_equal(const dms_interfaces__msg__FramePair * lhs, const dms_interfaces__msg__FramePair * rhs);

/// Copy a msg/FramePair message.
/**
 * This functions performs a deep copy, as opposed to the shallow copy that
 * plain assignment yields.
 *
 * \param[in] input The source message pointer.
 * \param[out] output The target message pointer, which must
 *   have been initialized before calling this function.
 * \return true if successful, or false if either pointer is null
 *   or memory allocation fails.
 */
ROSIDL_GENERATOR_C_PUBLIC_dms_interfaces
bool
dms_interfaces__msg__FramePair__copy(
  const dms_interfaces__msg__FramePair * input,
  dms_interfaces__msg__FramePair * output);

/// Initialize array of msg/FramePair messages.
/**
 * It allocates the memory for the number of elements and calls
 * dms_interfaces__msg__FramePair__init()
 * for each element of the array.
 * \param[in,out] array The allocated array pointer.
 * \param[in] size The size / capacity of the array.
 * \return true if initialization was successful, otherwise false
 * If the array pointer is valid and the size is zero it is guaranteed
 # to return true.
 */
ROSIDL_GENERATOR_C_PUBLIC_dms_interfaces
bool
dms_interfaces__msg__FramePair__Sequence__init(dms_interfaces__msg__FramePair__Sequence * array, size_t size);

/// Finalize array of msg/FramePair messages.
/**
 * It calls
 * dms_interfaces__msg__FramePair__fini()
 * for each element of the array and frees the memory for the number of
 * elements.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_dms_interfaces
void
dms_interfaces__msg__FramePair__Sequence__fini(dms_interfaces__msg__FramePair__Sequence * array);

/// Create array of msg/FramePair messages.
/**
 * It allocates the memory for the array and calls
 * dms_interfaces__msg__FramePair__Sequence__init().
 * \param[in] size The size / capacity of the array.
 * \return The pointer to the initialized array if successful, otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_dms_interfaces
dms_interfaces__msg__FramePair__Sequence *
dms_interfaces__msg__FramePair__Sequence__create(size_t size);

/// Destroy array of msg/FramePair messages.
/**
 * It calls
 * dms_interfaces__msg__FramePair__Sequence__fini()
 * on the array,
 * and frees the memory of the array.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_dms_interfaces
void
dms_interfaces__msg__FramePair__Sequence__destroy(dms_interfaces__msg__FramePair__Sequence * array);

/// Check for msg/FramePair message array equality.
/**
 * \param[in] lhs The message array on the left hand size of the equality operator.
 * \param[in] rhs The message array on the right hand size of the equality operator.
 * \return true if message arrays are equal in size and content, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_dms_interfaces
bool
dms_interfaces__msg__FramePair__Sequence__are_equal(const dms_interfaces__msg__FramePair__Sequence * lhs, const dms_interfaces__msg__FramePair__Sequence * rhs);

/// Copy an array of msg/FramePair messages.
/**
 * This functions performs a deep copy, as opposed to the shallow copy that
 * plain assignment yields.
 *
 * \param[in] input The source array pointer.
 * \param[out] output The target array pointer, which must
 *   have been initialized before calling this function.
 * \return true if successful, or false if either pointer
 *   is null or memory allocation fails.
 */
ROSIDL_GENERATOR_C_PUBLIC_dms_interfaces
bool
dms_interfaces__msg__FramePair__Sequence__copy(
  const dms_interfaces__msg__FramePair__Sequence * input,
  dms_interfaces__msg__FramePair__Sequence * output);

#ifdef __cplusplus
}
#endif

#endif  // DMS_INTERFACES__MSG__DETAIL__FRAME_PAIR__FUNCTIONS_H_

/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2018  Virginia Tech
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 *  Chong Peng
 *  Department of Chemistry, Virginia Tech
 *  May 8, 2019
 *
 */

#include <CL/sycl.hpp>
#include "dpct/dpct.hpp"
#include <TiledArray/gpu/kernel/reduce_kernel.h>
#include <TiledArray/gpu/kernel/reduce_kernel_impl.h>

#ifdef TILEDARRAY_HAS_SYCL

namespace TiledArray {

// foreach(i) result *= arg[i]
int product_sycl_kernel(const int *arg, std::size_t n, sycl::queue *stream,
                        int device_id) {
  return product_reduce_sycl_kernel_impl(arg, n, stream, device_id);

}

float product_sycl_kernel(const float *arg, std::size_t n, sycl::queue *stream,
                          int device_id) {
  return product_reduce_sycl_kernel_impl(arg, n, stream, device_id);
}

double product_sycl_kernel(const double *arg, std::size_t n,
                           sycl::queue *stream, int device_id) {
  return product_reduce_sycl_kernel_impl(arg, n, stream, device_id);
}


// foreach(i) result += arg[i]
int sum_sycl_kernel(const int *arg, std::size_t n, sycl::queue *stream,
                    int device_id) {
  return sum_reduce_sycl_kernel_impl(arg, n, stream, device_id);
}

float sum_sycl_kernel(const float *arg, std::size_t n, sycl::queue *stream,
                      int device_id) {
  return sum_reduce_sycl_kernel_impl(arg, n, stream, device_id);
}

double sum_sycl_kernel(const double *arg, std::size_t n, sycl::queue *stream,
                       int device_id) {
  return sum_reduce_sycl_kernel_impl(arg, n, stream, device_id);
}

// foreach(i) result = max(result, arg[i])
int max_sycl_kernel(const int *arg, std::size_t n, sycl::queue *stream,
                    int device_id) {
  return max_reduce_sycl_kernel_impl(arg, n, stream, device_id);
}

float max_sycl_kernel(const float *arg, std::size_t n, sycl::queue *stream,
                      int device_id) {
  return max_reduce_sycl_kernel_impl(arg, n, stream, device_id);
}

double max_sycl_kernel(const double *arg, std::size_t n, sycl::queue *stream,
                       int device_id) {
  return max_reduce_sycl_kernel_impl(arg, n, stream, device_id);
}

// foreach(i) result = min(result, arg[i])
int min_sycl_kernel(const int *arg, std::size_t n, sycl::queue *stream,
                    int device_id) {
  return min_reduce_sycl_kernel_impl(arg, n, stream, device_id);
}

float min_sycl_kernel(const float *arg, std::size_t n, sycl::queue *stream,
                      int device_id) {
  return min_reduce_sycl_kernel_impl(arg, n, stream, device_id);
}

double min_sycl_kernel(const double *arg, std::size_t n, sycl::queue *stream,
                       int device_id) {
  return min_reduce_sycl_kernel_impl(arg, n, stream, device_id);
}

// foreach(i) result = max(result, abs(arg[i]))
int absmax_sycl_kernel(const int *arg, std::size_t n, sycl::queue *stream,
                       int device_id) {
  return absmax_reduce_sycl_kernel_impl(arg, n, stream, device_id);
}

float absmax_sycl_kernel(const float *arg, std::size_t n, sycl::queue *stream,
                         int device_id) {
  return absmax_reduce_sycl_kernel_impl(arg, n, stream, device_id);
}

double absmax_sycl_kernel(const double *arg, std::size_t n, sycl::queue *stream,
                          int device_id) {
  return absmax_reduce_sycl_kernel_impl(arg, n, stream, device_id);
}

// foreach(i) result = min(result, abs(arg[i]))
int absmin_sycl_kernel(const int *arg, std::size_t n, sycl::queue *stream,
                       int device_id) {
  return absmin_reduce_sycl_kernel_impl(arg, n, stream, device_id);
}

float absmin_sycl_kernel(const float *arg, std::size_t n, sycl::queue *stream,
                         int device_id) {
  return absmin_reduce_sycl_kernel_impl(arg, n, stream, device_id);
}

double absmin_sycl_kernel(const double *arg, std::size_t n, sycl::queue *stream,
                          int device_id) {
  return absmin_reduce_sycl_kernel_impl(arg, n, stream, device_id);
}

}  // namespace TiledArray

#endif // TILEDARRAY_HAS_SYCL

/*
* Copyright (C) 2018 Ingenic Semiconductor Co.,Ltd
* File       : quantized_depthwise_conv_op.h
* Authors    : dzhang<dong.zhang@ingenic.com>
* Create Time: 2018-01-20:15:25:37
* Description: This file implements the quantize depthwise conv operation.
*              DepthwiseConv2dNative
*/
#ifndef THIRD_PARTY_TENSORFLOW_CORE_KERNELS_QUANTIZE_DEPTHWISE_CONV_OP_H_
#define THIRD_PARTY_TENSORFLOW_CORE_KERNELS_QUANTIZE_DEPTHWISE_CONV_OP_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/util/tensor_format.h"
#include "tensorflow/core/kernels/neon/types.h"

#include <cassert>
#include <cstdint>
#include <cstdlib>

namespace tensorflow {

using namespace neon;

#ifndef TFLITE_DCHECK
#define TFLITE_DCHECK(condition) (condition) ? (void)0 : assert(false)
#endif

#ifndef TFLITE_DCHECK_EQ
#define TFLITE_DCHECK_EQ(x, y) ((x) == (y)) ? (void)0 : assert(false)
#endif

#ifndef TFLITE_DCHECK_GE
#define TFLITE_DCHECK_GE(x, y) ((x) >= (y)) ? (void)0 : assert(false)
#endif

#ifndef TFLITE_DCHECK_GT
#define TFLITE_DCHECK_GT(x, y) ((x) > (y)) ? (void)0 : assert(false)
#endif

#ifndef TFLITE_DCHECK_LE
#define TFLITE_DCHECK_LE(x, y) ((x) <= (y)) ? (void)0 : assert(false)
#endif

#ifndef TFLITE_DCHECK_LT
#define TFLITE_DCHECK_LT(x, y) ((x) < (y)) ? (void)0 : assert(false)
#endif





struct QuantizeDepthwiseConv2dNativeArgs {
  // Input layer dimensions
  int batch;
  int in_rows;
  int in_cols;
  int in_depth;
  int filter_rows;
  int filter_cols;
  int depth_multiplier;
  int stride;
  int pad_rows;
  int pad_cols;

  // Output layer dimensions
  int out_rows;
  int out_cols;
  int out_depth;

  QuantizeDepthwiseConv2dNativeArgs()
      : batch(0),
        in_rows(0),
        in_cols(0),
        in_depth(0),
        filter_rows(0),
        filter_cols(0),
        depth_multiplier(0),
        stride(0),
        pad_rows(0),
        pad_cols(0),
        out_rows(0),
        out_cols(0),
        out_depth(0) {}
};





// TODO(ahentz): the implementations in kernels/internal/ take a Dims<4> object
// even if the original tensors were not 4D. We should consider rewriting them
// to take a more generic 'shape' object.

inline neon::Dims<4> GetTensorDims(const Tensor* tensor) {
  if (tensor == nullptr) {
    return neon::Dims<4>();
  }

  //tensor->dim_size
  const int size = tensor->dims();

  neon::Dims<4> d;
    for (int i = 0; i < 4; ++i) {
      int src = size - i - 1;
      if (src >= 0) {
        d.sizes[i] = tensor->dim_size(src);
      } else {
        d.sizes[i] = 1;
      }
    }
    d.strides[0] = 1;
    for (int i = 1; i < 4; i++) {
      d.strides[i] = d.strides[i - 1] * d.sizes[i - 1];
    }

  return d;
}





inline void SetNeonDimStrides(neon::Dims<4>* d) {
    int64 stride = 1;
    for (int i = 0; i < 4; ++i) {
      d->strides[i] = stride;
      stride *= d->sizes[i];
    }
  }

inline  neon::Dims<4> ToNeonDims(const TensorShape& input) {
    // Dims in the neon kernels are channel, x, y, batch order.
    neon::Dims<4> result;
    result.sizes[0] = input.dim_size(3);
    result.sizes[1] = input.dim_size(2);
    result.sizes[2] = input.dim_size(1);
    result.sizes[3] = input.dim_size(0);
    SetNeonDimStrides(&result);
    return result;
  }

inline neon::Dims<4> FilterToNeonDims(const TensorShape& filter) {
    // Dims in the neon kernels are channel, x, y, batch order.
    neon::Dims<4> result;
    result.sizes[0] = filter.dim_size(2) * filter.dim_size(3);
    result.sizes[1] = filter.dim_size(1);
    result.sizes[2] = filter.dim_size(0);
    result.sizes[3] = 1;
    SetNeonDimStrides(&result);

    return result;
  }

inline  neon::Dims<4> BiasNeonDims(const TensorShape& filter) {
    // Dims in the neon kernels are channel, x, y, batch order.
    // Bias has only output channel set.
    neon::Dims<4> result;
    result.sizes[0] =
        filter.dim_size(2) * filter.dim_size(3);  // output channels
    result.sizes[1] = 1;
    result.sizes[2] = 1;
    result.sizes[3] = 1;
    SetNeonDimStrides(&result);

    return result;
  }


template <int N>
inline bool IsPackedWithoutStrides(const Dims<N>& dims) {
  int expected_stride = 1;
  for (int d = 0; d < N; d++) {
    if (dims.strides[d] != expected_stride) return false;
    expected_stride *= dims.sizes[d];
  }
  return true;
}




}  // namespace tensorflow










namespace tensorflow {
namespace functor {

// Pads 'filter' to vector-register boundary along its inner dimension:
//   filter_inner_dim_size = in_depth * depth_multiplier
// Requires 'filter' to have the following storage order:
//   [filter_rows, filter_cols, in_depth, depth_multiplier]
// Returns zero-padded filter in 'padded_filter'.
//
// EX:
//   in_depth = 3, depth_multiplier = 2, filter [2, 2], register_width = 4
//   So we have a total of 3 * 2 = 6 filters, each of spatial size 2 x 2.
//
//   filter [rows, cols, in_depth, depth_multiplier]
//     [u0, v0, w0, x0] [y0, z0, u1, v1] [w1, x1, y1, z1]
//     [u2, v2, w2, x2] [y2, z2, u3, v3] [w3, x3, y3, z3]
//
//   padded_filter [rows, cols, in_depth, depth_multiplier]
//     [u0, v0, w0, x0] [y0, z0, 0, 0] [u1, v1, w1, x1] [y1, z1, 0, 0]
//     [u2, v2, w2, x2] [y2, z2, 0, 0] [u3, v3, w3, x3] [y3, z3, 0, 0]

template <typename T>
struct QuantizeDepthwiseConv2dNativeFilterPadOp {
  void operator()(const QuantizeDepthwiseConv2dNativeArgs& args, const T* filter,
                  T* padded_filter) {
    typedef typename Eigen::internal::packet_traits<T>::type Packet;
    static const int64 kPacketSize = (sizeof(Packet) / sizeof(T));

    // Calculate vectorized and scalar lengths of filter's inner dimension.
    const int64 filter_inner_dim_size = args.out_depth;
    const int64 vectorized_size =
        (filter_inner_dim_size / kPacketSize) * kPacketSize;
    const int64 scalar_size = filter_inner_dim_size - vectorized_size;
    // Calculate required padding and padded output buffer stride.
    const int64 pad_size = scalar_size > 0 ? kPacketSize - scalar_size : 0;
    const int64 padded_filter_stride = vectorized_size + kPacketSize;

    const int64 filter_spatial_size = args.filter_rows * args.filter_cols;
    for (int64 i = 0; i < filter_spatial_size; ++i) {
      const int64 input_base = i * filter_inner_dim_size;
      const int64 output_base = i * padded_filter_stride;
      // Write vectorized length of filter's inner dimension to output.
      for (int64 j = 0; j < vectorized_size; j += kPacketSize) {
        const auto v = Eigen::internal::ploadu<Packet>(filter + input_base + j);
        Eigen::internal::pstoreu<T>(padded_filter + output_base + j, v);
      }
      // Write scalar length of filter's inner dimension to output.
      for (int64 j = 0; j < scalar_size; ++j) {
        padded_filter[output_base + vectorized_size + j] =
            filter[input_base + vectorized_size + j];
      }
      // Pad the remainder of output to vector-register boundary.
      for (int64 j = 0; j < pad_size; ++j) {
        padded_filter[output_base + vectorized_size + scalar_size + j] = 0;
      }
    }
  }
};

// Copies data from local region in 'input' specified by 'out_r' and 'out_'c'
// to 'input_buffer'. The copied data is replicated by factor
// 'args.depth_mulitplier', and padded to vector register-width boundaries so
// that it is aligned for efficient traversal and vector multiply-add by the
// depthwise kernel.
//
// EX:
//   in_depth = 3, depth_multiplier = 2, filter [2, 2], register_width = 4
//
//   input: [batch, in_rows, in_cols, in_depth]
//
//     [a0, a1, a2, b0, b1, b2, ..., e0, e1, e2, f0, f1, f2, ...]
//
//   input_buffer (register boundaries shown):
//     [a0, a0, a1, a1] [a2, a2, 0, 0]   in_row = 0, in_col = 0
//     [b0, b0, b1, b1] [b2, b2, 0, 0]   in_row = 0, in_col = 1
//     [e0, e0, e1, e1] [e2, e2, 0, 0]   in_row = 1, in_col = 0
//     [f0, f0, f1, f1] [f2, f2, 0, 0]   in_row = 1, in_col = 1
//
// Returns replicated and padded data from specified input region in
// 'input_buffer'.

template <typename T>
struct QuantizeDepthwiseConv2dNativeInputCopyOp {
  void operator()(const QuantizeDepthwiseConv2dNativeArgs& args,
                  const int64 padded_filter_inner_dim_size, const int64 out_r,
                  const int64 out_c, const T* input, T* input_buffer) {
    typedef typename Eigen::internal::packet_traits<T>::type Packet;
    static const int64 kPacketSize = (sizeof(Packet) / sizeof(T));

    // Calculate vectorized and scalar (residual) lengths for 'in_depth'.
    const int64 input_vectorized_size =
        (args.in_depth / kPacketSize) * kPacketSize;
    const int64 input_scalar_size = args.in_depth % kPacketSize;

    // Calculate vectorized and scalar (residual) lengths for
    // 'depth_multiplier'. This is used to efficiently replicate data for
    // when 'depth_multiplier' > kPacketSize.
    const int64 dm_vectorized_size =
        (args.depth_multiplier / kPacketSize) * kPacketSize;
    const int64 dm_scalar_size = args.depth_multiplier % kPacketSize;

    // Calculate output padding length.
    const int64 output_scalar_size = args.out_depth % kPacketSize;
    const int64 output_pad_size =
        output_scalar_size > 0 ? kPacketSize - output_scalar_size : 0;

    const int64 replicated_packet_size = kPacketSize * args.depth_multiplier;

    // Iterate through all rows x cols reading 'in_depth' from 'input' and
    // replicating by 'depth_multiplier' into 'input_buffer' (otherwise
    // zero-padding input buffer as needed).
    auto* in_buf = input_buffer;
    const int64 in_r_start = out_r * args.stride - args.pad_rows;
    const int64 in_c_start = out_c * args.stride - args.pad_cols;

    for (int64 f_r = 0; f_r < args.filter_rows; ++f_r) {
      const int64 in_r = in_r_start + f_r;

      for (int64 f_c = 0; f_c < args.filter_cols; ++f_c) {
        const int64 in_c = in_c_start + f_c;

        if (in_r >= 0 && in_r < args.in_rows && in_c >= 0 &&
            in_c < args.in_cols) {
          auto* in = input + (in_r * args.in_cols + in_c) * args.in_depth;
          // Copy vectorized portion of inner dimension.
          for (int64 d = 0; d < input_vectorized_size; d += kPacketSize) {
            auto v = Eigen::internal::ploadu<Packet>(in + d);
            for (int dm = 0; dm < args.depth_multiplier; ++dm) {
              Eigen::internal::pscatter<T, Packet>(in_buf + dm, v,
                                                   args.depth_multiplier);
            }
            in_buf += replicated_packet_size;
          }

          // Copy scalar portion of inner dimension.
          for (int64 d = 0; d < input_scalar_size; ++d) {
            T v = in[input_vectorized_size + d];
            const int64 base = d * args.depth_multiplier;
            if (dm_vectorized_size > 0) {
              // Copy vectorized portion of replicated output.
              // This branch is only taken if 'args.depth_multiplier' is
              // vectorizable (i.e. args.depth_multiplier >= register width).
              auto p = Eigen::internal::pset1<Packet>(v);
              for (int64 dm = 0; dm < dm_vectorized_size; dm += kPacketSize) {
                Eigen::internal::pstoreu<T>(in_buf + base + dm, p);
              }
              // Copy scalar portion of replicated output.
              for (int64 dm = 0; dm < dm_scalar_size; ++dm) {
                in_buf[base + dm_vectorized_size + dm] = v;
              }
            } else {
              // Depth multiplier is less than one packet: scalar copy.
              for (int dm = 0; dm < args.depth_multiplier; ++dm) {
                in_buf[base + dm] = v;
              }
            }
          }
          in_buf += input_scalar_size * args.depth_multiplier;

          // Pad the remainder of the output to vector register boundary.
          for (int64 d = 0; d < output_pad_size; ++d) {
            in_buf[d] = 0;
          }
          in_buf += output_pad_size;

        } else {
          // Zero pad.
          memset(in_buf, 0, sizeof(T) * padded_filter_inner_dim_size);
          in_buf += padded_filter_inner_dim_size;
        }
      }
    }
  }
};

}  // namespace functor
}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CORE_KERNELS_QUANTIZE_DEPTHWISE_CONV_OP_H_

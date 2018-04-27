/*
* Copyright (C) 2018 Ingenic Semiconductor Co.,Ltd
* File       : quantized_depthwise_conv_op.cc
* Authors    : dzhang<dong.zhang@ingenic.com>
* Create Time: 2018-01-18:15:25:37
* Description: This file implements the quantize depthwise conv operation.
*              QuantizeDepthwiseConv2dNative
*/
#define EIGEN_USE_THREADS

#define GEMMLOWP_ALLOW_SLOW_SCALAR_FALLBACK

#include <algorithm>
#include <cmath>
#include <type_traits>

#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/kernels/conv_ops.h"

#include "public/gemmlowp.h"

#include "tensorflow/core/kernels/quantized_depthwise_conv_op.h"
#include "tensorflow/core/kernels/quantization_utils.h"
//#include "tensorflow/core/kernels/neon/depthwiseconv_float.h"
#include "tensorflow/core/kernels/neon/types.h"

#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"
#include "tensorflow/core/util/use_cudnn.h"
#include "tensorflow/core/util/work_sharder.h"

//#include "tensorflow/core/kernels/quantized_32bit_to_8bit.h"


//my head file
//#include "quantized_32bit_to_8bit.h"

#if GOOGLE_CUDA
#include "tensorflow/core/platform/stream_executor.h"
#endif  // GOOGLE_CUDA



static int step = 0;
float min_output_value;
float max_output_value;

using namespace tensorflow::neon;


namespace tensorflow {

using namespace std;

// In depthwise convolution, one input is convolved into depth_multipler
// outputs and the outputs don't need to be reduced again like what regular
// convolution does.
//  However, the way to apply filters to inputs is exactly the same as the
// regular convolution. Please refer to the regular convolution kernels for
// more details.

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;
using namespace gemmlowp;




template<class T1, class T2, class T3>
class QuantizeDepthwiseKernel {

public:

	// generic fallback of FloatDepthwiseConvAccumRow, portable, non-templatized.
	inline void IntDepthwiseConvAccumRowGeneric(int stride, int input_depth,
			int input_width, const T1* input_data, int pad_width,
			int depth_multiplier, int filter_width, const T2* filter_data,
			int out_x_buffer_start, int out_x_buffer_end, int output_depth,
			qint32* acc_buffer, int32 offset_filter, int32 offset_input) {
		gemmlowp::ScopedProfilingLabel label(
				"DepthwiseConvAccumRowGeneric (slow)");
		const T2* filter_base_ptr = filter_data;
		for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
			const int out_x_loop_start = std::max(out_x_buffer_start,
					(pad_width - filter_x + stride - 1) / stride);
			const int out_x_loop_end = std::min(out_x_buffer_end,
					(pad_width + input_width - filter_x + stride - 1) / stride);

			qint32* acc_buffer_ptr = acc_buffer
					+ (out_x_loop_start - out_x_buffer_start) * output_depth;
			const int in_x_origin = (out_x_loop_start * stride) - pad_width
					+ filter_x;
			const T1* input_ptr = input_data + in_x_origin * input_depth;
			const int input_ptr_increment = (stride - 1) * input_depth;
			for (int out_x = out_x_loop_start; out_x < out_x_loop_end;
					out_x++) {
				const T2* filter_ptr = filter_base_ptr;
				for (int ic = 0; ic < input_depth; ++ic) {
					const T1 input_val = *input_ptr++;
					for (int m = 0; m < depth_multiplier; m++) {
						const T2 filter_val = *filter_ptr++;

						*acc_buffer_ptr++ +=
								(static_cast<int32>(filter_val) - offset_filter)
										* (static_cast<int32>(input_val)
												- offset_input);
					}
				}
				input_ptr += input_ptr_increment;
			}
			filter_base_ptr += output_depth;
		}
	}

	// Initializes the accumulator buffer with bias values.
	inline void DepthwiseConvInitAccBuffer(int num_output_pixels,
			int output_depth, const int32* bias_data, qint32* acc_buffer) {
		// TODO(benoitjacob): This might need optimized specializations
		// for small output_depth values, if that ever becomes an important
		// case (like it was for some quantized DepthwiseConv cases).
		for (int i = 0; i < num_output_pixels; i++) {
			memcpy(acc_buffer + i * output_depth, bias_data,
					sizeof(acc_buffer[0]) * output_depth);
		}
	}

	void operator()(const T1* input_data, const Dims<4>& input_dims,
			int32 input_offset, const T2* filter_data,
			const Dims<4>& filter_dims, int32 filter_offset,
			const int32* bias_data, const Dims<4>& bias_dims, int stride_width,
			int stride_height, int pad_width, int pad_height,
			int depth_multiplier, int32 output_offset, int32 output_multiplier,
			int output_shiftt, int32 output_activation_min,
			int32 output_activation_max, T3* output_data,
			const Dims<4>& output_dims) {

//		std::cout << "############begin compute quantize depthwise conv:"
//				<< step << std::endl;

		const int batches = MatchingArraySize(input_dims, 3, output_dims, 3);
		const int output_depth = MatchingArraySize(filter_dims, 0, output_dims,
				0);
		const int input_height = ArraySize(input_dims, 2);
		const int input_width = ArraySize(input_dims, 1);
		const int input_depth = ArraySize(input_dims, 0);
		const int filter_height = ArraySize(filter_dims, 2);
		const int filter_width = ArraySize(filter_dims, 1);
		const int output_height = ArraySize(output_dims, 2);
		const int output_width = ArraySize(output_dims, 1);
		int32 stride = stride_height;


		static const int kAccBufferMaxSize = 1024;
		qint32 acc_buffer[kAccBufferMaxSize];
		DCHECK_GE(kAccBufferMaxSize, output_depth)
				<< "Too small kAccBufferMaxSize for this model!";
		const int kOutputPixelsInAccBuffer = kAccBufferMaxSize / output_depth;
		const int kAccBufferActualSize = kOutputPixelsInAccBuffer
				* output_depth;
		DCHECK_LE(kOutputPixelsInAccBuffer * output_depth,
				kAccBufferActualSize);
		DCHECK_LE(kAccBufferActualSize, kAccBufferMaxSize);
		DCHECK_GE(kOutputPixelsInAccBuffer, 1);

		const int kMaxFixedDepthMultiplier = 8;
		int fixed_depth_multiplier = 0;
		if (depth_multiplier <= kMaxFixedDepthMultiplier) {
			fixed_depth_multiplier = depth_multiplier;
		}
		// kMaxUnrolling is the max number of output values that we aim to handle
		// in one unrolled iteration of the inner loop. For practical performance
		// reasons, it is limited by the number of available registers. We could
		// fine-tune it depending on the architecture, but that's not worth doing
		// since this whole code is not very optimized to begin with. The
		// present value reflects what's realistic on ARM 32bit NEON with 16 128-bit
		// vector registers.
		const int kMaxUnrolling = 8;
		int fixed_input_depth = 0;
		if (fixed_depth_multiplier
				&& input_depth * fixed_depth_multiplier <= kMaxUnrolling) {
			fixed_input_depth = input_depth;
		}
		#define TF_NEON_USE_DEPTHWISECONV_KERNEL(ALLOW_STRIDED, FIXED_INPUT_DEPTH, \
		                                         FIXED_DEPTH_MULTIPLIER)           \
		  if ((stride == 1 || ALLOW_STRIDED) &&                                    \
		      fixed_input_depth == FIXED_INPUT_DEPTH &&                            \
		      fixed_depth_multiplier == FIXED_DEPTH_MULTIPLIER) {                  \
		    row_accum_func =                                                       \
		        FloatDepthwiseConvAccumRow<ALLOW_STRIDED, FIXED_INPUT_DEPTH,       \
		                                   FIXED_DEPTH_MULTIPLIER>;                \
		  }

		#undef TF_NEON_USE_DEPTHWISECONV_KERNEL

		// Now that we have determined row_accum_func, we can start work.
		T3* output_ptr = output_data;
	    const int32 highest = static_cast<int32>(Eigen::NumTraits<T3>::highest());
	    const int32 lowest = static_cast<int32>(Eigen::NumTraits<T3>::lowest());

		for (int b = 0; b < batches; ++b) {
			for (int out_y = 0; out_y < output_height; ++out_y) {
				const int in_y_origin = (out_y * stride) - pad_height;
				const int filter_y_start = std::max(0, -in_y_origin);
				const int filter_y_end = std::min(filter_height,
						input_height - in_y_origin);
				for (int out_x_buffer_start = 0;
						out_x_buffer_start < output_width; out_x_buffer_start +=
								kOutputPixelsInAccBuffer) {

					const int out_x_buffer_end = std::min(output_width,
							out_x_buffer_start + kOutputPixelsInAccBuffer);
					// We call a 'pixel' a group of activation that share all but the
					// 'depth'/'channel' coordinate. num_output_pixels is the number of
					// output pixels that we will accumulate in this loop iteration.
					const int num_output_pixels = out_x_buffer_end
							- out_x_buffer_start;
					// Initialize our local accumulator with the bias values, so we don't
					// have to add them later.
					DepthwiseConvInitAccBuffer(num_output_pixels, output_depth,
							bias_data, acc_buffer);
					// Accumulation loop. Most of the time should be spent in here.
					for (int filter_y = filter_y_start; filter_y < filter_y_end;
							++filter_y) {
						const int in_y = in_y_origin + filter_y;
						IntDepthwiseConvAccumRowGeneric(stride, input_depth,
								input_width,
								input_data + in_y * input_dims.strides[2]
										+ b * input_dims.strides[3], pad_width,
								depth_multiplier, filter_width,
								filter_data + filter_y * filter_dims.strides[2],
								out_x_buffer_start, out_x_buffer_end,
								output_depth, acc_buffer, filter_offset,
								input_offset);
					}
					// Finished accumulating. Now store to destination.
					const int num_output_values = output_depth
							* num_output_pixels;
					int i = 0;

					// Handle leftover values, one by one. This is very slow.
					for (; i < num_output_values; i++) {

						qint32 acc = acc_buffer[i];
						if (neon::FusedActivationFunctionType::kNone
								== FusedActivationFunctionType::kRelu) {
							acc = std::max(0, static_cast<int32>(acc));
						} else if (neon::FusedActivationFunctionType::kNone
								== FusedActivationFunctionType::kRelu6) {
							acc = std::max(0,
									std::min(6, static_cast<int32>(acc)));
						} else if (neon::FusedActivationFunctionType::kNone
								== FusedActivationFunctionType::kRelu1) {
							acc = std::max(-1,
									std::min(1, static_cast<int32>(acc)));
						}
			            acc = std::max(static_cast<int32>(acc), lowest);
			            acc = std::min(static_cast<int32>(acc), highest);


						*output_ptr++ = static_cast<qint32>(acc);
					}



				}
			}
		}

	}



};

//###################import 4 ned

template<typename Device, typename T1, typename T2, typename T3>

struct LaunchQuantizeDepthwiseConv2dNativeOp;
// Computes the vectorized product of 'input_buffer' and 'filter' and stores
// result in 'output' at location specified by 'out_r' and 'out_c'.
//
// EX:
//   in_depth = 3, depth_multiplier = 2, filter [2, 2], register_width = 4
//   Both 'input_buffer' and 'filter' are padded to register-width boundaries.
//
//   input_buffer [rows, cols, in_depth, depth_multiplier]
//     [a0, a0, a1, a1] [a2, a2, 0, 0] [b0, b0, b1, b1] [b2, b2, 0, 0]
//     [e0, e0, e1, e1] [e2, e2, 0, 0] [f0, f0, f1, f1] [f2, f2, 0, 0]
//
//   filter [rows, cols, in_depth, depth_multiplier]
//     [u0, v0, w0, x0] [y0, z0, 0, 0] [u1, v1, w1, x1] [y1, z1, 0, 0]
//     [u2, v2, w2, x2] [y2, z2, 0, 0] [u3, v3, w3, x3] [y3, z3, 0, 0]
//
//   First output register [in_depth, depth_multiplier]
//     [q0, q1, q2, q3] = ([a0, a0, a1, a1] x [u0, v0, w0, x0]) +
//                        ([b0, b0, b1, b1] x [u1, v1, w1, x1]) +
//                        ([e0, e0, e1, e1] x [u2, v2, w2, x2]) +
//                        ([f0, f0, f1, f1] x [u3, v3, w3, x3])
//
// TODO(andydavis) Experiment with processing multiple inputs per input buffer.

template<typename T1, typename T2, typename T3>
struct QuantizeDepthwiseConv2dNativeKernel {
	static void Run(const QuantizeDepthwiseConv2dNativeArgs& args,
			const Tensor& input_tensor, const Tensor& filter_tensor,
			int32 offset_input, int32 offset_filter, const T2* filter,
			const T1* input_buffer, T3* output, TensorFormat data_format) {

		auto input_neon_dims = ToNeonDims(input_tensor.shape());
		auto filter_neon_dims = FilterToNeonDims(filter_tensor.shape());
		auto bias_neon_dims = BiasNeonDims(filter_tensor.shape());
		TensorShape out_shape(
				{ args.batch, args.out_rows, args.out_cols, args.out_depth });
		auto output_dims = ToNeonDims(out_shape);

		int64 bias_size = bias_neon_dims.sizes[0];
		int32* bias_ptr = static_cast<int32*>(port::AlignedMalloc(
				bias_size * sizeof(int32), Allocator::kAllocatorAlignment));
		memset(bias_ptr, 0, bias_size * sizeof(int32));

		//const int32* bias_ptr  = nullptr;

		int32 output_activation_min = 0;
		int32 output_activation_max = 255;

		QuantizeDepthwiseKernel<T1, T2, T3> conv_functor;
		conv_functor(input_buffer, input_neon_dims, offset_input, filter,
				filter_neon_dims, offset_filter, bias_ptr, bias_neon_dims,
				args.stride, args.stride, args.pad_cols, args.pad_rows, 1, 0, 1,
				0, output_activation_min, output_activation_max, output,
				output_dims);

	}



};

// Extern template instantiated in conv_ops.cc.
extern template class LaunchConv2DOp<CPUDevice, float> ;

template<typename T1, typename T2, typename T3, typename Device>
class QuantizeDepthwiseConv2dNativeOp: public OpKernel {
public:
	explicit QuantizeDepthwiseConv2dNativeOp(OpKernelConstruction* context) :
			OpKernel(context) {
		OP_REQUIRES_OK(context, context->GetAttr("strides", &strides_));
		OP_REQUIRES(context, strides_.size() == 4,
				errors::InvalidArgument("Sliding window strides field must "
						"specify 4 dimensions"));
		OP_REQUIRES(context, strides_[1] == strides_[2],
				errors::InvalidArgument(
						"Current implementation only supports equal length "
								"strides in the row and column dimensions."));
		OP_REQUIRES(context, (strides_[0] == 1 && strides_[3] == 1),
				errors::InvalidArgument(
						"Current implementation does not yet support "
								"strides in the batch and depth dimensions."));
		OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
	}

	void Compute(OpKernelContext* context) override {

		step++;

		// Input tensor is of the following dimensions:
		// [ batch, in_rows, in_cols, in_depth ]
		const Tensor& input = context->input(0);

		// Input filter is of the following dimensions:
		// [ filter_rows, filter_cols, in_depth, out_depth]
		const Tensor& filter = context->input(1);

		// For 2D convolution, there should be 4 dimensions.
		OP_REQUIRES(context, input.dims() == 4,
				errors::InvalidArgument("input must be 4-dimensional",
						input.shape().DebugString()));
		OP_REQUIRES(context, filter.dims() == 4,
				errors::InvalidArgument("filter must be 4-dimensional: ",
						filter.shape().DebugString()));

		const float min_input = context->input(2).flat<float>()(0);
		const float max_input = context->input(3).flat<float>()(0);
		const float min_filter = context->input(4).flat<float>()(0);
		const float max_filter = context->input(5).flat<float>()(0);




		const int32 offset_input = FloatToQuantizedUnclamped < T1
				> (0.0f, min_input, max_input);
		const int32 offset_filter = FloatToQuantizedUnclamped < T2
				> (0.0f, min_filter, max_filter);



		// The last dimension for input is in_depth. It must be the same as the
		// filter's in_depth.
		const int64 in_depth = input.dim_size(3);
		OP_REQUIRES(context, in_depth == filter.dim_size(2),
				errors::InvalidArgument(
						"input and filter must have the same depth: ", in_depth,
						" vs ", filter.dim_size(2)));

		// The last dimension for filter is out_depth.
		const int64 out_depth = filter.dim_size(2);

		// The second dimension for input is rows/height.
		// The first dimension for filter is rows/height.
		const int64 input_rows = input.dim_size(1);
		const int64 filter_rows = filter.dim_size(0);

		// The third dimension for input is columns/width.
		// The second dimension for filter is columns/width.
		const int64 input_cols = input.dim_size(2);
		const int64 filter_cols = filter.dim_size(1);

		// The first dimension for input is batch.
		const int64 batch = input.dim_size(0);

		// For now we take the stride from the second dimension only (we
		// assume row = col stride, and do not support striding on the
		// batch or depth dimension).
		const int stride = strides_[1];

		int64 out_rows = 0, out_cols = 0, pad_rows = 0, pad_cols = 0;
		OP_REQUIRES_OK(context,
				GetWindowedOutputSize(input_rows, filter_rows, stride, padding_,
						&out_rows, &pad_rows));
		OP_REQUIRES_OK(context,
				GetWindowedOutputSize(input_cols, filter_cols, stride, padding_,
						&out_cols, &pad_cols));
		CHECK_GT(batch, 0);
		CHECK_GT(out_rows, 0);
		CHECK_GT(out_cols, 0);
		CHECK_GT(out_depth, 0);
		TensorShape out_shape( { batch, out_rows, out_cols, out_depth });

		// Output tensor is of the following dimensions:
		// [ in_batch, out_rows, out_cols, out_depth ]
		Tensor* output;
		OP_REQUIRES_OK(context,
				context->allocate_output(0, out_shape, &output));


		// If there is nothing to compute, return.
		if (out_shape.num_elements() == 0) {
			return;
		}

		QuantizeDepthwiseConv2dNativeArgs args;
		args.batch = batch;
		args.in_rows = input_rows;
		args.in_cols = input_cols;
		args.in_depth = in_depth;
		args.filter_rows = filter_rows;
		args.filter_cols = filter_cols;
		args.depth_multiplier = 1;
		args.stride = stride;
		args.pad_rows = pad_rows;
		args.pad_cols = pad_cols;
		args.out_rows = out_rows;
		args.out_cols = out_cols;
		args.out_depth = out_depth;



		auto input_ptr = input.template flat<T1>().data();
		auto filter_ptr = filter.template flat<T2>().data();
		auto output_ptr = output->template flat<T3>().data();

		QuantizeDepthwiseConv2dNativeKernel<T1, T2, T3>::Run(args, input, filter,
				offset_input, offset_filter, filter_ptr, input_ptr, output_ptr,
				data_format_);


	    float min_output_value;
	    float max_output_value;
	    QuantizationRangeForMultiplication<T1, T2, T3>(
	        min_input, max_input, min_filter, max_filter, &min_output_value,
	        &max_output_value);

		Tensor* output_min = nullptr;
		OP_REQUIRES_OK(context, context->allocate_output(1, { }, &output_min));
		output_min->flat<float>()(0) = min_output_value;

		Tensor* output_max = nullptr;
		OP_REQUIRES_OK(context, context->allocate_output(2, { }, &output_max));
		output_max->flat<float>()(0) = max_output_value;




	}

private:
	std::vector<int32> strides_;
	Padding padding_;
	TensorFormat data_format_;

	int64 stride_;  // in height/width dimension.


	TF_DISALLOW_COPY_AND_ASSIGN (QuantizeDepthwiseConv2dNativeOp);
};


REGISTER_KERNEL_BUILDER(
		Name("QuantizedDepthwiseConv2dNative") .Device(DEVICE_CPU) .TypeConstraint<quint8>("Tinput") .TypeConstraint<quint8>("Tfilter") .TypeConstraint<qint32>("out_type"),
		QuantizeDepthwiseConv2dNativeOp<quint8, quint8, qint32, CPUDevice>);


}
// namespace tensorflow








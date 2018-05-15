/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// See docs in ../ops/array_ops.cc.

#define EIGEN_USE_THREADS

#include <math.h>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/type_traits.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/meta_support.h"
#include "tensorflow/core/kernels/quantization_utils.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

template <class T1, class T2>
class RequantizeEightOp : public OpKernel {
 public:
  explicit RequantizeEightOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input = ctx->input(0);
    const float input_min_float = ctx->input(1).flat<float>()(0);
    const float input_max_float = ctx->input(2).flat<float>()(0);
    const float requested_output_min_float = ctx->input(3).flat<float>()(0);
    const float requested_output_max_float = ctx->input(4).flat<float>()(0);
    QuantizedParams input_quant_params, output_quant_params;
    input_quant_params = ChooseQuantizationParams<int32>(input_min_float, input_max_float);
    output_quant_params = ChooseQuantizationParams<T2>(requested_output_min_float, requested_output_max_float);
    double origin_multiplier = static_cast<double>(input_quant_params.scale) / static_cast<double>(output_quant_params.scale);
    int32_t quantized_multiplier;
    int right_shift;
    QuantizeMultiplierEightBits(origin_multiplier, &quantized_multiplier, &right_shift);

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input.shape(), &output));
    Tensor* output_min = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, TensorShape({}), &output_min));
    Tensor* output_max = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(2, TensorShape({}), &output_max));

    OP_REQUIRES(
        ctx, requested_output_min_float <= 0.0f,
        errors::InvalidArgument("requested_output_min must be <= 0, but got ",
                                requested_output_min_float));
    OP_REQUIRES(
        ctx, requested_output_max_float >= requested_output_min_float,
        errors::InvalidArgument(
            "requested_output_max must be >= requested_output_min, but got ",
            requested_output_max_float, " and ", requested_output_min_float));

    auto input_array = input.flat<T1>().data();
    auto output_array = output->flat<T2>().data();
    for (int i = 0; i < input.NumElements(); i++){
      int32 tmpdata;
      tmpdata = static_cast<int32>(input_array[i]) - input_quant_params.zero_point;
      tmpdata = (tmpdata * quantized_multiplier + (1 << (right_shift -1))) >> right_shift;
      tmpdata += output_quant_params.zero_point;
      tmpdata = std::min(tmpdata, 255);
      tmpdata = std::max(tmpdata, 0);
      output_array[i] = static_cast<quint8>(tmpdata);
    }
    

    output_min->flat<float>().setConstant(requested_output_min_float);
    output_max->flat<float>().setConstant(requested_output_max_float);
  }
};

REGISTER_KERNEL_BUILDER(Name("RequantizeEight")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<qint32>("Tinput")
                            .TypeConstraint<quint8>("out_type"),
                        RequantizeEightOp<qint32, quint8>);

}  // namespace tensorflow

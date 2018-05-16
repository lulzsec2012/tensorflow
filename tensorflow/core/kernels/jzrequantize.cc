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
typedef int32 Dtype;

// calculate M , N
void QuantizeMultiplier(double double_multiplier, int32_t* quantized_multiplier,
                        int* shift) {
  if (double_multiplier == 0.) {
    *quantized_multiplier = 0;
    *shift = 0;
    return;
  }
  const int number_of_bits = sizeof(Dtype) * 8;
  const double q = std::frexp(double_multiplier, shift);
  
  auto q_fixed = static_cast<int64_t>(round(q * (1ll << (number_of_bits -1))));
  DCHECK_LE(q_fixed , (1ll << (number_of_bits -1)));
  if (q_fixed == (1ll << (number_of_bits -1))) {
    q_fixed /= 2;
    ++*shift;
  }
  
  DCHECK_LE(q_fixed, std::numeric_limits<Dtype>::max());
  *quantized_multiplier = static_cast<int32_t>(q_fixed);
}

void QuantizeMultiplierSmallerThanOne(double double_multiplier,
                                      int32_t* quantized_multiplier,
                                      int* right_shift) {
  DCHECK_LT(double_multiplier, 1.);
  DCHECK_GT(double_multiplier, 0.);
  int shift;
  QuantizeMultiplier(double_multiplier, quantized_multiplier, &shift);
  DCHECK_LE(shift, 0);
  *right_shift = -shift;
}

  
int32 Dup(int32 x) {
  return x;
}

int32 BitAnd(int32 a, int32 b) {
  return a & b;
}

// Plain bit-wise NOT
int32 BitNot(int32 a) {
  return ~a;
}



// Integer addition. Not saturating. Overflow is undefined behavior.
int32 Add(int32 a, int32 b) {
  return a + b;
}


// Integer arithmetic right-shift. Not rounding.
// Relying on implementation-defined, but in-practice-consistent,
// C++ compiler behavior.
int32 ShiftRight(int32 a, int offset) {
  return a >> offset;
}

// For each input scalar, the corresponding bits of the result are set if the
// input scalar is non-zero.
int32 MaskIfNonZero(int32 a) {
  static const int32 zero = 0;
  return a ? BitNot(zero) : zero;
}

// For each pair of input scalars, the corresponding bits of the result are
// set if the input scalars a, b satisfy a < b.
int32 MaskIfLessThan(int32 a, int32 b) {
  return MaskIfNonZero(a < b);
}

// For each pair of input scalars, the corresponding bits of the result are
// set if the input scalars a, b satisfy a > b.
int32 MaskIfGreaterThan(int32 a, int32 b) {
  return MaskIfNonZero(a > b);
}

  
  
int32 RoundingDivideByPOT(int32 x, int exponent) {

  assert(exponent >= 0);
  assert(exponent <= 31);
  const int32 mask = Dup((1ll << exponent) - 1);
  const int32 zero = Dup(0);
  const int32 one = Dup(1);
  const int32 remainder = BitAnd(x, mask);
  const int32 threshold =
      Add(ShiftRight(mask, 1), BitAnd(MaskIfLessThan(x, zero), one));
  return Add(ShiftRight(x, exponent),
             BitAnd(MaskIfGreaterThan(remainder, threshold), one));
}


// This function implements the same computation as the ARMv7 NEON VQRDMULH
// instruction.
int32 SaturatingRoundingDoublingHighMul(int32 a,
		int32 b) {

  
  const int number_of_bits = sizeof(Dtype) * 8;
  
  bool overflow = a == b && a == std::numeric_limits<int32>::min();
  int64 a_64(a);
  int64 b_64(b);
  int64 ab_64 = a_64 * b_64;
  
  int32 nudge = ab_64 >= 0 ? (1 << (number_of_bits - 2)) : (1 - (1 << (number_of_bits - 2)));
  int32 ab_x2_high32 =
    static_cast<int32>((ab_64 + nudge) / (1ll << (number_of_bits - 1)));
  return overflow ? std::numeric_limits<int32>::max() : ab_x2_high32;


}

int32 MultiplyByQuantizedMultiplierSmallerThanOne(
    int32 x, int32 quantized_multiplier, int right_shift) {
  

  return RoundingDivideByPOT(
      SaturatingRoundingDoublingHighMul(x, quantized_multiplier), right_shift);
}

  
  
template <class T1, class T2>
class JzRequantizeOp : public OpKernel {
 public:
  explicit JzRequantizeOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

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
    QuantizeMultiplierSmallerThanOne(origin_multiplier, &quantized_multiplier, &right_shift);

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

      tmpdata = MultiplyByQuantizedMultiplierSmallerThanOne(static_cast<int32>(input_array[i]) - input_quant_params.zero_point, quantized_multiplier, right_shift);
      tmpdata += output_quant_params.zero_point;
      tmpdata = std::min(tmpdata, 255);
      tmpdata = std::max(tmpdata, 0);
      output_array[i] = static_cast<quint8>(tmpdata);
    }
    

    output_min->flat<float>().setConstant(requested_output_min_float);
    output_max->flat<float>().setConstant(requested_output_max_float);
  }
};

REGISTER_KERNEL_BUILDER(Name("JzRequantize")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<qint32>("Tinput")
                            .TypeConstraint<quint8>("out_type"),
                        JzRequantizeOp<qint32, quint8>);

}  // namespace tensorflow

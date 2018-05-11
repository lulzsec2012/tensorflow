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

struct QuantizedParams{
  float scale;
  int32 zero_point;
};
// calculate the multiplier
template <typename T>
QuantizedParams ChooseQuantizationParams(float rmin, float rmax) {
  const T qmin = std::numeric_limits<T>::min();
  const T qmax = std::numeric_limits<T>::max();
  const double qmin_double = qmin;
  const double qmax_double = qmax;
  // 0 should always be a representable value. Let's assume that the initial
  // min,max range contains 0.
  DCHECK_LE(rmin, 0.);
  DCHECK_GE(rmax, 0.);
  if (rmin == rmax) {
    // Special case where the min,max range is a point. Should be {0}.
    DCHECK_EQ(rmin, 0.);
    DCHECK_EQ(rmax, 0.);
    QuantizedParams quantization_params;
    quantization_params.zero_point = 0;
    quantization_params.scale = 0.;
    return quantization_params;
  }

  // General case.
  //
  // First determine the scale.
  const double scale = (rmax - rmin) / (qmax_double - qmin_double);

  // Zero-point computation.
  // First the initial floating-point computation. The zero-point can be
  // determined from solving an affine equation for any known pair
  // (real value, corresponding quantized value).
  // We know two such pairs: (rmin, qmin) and (rmax, qmax).
  // The arithmetic error on the zero point computed from either pair
  // will be roughly machine_epsilon * (sum of absolute values of terms)
  // so we want to use the variant that adds the smaller terms.
  const double zero_point_from_min = qmin_double - rmin / scale;
  const double zero_point_from_max = qmax_double - rmax / scale;
  const double zero_point_from_min_error =
      std::abs(qmin_double) + std::abs(rmin / scale);
  const double zero_point_from_max_error =
      std::abs(qmax_double) + std::abs(rmax / scale);

  const double zero_point_double =
      zero_point_from_min_error < zero_point_from_max_error
          ? zero_point_from_min
          : zero_point_from_max;

  // Now we need to nudge the zero point to be an integer
  // (our zero points are integer, and this is motivated by the requirement
  // to be able to represent the real value "0" exactly as a quantized value,
  // which is required in multiple places, for example in Im2col with SAME
  // padding).
  T nudged_zero_point = 0;
  if (zero_point_double < qmin_double) {
    nudged_zero_point = qmin;
  } else if (zero_point_double > qmax_double) {
    nudged_zero_point = qmax;
  } else {
    nudged_zero_point = static_cast<T>(round(zero_point_double));
  }
  // The zero point should always be in the range of quantized value,
  // [qmin, qmax].
  DCHECK_GE(nudged_zero_point, qmin);
  DCHECK_LE(nudged_zero_point, qmax);

  // Finally, store the result nudged quantization params.
  QuantizedParams quantization_params;
  quantization_params.zero_point = nudged_zero_point;
  quantization_params.scale = scale;
  return quantization_params;
}

//double_multiplier = input_scale / output_scale
  

// calculate M , N
void QuantizeMultiplier(double double_multiplier, int32_t* quantized_multiplier,
                        int* shift) {
  if (double_multiplier == 0.) {
    *quantized_multiplier = 0;
    *shift = 0;
    return;
  }
  const double q = std::frexp(double_multiplier, shift);
  auto q_fixed = static_cast<int64_t>(round(q * (1ll << 31)));
  DCHECK_LE(q_fixed , (1ll << 31));
  if (q_fixed == (1ll << 31)) {
    q_fixed /= 2;
    ++*shift;
  }
  DCHECK_LE(q_fixed, std::numeric_limits<int32_t>::max());
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

  
  
  bool overflow = a == b && a == std::numeric_limits<int32>::min();
  int64 a_64(a);
  int64 b_64(b);
  int64 ab_64 = a_64 * b_64;
  int32 nudge = ab_64 >= 0 ? (1 << 30) : (1 - (1 << 30));
  int32 ab_x2_high32 =
      static_cast<int32>((ab_64 + nudge) / (1ll << 31));
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
    //std::FILE* fp = std::fopen("/tmp/cal_mn.txt", "a+");
    //std::fprintf(fp, "M: %d, N: %s\n", quantized_multiplier, right_shift);
    //std::fclose(fp);
    //std::cout <<"step: " << ctx->step_id() <<" quantized_multiplier: " << quantized_multiplier << " right_shift: " << right_shift << std::endl;
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
      //tmpdata = MultiplyByQuantizedMultiplierSmallerThanOne(static_cast<int32>(input_array[i]), quantized_multiplier, right_shift);
      //tmpdata= round(static_cast<float>(static_cast<int32>(input_array[i]) - input_quant_params.zero_point) * origin_multiplier);
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

/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#define EIGEN_USE_THREADS

#include "tensorflow/core/common_runtime/constant_folding.h"
#include "tensorflow/core/common_runtime/threadpool_device.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/subgraph.h"
#include "tensorflow/core/kernels/quantization_utils.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow/tools/graph_transforms/transform_utils.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace tensorflow {
namespace graph_transforms {
typedef int32 Dtype;
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
  const int number_of_bits = sizeof(Dtype) * 8;
  const double q = std::frexp(double_multiplier, shift);
  auto q_fixed = static_cast<int64_t>(round(q * (1ll << (number_of_bits - 1))));
  DCHECK_LE(q_fixed , (1ll << (number_of_bits - 1)));
  if (q_fixed == (1ll << (number_of_bits - 1))) {
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
  
Status CalculateMn(const GraphDef& input_graph_def,
                       const TransformFuncContext& context,
                       GraphDef* output_graph_def) {
  TF_RETURN_IF_ERROR(ReplaceMatchingOpTypes(
					    input_graph_def,
					  {"Requantize",
					      {
						{"*",{{"*"},{"*"},{"*"},{"*"},{"*"}, {"*"}}},
						{"*"},
						{"*"},
						{"*"},
						{"*"},
					      }
					  },

        [](const NodeMatch& match,
                     const std::set<string>& input_nodes,
                     const std::set<string>& output_nodes,
                     std::vector<NodeDef>* new_nodes) {
	  NodeDef input_min_node, input_max_node;
	  NodeDef output_min_node, output_max_node;
	CopyOriginalMatch(match, new_nodes);
	input_min_node = match.inputs[0].inputs[4].node;
	input_max_node = match.inputs[0].inputs[5].node;
	output_min_node = match.inputs[3].node;
	output_max_node = match.inputs[4].node;
        Tensor input_min_tensor, input_max_tensor;
	Tensor output_min_tensor, output_max_tensor;

        if (!input_min_tensor.FromProto(input_min_node.attr().at("value").tensor())) {
          return errors::InvalidArgument("Decoding Tensor failed for node",
                                         input_min_node.name());
        }
        if (!input_max_tensor.FromProto(input_max_node.attr().at("value").tensor())) {
          return errors::InvalidArgument("Decoding Tensor failed for node",
                                         input_max_node.name());
        }

	if (!output_min_tensor.FromProto(output_min_node.attr().at("value").tensor())) {
          return errors::InvalidArgument("Decoding Tensor failed for node",
                                         output_min_node.name());
        }
        if (!output_max_tensor.FromProto(output_max_node.attr().at("value").tensor())) {
          return errors::InvalidArgument("Decoding Tensor failed for node",
                                         output_max_node.name());
        }

        const float* input_min_value = input_min_tensor.flat<float>().data();
        const float* input_max_value = input_max_tensor.flat<float>().data();
        const float* output_min_value = output_min_tensor.flat<float>().data();
        const float* output_max_value = output_max_tensor.flat<float>().data();

	QuantizedParams input_quant_params, output_quant_params;
	input_quant_params = ChooseQuantizationParams<int32>(input_min_value[0], input_max_value[0]);
	output_quant_params = ChooseQuantizationParams<quint8>(output_min_value[0], output_max_value[0]);	
	double origin_multiplier = static_cast<double>(input_quant_params.scale) / static_cast<double>(output_quant_params.scale);
	int32_t quantized_multiplier;
	int right_shift;
	QuantizeMultiplierSmallerThanOne(origin_multiplier, &quantized_multiplier, &right_shift);
	std::FILE* fp = std::fopen("/tmp/cal_mn.txt", "a+");
	std::fprintf(fp, "M: %d, N: %d\n", quantized_multiplier, right_shift);
	std::fclose(fp);

	return Status::OK();
	}, {}, output_graph_def));

  return Status::OK();

}
  

REGISTER_GRAPH_TRANSFORM("calculate_mn", CalculateMn);

}  // namespace graph_transforms
}  // namespace tensorflow

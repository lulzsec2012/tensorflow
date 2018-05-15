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
#define TF_METHOD 0
#if TF_METHOD
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
#endif
  
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
#if TF_METHOD
	QuantizeMultiplierSmallerThanOne(origin_multiplier, &quantized_multiplier, &right_shift);	
#else
	QuantizeMultiplierEightBits(origin_multiplier, &quantized_multiplier, &right_shift);
#endif	
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

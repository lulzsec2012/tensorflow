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

namespace tensorflow {
namespace graph_transforms {

// Converts any large float constants into eight-bit equivalents, with a
// Dequantize op so that subsequent nodes can still access the results in a
// float form.
Status QuantizeWeights(const GraphDef& input_graph_def,
                       const TransformFuncContext& context,
                       GraphDef* output_graph_def) {
  int32 minimum_size;
  TF_RETURN_IF_ERROR(
      context.GetOneInt32Parameter("minimum_size", 2, &minimum_size));
  //add ingenic
  std::set<string> ops_to_ignore;
  if (context.params.count("ignore_op") > 0) {
    for (const string& name : context.params.at("ignore_op")) {
      ops_to_ignore.insert(name);
    }
  }
  //add ingenic
  std::vector<string> quant_bias_names;
  GraphDef tmp_output_graph_def;
  GraphDef quant_placeholder_graph_def;
  TF_RETURN_IF_ERROR(ReplaceMatchingOpTypes(
					    input_graph_def,
					    {"Add|BiasAdd",
						{
						  {"Conv2D|DepthwiseConv2dNative",
						      {
							{"Placeholder"},
							{"*"},							  
							
						      },
						  },
						  {"*"},
						},
					    },

        [minimum_size, &quant_bias_names, &ops_to_ignore](const NodeMatch& match,
                     const std::set<string>& input_nodes,
                     const std::set<string>& output_nodes,
                     std::vector<NodeDef>* new_nodes) {
	const NodeDef& origin_node = match.node;
        const NodeDef& conv_const_node = match.inputs[0].node;
	const NodeDef& old_const_node = match.inputs[1].node;
	const NodeDef& fake_quant_weight_const_node = match.inputs[0].inputs[1].node;
	const NodeDef& fake_quant_acvation_const_node = match.inputs[0].inputs[0].node;
	new_nodes->push_back(conv_const_node);
	new_nodes->push_back(match.node);
	new_nodes->push_back(fake_quant_weight_const_node);
	new_nodes->push_back(fake_quant_acvation_const_node);
        if (!old_const_node.attr().count("dtype")) {
          return errors::InvalidArgument("No 'dtype' attribute for Const node ",
                                         old_const_node.name());
        }
        if (!old_const_node.attr().count("value")) {
          return errors::InvalidArgument("No 'value' attribute for Const node ",
                                         old_const_node.name());
        }
        const DataType old_dtype = old_const_node.attr().at("dtype").type();
        Tensor old_tensor;
        if (!old_tensor.FromProto(old_const_node.attr().at("value").tensor())) {
          return errors::InvalidArgument("Decoding Tensor failed for node",
                                         old_const_node.name());
        }
	float* old_tensor_value = old_tensor.flat<float>().data(); 
        const size_t num_elements = old_tensor.NumElements();
        // If this isn't a float constant, or it's too small, then reuse the
        // same node with no changes.
        if ((old_dtype != DT_FLOAT) || (num_elements < minimum_size)) {
          new_nodes->push_back(old_const_node);
          return Status::OK();
        }
	const NodeDef& weight_node = match.inputs[0].inputs[1].node;
	Tensor weight_tensor;
	if (!weight_tensor.FromProto(weight_node.attr().at("value").tensor())) {
	  return errors::InvalidArgument("Decoding Tensor faild for node",
					weight_node.name());
	}
	const float* weight_values = weight_tensor.flat<float>().data();
	const size_t weight_num_elements = weight_tensor.NumElements();
	float weight_min = std::numeric_limits<float>::max();
	float weight_max = std::numeric_limits<float>::min();
        for (int i = 0; i < weight_num_elements; ++i) {
          const float value = weight_values[i];
          weight_min = std::min(weight_min, value);
          weight_max = std::max(weight_max, value);
        }
	weight_min = std::min(weight_min, 0.0f);
	weight_max = std::max(0.0f, weight_max);
	if (weight_min == weight_max){
	  if (std::abs(weight_min) < 0.000001f){
	    weight_max = weight_min + 1.0f;
	  }else if (weight_min > 0.0f){
	    weight_max = 2.0f * weight_min;
	  }else {
	    weight_max = weight_min / 2.0f;
	  }
	}
	float weight_scale = 255 / (weight_max - weight_min);
	//
	float last_activation_scale = 1.0;//255 / (last_activation_max_value[0] - last_activation_min_value[0]);	
	//
        Tensor quantized_tensor(DT_QINT32, old_tensor.shape());
	qint32* quantized_tensor_value = quantized_tensor.flat<qint32>().data();
	const size_t quantized_tensor_elements = quantized_tensor.NumElements();

	for (int i = 0; i < quantized_tensor_elements; i++){

	  quantized_tensor_value[i] = static_cast<int32_t>(round(old_tensor_value[i] * weight_scale * last_activation_scale));	  
	 
	}
	float min = static_cast<float>((1 - (1 << 31))  / weight_scale / last_activation_scale);
	float max = static_cast<float>((1 << 31) / weight_scale / last_activation_scale);
        NodeDef quantized_const_node;
        quantized_const_node.set_op("Const");
        quantized_const_node.set_name(old_const_node.name() +
                                      "_quantized_const");
        SetNodeAttr("dtype", DT_QINT32, &quantized_const_node);
        SetNodeTensorAttr<float>("value", quantized_tensor,
                                 &quantized_const_node);
        new_nodes->push_back(quantized_const_node);
	quant_bias_names.push_back(quantized_const_node.name());

        NodeDef min_node;
        min_node.set_op("Const");
        min_node.set_name(old_const_node.name() + "_quantized_min");
        SetNodeAttr("dtype", DT_FLOAT, &min_node);
        Tensor min_tensor(DT_FLOAT, {});
        min_tensor.scalar<float>()() = min;
        SetNodeTensorAttr<float>("value", min_tensor, &min_node);
        new_nodes->push_back(min_node);
	quant_bias_names.push_back(min_node.name());
        NodeDef max_node;
        max_node.set_op("Const");
        max_node.set_name(old_const_node.name() + "_quantized_max");
        SetNodeAttr("dtype", DT_FLOAT, &max_node);
        Tensor max_tensor(DT_FLOAT, {});
        max_tensor.scalar<float>()() = max;
        SetNodeTensorAttr<float>("value", max_tensor, &max_node);
        new_nodes->push_back(max_node);
	quant_bias_names.push_back(max_node.name());

        NodeDef dequantize_node;
        dequantize_node.set_op("Dequantize");
        dequantize_node.set_name(old_const_node.name());
        SetNodeAttr("T", DT_QINT32, &dequantize_node);
        SetNodeAttr("mode", "SCALED", &dequantize_node);
        AddNodeInput(quantized_const_node.name(), &dequantize_node);
        AddNodeInput(min_node.name(), &dequantize_node);
	AddNodeInput(max_node.name(), &dequantize_node);
        new_nodes->push_back(dequantize_node);

        return Status::OK();
      },
      {}, &quant_placeholder_graph_def));






  
  TF_RETURN_IF_ERROR(ReplaceMatchingOpTypes(
					    quant_placeholder_graph_def,
					    {"Add|BiasAdd",
						{
						  {"Conv2D|DepthwiseConv2dNative",
						      {
							{"*",
							    {
							      {"*"},
							      {"*"},
							      {"*"},
							    },
							},
							{"*"},							  
							
						      },
						  },
						  {"*"},
						},
					    },

	[minimum_size, &quant_bias_names](const NodeMatch& match,
                     const std::set<string>& input_nodes,
                     const std::set<string>& output_nodes,
                     std::vector<NodeDef>* new_nodes) {
	
	const NodeDef& origin_node = match.node;
        const NodeDef& conv_const_node = match.inputs[0].node;
	const NodeDef& old_const_node = match.inputs[1].node;
	const NodeDef& fake_quant_weight_const_node = match.inputs[0].inputs[1].node;
	const NodeDef& fake_quant_acvation_const_node = match.inputs[0].inputs[0].node;
	new_nodes->push_back(conv_const_node);
	new_nodes->push_back(match.node);
	new_nodes->push_back(fake_quant_weight_const_node);
	new_nodes->push_back(fake_quant_acvation_const_node);
	new_nodes->push_back(match.inputs[0].inputs[0].inputs[0].node);
	new_nodes->push_back(match.inputs[0].inputs[0].inputs[1].node);
	new_nodes->push_back(match.inputs[0].inputs[0].inputs[2].node); 
        if (!old_const_node.attr().count("dtype")) {
          return errors::InvalidArgument("No 'dtype' attribute for Const node ",
                                         old_const_node.name());
        }
        if (!old_const_node.attr().count("value")) {
          return errors::InvalidArgument("No 'value' attribute for Const node ",
                                         old_const_node.name());
        }
        const DataType old_dtype = old_const_node.attr().at("dtype").type();
        Tensor old_tensor;
        if (!old_tensor.FromProto(old_const_node.attr().at("value").tensor())) {
          return errors::InvalidArgument("Decoding Tensor failed for node",
                                         old_const_node.name());
        }
	float* old_tensor_value = old_tensor.flat<float>().data(); 
        const size_t num_elements = old_tensor.NumElements();
        // If this isn't a float constant, or it's too small, then reuse the
        // same node with no changes.
        if ((old_dtype != DT_FLOAT) || (num_elements < minimum_size)) {
          new_nodes->push_back(old_const_node);
          return Status::OK();
        }
	const NodeDef& weight_node = match.inputs[0].inputs[1].node;
	Tensor weight_tensor;
	if (!weight_tensor.FromProto(weight_node.attr().at("value").tensor())) {
	  return errors::InvalidArgument("Decoding Tensor faild for node",
					weight_node.name());
	}
	const float* weight_values = weight_tensor.flat<float>().data();
	const size_t weight_num_elements = weight_tensor.NumElements();
	float weight_min = std::numeric_limits<float>::max();
	float weight_max = std::numeric_limits<float>::min();
        for (int i = 0; i < weight_num_elements; ++i) {
          const float value = weight_values[i];
          weight_min = std::min(weight_min, value);
          weight_max = std::max(weight_max, value);
        }
	weight_min = std::min(weight_min, 0.0f);
	weight_max = std::max(0.0f, weight_max);
	if (weight_min == weight_max){
	  if (std::abs(weight_min) < 0.000001f){
	    weight_max = weight_min + 1.0f;
	  }else if (weight_min > 0.0f){
	    weight_max = 2.0f * weight_min;
	  }else {
	    weight_max = weight_min / 2.0f;
	  }
	}
	float weight_scale = 255 / (weight_max - weight_min);
	
	const NodeDef& last_activation_min_node = match.inputs[0].inputs[0].inputs[1].node;
	const NodeDef& last_activation_max_node = match.inputs[0].inputs[0].inputs[2].node;
	Tensor last_activation_min, last_activation_max;
	if(!last_activation_min.FromProto(last_activation_min_node.attr().at("value").tensor())){
	  return errors::InvalidArgument("Decoding Tensor faild for node",
					last_activation_min_node.name());
	}
	if(!last_activation_max.FromProto(last_activation_max_node.attr().at("value").tensor())){
	  return errors::InvalidArgument("Decoding Tensor faild for node",
					last_activation_max_node.name());
	}
	const float* last_activation_min_value = last_activation_min.flat<float>().data();
	const float* last_activation_max_value = last_activation_max.flat<float>().data();
	float last_activation_scale = 255 / (last_activation_max_value[0] - last_activation_min_value[0]);	

        Tensor quantized_tensor(DT_QINT32, old_tensor.shape());
	qint32* quantized_tensor_value = quantized_tensor.flat<qint32>().data();
	const size_t quantized_tensor_elements = quantized_tensor.NumElements();

	for (int i = 0; i < quantized_tensor_elements; i++){

	  quantized_tensor_value[i] = static_cast<int32_t>(round(old_tensor_value[i] * weight_scale * last_activation_scale));	  
	 
	}
	float min = static_cast<float>(-(1 << 31) / weight_scale / last_activation_scale);
	float max = static_cast<float>((1 << 31) / weight_scale / last_activation_scale);
        NodeDef quantized_const_node;
        quantized_const_node.set_op("Const");
        quantized_const_node.set_name(old_const_node.name() +
                                      "_quantized_const");
        SetNodeAttr("dtype", DT_QINT32, &quantized_const_node);
        SetNodeTensorAttr<float>("value", quantized_tensor,
                                 &quantized_const_node);
        new_nodes->push_back(quantized_const_node);
	quant_bias_names.push_back(quantized_const_node.name());

        NodeDef min_node;
        min_node.set_op("Const");
        min_node.set_name(old_const_node.name() + "_quantized_min");
        SetNodeAttr("dtype", DT_FLOAT, &min_node);
        Tensor min_tensor(DT_FLOAT, {});
        min_tensor.scalar<float>()() = min;
        SetNodeTensorAttr<float>("value", min_tensor, &min_node);
        new_nodes->push_back(min_node);
	quant_bias_names.push_back(min_node.name());
        NodeDef max_node;
        max_node.set_op("Const");
        max_node.set_name(old_const_node.name() + "_quantized_max");
        SetNodeAttr("dtype", DT_FLOAT, &max_node);
        Tensor max_tensor(DT_FLOAT, {});
        max_tensor.scalar<float>()() = max;
        SetNodeTensorAttr<float>("value", max_tensor, &max_node);
        new_nodes->push_back(max_node);
	quant_bias_names.push_back(max_node.name());

        NodeDef dequantize_node;
        dequantize_node.set_op("Dequantize");
        dequantize_node.set_name(old_const_node.name());
        SetNodeAttr("T", DT_QINT32, &dequantize_node);
        SetNodeAttr("mode", "SCALED", &dequantize_node);
        AddNodeInput(quantized_const_node.name(), &dequantize_node);
        AddNodeInput(min_node.name(), &dequantize_node);
	AddNodeInput(max_node.name(), &dequantize_node);
        new_nodes->push_back(dequantize_node);

        return Status::OK();
      },
      {}, &tmp_output_graph_def));
  //add end







  TF_RETURN_IF_ERROR(ReplaceMatchingOpTypes(
      tmp_output_graph_def, {"Const"},
      [minimum_size, &quant_bias_names](const NodeMatch& match,
                     const std::set<string>& input_nodes,
                     const std::set<string>& output_nodes,
                     std::vector<NodeDef>* new_nodes) {
        const NodeDef& old_const_node = match.node;
	bool bias_flag = false;

	for (int i = 0; i < quant_bias_names.size(); i++){
	  if (match.node.name() == quant_bias_names[i]){
	    bias_flag = true;
	    new_nodes->push_back(match.node);
	    break;
	  }
	}

	if (bias_flag == false){
	  
        if (!old_const_node.attr().count("dtype")) {
          return errors::InvalidArgument("No 'dtype' attribute for Const node ",
                                         old_const_node.name());
        }
        if (!old_const_node.attr().count("value")) {
          return errors::InvalidArgument("No 'value' attribute for Const node ",
                                         old_const_node.name());
        }
        const DataType old_dtype = old_const_node.attr().at("dtype").type();
        Tensor old_tensor;
        if (!old_tensor.FromProto(old_const_node.attr().at("value").tensor())) {
          return errors::InvalidArgument("Decoding Tensor failed for node",
                                         old_const_node.name());
        }
        const size_t num_elements = old_tensor.NumElements();
        // If this isn't a float constant, or it's too small, then reuse the
        // same node with no changes.
        if ((old_dtype != DT_FLOAT) || (num_elements < minimum_size)) {
          new_nodes->push_back(old_const_node);
          return Status::OK();
        }
	//
	//std::cout << "batch: " << old_tensor.dim_size(0) << std::endl;
	//std::cout << "batch: " << old_tensor.dim_size(0) << std::endl;
	//std::cout << "height: " << old_tensor.shape(1) << std::endl;
	//std::cout << "width: " << old_tensor.shape(2) << std::endl;
	//std::cout << "channel: " << old_tensor.shape(3) << std::endl;	
	//

        const float* old_values = old_tensor.flat<float>().data();
        float min = std::numeric_limits<float>::max();
        float max = std::numeric_limits<float>::min();
        for (int i = 0; i < num_elements; ++i) {
          const float value = old_values[i];
          min = std::min(min, value);
          max = std::max(max, value);
        }
        // Make sure the quantization range includes 0.0f. Not all quantized
        // Ops behave properly if 0.0f is not in the range.
        min = std::min(min, 0.0f);
        max = std::max(0.0f, max);
        // min_value == max_value is a tricky case. It can occur for general
        // tensors, and of course for scalars. The quantized ops cannot deal
        // with this case, so we set max_value to something else.
        // It's a tricky question what is the numerically best solution to
        // deal with this degeneracy.
        // TODO(petewarden): Better use a tolerance than a hard comparison?
        if (min == max) {
          if (std::abs(min) < 0.000001f) {
            max = min + 1.0f;
          } else if (min > 0) {
            max = 2.0f * min;
          } else {
            max = min / 2.0f;
          }
        }
        Tensor quantized_tensor(DT_QUINT8, old_tensor.shape());
        FloatTensorToQuantizedInPlace<quint8>(old_tensor, min, max,
                                              &quantized_tensor);

        NodeDef quantized_const_node;
        quantized_const_node.set_op("Const");
        quantized_const_node.set_name(old_const_node.name() +
                                      "_quantized_const");
        SetNodeAttr("dtype", DT_QUINT8, &quantized_const_node);
        SetNodeTensorAttr<float>("value", quantized_tensor,
                                 &quantized_const_node);
        new_nodes->push_back(quantized_const_node);

        NodeDef min_node;
        min_node.set_op("Const");
        min_node.set_name(old_const_node.name() + "_quantized_min");
        SetNodeAttr("dtype", DT_FLOAT, &min_node);
        Tensor min_tensor(DT_FLOAT, {});
        min_tensor.scalar<float>()() = min;
        SetNodeTensorAttr<float>("value", min_tensor, &min_node);
        new_nodes->push_back(min_node);

        NodeDef max_node;
        max_node.set_op("Const");
        max_node.set_name(old_const_node.name() + "_quantized_max");
        SetNodeAttr("dtype", DT_FLOAT, &max_node);
        Tensor max_tensor(DT_FLOAT, {});
        max_tensor.scalar<float>()() = max;
        SetNodeTensorAttr<float>("value", max_tensor, &max_node);
        new_nodes->push_back(max_node);

        NodeDef dequantize_node;
        dequantize_node.set_op("Dequantize");
        dequantize_node.set_name(old_const_node.name());
        SetNodeAttr("T", DT_QUINT8, &dequantize_node);
        SetNodeAttr("mode", "MIN_FIRST", &dequantize_node);
        AddNodeInput(quantized_const_node.name(), &dequantize_node);
        AddNodeInput(min_node.name(), &dequantize_node);
        AddNodeInput(max_node.name(), &dequantize_node);
        new_nodes->push_back(dequantize_node);

	}
        return Status::OK();
      },
      {}, output_graph_def));

  return Status::OK();
}

REGISTER_GRAPH_TRANSFORM("quantize_weights", QuantizeWeights);

}  // namespace graph_transforms
}  // namespace tensorflow

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

template <class T>
float FloatForOneQuantizedLevel(float range_min, float range_max) {
  const int64 highest = static_cast<int64>(Eigen::NumTraits<T>::highest());
  const int64 lowest = static_cast<int64>(Eigen::NumTraits<T>::lowest());
  const float float_for_one_quantized_level =
      (range_max - range_min) / (highest - lowest);
  return float_for_one_quantized_level;
}

template <class T1, class T2, class T3>
void QuantizationRangeForMultiplication(float min_a, float max_a, float min_b,
                                        float max_b, float* min_c,
                                        float* max_c) {
  const float a_float_for_one_quant_level =
      FloatForOneQuantizedLevel<T1>(min_a, max_a);
  const float b_float_for_one_quant_level =
      FloatForOneQuantizedLevel<T2>(min_b, max_b);

  const int64 c_highest = static_cast<int64>(Eigen::NumTraits<T3>::highest());
  const int64 c_lowest = static_cast<int64>(Eigen::NumTraits<T3>::lowest());
  const float c_float_for_one_quant_level =
      a_float_for_one_quant_level * b_float_for_one_quant_level;

  *min_c = c_float_for_one_quant_level * c_lowest;
  *max_c = c_float_for_one_quant_level * c_highest;
}

void CopyOriginalMatchExceptNode(const NodeMatch& match,std::vector<NodeDef>* new_nodes, string bias_node_name) {
  std::vector<NodeDef> old_nodes;
  MatchedNodesAsArray(match, &old_nodes);
  for (const NodeDef& old_node : old_nodes) {
    if(old_node.name() != bias_node_name){
      new_nodes->push_back(old_node);	
    }
  }
}

Status getConstNodeMinMax(const NodeDef &old_const_node,float &rst_min,float &rst_max){
        Tensor old_tensor;
        if (!old_tensor.FromProto(old_const_node.attr().at("value").tensor())) {
          return errors::InvalidArgument("Decoding Tensor failed for node",
                                         old_const_node.name());
        }
        const size_t num_elements = old_tensor.NumElements();	
        const float* old_values = old_tensor.flat<float>().data();
        float min = std::numeric_limits<float>::max();
        float max = std::numeric_limits<float>::min();
        for (int i = 0; i < num_elements; ++i) {
          const float value = old_values[i];
          min = std::min(min, value);
          max = std::max(max, value);
        }

        min = std::min(min, 0.0f);
        max = std::max(0.0f, max);

        if (min == max) {
          if (std::abs(min) < 0.000001f) {
            max = min + 1.0f;
          } else if (min > 0) {
            max = 2.0f * min;
          } else {
            max = min / 2.0f;
          }
        }
	rst_min = min;
	rst_max = max;
	return Status::OK();	
  }

// Converts any large float constants into eight-bit equivalents, with a
// Dequantize op so that subsequent nodes can still access the results in a
// float form.
Status QuantizeWeights(const GraphDef& input_graph_def,
                       const TransformFuncContext& context,
                       GraphDef* output_graph_def) {
  int32 minimum_size;
  TF_RETURN_IF_ERROR(
      context.GetOneInt32Parameter("minimum_size", 1024, &minimum_size));
  bool Ti_quant;
  TF_RETURN_IF_ERROR(
      context.GetOneBoolParameter("Ti_quant", false, &Ti_quant));
  int match_idx = 0;
  const int max_depth = 3;
  const int pattern_num = 2;
  std::vector<string> quant_bias_names;
  const std::function<Status(const NodeMatch&, const std::set<string>&,
                               const std::set<string>&, std::vector<NodeDef>*)>&
    node_generator = [pattern_num, max_depth, &match_idx,minimum_size, Ti_quant,&quant_bias_names](const NodeMatch& match,
                     const std::set<string>& input_nodes,
                     const std::set<string>& output_nodes,
                     std::vector<NodeDef>* new_nodes) {
        NodeDef old_const_node;
	if(match_idx < pattern_num*max_depth){
	  // std::cout<<"match.node.name():"<<match.node.name()<<"::"<<"match.node.op():"<<match.node.op()<<" match_idx="<<match_idx<<std::endl;
	  //std::cout<<"match.DebugString()="<<std::endl<<match.DebugString()<<std::endl<<std::endl;	
	  old_const_node = match.inputs[1].node;
	  CopyOriginalMatchExceptNode(match, new_nodes, old_const_node.name());
	}else{
	  old_const_node = match.node;
	}
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
	if(match_idx < pattern_num*max_depth){
	  for (int i = 0; i < quant_bias_names.size(); i++){
	    if (match.node.name() == quant_bias_names[i]){
	      new_nodes->push_back(match.node);
	      return Status::OK();
	    }
	  }
	}
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
	if(Ti_quant){
	  max = std::max(std::abs(min),std::abs(max));
	  min = -max;
	}
#if 1
	Tensor *pquantized_tensor;
	Tensor quantized_tensor_8(DT_QUINT8, old_tensor.shape());
	Tensor quantized_tensor_32(DT_QINT32, old_tensor.shape());
	if(match_idx < pattern_num*max_depth){
	  //std::cout<<"match_idx="<<match_idx<<std::endl;
	  NodeMatch current_match = static_cast<NodeMatch>(match.inputs[0]);
	  for(int i=0;i<=match_idx%max_depth;i++){
	    current_match = static_cast<NodeMatch>(current_match.inputs[0]);
	  }
	  //x_scale
	  float x_scale = 1.0;
	  float x_min, x_max;
	  //std::cout<<"current_match.inputs[0].node.op():"<<current_match.node.op()<<std::endl;
	  if("FakeQuantWithMinMaxVars" == current_match.node.op()){
	    const NodeDef& last_activation_min_node = current_match.inputs[1].node;
	    const NodeDef& last_activation_max_node = current_match.inputs[2].node;
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
	    x_min = last_activation_min_value[0];
	    x_max = last_activation_max_value[0];
	    x_scale = 255 / (x_max - x_min);
	  }else if("Placeholder" == current_match.node.op()){
	    x_min = -1.0;
	    x_max = 1.0;
	    x_scale = 255 / (x_max - x_min);
	  }else{
	    return errors::InvalidArgument("current_match.node.op() should be 'FakeQuantWithMinMaxVars' or 'Placeholder'",
					   current_match.node.name());
	  }
	  //w_scale
	  const NodeDef& weight_node = match.inputs[0].inputs[1].node;
	  float w_min,w_max;
	  getConstNodeMinMax(weight_node, w_min, w_max);
	  if(Ti_quant){
	    w_max = std::max(std::abs(w_min),std::abs(w_max));
	    w_min = -w_max;
	  }
	  float w_scale = 255 / (w_max - w_min);
	  qint32* quantized_tensor_value = quantized_tensor_32.flat<qint32>().data();
	  const size_t quantized_tensor_elements = quantized_tensor_32.NumElements();
	  float* old_tensor_value = old_tensor.flat<float>().data();
	  QuantizationRangeForMultiplication<quint8,quint8,qint32>(w_min, w_max, x_min, x_max, &min,&max);
	  for (int i = 0; i < quantized_tensor_elements; i++){
	    quantized_tensor_value[i] = static_cast<int32_t>(round(old_tensor_value[i] * w_scale * x_scale));	  
	  }
	  pquantized_tensor = &quantized_tensor_32;
	}else{
	  FloatTensorToQuantizedInPlace<quint8>(old_tensor, min, max,
						&quantized_tensor_8);
	  pquantized_tensor = &quantized_tensor_8;
	}
#else
	Tensor quantized_tensor(DT_QUINT8, old_tensor.shape());
	FloatTensorToQuantizedInPlace<quint8>(old_tensor, min, max,
	 					&quantized_tensor);
#endif
        NodeDef quantized_const_node;
        quantized_const_node.set_op("Const");
        quantized_const_node.set_name(old_const_node.name() +
                                      "_quantized_const");
	if(match_idx < pattern_num*max_depth){
	  SetNodeAttr("dtype", DT_QINT32, &quantized_const_node);
	}else{
	  SetNodeAttr("dtype", DT_QUINT8, &quantized_const_node);
	}
        SetNodeTensorAttr<float>("value", *pquantized_tensor,
                                 &quantized_const_node);
        new_nodes->push_back(quantized_const_node);
	if(match_idx < pattern_num*max_depth)quant_bias_names.push_back(quantized_const_node.name());

        NodeDef min_node;
        min_node.set_op("Const");
        min_node.set_name(old_const_node.name() + "_quantized_min");
        SetNodeAttr("dtype", DT_FLOAT, &min_node);
        Tensor min_tensor(DT_FLOAT, {});
        min_tensor.scalar<float>()() = min;
        SetNodeTensorAttr<float>("value", min_tensor, &min_node);
        new_nodes->push_back(min_node);
	if(match_idx < pattern_num*max_depth)quant_bias_names.push_back(min_node.name());

        NodeDef max_node;
        max_node.set_op("Const");
        max_node.set_name(old_const_node.name() + "_quantized_max");
        SetNodeAttr("dtype", DT_FLOAT, &max_node);
        Tensor max_tensor(DT_FLOAT, {});
        max_tensor.scalar<float>()() = max;
        SetNodeTensorAttr<float>("value", max_tensor, &max_node);
        new_nodes->push_back(max_node);
	if(match_idx < pattern_num*max_depth)quant_bias_names.push_back(max_node.name());

        NodeDef dequantize_node;
        dequantize_node.set_op("Dequantize");
        dequantize_node.set_name(old_const_node.name());
	if(match_idx < pattern_num*max_depth){
	  SetNodeAttr("T", DT_QINT32, &dequantize_node);
	  SetNodeAttr("mode", "SCALED", &dequantize_node);
	}else{
	  SetNodeAttr("T", DT_QUINT8, &dequantize_node);
	  SetNodeAttr("mode", "MIN_FIRST", &dequantize_node);
	}
        AddNodeInput(quantized_const_node.name(), &dequantize_node);
        AddNodeInput(min_node.name(), &dequantize_node);
        AddNodeInput(max_node.name(), &dequantize_node);
        new_nodes->push_back(dequantize_node);

        return Status::OK();
  };
  OpTypePattern pattern_hold = {"Placeholderxx"};
  OpTypePattern pattern_fake = {"FakeQuantWithMinMaxVars", {{"*"}, {"Const"}, {"Const"}}};    
  std::vector<OpTypePattern> pattern_vec = {pattern_hold,pattern_fake};
  //
  OpTypePattern pattern_avg = {"AvgPool",{pattern_fake}};
  OpTypePattern pattern_reshape = {"Reshape",{pattern_avg,{"*"}}};
  //
  assert(pattern_num*max_depth<20);
  GraphDef graph_def_tmp[20];
  graph_def_tmp[0] = input_graph_def;
  for(std::vector<OpTypePattern>::iterator it=pattern_vec.begin();it != pattern_vec.end();it++){
    for (int depth = 0; depth < max_depth; depth++) {
      OpTypePattern pattern = *it;
      for (int i = 0; i < depth; i++) {
	pattern = {"*", {pattern}};
      }
      //
      if(match_idx==5){
	pattern = pattern_reshape;
      }
      //
      OpTypePattern pattern_conv = {"Conv2D|DepthwiseConv2dNative|MatMul",{pattern,{"*"}}};
      OpTypePattern pattern_bias = {"Add|BiasAdd",{pattern_conv,{"*"}}};
      TF_RETURN_IF_ERROR(ReplaceMatchingOpTypes(graph_def_tmp[match_idx],pattern_bias,node_generator,{}, &graph_def_tmp[match_idx+1]));
      TF_RETURN_IF_ERROR(IsGraphValid(graph_def_tmp[match_idx+1]));
      match_idx++;
    }
  }
  TF_RETURN_IF_ERROR(ReplaceMatchingOpTypes(graph_def_tmp[match_idx],{"Const"},node_generator,{}, output_graph_def));
  TF_RETURN_IF_ERROR(IsGraphValid(*output_graph_def));
  return Status::OK();
}

REGISTER_GRAPH_TRANSFORM("quantize_weights", QuantizeWeights);

}  // namespace graph_transforms
}  // namespace tensorflow

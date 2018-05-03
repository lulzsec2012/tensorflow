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
void CopyOriginalMatchExceptBiasNode(const NodeMatch& match,std::vector<NodeDef>* new_nodes) {
  std::vector<NodeDef> old_nodes;
  MatchedNodesAsArray(match, &old_nodes);
  for (const NodeDef& old_node : old_nodes) {
    if(old_node.name() != match.inputs[1].node.name()){
      new_nodes->push_back(old_node);	
    }
  }
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
  int match_idx = 0;
  std::vector<string> quant_bias_names;
  const std::function<Status(const NodeMatch&, const std::set<string>&,
                               const std::set<string>&, std::vector<NodeDef>*)>&
    node_generator = [&match_idx,minimum_size, &quant_bias_names](const NodeMatch& match,
                     const std::set<string>& input_nodes,
                     const std::set<string>& output_nodes,
                     std::vector<NodeDef>* new_nodes) {
    if(match_idx==6)std::cout<<std::endl<<"XXXXX:"<<"1"<<std::endl;
        NodeDef old_const_node;
	if(match_idx < 6){
	  old_const_node = match.inputs[1].node;
	  std::cout<<"work:match.node.name():"<<match.node.name()<<"::"<<"match.node.op():"<<match.node.op()<<std::endl;
	}else{
	  old_const_node = match.node;
	  if(match_idx==6)
	    std::cout<<"match.node.name():"<<match.node.name()<<"::"<<"match.node.op():"<<match.node.op()<<std::endl;
	}

        if (!old_const_node.attr().count("dtype")) {
          return errors::InvalidArgument("No 'dtype' attribute for Const node ",
                                         old_const_node.name());
        }
        if (!old_const_node.attr().count("value")) {
          return errors::InvalidArgument("No 'value' attribute for Const node ",
                                         old_const_node.name());
        }
    if(match_idx==6)std::cout<<"XXXXX:"<<"2"<<std::endl;
        const DataType old_dtype = old_const_node.attr().at("dtype").type();
        Tensor old_tensor;
        if (!old_tensor.FromProto(old_const_node.attr().at("value").tensor())) {
          return errors::InvalidArgument("Decoding Tensor failed for node",
                                         old_const_node.name());
        }
    if(match_idx==6)std::cout<<"XXXXX:"<<"3"<<std::endl;
        const size_t num_elements = old_tensor.NumElements();
        // If this isn't a float constant, or it's too small, then reuse the
        // same node with no changes.
        if ((old_dtype != DT_FLOAT) || (num_elements < minimum_size)) {
	  if(match_idx==6)std::cout<<"XXXXX:"<<"4"<<std::endl;
          new_nodes->push_back(old_const_node);
	  if(match_idx==6)std::cout<<"XXXXX:"<<"5"<<std::endl;
          return Status::OK();
        }
    if(match_idx==6)std::cout<<"XXXXX:"<<"A"<<std::endl;
	float last_activation_scale = 1.0;
	if(match_idx < 3){
	  //pattern_flag = HOLD;
	  last_activation_scale = 1.0;
	}else if(match_idx < 6){
	  //pattern_flag = BIAS;
	  NodeMatch current_match = match.inputs[0];
	  for(int i=0;i<=match_idx%3;i++){
	    current_match = current_match.inputs[0];
	  }
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
	  last_activation_scale = 255 / (last_activation_max_value[0] - last_activation_min_value[0]);	
	}else{
	  //pattern_flag = WEIGHT;
	  for (int i = 0; i < quant_bias_names.size(); i++){
	    if (match.node.name() == quant_bias_names[i]){
	      new_nodes->push_back(match.node);
	      return Status::OK();
	    }
	  }
	  std::cout<<"match_idx:"<<match_idx<<"match.node.name():"<<match.node.name()<<"::"<<"match.node.op():"<<match.node.op()<<std::endl;
	}
    if(match_idx==6)std::cout<<"XXXXX:"<<"B"<<std::endl;
	float min,max;
	NodeDef quantized_const_node;
	if(match_idx < 6){
	  CopyOriginalMatchExceptBiasNode(match, new_nodes);
	  
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
	  Tensor quantized_tensor(DT_QINT32, old_tensor.shape());
	  qint32* quantized_tensor_value = quantized_tensor.flat<qint32>().data();
	  const size_t quantized_tensor_elements = quantized_tensor.NumElements();
	  float* old_tensor_value = old_tensor.flat<float>().data(); 
	  for (int i = 0; i < quantized_tensor_elements; i++){
	    
	    quantized_tensor_value[i] = static_cast<int32_t>(round(old_tensor_value[i] * weight_scale * last_activation_scale));	  
	    
	  }
	  min = static_cast<float>(-(1 << 31) / weight_scale / last_activation_scale);
	  max = static_cast<float>((1 << 31) / weight_scale / last_activation_scale);
	  quantized_const_node.set_op("Const");
	  quantized_const_node.set_name(old_const_node.name() +
					"_quantized_const");
	  SetNodeAttr("dtype", DT_QINT32, &quantized_const_node);
	  SetNodeTensorAttr<float>("value", quantized_tensor,
                                 &quantized_const_node);
	  new_nodes->push_back(quantized_const_node);
	  quant_bias_names.push_back(quantized_const_node.name());
	}else{
	
        const float* old_values = old_tensor.flat<float>().data();
        min = std::numeric_limits<float>::max();
        max = std::numeric_limits<float>::min();
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
	if(match_idx!=6)quant_bias_names.push_back(quantized_const_node.name());
	}//
    if(match_idx==6)std::cout<<"XXXXX:"<<"C"<<std::endl;
        NodeDef min_node;
        min_node.set_op("Const");
        min_node.set_name(old_const_node.name() + "_quantized_min");
        SetNodeAttr("dtype", DT_FLOAT, &min_node);
        Tensor min_tensor(DT_FLOAT, {});
        min_tensor.scalar<float>()() = min;
        SetNodeTensorAttr<float>("value", min_tensor, &min_node);
        new_nodes->push_back(min_node);
        if(match_idx!=6)quant_bias_names.push_back(min_node.name());
    if(match_idx==6)std::cout<<"XXXXX:"<<"D"<<std::endl;
        NodeDef max_node;
        max_node.set_op("Const");
        max_node.set_name(old_const_node.name() + "_quantized_max");
        SetNodeAttr("dtype", DT_FLOAT, &max_node);
        Tensor max_tensor(DT_FLOAT, {});
        max_tensor.scalar<float>()() = max;
        SetNodeTensorAttr<float>("value", max_tensor, &max_node);
        new_nodes->push_back(max_node);
        if(match_idx!=6)quant_bias_names.push_back(max_node.name());
    if(match_idx==6)std::cout<<"XXXXX:"<<"E"<<std::endl;
        NodeDef dequantize_node;
        dequantize_node.set_op("Dequantize");
        dequantize_node.set_name(old_const_node.name());
        if(match_idx!=6){
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

  
  OpTypePattern pattern_hold = {"Placeholder"};
  OpTypePattern pattern_fake = {"FakeQuantWithMinMaxVars", {{"*"}, {"Const"}, {"Const"}}};    
  std::vector<OpTypePattern> pattern_vec = {pattern_hold,pattern_fake};
  
  GraphDef graph_def_tmp[7];
  graph_def_tmp[0] = input_graph_def;
  for(std::vector<OpTypePattern>::iterator it=pattern_vec.begin();it != pattern_vec.end();it++){
    OpTypePattern pattern = *it;
    const int max_depth = 3;
    for (int depth = 0; depth < max_depth; depth++) {
      for (int i = 0; i < depth; i++) {
	pattern = {"*", {pattern}};
      }
      OpTypePattern pattern_conv = {"Conv2D|DepthwiseConv2dNative",{pattern,{"*"}}};
      OpTypePattern pattern_bias = {"Add|BiasAdd",{pattern_conv,{"*"}}};
      TF_RETURN_IF_ERROR(ReplaceMatchingOpTypes(graph_def_tmp[match_idx],pattern_bias,node_generator,{}, &graph_def_tmp[match_idx+1]));
      TF_RETURN_IF_ERROR(IsGraphValid(graph_def_tmp[match_idx+1]));
      std::cout<<"OK:"<<match_idx<<std::endl;
      match_idx++;
    }
  }
  std::cout<<"match_idxAX:"<<match_idx<<std::endl;
  //TF_RETURN_IF_ERROR(ReplaceMatchingOpTypes(graph_def_tmp[match_idx], {"Const"},node_generator,{}, output_graph_def));
  //TF_RETURN_IF_ERROR(IsGraphValid(*output_graph_def));
  std::cout<<"OK:"<<match_idx<<std::endl;
    TF_RETURN_IF_ERROR(ReplaceMatchingOpTypes(
      graph_def_tmp[match_idx], {"Const"},
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
	  //std::cout << "name: " << old_const_node.name() <<" "<<  old_const_node.op()<<std::endl;
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
    //
  return Status::OK();
}

REGISTER_GRAPH_TRANSFORM("quantize_weights", QuantizeWeights);

}  // namespace graph_transforms
}  // namespace tensorflow

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
#include "tensorflow/core/lib/strings/str_util.h"

namespace tensorflow {
namespace graph_transforms {

// Converts any large float constants into eight-bit equivalents, with a
// Dequantize op so that subsequent nodes can still access the results in a
// float form.
Status FoldMeanAndScale(const GraphDef& input_graph_def,
                       const TransformFuncContext& context,
                       GraphDef* output_graph_def) {
  float scale_value;
  TF_RETURN_IF_ERROR(
      context.GetOneFloatParameter("scale", 1.0, &scale_value));
  

  std::set<string> ops_to_ignore;
  if (context.params.count("ignore_op") > 0) {
    for (const string& name : context.params.at("ignore_op")) {
      ops_to_ignore.insert(name);
    }
  }
  std::vector<float> mean_value;
  if (context.params.count("mean")) {
    if (context.params.at("mean").size() != 1) {
      return errors::InvalidArgument(
          "You must pass no more than one default 'mean' to "
          "fold_mean_node");
    }
    const string& mean_string = context.params.at("mean")[0];
    if (!str_util::SplitAndParseAsFloats(mean_string, ',', &mean_value)) {
      return errors::InvalidArgument("Could parse as mean value: '", mean_string,
				     "'");
    }    
  }

  
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

        [&mean_value, scale_value, &ops_to_ignore](const NodeMatch& match,
                     const std::set<string>& input_nodes,
                     const std::set<string>& output_nodes,
                     std::vector<NodeDef>* new_nodes) {
	const NodeDef& origin_node = match.node;
        const NodeDef& conv_const_node = match.inputs[0].node;
	const NodeDef& bias_const_node = match.inputs[1].node;
	const NodeDef& weight_const_node = match.inputs[0].inputs[1].node;
	const NodeDef& fake_quant_acvation_const_node = match.inputs[0].inputs[0].node;
	//std::cerr << "bias_node.name: "<< old_const_node.name() << std::endl;
	new_nodes->push_back(conv_const_node);
	new_nodes->push_back(match.node);
	//new_nodes->push_back(fake_quant_weight_const_node);
	new_nodes->push_back(fake_quant_acvation_const_node);
        if (!bias_const_node.attr().count("dtype")) {
          return errors::InvalidArgument("No 'dtype' attribute for Const node ",
                                         bias_const_node.name());
        }
        if (!bias_const_node.attr().count("value")) {
          return errors::InvalidArgument("No 'value' attribute for Const node ",
                                         bias_const_node.name());
        }
        const DataType bias_dtype = bias_const_node.attr().at("dtype").type();
        Tensor bias_tensor;
        if (!bias_tensor.FromProto(bias_const_node.attr().at("value").tensor())) {
          return errors::InvalidArgument("Decoding Tensor failed for node",
                                         bias_const_node.name());
        }
	float* bias_tensor_value = bias_tensor.flat<float>().data(); 
        const size_t num_elements = bias_tensor.NumElements();
        // If this isn't a float constant, or it's too small, then reuse the
        // same node with no changes.
        //if ((bias_dtype != DT_FLOAT)) {
        //  new_nodes->push_back(bias_const_node);
        //  return Status::OK();
        //}
	Tensor weight_tensor;
	if (!weight_tensor.FromProto(weight_const_node.attr().at("value").tensor())) {
	  return errors::InvalidArgument("Decoding Tensor faild for node",
					weight_const_node.name());
	}
	const float* weight_tensor_value = weight_tensor.flat<float>().data();
	const size_t weight_num_elements = weight_tensor.NumElements();


	Tensor new_weight_tensor(DT_FLOAT, weight_tensor.shape());
	float* new_weight_tensor_value = new_weight_tensor.flat<float>().data();
	const size_t new_weight_tensor_elements = new_weight_tensor.NumElements();

	for (int i = 0; i < new_weight_tensor_elements; i++){

	  new_weight_tensor_value[i] = static_cast<float>(weight_tensor_value[i] / scale_value);	  
	 
	}


	Tensor new_bias_tensor(DT_FLOAT, bias_tensor.shape());
	float* new_bias_tensor_value = new_bias_tensor.flat<float>().data();
	const size_t new_bias_tensor_elements = new_bias_tensor.NumElements();

	//conv2d
	int weight_height = weight_tensor.dim_size(0);
	int weight_width = weight_tensor.dim_size(1);
	int in_channel = weight_tensor.dim_size(2);
	int out_channel = weight_tensor.dim_size(3);
	int h_size = out_channel * in_channel * weight_width;
	int w_size = out_channel * in_channel;
	//	
	
	for (int i = 0; i < out_channel; i++){
	  float tmp = 0;
	  for (int j = 0; j < in_channel; j++){
	    for (int h = 0; h < weight_height; h++){
	      for (int w = 0; w < weight_width; w++){
		tmp += new_weight_tensor_value[i + j * out_channel + w * w_size + h * h_size] * mean_value[j];				
	      }
	    }
	  }
	  new_bias_tensor_value[i] = bias_tensor_value[i] - tmp;	  
	}

        NodeDef new_bias_const_node;
        new_bias_const_node.set_op("Const");
        new_bias_const_node.set_name(bias_const_node.name());
        SetNodeAttr("dtype", DT_FLOAT, &new_bias_const_node);
        SetNodeTensorAttr<float>("value", new_bias_tensor,
                                 &new_bias_const_node);
        new_nodes->push_back(new_bias_const_node);

        NodeDef new_weight_const_node;
        new_weight_const_node.set_op("Const");
        new_weight_const_node.set_name(weight_const_node.name());
        SetNodeAttr("dtype", DT_FLOAT, &new_weight_const_node);
        SetNodeTensorAttr<float>("value", new_weight_tensor,
                                 &new_weight_const_node);
        new_nodes->push_back(new_weight_const_node);
	

        return Status::OK();
      },
      {}, output_graph_def));


  return Status::OK();
}

REGISTER_GRAPH_TRANSFORM("fold_mean_value", FoldMeanAndScale);

}  // namespace graph_transforms
}  // namespace tensorflow

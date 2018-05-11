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

Status BiasAdd2Add(const GraphDef& input_graph_def,
                                    const TransformFuncContext& context,
                                    GraphDef* output_graph_def) {
  GraphDef replaced_graph_def;
  TF_RETURN_IF_ERROR(ReplaceMatchingOpTypes(
      input_graph_def,  // clang-format off
      {"Squeeze",
        {
      	  {"*"},
        }
      },  // clang-format on
      [](const NodeMatch& match, const std::set<string>& input_nodes,
         const std::set<string>& output_nodes,
         std::vector<NodeDef>* new_nodes) {

        const NodeDef& old_op = match.node;
	
	std::cout << "=============================================" << std::endl;
	std::cout << "old_op = " << old_op.name() << std::endl;
	std::cout << "match.inputs[0].node.name() = " << match.inputs[0].node.name() << std::endl;
	// std::cout << "match.inputs[1].node.name() = " << match.inputs[1].node.name() << std::endl;
	// std::cout << "match.inputs[2].node.name() = " << match.inputs[2].node.name() << std::endl;


        new_nodes->push_back(match.inputs[0].node);
        // new_nodes->push_back(match.inputs[1].node);
        // new_nodes->push_back(match.inputs[2].node:1);
        // new_nodes->push_back(match.inputs[3].node:2);
        // new_nodes->push_back(match.inputs[4].node);
        // new_nodes->push_back(match.inputs[5].node);

	// TensorShape s({1, 1});
	// Tensor shape_tensor(DT_INT32, s);
	Tensor shape_tensor(DT_INT32, TensorShape({2}));
        int32* shape_values = shape_tensor.flat<int32>().data();
	shape_values[0] = -1;
	shape_values[1] = 1024;
	std::cout << "shape_values = " << shape_values << std::endl;	

	NodeDef const_shape_op;
        const_shape_op.set_op("Const");
        const_shape_op.set_name("logits/shape");
        // SetNodeAttr("dtype", DT_FLOAT, &rounded_const_node);
        SetNodeAttr("dtype", DT_INT32, &const_shape_op);
	
        // SetNodeTensorAttr<float>("value", shape_tensor, &const_shape_op);
	SetNodeTensorAttr<int>("value", shape_tensor, &const_shape_op);

	new_nodes->push_back(const_shape_op);


	
	
        NodeDef new_op;
        new_op.set_op("Reshape");
        new_op.set_name(old_op.name());
        SetNodeAttr("T", DT_FLOAT, &new_op);
        SetNodeAttr("Tshape", DT_INT32, &new_op);
        AddNodeInput(match.inputs[0].node.name(), &new_op);
        AddNodeInput(const_shape_op.name(), &new_op);	
	new_nodes->push_back(new_op);


	std::cout << "=============================================" << std::endl;	
        return Status::OK();
      },
      {}, &replaced_graph_def));
  *output_graph_def = replaced_graph_def;
  return Status::OK();
}

  /************************/

REGISTER_GRAPH_TRANSFORM("biasadd_to_add", BiasAdd2Add);
  
}  // namespace graph_transforms
}  // namespace tensorflow

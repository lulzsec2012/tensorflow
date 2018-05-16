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

#include "tensorflow/core/kernels/quantization_utils.h"

namespace tensorflow {

void GetOutputMinAndMaxForQuantizedAdd(float input_min, float input_max,
                                       float smaller_input_min,
                                       float smaller_input_max,
                                       float* output_min, float* output_max) {
  // We need to have a good range to add our two arguments together in. This
  // is surprisingly tricky, since it has to satisfy a few different needs:
  //  - Must be symmetrical around zero, so that 0 + 0 = 0.
  //  - Must hold the largest of the argument ranges.
  //  - Should have enough range that the bits of the lowest and highest
  //    arguments overlap if possible without the lower getting truncated.
  //  - Should have some headroom so that there's no overflow.
  //  - Needs to be signed.
  // This leads us to use a scheme where we (assuming the inputs are eight bit
  // and the output is 32-bit) use the bottom 32 - 17 = 15 bits to store the
  // accumulated results. This gives us all the properties we need.
  *output_max =
      std::max(input_max, std::max(-input_min, std::max(smaller_input_max,
                                                        -smaller_input_min))) *
      (1 << 17);
  *output_min = -(*output_max);
}
  
void QuantizeMultiplierEightBits(double double_multiplier, int32_t* quantized_multiplier,
                        int* shift) {
  if (double_multiplier == 0.) {
    *quantized_multiplier = 0;
    *shift = 0;
    return;
  }
  float min_loss = 100.0;
  for (int n = -32; n < 32; n++){
    int m = round(double_multiplier * pow(2, n));
    if (m > 0 && m < 256){
      float loss = std::fabs(double_multiplier - m /(pow(2, n) + 0.0));
      if (min_loss > loss){
	min_loss = loss;
	*shift = n;
      }
    }
  }
  *quantized_multiplier = static_cast<int32_t>(round(double_multiplier * pow(2, *shift)));
}
  
}  // namespace tensorflow

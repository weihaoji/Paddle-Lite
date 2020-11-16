// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "lite/kernels/xpu/__xpu__block_fuse_compute.h"
#include <vector>
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

void XPUBlockFuseCompute::PrepareForRun() {
  auto& param = this->Param<param_t>();
  auto op_type = param.op_type;

  auto place_x = param.place_x;
  auto place_y = param.place_y;
  auto place_z = param.place_z;

  auto& has_bias = param.has_bias;
  auto& filter_dims = param.filter_dims;
  auto& conv_strides = param.conv_strides;
  auto& conv_paddings = param.conv_paddings;
  auto& conv_dilations = param.conv_dilations;
  auto& conv_groups = param.conv_groups;
  auto& act_type = param.act_type;
  auto& act_param = param.act_param;
  auto& block_lod = param.block_lod;

  int op_count = 0;
  int conv_count = 0;
  int conv_bias_count = 0;
  for (int block_idx = 0; block_idx < block_lod.size(); block_idx++) {
    int cur_block_op_num = block_lod[block_idx];
    xdnn::fusion_block<float, int16_t, int16_t, float> cur_block;
    for (int op_idx = 0; op_idx < cur_block_op_num; op_idx++) {
      xdnn::Activation_t act((xdnn::Activation_t::act_enum)act_type[op_count]);
      if (act_type[op_count] == 5) {
        act.leaky_alpha = act_param[op_count];
      } else if (act_type[op_count] == 15) {
        act.hard_sigmoid_slope = act_param[op_count];
      }
      switch (op_type[op_idx]) {
        case 0: {
          int r = cur_block.add_conv_layer(
              place_x[op_count],
              place_y[op_count],
              place_z[op_count],
              reinterpret_cast<const int16_t*>(
                  param.filter[conv_count]->data<float>()),
              filter_dims[conv_count * 4 + 1] * conv_groups[conv_count],
              filter_dims[conv_count * 4 + 0],
              {filter_dims[conv_count * 4 + 2],
               filter_dims[conv_count * 4 + 3]},
              {conv_strides[conv_count * 2 + 0],
               conv_strides[conv_count * 2 + 1]},
              {conv_paddings[conv_count * 4 + 0],
               conv_paddings[conv_count * 4 + 1],
               conv_paddings[conv_count * 4 + 2],
               conv_paddings[conv_count * 4 + 3]},
              {conv_dilations[conv_count * 2 + 0],
               conv_dilations[conv_count * 2 + 1]},
              conv_groups[conv_count],
              param.max_filter[conv_count]->data<float>(),
              true,
              has_bias[conv_count] ? param.bias[conv_bias_count]->data<float>()
                                   : nullptr,
              act);
          CHECK_EQ(r, 0);
          conv_bias_count =
              has_bias[conv_count] ? (conv_bias_count + 1) : conv_bias_count;
          conv_count += 1;
          op_count += 1;
          break;
        }
        case 1: {
          // ew_add
          int r = cur_block.add_ew_layer(
              place_x[op_count], place_y[op_count], place_z[op_count], act);
          CHECK_EQ(r, 0);
          op_count += 1;
          break;
        }
        default: { break; }
      }
    }
    xpu_fusion_block.push_back(cur_block);
  }
}

void XPUBlockFuseCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->As<XPUContext>();

  auto& input_dims = param.input->dims();
  int n = static_cast<int>(input_dims[0]);
  int c = static_cast<int>(input_dims[1]);
  int h = static_cast<int>(input_dims[2]);
  int w = static_cast<int>(input_dims[3]);
  auto& has_block_output = param.has_block_output;

  std::vector<float*> feature_list;
  int f = c, yh = h, yw = w;
  int feature_list_count = 0;
  for (int block_idx = 0; block_idx < xpu_fusion_block.size(); block_idx++) {
    int cur_block_c = f;
    int cur_block_h = yh;
    int cur_block_w = yw;
    int r = xpu_fusion_block[block_idx].infer_shape(
        n, cur_block_c, cur_block_h, cur_block_w, &f, &yh, &yw);
    CHECK_EQ(r, 0);
    if (block_idx != xpu_fusion_block.size() - 1) {
      if (has_block_output[block_idx]) {
        param.block_output[feature_list_count]->Resize({n, f, yh, yw});
        feature_list.push_back(
            param.block_output[feature_list_count]->mutable_data<float>(
                TARGET(kXPU)));
        feature_list_count += 1;
      } else {
        feature_list.push_back(nullptr);
      }
    }
  }
  CHECK_EQ(feature_list_count, param.block_output.size());
  auto output = param.output;
  output->Resize({n, f, yh, yw});
  const float* input_max =
      param.input_max ? param.input_max->data<float>() : nullptr;
  float* output_max = param.output_max->mutable_data<float>(TARGET(kXPU));

  int r = xdnn::run_fusion_block_list<float, int16_t, int16_t, float>(
      ctx.GetRawContext(),
      param.input->data<float>(),
      output->mutable_data<float>(TARGET(kXPU)),
      input_max,
      output_max,
      n,
      c,
      h,
      w,
      xpu_fusion_block,
      feature_list);

  CHECK_EQ(r, 0);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(__xpu__block_fuse_op,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::XPUBlockFuseCompute,
                     def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Filter", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("InputMax", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("FilterMax", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Output", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("OutputMax", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();

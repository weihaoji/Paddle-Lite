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

#include "lite/kernels/xpu/__xpu__conv2d_compute.h"
#include <string>
#include <vector>
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

void XPUConv2dCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->As<XPUContext>();

  auto& input_dims = param.Input->dims();
  auto& filter_dims = param.Filter->dims();
  int batch = static_cast<int>(input_dims[0]);
  int img_c = static_cast<int>(input_dims[1]);
  int img_h = static_cast<int>(input_dims[2]);
  int img_w = static_cast<int>(input_dims[3]);
  int filter_num = static_cast<int>(filter_dims[0]);
  int win_h = static_cast<int>(filter_dims[2]);
  int win_w = static_cast<int>(filter_dims[3]);

  auto paddings = *param.paddings;
  auto dilations = *param.dilations;
  int stride_h = param.strides[0];
  int stride_w = param.strides[1];
  int paddings_h = paddings[0];
  int paddings_w = paddings[1];
  int dilations_h = dilations[0];
  int dilations_w = dilations[1];

  std::string filter_type = param.filter_type;
  int groups = param.groups;
  int act_type = (param.act_type == -1) ? 0 : param.act_type;
  const auto* bias = param.Bias ? param.Bias->data<float>() : nullptr;
  const auto* branch = param.Branch ? param.Branch->data<float>() : nullptr;
  const float* input_max =
      param.InputMax ? param.InputMax->data<float>() : nullptr;
  float* output_max = param.OutputMax
                          ? param.OutputMax->mutable_data<float>(TARGET(kXPU))
                          : nullptr;
  float* output = param.Output->mutable_data<float>(TARGET(kXPU));

  // TODO(luohang): now support for resnet50 first
  CHECK_EQ(filter_type, "int16");
  xdnn::Activation_t act((xdnn::Activation_t::act_enum)act_type);
  if (act_type == 5) {
    act.leaky_alpha = param.leaky_relu_alpha;
    CHECK(act.leaky_alpha >= 0.0001 && act.leaky_alpha <= 10);
  } else if (act_type == 15) {
    act.hard_sigmoid_slope = param.hard_sigmoid_slope;
  }

  if ((win_h > 11) || (win_w > 11) || (stride_h > 10) || (stride_w > 10) ||
      (paddings_h > 5) || (paddings_w > 5) ||
      (img_h + paddings_h * 2 - win_h < 0) ||
      (img_w + paddings_w * 2 - win_w < 0)) {
    int r = xdnn::conv2d<float, int16_t, float, int16_t>(
        ctx.GetRawContext(),           /* context */
        param.Input->data<float>(),    /* input bottom */
        param.Filter->data<int16_t>(), /* filter weight */
        output,
        batch,
        img_c,
        img_h,
        img_w,
        filter_num,
        std::vector<int>{win_h, win_w},
        param.strides,
        paddings,
        dilations,
        groups,
        input_max,
        param.FilterMax->data<float>(), /* max_filter_ptr */
        output_max,
        true);

    CHECK_EQ(r, 0);

    float* y_broadcast = nullptr;

    r = xpu_malloc(reinterpret_cast<void**>(&y_broadcast),
                   param.Output->numel() * sizeof(float));
    CHECK_EQ(r, 0);

    if (bias) {
      r = xdnn::broadcast_ew(ctx.GetRawContext(),
                             bias,
                             y_broadcast,
                             batch,
                             filter_num,
                             img_h * img_w,
                             xdnn::ElementwiseOp::ASSIGN);

      CHECK_EQ(r, 0);

      r = xdnn::elementwise_add(ctx.GetRawContext(), /* context */
                                output,
                                y_broadcast, /* y */
                                y_broadcast,
                                param.Output->numel());
      CHECK_EQ(r, 0);
    }

    r = xdnn::activation_forward(ctx.GetRawContext(), /* context */
                                 act,
                                 param.Output->numel(),
                                 y_broadcast,
                                 output);
    CHECK_EQ(r, 0);

    xpu_free(y_broadcast);

  } else if (act_type >= 14) {
    int r = xdnn::conv2d_fusion<float, int16_t, float, int16_t>(
        ctx.GetRawContext(),           /* context */
        param.Input->data<float>(),    /* input bottom */
        param.Filter->data<int16_t>(), /* filter weight */
        output,
        batch,
        img_c,
        img_h,
        img_w,
        filter_num,
        std::vector<int>{win_h, win_w},
        param.strides,
        paddings,
        dilations,
        groups,
        input_max,
        param.FilterMax->data<float>(), /* max_filter_ptr */
        output_max,
        true,
        bias,   /* bias */
        branch, /* branch */
        act);
    CHECK_EQ(r, 0);

  } else if (groups == 1) {
    int r = xdnn::conv2d_forward_int16<float, int16_t, float, float>(
        ctx.GetRawContext(),            /* context */
        batch,                          /* batch */
        img_c,                          /* input_c */
        img_h,                          /* input_h */
        img_w,                          /* input_w */
        filter_num,                     /* num_filter */
        win_h,                          /* kernel_h */
        win_w,                          /* kernel_w */
        stride_h,                       /* stride_h */
        stride_w,                       /* stride_w */
        paddings_h,                     /* pad_h */
        paddings_w,                     /* pad_w */
        dilations_h,                    /* dilation_h */
        dilations_w,                    /* dilation_w */
        groups,                         /* group */
        param.Input->data<float>(),     /* input bottom */
        param.Filter->data<int16_t>(),  /* filter weight */
        output,                         /* output top */
        bias,                           /* bias */
        branch,                         /* branch */
        act,                            /* act type */
        input_max,                      /* max_image_ptr */
        param.FilterMax->data<float>(), /* max_filter_ptr */
        output_max /* max_result_ptr */);

    CHECK_EQ(r, 0);

  } else {
    int r = xdnn::conv2d_int16_with_group<float, int16_t, float>(
        ctx.GetRawContext(), /* context */
        param.Input->data<float>(),
        param.Filter->data<int16_t>(),
        output,
        batch,
        img_c,
        img_h,
        img_w,
        filter_num,
        win_h,
        win_w,
        groups,
        stride_h,
        stride_w,
        paddings_h,
        paddings_w,
        input_max,
        param.FilterMax->data<float>(), /* max_filter_ptr */
        output_max,
        bias,
        act);
    CHECK_EQ(r, 0);
  }
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(__xpu__conv2d,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::XPUConv2dCompute,
                     def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Filter", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("InputMax", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("FilterMax", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Branch", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Output", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("OutputMax", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();

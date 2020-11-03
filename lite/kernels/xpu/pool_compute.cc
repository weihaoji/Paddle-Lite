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

#include "lite/kernels/xpu/pool_compute.h"
#include <vector>
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

void Pool2DCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->As<XPUContext>();

  auto& x_dims = param.x->dims();
  CHECK_EQ(x_dims.size(), 4);
  CHECK_EQ(param.ksize.size(), 2);
  if (param.global_pooling) {
    param.ksize[0] = x_dims[2];
    param.ksize[1] = x_dims[3];
  }
  CHECK_EQ(param.strides.size(), 2);
  CHECK_EQ(param.paddings->size(), 4);
  std::vector<int> paddings{(*param.paddings)[0],
                            (*param.paddings)[1],
                            (*param.paddings)[2],
                            (*param.paddings)[3]};
  if (param.ceil_mode) {
    paddings[1] += param.strides[0] - 1;
    paddings[3] += param.strides[1] - 1;
  }

  if (param.adaptive) {
    if (param.pooling_type == "avg") {
      int r = xdnn::adaptive_avg_pool2d(
          ctx.GetRawContext(),                             /* context */
          param.x->data<float>(),                          /* x */
          param.output->mutable_data<float>(TARGET(kXPU)), /* y */
          x_dims[0],
          x_dims[1],
          x_dims[2],
          x_dims[3],
          param.ksize[0],
          param.ksize[1],
          true);
      CHECK_EQ(r, 0);
    } else {
      int r = xdnn::adaptive_max_pool2d(
          ctx.GetRawContext(),                             /* context */
          param.x->data<float>(),                          /* x */
          param.output->mutable_data<float>(TARGET(kXPU)), /* y */
          nullptr,
          x_dims[0],
          x_dims[1],
          x_dims[2],
          x_dims[3],
          param.ksize[0],
          param.ksize[1],
          true);
      CHECK_EQ(r, 0);
    }
  } else {
    if (param.pooling_type == "avg") {
      bool count_include_pad = param.exclusive ? false : true;
      int r = xdnn::avg_pool2d<float>(
          ctx.GetRawContext(),                             /* context */
          param.x->data<float>(),                          /* x */
          param.output->mutable_data<float>(TARGET(kXPU)), /* y */
          x_dims[0],
          x_dims[1],
          x_dims[2],
          x_dims[3],
          param.ksize,
          param.strides,
          paddings,
          count_include_pad,
          true);
      CHECK_EQ(r, 0);
    } else {
      // handle max pool error
      if (param.ksize[0] == 3 && param.ksize[1] == 3 && param.strides[0] == 2 &&
          param.strides[1] == 1 && paddings[0] == 1 && paddings[1] == 1) {
        float* y_xpu = param.output->mutable_data<float>(TARGET(kXPU));

        float* y_cpu = reinterpret_cast<float*>(
            malloc(param.output->numel() * sizeof(float)));
        float* x_cpu =
            reinterpret_cast<float*>(malloc(param.x->numel() * sizeof(float)));
        XPU_CALL(xpu_memcpy(x_cpu,
                            param.x->data<float>(),
                            param.x->numel() * sizeof(float),
                            XPU_DEVICE_TO_HOST));
        XPU_CALL(xpu_wait());
        xdnn::Context ctx_cpu(xdnn::kCPU);
        int r = xdnn::max_pool2d<float>(&ctx_cpu,
                                        x_cpu,
                                        y_cpu,
                                        nullptr,
                                        x_dims[0],
                                        x_dims[1],
                                        x_dims[2],
                                        x_dims[3],
                                        param.ksize,
                                        param.strides,
                                        paddings,
                                        true);
        CHECK_EQ(r, 0);
        XPU_CALL(xpu_memcpy(y_xpu,
                            y_cpu,
                            param.output->numel() * sizeof(float),
                            XPU_HOST_TO_DEVICE));
        XPU_CALL(xpu_wait());
        free(y_cpu);
        free(x_cpu);
      } else {
        int r = xdnn::max_pool2d<float>(
            ctx.GetRawContext(),                             /* context */
            param.x->data<float>(),                          /* x */
            param.output->mutable_data<float>(TARGET(kXPU)), /* y */
            nullptr,
            x_dims[0],
            x_dims[1],
            x_dims[2],
            x_dims[3],
            param.ksize,
            param.strides,
            paddings,
            true);
        CHECK_EQ(r, 0);
      }
    }
  }
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(
    pool2d, kXPU, kFloat, kNCHW, paddle::lite::kernels::xpu::Pool2DCompute, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();

REGISTER_LITE_KERNEL(max_pool2d_with_index,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::Pool2DCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Mask", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();

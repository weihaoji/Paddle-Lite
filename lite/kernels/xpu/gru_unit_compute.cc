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
#include "lite/kernels/xpu/gru_unit_compute.h"
#include <math.h>
#include <algorithm>
#include <map>
#include <string>
#include <vector>
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

inline xdnn::Activation_t GetGruActType(int act_type) {
  switch (act_type) {
    case 1:
      return xdnn::Activation_t::SIGMOID;
    case 2:
      return xdnn::Activation_t::TANH;
    case 3:
      return xdnn::Activation_t::RELU;
    default:
      LOG(FATAL) << "unsupported activation type";
      return xdnn::Activation_t(xdnn::Activation_t::act_enum(0));
  }
}

void GRUUnitCompute::PrepareForRun() {
  // find max
  maxs_xpu_guard_ =
      TargetWrapperXPU::MallocScratchPad(8 * sizeof(float), false /* use_l3 */);
  auto& ctx = this->ctx_->As<XPUContext>();
  auto& param = this->Param<param_t>();
  int frame_size = param.hidden_prev->dims()[1];
  float* weight_ur_max_ptr_xpu =
      reinterpret_cast<float*>(maxs_xpu_guard_->addr_);
  float* weight_c_max_ptr_xpu = weight_ur_max_ptr_xpu + 4;

  // weight_ur_max
  int ret = xdnn::findmax(ctx.GetRawContext(),
                          param.weight->data<float>(),
                          frame_size * frame_size * 2,
                          weight_ur_max_ptr_xpu);
  CHECK_EQ(ret, 0);
  // weight_c_max
  ret = xdnn::findmax(ctx.GetRawContext(),
                      param.weight->data<float>() + frame_size * frame_size * 2,
                      frame_size * frame_size,
                      weight_c_max_ptr_xpu);
  CHECK_EQ(ret, 0);

  float weight_ur_max_cpu[4];
  XPU_CALL(xpu_memcpy(weight_ur_max_cpu,
                      weight_ur_max_ptr_xpu,
                      sizeof(float) * 4,
                      XPUMemcpyKind::XPU_DEVICE_TO_HOST));
  weight_u_r_max_value =
      std::max(std::max(weight_ur_max_cpu[0], weight_ur_max_cpu[1]),
               std::max(weight_ur_max_cpu[2], weight_ur_max_cpu[3]));

  float weight_c_max_cpu[4];
  XPU_CALL(xpu_memcpy(weight_c_max_cpu,
                      weight_c_max_ptr_xpu,
                      sizeof(float) * 4,
                      XPUMemcpyKind::XPU_DEVICE_TO_HOST));
  weight_c_max_value =
      std::max(std::max(weight_c_max_cpu[0], weight_c_max_cpu[1]),
               std::max(weight_c_max_cpu[2], weight_c_max_cpu[3]));
}

void GRUUnitCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->As<XPUContext>();

  auto input = param.input;
  auto hidden_prev = param.hidden_prev;
  const float* hidden_prev_data = hidden_prev->data<float>();

  float* gate_data = param.gate->mutable_data<float>(TARGET(kXPU));
  float* reset_hidden_prev_data =
      param.reset_hidden_prev->mutable_data<float>(TARGET(kXPU));
  float* hidden_data = param.hidden->mutable_data<float>(TARGET(kXPU));

  int batch_size = input->dims()[0];
  int frame_size = hidden_prev->dims()[1];

  int ret = xdnn::gru_unit_int31(ctx.GetRawContext(),
                                 batch_size,
                                 frame_size,
                                 param.origin_mode,
                                 GetGruActType(param.gate_activation),
                                 GetGruActType(param.activation),
                                 input->data<float>(),
                                 hidden_prev_data,
                                 param.weight->data<float>(),
                                 weight_u_r_max_value,
                                 weight_c_max_value,
                                 param.bias->data<float>(),
                                 gate_data,
                                 reset_hidden_prev_data,
                                 hidden_data);
  CHECK_EQ(ret, 0);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(gru_unit,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::GRUUnitCompute,
                     def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("HiddenPrev", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Weight", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Gate", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("ResetHiddenPrev", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Hidden", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();

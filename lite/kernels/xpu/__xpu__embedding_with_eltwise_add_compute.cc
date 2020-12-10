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

#include "lite/kernels/xpu/__xpu__embedding_with_eltwise_add_compute.h"
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

inline int ComputeMaskLen(const float* seq, int seq_len) {
  // debug
  // for (int i = 0; i < seq_len; i++) {
  //   std::cout << seq[i] << "    ";
  // }
  // std::cout << "seq end" << std::endl;

  int begin = 0;
  int end = seq_len - 1;
  if (seq_len <= 0) {
    return 0;
  } else if (seq[0] <= 1e-5) {
    return 0;
  }
  while (begin < end) {
    int mid = (begin + end) / 2 + 1;
    if (std::abs(seq[mid] - 1.0f) <= 1e-5) {
      begin = mid;
    } else {
      end = mid - 1;
    }
  }
  return begin + 1;
}

void XPUEmbeddingWithEltwiseAddCompute::PrepareForRun() {
  auto& param = this->Param<param_t>();

  arg_ids_.reserve(param.Ids.size());
  arg_tables_.reserve(param.Tables.size());
  for (auto* table : param.Tables) {
    auto& table_dims = table->dims();
    CHECK_EQ(table_dims.size(), 2); /* shape like [table_len, embed_dim] */
    table_lens_cpu_.push_back(table_dims[0]);
  }

  size_t lens_size = table_lens_cpu_.size() * sizeof(int);
  table_lens_guard_ =
      TargetWrapperXPU::MallocScratchPad(lens_size, false /* use_l3 */);
  XPU_CALL(xpu_memcpy(table_lens_guard_->addr_,
                      &table_lens_cpu_[0],
                      lens_size,
                      XPU_HOST_TO_DEVICE));
}

void XPUEmbeddingWithEltwiseAddCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->As<XPUContext>();

  for (size_t i = 0; i < param.Tables.size(); ++i) {
    arg_tables_[i] = param.Tables[i]->data<float>();
  }

  auto& id_dims = param.Ids[0]->dims();
  int batch_size = id_dims[0];
  int pad_seq_len = id_dims[1];
  auto& table_dims = param.Tables[0]->dims();
  int embed_dim = table_dims[1];
  int emb_layer_num = param.Ids.size();

  std::vector<int> ids_lod(id_dims[0] + 1, 0);
  std::vector<int64_t> cpu_ids;
  // calculate lod
  if (param.adaptive_seq_len == true) {
    for (size_t batch_idx = 0; batch_idx < batch_size; batch_idx++) {
      int cur_seq_len = ComputeMaskLen(
          param.InputMask->data<float>() + batch_idx * pad_seq_len,
          pad_seq_len);
      std::cout << cur_seq_len << std::endl;
      ids_lod[batch_idx + 1] = ids_lod[batch_idx] + cur_seq_len;
    }
    param.OutputMaskLod->Resize({static_cast<int64_t>(ids_lod.size())});
    XPU_CALL(xpu_memcpy(param.OutputMaskLod->mutable_data<int>(TARGET(kXPU)),
                        ids_lod.data(),
                        sizeof(int) * ids_lod.size(),
                        XPUMemcpyKind::XPU_HOST_TO_DEVICE));
  } else {
    for (size_t batch_idx = 0; batch_idx < batch_size; batch_idx++) {
      ids_lod[batch_idx + 1] = ids_lod[batch_idx] + pad_seq_len;
    }
  }

  // memcpy
  xpu_ids_guard_ = TargetWrapperXPU::MallocScratchPad(
      param.Ids.size() * ids_lod.back() * sizeof(int64_t), false);
  int64_t* xpu_ids = reinterpret_cast<int64_t*>(xpu_ids_guard_->addr_);
  cpu_ids.resize(param.Ids.size() * ids_lod.back());
  int ids_pos = 0;
  for (size_t i = 0; i < param.Ids.size(); i++) {
    for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
      memcpy(cpu_ids.data() + ids_pos,
             param.Ids[i]->data<int64_t>() + batch_idx * pad_seq_len,
             sizeof(int64_t) * (ids_lod[batch_idx + 1] - ids_lod[batch_idx]));
      ids_pos += ids_lod[batch_idx + 1] - ids_lod[batch_idx];
    }
    XPU_CALL(xpu_memcpy(xpu_ids + i * ids_lod.back(),
                        cpu_ids.data() + i * ids_lod.back(),
                        sizeof(int64_t) * ids_lod.back(),
                        XPUMemcpyKind::XPU_HOST_TO_DEVICE));
    arg_ids_[i] = xpu_ids + i * ids_lod.back();
  }

  // debug
  // std::cout << "ids lod is " << std::endl;
  // for (int i = 0; i < ids_lod.size(); i++) {
  //   std::cout << ids_lod[i] << "    ";
  // }
  // std::cout << std::endl;
  // std::vector<int64_t> debug_v(ids_lod.back(), 0);
  // for (int i = 0; i < emb_layer_num; i++) {
  //   xpu_memcpy(debug_v.data(), arg_ids_[i], sizeof(int64_t) * ids_lod.back(),
  //   XPUMemcpyKind::XPU_DEVICE_TO_HOST);
  //   for (int j = 0; j < ids_lod.back(); j++) {
  //     std::cout << debug_v[j] << "    ";
  //   }
  //   std::cout << std::endl;
  // }

  int r = xdnn::embedding_with_ewadd<float, int64_t, false, false>(
      ctx.GetRawContext(),                         /* context */
      embed_dim,                                   /* embed_dim */
      ids_lod.back(),                              /* idx_len */
      emb_layer_num,                               /* emb_layer_num */
      param.padding_idx,                           /* padding_idx */
      &arg_tables_[0],                             /* tables */
      &arg_ids_[0],                                /* indices */
      static_cast<int*>(table_lens_guard_->addr_), /* table_lens */
      nullptr,                                     /* scale_after_emb */
      nullptr,                                     /* scale_after_ewadd */
      param.Out->mutable_data<float>(TARGET(kXPU)) /* top */);
  CHECK_EQ(r, 0);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(
    __xpu__embedding_with_eltwise_add,
    kXPU,
    kFloat,
    kNCHW,
    paddle::lite::kernels::xpu::XPUEmbeddingWithEltwiseAddCompute,
    def)
    .BindInput("Ids", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt64))})
    .BindInput("Tables", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("InputMask", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Output", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("OutputMaskLod", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();

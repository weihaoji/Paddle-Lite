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

#pragma once
#include <string>
#include <vector>
#include "lite/core/kernel.h"
#include "lite/core/op_lite.h"
#include "lite/core/scope.h"
#include "lite/operators/op_params.h"
#include "lite/utils/all.h"

namespace paddle {
namespace lite {
namespace operators {

class PadConstantLikeOpLite : public OpLite {
 public:
  PadConstantLikeOpLite() {}

  explicit PadConstantLikeOpLite(const std::string &type) : OpLite(type) {}

  bool CheckShape() const override;

  bool InferShapeImpl() const override;

  void AttachKernel(KernelBase *kernel) override { kernel->SetParam(param_); }
  // TODO(Superjomn) replace framework::OpDesc with a lite one.
  bool AttachImpl(const cpp::OpDesc &op_desc, lite::Scope *scope) override {
    AttachParam(&param_);

    CHECK(!op_desc.Input("X").empty());
    CHECK(!op_desc.Input("Y").empty());
    CHECK(!op_desc.Output("Out").empty());

    auto x = op_desc.Input("X").front();
    auto y = op_desc.Input("Y").front();
    auto out = op_desc.Output("Out").front();
    auto *var = scope->FindVar(x);
    CHECK(var);
    param_.x = &var->Get<Tensor>();
    var = scope->FindVar(y);
    CHECK(var);
    param_.y = &var->Get<Tensor>();
    var = scope->FindVar(out);
    CHECK(var);
    param_.output = var->GetMutable<Tensor>();

    param_.pad_value = op_desc.GetAttr<float>("pad_value");
    return true;
  }

  std::string DebugString() const override { return "pad_constant_like"; }

#ifdef LITE_WITH_PROFILE
  void GetOpRuntimeInfo(paddle::lite::profile::OpCharacter *ch) {
    ch->input_shape = ch->DimToStr(param_.y->dims());
    ch->output_shape = ch->DimToStr(param_.output->dims());
    // ch->remark = "";
    auto x_dims = param_.x->dims();
    auto y_dims = param_.y->dims();
  }
#endif

 private:
  mutable PadConstantLikeParam param_;
};

}  // namespace operators
}  // namespace lite
}  // namespace paddle

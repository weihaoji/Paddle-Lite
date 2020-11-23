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

#include "lite/operators/__xpu__block_fuse_op.h"
#include <vector>
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool XPUBlockFuseOp::CheckShape() const {
  CHECK(param_.input) << "Input(input) of XPUBlockFuseOp should not be null.";
  CHECK(param_.output)
      << "Output(output) of XPUBlockFuseOp should not be null.";
  return true;
}

bool XPUBlockFuseOp::InferShapeImpl() const {
  param_.output_max->Resize({4});
  param_.output->set_lod(param_.input->lod());
  return true;
}

bool XPUBlockFuseOp::AttachImpl(const cpp::OpDesc& op_desc,
                                lite::Scope* scope) {
  AttachParam(&param_);
  CHECK(scope->FindVar(op_desc.Input("Input").front()));
  CHECK(scope->FindVar(op_desc.Output("Output").front()));
  CHECK(scope->FindVar(op_desc.Output("OutputMax").front()));

  param_.input =
      scope->FindVar(op_desc.Input("Input").front())->GetMutable<Tensor>();
  param_.output =
      scope->FindVar(op_desc.Output("Output").front())->GetMutable<Tensor>();
  param_.output_max =
      scope->FindVar(op_desc.Output("OutputMax").front())->GetMutable<Tensor>();
  param_.filter =
      scope->FindVar(op_desc.Input("Filter").front())->GetMutable<Tensor>();
  param_.max_filter =
      scope->FindVar(op_desc.Input("FilterMax").front())->GetMutable<Tensor>();

  param_.op_type = op_desc.GetAttr<std::vector<int>>("OpType");
  param_.place_x = op_desc.GetAttr<std::vector<int>>("PlaceX");
  param_.place_y = op_desc.GetAttr<std::vector<int>>("PlaceY");
  param_.place_z = op_desc.GetAttr<std::vector<int>>("PlaceZ");

  param_.has_bias = op_desc.GetAttr<std::vector<int>>("HasBias");
  param_.filter_dims = op_desc.GetAttr<std::vector<int>>("FilterDims");
  param_.conv_strides = op_desc.GetAttr<std::vector<int>>("ConvStrides");
  param_.conv_paddings = op_desc.GetAttr<std::vector<int>>("ConvPaddings");
  param_.conv_dilations = op_desc.GetAttr<std::vector<int>>("ConvDilations");
  param_.conv_groups = op_desc.GetAttr<std::vector<int>>("ConvGroups");

  param_.act_type = op_desc.GetAttr<std::vector<int>>("ActType");
  param_.act_param = op_desc.GetAttr<std::vector<float>>("ActParam");
  param_.block_lod = op_desc.GetAttr<std::vector<int>>("BlockLod");
  param_.has_block_output = op_desc.GetAttr<std::vector<int>>("HasBlockOutput");

  // optional params
  std::vector<std::string> input_arg_names = op_desc.InputArgumentNames();
  if (std::find(input_arg_names.begin(), input_arg_names.end(), "Bias") !=
      input_arg_names.end()) {
    auto bias_arguments = op_desc.Input("Bias");
    if (bias_arguments.size() > 0) {
      auto bias_var = scope->FindVar(bias_arguments.front());
      if (bias_var != nullptr) {
        param_.bias =
            const_cast<lite::Tensor*>(&(bias_var->Get<lite::Tensor>()));
      }
    }
  }
  std::vector<std::string> output_arg_names = op_desc.OutputArgumentNames();
  param_.block_output.clear();
  if (std::find(output_arg_names.begin(),
                output_arg_names.end(),
                "BlockOutput") != output_arg_names.end()) {
    auto args = op_desc.Output("BlockOutput");
    if (!args.empty()) {
      for (auto var : args) {
        param_.block_output.push_back(
            scope->FindVar(var)->GetMutable<lite::Tensor>());
      }
    }
  }
  if (op_desc.HasAttr("has_input_max") &&
      op_desc.GetAttr<bool>("has_input_max")) {
    CHECK(scope->FindVar(op_desc.Input("InputMax").front()));
    param_.input_max =
        scope->FindVar(op_desc.Input("InputMax").front())->GetMutable<Tensor>();
  }
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(__xpu__block_fuse_op, paddle::lite::operators::XPUBlockFuseOp);

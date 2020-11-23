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

#include <memory>
#include <string>
#include "lite/backends/xpu/math.h"
#include "lite/core/mir/pass_registry.h"
#include "lite/core/mir/pattern_matcher_high_api.h"

namespace paddle {
namespace lite {
namespace mir {
namespace fusion {
class XPUConsecutiveBlockFuser : public FuseBase {
 public:
  void BuildPattern() override {
    auto* input = VarNode("input")
                      ->assert_is_op_input("__xpu__block_fuse_op", "Input")
                      ->AsInput();
    auto* filter0 = VarNode("filter0")
                        ->assert_is_op_input("__xpu__block_fuse_op", "Filter")
                        ->AsIntermediate();
    auto* filter_max0 =
        VarNode("filter_max0")
            ->assert_is_op_input("__xpu__block_fuse_op", "FilterMax")
            ->AsIntermediate();
    auto* bias0 = VarNode("bias0")
                      ->assert_is_op_input("__xpu__block_fuse_op", "Bias")
                      ->AsIntermediate();
    auto* block0 = OpNode("block0", "__xpu__block_fuse_op")->AsIntermediate();
    auto* block_out0 =
        VarNode("block_out0")
            ->assert_is_op_output("__xpu__block_fuse_op", "Output")
            ->AsIntermediate();
    auto* block_out_max0 =
        VarNode("block_out_max0")
            ->assert_is_op_output("__xpu__block_fuse_op", "OutputMax")
            ->AsIntermediate();
    auto* filter1 = VarNode("filter1")
                        ->assert_is_op_input("__xpu__block_fuse_op", "Filter")
                        ->AsIntermediate();
    auto* filter_max1 =
        VarNode("filter_max1")
            ->assert_is_op_input("__xpu__block_fuse_op", "FilterMax")
            ->AsIntermediate();
    auto* bias1 = VarNode("bias1")
                      ->assert_is_op_input("__xpu__block_fuse_op", "Bias")
                      ->AsIntermediate();
    auto* block1 = OpNode("block1", "__xpu__block_fuse_op")->AsIntermediate();
    auto* block_out1 =
        VarNode("block_out1")
            ->assert_is_op_output("__xpu__block_fuse_op", "Output")
            ->AsOutput();
    auto* block_out_max1 =
        VarNode("block_out_max1")
            ->assert_is_op_output("__xpu__block_fuse_op", "OutputMax")
            ->AsOutput();

    *input >> *block0 >> *block_out0 >> *block1 >> *block_out1;

    *filter0 >> *block0;
    *filter_max0 >> *block0;
    *bias0 >> *block0;
    *block0 >> *block_out_max0;

    *filter1 >> *block1;
    *filter_max1 >> *block1;
    *bias1 >> *block1;
    *block1 >> *block_out_max1;
  }

  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override {
    std::vector<std::string> block_name{"block0", "block1"};
    std::vector<std::string> filter_name{matched.at("filter0")->arg()->name,
                                         matched.at("filter1")->arg()->name};
    std::vector<std::string> bias_name{matched.at("bias0")->arg()->name,
                                       matched.at("bias1")->arg()->name};
    std::vector<std::string> filter_max_name{
        matched.at("filter_max0")->arg()->name,
        matched.at("filter_max1")->arg()->name};

    auto op_desc = *matched.at("block0")->stmt()->op_info();
    auto block_0 = matched.at("block0")->stmt()->op();
    auto* scope = block_0->scope();

    op_desc.mutable_inputs()->clear();
    op_desc.mutable_outputs()->clear();

    op_desc.SetType("__xpu__block_fuse_op");
    op_desc.SetInput("Input", {matched.at("input")->arg()->name});
    op_desc.SetOutput("Output", {matched.at("block_out1")->arg()->name});
    op_desc.SetOutput("OutputMax", {matched.at("block_out_max1")->arg()->name});

    std::vector<int> op_type;
    std::vector<int> place_x;
    std::vector<int> place_y;
    std::vector<int> place_z;
    std::vector<int> has_bias;
    std::vector<int> filter_dims;
    std::vector<int> conv_strides;
    std::vector<int> conv_paddings;
    std::vector<int> conv_dilations;
    std::vector<int> conv_groups;
    std::vector<int> act_type;
    std::vector<float> act_param;
    std::vector<int> block_lod;
    std::vector<int> has_block_out;

    for (auto name : block_name) {
      auto cur_op_type =
          matched.at(name)->stmt()->op_info()->GetAttr<std::vector<int>>(
              "OpType");
      auto cur_place_x =
          matched.at(name)->stmt()->op_info()->GetAttr<std::vector<int>>(
              "PlaceX");
      auto cur_place_y =
          matched.at(name)->stmt()->op_info()->GetAttr<std::vector<int>>(
              "PlaceY");
      auto cur_place_z =
          matched.at(name)->stmt()->op_info()->GetAttr<std::vector<int>>(
              "PlaceZ");
      auto cur_has_bias =
          matched.at(name)->stmt()->op_info()->GetAttr<std::vector<int>>(
              "HasBias");
      auto cur_filter_dims =
          matched.at(name)->stmt()->op_info()->GetAttr<std::vector<int>>(
              "FilterDims");
      auto cur_conv_strides =
          matched.at(name)->stmt()->op_info()->GetAttr<std::vector<int>>(
              "ConvStrides");
      auto cur_conv_paddings =
          matched.at(name)->stmt()->op_info()->GetAttr<std::vector<int>>(
              "ConvPaddings");
      auto cur_conv_dilations =
          matched.at(name)->stmt()->op_info()->GetAttr<std::vector<int>>(
              "ConvDilations");
      auto cur_conv_groups =
          matched.at(name)->stmt()->op_info()->GetAttr<std::vector<int>>(
              "ConvGroups");
      auto cur_act_type =
          matched.at(name)->stmt()->op_info()->GetAttr<std::vector<int>>(
              "ActType");
      auto cur_act_param =
          matched.at(name)->stmt()->op_info()->GetAttr<std::vector<float>>(
              "ActParam");
      auto cur_block_lod =
          matched.at(name)->stmt()->op_info()->GetAttr<std::vector<int>>(
              "BlockLod");
      auto cur_has_block_out =
          matched.at(name)->stmt()->op_info()->GetAttr<std::vector<int>>(
              "HasBlockOutput");
      op_type.insert(op_type.end(), cur_op_type.begin(), cur_op_type.end());
      place_x.insert(place_x.end(), cur_place_x.begin(), cur_place_x.end());
      place_y.insert(place_y.end(), cur_place_y.begin(), cur_place_y.end());
      place_z.insert(place_z.end(), cur_place_z.begin(), cur_place_z.end());

      has_bias.insert(has_bias.end(), cur_has_bias.begin(), cur_has_bias.end());
      filter_dims.insert(
          filter_dims.end(), cur_filter_dims.begin(), cur_filter_dims.end());

      conv_strides.insert(
          conv_strides.end(), cur_conv_strides.begin(), cur_conv_strides.end());
      conv_paddings.insert(conv_paddings.end(),
                           cur_conv_paddings.begin(),
                           cur_conv_paddings.end());
      conv_dilations.insert(conv_dilations.end(),
                            cur_conv_dilations.begin(),
                            cur_conv_dilations.end());
      conv_groups.insert(
          conv_groups.end(), cur_conv_groups.begin(), cur_conv_groups.end());

      act_type.insert(act_type.end(), cur_act_type.begin(), cur_act_type.end());
      act_param.insert(
          act_param.end(), cur_act_param.begin(), cur_act_param.end());
      block_lod.insert(
          block_lod.end(), cur_block_lod.begin(), cur_block_lod.end());
      has_block_out.insert(has_block_out.end(),
                           cur_has_block_out.begin(),
                           cur_has_block_out.end());
    }

    op_desc.SetAttr("OpType", op_type);
    op_desc.SetAttr("PlaceX", place_x);
    op_desc.SetAttr("PlaceY", place_y);
    op_desc.SetAttr("PlaceZ", place_z);
    op_desc.SetAttr("HasBias", has_bias);
    op_desc.SetAttr("FilterDims", filter_dims);
    op_desc.SetAttr("ConvStrides", conv_strides);
    op_desc.SetAttr("ConvPaddings", conv_paddings);
    op_desc.SetAttr("ConvDilations", conv_dilations);
    op_desc.SetAttr("ConvGroups", conv_groups);
    op_desc.SetAttr("ActType", act_type);
    op_desc.SetAttr("ActParam", act_param);
    op_desc.SetAttr("BlockLod", block_lod);
    op_desc.SetAttr("HasBlockOutput", has_block_out);

    auto* filter0_t = scope->FindMutableTensor(filter_name[0]);
    auto* filter1_t = scope->FindMutableTensor(filter_name[1]);
    int filter0_numel = filter0_t->numel();
    int filter1_numel = filter1_t->numel();
    std::unique_ptr<int16_t[]> encode_filter_int16(
        new int16_t[filter0_numel + filter1_numel]);
    int16_t* filter0_on_host = filter0_t->mutable_data<int16_t>();
    int16_t* filter1_on_host = filter1_t->mutable_data<int16_t>();
    memcpy(encode_filter_int16.get(),
           filter0_on_host,
           filter0_numel * sizeof(int16_t));
    memcpy(encode_filter_int16.get() + filter0_numel,
           filter1_on_host,
           filter1_numel * sizeof(int16_t));
    std::string new_filter_name = "block_" + filter_name[0];
    auto* new_filter_node = graph->NewArgumentNode(new_filter_name);
    new_filter_node->arg()->is_weight = true;
    new_filter_node->arg()->type = LiteType::GetTensorTy(
        TARGET(kHost), PRECISION(kInt16), DATALAYOUT(kNCHW));
    auto* new_filter_t = scope->NewTensor(new_filter_name);
    new_filter_t->Resize({filter0_numel + filter1_numel});
    int16_t* new_filter_ptr = new_filter_t->mutable_data<int16_t>();
    memcpy(new_filter_ptr,
           encode_filter_int16.get(),
           (filter0_numel + filter1_numel) * sizeof(int16_t));
    op_desc.SetInput("Filter", {new_filter_name});

    auto* bias0_t = scope->FindMutableTensor(bias_name[0]);
    auto* bias1_t = scope->FindMutableTensor(bias_name[1]);
    int bias0_numel = bias0_t->numel();
    int bias1_numel = bias1_t->numel();
    std::unique_ptr<float[]> encode_bias(new float[bias0_numel + bias1_numel]);
    float* bias0_on_host = bias0_t->mutable_data<float>();
    float* bias1_on_host = bias1_t->mutable_data<float>();
    memcpy(encode_bias.get(), bias0_on_host, bias0_numel * sizeof(float));
    memcpy(encode_bias.get() + bias0_numel,
           bias1_on_host,
           bias1_numel * sizeof(float));
    std::string new_bias_name = "block_" + bias_name[0];
    auto* new_bias_node = graph->NewArgumentNode(new_bias_name);
    new_bias_node->arg()->is_weight = true;
    new_bias_node->arg()->type = LiteType::GetTensorTy(
        TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kNCHW));
    auto* new_bias_t = scope->NewTensor(new_bias_name);
    new_bias_t->Resize({bias0_numel + bias1_numel});
    float* new_bias_ptr = new_bias_t->mutable_data<float>();
    memcpy(new_bias_ptr,
           encode_bias.get(),
           (bias0_numel + bias1_numel) * sizeof(float));
    op_desc.SetInput("Bias", {new_bias_name});

    auto* filter_max0_t = scope->FindMutableTensor(filter_max_name[0]);
    auto* filter_max1_t = scope->FindMutableTensor(filter_max_name[1]);
    int filter_max0_numel = filter_max0_t->numel();
    int filter_max1_numel = filter_max1_t->numel();
    std::unique_ptr<float[]> encode_filter_max(
        new float[filter_max0_numel + filter_max1_numel]);
    float* filter_max0_on_host = filter_max0_t->mutable_data<float>();
    float* filter_max1_on_host = filter_max1_t->mutable_data<float>();
    memcpy(encode_filter_max.get(),
           filter_max0_on_host,
           filter_max0_numel * sizeof(float));
    memcpy(encode_filter_max.get() + filter_max0_numel,
           filter_max1_on_host,
           filter_max1_numel * sizeof(float));
    std::string new_filter_max_name = "block_" + filter_max_name[0];
    auto* new_filter_max_node = graph->NewArgumentNode(new_filter_max_name);
    new_filter_max_node->arg()->is_weight = true;
    new_filter_max_node->arg()->type = LiteType::GetTensorTy(
        TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kNCHW));
    auto* new_filter_max_t = scope->NewTensor(new_filter_max_name);
    new_filter_max_t->Resize({filter_max0_numel + filter_max1_numel});
    float* new_filter_max_ptr = new_filter_max_t->mutable_data<float>();
    memcpy(new_filter_max_ptr,
           encode_filter_max.get(),
           (filter_max0_numel + filter_max1_numel) * sizeof(float));
    op_desc.SetInput("FilterMax", {new_filter_max_name});

    auto& valid_places = block_0->valid_places();
    auto block_op = LiteOpRegistry::Global().Create(op_desc.Type());
    block_op->Attach(op_desc, scope);
    auto* new_op_node = graph->GraphCreateInstructNode(block_op, valid_places);

    IR_NODE_LINK_TO(matched.at("input"), new_op_node);
    IR_NODE_LINK_TO(new_filter_node, new_op_node);
    IR_NODE_LINK_TO(new_filter_max_node, new_op_node);
    IR_NODE_LINK_TO(new_bias_node, new_op_node);
    IR_NODE_LINK_TO(new_op_node, matched.at("block_out1"));
    IR_NODE_LINK_TO(new_op_node, matched.at("block_out_max1"));
  }
};

}  // namespace fusion

class XPUConsecutiveBlockFusePass : public ProgramPass {
 public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override {
    bool changed = true;
    while (changed) {
      changed = false;
      fusion::XPUConsecutiveBlockFuser fuser;
      changed = fuser(graph.get());
    }
  }
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(__xpu__consecutive_block_fuse_pass,
                  paddle::lite::mir::XPUConsecutiveBlockFusePass)
    .BindTargets({TARGET(kXPU)})
    .BindKernel("__xpu__block_fuse_op");

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

class XPUResBlockReductionFuser : public FuseBase {
 public:
  void BuildPattern() override {
    auto* input = VarNode("input")
                      ->assert_is_op_input("__xpu__conv2d", "Input")
                      ->AsInput();

    auto* left_conv1_weight =
        VarNode("left_conv1_weight")
            ->assert_is_op_input("__xpu__conv2d", "Filter")
            ->AsIntermediate();
    auto* left_conv1_weight_max =
        VarNode("left_conv1_weight_max")
            ->assert_is_op_input("__xpu__conv2d", "FilterMax")
            ->AsIntermediate();
    auto* left_conv1_bias = VarNode("left_conv1_bias")
                                ->assert_is_op_input("__xpu__conv2d", "Bias")
                                ->AsIntermediate();
    auto* left_conv1 = OpNode("left_conv1", "__xpu__conv2d")->AsIntermediate();

    auto* left_conv1_out = VarNode("left_conv1_out")
                               ->assert_is_op_output("__xpu__conv2d", "Output")
                               ->assert_is_op_input("__xpu__conv2d", "Input")
                               ->AsIntermediate();

    auto* left_conv1_out_max =
        VarNode("left_conv1_out_max")
            ->assert_is_op_output("__xpu__conv2d", "OutputMax")
            ->AsIntermediate();

    auto* left_conv2_weight =
        VarNode("left_conv2_weight")
            ->assert_is_op_input("__xpu__conv2d", "Filter")
            ->AsIntermediate();
    auto* left_conv2_weight_max =
        VarNode("left_conv2_weight_max")
            ->assert_is_op_input("__xpu__conv2d", "FilterMax")
            ->AsIntermediate();
    auto* left_conv2_bias = VarNode("left_conv2_bias")
                                ->assert_is_op_input("__xpu__conv2d", "Bias")
                                ->AsIntermediate();

    auto* left_conv2 = OpNode("left_conv2", "__xpu__conv2d")->AsIntermediate();

    auto* left_conv2_out = VarNode("left_conv2_out")
                               ->assert_is_op_output("__xpu__conv2d", "Output")
                               ->assert_is_op_input("__xpu__conv2d", "Input")
                               ->AsIntermediate();

    auto* left_conv2_out_max =
        VarNode("left_conv2_out_max")
            ->assert_is_op_output("__xpu__conv2d", "OutputMax")
            ->AsIntermediate();

    auto* right_conv1_weight =
        VarNode("right_conv1_weight")
            ->assert_is_op_input("__xpu__conv2d", "Filter")
            ->AsIntermediate();
    auto* right_conv1_weight_max =
        VarNode("right_conv1_weight_max")
            ->assert_is_op_input("__xpu__conv2d", "FilterMax")
            ->AsIntermediate();
    auto* right_conv1_bias = VarNode("right_conv1_bias")
                                 ->assert_is_op_input("__xpu__conv2d", "Bias")
                                 ->AsIntermediate();
    auto* right_conv1 =
        OpNode("right_conv1", "__xpu__conv2d")->AsIntermediate();

    auto* right_conv1_out = VarNode("right_conv1_out")
                                ->assert_is_op_output("__xpu__conv2d", "Output")
                                ->assert_is_op_input("__xpu__conv2d", "Branch")
                                ->AsIntermediate();

    auto* right_conv1_out_max =
        VarNode("right_conv1_out_max")
            ->assert_is_op_output("__xpu__conv2d", "OutputMax")
            ->AsIntermediate();

    auto* left_conv3_weight =
        VarNode("left_conv3_weight")
            ->assert_is_op_input("__xpu__conv2d", "Filter")
            ->AsIntermediate();
    auto* left_conv3_weight_max =
        VarNode("left_conv3_weight_max")
            ->assert_is_op_input("__xpu__conv2d", "FilterMax")
            ->AsIntermediate();
    auto* left_conv3_bias = VarNode("left_conv3_bias")
                                ->assert_is_op_input("__xpu__conv2d", "Bias")
                                ->AsIntermediate();
    auto* left_conv3 = OpNode("left_conv3", "__xpu__conv2d")->AsIntermediate();

    auto* left_conv3_out = VarNode("left_conv3_out")
                               ->assert_is_op_output("__xpu__conv2d", "Output")
                               ->AsOutput();

    auto* left_conv3_out_max =
        VarNode("left_conv3_out_max")
            ->assert_is_op_output("__xpu__conv2d", "OutputMax")
            ->AsOutput();

    *input >> *left_conv1 >> *left_conv1_out >> *left_conv2 >>
        *left_conv2_out >> *left_conv3;
    *input >> *right_conv1 >> *right_conv1_out >> *left_conv3;
    *left_conv3 >> *left_conv3_out;

    *left_conv1_weight >> *left_conv1;
    *left_conv1_weight_max >> *left_conv1;
    *left_conv1_bias >> *left_conv1;
    *left_conv1 >> *left_conv1_out_max;

    *left_conv2_weight >> *left_conv2;
    *left_conv2_weight_max >> *left_conv2;
    *left_conv2_bias >> *left_conv2;
    *left_conv2 >> *left_conv2_out_max;

    *right_conv1_weight >> *right_conv1;
    *right_conv1_weight_max >> *right_conv1;
    *right_conv1_bias >> *right_conv1;
    *right_conv1 >> *right_conv1_out_max;

    *left_conv3_weight >> *left_conv3;
    *left_conv3_weight_max >> *left_conv3;
    *left_conv3_bias >> *left_conv3;
    *left_conv3 >> *left_conv3_out_max;
  }
  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override {
    std::vector<std::string> conv_name{
        "left_conv1", "left_conv2", "right_conv1", "left_conv3"};

    std::vector<std::string> filter_name{
        matched.at("left_conv1_weight")->arg()->name,
        matched.at("left_conv2_weight")->arg()->name,
        matched.at("right_conv1_weight")->arg()->name,
        matched.at("left_conv3_weight")->arg()->name};

    std::vector<std::string> bias_name = {
        matched.at("left_conv1_bias")->arg()->name,
        matched.at("left_conv2_bias")->arg()->name,
        matched.at("right_conv1_bias")->arg()->name,
        matched.at("left_conv3_bias")->arg()->name};

    std::vector<std::string> filter_max_name{
        matched.at("left_conv1_weight_max")->arg()->name,
        matched.at("left_conv2_weight_max")->arg()->name,
        matched.at("right_conv1_weight_max")->arg()->name,
        matched.at("left_conv3_weight_max")->arg()->name};

    auto op_desc = *matched.at("left_conv1")->stmt()->op_info();
    auto left_conv1 = matched.at("left_conv1")->stmt()->op();
    auto* scope = left_conv1->scope();

    op_desc.mutable_inputs()->clear();
    op_desc.mutable_outputs()->clear();

    op_desc.SetType("__xpu__block_fuse_op");
    op_desc.SetInput("Input", {matched.at("input")->arg()->name});
    op_desc.SetOutput("Output", {matched.at("left_conv3_out")->arg()->name});
    op_desc.SetOutput("OutputMax",
                      {matched.at("left_conv3_out_max")->arg()->name});

    static const int PX = 0;
    static const int P1 = 1;
    static const int P2 = 2;
    static const int P3 = 3;
    // static const int P4 = 4;
    static const int PNONE = 9;
    static const int PY = 10;

    std::vector<int> op_type{0, 0, 0, 0};
    std::vector<int> place_x{PX, P1, PX, P2};
    std::vector<int> place_y{PNONE, PNONE, PNONE, P3};
    std::vector<int> place_z{P1, P2, P3, PY};
    std::vector<int> block_lod{4};
    std::vector<int> has_block_output{0};
    std::vector<int> has_bias{1, 1, 1, 1};

    std::vector<int> filter_dims;
    std::vector<int> conv_strides;
    std::vector<int> conv_paddings;
    std::vector<int> conv_dilations;
    std::vector<int> conv_groups;
    std::vector<int> act_type;
    std::vector<float> act_param;

    std::vector<int> encode_filter_size{0};
    std::vector<int> encode_bias_size{0};
    std::vector<int> encode_filter_max_size{0};
    for (auto name : conv_name) {
      auto cur_filter_dims =
          matched.at(name)->stmt()->op_info()->GetAttr<std::vector<int>>(
              "filter_dims");
      auto cur_strides =
          matched.at(name)->stmt()->op_info()->GetAttr<std::vector<int>>(
              "strides");
      auto cur_paddings =
          matched.at(name)->stmt()->op_info()->GetAttr<std::vector<int>>(
              "paddings");
      auto cur_dilations =
          matched.at(name)->stmt()->op_info()->GetAttr<std::vector<int>>(
              "dilations");
      auto cur_groups =
          matched.at(name)->stmt()->op_info()->GetAttr<int>("groups");
      auto cur_act_type =
          matched.at(name)->stmt()->op_info()->GetAttr<int>("act_type");
      auto cur_act_param =
          matched.at(name)->stmt()->op_info()->GetAttr<float>("act_param");
      filter_dims.insert(
          filter_dims.end(), cur_filter_dims.begin(), cur_filter_dims.end());

      encode_filter_size.push_back(encode_filter_size.back() +
                                   cur_filter_dims[0] * cur_filter_dims[1] *
                                       cur_filter_dims[2] * cur_filter_dims[3]);
      encode_bias_size.push_back(encode_bias_size.back() + cur_filter_dims[0]);
      encode_filter_max_size.push_back(encode_filter_max_size.back() + 4);

      conv_strides.insert(
          conv_strides.end(), cur_strides.begin(), cur_strides.end());
      if (cur_paddings.size() == 2) {
        for (size_t i = 0; i < cur_strides.size(); ++i) {
          int copy_pad = *(cur_paddings.begin() + 2 * i);
          cur_paddings.insert(cur_paddings.begin() + 2 * i + 1, copy_pad);
        }
      } else {
        if (cur_paddings.size() != 4) {
          LOG(FATAL)
              << "Paddings size should be the same or twice as the input size.";
        }
      }
      conv_paddings.insert(
          conv_paddings.end(), cur_paddings.begin(), cur_paddings.end());
      conv_dilations.insert(
          conv_dilations.end(), cur_dilations.begin(), cur_dilations.end());
      conv_groups.push_back(cur_groups);
      act_type.push_back(cur_act_type);
      act_param.push_back(cur_act_param);
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
    op_desc.SetAttr("HasBlockOutput", has_block_output);

    std::unique_ptr<int16_t[]> encode_filter_int16(
        new int16_t[encode_filter_size.back()]);
    for (int i = 0; i < filter_name.size(); i++) {
      auto* filter_t = scope->FindMutableTensor(filter_name[i]);
      int16_t* filter_on_host = filter_t->mutable_data<int16_t>();
      memcpy(encode_filter_int16.get() + encode_filter_size[i],
             filter_on_host,
             (encode_filter_size[i + 1] - encode_filter_size[i]) *
                 sizeof(int16_t));
    }
    std::string new_filter_name = "block_" + filter_name[0];
    auto* new_filter_node = graph->NewArgumentNode(new_filter_name);
    new_filter_node->arg()->is_weight = true;
    new_filter_node->arg()->type = LiteType::GetTensorTy(
        TARGET(kHost), PRECISION(kInt16), DATALAYOUT(kNCHW));
    auto* new_filter_t = scope->NewTensor(new_filter_name);
    new_filter_t->Resize({encode_filter_size.back()});
    int16_t* new_filter_ptr = new_filter_t->mutable_data<int16_t>();
    memcpy(new_filter_ptr,
           encode_filter_int16.get(),
           encode_filter_size.back() * sizeof(int16_t));
    op_desc.SetInput("Filter", {new_filter_name});

    std::unique_ptr<float[]> encode_bias(new float[encode_bias_size.back()]);
    for (int i = 0; i < bias_name.size(); i++) {
      auto* bias_t = scope->FindMutableTensor(bias_name[i]);
      float* bias_on_host = bias_t->mutable_data<float>();
      memcpy(encode_bias.get() + encode_bias_size[i],
             bias_on_host,
             (encode_bias_size[i + 1] - encode_bias_size[i]) * sizeof(float));
    }
    std::string new_bias_name = "block_" + bias_name[0];
    auto* new_bias_node = graph->NewArgumentNode(new_bias_name);
    new_bias_node->arg()->is_weight = true;
    new_bias_node->arg()->type = LiteType::GetTensorTy(
        TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kNCHW));
    auto* new_bias_t = scope->NewTensor(new_bias_name);
    new_bias_t->Resize({encode_bias_size.back()});
    float* new_bias_ptr = new_bias_t->mutable_data<float>();
    memcpy(new_bias_ptr,
           encode_bias.get(),
           encode_bias_size.back() * sizeof(float));
    op_desc.SetInput("Bias", {new_bias_name});

    std::unique_ptr<float[]> encode_filter_max(
        new float[encode_filter_max_size.back()]);
    for (int i = 0; i < filter_max_name.size(); i++) {
      auto* filter_max_t = scope->FindMutableTensor(filter_max_name[i]);
      float* filter_max_on_host = filter_max_t->mutable_data<float>();
      memcpy(encode_filter_max.get() + encode_filter_max_size[i],
             filter_max_on_host,
             (encode_filter_max_size[i + 1] - encode_filter_max_size[i]) *
                 sizeof(float));
    }
    std::string new_filter_max_name = "block_" + filter_max_name[0];
    auto* new_filter_max_node = graph->NewArgumentNode(new_filter_max_name);
    new_filter_max_node->arg()->is_weight = true;
    new_filter_max_node->arg()->type = LiteType::GetTensorTy(
        TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kNCHW));
    auto* new_filter_max_t = scope->NewTensor(new_filter_max_name);
    new_filter_max_t->Resize({encode_filter_max_size.back()});
    float* new_filter_max_ptr = new_filter_max_t->mutable_data<float>();
    memcpy(new_filter_max_ptr,
           encode_filter_max.get(),
           encode_filter_max_size.back() * sizeof(float));
    op_desc.SetInput("FilterMax", {new_filter_max_name});

    auto& valid_places = left_conv1->valid_places();
    auto resblock_reduction_op =
        LiteOpRegistry::Global().Create(op_desc.Type());
    resblock_reduction_op->Attach(op_desc, scope);
    auto* new_op_node =
        graph->GraphCreateInstructNode(resblock_reduction_op, valid_places);

    IR_NODE_LINK_TO(matched.at("input"), new_op_node);
    IR_NODE_LINK_TO(new_filter_node, new_op_node);
    IR_NODE_LINK_TO(new_filter_max_node, new_op_node);
    IR_NODE_LINK_TO(new_bias_node, new_op_node);

    IR_NODE_LINK_TO(new_op_node, matched.at("left_conv3_out"));
    IR_NODE_LINK_TO(new_op_node, matched.at("left_conv3_out_max"));
  }
};

}  // namespace fusion

class XPUResBlockReductionFusePass : public ProgramPass {
 public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override {
    fusion::XPUResBlockReductionFuser fuser;
    fuser(graph.get());
  }
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(__xpu__resblock_reduction_fuse_pass,
                  paddle::lite::mir::XPUResBlockReductionFusePass)
    .BindTargets({TARGET(kXPU)})
    .BindKernel("__xpu__block_fuse_op");

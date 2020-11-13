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

#define STR1(R) #R
#define STR2(R) STR1(R)

#define CONV_PATTERN(num)                                                      \
  auto* weight_##num = VarNode(STR2(weight_##num))                             \
                           ->assert_is_op_input("__xpu__conv2d", "Filter")     \
                           ->AsInput();                                        \
  auto* weight_max_##num =                                                     \
      VarNode(STR2(weight_max_##num))                                          \
          ->assert_is_op_input("__xpu__conv2d", "FilterMax")                   \
          ->AsInput();                                                         \
  auto* bias_##num = VarNode(STR2(bias_##num))                                 \
                         ->assert_is_op_input("__xpu__conv2d", "Bias")         \
                         ->AsInput();                                          \
  auto* conv_##num =                                                           \
      OpNode(STR2(conv_##num), "__xpu__conv2d")->AsIntermediate();             \
  auto* conv_out_##num = VarNode(STR2(conv_out_##num))                         \
                             ->assert_is_op_output("__xpu__conv2d", "Output"); \
  auto* conv_out_max_##num =                                                   \
      VarNode(STR2(conv_out_max_##num))                                        \
          ->assert_is_op_output("__xpu__conv2d", "OutputMax");

#define CONV_CONNECT(num)           \
  *weight_##num >> *conv_##num;     \
  *weight_max_##num >> *conv_##num; \
  *bias_##num >> *conv_##num;       \
  *conv_##num >> *conv_out_max_##num;

class XPUConsecutiveConv2dFuser : public FuseBase {
 public:
  explicit XPUConsecutiveConv2dFuser(int conv_num) { conv_num_ = conv_num; }
  void BuildPattern() override {
    auto* input = VarNode("input")
                      ->assert_is_op_input("__xpu__conv2d", "Input")
                      ->AsInput();
    CONV_PATTERN(0);
    conv_out_0->assert_is_op_input("__xpu__conv2d", "Input")->AsIntermediate();
    conv_out_max_0->AsIntermediate();

    CONV_PATTERN(1);
    conv_out_1->assert_is_op_input("__xpu__conv2d", "Input")->AsIntermediate();
    conv_out_max_1->AsIntermediate();

    CONV_PATTERN(2);

    if (conv_num_ == 3) {
      conv_out_2->AsOutput();
      conv_out_max_2->AsOutput();
      *input >> *conv_0 >> *conv_out_0 >> *conv_1 >> *conv_out_1 >> *conv_2 >>
          *conv_out_2;
      CONV_CONNECT(0);
      CONV_CONNECT(1);
      CONV_CONNECT(2);
    } else if (conv_num_ == 4) {
      conv_out_2->assert_is_op_input("__xpu__conv2d", "Input")
          ->AsIntermediate();
      conv_out_max_2->AsIntermediate();
      CONV_PATTERN(3);
      conv_out_3->AsOutput();
      conv_out_max_3->AsOutput();
      *input >> *conv_0 >> *conv_out_0 >> *conv_1 >> *conv_out_1 >> *conv_2 >>
          *conv_out_2 >> *conv_3 >> *conv_out_3;
      CONV_CONNECT(0);
      CONV_CONNECT(1);
      CONV_CONNECT(2);
      CONV_CONNECT(3);
    } else {
      conv_out_2->assert_is_op_input("__xpu__conv2d", "Input")
          ->AsIntermediate();
      conv_out_max_2->AsIntermediate();
      CONV_PATTERN(3);
      conv_out_3->assert_is_op_input("__xpu__conv2d", "Input")
          ->AsIntermediate();
      conv_out_max_3->AsIntermediate();
      CONV_PATTERN(4);
      conv_out_4->AsOutput();
      conv_out_max_4->AsOutput();
      *input >> *conv_0 >> *conv_out_0 >> *conv_1 >> *conv_out_1 >> *conv_2 >>
          *conv_out_2 >> *conv_3 >> *conv_out_3 >> *conv_4 >> *conv_out_4;
      CONV_CONNECT(0);
      CONV_CONNECT(1);
      CONV_CONNECT(2);
      CONV_CONNECT(3);
      CONV_CONNECT(4);
    }
  }
  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override {
    std::vector<std::string> conv_name{"conv_0", "conv_1", "conv_2"};

    std::vector<std::string> filter_name{matched.at("weight_0")->arg()->name,
                                         matched.at("weight_1")->arg()->name,
                                         matched.at("weight_2")->arg()->name};

    std::vector<std::string> bias_name = {matched.at("bias_0")->arg()->name,
                                          matched.at("bias_1")->arg()->name,
                                          matched.at("bias_2")->arg()->name};

    std::vector<std::string> filter_max_name{
        matched.at("weight_max_0")->arg()->name,
        matched.at("weight_max_1")->arg()->name,
        matched.at("weight_max_2")->arg()->name};

    if (conv_num_ > 3) {
      conv_name.push_back("conv_3");
      filter_name.push_back(matched.at("weight_3")->arg()->name);
      bias_name.push_back(matched.at("bias_3")->arg()->name);
      filter_max_name.push_back(matched.at("weight_max_3")->arg()->name);
    }
    if (conv_num_ > 4) {
      conv_name.push_back("conv_4");
      filter_name.push_back(matched.at("weight_4")->arg()->name);
      bias_name.push_back(matched.at("bias_4")->arg()->name);
      filter_max_name.push_back(matched.at("weight_max_4")->arg()->name);
    }

    auto op_desc = *matched.at("conv_0")->stmt()->op_info();
    auto conv_0 = matched.at("conv_0")->stmt()->op();
    auto* scope = conv_0->scope();

    op_desc.mutable_inputs()->clear();
    op_desc.mutable_outputs()->clear();

    op_desc.SetType("__xpu__block_fuse_op");
    op_desc.SetInput("Input", {matched.at("input")->arg()->name});
    op_desc.SetInput("Filter", filter_name);
    op_desc.SetInput("Bias", bias_name);
    op_desc.SetInput("FilterMax", filter_max_name);

    if (conv_num_ == 3) {
      op_desc.SetOutput("Output", {matched.at("conv_out_2")->arg()->name});
      op_desc.SetOutput("OutputMax",
                        {matched.at("conv_out_max_2")->arg()->name});
    } else if (conv_num_ == 4) {
      op_desc.SetOutput("Output", {matched.at("conv_out_3")->arg()->name});
      op_desc.SetOutput("OutputMax",
                        {matched.at("conv_out_max_3")->arg()->name});
    } else {
      op_desc.SetOutput("Output", {matched.at("conv_out_4")->arg()->name});
      op_desc.SetOutput("OutputMax",
                        {matched.at("conv_out_max_4")->arg()->name});
    }

    static const int PX = 0;
    static const int P1 = 1;
    static const int P2 = 2;
    static const int P3 = 3;
    static const int P4 = 4;
    static const int PNONE = 9;
    static const int PY = 10;

    std::vector<int> place_x{PX, P1, P2};
    std::vector<int> place_z{P1, P2};
    if (conv_num_ == 3) {
      place_z.push_back(PY);
    } else if (conv_num_ == 4) {
      place_x.push_back(P3);
      place_z.push_back(P3);
      place_z.push_back(PY);
    } else {
      place_x.push_back(P3);
      place_x.push_back(P4);
      place_z.push_back(P3);
      place_z.push_back(P4);
      place_z.push_back(PY);
    }

    std::vector<int> place_y;
    std::vector<int> block_lod{conv_num_};
    std::vector<int> has_block_output{0};
    std::vector<int> has_bias;
    std::vector<int> op_type;
    for (int i = 0; i < conv_num_; i++) {
      has_bias.push_back(1);
      op_type.push_back(0);
      place_y.push_back(PNONE);
    }

    std::vector<int> filter_dims;
    std::vector<int> conv_strides;
    std::vector<int> conv_paddings;
    std::vector<int> conv_dilations;
    std::vector<int> conv_groups;
    std::vector<int> act_type;
    std::vector<float> act_param;

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

    auto& valid_places = conv_0->valid_places();
    auto block_op = LiteOpRegistry::Global().Create(op_desc.Type());
    block_op->Attach(op_desc, scope);
    auto* new_op_node = graph->GraphCreateInstructNode(block_op, valid_places);

    std::vector<std::string> froms{
        "input",
        "weight_0",
        "weight_1",
        "weight_2",
        "weight_max_0",
        "weight_max_1",
        "weight_max_2",
        "bias_0",
        "bias_1",
        "bias_2",
    };
    if (conv_num_ > 3) {
      froms.push_back("weight_3");
      froms.push_back("weight_max_3");
      froms.push_back("bias_3");
    }
    if (conv_num_ > 4) {
      froms.push_back("weight_4");
      froms.push_back("weight_max_4");
      froms.push_back("bias_4");
    }
    for (auto& from : froms) {
      IR_NODE_LINK_TO(matched.at(from), new_op_node);
    }
    if (conv_num_ == 3) {
      IR_NODE_LINK_TO(new_op_node, matched.at("conv_out_2"));
      IR_NODE_LINK_TO(new_op_node, matched.at("conv_out_max_2"));
    } else if (conv_num_ == 4) {
      IR_NODE_LINK_TO(new_op_node, matched.at("conv_out_3"));
      IR_NODE_LINK_TO(new_op_node, matched.at("conv_out_max_3"));
    } else {
      IR_NODE_LINK_TO(new_op_node, matched.at("conv_out_4"));
      IR_NODE_LINK_TO(new_op_node, matched.at("conv_out_max_4"));
    }
  }

 private:
  int conv_num_;
};
#undef CONV_CONNECT
#undef CONV_PATTERN
#undef STR1
#undef STR2
}  // namespace fusion

class XPUConsecutiveConv2dFusePass : public ProgramPass {
 public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override {
    // only support conv_num in {5, 4, 3}
    for (auto conv_num : {5, 4, 3}) {
      fusion::XPUConsecutiveConv2dFuser fuser(conv_num);
      fuser(graph.get());
    }
  }
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(__xpu__consecutive_conv2d_fuse_pass,
                  paddle::lite::mir::XPUConsecutiveConv2dFusePass)
    .BindTargets({TARGET(kXPU)})
    .BindKernel("__xpu__block_fuse_op");

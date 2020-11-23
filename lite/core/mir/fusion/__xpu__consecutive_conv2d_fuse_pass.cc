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

#define CONV_PATTERN(num)                                                   \
  weight_##num = VarNode(STR2(weight_##num))                                \
                     ->assert_is_op_input("__xpu__conv2d", "Filter")        \
                     ->AsIntermediate();                                    \
  weight_max_##num = VarNode(STR2(weight_max_##num))                        \
                         ->assert_is_op_input("__xpu__conv2d", "FilterMax") \
                         ->AsIntermediate();                                \
  bias_##num = VarNode(STR2(bias_##num))                                    \
                   ->assert_is_op_input("__xpu__conv2d", "Bias")            \
                   ->AsIntermediate();                                      \
  conv_##num = OpNode(STR2(conv_##num), "__xpu__conv2d")->AsIntermediate(); \
  conv_out_##num = VarNode(STR2(conv_out_##num))                            \
                       ->assert_is_op_output("__xpu__conv2d", "Output");    \
  conv_out_max_##num =                                                      \
      VarNode(STR2(conv_out_max_##num))                                     \
          ->assert_is_op_output("__xpu__conv2d", "OutputMax");

#define CONV_CONNECT(num)           \
  *weight_##num >> *conv_##num;     \
  *weight_max_##num >> *conv_##num; \
  *bias_##num >> *conv_##num;       \
  *conv_##num >> *conv_out_max_##num;

#define NODE_INIT(num)                \
  PMNode* weight_##num = nullptr;     \
  PMNode* weight_max_##num = nullptr; \
  PMNode* bias_##num = nullptr;       \
  PMNode* conv_##num = nullptr;       \
  PMNode* conv_out_##num = nullptr;   \
  PMNode* conv_out_max_##num = nullptr;

class XPUConsecutiveConv2dFuser : public FuseBase {
 public:
  explicit XPUConsecutiveConv2dFuser(int conv_num) { conv_num_ = conv_num; }
  void BuildPattern() override {
    auto* input = VarNode("input")
                      ->assert_is_op_input("__xpu__conv2d", "Input")
                      ->AsInput();
    NODE_INIT(0);
    NODE_INIT(1);
    NODE_INIT(2);
    NODE_INIT(3);
    NODE_INIT(4);

    CONV_PATTERN(0);
    *input >> *conv_0 >> *conv_out_0;
    if (conv_num_ > 1) {
      CONV_PATTERN(1);
      conv_out_0->assert_is_op_input("__xpu__conv2d", "Input")
          ->AsIntermediate();
      conv_out_max_0->AsIntermediate();
      *conv_out_0 >> *conv_1 >> *conv_out_1;
    }
    if (conv_num_ > 2) {
      CONV_PATTERN(2);
      conv_out_1->assert_is_op_input("__xpu__conv2d", "Input")
          ->AsIntermediate();
      conv_out_max_1->AsIntermediate();
      *conv_out_1 >> *conv_2 >> *conv_out_2;
    }
    if (conv_num_ > 3) {
      CONV_PATTERN(3);
      *conv_out_2 >> *conv_3 >> *conv_out_3;
      conv_out_2->assert_is_op_input("__xpu__conv2d", "Input")
          ->AsIntermediate();
      conv_out_max_2->AsIntermediate();
    }
    if (conv_num_ > 4) {
      CONV_PATTERN(4);
      *conv_out_3 >> *conv_4 >> *conv_out_4;
      conv_out_3->assert_is_op_input("__xpu__conv2d", "Input")
          ->AsIntermediate();
      conv_out_max_3->AsIntermediate();
    }
    if (conv_num_ == 1) {
      CONV_CONNECT(0);
      conv_out_0->AsOutput();
      conv_out_max_0->AsOutput();
    } else if (conv_num_ == 2) {
      CONV_CONNECT(0);
      CONV_CONNECT(1);
      conv_out_1->AsOutput();
      conv_out_max_1->AsOutput();
    } else if (conv_num_ == 3) {
      CONV_CONNECT(0);
      CONV_CONNECT(1);
      CONV_CONNECT(2);
      conv_out_2->AsOutput();
      conv_out_max_2->AsOutput();
    } else if (conv_num_ == 4) {
      CONV_CONNECT(0);
      CONV_CONNECT(1);
      CONV_CONNECT(2);
      CONV_CONNECT(3);
      conv_out_3->AsOutput();
      conv_out_max_3->AsOutput();
    } else {
      CONV_CONNECT(0);
      CONV_CONNECT(1);
      CONV_CONNECT(2);
      CONV_CONNECT(3);
      CONV_CONNECT(4);
      conv_out_4->AsOutput();
      conv_out_max_4->AsOutput();
    }
  }
  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override {
    std::vector<std::string> conv_name{"conv_0"};
    if (conv_num_ > 1) {
      conv_name.push_back("conv_1");
    }
    if (conv_num_ > 2) {
      conv_name.push_back("conv_2");
    }
    if (conv_num_ > 3) {
      conv_name.push_back("conv_3");
    }
    if (conv_num_ > 4) {
      conv_name.push_back("conv_4");
    }

    std::vector<std::string> filter_name{matched.at("weight_0")->arg()->name};
    if (conv_num_ > 1) {
      filter_name.push_back(matched.at("weight_1")->arg()->name);
    }
    if (conv_num_ > 2) {
      filter_name.push_back(matched.at("weight_2")->arg()->name);
    }
    if (conv_num_ > 3) {
      filter_name.push_back(matched.at("weight_3")->arg()->name);
    }
    if (conv_num_ > 4) {
      filter_name.push_back(matched.at("weight_4")->arg()->name);
    }

    std::vector<std::string> bias_name = {matched.at("bias_0")->arg()->name};
    if (conv_num_ > 1) {
      bias_name.push_back(matched.at("bias_1")->arg()->name);
    }
    if (conv_num_ > 2) {
      bias_name.push_back(matched.at("bias_2")->arg()->name);
    }
    if (conv_num_ > 3) {
      bias_name.push_back(matched.at("bias_3")->arg()->name);
    }
    if (conv_num_ > 4) {
      bias_name.push_back(matched.at("bias_4")->arg()->name);
    }

    std::vector<std::string> filter_max_name{
        matched.at("weight_max_0")->arg()->name};
    if (conv_num_ > 1) {
      filter_max_name.push_back(matched.at("weight_max_1")->arg()->name);
    }
    if (conv_num_ > 2) {
      filter_max_name.push_back(matched.at("weight_max_2")->arg()->name);
    }
    if (conv_num_ > 3) {
      filter_max_name.push_back(matched.at("weight_max_3")->arg()->name);
    }
    if (conv_num_ > 4) {
      filter_max_name.push_back(matched.at("weight_max_4")->arg()->name);
    }

    auto op_desc = *matched.at("conv_0")->stmt()->op_info();
    auto conv_0 = matched.at("conv_0")->stmt()->op();
    auto* scope = conv_0->scope();

    op_desc.mutable_inputs()->clear();
    op_desc.mutable_outputs()->clear();

    op_desc.SetType("__xpu__block_fuse_op");
    op_desc.SetInput("Input", {matched.at("input")->arg()->name});
    if (conv_num_ == 1) {
      op_desc.SetOutput("Output", {matched.at("conv_out_0")->arg()->name});
      op_desc.SetOutput("OutputMax",
                        {matched.at("conv_out_max_0")->arg()->name});
    } else if (conv_num_ == 2) {
      op_desc.SetOutput("Output", {matched.at("conv_out_1")->arg()->name});
      op_desc.SetOutput("OutputMax",
                        {matched.at("conv_out_max_1")->arg()->name});
    } else if (conv_num_ == 3) {
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

    std::vector<int> place_x{PX};
    std::vector<int> place_z{P1};
    if (conv_num_ > 1) {
      place_x.push_back(P1);
      place_z.push_back(P2);
    }
    if (conv_num_ > 2) {
      place_x.push_back(P2);
      place_z.push_back(P3);
    }
    if (conv_num_ > 3) {
      place_x.push_back(P3);
      place_z.push_back(P4);
    }
    if (conv_num_ > 4) {
      place_x.push_back(P4);
      place_z.push_back(PY);
    }
    place_z[conv_num_ - 1] = PY;

    std::vector<int> place_y(conv_num_, PNONE);
    std::vector<int> block_lod{conv_num_};
    std::vector<int> has_block_output{0};
    std::vector<int> has_bias(conv_num_, 1);
    std::vector<int> op_type(conv_num_, 0);

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

    auto& valid_places = conv_0->valid_places();
    auto block_op = LiteOpRegistry::Global().Create(op_desc.Type());
    block_op->Attach(op_desc, scope);
    auto* new_op_node = graph->GraphCreateInstructNode(block_op, valid_places);

    IR_NODE_LINK_TO(matched.at("input"), new_op_node);
    IR_NODE_LINK_TO(new_filter_node, new_op_node);
    IR_NODE_LINK_TO(new_filter_max_node, new_op_node);
    IR_NODE_LINK_TO(new_bias_node, new_op_node);

    if (conv_num_ == 1) {
      IR_NODE_LINK_TO(new_op_node, matched.at("conv_out_0"));
      IR_NODE_LINK_TO(new_op_node, matched.at("conv_out_max_0"));
    } else if (conv_num_ == 2) {
      IR_NODE_LINK_TO(new_op_node, matched.at("conv_out_1"));
      IR_NODE_LINK_TO(new_op_node, matched.at("conv_out_max_1"));
    } else if (conv_num_ == 3) {
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
#undef NODE_INIT
#undef CONV_CONNECT
#undef CONV_PATTERN
#undef STR1
#undef STR2
}  // namespace fusion

class XPUConsecutiveConv2dFusePass : public ProgramPass {
 public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override {
    // only support conv_num in {5, 4, 3}
    for (auto conv_num : {5, 4, 3, 2, 1}) {
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

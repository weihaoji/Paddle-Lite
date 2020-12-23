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

class XPUConv2dConcatPool2dFuser : public FuseBase {
 public:
  explicit XPUConv2dConcatPool2dFuser(bool has_mid_conv, bool has_pool2d) {
    has_mid_conv_ = has_mid_conv;
    has_pool2d_ = has_pool2d;
  }
  static bool Pool2dCheck(const Node* x) {
    if (x && x->IsStmt()) {
      auto* op_info = x->stmt()->op_info();
      if (op_info->HasAttr("adaptive")) {
        auto attr_type = op_info->GetAttrType("adaptive");
        if (attr_type == paddle::lite::OpDescAPI::AttrType::BOOLEAN &&
            op_info->GetAttr<bool>("adaptive") == true) {
          return false;
        }
      }
      if (op_info->HasAttr("padding_algorithm") &&
          op_info->GetAttrType("padding_algorithm") ==
              paddle::lite::OpDescAPI::AttrType::STRING &&
          op_info->GetAttr<std::string>("padding_algorithm") == "SAME") {
        return false;
      }
    }
    return true;
  }

  void BuildPattern() override {
    auto* input = VarNode("input")
                      ->assert_is_op_input("__xpu__conv2d", "Input")
                      ->AsInput();
    auto* left_weight0 = VarNode("left_weight0")
                             ->assert_is_op_input("__xpu__conv2d", "Filter")
                             ->assert_is_persistable_var()
                             ->AsInput();
    auto* left_weight_max0 =
        VarNode("left_weight_max0")
            ->assert_is_op_input("__xpu__conv2d", "FilterMax")
            ->assert_is_persistable_var()
            ->AsInput();
    auto* left_bias0 = VarNode("left_bias0")
                           ->assert_is_op_input("__xpu__conv2d", "Bias")
                           ->assert_is_persistable_var()
                           ->AsInput();
    auto* left_xpu_conv0 =
        OpNode("left_xpu_conv0", "__xpu__conv2d")
            ->assert_op_attr_satisfied<bool>(
                "has_branch", [](const bool& attr) { return attr == false; })
            ->assert_op_attr_satisfied<int>(
                "act_type",
                [](const int& attr) {
                  return attr == 1 || attr == 2; /* support relu and sigmoid */
                })
            ->AsIntermediate();
    auto* left_conv_out0 = VarNode("left_conv_out0")
                               ->assert_is_op_output("__xpu__conv2d", "Output")
                               ->assert_is_op_nth_input("concat", "X", 0)
                               ->AsIntermediate();
    auto* left_conv_out_max0 =
        VarNode("left_conv_out_max0")
            ->assert_is_op_output("__xpu__conv2d", "OutputMax")
            ->AsIntermediate();
    PMNode* right_weight0 = nullptr;
    PMNode* right_weight_max0 = nullptr;
    PMNode* right_bias0 = nullptr;
    PMNode* right_xpu_conv0 = nullptr;
    PMNode* right_conv_out0 = nullptr;
    PMNode* right_conv_out_max0 = nullptr;
    if (has_mid_conv_) {
      right_weight0 = VarNode("right_weight0")
                          ->assert_is_op_input("__xpu__conv2d", "Filter")
                          ->assert_is_persistable_var()
                          ->AsInput();
      right_weight_max0 = VarNode("right_weight_max0")
                              ->assert_is_op_input("__xpu__conv2d", "FilterMax")
                              ->assert_is_persistable_var()
                              ->AsInput();
      right_bias0 = VarNode("right_bias0")
                        ->assert_is_op_input("__xpu__conv2d", "Bias")
                        ->assert_is_persistable_var()
                        ->AsInput();
      right_xpu_conv0 =
          OpNode("right_xpu_conv0", "__xpu__conv2d")
              ->assert_op_attr_satisfied<bool>(
                  "has_branch", [](const bool& attr) { return attr == false; })
              ->assert_op_attr_satisfied<int>(
                  "act_type",
                  [](const int& attr) {
                    return attr == 1 ||
                           attr == 2; /* support relu and sigmoid */
                  })
              ->AsIntermediate();
      right_conv_out0 = VarNode("right_conv_out0")
                            ->assert_is_op_output("__xpu__conv2d", "Output")
                            ->assert_is_op_input("__xpu__conv2d", "Input")
                            ->AsIntermediate();
      right_conv_out_max0 =
          VarNode("right_conv_out_max0")
              ->assert_is_op_output("__xpu__conv2d", "OutputMax")
              ->AsIntermediate();
    }
    auto* right_weight1 = VarNode("right_weight1")
                              ->assert_is_op_input("__xpu__conv2d", "Filter")
                              ->assert_is_persistable_var()
                              ->AsInput();
    auto* right_weight_max1 =
        VarNode("right_weight_max1")
            ->assert_is_op_input("__xpu__conv2d", "FilterMax")
            ->assert_is_persistable_var()
            ->AsInput();
    auto* right_bias1 = VarNode("right_bias1")
                            ->assert_is_op_input("__xpu__conv2d", "Bias")
                            ->assert_is_persistable_var()
                            ->AsInput();
    auto* right_xpu_conv1 =
        OpNode("right_xpu_conv1", "__xpu__conv2d")
            ->assert_op_attr_satisfied<bool>(
                "has_branch", [](const bool& attr) { return attr == false; })
            ->assert_op_attr_satisfied<int>(
                "act_type",
                [](const int& attr) {
                  return attr == 1 || attr == 2; /* support relu and sigmoid */
                })
            ->AsIntermediate();
    auto* right_conv_out1 = VarNode("right_conv_out1")
                                ->assert_is_op_output("__xpu__conv2d", "Output")
                                ->assert_is_op_nth_input("concat", "X", 1)
                                ->AsIntermediate();
    auto* right_conv_out_max1 =
        VarNode("right_conv_out_max1")
            ->assert_is_op_output("__xpu__conv2d", "OutputMax")
            ->AsIntermediate();
    auto* concat = OpNode("concat", "concat")
                       ->assert_op_attr_satisfied<int>(
                           "axis",
                           [](const int& attr) {
                             return attr == 1; /* support relu and sigmoid */
                           })
                       ->AsIntermediate();
    auto* concat_out =
        VarNode("concat_out")->assert_is_op_output("concat", "Out");
    PMNode* pool2d = nullptr;
    PMNode* pool2d_out = nullptr;
    if (has_pool2d_) {
      concat_out->assert_is_op_input("pool2d", "X")->AsIntermediate();
      pool2d =
          OpNode("pool2d", "pool2d")
              ->assert_op_attr_satisfied<bool>(
                  "global_pooling",
                  [](const bool& attr) { return attr == false; })
              ->assert_node_satisfied(XPUConv2dConcatPool2dFuser::Pool2dCheck)
              ->AsIntermediate();
      pool2d_out = VarNode("pool2d_out")
                       ->assert_is_op_output("pool2d", "Out")
                       ->AsOutput();
    } else {
      concat_out->AsOutput();
    }

    *input >> *left_xpu_conv0 >> *left_conv_out0 >> *concat;
    if (has_mid_conv_) {
      *input >> *right_xpu_conv0 >> *right_conv_out0 >> *right_xpu_conv1 >>
          *right_conv_out1 >> *concat;
      *right_weight0 >> *right_xpu_conv0;
      *right_weight_max0 >> *right_xpu_conv0;
      *right_bias0 >> *right_xpu_conv0;
      *right_xpu_conv0 >> *right_conv_out_max0;
    } else {
      *input >> *right_xpu_conv1 >> *right_conv_out1 >> *concat;
    }
    *concat >> *concat_out;
    if (has_pool2d_) {
      *concat_out >> *pool2d >> *pool2d_out;
    }
    *left_weight0 >> *left_xpu_conv0;
    *left_weight_max0 >> *left_xpu_conv0;
    *left_bias0 >> *left_xpu_conv0;
    *left_xpu_conv0 >> *left_conv_out_max0;

    *right_weight1 >> *right_xpu_conv1;
    *right_weight_max1 >> *right_xpu_conv1;
    *right_bias1 >> *right_xpu_conv1;
    *right_xpu_conv1 >> *right_conv_out_max1;
  }
  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override {
    std::vector<std::string> conv_name{"left_xpu_conv0", "right_xpu_conv1"};
    std::vector<std::string> filter_name{
        matched.at("left_weight0")->arg()->name,
        matched.at("right_weight1")->arg()->name};
    std::vector<std::string> bias_name = {
        matched.at("left_bias0")->arg()->name,
        matched.at("right_bias1")->arg()->name};
    std::vector<std::string> filter_max_name{
        matched.at("left_weight_max0")->arg()->name,
        matched.at("right_weight_max1")->arg()->name};
    if (has_mid_conv_) {
      conv_name.insert(conv_name.begin(), "right_xpu_conv0");
      filter_name.insert(filter_name.begin(),
                         matched.at("right_weight0")->arg()->name);
      bias_name.insert(bias_name.begin(),
                       matched.at("right_bias0")->arg()->name);
      filter_max_name.insert(filter_max_name.begin(),
                             matched.at("right_weight_max0")->arg()->name);
    }
    std::string output_name = "";
    if (has_pool2d_) {
      output_name = matched.at("pool2d_out")->arg()->name;
    } else {
      output_name = matched.at("concat_out")->arg()->name;
    }
    auto op_desc = *matched.at("left_xpu_conv0")->stmt()->op_info();
    auto conv = matched.at("left_xpu_conv0")->stmt()->op();
    auto* scope = conv->scope();
    op_desc.mutable_inputs()->clear();
    op_desc.mutable_outputs()->clear();

    // add new arg output_max
    std::string max_output_name = output_name + "_max";
    auto* max_output_node = graph->NewArgumentNode(max_output_name);
    max_output_node->arg()->type = LiteType::GetTensorTy(
        TARGET(kXPU), PRECISION(kFloat), DATALAYOUT(kNCHW));
    scope->NewTensor(max_output_name);

    op_desc.SetType("__xpu__block_fuse_op");
    op_desc.SetInput("Input", {matched.at("input")->arg()->name});
    op_desc.SetInput("Filter", {filter_name});
    op_desc.SetInput("Bias", {bias_name});
    op_desc.SetInput("FilterMax", {filter_max_name});
    op_desc.SetOutput("Output", {output_name});
    op_desc.SetOutput("OutputMax", {max_output_name});
    static const int PX = 0;
    static const int P1 = 1;
    static const int P2 = 2;
    static const int P3 = 3;
    static const int P4 = 4;
    static const int PNONE = 9;
    static const int PY = 10;
    std::vector<int> place_x;
    std::vector<int> place_y;
    std::vector<int> place_z;
    std::vector<int> block_lod;
    std::vector<int> op_type;
    if (has_mid_conv_) {
      place_x = {PX, PX, P2, P1};
      place_y = {PNONE, PNONE, PNONE, P3};
      place_z = {P2, P1, P3, P4};
      op_type = {0, 0, 0, 20};
    } else {
      place_x = {PX, PX, P1};
      place_y = {PNONE, PNONE, P2};
      place_z = {P1, P2, P3};
      op_type = {0, 0, 20};
    }
    if (has_pool2d_) {
      auto last_place = place_z.back();
      place_x.push_back(last_place);
      place_y.push_back(PNONE);
      place_z.push_back(PY);
      int pooling_type = -1;
      if (matched.at("pool2d")->stmt()->op_info()->GetAttr<std::string>(
              "pooling_type") == "avg") {
        if (matched.at("pool2d")->stmt()->op_info()->GetAttr<bool>(
                "exclusive") == true) {
          pooling_type = 1;
        } else {
          pooling_type = 2;
        }
      } else {
        pooling_type = 3;
      }
      op_type.push_back(pooling_type);
    } else {
      place_z[place_z.size() - 1] = PY;
    }
    if (has_mid_conv_ && has_pool2d_) {
      block_lod = {5};
    } else if (has_mid_conv_ || has_pool2d_) {
      block_lod = {4};
    } else {
      block_lod = {3};
    }

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

    if (has_pool2d_) {
      auto pool_kernel =
          matched.at("pool2d")->stmt()->op_info()->GetAttr<std::vector<int>>(
              "ksize");
      filter_dims.insert(
          filter_dims.end(), pool_kernel.begin(), pool_kernel.end());
      auto pool_strides =
          matched.at("pool2d")->stmt()->op_info()->GetAttr<std::vector<int>>(
              "strides");
      conv_strides.insert(
          conv_strides.end(), pool_strides.begin(), pool_strides.end());
      auto pool_paddings =
          matched.at("pool2d")->stmt()->op_info()->GetAttr<std::vector<int>>(
              "paddings");
      if (pool_paddings.size() == 2) {
        for (size_t i = 0; i < pool_strides.size(); ++i) {
          int copy_pad = *(pool_paddings.begin() + 2 * i);
          pool_paddings.insert(pool_paddings.begin() + 2 * i + 1, copy_pad);
        }
      } else {
        if (pool_paddings.size() != 4) {
          LOG(FATAL)
              << "Paddings size should be the same or twice as the input size.";
        }
      }
      if ((matched.at("pool2d")->stmt()->op_info()->HasAttr(
              "padding_algorithm")) &&
          (matched.at("pool2d")->stmt()->op_info()->GetAttr<std::string>(
               "padding_algorithm") == "VALID")) {
        pool_paddings[0] = 0;
        pool_paddings[1] = 0;
        pool_paddings[2] = 0;
        pool_paddings[3] = 0;
      }
      if ((matched.at("pool2d")->stmt()->op_info()->HasAttr("ceil_mode")) &&
          (matched.at("pool2d")->stmt()->op_info()->GetAttr<bool>(
              "ceil_mode"))) {
        pool_paddings[1] += pool_strides[0] - 1;
        pool_paddings[3] += pool_strides[1] - 1;
      }
      conv_paddings.insert(
          conv_paddings.end(), pool_paddings.begin(), pool_paddings.end());
    }

    op_desc.SetAttr("op_type", op_type);
    op_desc.SetAttr("place_x", place_x);
    op_desc.SetAttr("place_y", place_y);
    op_desc.SetAttr("place_z", place_z);
    op_desc.SetAttr("filter_dims", filter_dims);
    op_desc.SetAttr("strides", conv_strides);
    op_desc.SetAttr("paddings", conv_paddings);
    op_desc.SetAttr("dilations", conv_dilations);
    op_desc.SetAttr("groups", conv_groups);
    op_desc.SetAttr("act_type", act_type);
    op_desc.SetAttr("act_param", act_param);
    op_desc.SetAttr("block_lod", block_lod);

    auto& valid_places = conv->valid_places();
    auto block_op = LiteOpRegistry::Global().Create(op_desc.Type());
    block_op->Attach(op_desc, scope);
    auto* new_op_node = graph->GraphCreateInstructNode(block_op, valid_places);

    IR_NODE_LINK_TO(matched.at("input"), new_op_node);
    IR_NODE_LINK_TO(new_filter_node, new_op_node);
    IR_NODE_LINK_TO(new_filter_max_node, new_op_node);
    IR_NODE_LINK_TO(new_bias_node, new_op_node);
    IR_NODE_LINK_TO(new_op_node, max_output_node);
    if (has_pool2d_) {
      IR_NODE_LINK_TO(new_op_node, matched.at("pool2d_out"));
    } else {
      IR_NODE_LINK_TO(new_op_node, matched.at("concat_out"));
    }
  }

 private:
  bool has_mid_conv_;
  bool has_pool2d_;
};

}  // namespace fusion

class XPUConv2dConcatPool2dFusePass : public ProgramPass {
 public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override {
    for (auto has_pool2d : {true, false}) {
      for (auto has_mid_conv : {true, false}) {
        fusion::XPUConv2dConcatPool2dFuser fuser(has_mid_conv, has_pool2d);
        fuser(graph.get());
      }
    }
  }
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(__xpu__conv2d_concat_pool2d_fuse_pass,
                  paddle::lite::mir::XPUConv2dConcatPool2dFusePass)
    .BindTargets({TARGET(kXPU)})
    .BindKernel("__xpu__block_fuse_op");

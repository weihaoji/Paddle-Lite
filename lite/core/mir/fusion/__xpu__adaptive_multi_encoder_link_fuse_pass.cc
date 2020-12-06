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

class XPUAdaptiveMultiEncoderLinkFuser : public FuseBase {
 public:
  void BuildPattern() override {
    // auto* idx = VarNode("ids")
    //                  ->assert_is_op_input("__xpu__embedding_with_eltwise_add",
    //                  "Ids");

    // auto* input_mask = VarNode("input_mask")
    //                  ->assert_is_op_input("__xpu__embedding_with_eltwise_add",
    //                  "InputMask");

    // auto* tables = VarNode("tables")
    //                  ->assert_is_op_input("__xpu__embedding_with_eltwise_add",
    //                  "Tables");

    auto* xpu_embedding =
        OpNode("xpu_embedding", "__xpu__embedding_with_eltwise_add");

    auto* embedding_out =
        VarNode("embedding_out")
            ->assert_is_op_output("__xpu__embedding_with_eltwise_add", "Output")
            ->assert_is_op_input("layer_norm", "X");

    auto* embedding_mask_lod =
        VarNode("embedding_mask_lod")
            ->assert_is_op_output("__xpu__embedding_with_eltwise_add",
                                  "OutputMaskLod");

    // auto* layer_norm_bias = VarNode("layer_norm_bias")
    //                         ->assert_is_op_input("layer_norm", "Bias");

    // auto* layer_norm_scale = VarNode("layer_norm_scale")
    //                         ->assert_is_op_input("layer_norm", "Scale");

    // auto* layer_norm_mean = VarNode("layer_norm_mean")
    //                          ->assert_is_op_output("layer_norm", "Mean");

    // auto* layer_norm_variance = VarNode("layer_norm_variance")
    //                          ->assert_is_op_output("layer_norm", "Variance");

    auto* layer_norm_y =
        VarNode("layer_norm_y")
            ->assert_is_op_output("layer_norm", "Y")
            ->assert_is_op_input("__xpu__multi_encoder", "Input");

    auto* layer_norm = OpNode("layer_norm", "layer_norm");

    auto* xpu_encoder = OpNode("xpu_encoder", "__xpu__multi_encoder");

    auto* encoder_out =
        VarNode("encoder_out")
            ->assert_is_op_output("__xpu__multi_encoder", "Output")
            ->assert_is_op_input("slice", "Input")
            ->AsIntermediate();

    // auto* encoder_out_mask_lod = VarNode("encoder_out_mask_lod")
    //                          ->assert_is_op_output("__xpu__multi_encoder",
    //                          "OutputMaskLod");

    auto* matmul_in = VarNode("matmul_in")
                          ->assert_is_op_input("matmul", "X")
                          ->assert_is_op_input("matmul", "Y");

    auto* matmul = OpNode("matmul", "matmul")->AsIntermediate();

    auto* matmul_out = VarNode("matmul_out")
                           ->assert_is_op_output("matmul", "Out")
                           ->assert_is_op_input("scale", "X")
                           ->AsIntermediate();

    auto* scale = OpNode("scale", "scale")->AsIntermediate();

    auto* scale_out = VarNode("scale_out")
                          ->assert_is_op_output("scale", "Out")
                          ->assert_is_op_input("stack", "X")
                          ->AsIntermediate();

    auto* stack = OpNode("stack", "stack")->AsIntermediate();

    auto* stack_y = VarNode("stack_y")
                        ->assert_is_op_output("stack", "Y")
                        ->assert_is_op_input("__xpu__multi_encoder", "Mask")
                        ->AsIntermediate();

    auto* slice = OpNode("slice", "slice")->AsIntermediate();
    auto* slice_out = VarNode("slice_out")->assert_is_op_output("slice", "Out");

    *matmul_in >> *matmul >> *matmul_out >> *scale >> *scale_out >> *stack >>
        *stack_y >> *xpu_encoder;

    *xpu_embedding >> *embedding_out >> *layer_norm >> *layer_norm_y >>
        *xpu_encoder;

    *xpu_encoder >> *encoder_out >> *slice >> *slice_out;

    *xpu_embedding >> *embedding_mask_lod;
  }

  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override {
    auto* embedding_instruct = matched.at("xpu_embedding")->stmt();
    auto embedding_op_desc = *embedding_instruct->mutable_op_info();
    auto embedding_op = embedding_instruct->op();

    auto* encoder_instruct = matched.at("xpu_encoder")->stmt();
    auto encoder_op_desc = *encoder_instruct->mutable_op_info();
    auto encoder_op = encoder_instruct->op();

    auto* slice_instruct = matched.at("slice")->stmt();
    auto slice_op_desc = *slice_instruct->op_info();
    auto slice_op = slice_instruct->op();

    std::string input_mask_name = matched.at("matmul_in")->arg()->name;
    auto* input_mask_node = graph->RetrieveArgument(input_mask_name);

    std::string embedding_out_mask_lod_name =
        matched.at("embedding_mask_lod")->arg()->name;
    auto* embedding_out_mask_lod_node =
        graph->RetrieveArgument(embedding_out_mask_lod_name);

    std::string slice_out_name = matched.at("slice_out")->arg()->name;
    auto* slice_out_node = graph->RetrieveArgument(slice_out_name);

    if ((input_mask_node != nullptr) &&
        (embedding_out_mask_lod_node != nullptr) &&
        (slice_out_node != nullptr)) {
      embedding_op_desc.SetInput("InputMask", {input_mask_name});
      embedding_op_desc.SetAttr("adaptive_seq_len", true);
      encoder_op_desc.SetInput("InputMaskLod", {embedding_out_mask_lod_name});
      encoder_op_desc.SetAttr("adaptive_seq_len", true);
      encoder_op_desc.SetOutput("Output", {slice_out_name});

      if (slice_op_desc.HasAttr("axes")) {
        auto slice_axes = slice_op_desc.GetAttr<std::vector<int>>("axes");
        encoder_op_desc.SetAttr("slice_axes", slice_axes);
      }
      if (slice_op_desc.HasAttr("starts")) {
        auto slice_starts = slice_op_desc.GetAttr<std::vector<int>>("starts");
        encoder_op_desc.SetAttr("slice_starts", slice_starts);
      }
      if (slice_op_desc.HasAttr("ends")) {
        auto slice_ends = slice_op_desc.GetAttr<std::vector<int>>("ends");
        encoder_op_desc.SetAttr("slice_ends", slice_ends);
      }

      embedding_instruct->ResetOp(embedding_op_desc,
                                  embedding_op->valid_places());
      DirectedLink(input_mask_node, matched.at("xpu_embedding"));
      encoder_instruct->ResetOp(encoder_op_desc, encoder_op->valid_places());
      DirectedLink(embedding_out_mask_lod_node, matched.at("xpu_encoder"));
      DirectedLink(matched.at("xpu_encoder"), matched.at("slice_out"));
    }
  }
};

}  // namespace fusion

class XPUAdaptiveMultiEncoderLinkFusePass : public ProgramPass {
 public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override {
    fusion::XPUAdaptiveMultiEncoderLinkFuser fuser;
    fuser(graph.get());
  }
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(__xpu__adaptive_multi_encoder_link_fuse_pass,
                  paddle::lite::mir::XPUAdaptiveMultiEncoderLinkFusePass)
    .BindTargets({TARGET(kXPU)})
    .BindKernel("__xpu__multi_encoder");

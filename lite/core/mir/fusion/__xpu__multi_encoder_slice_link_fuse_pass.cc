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

class XPUMultiEncoderSliceLinkFuser : public FuseBase {
 public:
  void BuildPattern() override {
    auto* xpu_encoder = OpNode("xpu_encoder", "__xpu__multi_encoder");

    auto* encoder_out =
        VarNode("encoder_out")
            ->assert_is_op_output("__xpu__multi_encoder", "Output")
            ->assert_is_op_input("slice", "Input")
            ->AsIntermediate();

    auto* slice = OpNode("slice", "slice")
                      ->assert_op_attr_satisfied<std::vector<int>>(
                          "axes",
                          [](const std::vector<int>& attr) {
                            return attr.size() == 1 && attr[0] == 1;
                          })
                      ->assert_op_attr_satisfied<std::vector<int>>(
                          "starts",
                          [](const std::vector<int>& attr) {
                            return attr.size() == 1 && attr[0] == 0;
                          })
                      ->assert_op_attr_satisfied<std::vector<int>>(
                          "ends",
                          [](const std::vector<int>& attr) {
                            return attr.size() == 1 && attr[0] == 1;
                          })
                      ->AsIntermediate();
    auto* slice_out = VarNode("slice_out")->assert_is_op_output("slice", "Out");

    *xpu_encoder >> *encoder_out >> *slice >> *slice_out;
  }

  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override {
    auto* encoder_instruct = matched.at("xpu_encoder")->stmt();
    auto encoder_op_desc = *encoder_instruct->mutable_op_info();
    auto encoder_op = encoder_instruct->op();

    auto* slice_instruct = matched.at("slice")->stmt();
    auto slice_op_desc = *slice_instruct->op_info();
    auto slice_op = slice_instruct->op();

    std::string slice_out_name = matched.at("slice_out")->arg()->name;
    auto* slice_out_node = graph->RetrieveArgument(slice_out_name);

    if (slice_out_node != nullptr) {
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

      encoder_instruct->ResetOp(encoder_op_desc, encoder_op->valid_places());
      DirectedLink(matched.at("xpu_encoder"), matched.at("slice_out"));
    }
  }
};

}  // namespace fusion

class XPUMultiEncoderSliceLinkFusePass : public ProgramPass {
 public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override {
    fusion::XPUMultiEncoderSliceLinkFuser fuser;
    fuser(graph.get());
  }
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(__xpu__multi_encoder_slice_link_fuse_pass,
                  paddle::lite::mir::XPUMultiEncoderSliceLinkFusePass)
    .BindTargets({TARGET(kXPU)})
    .BindKernel("__xpu__multi_encoder");

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
/* link the previous __xpu__block_fuse_op's OutputMax to   */
/* next __xpu__block_fuse_op as InputMax                   */
/* For example:                                     */
/* graph[1]: sub block                              */
/*                     in_Input                     */
/*        in_Filter      |     in_FilterMax         */
/*                  \    |    /                     */
/*                   \   |   /                      */
/*     in_Bias ------- __xpu__block_fuse_op         */
/*                       |      \                   */
/*                       |       \                  */
/*                out_Output      out_OutputMax     */
/*                       |                          */
/*                       |                          */
/*                    __xpu__block_fuse_op          */
/*                       |                          */
/*                       |                          */
/*                     out_Output                   */
/*                                                  */
/* After the pass is applied:                       */
/*                     in_Input                     */
/*        in_Filter      |     in_FilterMax         */
/*                  \    |    /                     */
/*                   \   |   /                      */
/*     in_Bias ------- __xpu__block_fuse_op         */
/*                       |      \                   */
/*                       |       \                  */
/*                out_Output      out_OutputMax     */
/*                       |       /                  */
/*                       |      /                   */
/*                    __xpu__block_fuse_op          */
/*                       |                          */
/*                       |                          */
/*                     out_Output                   */

class XPUBlockLinkFuser : public FuseBase {
 public:
  void BuildPattern() override {
    auto* input = VarNode("input")
                      ->assert_is_op_input("__xpu__block_fuse_op", "Input")
                      ->AsInput();
    auto* filter = VarNode("filter")
                       ->assert_is_op_input("__xpu__block_fuse_op", "Filter")
                       ->AsInput();
    auto* filter_max =
        VarNode("filter_max")
            ->assert_is_op_input("__xpu__block_fuse_op", "FilterMax")
            ->AsInput();
    auto* bias = VarNode("bias")
                     ->assert_is_op_input("__xpu__block_fuse_op", "Bias")
                     ->AsInput();
    auto* xpu_block = OpNode("__xpu__block_fuse_op", "__xpu__block_fuse_op");
    auto* xpu_block_out =
        VarNode("xpu_block_out")
            ->assert_is_op_output("__xpu__block_fuse_op", "Output")
            ->AsOutput();
    auto* xpu_block_out_max =
        VarNode("xpu_block_out_max")
            ->assert_is_op_output("__xpu__block_fuse_op", "OutputMax")
            ->AsOutput();

    *input >> *xpu_block >> *xpu_block_out;
    *filter >> *xpu_block;
    *filter_max >> *xpu_block;
    *bias >> *xpu_block;
    *xpu_block >> *xpu_block_out_max;
  }

  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override {
    auto block_instruct = matched.at("__xpu__block_fuse_op")->stmt();
    auto op_desc = *block_instruct->mutable_op_info();
    auto block_old = block_instruct->op();

    // try to find input_max
    std::string max_input_name = matched.at("input")->arg()->name + "_max";
    auto* max_input_node = graph->RetrieveArgument(max_input_name);
    if (max_input_node != nullptr &&
        (!op_desc.HasAttr("has_input_max") ||
         !op_desc.GetAttr<bool>("has_input_max"))) {
      op_desc.SetInput("InputMax", {max_input_name});
      op_desc.SetAttr("has_input_max", true);
      block_instruct->ResetOp(op_desc, block_old->valid_places());
      DirectedLink(max_input_node, matched.at("__xpu__block_fuse_op"));
    }
  }

 private:
};

}  // namespace fusion

class XPUBlockLinkPass : public ProgramPass {
 public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override {
    fusion::XPUBlockLinkFuser fuser;
    fuser(graph.get());
  }
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(__xpu__block_link_previous_out_max_pass,
                  paddle::lite::mir::XPUBlockLinkPass)
    .BindTargets({TARGET(kXPU)})
    .BindKernel("__xpu__block_fuse_op");

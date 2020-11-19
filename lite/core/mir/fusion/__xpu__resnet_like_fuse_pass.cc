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

#define NORMAL_BLOCK_PATTERN(num)                                          \
  auto* normal_block_weight0_##num =                                       \
      VarNode(STR2(normal_block_weight0_##num))                            \
          ->assert_is_op_nth_input("__xpu__block_fuse_op", "Filter", 0)    \
          ->AsInput();                                                     \
  auto* normal_block_weight1_##num =                                       \
      VarNode(STR2(normal_block_weight1_##num))                            \
          ->assert_is_op_nth_input("__xpu__block_fuse_op", "Filter", 1)    \
          ->AsInput();                                                     \
  auto* normal_block_weight2_##num =                                       \
      VarNode(STR2(normal_block_weight2_##num))                            \
          ->assert_is_op_nth_input("__xpu__block_fuse_op", "Filter", 2)    \
          ->AsInput();                                                     \
  auto* normal_block_weight_max0_##num =                                   \
      VarNode(STR2(normal_block_weight_max0_##num))                        \
          ->assert_is_op_nth_input("__xpu__block_fuse_op", "FilterMax", 0) \
          ->AsInput();                                                     \
  auto* normal_block_weight_max1_##num =                                   \
      VarNode(STR2(normal_block_weight_max1_##num))                        \
          ->assert_is_op_nth_input("__xpu__block_fuse_op", "FilterMax", 1) \
          ->AsInput();                                                     \
  auto* normal_block_weight_max2_##num =                                   \
      VarNode(STR2(normal_block_weight_max2_##num))                        \
          ->assert_is_op_nth_input("__xpu__block_fuse_op", "FilterMax", 2) \
          ->AsInput();                                                     \
  auto* normal_block_bias0_##num =                                         \
      VarNode(STR2(normal_block_bias0_##num))                              \
          ->assert_is_op_nth_input("__xpu__block_fuse_op", "Bias", 0)      \
          ->AsInput();                                                     \
  auto* normal_block_bias1_##num =                                         \
      VarNode(STR2(normal_block_bias1_##num))                              \
          ->assert_is_op_nth_input("__xpu__block_fuse_op", "Bias", 1)      \
          ->AsInput();                                                     \
  auto* normal_block_bias2_##num =                                         \
      VarNode(STR2(normal_block_bias2_##num))                              \
          ->assert_is_op_nth_input("__xpu__block_fuse_op", "Bias", 2)      \
          ->AsInput();                                                     \
  auto* normal_block_op_##num =                                            \
      OpNode(STR2(normal_block_op_##num), "__xpu__block_fuse_op")          \
          ->AsIntermediate();                                              \
  auto* normal_block_op_out_##num =                                        \
      VarNode(STR2(normal_block_op_out_##num))                             \
          ->assert_is_op_output("__xpu__block_fuse_op", "Output");         \
  auto* normal_block_op_out_max_##num =                                    \
      VarNode(STR2(normal_block_op_out_max_##num))                         \
          ->assert_is_op_output("__xpu__block_fuse_op", "OutputMax");

#define NORMAL_BLOCK_CONNECT(num)                            \
  *normal_block_weight0_##num >> *normal_block_op_##num;     \
  *normal_block_weight1_##num >> *normal_block_op_##num;     \
  *normal_block_weight2_##num >> *normal_block_op_##num;     \
  *normal_block_weight_max0_##num >> *normal_block_op_##num; \
  *normal_block_weight_max1_##num >> *normal_block_op_##num; \
  *normal_block_weight_max2_##num >> *normal_block_op_##num; \
  *normal_block_bias0_##num >> *normal_block_op_##num;       \
  *normal_block_bias1_##num >> *normal_block_op_##num;       \
  *normal_block_bias2_##num >> *normal_block_op_##num;       \
  *normal_block_op_##num >> *normal_block_op_out_max_##num;  \
  *normal_block_op_##num >> *normal_block_op_out_##num;      \
  normal_block_op_out_##num->AsOutput();                     \
  normal_block_op_out_max_##num->AsOutput();

#define NORMAL_BLOCK_INTERNAL_CONNECT(num)                   \
  *normal_block_weight0_##num >> *normal_block_op_##num;     \
  *normal_block_weight1_##num >> *normal_block_op_##num;     \
  *normal_block_weight2_##num >> *normal_block_op_##num;     \
  *normal_block_weight_max0_##num >> *normal_block_op_##num; \
  *normal_block_weight_max1_##num >> *normal_block_op_##num; \
  *normal_block_weight_max2_##num >> *normal_block_op_##num; \
  *normal_block_bias0_##num >> *normal_block_op_##num;       \
  *normal_block_bias1_##num >> *normal_block_op_##num;       \
  *normal_block_bias2_##num >> *normal_block_op_##num;       \
  *normal_block_op_##num >> *normal_block_op_out_max_##num;  \
  *normal_block_op_##num >> *normal_block_op_out_##num;      \
  normal_block_op_out_##num->AsIntermediate();               \
  normal_block_op_out_max_##num->AsIntermediate();

#define NORMAL_BLOCK_FILTER_NAME(num)                            \
  matched.at(STR2(normal_block_weight0_##num))->arg()->name,     \
      matched.at(STR2(normal_block_weight1_##num))->arg()->name, \
      matched.at(STR2(normal_block_weight2_##num))->arg()->name,

#define NORMAL_BLOCK_FILTERMAX_NAME(num)                             \
  matched.at(STR2(normal_block_weight_max0_##num))->arg()->name,     \
      matched.at(STR2(normal_block_weight_max1_##num))->arg()->name, \
      matched.at(STR2(normal_block_weight_max2_##num))->arg()->name,

#define NORMAL_BLOCK_BIAS_NAME(num)                            \
  matched.at(STR2(normal_block_bias0_##num))->arg()->name,     \
      matched.at(STR2(normal_block_bias1_##num))->arg()->name, \
      matched.at(STR2(normal_block_bias2_##num))->arg()->name,

#define NORMAL_BLOCK_VAR_NODE(num)                                          \
  STR2(normal_block_weight0_##num)                                          \
  , STR2(normal_block_weight1_##num), STR2(normal_block_weight2_##num),     \
      STR2(normal_block_weight_max0_##num),                                 \
      STR2(normal_block_weight_max1_##num),                                 \
      STR2(normal_block_weight_max2_##num), STR2(normal_block_bias0_##num), \
      STR2(normal_block_bias1_##num), STR2(normal_block_bias2_##num),

#define REDUCTION_BLOCK_PATTERN(num)                                       \
  auto* reduct_block_weight0_##num =                                       \
      VarNode(STR2(reduct_block_weight0_##num))                            \
          ->assert_is_op_nth_input("__xpu__block_fuse_op", "Filter", 0)    \
          ->AsInput();                                                     \
  auto* reduct_block_weight1_##num =                                       \
      VarNode(STR2(reduct_block_weight1_##num))                            \
          ->assert_is_op_nth_input("__xpu__block_fuse_op", "Filter", 1)    \
          ->AsInput();                                                     \
  auto* reduct_block_weight2_##num =                                       \
      VarNode(STR2(reduct_block_weight2_##num))                            \
          ->assert_is_op_nth_input("__xpu__block_fuse_op", "Filter", 2)    \
          ->AsInput();                                                     \
  auto* reduct_block_weight3_##num =                                       \
      VarNode(STR2(reduct_block_weight3_##num))                            \
          ->assert_is_op_nth_input("__xpu__block_fuse_op", "Filter", 3)    \
          ->AsInput();                                                     \
  auto* reduct_block_weight_max0_##num =                                   \
      VarNode(STR2(reduct_block_weight_max0_##num))                        \
          ->assert_is_op_nth_input("__xpu__block_fuse_op", "FilterMax", 0) \
          ->AsInput();                                                     \
  auto* reduct_block_weight_max1_##num =                                   \
      VarNode(STR2(reduct_block_weight_max1_##num))                        \
          ->assert_is_op_nth_input("__xpu__block_fuse_op", "FilterMax", 1) \
          ->AsInput();                                                     \
  auto* reduct_block_weight_max2_##num =                                   \
      VarNode(STR2(reduct_block_weight_max2_##num))                        \
          ->assert_is_op_nth_input("__xpu__block_fuse_op", "FilterMax", 2) \
          ->AsInput();                                                     \
  auto* reduct_block_weight_max3_##num =                                   \
      VarNode(STR2(reduct_block_weight_max3_##num))                        \
          ->assert_is_op_nth_input("__xpu__block_fuse_op", "FilterMax", 3) \
          ->AsInput();                                                     \
  auto* reduct_block_bias0_##num =                                         \
      VarNode(STR2(reduct_block_bias0_##num))                              \
          ->assert_is_op_nth_input("__xpu__block_fuse_op", "Bias", 0)      \
          ->AsInput();                                                     \
  auto* reduct_block_bias1_##num =                                         \
      VarNode(STR2(reduct_block_bias1_##num))                              \
          ->assert_is_op_nth_input("__xpu__block_fuse_op", "Bias", 1)      \
          ->AsInput();                                                     \
  auto* reduct_block_bias2_##num =                                         \
      VarNode(STR2(reduct_block_bias2_##num))                              \
          ->assert_is_op_nth_input("__xpu__block_fuse_op", "Bias", 2)      \
          ->AsInput();                                                     \
  auto* reduct_block_bias3_##num =                                         \
      VarNode(STR2(reduct_block_bias3_##num))                              \
          ->assert_is_op_nth_input("__xpu__block_fuse_op", "Bias", 3)      \
          ->AsInput();                                                     \
  auto* reduct_block_op_##num =                                            \
      OpNode(STR2(reduct_block_op_##num), "__xpu__block_fuse_op")          \
          ->AsIntermediate();                                              \
  auto* reduct_block_op_out_##num =                                        \
      VarNode(STR2(reduct_block_op_out_##num))                             \
          ->assert_is_op_output("__xpu__block_fuse_op", "Output")          \
          ->AsIntermediate();                                              \
  auto* reduct_block_op_out_max_##num =                                    \
      VarNode(STR2(reduct_block_op_out_max_##num))                         \
          ->assert_is_op_output("__xpu__block_fuse_op", "OutputMax")       \
          ->AsIntermediate();

#define REDUCT_BLOCK_CONNECT(num)                            \
  *reduct_block_weight0_##num >> *reduct_block_op_##num;     \
  *reduct_block_weight1_##num >> *reduct_block_op_##num;     \
  *reduct_block_weight2_##num >> *reduct_block_op_##num;     \
  *reduct_block_weight3_##num >> *reduct_block_op_##num;     \
  *reduct_block_weight_max0_##num >> *reduct_block_op_##num; \
  *reduct_block_weight_max1_##num >> *reduct_block_op_##num; \
  *reduct_block_weight_max2_##num >> *reduct_block_op_##num; \
  *reduct_block_weight_max3_##num >> *reduct_block_op_##num; \
  *reduct_block_bias0_##num >> *reduct_block_op_##num;       \
  *reduct_block_bias1_##num >> *reduct_block_op_##num;       \
  *reduct_block_bias2_##num >> *reduct_block_op_##num;       \
  *reduct_block_bias3_##num >> *reduct_block_op_##num;       \
  *reduct_block_op_##num >> *reduct_block_op_out_max_##num;  \
  *reduct_block_op_##num >> *reduct_block_op_out_##num;

#define REDUCT_BLOCK_FILTER_NAME(num)                            \
  matched.at(STR2(reduct_block_weight0_##num))->arg()->name,     \
      matched.at(STR2(reduct_block_weight1_##num))->arg()->name, \
      matched.at(STR2(reduct_block_weight2_##num))->arg()->name, \
      matched.at(STR2(reduct_block_weight3_##num))->arg()->name,

#define REDUCT_BLOCK_FILTERMAX_NAME(num)                             \
  matched.at(STR2(reduct_block_weight_max0_##num))->arg()->name,     \
      matched.at(STR2(reduct_block_weight_max1_##num))->arg()->name, \
      matched.at(STR2(reduct_block_weight_max2_##num))->arg()->name, \
      matched.at(STR2(reduct_block_weight_max3_##num))->arg()->name,

#define REDUCT_BLOCK_BIAS_NAME(num)                            \
  matched.at(STR2(reduct_block_bias0_##num))->arg()->name,     \
      matched.at(STR2(reduct_block_bias1_##num))->arg()->name, \
      matched.at(STR2(reduct_block_bias2_##num))->arg()->name, \
      matched.at(STR2(reduct_block_bias3_##num))->arg()->name,

#define REDUCT_BLOCK_VAR_NODE(num)                                            \
  STR2(reduct_block_weight0_##num)                                            \
  , STR2(reduct_block_weight1_##num), STR2(reduct_block_weight2_##num),       \
      STR2(reduct_block_weight3_##num), STR2(reduct_block_weight_max0_##num), \
      STR2(reduct_block_weight_max1_##num),                                   \
      STR2(reduct_block_weight_max2_##num),                                   \
      STR2(reduct_block_weight_max3_##num), STR2(reduct_block_bias0_##num),   \
      STR2(reduct_block_bias1_##num), STR2(reduct_block_bias2_##num),         \
      STR2(reduct_block_bias3_##num),

class XPUResNetLikeFuser : public FuseBase {
 public:
  explicit XPUResNetLikeFuser(bool stage1_out,
                              bool stage2_out,
                              bool stage3_out,
                              bool is_resnet101) {
    has_stage1_out = stage1_out;
    has_stage2_out = stage2_out;
    has_stage3_out = stage3_out;
    is_resnet101_ = is_resnet101;
  }
  void BuildPattern() override {
    auto* input = VarNode("input")
                      ->assert_is_op_input("__xpu__block_fuse_op", "Input")
                      ->AsInput();

    REDUCTION_BLOCK_PATTERN(1);
    NORMAL_BLOCK_PATTERN(11);
    NORMAL_BLOCK_PATTERN(12);

    REDUCTION_BLOCK_PATTERN(2);
    NORMAL_BLOCK_PATTERN(21);
    NORMAL_BLOCK_PATTERN(22);
    NORMAL_BLOCK_PATTERN(23);

    REDUCTION_BLOCK_PATTERN(3);
    NORMAL_BLOCK_PATTERN(31);
    NORMAL_BLOCK_PATTERN(32);
    NORMAL_BLOCK_PATTERN(33);
    NORMAL_BLOCK_PATTERN(34);
    NORMAL_BLOCK_PATTERN(35);

    REDUCTION_BLOCK_PATTERN(4);
    NORMAL_BLOCK_PATTERN(41);
    NORMAL_BLOCK_PATTERN(42);

    *input >> *reduct_block_op_1 >> *reduct_block_op_out_1 >>
        *normal_block_op_11 >> *normal_block_op_out_11 >> *normal_block_op_12 >>
        *normal_block_op_out_12;

    *normal_block_op_out_12 >> *reduct_block_op_2 >> *reduct_block_op_out_2 >>
        *normal_block_op_21 >> *normal_block_op_out_21 >> *normal_block_op_22 >>
        *normal_block_op_out_22 >> *normal_block_op_23 >>
        *normal_block_op_out_23;

    *normal_block_op_out_23 >> *reduct_block_op_3 >> *reduct_block_op_out_3 >>
        *normal_block_op_31 >> *normal_block_op_out_31 >> *normal_block_op_32 >>
        *normal_block_op_out_32 >> *normal_block_op_33 >>
        *normal_block_op_out_33 >> *normal_block_op_34 >>
        *normal_block_op_out_34 >> *normal_block_op_35 >>
        *normal_block_op_out_35;

    *reduct_block_op_out_4 >> *normal_block_op_41 >> *normal_block_op_out_41 >>
        *normal_block_op_42 >> *normal_block_op_out_42;

    REDUCT_BLOCK_CONNECT(1);
    NORMAL_BLOCK_INTERNAL_CONNECT(11);
    if (has_stage1_out) {
      NORMAL_BLOCK_CONNECT(12);
    } else {
      NORMAL_BLOCK_INTERNAL_CONNECT(12);
    }

    REDUCT_BLOCK_CONNECT(2);
    NORMAL_BLOCK_INTERNAL_CONNECT(21);
    NORMAL_BLOCK_INTERNAL_CONNECT(22);
    if (has_stage2_out) {
      NORMAL_BLOCK_CONNECT(23);
    } else {
      NORMAL_BLOCK_INTERNAL_CONNECT(23);
    }

    REDUCT_BLOCK_CONNECT(3);
    NORMAL_BLOCK_INTERNAL_CONNECT(31);
    NORMAL_BLOCK_INTERNAL_CONNECT(32);
    NORMAL_BLOCK_INTERNAL_CONNECT(33);
    NORMAL_BLOCK_INTERNAL_CONNECT(34);

    REDUCT_BLOCK_CONNECT(4);
    NORMAL_BLOCK_INTERNAL_CONNECT(41);
    NORMAL_BLOCK_CONNECT(42);

    if (is_resnet101_) {
      NORMAL_BLOCK_PATTERN(36);
      NORMAL_BLOCK_PATTERN(37);
      NORMAL_BLOCK_PATTERN(38);
      NORMAL_BLOCK_PATTERN(39);
      NORMAL_BLOCK_PATTERN(310);
      NORMAL_BLOCK_PATTERN(311);
      NORMAL_BLOCK_PATTERN(312);
      NORMAL_BLOCK_PATTERN(313);
      NORMAL_BLOCK_PATTERN(314);
      NORMAL_BLOCK_PATTERN(315);
      NORMAL_BLOCK_PATTERN(316);
      NORMAL_BLOCK_PATTERN(317);
      NORMAL_BLOCK_PATTERN(318);
      NORMAL_BLOCK_PATTERN(319);
      NORMAL_BLOCK_PATTERN(320);
      NORMAL_BLOCK_PATTERN(321);
      NORMAL_BLOCK_PATTERN(322);

      *normal_block_op_out_35 >> *normal_block_op_36 >>
          *normal_block_op_out_36 >> *normal_block_op_37 >>
          *normal_block_op_out_37 >> *normal_block_op_38 >>
          *normal_block_op_out_38 >> *normal_block_op_39 >>
          *normal_block_op_out_39 >> *normal_block_op_310 >>
          *normal_block_op_out_310 >> *normal_block_op_311 >>
          *normal_block_op_out_311 >> *normal_block_op_312 >>
          *normal_block_op_out_312 >> *normal_block_op_313 >>
          *normal_block_op_out_313 >> *normal_block_op_314 >>
          *normal_block_op_out_314 >> *normal_block_op_315 >>
          *normal_block_op_out_315 >> *normal_block_op_316 >>
          *normal_block_op_out_316 >> *normal_block_op_317 >>
          *normal_block_op_out_317 >> *normal_block_op_318 >>
          *normal_block_op_out_318 >> *normal_block_op_319 >>
          *normal_block_op_out_319 >> *normal_block_op_320 >>
          *normal_block_op_out_320 >> *normal_block_op_321 >>
          *normal_block_op_out_321 >> *normal_block_op_322 >>
          *normal_block_op_out_322;

      *normal_block_op_out_322 >> *reduct_block_op_4 >> *reduct_block_op_out_4;

      NORMAL_BLOCK_INTERNAL_CONNECT(35);
      NORMAL_BLOCK_INTERNAL_CONNECT(36);
      NORMAL_BLOCK_INTERNAL_CONNECT(37);
      NORMAL_BLOCK_INTERNAL_CONNECT(38);
      NORMAL_BLOCK_INTERNAL_CONNECT(39);
      NORMAL_BLOCK_INTERNAL_CONNECT(310);
      NORMAL_BLOCK_INTERNAL_CONNECT(311);
      NORMAL_BLOCK_INTERNAL_CONNECT(312);
      NORMAL_BLOCK_INTERNAL_CONNECT(313);
      NORMAL_BLOCK_INTERNAL_CONNECT(314);
      NORMAL_BLOCK_INTERNAL_CONNECT(315);
      NORMAL_BLOCK_INTERNAL_CONNECT(316);
      NORMAL_BLOCK_INTERNAL_CONNECT(317);
      NORMAL_BLOCK_INTERNAL_CONNECT(318);
      NORMAL_BLOCK_INTERNAL_CONNECT(319);
      NORMAL_BLOCK_INTERNAL_CONNECT(320);
      NORMAL_BLOCK_INTERNAL_CONNECT(321);

      if (has_stage3_out) {
        NORMAL_BLOCK_CONNECT(322);
      } else {
        NORMAL_BLOCK_INTERNAL_CONNECT(322);
      }
    } else {
      *normal_block_op_out_35 >> *reduct_block_op_4 >> *reduct_block_op_out_4;

      if (has_stage3_out) {
        NORMAL_BLOCK_CONNECT(35);
      } else {
        NORMAL_BLOCK_INTERNAL_CONNECT(35);
      }
    }
  }
  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override {
    std::vector<std::string> block_name{"reduct_block_op_1",
                                        "normal_block_op_11",
                                        "normal_block_op_12",
                                        "reduct_block_op_2",
                                        "normal_block_op_21",
                                        "normal_block_op_22",
                                        "normal_block_op_23",
                                        "reduct_block_op_3",
                                        "normal_block_op_31",
                                        "normal_block_op_32",
                                        "normal_block_op_33",
                                        "normal_block_op_34",
                                        "normal_block_op_35",
                                        "reduct_block_op_4",
                                        "normal_block_op_41",
                                        "normal_block_op_42"};
    if (is_resnet101_) {
      std::vector<std::string> resnet101_block_name{"normal_block_op_36",
                                                    "normal_block_op_37",
                                                    "normal_block_op_38",
                                                    "normal_block_op_39",
                                                    "normal_block_op_310",
                                                    "normal_block_op_311",
                                                    "normal_block_op_312",
                                                    "normal_block_op_313",
                                                    "normal_block_op_314",
                                                    "normal_block_op_315",
                                                    "normal_block_op_316",
                                                    "normal_block_op_317",
                                                    "normal_block_op_318",
                                                    "normal_block_op_319",
                                                    "normal_block_op_320",
                                                    "normal_block_op_321",
                                                    "normal_block_op_322"};
      block_name.insert(block_name.begin() + 13,
                        resnet101_block_name.begin(),
                        resnet101_block_name.end());
    }

    std::vector<std::string> filter_name{
        REDUCT_BLOCK_FILTER_NAME(1) NORMAL_BLOCK_FILTER_NAME(
            11) NORMAL_BLOCK_FILTER_NAME(12) REDUCT_BLOCK_FILTER_NAME(2)
            NORMAL_BLOCK_FILTER_NAME(21) NORMAL_BLOCK_FILTER_NAME(
                22) NORMAL_BLOCK_FILTER_NAME(23) REDUCT_BLOCK_FILTER_NAME(3)
                NORMAL_BLOCK_FILTER_NAME(31) NORMAL_BLOCK_FILTER_NAME(32)
                    NORMAL_BLOCK_FILTER_NAME(33) NORMAL_BLOCK_FILTER_NAME(34)
                        NORMAL_BLOCK_FILTER_NAME(35) REDUCT_BLOCK_FILTER_NAME(4)
                            NORMAL_BLOCK_FILTER_NAME(41)
                                NORMAL_BLOCK_FILTER_NAME(42)};
    if (is_resnet101_) {
      std::vector<std::string> resnet101_filter_name = {
          NORMAL_BLOCK_FILTER_NAME(36) NORMAL_BLOCK_FILTER_NAME(
              37) NORMAL_BLOCK_FILTER_NAME(38) NORMAL_BLOCK_FILTER_NAME(39)
              NORMAL_BLOCK_FILTER_NAME(310) NORMAL_BLOCK_FILTER_NAME(311)
                  NORMAL_BLOCK_FILTER_NAME(312) NORMAL_BLOCK_FILTER_NAME(313)
                      NORMAL_BLOCK_FILTER_NAME(314) NORMAL_BLOCK_FILTER_NAME(
                          315) NORMAL_BLOCK_FILTER_NAME(316)
                          NORMAL_BLOCK_FILTER_NAME(317)
                              NORMAL_BLOCK_FILTER_NAME(318)
                                  NORMAL_BLOCK_FILTER_NAME(319)
                                      NORMAL_BLOCK_FILTER_NAME(320)
                                          NORMAL_BLOCK_FILTER_NAME(321)
                                              NORMAL_BLOCK_FILTER_NAME(322)};
      filter_name.insert(filter_name.begin() + 42,
                         resnet101_filter_name.begin(),
                         resnet101_filter_name.end());
    }

    std::vector<std::string> bias_name = {
        REDUCT_BLOCK_BIAS_NAME(1) NORMAL_BLOCK_BIAS_NAME(
            11) NORMAL_BLOCK_BIAS_NAME(12) REDUCT_BLOCK_BIAS_NAME(2)
            NORMAL_BLOCK_BIAS_NAME(21) NORMAL_BLOCK_BIAS_NAME(22)
                NORMAL_BLOCK_BIAS_NAME(23) REDUCT_BLOCK_BIAS_NAME(3)
                    NORMAL_BLOCK_BIAS_NAME(31) NORMAL_BLOCK_BIAS_NAME(32)
                        NORMAL_BLOCK_BIAS_NAME(33) NORMAL_BLOCK_BIAS_NAME(34)
                            NORMAL_BLOCK_BIAS_NAME(35) REDUCT_BLOCK_BIAS_NAME(4)
                                NORMAL_BLOCK_BIAS_NAME(41)
                                    NORMAL_BLOCK_BIAS_NAME(42)};
    if (is_resnet101_) {
      std::vector<std::string> resnet101_bias_name = {
          NORMAL_BLOCK_BIAS_NAME(36) NORMAL_BLOCK_BIAS_NAME(
              37) NORMAL_BLOCK_BIAS_NAME(38) NORMAL_BLOCK_BIAS_NAME(39)
              NORMAL_BLOCK_BIAS_NAME(310) NORMAL_BLOCK_BIAS_NAME(311)
                  NORMAL_BLOCK_BIAS_NAME(312) NORMAL_BLOCK_BIAS_NAME(313)
                      NORMAL_BLOCK_BIAS_NAME(314) NORMAL_BLOCK_BIAS_NAME(315)
                          NORMAL_BLOCK_BIAS_NAME(316)
                              NORMAL_BLOCK_BIAS_NAME(317)
                                  NORMAL_BLOCK_BIAS_NAME(318)
                                      NORMAL_BLOCK_BIAS_NAME(319)
                                          NORMAL_BLOCK_BIAS_NAME(320)
                                              NORMAL_BLOCK_BIAS_NAME(321)
                                                  NORMAL_BLOCK_BIAS_NAME(322)};
      bias_name.insert(bias_name.begin() + 42,
                       resnet101_bias_name.begin(),
                       resnet101_bias_name.end());
    }

    std::vector<std::string> filter_max_name{
        REDUCT_BLOCK_FILTERMAX_NAME(1) NORMAL_BLOCK_FILTERMAX_NAME(
            11) NORMAL_BLOCK_FILTERMAX_NAME(12) REDUCT_BLOCK_FILTERMAX_NAME(2)
            NORMAL_BLOCK_FILTERMAX_NAME(21) NORMAL_BLOCK_FILTERMAX_NAME(22)
                NORMAL_BLOCK_FILTERMAX_NAME(23) REDUCT_BLOCK_FILTERMAX_NAME(3)
                    NORMAL_BLOCK_FILTERMAX_NAME(31) NORMAL_BLOCK_FILTERMAX_NAME(
                        32) NORMAL_BLOCK_FILTERMAX_NAME(33)
                        NORMAL_BLOCK_FILTERMAX_NAME(34)
                            NORMAL_BLOCK_FILTERMAX_NAME(35)
                                REDUCT_BLOCK_FILTERMAX_NAME(4)
                                    NORMAL_BLOCK_FILTERMAX_NAME(41)
                                        NORMAL_BLOCK_FILTERMAX_NAME(42)};

    if (is_resnet101_) {
      std::vector<std::string> resnet101_filter_max_name{
          NORMAL_BLOCK_FILTERMAX_NAME(36) NORMAL_BLOCK_FILTERMAX_NAME(
              37) NORMAL_BLOCK_FILTERMAX_NAME(38)
              NORMAL_BLOCK_FILTERMAX_NAME(39) NORMAL_BLOCK_FILTERMAX_NAME(
                  310) NORMAL_BLOCK_FILTERMAX_NAME(311)
                  NORMAL_BLOCK_FILTERMAX_NAME(312) NORMAL_BLOCK_FILTERMAX_NAME(
                      313) NORMAL_BLOCK_FILTERMAX_NAME(314)
                      NORMAL_BLOCK_FILTERMAX_NAME(
                          315) NORMAL_BLOCK_FILTERMAX_NAME(316)
                          NORMAL_BLOCK_FILTERMAX_NAME(317)
                              NORMAL_BLOCK_FILTERMAX_NAME(318)
                                  NORMAL_BLOCK_FILTERMAX_NAME(319)
                                      NORMAL_BLOCK_FILTERMAX_NAME(320)
                                          NORMAL_BLOCK_FILTERMAX_NAME(321)
                                              NORMAL_BLOCK_FILTERMAX_NAME(322)};
      filter_max_name.insert(filter_max_name.begin() + 42,
                             resnet101_filter_max_name.begin(),
                             resnet101_filter_max_name.end());
    }

    std::vector<std::string> feature_out_list;
    std::vector<int> has_block_out(16, 0);
    if (is_resnet101_) {
      std::vector<int> resnet101_has_block_out(17, 0);
      has_block_out.insert(has_block_out.end(),
                           resnet101_has_block_out.begin(),
                           resnet101_has_block_out.end());
    }
    if (has_stage1_out) {
      feature_out_list.push_back(
          matched.at("normal_block_op_out_12")->arg()->name);
      has_block_out[2] = 1;
    }
    if (has_stage2_out) {
      feature_out_list.push_back(
          matched.at("normal_block_op_out_23")->arg()->name);
      has_block_out[6] = 1;
    }
    if (is_resnet101_) {
      if (has_stage3_out) {
        feature_out_list.push_back(
            matched.at("normal_block_op_out_322")->arg()->name);
        has_block_out[29] = 1;
      }
    } else {
      if (has_stage3_out) {
        feature_out_list.push_back(
            matched.at("normal_block_op_out_35")->arg()->name);
        has_block_out[12] = 1;
      }
    }

    auto op_desc = *matched.at("reduct_block_op_1")->stmt()->op_info();
    auto block_first = matched.at("reduct_block_op_1")->stmt()->op();
    auto* scope = block_first->scope();

    op_desc.mutable_inputs()->clear();
    op_desc.mutable_outputs()->clear();

    op_desc.SetType("__xpu__block_fuse_op");
    op_desc.SetInput("Input", {matched.at("input")->arg()->name});
    op_desc.SetInput("Filter", filter_name);
    op_desc.SetInput("Bias", bias_name);
    op_desc.SetInput("FilterMax", filter_max_name);

    op_desc.SetOutput("Output",
                      {matched.at("normal_block_op_out_42")->arg()->name});
    op_desc.SetOutput("OutputMax",
                      {matched.at("normal_block_op_out_max_42")->arg()->name});
    if (feature_out_list.size() > 0) {
      op_desc.SetOutput("BlockOutput", feature_out_list);
    }

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

    auto& valid_places = block_first->valid_places();
    auto block_op = LiteOpRegistry::Global().Create(op_desc.Type());
    block_op->Attach(op_desc, scope);
    auto* new_op_node = graph->GraphCreateInstructNode(block_op, valid_places);

    std::vector<std::string> froms = {
        "input",
        REDUCT_BLOCK_VAR_NODE(1) NORMAL_BLOCK_VAR_NODE(11)
            NORMAL_BLOCK_VAR_NODE(12) REDUCT_BLOCK_VAR_NODE(2)
                NORMAL_BLOCK_VAR_NODE(21) NORMAL_BLOCK_VAR_NODE(22)
                    NORMAL_BLOCK_VAR_NODE(23) REDUCT_BLOCK_VAR_NODE(3)
                        NORMAL_BLOCK_VAR_NODE(31) NORMAL_BLOCK_VAR_NODE(32)
                            NORMAL_BLOCK_VAR_NODE(33) NORMAL_BLOCK_VAR_NODE(34)
                                NORMAL_BLOCK_VAR_NODE(35)
                                    REDUCT_BLOCK_VAR_NODE(4)
                                        NORMAL_BLOCK_VAR_NODE(41)
                                            NORMAL_BLOCK_VAR_NODE(42)};
    if (is_resnet101_) {
      std::vector<std::string> resnet101_froms = {
          NORMAL_BLOCK_VAR_NODE(36) NORMAL_BLOCK_VAR_NODE(
              37) NORMAL_BLOCK_VAR_NODE(38) NORMAL_BLOCK_VAR_NODE(39)
              NORMAL_BLOCK_VAR_NODE(310) NORMAL_BLOCK_VAR_NODE(311)
                  NORMAL_BLOCK_VAR_NODE(312) NORMAL_BLOCK_VAR_NODE(313)
                      NORMAL_BLOCK_VAR_NODE(314) NORMAL_BLOCK_VAR_NODE(315)
                          NORMAL_BLOCK_VAR_NODE(316) NORMAL_BLOCK_VAR_NODE(317)
                              NORMAL_BLOCK_VAR_NODE(318)
                                  NORMAL_BLOCK_VAR_NODE(319)
                                      NORMAL_BLOCK_VAR_NODE(320)
                                          NORMAL_BLOCK_VAR_NODE(321)
                                              NORMAL_BLOCK_VAR_NODE(322)};
      froms.insert(froms.end(), resnet101_froms.begin(), resnet101_froms.end());
    }
    for (auto& from : froms) {
      IR_NODE_LINK_TO(matched.at(from), new_op_node);
    }
    IR_NODE_LINK_TO(new_op_node, matched.at("normal_block_op_out_42"));
    IR_NODE_LINK_TO(new_op_node, matched.at("normal_block_op_out_max_42"));

    if (has_stage1_out) {
      IR_NODE_LINK_TO(new_op_node, matched.at("normal_block_op_out_12"));
    }
    if (has_stage2_out) {
      IR_NODE_LINK_TO(new_op_node, matched.at("normal_block_op_out_23"));
    }
    if (is_resnet101_) {
      if (has_stage3_out) {
        IR_NODE_LINK_TO(new_op_node, matched.at("normal_block_op_out_322"));
      }
    } else {
      if (has_stage3_out) {
        IR_NODE_LINK_TO(new_op_node, matched.at("normal_block_op_out_35"));
      }
    }
  }

 private:
  bool has_stage1_out;
  bool has_stage2_out;
  bool has_stage3_out;
  bool is_resnet101_;
};
#undef REDUCT_BLOCK_VAR_NODE
#undef REDUCT_BLOCK_BIAS_NAME
#undef REDUCT_BLOCK_FILTERMAX_NAME
#undef REDUCT_BLOCK_FILTER_NAME
#undef REDUCT_BLOCK_CONNECT
#undef REDUCTION_BLOCK_PATTERN

#undef NORMAL_BLOCK_VAR_NODE
#undef NORMAL_BLOCK_BIAS_NAME
#undef NORMAL_BLOCK_FILTERMAX_NAME
#undef NORMAL_BLOCK_FILTER_NAME
#undef NORMAL_BLOCK_INTERNAL_CONNECT
#undef NORMAL_BLOCK_CONNECT
#undef NORMAL_BLOCK_PATTERN

#undef STR1
#undef STR2
}  // namespace fusion

class XPUResNetLikeFusePass : public ProgramPass {
 public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override {
    // only support conv_num in {5, 4, 3}
    for (int i = 0; i < 16; i++) {
      fusion::XPUResNetLikeFuser fuser((i & 1), (i & 2), (i & 4), (i & 8));
      fuser(graph.get());
    }
  }
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(__xpu__resnet_like_fuse_pass,
                  paddle::lite::mir::XPUResNetLikeFusePass)
    .BindTargets({TARGET(kXPU)})
    .BindKernel("__xpu__block_fuse_op");

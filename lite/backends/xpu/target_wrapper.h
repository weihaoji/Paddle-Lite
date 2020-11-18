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

#pragma once

#include <fcntl.h>
#include <memory>                                 // std::unique_ptr
#include "lite/backends/xpu/xpu_header_sitter.h"  // xpu_free
#include "lite/core/target_wrapper.h"             // TargetWrapper
#include "lite/utils/cp_logging.h"                // CHECK_EQ
#include "lite/utils/macros.h"

#define XPU_CALL(func)                                        \
  {                                                           \
    auto e = (func);                                          \
    CHECK_EQ(e, 0) << "XPU: (" << #func << ") returns " << e; \
  }

namespace paddle {
namespace lite {

// MAX(lod.size()) = 64
const int XPU_MAX_LOD_SIZE = 64;
// MAX(lod[i + 1] - lod[i]) = 512
const int XPU_MAX_LOD_SEQ_LEN = 512;

using TargetWrapperXPU = TargetWrapper<TARGET(kXPU)>;

struct XPUScratchPad {
  XPUScratchPad(void* addr, size_t size, bool is_l3)
      : addr_(addr), size_(size), is_l3_(is_l3) {}

  // XXX(miaotianxiang): |size_| increases monotonically
  void Reserve(size_t new_size);

  void* addr_{nullptr};
  size_t size_{0};
  bool is_l3_{false};
};

struct XPUScratchPadDeleter {
  void operator()(XPUScratchPad* sp) const;
};

using XPUScratchPadGuard = std::unique_ptr<XPUScratchPad, XPUScratchPadDeleter>;

template <>
class TargetWrapper<TARGET(kXPU)> {
 public:
  static size_t num_devices() { return 1; }
  static size_t maximum_stream() { return 0; }

  static void* Malloc(size_t size);
  static void Free(void* ptr);

  static void MemcpySync(void* dst,
                         const void* src,
                         size_t size,
                         IoDirection dir);

  static XPUScratchPadGuard MallocScratchPad(size_t size, bool use_l3 = false);

  static xdnn::Context* GetRawContext() {
    if (tls_raw_ctx_ == nullptr) {
      tls_raw_ctx_ = xdnn::create_context();
      CHECK(tls_raw_ctx_);
      if (workspace_l3_size_per_thread) {
        void* xpu_l3_ptr = nullptr;
        XPU_CALL(xpu_malloc(reinterpret_cast<void**>(&xpu_l3_ptr),
                            workspace_l3_size_per_thread,
                            XPU_MEM_L3));
        if (xpu_l3_ptr != nullptr) {
          tls_raw_ctx_->_l3_mgr.set(xpu_l3_ptr, workspace_l3_size_per_thread);
        } else {
          LOG(WARNING) << "XPU L3 Cache Malloc Fail, Set L3 WorkSpace as 0";
        }
      }

      if (set_xpu_auto_tune) {
        tls_raw_ctx_->_xpu1_conv_selector.set_autotune_loop(true);
        tls_raw_ctx_->_xpu1_conv_selector.set_inference_mode(true);
        const char* autotune_file = std::getenv("XPU_AUTOTUNE_FILE");
        if (autotune_file) {
          tls_raw_ctx_->_xpu1_conv_selector.set_autotune_file(autotune_file);
        }
      }

      // int r = xdnn::set_workspace_l3_size(tls_raw_ctx_,
      //                                    workspace_l3_size_per_thread);

      // if (r != 0) {
      //  LOG(WARNING) << "xdnn::set_workspace_l3_size() failed, r = " << r
      //               << ", workspace_l3_size_per_thread = "
      //               << workspace_l3_size_per_thread;
      //}
    }
    return tls_raw_ctx_;
  }

  // **DEPRECATED**, use xpu_set_device() at the very beginning of each worker
  // thread
  static void SetDev(int dev_no = 0) {
    const char* dev_env = std::getenv("LITE_XPU_DEV");
    if (dev_env) {
      dev_no = atoi(dev_env);
    }
    XPU_CALL(xpu_set_device(dev_no));
  }

  static void LockL3Cache() {
    auto xpu_l3_lock_size_str = std::getenv("XPU_L3_LOCK_SIZE");
    xpu_l3_lock_size = -1;
    xpu_l3_lock_fd = -1;
    struct flock f_lock;
    f_lock.l_whence = 0;
    f_lock.l_len = 0;

    if (xpu_l3_lock_size_str && (std::atoi(xpu_l3_lock_size_str) > 0)) {
      int pd = -1;
      XPU_CALL(xpu_current_device(&pd));
      CHECK(pd > 0) << "Wrong Current XPU Device Num";
      std::string buf = "/opt/xpu_lock" + std::to_string(pd);

      xpu_l3_lock_fd = open(buf.c_str(), O_RDWR);
      CHECK(xpu_l3_lock_fd > 0) << "open " << buf << " failed "
                                << xpu_l3_lock_fd;

      // lock
      f_lock.l_type = F_WRLCK;
      fcntl(xpu_l3_lock_fd, F_SETLKW, &f_lock);
      // check ctx and init
      if (tls_raw_ctx_ == nullptr) {
        tls_raw_ctx_ = xdnn::create_context();
        CHECK(tls_raw_ctx_);

        if (set_xpu_auto_tune) {
          tls_raw_ctx_->_xpu1_conv_selector.set_autotune_loop(true);
          tls_raw_ctx_->_xpu1_conv_selector.set_inference_mode(true);
          const char* autotune_file = std::getenv("XPU_AUTOTUNE_FILE");
          if (autotune_file) {
            tls_raw_ctx_->_xpu1_conv_selector.set_autotune_file(autotune_file);
          }
        }
      }
      // set l3 cache
      void* xpu_l3_ptr = nullptr;
      // free thread level L3 Cache
      xpu_l3_ptr = tls_raw_ctx_->_l3_mgr.get_ptr();
      if (xpu_l3_ptr != nullptr) {
        XPU_CALL(xpu_free(xpu_l3_ptr));
      }
      // malloc process level L3 Cache
      xpu_l3_lock_size = std::atoi(xpu_l3_lock_size_str);
      XPU_CALL(xpu_malloc(
          reinterpret_cast<void**>(&xpu_l3_ptr), xpu_l3_lock_size, XPU_MEM_L3));
      CHECK(xpu_l3_ptr != nullptr)
          << "XPU L3 Cache Malloc Fail, No Enough L3 Cache";
      tls_raw_ctx_->_l3_mgr.set(xpu_l3_ptr, xpu_l3_lock_size);
    }
  }

  static void FreeL3Cache() {
    if (xpu_l3_lock_size > 0) {
      void* l3_ptr = tls_raw_ctx_->_l3_mgr.get_ptr();
      if (l3_ptr != nullptr) {
        XPU_CALL(xpu_free(l3_ptr));
      }
      xpu_l3_lock_size = -1;
      xpu_l3_lock_fd = -1;
      struct flock f_lock;
      f_lock.l_whence = 0;
      f_lock.l_len = 0;
      f_lock.l_type = F_UNLCK;
      fcntl(xpu_l3_lock_fd, F_SETLKW, &f_lock);
      close(xpu_l3_lock_fd);
    }
  }

  static std::string multi_encoder_precision;  // NOLINT
  static int workspace_l3_size_per_thread;
  static bool set_xpu_auto_tune;

 private:
  static LITE_THREAD_LOCAL xdnn::Context* tls_raw_ctx_;
  static int xpu_l3_lock_size;
  static int xpu_l3_lock_fd;
};

}  // namespace lite
}  // namespace paddle

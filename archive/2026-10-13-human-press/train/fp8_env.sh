# Transformer Engine FP8 on an RTX 5090 (sm_120), torch 2.11+cu130, TE 2.20.2.
#
# Build (the torch extension has no wheel; the core library does):
#   SP=<training env>/lib/python3.12/site-packages
#   CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH \
#   CUDNN_PATH=$SP/nvidia/cudnn NVTE_CUDNN_INCLUDE_DIR=$SP/nvidia/cudnn/include \
#   CPATH=$SP/nvidia/cudnn/include:$SP/nvidia/nccl/include:$SP/nvidia/cu13/include:/usr/local/cuda-13.0/include \
#   LIBRARY_PATH=$SP/nvidia/cudnn/lib:$SP/nvidia/nccl/lib:/usr/local/cuda-13.0/lib64 \
#   NVTE_FRAMEWORK=pytorch NVTE_CUDA_ARCHS=120 MAX_JOBS=12 \
#   python -m pip install --no-build-isolation --no-deps \
#       transformer_engine_cu13==2.20.2 transformer_engine==2.20.2 transformer_engine_torch==2.20.2
#   python -m pip install --no-deps onnxscript onnx onnx_ir ml_dtypes
#
# Three things that are not in any instructions:
#   * cudnn.h and nccl.h ship inside torch's own nvidia wheels, not the system
#     CUDA toolkit; without CPATH pointing there the build fails immediately.
#   * `pip install transformer_engine` alone installs an empty meta package.
#     The core library is a separate prebuilt wheel, transformer_engine_cu13.
#   * NVTE_DISABLE_NVRTC=1 is load-bearing. TE compiles some kernels at runtime
#     with NVRTC against nvidia/cu13/include/cuda_fp8.hpp, which declares the
#     E8M0 types MXFP8 uses; this NVRTC cannot parse them, so every quantize call
#     dies with NVRTC_ERROR_COMPILATION. Disabling RTC uses precompiled kernels
#     and FP8 then works. (MXFP8 itself stays unavailable on sm_120.)
SP=/home/hanszhu/vla2-train/env/lib/python3.12/site-packages
export NVTE_DISABLE_NVRTC=1
export LD_LIBRARY_PATH="$SP/nvidia/cudnn/lib:$SP/nvidia/nccl/lib:${LD_LIBRARY_PATH:-}"

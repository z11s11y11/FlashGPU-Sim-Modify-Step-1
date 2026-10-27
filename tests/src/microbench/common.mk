# Shared settings for standalone CUDA microbenchmark Makefiles.
#
# GTest-based microbenchmarks are built by tests/Makefile. The .cu
# calibration binaries under cp_async, mma, tma, and memory use this file.

CUDA_ENV_PATH := $(if $(CUDA_HOME),$(CUDA_HOME),$(CUDA_PATH))
CUDA_DEFAULT_PATH := $(if $(wildcard /usr/local/cuda-12.8/bin/nvcc),/usr/local/cuda-12.8,/usr/local/cuda)
CUDA_INSTALL_PATH ?= $(if $(strip $(CUDA_ENV_PATH)),$(CUDA_ENV_PATH),$(CUDA_DEFAULT_PATH))

# 				$(MAKEFILE_LIST)
# 这是 GNU make 的一个内置变量，里面记录了所有已经被 make 读取的 Makefile 文件名，
# 按读取顺序排列，用空格分隔。
# 				$(lastword ...)
# lastword 是取字符串的最后一个单词。
# 所以 $(lastword $(MAKEFILE_LIST)) 就是最后一个被读取的 Makefile 文件名，
# 通常就是当前正在解析的这个 Makefile。
# 				$(dir ...)
# dir 函数用来取一个文件路径的目录部分，并且会在末尾加上 /
# 				$(abspath ...)
# abspath 把相对路径转换成绝对路径，并规范化路径（去掉多余的 .、.. 等）

MICROBENCH_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
# MICROBENCH_ROOT = FlashGPU_Sim_Baseline/tests/src/microbench
TEST_DIR := $(abspath $(MICROBENCH_ROOT)/../..)
# TEST_DIR = FlashGPU_Sim_Baseline/tests
STANDALONE_BIN_ROOT ?= $(TEST_DIR)/build/bin/microbench
# STANDALONE_BIN_ROOT = ~/FlashGPU_Sim_Baseline/tests/build/bin/microbench
STANDALONE_RUN_ROOT ?= $(TEST_DIR)/run/microbench
# STANDALONE_RUN_ROOT = ~/FlashGPU_Sim_Baseline/tests/run/microbench

NVCC ?= $(CUDA_INSTALL_PATH)/bin/nvcc
NVCC_BIN := $(shell command -v $(NVCC) 2>/dev/null)
CUDA_ROOT := $(if $(strip $(NVCC_BIN)),$(abspath $(dir $(NVCC_BIN))/..),$(CUDA_INSTALL_PATH))
CUOBJDUMP ?= $(CUDA_ROOT)/bin/cuobjdump
NCU ?= ncu

ARCH ?= sm_120a
PTX_PROFILE ?= compute_120a
CUDA_LIB_DIR ?= $(if $(wildcard $(CUDA_ROOT)/lib64/libcudart.so*),$(CUDA_ROOT)/lib64,$(CUDA_ROOT)/lib)

NVCC_FLAGS ?= -std=c++17 -O3 -lineinfo -cudart shared
NVCC_FLAGS += -gencode arch=$(PTX_PROFILE),code=$(ARCH)
NVCC_FLAGS += -gencode arch=$(PTX_PROFILE),code=$(PTX_PROFILE)
NVCC_FLAGS += -L$(CUDA_LIB_DIR) -Xlinker -rpath -Xlinker $(CUDA_LIB_DIR)

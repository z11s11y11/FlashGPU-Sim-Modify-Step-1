# 运行 FlashGPU-Sim 官方矩阵乘教程（Triton GEMM）

仓库自带的矩阵乘示例位于 [`tutorials/triton-gemm/`](tutorials/triton-gemm/)。默认计算规模为 `M=2560, N=64, K=2560`，使用 `SM120_RTX5090` 配置。仓库已附带录制好的输入、参考输出和内核文件，所以按以下步骤回放时**不需要物理 GPU**，也不需要安装 PyTorch、Triton 或 TritonTrace。

## 1. 准备并编译模拟器

在 Bash 终端中执行。此处使用本机的 CUDA 12.8 路径；如果 CUDA Toolkit 安装在别处，请修改 `CUDA_INSTALL_PATH`。编译过模拟器后，新开终端仍需重新设置环境，但可以跳过 `make`。

```bash
cd /home/zhangshuyi/MyExperiment/FlashGPU-Sim-Baseline/FlashGPU-Sim
export CUDA_INSTALL_PATH=/usr/local/cuda
source setup_environment
make -j "$(nproc)"
```

模拟器还需要支持 C++17 和 OpenMP 的 GCC/G++、GNU Make、Python 3、Flex、Bison、zlib 开发头文件和 X11/OpenGL 开发头文件；CUDA Toolkit 需为 12.8 或更新版本。

## 2. 运行官方矩阵乘示例

接着在**同一终端**执行：

```bash
export OMP_NUM_THREADS=8
cd tutorials/triton-gemm
./run.sh
```

`run.sh` 会使用仓库附带的录制数据，复制 `SM120_RTX5090` 配置，编译独立启动程序，并通过 FlashGPU-Sim 回放矩阵乘内核。成功时终端会出现 `Validation PASSED` 和 `gpu_tot_sim_cycle`。完整输出保存在 `tutorials/triton-gemm/run/simulation.log`；如果想查看这两项结果，可执行：

```bash
grep -E 'Validation PASSED|^gpu_tot_sim_cycle' run/simulation.log
```

## 可选：在 RTX 5090 上重新录制矩阵乘

只有需要重新生成录制数据时才执行本节。另开一个**未执行过 `source setup_environment` 的干净终端**，准备好可访问 RTX 5090 的 Python 环境（PyTorch、Triton、NumPy），然后执行：

```bash
cd /home/zhangshuyi/MyExperiment/FlashGPU-Sim-Baseline/FlashGPU-Sim
export CUDA_INSTALL_PATH=/usr/local/cuda
cd tutorials/triton-gemm
python -m pip install -e ../../tools
./capture.sh
```

录制完成后，回到已设置模拟器环境的终端，在 `tutorials/triton-gemm` 目录执行 `./run.sh`。录制日志位于 `run/capture.log`。上述离线回放使用的是仓库附带的默认规模；更改录制规模时应重新录制，不能只修改回放命令。

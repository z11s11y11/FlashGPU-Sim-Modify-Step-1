# 编译 FlashGPU-Sim 并运行 vectorAdd 教程

以下命令在 **同一个 Bash 终端**中依次执行。仓库的 `README.md` 要求 CUDA Toolkit 12.8 或更新版本，以及支持 C++17 和 OpenMP 的 GCC/G++、GNU Make、Flex、Bison、zlib 开发头文件和 X11/OpenGL 开发头文件。运行这个教程不需要物理 GPU。

## 1. 进入仓库并设置 CUDA 环境

本机的 `/usr/local/cuda` 当前指向 CUDA 12.8。如果你在另一台机器运行，请把 `CUDA_INSTALL_PATH` 改为该机器的 CUDA Toolkit 根目录。

```bash
cd /home/zhangshuyi/MyExperiment/FlashGPU-Sim-Baseline/FlashGPU-Sim
export CUDA_INSTALL_PATH=/usr/local/cuda
"$CUDA_INSTALL_PATH/bin/nvcc" --version
source setup_environment
```

确认 `nvcc` 显示的版本至少为 12.8，并且环境脚本输出 `setup_environment succeeded`。以后打开新的终端编译或运行时，需要重新执行本节的 `export` 和 `source` 命令。

## 2. 编译模拟器

仍在仓库根目录执行：

```bash
make -j "$(nproc)"
test -f "lib/$GPGPUSIM_CONFIG/libcudart.so" && echo "模拟器编译产物存在"
```

`make` 应正常结束（退出码为 0）；第二条命令会检查当前环境对应的模拟器运行库。如果 `make` 报缺少编译工具或头文件，请先安装上文列出的依赖，再重新执行编译命令。

## 3. 运行官方 vectorAdd 教程

```bash
export OMP_NUM_THREADS=8
cd tutorials/vectorAdd
./run.sh
```

脚本会复制默认的 `SM120_RTX5090` 配置、使用 `nvcc` 编译 `vectorAdd.cu`，并启动模拟。终端出现 `Test PASSED` 表示功能验证通过；脚本还会显示日志路径。可以用下面的命令再次检查日志：

```bash
grep -F 'Test PASSED' run/simulation.log
grep -F 'gpu_tot_sim_cycle' run/simulation.log
```

完整模拟输出位于 `tutorials/vectorAdd/run/simulation.log`。如果脚本报告找不到模拟器运行库，请确认第 2 步已成功，且当前终端已经执行 `source setup_environment`。

成功在本地编译运行的flashgpu sim源码，后面的修改可以基于此baseline

## 1. 进入仓库并设置 CUDA 环境

本机的 `/usr/local/cuda` 当前指向 CUDA 12.8。如果你在另一台机器运行，请把 `CUDA_INSTALL_PATH` 改为该机器的 CUDA Toolkit 根目录。

```bash
cd /home/zhangshuyi/MyExperiment/FlashGPU-Sim-Baseline/FlashGPU-Sim
export CUDA_INSTALL_PATH=/usr/local/cuda
"$CUDA_INSTALL_PATH/bin/nvcc" --version
source setup_environment
```

## 2. 编译模拟器

仍在仓库根目录执行：

```bash
make -j "$(nproc)"
test -f "lib/$GPGPUSIM_CONFIG/libcudart.so" && echo "模拟器编译产物存在"
```
## 3. 代码运行

运行代码。PTX_SIM_MODE_FUNC=1（纯功能仿真模式）。 
PTX_SIM_MODE_FUNC=0（详细性能仿真模式）这是模拟器的默认模式，旨在精确模拟 GPU 的微架构行为。

OMP_NUM_THREADS=8 可以拿来加速用

```bash
OMP_NUM_THREADS=8 (echo $OMP_NUM_THREADS)
cd tutorials/vectorAdd/
PTX_SIM_MODE_FUNC=1 ./run.sh
```

## 关键的测试代码

FlashGPU_Sim_Baseline/tests/src/microbench/memory 这里面的代码还比较重要的
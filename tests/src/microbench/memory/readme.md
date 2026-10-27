## 运行这些测试文件的代码

首先需要运行FlashGPU_Sim文件夹下面的readme，把模拟器的环境配置起来

# 可以直接使用该目录现成的 Makefile：

cd ~/FlashGPU_Sim_Baseline

<!--编译memory 里面的4个cuda文件 -->
```
make -C tests/src/microbench/memory \
    ARCH=sm_90a \
    PTX_PROFILE=compute_90a \
    all
```

# 编译完成后检查：
检查编译好的可执行文件是不是都出来了

ls -lh tests/build/bin/microbench/memory

# 然后配置 H100 模拟环境：

```
source setup_environment

mkdir -p tests/run/microbench_memory

cp -a configs/SM90_H100/. tests/run/microbench_memory/

cp -a tests/build/bin/microbench/memory/. \tests/run/microbench_memory/

cd tests/run/microbench_memory
```

ls一下，会发现里面有编译完成后的文件如l2_partition_latency_probe_cuda128_sm_90a

# 运行最小规模测试：
加速：OMP_NUM_THREADS=8
--warmup-steps=2048 可能太大了，要运行很久

./l2_latency_bench_cuda128_sm_90a \
    --case=l2-hit \
    --bytes=4096 \
    --steps=16 \
    --samples=8 \
    --warmup-steps=2 \
    --cache-op=cg >/home/zhangshuyi/FlashGPU_Sim_Baseline/tests/src/microbench/memory/reslut.log

<!-- 
  --blocks=84 \ CTA 个数
  --threads=32 \ 每 block 线程数
  --repeats=256 \
  --warmup=64 \
  --bit-first=7 \
  --bit-last=27 \
  --dense-offsets=64 \
  --dense-stride=128 \
  --data-bytes=256M \
  --csv=sim_results.csv \
-->

L2 Partition部分
```
./l2_partition_latency_probe_cuda128_sm_90a \
  --blocks=4 \
  --threads=32 \
  --repeats=32 \
  --warmup=8 \
  --bit-first=7 \
  --bit-last=12 \
  --dense-offsets=8 \
  --data-bytes=4M \
  --csv=sim_results.csv \
  2>&1 | tee sim_run.log
```


## 在真机上测量

/home/zhangshuyi/FlashGPU_Sim_Baseline/tests/src/microbench/memory/l2_partition_latency_probe.cu

# 在真机的 Linux 终端执行：

```
cd /home/zhangshuyi/FlashGPU_Sim_Baseline

# 查看测试源码
sed -n '1,260p' tests/src/microbench/memory/l2_partition_latency_probe.cu

# 查看同目录的测试文件
ls tests/src/microbench/memory/

# 查看测试框架是否已经注册这个测试
grep -R -n "l2_partition_latency_probe" tests/
```

# 第二步：确认本机有可用的 NVIDIA GPU 和 CUDA 编译器

```
如果没有环境的话需要export一下路径
export PATH=$PATH:/usr/local/cuda/bin
nvidia-smi
nvcc --version
```

# 第三步：编译这个测试程序

```
cd /home/zhangshuyi/FlashGPU_Sim_Baseline

mkdir -p tests/run/memory_RealGPU

nvcc -O2 -std=c++17 \
  tests/src/microbench/memory/l2_partition_latency_probe.cu \
  -o tests/run/memory_RealGPU/l2_partition_latency_probe
```

如果编译成功，执行文件就会生成在：

tests/run/memory_RealGPU/l2_partition_latency_probe

# 第四步：运行并保存测量结果

```
cd /home/zhangshuyi/FlashGPU_Sim_Baseline

./tests/run/memory_RealGPU/l2_partition_latency_probe \
  --blocks=4 \
  --threads=32 \
  --repeats=32 \
  --warmup=8 \
  --bit-first=7 \
  --bit-last=17 \
  --dense-offsets=8 \
  --data-bytes=4M \
  --csv=tests/run/memory_RealGPU/results.csv \
  2>&1 | tee tests/run/memory_RealGPU/run.log
```
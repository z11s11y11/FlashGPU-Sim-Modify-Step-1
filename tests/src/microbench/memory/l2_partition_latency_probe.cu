/*
这个文件是一个针对 L2 分区（L2 partition / slice）行为的延迟探测微
基准，用途是标定/验证 FlashGPU-Sim 里 L2 切片划分和地址散列（hash）的建模是否正确。

核心思想：让每个 SM 上的线程对一批精心挑选的地址偏移做串行指针追逐（pointer chase），
用 clock64() 计时，得到一张 (SM, 地址偏移) → 访问延迟 的表。如果 GPU 的 L2 
是按地址哈希分成多个 slice/partition 的，不同偏移的地址会落到不同 slice，从某个 SM 
出发到"近 slice"和"远 slice"的延迟就会不同——从这张表就能反推出哈希函数用了哪些地址位、
每个 SM 到各 slice 的远近关系。

仓库是个 GPU 模拟器（FlashGPU-Sim）。模拟器必须自己实现那套哈希规则，但真芯片的规则
是不公开的。所以作者的思路是：先在真 GPU 上跑这个探测程序拿到真实延迟图，再让模拟器跑
同样的程序——两边图对上了，说明模拟器的 L2 建模是对的。这类程序叫"校准微基准"（
calibration microbenchmark），本身不产生有用计算结果，纯粹是给模拟器当"标尺"用。
*/

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

namespace {

struct Options {
  int device = 0; //CUDA 设备号
  int blocks = 0;
  int threads = 32;
  int repeats = 256;
  int warmup = 64; //	计时前先空跑 64 次追逐，让流水线、TLB、缓存进入稳态，不计入结果
  int bit_first = 7;
  int bit_last = 30;
  int dense_offsets = 256;
  size_t dense_stride = 128;
  size_t base_offset = 0;
  size_t data_bytes = 2ULL * 1024 * 1024 * 1024;
  size_t smem_bytes = 0;
  bool powers_only = false;
  std::string csv_path;
};

struct Sample {
  uint32_t block;
  uint32_t smid;
  uint32_t offset_idx;
  uint64_t offset;
  uint64_t cycles;
  uint64_t sink;
};

#define CUDA_CHECK(expr)                                                 \
  do {                                                                   \
    cudaError_t err__ = (expr);                                          \
    if (err__ != cudaSuccess) {                                          \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(err__));                           \
      std::exit(1);                                                      \
    }                                                                    \
  } while (0)

__device__ __forceinline__ uint32_t smid_now() {
  uint32_t value = 0;
  /*
    %%smid 是 PTX 的一个特殊只读寄存器（special register），
    硬件在每次发射 warp 时会把该 warp 所在 SM 的编号写进去；
    mov.u32 %0, %%smid 就是"把这个 SM 编号读到普通寄存器里"；
    asm volatile 告诉编译器不许优化掉这次读取（否则结果没人用就可能被删）；
  */
  asm volatile("mov.u32 %0, %%smid;" : "=r"(value));
  // mov.u32 %0, %%smid; 里的 %0 最终会被替换成编译器给 value 选的寄存器名（比如 %r1）
  // %0：在汇编模板字符串里，%0、%1…是按操作数列表顺序编号的占位符
  return value;
}

__device__ __forceinline__ uint64_t clock64_now() {
  uint64_t value = 0;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(value)::"memory");
  return value;
}

__device__ __forceinline__ uint64_t gmem_addr_u64(const void* ptr) {
  uint64_t out = 0;
  // cvta.to.global.u64 %0, %1;"：核心 PTX 指令：
  asm volatile("cvta.to.global.u64 %0, %1;" : "=l"(out) : "l"(ptr));
  /*
  cvta：convert address，地址转换。
  .to.global：转换到全局内存状态空间。
  .u64：操作数为 64 位无符号整数。
  %0：输出操作数，即 out。
  %1：输入操作数，即 ptr
  */

  return out;
}

// 这段代码定义了一个 CUDA 设备端强制内联函数，用途是用内联 PTX 汇编
// 从 全局内存 加载一个 64 位无符号整数，并指定缓存策略为 .cg

// uint64_t addr：传入一个 64 位地址。通常这是某个全局内存地址；
__device__ __forceinline__ uint64_t ld_global_cg_u64(uint64_t addr) {

  // uint64_t value = 0;：输出变量，后面的汇编会把加载结果写入它。
  uint64_t value = 0;
  asm volatile("ld.global.cg.u64 %0, [%1];\n"
               : "=l"(value)
               : "l"(addr)
               : "memory");
  return value;
}

__global__ void init_self_pointer_kernel(uint8_t* data,
                                         const uint64_t* offsets,
                                         int num_offsets) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_offsets) return;
  const uint64_t addr = gmem_addr_u64(data + offsets[idx]);
  *reinterpret_cast<uint64_t*>(data + offsets[idx]) = addr;
}

// 比较关键的一个函数 l2_partition_latency_kernel
/*
不只是测量一次 GPU 内存访问需要多少时间，而是想通过改变内存访问地址，
观察不同地址是否映射到不同的 L2 Partition，以及这些分区的访问延迟是否存在差异

最理想的验证方法就是：先在真实 GPU 上运行微基准测试，再在 FlashGPU-Sim 中运行相同的测试，
最后比较两者的结果。
*/
__global__ void l2_partition_latency_kernel(const uint8_t* data,
                                            const uint64_t* offsets,
                                            int num_offsets, int repeats,
                                            int warmup, Sample* samples) {
  const int tid = threadIdx.x;
  // 读取当前线程运行在哪个 SM（流多处理器）上，返回那个 SM 的编号。
  const uint32_t smid = smid_now();

  for (int logical = 0; logical < num_offsets; ++logical) {
    const int offset_idx = (logical + blockIdx.x) % num_offsets;
    const uint64_t offset = offsets[offset_idx];
    uint64_t ptr = gmem_addr_u64(data + offset);
    uint64_t sink = ptr;

    if (tid == 0) {
      for (int i = 0; i < warmup; ++i) {
        // .cg 表示主要走 global/L2 缓存路径，尽量避免 L1 命中干扰测量
        ptr = ld_global_cg_u64(ptr);
      }
      const uint64_t start = clock64_now();
      for (int i = 0; i < repeats; ++i) {
        ptr = ld_global_cg_u64(ptr);
      }
      const uint64_t end = clock64_now();
      // load 的结果 ptr 必须被后续写回使用，计时循环才不会被当成无副作用代码删掉
      sink ^= ptr;

      /*
      out_idx
      把二维下标 (blockIdx.x, offset_idx) 线性化成一维：
      out_idx = block * num_offsets + offset_idx
      */
      const size_t out_idx =
          static_cast<size_t>(blockIdx.x) * static_cast<size_t>(num_offsets) +
          static_cast<size_t>(offset_idx);

      /*
      {
        blockIdx.x,   // block：第几个 CTA
        smid,         // 当前 SM 编号，来自 PTX %smid
        offset_idx,   // 在 offsets 数组中的下标
        offset,       // 相对 data 基址的字节偏移
        end - start,  // repeats 次 pointer-chase load 的总周期数
        sink          // 上面的依赖/校验值
      }
      */
      samples[out_idx] = {static_cast<uint32_t>(blockIdx.x), smid,
                          static_cast<uint32_t>(offset_idx), offset,
                          end - start, sink};
    }
  }
}

void usage(const char* argv0) {
  std::printf(
      "Usage: %s [--device=N] [--blocks=N] [--threads=N]\n"
      "          [--repeats=N] [--warmup=N]\n"
      "          [--bit-first=N] [--bit-last=N] [--dense-offsets=N]\n"
      "          [--dense-stride=N] [--base-offset=N] [--data-bytes=N]\n"
      "          [--smem-bytes=N] [--powers-only] [--csv=path]\n\n"
      "The kernel records a latency vector indexed by (smid, address offset).\n"
      "Offsets include 0, powers of two in [bit-first, bit-last], and a dense\n"
      "stream unless --powers-only is used.\n",
      argv0);
}

bool parse_int_arg(const char* arg, const char* name, int* value) {
  const size_t n = std::strlen(name);
  if (std::strncmp(arg, name, n) == 0 && arg[n] == '=') {
    *value = std::atoi(arg + n + 1);
    return true;
  }
  return false;
}

bool parse_size_arg(const char* arg, const char* name, size_t* value) {
  const size_t n = std::strlen(name);
  if (std::strncmp(arg, name, n) != 0 || arg[n] != '=') return false;
  char* end = nullptr;
  unsigned long long parsed = std::strtoull(arg + n + 1, &end, 0);
  if (end == arg + n + 1) {
    std::fprintf(stderr, "Invalid integer argument: %s\n", arg);
    std::exit(1);
  }
  if (*end == 'k' || *end == 'K') {
    parsed *= 1024ULL;
    ++end;
  } else if (*end == 'm' || *end == 'M') {
    parsed *= 1024ULL * 1024ULL;
    ++end;
  } else if (*end == 'g' || *end == 'G') {
    parsed *= 1024ULL * 1024ULL * 1024ULL;
    ++end;
  }
  if (*end != '\0') {
    std::fprintf(stderr, "Invalid integer argument: %s\n", arg);
    std::exit(1);
  }
  *value = static_cast<size_t>(parsed);
  return true;
}

Options parse_options(int argc, char** argv) {
  Options opts;
  for (int i = 1; i < argc; ++i) {
    const char* arg = argv[i];
    if (std::strcmp(arg, "--help") == 0 || std::strcmp(arg, "-h") == 0) {
      usage(argv[0]);
      std::exit(0);
    } else if (parse_int_arg(arg, "--device", &opts.device) ||
               parse_int_arg(arg, "--blocks", &opts.blocks) ||
               parse_int_arg(arg, "--threads", &opts.threads) ||
               parse_int_arg(arg, "--repeats", &opts.repeats) ||
               parse_int_arg(arg, "--warmup", &opts.warmup) ||
               parse_int_arg(arg, "--bit-first", &opts.bit_first) ||
               parse_int_arg(arg, "--bit-last", &opts.bit_last) ||
               parse_int_arg(arg, "--dense-offsets", &opts.dense_offsets)) {
      continue;
    } else if (parse_size_arg(arg, "--dense-stride", &opts.dense_stride) ||
               parse_size_arg(arg, "--base-offset", &opts.base_offset) ||
               parse_size_arg(arg, "--data-bytes", &opts.data_bytes) ||
               parse_size_arg(arg, "--smem-bytes", &opts.smem_bytes)) {
      continue;
    } else if (std::strcmp(arg, "--powers-only") == 0) {
      opts.powers_only = true;
    } else if (std::strncmp(arg, "--csv=", 6) == 0) {
      opts.csv_path = arg + 6;
    } else {
      std::fprintf(stderr, "Unknown option: %s\n", arg);
      usage(argv[0]);
      std::exit(1);
    }
  }
  if (opts.threads <= 0 || opts.threads > 1024 || opts.threads % 32 != 0 ||
      opts.repeats <= 0 || opts.warmup < 0 || opts.bit_first < 3 ||
      opts.bit_last < opts.bit_first || opts.bit_last >= 62 ||
      opts.dense_offsets < 0 || opts.dense_stride < 8 ||
      opts.data_bytes < 4096) {
    std::fprintf(stderr, "Invalid options\n");
    std::exit(1);
  }
  opts.base_offset -= opts.base_offset % 8;
  opts.dense_stride -= opts.dense_stride % 8;
  opts.smem_bytes -= opts.smem_bytes % 256;
  return opts;
}

std::vector<uint64_t> build_offsets(const Options& opts) {
  std::vector<uint64_t> offsets;
  offsets.push_back(opts.base_offset);
  for (int bit = opts.bit_first; bit <= opts.bit_last; ++bit) {
    offsets.push_back(opts.base_offset + (1ULL << bit));
  }
  if (!opts.powers_only) {
    for (int i = 0; i < opts.dense_offsets; ++i) {
      offsets.push_back(opts.base_offset +
                        static_cast<uint64_t>(i + 1) * opts.dense_stride);
    }
  }
  std::sort(offsets.begin(), offsets.end());
  offsets.erase(std::unique(offsets.begin(), offsets.end()), offsets.end());
  for (uint64_t offset : offsets) {
    if (offset + sizeof(uint64_t) > opts.data_bytes) {
      std::fprintf(stderr,
                   "Offset %llu exceeds data allocation %zu; increase "
                   "--data-bytes or lower bit range\n",
                   static_cast<unsigned long long>(offset), opts.data_bytes);
      std::exit(1);
    }
  }
  return offsets;
}

void write_csv(const std::string& path, const std::vector<Sample>& samples,
               int repeats) {
  std::ofstream out(path);
  if (!out) {
    std::fprintf(stderr, "Failed to open CSV path: %s\n", path.c_str());
    std::exit(1);
  }
  out << "block,smid,offset_idx,offset,cycles,repeats,cycles_per_load,sink\n";
  for (const Sample& s : samples) {
    out << s.block << "," << s.smid << "," << s.offset_idx << ","
        << s.offset << "," << s.cycles << "," << repeats << ","
        << (static_cast<double>(s.cycles) / static_cast<double>(repeats))
        << "," << s.sink << "\n";
  }
}

void print_summary(const std::vector<Sample>& samples, int repeats,
                   int num_offsets) {
  std::vector<uint32_t> smids;
  for (const Sample& s : samples) smids.push_back(s.smid);
  std::sort(smids.begin(), smids.end());
  smids.erase(std::unique(smids.begin(), smids.end()), smids.end());

  double total = 0.0;
  uint64_t min_cycles = UINT64_MAX;
  uint64_t max_cycles = 0;
  for (const Sample& s : samples) {
    // 样本累加
    total += static_cast<double>(s.cycles);
    min_cycles = std::min(min_cycles, s.cycles);
    max_cycles = std::max(max_cycles, s.cycles);
  }
  const double mean = total / static_cast<double>(samples.size());
  std::printf(
      "samples=%zu offsets=%d unique_smids=%zu repeats=%d "
      "cycles_mean=%.2f cycles_per_load_mean=%.3f min=%llu max=%llu\n",
      samples.size(), num_offsets, smids.size(), repeats, mean,
      mean / static_cast<double>(repeats),
      static_cast<unsigned long long>(min_cycles),
      static_cast<unsigned long long>(max_cycles));
}

}  // namespace

int main(int argc, char** argv) {
  const Options opts = parse_options(argc, argv);
  CUDA_CHECK(cudaSetDevice(opts.device));

  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, opts.device));
  int clock_rate_khz = 0;
  CUDA_CHECK(
      cudaDeviceGetAttribute(&clock_rate_khz, cudaDevAttrClockRate, opts.device));
  const int blocks = opts.blocks > 0 ? opts.blocks : prop.multiProcessorCount;
  const std::vector<uint64_t> offsets = build_offsets(opts);

  std::printf("device=%d name=\"%s\" sm=%d.%d sms=%d clockRateKHz=%d\n",
              opts.device, prop.name, prop.major, prop.minor,
              prop.multiProcessorCount, clock_rate_khz);
  std::printf(
      "case=l2-partition-latency blocks=%d threads=%d repeats=%d warmup=%d "
      "offsets=%zu bit_first=%d bit_last=%d dense_offsets=%d "
      "dense_stride=%zu base_offset=%zu data_bytes=%zu smem_bytes=%zu\n",
      blocks, opts.threads, opts.repeats, opts.warmup, offsets.size(),
      opts.bit_first, opts.bit_last, opts.dense_offsets, opts.dense_stride,
      opts.base_offset, opts.data_bytes, opts.smem_bytes);

  uint8_t* d_data = nullptr;
  uint64_t* d_offsets = nullptr;
  Sample* d_samples = nullptr;
  const size_t num_samples =
      static_cast<size_t>(blocks) * static_cast<size_t>(offsets.size());
  CUDA_CHECK(cudaMalloc(&d_data, opts.data_bytes));
  CUDA_CHECK(cudaMalloc(&d_offsets, offsets.size() * sizeof(uint64_t)));
  CUDA_CHECK(cudaMalloc(&d_samples, num_samples * sizeof(Sample)));
  CUDA_CHECK(cudaMemset(d_data, 0, opts.data_bytes));
  CUDA_CHECK(cudaMemcpy(d_offsets, offsets.data(),
                        offsets.size() * sizeof(uint64_t),
                        cudaMemcpyHostToDevice));

  init_self_pointer_kernel<<<(offsets.size() + 255) / 256, 256>>>(
      d_data, d_offsets, static_cast<int>(offsets.size()));
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  if (opts.smem_bytes > 0) {
    CUDA_CHECK(cudaFuncSetAttribute(
        l2_partition_latency_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(opts.smem_bytes)));
    CUDA_CHECK(cudaFuncSetAttribute(l2_partition_latency_kernel,
                                    cudaFuncAttributePreferredSharedMemoryCarveout,
                                    100));
  }

  l2_partition_latency_kernel<<<blocks, opts.threads, opts.smem_bytes>>>(
      d_data, d_offsets, static_cast<int>(offsets.size()), opts.repeats,
      opts.warmup, d_samples);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<Sample> samples(num_samples);
  CUDA_CHECK(cudaMemcpy(samples.data(), d_samples, num_samples * sizeof(Sample),
                        cudaMemcpyDeviceToHost));
  print_summary(samples, opts.repeats, static_cast<int>(offsets.size()));

  if (!opts.csv_path.empty()) {
    write_csv(opts.csv_path, samples, opts.repeats);
    std::printf("csv=%s\n", opts.csv_path.c_str());
  }

  CUDA_CHECK(cudaFree(d_samples));
  CUDA_CHECK(cudaFree(d_offsets));
  CUDA_CHECK(cudaFree(d_data));
  return 0;
}

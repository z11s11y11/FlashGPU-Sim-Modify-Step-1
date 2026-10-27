// Small deterministic workload for read traffic, dirty L2 eviction and replay.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>

#define CUDA_OK(call) do { \
  cudaError_t error = (call); \
  if (error != cudaSuccess) { \
    fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(error)); \
    std::exit(1); \
  } \
} while (0)

__device__ __forceinline__ unsigned load_l2(const unsigned *address) {
  unsigned result;
  asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(result) : "l"(address) : "memory");
  return result;
}
__device__ __forceinline__ void store_l2(unsigned *address, unsigned value) {
  asm volatile("st.global.cg.u32 [%0], %1;" :: "l"(address), "r"(value) : "memory");
}
__global__ void fill(unsigned *data, unsigned count) {
  for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
       i += gridDim.x * blockDim.x)
    store_l2(data + i, load_l2(data + i) + 1u);
}
__global__ void transform(const unsigned *data, unsigned *output, unsigned count) {
  for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
       i += gridDim.x * blockDim.x) {
    unsigned j = (i + 4093u) % count;
    store_l2(output + i, load_l2(data + i) ^ (load_l2(data + j) + 7u));
  }
}
int main() {
  constexpr unsigned count = 32768;
  unsigned *data, *output;
  CUDA_OK(cudaMalloc(&data, count * sizeof(unsigned)));
  CUDA_OK(cudaMalloc(&output, count * sizeof(unsigned)));
  std::vector<unsigned> seed(count);
  for (unsigned i = 0; i < count; ++i) seed[i] = i * 17u + 22u;
  CUDA_OK(cudaMemcpy(data, seed.data(), count * sizeof(unsigned),
                          cudaMemcpyHostToDevice));
  fill<<<16, 128>>>(data, count);
  CUDA_OK(cudaGetLastError());
  CUDA_OK(cudaDeviceSynchronize());
  transform<<<16, 128>>>(data, output, count);
  CUDA_OK(cudaGetLastError());
  CUDA_OK(cudaDeviceSynchronize());
  std::vector<unsigned> actual(count);
  CUDA_OK(cudaMemcpy(actual.data(), output, count * sizeof(unsigned),
                          cudaMemcpyDeviceToHost));
  uint64_t digest = 1469598103934665603ull;
  for (unsigned i = 0; i < count; ++i) {
    unsigned j = (i + 4093u) % count;
    unsigned expected = (i * 17u + 23u) ^ (j * 17u + 23u + 7u);
    if (actual[i] != expected) {
      fprintf(stderr, "Mismatch %u: expected=%u actual=%u\n", i, expected, actual[i]);
      return 1;
    }
    digest = (digest ^ actual[i]) * 1099511628211ull;
  }
  printf("POOL_PATH_PROBE_PASSED digest=%016llx\n",
         static_cast<unsigned long long>(digest));
  CUDA_OK(cudaFree(data));
  CUDA_OK(cudaFree(output));
}

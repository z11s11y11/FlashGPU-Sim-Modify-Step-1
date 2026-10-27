// Exercise the real pool and real DRAM model through two independent clients.
// Both clients deliberately use the same GPU-local subpartition ID.
#include "../libcuda/gpgpu_context.h"
#include "../src/gpgpu-sim/gpu-sim.h"
#include "../src/gpgpu-sim/mem_fetch.h"
#include "../src/gpgpu-sim/mem_latency_stat.h"
#include <cassert>
#include <deque>
#include <set>

class test_client : public memory_pool_client {
 public:
  std::deque<mem_fetch *> requests;
  std::set<mem_fetch *> owned;
  unsigned responses = 0;
  unsigned writebacks = 0;
  bool blocked = false;

  unsigned sub_partition_count() const override { return 1; }
  bool request_empty(unsigned local) const override {
    assert(local == 0);
    return requests.empty();
  }
  mem_fetch *request_top(unsigned local) const override {
    assert(local == 0);
    return requests.front();
  }
  void request_pop(unsigned local) override {
    assert(local == 0);
    requests.pop_front();
  }
  bool response_full(unsigned local) const override {
    assert(local == 0);
    return blocked;
  }
  void response_push(unsigned local, mem_fetch *request) override {
    assert(local == 0 && !blocked);
    assert(owned.erase(request) == 1);  // Also detects cross-client routing.
    assert(request->get_type() == READ_REPLY || request->get_type() == WRITE_ACK);
    delivered.push_back(request);
    ++responses;
  }
  void writeback_done(unsigned local, mem_fetch *request) override {
    assert(local == 0);
    assert(owned.erase(request) == 1);
    ++writebacks;  // DRAM deletes writebacks; the client must not delete them.
  }
  std::vector<mem_fetch *> delivered;
};

static void run_case(gpgpu_context &context, gpgpu_sim &gpu,
                     bool simple, bool separate, unsigned scheduler) {
  memory_config config = *gpu.getMemoryConfig();
  config.simple_dram_model = simple;
  config.seperate_write_queue_enabled = separate;
  config.scheduler_type = static_cast<dram_ctrl_t>(scheduler);
  config.dram_latency = 3;
  config.gpgpu_frfcfs_dram_sched_queue_size = 8;
  config.gpgpu_frfcfs_dram_write_queue_size = 8;
  config.gpgpu_dram_return_queue_size = 1;
  config.write_high_watermark = 1;
  config.write_low_watermark = 1;
  config.gpgpu_memlatency_stat = 0;

  memory_stats_manager_t stats(1, &config, gpu.getShaderCoreConfig(), &gpu);
  memory_pool pool(1, &config, &stats);  // Two clients share one channel.
  test_client first, second;
  test_client *clients[] = {&first, &second};
  const mem_access_type kinds[] = {
      GLOBAL_ACC_R, GLOBAL_ACC_W, L1_WRBK_ACC, L2_WRBK_ACC};
  unsigned index = 0;
  for (auto *client : clients) {
    for (auto kind : kinds) {
      mem_access_t access(kind, 0x10000 + index++ * 256, 32,
                          kind != GLOBAL_ACC_R, &context);
      mem_fetch *request = new mem_fetch(access, nullptr, 0, 8, 0, 0, 0,
                                         &config, 0);
      request->set_chip(0);
      request->set_partition(0);  // Identical IDs across clients are intentional.
      client->requests.push_back(request);
      client->owned.insert(request);
    }
    pool.register_client(0, *client);
  }
  pool.seal();
  unsigned long long tick = 0;
  for (; tick < 10000; ++tick) {
    first.blocked = second.blocked = tick >= 4 && tick < 800;
    pool.cycle(tick, 1000);
    if (tick == 799) assert(pool.busy());
    if (first.requests.empty() && second.requests.empty() && !pool.busy()) break;
  }
  assert(tick < 10000);
  auto counters = pool.get_counters();
  assert(counters.accepted == 8 && counters.responses == 4);
  assert(counters.writebacks == 4 && counters.write_acks == 2);
  assert(counters.l1_writebacks == 2 && counters.l2_writebacks == 2);
  assert(counters.pending == 0 && counters.blocked_responses > 0);
  assert(first.responses == 2 && second.responses == 2);
  assert(first.writebacks == 2 && second.writebacks == 2);
  assert(first.owned.empty() && second.owned.empty());
  for (auto *client : clients) {
    for (auto *request : client->delivered) delete request;
    pool.unregister_client(*client);
  }
  printf("POOL_COMPONENT_PASSED simple=%u separate=%u scheduler=%u ticks=%llu blocked=%llu\n",
         simple, separate, scheduler, tick, counters.blocked_responses);
}

int main() {
  gpgpu_context context;
  gpgpu_sim *gpu = context.gpgpu_ptx_sim_init_perf();
  run_case(context, *gpu, false, false, DRAM_FRFCFS);
  run_case(context, *gpu, false, true, DRAM_FRFCFS);
  run_case(context, *gpu, false, false, DRAM_FIFO);
  run_case(context, *gpu, true, false, DRAM_FRFCFS);
  puts("ALL_POOL_COMPONENT_TESTS_PASSED");
}

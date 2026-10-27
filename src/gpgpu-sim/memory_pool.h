// Pooled ownership and cycle orchestration for simulated DRAM channels.
#ifndef FLASHGPU_MEMORY_POOL_H
#define FLASHGPU_MEMORY_POOL_H

#include <cstdio>
#include <memory>
#include <vector>
#include <zlib.h>
#include "memory_pool_interface.h"

class memory_config;
class memory_stats_manager_t;

class memory_pool {
 public:
  struct counters {
    unsigned long long accepted = 0;
    unsigned long long responses = 0;
    unsigned long long writebacks = 0;
    unsigned long long write_acks = 0;
    unsigned long long l1_writebacks = 0;
    unsigned long long l2_writebacks = 0;
    unsigned long long blocked_responses = 0;
    unsigned long long pending = 0;
  };

  memory_pool(unsigned channel_count, const memory_config *config,
              memory_stats_manager_t *stats);
  ~memory_pool();
  memory_pool(const memory_pool &) = delete;
  memory_pool &operator=(const memory_pool &) = delete;

  // Multiple clients may register against one channel. The single-GPU adapter
  // initially uses the legacy identity mapping. Registration precedes seal().
  void register_client(unsigned channel_id, memory_pool_client &client);
  void unregister_client(memory_pool_client &client);
  void seal();

  // Called exactly once per DRAM clock event by the simulation driver.
  // Keep kernel-local and cumulative time separate to preserve legacy stats.
  void cycle(unsigned long long kernel_cycle, unsigned long long prior_cycles);
  bool busy() const;
  counters get_counters() const;
  void print_summary(FILE *fp) const;

  void print(unsigned channel_id, FILE *fp) const;
  void print_stat(unsigned channel_id, FILE *fp);
  void visualize(unsigned channel_id) const;
  void visualizer_print(unsigned channel_id, gzFile file) const;
  void set_dram_power_stats(unsigned channel_id, unsigned &cmd,
                            unsigned &activity, unsigned &nop, unsigned &act,
                            unsigned &pre, unsigned &rd, unsigned &wr,
                            unsigned &wr_wb, unsigned &req) const;

 private:
  struct channel_state;
  channel_state &channel(unsigned id) const;
  const memory_config *m_config;
  bool m_sealed = false;
  unsigned long long m_current_cycle = 0;
  unsigned long long m_kernel_cycle = 0;
  std::vector<std::unique_ptr<channel_state>> m_channels;
};

#endif

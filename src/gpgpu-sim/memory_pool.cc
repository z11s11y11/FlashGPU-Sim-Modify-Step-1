// The channel pipeline is migrated from l2cache.cc. Its ordering, admission
// limits and legacy single-GPU address mapping are deliberately preserved.
#include "memory_pool.h"

#include <cassert>
#include <list>
#include <unordered_map>
#include "dram.h"
#include "gpu-sim.h"
#include "mem_fetch.h"

struct memory_pool::channel_state : dram_context {
  struct endpoint {
    memory_pool_client *client;
    unsigned local_id;
  };
  struct delayed_request {
    unsigned long long ready_cycle;
    mem_fetch *request;
  };

  channel_state(memory_pool &owner, unsigned id, const memory_config *config,
                memory_stats_manager_t *stats)
      : pool(owner), config(config),
        dram(new dram_t(id, config, stats, this)) {}

  unsigned long long current_cycle() const override {
    return pool.m_current_cycle;
  }
  unsigned long long kernel_cycle() const override {
    return pool.m_kernel_cycle;
  }

  void initialize_credits() {
    private_credit.assign(endpoints.size(), 0);
    last_borrower = endpoints.empty() ? 0 : endpoints.size() - 1;
    shared_limit = static_cast<int>(config->gpgpu_frfcfs_dram_sched_queue_size)
                 + static_cast<int>(config->gpgpu_dram_return_queue_size)
                 - (static_cast<int>(endpoints.size()) - 1);
    if (config->seperate_write_queue_enabled)
      shared_limit += config->gpgpu_frfcfs_dram_write_queue_size;
    if (config->gpgpu_frfcfs_dram_sched_queue_size == 0 ||
        config->gpgpu_dram_return_queue_size == 0)
      shared_limit = 0;
    assert(shared_limit >= 0);
  }

  bool has_credit(unsigned slot) const {
    return private_credit[slot] < 1 || shared_limit == 0 ||
           shared_credit < shared_limit;
  }
  void borrow_credit(unsigned slot) {
    assert(has_credit(slot));
    if (private_credit[slot] < 1)
      ++private_credit[slot];
    else
      ++shared_credit;
    last_borrower = slot;
  }
  void return_credit(unsigned slot) {
    if (private_credit[slot] > 0)
      --private_credit[slot];
    else
      --shared_credit;
    assert(shared_credit >= 0);
  }

  unsigned source(mem_fetch *request) const {
    const auto found = origins.find(request);
    assert(found != origins.end());
    return found->second;
  }

  bool deliver_response(mem_fetch *request) {
    const unsigned slot = source(request);
    const endpoint &destination = endpoints[slot];
    assert(destination.client);
    if (destination.client->response_full(destination.local_id)) {
      ++counts.blocked_responses;
      return false;
    }
    destination.client->response_push(destination.local_id, request);
    request->set_status(IN_PARTITION_DRAM_TO_L2_QUEUE, current_cycle());
    return_credit(slot);
    origins.erase(request);
    ++counts.responses;
    if (request->get_type() == WRITE_ACK) ++counts.write_acks;
    return true;
  }

  void complete_writeback(mem_fetch *request) override {
    const unsigned slot = source(request);
    const endpoint &destination = endpoints[slot];
    assert(destination.client);
    assert(request->get_access_type() == L1_WRBK_ACC ||
           request->get_access_type() == L2_WRBK_ACC);
    return_credit(slot);
    destination.client->writeback_done(destination.local_id, request);
    origins.erase(request);
    ++counts.writebacks;
    if (request->get_access_type() == L1_WRBK_ACC)
      ++counts.l1_writebacks;
    else
      ++counts.l2_writebacks;
    // The DRAM completion path deletes the request after this callback.
  }

  void arbitrate() {
    const unsigned count = endpoints.size();
    const unsigned previous = last_borrower;
    for (unsigned p = 0; p < count; ++p) {
      const unsigned slot = (p + previous + 1) % count;
      const endpoint &candidate = endpoints[slot];
      if (!candidate.client || candidate.client->request_empty(candidate.local_id) ||
          !has_credit(slot) ||
          candidate.client->response_full(candidate.local_id))
        continue;
      mem_fetch *request = candidate.client->request_top(candidate.local_id);
      // Preserve the legacy head-of-line policy: stop on this selected
      // candidate when its read/write scheduler queue is full.
      if (dram->full(request->is_write())) break;
      candidate.client->request_pop(candidate.local_id);
      const bool inserted = origins.emplace(request, slot).second;
      assert(inserted);
      latency_queue.push_back(
          {current_cycle() + config->dram_latency, request});
      request->set_status(IN_PARTITION_DRAM_LATENCY_QUEUE, current_cycle());
      borrow_credit(slot);
      ++counts.accepted;
      break;  // One accepted request per channel per DRAM clock event.
    }
  }

  void cycle() {
    if (config->simple_dram_model) {
      if (!latency_queue.empty() &&
          current_cycle() >= latency_queue.front().ready_cycle) {
        mem_fetch *request = latency_queue.front().request;
        if (request->get_access_type() == L1_WRBK_ACC ||
            request->get_access_type() == L2_WRBK_ACC) {
          complete_writeback(request);
          delete request;
          latency_queue.pop_front();
        } else {
          request->set_reply();
          if (deliver_response(request)) latency_queue.pop_front();
        }
      }
      arbitrate();
    } else {
      // Preserve the legacy sequence within EACH channel:
      // return response -> advance DRAM -> arbitrate -> submit ready request.
      mem_fetch *request = dram->return_queue_top();
      if (!request || deliver_response(request)) dram->return_queue_pop();
      dram->cycle();
      dram->dram_log(SAMPLELOG);
      arbitrate();
      if (!latency_queue.empty() &&
          current_cycle() >= latency_queue.front().ready_cycle &&
          !dram->full(latency_queue.front().request->is_write())) {
        request = latency_queue.front().request;
        latency_queue.pop_front();
        dram->push(request);
      }
    }
    assert(counts.accepted ==
           counts.responses + counts.writebacks + origins.size());
  }

  memory_pool &pool;
  const memory_config *config;
  std::unique_ptr<dram_t> dram;
  std::vector<endpoint> endpoints;
  std::list<delayed_request> latency_queue;
  // Track the actual source independently of channel and GPU-local L2 IDs.
  std::unordered_map<mem_fetch *, unsigned> origins;
  std::vector<int> private_credit;
  unsigned last_borrower = 0;
  int shared_credit = 0;
  int shared_limit = 0;
  counters counts;
};

memory_pool::memory_pool(unsigned count, const memory_config *config,
                         memory_stats_manager_t *stats)
    : m_config(config) {
  assert(config && stats && count);
  for (unsigned id = 0; id < count; ++id)
    m_channels.emplace_back(new channel_state(*this, id, config, stats));
}

memory_pool::~memory_pool() = default;

memory_pool::channel_state &memory_pool::channel(unsigned id) const {
  assert(id < m_channels.size());
  return *m_channels[id];
}

void memory_pool::register_client(unsigned id, memory_pool_client &client) {
  assert(!m_sealed);
  channel_state &state = channel(id);
  for (const auto &ep : state.endpoints) assert(ep.client != &client);
  for (unsigned local = 0; local < client.sub_partition_count(); ++local)
    state.endpoints.push_back({&client, local});
}

void memory_pool::unregister_client(memory_pool_client &client) {
  for (auto &state : m_channels) {
    for (const auto &origin : state->origins)
      assert(state->endpoints[origin.second].client != &client);
    for (auto &ep : state->endpoints)
      if (ep.client == &client) ep.client = nullptr;
  }
}

void memory_pool::seal() {
  assert(!m_sealed);
  for (auto &state : m_channels) state->initialize_credits();
  m_sealed = true;
}

void memory_pool::cycle(unsigned long long kernel,
                        unsigned long long prior) {
  assert(m_sealed);
  m_kernel_cycle = kernel;
  m_current_cycle = kernel + prior;
  for (auto &state : m_channels) state->cycle();
}

bool memory_pool::busy() const {
  for (const auto &state : m_channels)
    if (!state->origins.empty()) return true;
  return false;
}

memory_pool::counters memory_pool::get_counters() const {
  counters total;
  for (const auto &state : m_channels) {
    total.accepted += state->counts.accepted;
    total.responses += state->counts.responses;
    total.writebacks += state->counts.writebacks;
    total.write_acks += state->counts.write_acks;
    total.l1_writebacks += state->counts.l1_writebacks;
    total.l2_writebacks += state->counts.l2_writebacks;
    total.blocked_responses += state->counts.blocked_responses;
    total.pending += state->origins.size();
  }
  return total;
}

void memory_pool::print_summary(FILE *fp) const {
  const counters total = get_counters();
  fprintf(fp, "memory_pool_accepted = %llu\n", total.accepted);
  fprintf(fp, "memory_pool_responses = %llu\n", total.responses);
  fprintf(fp, "memory_pool_writebacks = %llu\n", total.writebacks);
  fprintf(fp, "memory_pool_write_acks = %llu\n", total.write_acks);
  fprintf(fp, "memory_pool_l1_writebacks = %llu\n", total.l1_writebacks);
  fprintf(fp, "memory_pool_l2_writebacks = %llu\n", total.l2_writebacks);
  fprintf(fp, "memory_pool_blocked_responses = %llu\n", total.blocked_responses);
  fprintf(fp, "memory_pool_pending = %llu\n", total.pending);
}

void memory_pool::print(unsigned id, FILE *fp) const {
  const channel_state &state = channel(id);
  fprintf(fp, "In Dram Latency Queue (total = %zd): \n",
          state.latency_queue.size());
  for (const auto &delayed : state.latency_queue) {
    fprintf(fp, "Ready @ %llu - ", delayed.ready_cycle);
    delayed.request->print(fp);
  }
  state.dram->print(fp);
}

void memory_pool::print_stat(unsigned id, FILE *fp) {
  channel(id).dram->print_stat(fp);
}
void memory_pool::visualize(unsigned id) const {
  channel(id).dram->visualize();
}
void memory_pool::visualizer_print(unsigned id, gzFile file) const {
  channel(id).dram->visualizer_print(file);
}
void memory_pool::set_dram_power_stats(
    unsigned id, unsigned &cmd, unsigned &activity, unsigned &nop, unsigned &act,
    unsigned &pre, unsigned &rd, unsigned &wr, unsigned &wr_wb,
    unsigned &req) const {
  channel(id).dram->set_dram_power_stats(cmd, activity, nop, act, pre, rd, wr,
                                        wr_wb, req);
}

// Interfaces at the L2 / pooled DRAM boundary.
#ifndef FLASHGPU_MEMORY_POOL_INTERFACE_H
#define FLASHGPU_MEMORY_POOL_INTERFACE_H

class mem_fetch;

// A client owns its L2 queues and request trackers. The pool borrows this
// interface; clients must outlive their registrations and outstanding requests.
class memory_pool_client {
 public:
  virtual ~memory_pool_client() = default;
  virtual unsigned sub_partition_count() const = 0;
  virtual bool request_empty(unsigned local_id) const = 0;
  virtual mem_fetch *request_top(unsigned local_id) const = 0;
  virtual void request_pop(unsigned local_id) = 0;
  virtual bool response_full(unsigned local_id) const = 0;
  virtual void response_push(unsigned local_id, mem_fetch *request) = 0;
  virtual void writeback_done(unsigned local_id, mem_fetch *request) = 0;
};

// DRAM obtains time and writeback completion through a channel context, without
// depending on a particular GPU or memory_partition_unit.
class dram_context {
 public:
  virtual ~dram_context() = default;
  virtual unsigned long long current_cycle() const = 0;
  virtual unsigned long long kernel_cycle() const = 0;
  virtual void complete_writeback(mem_fetch *request) = 0;
};

#endif

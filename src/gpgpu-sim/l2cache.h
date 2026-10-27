// Copyright (c) 2009-2021, Tor M. Aamodt, Vijay Kandiah, Nikos Hardavellas,
// Mahmoud Khairy, Junrui Pan, Timothy G. Rogers
// The University of British Columbia, Northwestern University, Purdue
// University All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
// this
//    list of conditions and the following disclaimer;
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution;
// 3. Neither the names of The University of British Columbia, Northwestern
//    University nor the names of their contributors may be used to
//    endorse or promote products derived from this software without specific
//    prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#ifndef MC_PARTITION_INCLUDED
#define MC_PARTITION_INCLUDED

#include "../abstract_hardware_model.h"
#include "memory_pool.h"

#include <list>
#include <queue>
#include <vector>

class mem_fetch;

enum mem_sub_partition_full_stat {
  MSP_FULL_ICNT_TO_L2_NOT_ENOUGH_SECTOR_SLOTS = 0,
  MSP_FULL_ICNT_TO_L2_QUEUE_FULL,
  MSP_FULL_ICNT_TO_L2_QUEUE_NEAR_FULL,
  MSP_FULL_L2_DRAM_QUEUE_FULL,
  MSP_FULL_DRAM_L2_QUEUE_FULL,
  MSP_FULL_L2_ICNT_QUEUE_FULL,
  MSP_FULL_L2_DATA_PORT_BUSY,
  MSP_FULL_L2_FILL_PORT_BUSY,
  MSP_FULL_L2_READY_BLOCKED_BY_L2_ICNT_QUEUE,
  NUM_MEM_SUB_PARTITION_FULL_STATS
};

const char *mem_sub_partition_full_stat_str(
    enum mem_sub_partition_full_stat stat);

class partition_mf_allocator : public mem_fetch_allocator {
 public:
  partition_mf_allocator(const memory_config *config) {
    m_memory_config = config;
  }
  virtual mem_fetch *alloc(const class warp_inst_t &inst,
                           const mem_access_t &access,
                           unsigned long long cycle) const {
    abort();
    return NULL;
  }
  virtual mem_fetch *alloc(new_addr_type addr, mem_access_type type,
                           unsigned size, bool wr, unsigned long long cycle,
                           unsigned long long streamID) const;
  virtual mem_fetch *alloc(new_addr_type addr, mem_access_type type,
                           const active_mask_t &active_mask,
                           const mem_access_byte_mask_t &byte_mask,
                           const mem_access_sector_mask_t &sector_mask,
                           unsigned size, bool wr, unsigned long long cycle,
                           unsigned wid, unsigned sid, unsigned tpc,
                           mem_fetch *original_mf,
                           unsigned long long streamID) const;

 private:
  const memory_config *m_memory_config;
};

// GPU-side L2 frontend. DRAM ownership, arbitration and timing live in memory_pool.
class memory_partition_unit : public memory_pool_client {
 public:
  memory_partition_unit(unsigned partition_id, const memory_config *config,
                        memory_stats_manager_t *stats, class gpgpu_sim *gpu,
                        memory_pool &pool, unsigned channel_id);
  ~memory_partition_unit() override;

  bool busy() const;
  void cache_cycle(unsigned cycle);
  void visualizer_print(gzFile visualizer_file) const;
  void print_stat(FILE *fp) { m_pool.print_stat(m_channel_id, fp); }
  void visualize() const { m_pool.visualize(m_channel_id); }
  void print(FILE *fp) const;
  void handle_memcpy_to_gpu(size_t dst_start_addr, unsigned subpart_id,
                            mem_access_sector_mask_t mask);
  class memory_sub_partition *get_sub_partition(int local_id) {
    return m_sub_partition[local_id];
  }
  void set_dram_power_stats(unsigned &n_cmd, unsigned &n_activity,
                            unsigned &n_nop, unsigned &n_act, unsigned &n_pre,
                            unsigned &n_rd, unsigned &n_wr, unsigned &n_wr_WB,
                            unsigned &n_req) const;
  int global_sub_partition_id_to_local_id(int global_id) const;
  unsigned get_mpid() const { return m_id; }

  // memory_pool_client: local IDs belong to this client, not to a DRAM channel.
  unsigned sub_partition_count() const override;
  bool request_empty(unsigned local_id) const override;
  mem_fetch *request_top(unsigned local_id) const override;
  void request_pop(unsigned local_id) override;
  bool response_full(unsigned local_id) const override;
  void response_push(unsigned local_id, mem_fetch *request) override;
  void writeback_done(unsigned local_id, mem_fetch *request) override;

 private:
  unsigned m_id;
  const memory_config *m_config;
  memory_stats_manager_t *m_mem_stats;
  class memory_sub_partition **m_sub_partition;
  class gpgpu_sim *m_gpu;
  memory_pool &m_pool;  // Borrowed; the simulation driver owns the pool.
  unsigned m_channel_id;
};

class memory_sub_partition {
 public:
  memory_sub_partition(unsigned sub_partition_id, const memory_config *config,
                       memory_stats_manager_t *stats, class gpgpu_sim *gpu);
  ~memory_sub_partition();

  // ============ 身份与状态 ============
  unsigned get_id() const { return m_id; }   // 返回全局 subpartition 号（注意是全局 id：
                                             // partition_id * 每 partition 的 subpartition 数 + 局部号，
                                             // icnt 路由和统计都用它——这个"全局编号"思路
                                             // 正是你将来给 mem_fetch 加 gpu_id 时的参照）
  bool busy() const;                         // 本 subpartition 是否还有未完成请求
                                             // （各队列非空或 L2 MSHR 非空）

  // ============ 每周期驱动 ============
  void cache_cycle(unsigned cycle);          // L2 流水：icnt→L2 入队、L2 查找、
                                             // miss 进 m_L2_dram_queue、hit 进 m_L2_icnt_queue

  // ============ 背压与满状态统计 ============
  bool full() const;                         // 入口队列（m_icnt_L2_queue）是否满
  bool full(unsigned size) const;            // 带尺寸版本：判断加上 size 字节后是否超限
  void record_full_state(unsigned size);     // 记录一次"因满被挡"的事件（按 size 分桶）
  void accumulate_full_state_stats(unsigned long long *stats) const;  // 把满状态计数导出累加
  void accumulate_l2_partition_stats(unsigned long long &remote_accesses,
                                     unsigned long long &extra_latency) const;
                                             // 导出"跨 partition 远程访问"次数及其附加延迟
                                             // （ROP 路径中 remote 类事件的统计，用于分析
                                             // 跨 partition 流量的代价）

  // ============ 请求入口/出口（FIFO 操作语义） ============
  void push(class mem_fetch *mf, unsigned long long clock_cycle);
                                             // 入口：互连(icnt)送来的请求入 m_icnt_L2_queue
  class mem_fetch *pop();                    // 出口：取走一个已完成的响应（L2 hit 回填或 DRAM 返回）
  class mem_fetch *top();                    // 窥探出口队列头部（配合 pop 的 peek-pop 协议）
  void set_done(mem_fetch *mf);              // 写回完成回调：释放 mf 在本 subpartition 中
                                             // 占用的资源（经 m_request_tracker 定位）

  // ============ L2 维护操作 ============
  unsigned flushL2();                        // 冲刷 L2：把脏行写回 DRAM，返回 flush 的请求数
  unsigned invalidateL2();                   // 失效整个 L2（丢弃所有行，内核边界/重置用）

  // ============ 到 DRAM 方向的队列接口（解耦的关键接缝！） ============
  bool L2_dram_queue_empty() const;          // L2 miss 队列（m_L2_dram_queue）是否空
  class mem_fetch *L2_dram_queue_top() const;// 窥探队头（memory_pool::cycle
                                             // 从这里取请求，经延迟队列推入 dram_t）
  void L2_dram_queue_pop();                  // 弹出队头

  // ============ DRAM 返回方向的队列接口 ============
  bool dram_L2_queue_full() const;           // DRAM 返回队列（m_dram_L2_queue）是否满
                                             // （是 DRAM→L2 方向的背压信号）
  void dram_L2_queue_push(class mem_fetch *mf);  // dram_t 的 returnq 出队后经此回到 L2 侧

  // ============ 统计输出 ============
  void visualizer_print(gzFile visualizer_file);   // 窗口统计写入可视化 gzip 文件
  void print_cache_stat(unsigned &accesses, unsigned &misses) const;  // 导出 L2 命中/缺失计数
  void print(FILE *fp) const;                      // 打印本 subpartition 的汇总统计

  void accumulate_L2cache_stats(class cache_stats &l2_stats) const;      // 全量 L2 统计导出
  void get_L2cache_sub_stats(struct cache_sub_stats &css) const;         // 当前累计子集

  // Support for getting per-window L2 stats for AerialVision
  void get_L2cache_sub_stats_pw(struct cache_sub_stats_pw &css) const;   // 本窗口 L2 统计（可视化用）
  void clear_L2cache_stats_pw();                                         // 清零窗口统计

  void force_l2_tag_update(new_addr_type addr, unsigned time,
                           mem_access_sector_mask_t mask) {
    m_L2cache->force_tag_access(addr, m_memcpy_cycle_offset + time, mask);
    m_memcpy_cycle_offset += 1;            // cudaMemcpy(H2D)直接改 L2 tag：把目的地址
  }                                        // 的块预置为"已在 L2"，时间戳加 memcpy 周期
                                           // 偏移；每调一次偏移 +1，保证多次 memcpy 的
                                           // tag 时间戳严格递增（替换策略正确性）

 private:
  // ============ 核心数据成员 ============
  unsigned m_id;  //< the global sub partition ID
                  // 全局 subpartition 编号（见 get_id 注释；全局唯一，跨 partition 可比较）
  const memory_config *m_config;           // 内存配置（外部持有）
  class l2_cache *m_L2cache;               // 本 subpartition 的 L2 cache 本体（tag/数据/MSHR）
  class L2interface *m_L2interface;        // L2 与 cache 层次通信的回调接口对象
                                           // （cache 命中/缺失时反向通知本类，friend 互访）
  class gpgpu_sim *m_gpu;                  // 所属 GPU（取时钟/发 icnt 用，解耦时处理）
  partition_mf_allocator *m_mf_allocator;  // mem_fetch 对象分配器（分区域预分配，
                                           // 避免仿真热路径频繁 new/delete）

  // ============ ROP 固定延迟模型（用优先级队列模拟 ROP 单元的处理流水） ============
  // model delay of ROP units with a fixed latency
  struct rop_delay_t {
    unsigned long long ready_cycle;        // 请求"处理完成"的周期
    unsigned long long sequence;           // 入队序号（同周期到期时按 FIFO 顺序出队）
    class mem_fetch *req;                  // 被延迟的请求
  };
  struct rop_delay_compare {               // 优先级队列的比较器：先比 ready_cycle
    bool operator()(const rop_delay_t &lhs, const rop_delay_t &rhs) const {
      if (lhs.ready_cycle != rhs.ready_cycle)
        return lhs.ready_cycle > rhs.ready_cycle;   // ready_cycle 小者先出（堆顶=最早到期）
      return lhs.sequence > rhs.sequence;           // 同周期则序号小者先出（保序）
    }
  };
  typedef std::priority_queue<rop_delay_t, std::vector<rop_delay_t>,
                              rop_delay_compare>
      rop_delay_queue_t;
  rop_delay_queue_t m_rop_local;           // 本地 ROP 延迟队列（常规 L2 miss 路径）
  rop_delay_queue_t m_rop_remote;          // 远程 ROP 延迟队列（跨 partition 的请求，
                                           // 响应要经互连绕回，附加额外延迟）
  unsigned long long m_next_rop_sequence;  // ROP 入队序号发生器

  // ============ 单元间 FIFO（本 subpartition 内部四级流水） ============
  // these are various FIFOs between units within a memory partition
  fifo_pipeline<mem_fetch> *m_icnt_L2_queue;   // ① icnt → L2：互连送来的请求缓冲
  fifo_pipeline<mem_fetch> *m_L2_dram_queue;   // ② L2 → DRAM：L2 miss 等待发往 channel
  fifo_pipeline<mem_fetch> *m_dram_L2_queue;   // ③ DRAM → L2：channel 返回数据缓冲
  fifo_pipeline<mem_fetch> *m_L2_icnt_queue;   // ④ L2 hit → icnt：L2 命中的响应缓冲

  // ============ 满状态与远程访问统计 ============
  unsigned long long
      m_full_state_stats[NUM_MEM_SUB_PARTITION_FULL_STATS];  // 各类"满事件"计数
                                                             // （按尺寸/原因分桶）
  unsigned long long m_l2_partition_remote_accesses;       // 跨 partition 远程访问计数
  unsigned long long m_l2_partition_extra_latency_cycles;  // 远程访问附加延迟周期累加

  class mem_fetch *L2dramout;              // L2→DRAM 路径上的当前出队请求（暂存指针）
  unsigned long long int wb_addr;          // writeback 地址暂存（L2 驱逐脏行时记录）

  class memory_stats_manager_t *m_mem_stats;  // 延迟统计管理器

  std::set<mem_fetch *> m_request_tracker;    // 在途请求追踪表：本 subpartition 发出的
                                              // 所有未完成 mem_fetch，set_done 据此
                                              // 找到自己的请求做收尾（防止悬挂/重复释放）

  friend class L2interface;              // 友元：允许 L2interface 回调时直接访问本类私有成员

  unsigned l2_partition_extra_latency(const mem_fetch *mf) const;
                                           // 计算 mf 应附加的远程访问延迟（remote ROP 用）
  std::vector<mem_fetch *> breakdown_request_to_sector_requests(mem_fetch *mf);
                                           // 把一个请求按 sector 粒度拆成多个 mem_fetch
                                           // （L2 按 32B sector 管理，跨 sector 请求拆分处理）
  void push_rop_delay(mem_fetch *mf, unsigned long long ready_cycle,
                      bool remote);        // 请求入 ROP 延迟队列（remote 决定进哪个队列）
  bool pop_ready_rop(unsigned long long cycle, mem_fetch *&mf);
                                           // 到期请求出队（当前周期 ≥ ready_cycle 才弹出）

  // This is a cycle offset that has to be applied to the l2 accesses to account
  // for the cudamemcpy read/writes. We want GPGPU-Sim to only count cycles for
  // kernel execution but we want cudamemcpy to go through the L2. Everytime an
  // access is made from cudamemcpy this counter is incremented, and when the l2
  // is accessed (in both cudamemcpyies and otherwise) this value is added to the
  // gpgpu-sim cycle counters.
  unsigned m_memcpy_cycle_offset;          // memcpy 周期偏移计数器（原文注释已说明动机：
                                           // 想只统计 kernel 执行周期，又要让 memcpy
                                           // 走 L2 通路；memcpy 产生的 L2 访问时间戳
                                           // 都加上这个偏移，且每次访问 +1 保序）
};

class L2interface : public mem_fetch_interface {
 public:
  L2interface(memory_sub_partition *unit) { m_unit = unit; }
  virtual ~L2interface() {}
  virtual bool full(unsigned size, bool write) const {
    // assume read and write packets all same size
    return m_unit->m_L2_dram_queue->full();
  }
  virtual void push(mem_fetch *mf) {
    mf->set_status(IN_PARTITION_L2_TO_DRAM_QUEUE, 0 /*FIXME*/);
    m_unit->m_L2_dram_queue->push(mf);
  }

 private:
  memory_sub_partition *m_unit;
};

#endif

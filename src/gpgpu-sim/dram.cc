// Copyright (c) 2009-2021, Tor M. Aamodt, Wilson W.L. Fung, Ali Bakhoda,
// Ivan Sham, George L. Yuan, Vijay Kandiah, Nikos Hardavellas,
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

#include "dram.h"
#include "dram_sched.h"
#include "gpu-misc.h"
#include "gpu-sim.h"
#include "hashing.h"
#include "mem_fetch.h"
#include "mem_latency_stat.h"

#ifdef DRAM_VERIFY
int PRINT_CYCLE = 0;
#endif

template class fifo_pipeline<mem_fetch>;
template class fifo_pipeline<dram_req_t>;

dram_t::dram_t(unsigned int partition_id, const memory_config *config,
               memory_stats_manager_t *stats,
               dram_context *context) {

  id = partition_id;                       // 本 DRAM channel（memory partition）编号
  m_context = context;
  m_mem_stats = stats;                     // 访存延迟统计管理器（统计文件输出用）
  m_config = config;                       // DRAM/HBM 配置（时序参数、bank 数等，源自 gpgpusim.config）

  // rowblp：行局部性（row locality）与 bank 级并行度（BLP）统计
  access_num = 0;                          // 已发射命令总数（ACT + RD/WR，行命中率的分子分母）
  hits_num = 0;                            // 行命中次数（ACT 时目标行恰好已打开）
  read_num = 0;                            // 读命令（RD）发射总数
  write_num = 0;                           // 写命令（WR）发射总数
  hits_read_num = 0;                       // 行命中的读命令数
  hits_write_num = 0;                      // 行命中的写命令数
  banks_1time = 0;                         // 单周期并行访问 bank 数的峰值（观测到的 BLP 上限）
  banks_acess_total = 0;                   // 各周期并行访问 bank 数的累加和（BLP 平均值分子）
  banks_acess_total_after = 0;             // 激活后口径的并行 bank 数累加（row locality 折损分析）
  banks_time_ready = 0;                    // 存在 bank 被访问的周期数（BLP 平均值分母）
  banks_access_ready_total = 0;            // 处于可服务状态（无阻塞计数器）的 bank 数累加
  issued_two = 0;                          // 同一周期发射 2 条命令的周期数
  issued_total = 0;                        // 发射命令总次数
  issued_total_row = 0;                    // 行命令（ACT/PRE）发射总数
  issued_total_col = 0;                    // 列命令（RD/WR）发射总数
  last_cmd_cycle = 0;                      // 上次发射任意命令的全局周期（算总线利用率用）
  last_col_cycle = 0;                      // 上次发射列命令的全局周期
  last_read_col_cycle = 0;                 // 上次发射读命令的全局周期
  last_write_col_cycle = 0;                // 上次发射写命令的全局周期
  last_row_cycle = 0;                      // 上次发射行命令的全局周期

  CCDc = 0;                                // tCCD 计数器：列命令最小间隔，倒数到 0 才能发下一条列命令
  RRDc = 0;                                // tRRD 计数器：ACT-ACT 最小间隔（跨 bank）
  RTWc = 0;                                // tRTW 计数器：读→写总线转向惩罚（channel 级，跨 bank 共享）
  WTRc = 0;                                // tWTR 计数器：写→读总线转向惩罚（channel 级，跨 bank 共享）

  wasted_bw_row = 0;                       // 因发 ACT/PRE 占用总线而损失的带宽
  wasted_bw_col = 0;                       // 因读写转向等时序限制损失的列数据传输带宽
  util_bw = 0;                             // 有效数据传输带宽（实际传数据的周期累计）
  idle_bw = 0;                             // 总线完全空闲损失的带宽
  RCDc_limit = 0;                          // 因 tRCD（行激活延迟）阻塞导致总线空转的周期数
  CCDLc_limit = 0;                         // 因 tCCD_L（同 bank group 列命令间隔）阻塞的周期数
  CCDLc_limit_alone = 0;                   // 仅由 tCCD_L 单独造成阻塞的周期数（与其他约束归因去重）
  CCDc_limit = 0;                          // 因 tCCD 阻塞导致空转的周期数
  WTRc_limit = 0;                          // 因 tWTR 阻塞导致空转的周期数
  WTRc_limit_alone = 0;                    // 仅由 tWTR 单独造成阻塞的周期数
  RCDWRc_limit = 0;                        // 因 tRCDWR（读改写延迟）阻塞导致空转的周期数
  RTWc_limit = 0;                          // 因 tRTW 阻塞导致空转的周期数
  RTWc_limit_alone = 0;                    // 仅由 tRTW 单独造成阻塞的周期数
  rwq_limit = 0;                           // 因 rwq（数据总线队列）满而无法发射列命令的次数
  write_to_read_ratio_blp_rw_average = 0;  // 读写混合流下按写读比加权的 BLP 平均值
  bkgrp_parallsim_rw = 0;                  // 读写操作时的 bank group 并行度统计

  rw = READ;  // read mode is default

  bkgrp = (bankgrp_t **)calloc(sizeof(bankgrp_t *), m_config->nbkgrp);
  bkgrp[0] = (bankgrp_t *)calloc(sizeof(bank_t), m_config->nbkgrp);
  for (unsigned i = 1; i < m_config->nbkgrp; i++) {
    bkgrp[i] = bkgrp[0] + i;
  }
  for (unsigned i = 0; i < m_config->nbkgrp; i++) {
    bkgrp[i]->CCDLc = 0;
    bkgrp[i]->RTPLc = 0;
  }

  bk = (bank_t **)calloc(sizeof(bank_t *), m_config->nbk);
  bk[0] = (bank_t *)calloc(sizeof(bank_t), m_config->nbk);
  for (unsigned i = 1; i < m_config->nbk; i++) bk[i] = bk[0] + i;
  for (unsigned i = 0; i < m_config->nbk; i++) {
    bk[i]->state = BANK_IDLE;
    bk[i]->bkgrpindex = i / (m_config->nbk / m_config->nbkgrp);
  }
  prio = 0;

  rwq = new fifo_pipeline<dram_req_t>("rwq", m_config->CL, m_config->CL + 1);
  mrqq = new fifo_pipeline<dram_req_t>("mrqq", 0, 2);

  // returnq的定义
  /*
  fifo_pipeline 定义在 src/gpgpu-sim/delayqueue.h 里

  ····"dramreturnq" 队列的名字字符串，只用于调试打印
  ····0 —— 最小驻留延迟（min_len）fifo_pipeline 不只是个普通 FIFO，它内置了"延迟流水线"语义：元素必须在里面待满 min_len 拍才允许出队。
  ····容量 配置项 -gpgpu_dram_return_queue_size 配了值就用配置值；配 0 时不是"无限"，而是回落到 1024

  */
  returnq = new fifo_pipeline<mem_fetch>(
      "dramreturnq", 0,
      m_config->gpgpu_dram_return_queue_size == 0
          ? 1024
          : m_config->gpgpu_dram_return_queue_size);
  m_frfcfs_scheduler = NULL;
  if (m_config->scheduler_type == DRAM_FRFCFS)
    m_frfcfs_scheduler = new frfcfs_scheduler(m_config, this, stats);
  n_cmd = 0;
  n_activity = 0;
  n_nop = 0;
  n_act = 0;
  n_pre = 0;
  n_rd = 0;
  n_wr = 0;
  n_wr_WB = 0;
  n_rd_L2_A = 0;
  n_req = 0;
  max_mrqs_temp = 0;
  bwutil = 0;
  max_mrqs = 0;
  ave_mrqs = 0;

  for (unsigned i = 0; i < 10; i++) {
    dram_util_bins[i] = 0;
    dram_eff_bins[i] = 0;
  }
  last_n_cmd = last_n_activity = last_bwutil = 0;

  n_cmd_partial = 0;
  n_activity_partial = 0;
  n_nop_partial = 0;
  n_act_partial = 0;
  n_pre_partial = 0;
  n_req_partial = 0;
  ave_mrqs_partial = 0;
  bwutil_partial = 0;

  if (queue_limit())
    mrqq_Dist = StatCreate("mrqq_length", 1, queue_limit());
  else                                             // queue length is unlimited;
    mrqq_Dist = StatCreate("mrqq_length", 1, 64);  // track up to 64 entries
}

bool dram_t::full(bool is_write) const {
  if (m_config->scheduler_type == DRAM_FRFCFS) {
    if (m_config->gpgpu_frfcfs_dram_sched_queue_size == 0) return false; //无限队列
    
    // 读写分队列
    if (m_config->seperate_write_queue_enabled) {
      // 如果是写
      if (is_write)
        return m_frfcfs_scheduler->num_write_pending() >=
               m_config->gpgpu_frfcfs_dram_write_queue_size;
        // gpgpu_frfcfs_dram_write_queue_size 是 memory_config 里的一个配置参数，
        // 定义在 gpu-sim.h、解析在 gpu-sim.cc
        // 作用是：限制调度器里独立的写队列最多能积压多少个写请求

      else
        return m_frfcfs_scheduler->num_pending() >=
               m_config->gpgpu_frfcfs_dram_sched_queue_size;
        // 读队列 容量上限由 gpgpu_frfcfs_dram_sched_queue_size 控制
    } else
      return m_frfcfs_scheduler->num_pending() >=
             m_config->gpgpu_frfcfs_dram_sched_queue_size;
  } else
    return mrqq->full();
}

unsigned dram_t::que_length() const {
  unsigned nreqs = 0;
  if (m_config->scheduler_type == DRAM_FRFCFS) {
    nreqs = m_frfcfs_scheduler->num_pending();
  } else {
    nreqs = mrqq->get_length();
  }
  return nreqs;
}

bool dram_t::returnq_full() const { return returnq->full(); }

unsigned int dram_t::queue_limit() const {
  return m_config->gpgpu_frfcfs_dram_sched_queue_size;
}

dram_req_t::dram_req_t(class mem_fetch *mf, unsigned banks,
                       unsigned dram_bnk_indexing_policy,
                       unsigned long long cycle,
                       unsigned long long kernel_cycle) {
  txbytes = 0; // 已传输字节数：每完成一个 burst 累加 atom_size，达到 nbytes 即服务完成
  dqbytes = 0; // 本周期数据总线已占用字节数：统计/限流用，防止一个周期超额占线
  data = mf; // 原始请求指针：关联的 mem_fetch（含 SM 来源、TLB 等信息）

  const addrdec_t &tlx = mf->get_tlx_addr();

  switch (dram_bnk_indexing_policy) {
    case LINEAR_BK_INDEX: {
      // // bank 编号：请求落在本 channel 内 16 个 bank 中的哪一个
      bk = tlx.bk;
      break;
    }
    case BITWISE_XORING_BK_INDEX: {
      // xoring bank bits with lower bits of the page
      bk = bitwise_hash_function(tlx.row, tlx.bk, banks);
      assert(bk < banks);
      break;
    }
    case IPOLY_BK_INDEX: {
      /*IPOLY for bank indexing function from "Pseudo-randomly interleaved
       * memory." Rau, B. R et al. ISCA 1991
       * http://citeseerx.ist.psu.edu/viewdoc/download;jsessionid=348DEA37A3E440473B3C075EAABC63B6?doi=10.1.1.12.7149&rep=rep1&type=pdf
       */
      // xoring bank bits with lower bits of the page
      bk = ipoly_hash_function(tlx.row, tlx.bk, banks);
      assert(bk < banks);
      break;
    }
    case CUSTOM_BK_INDEX:
      /* No custom set function implemented */
      // Do you custom index here
      break;
    default:
      assert("\nUndefined bank index function.\n" && 0);
      break;
  }

  row = tlx.row; // 目标行地址：请求要访问的 DRAM row（页），ACT 命令打开的对象
  col = tlx.col; // 列地址：row 内的起始 burst 位置，RD/WR 命令的寻址字段
  nbytes = mf->get_data_size(); // 待传输总字节数：本次请求需要搬完的数据量（= access size）

  // 进入 DRAM 的时刻（全局周期）：算 queuing latency 的起点
  timestamp = cycle;
  // 原始物理地址：请求的完整地址，用于统计与调试回溯
  addr = mf->get_addr();
  // 入队时刻：进入 mrqq/调度队列的周期，算调度等待时间
  insertion_time = (unsigned)kernel_cycle;
  // 读写标志：1 = 读请求（要返回数据），0 = 写请求
  rw = data->get_is_write() ? WRITE : READ;
}

void dram_t::push(class mem_fetch *data) {
  assert(id == data->get_tlx_addr()
                   .chip);  // Ensure request is in correct memory partition

  dram_req_t *mrq =
      new dram_req_t(data, m_config->nbk, m_config->dram_bnk_indexing_policy,
                     m_context->current_cycle(), m_context->kernel_cycle());

  data->set_status(IN_PARTITION_MC_INTERFACE_QUEUE,
                   m_context->current_cycle());
  mrqq->push(mrq);

  // stats...
  n_req += 1;
  n_req_partial += 1;
  if (m_config->scheduler_type == DRAM_FRFCFS) {
    unsigned nreqs = m_frfcfs_scheduler->num_pending();
    if (nreqs > max_mrqs_temp) max_mrqs_temp = nreqs;
  } else {
    max_mrqs_temp = (max_mrqs_temp > mrqq->get_length()) ? max_mrqs_temp
                                                         : mrqq->get_length();
  }

  m_mem_stats->get_stats()->memlatstat_dram_access(data);
}

void dram_t::scheduler_fifo() {
  if (!mrqq->empty()) {
    unsigned int bkn;
    dram_req_t *head_mrqq = mrqq->top();
    head_mrqq->data->set_status(
        IN_PARTITION_MC_BANK_ARB_QUEUE,
        m_context->current_cycle());
    bkn = head_mrqq->bk;
    if (!bk[bkn]->mrq) bk[bkn]->mrq = mrqq->pop();
  }
}

#define DEC2ZERO(x) x = (x) ? (x - 1) : 0;
#define SWAP(a, b) \
  a ^= b;          \
  b ^= a;          \
  a ^= b;

void dram_t::cycle() {
  if (!returnq->full()) {
    dram_req_t *cmd = rwq->pop();
    if (cmd) {
#ifdef DRAM_VIEWCMD
      printf("\tDQ: BK%d Row:%03x Col:%03x", cmd->bk, cmd->row,
             cmd->col + cmd->dqbytes);
#endif
      cmd->dqbytes += m_config->dram_atom_size;

      if (cmd->dqbytes >= cmd->nbytes) {
        mem_fetch *data = cmd->data;
        data->set_status(IN_PARTITION_MC_RETURNQ,
                         m_context->current_cycle());
        if (data->get_access_type() != L1_WRBK_ACC &&
            data->get_access_type() != L2_WRBK_ACC) {
          data->set_reply();
          returnq->push(data);
        } else {
          m_context->complete_writeback(data);
          delete data;
        }
        delete cmd;
      }
#ifdef DRAM_VIEWCMD
      printf("\n");
#endif
    }
  }

  /* check if the upcoming request is on an idle bank */
  /* Should we modify this so that multiple requests are checked? */

  switch (m_config->scheduler_type) {
    case DRAM_FIFO:
      scheduler_fifo();
      break;
    case DRAM_FRFCFS:
      scheduler_frfcfs();
      break;
    default:
      printf("Error: Unknown DRAM scheduler type\n");
      assert(0);
  }
  if (m_config->scheduler_type == DRAM_FRFCFS) {
    unsigned nreqs = m_frfcfs_scheduler->num_pending();
    if (nreqs > max_mrqs) {
      max_mrqs = nreqs;
    }
    ave_mrqs += nreqs;
    ave_mrqs_partial += nreqs;
  } else {
    if (mrqq->get_length() > max_mrqs) {
      max_mrqs = mrqq->get_length();
    }
    ave_mrqs += mrqq->get_length();
    ave_mrqs_partial += mrqq->get_length();
  }

  unsigned k = m_config->nbk;
  bool issued = false;

  // collect row buffer locality, BLP and other statistics
  /////////////////////////////////////////////////////////////////////////
  unsigned int memory_pending = 0;
  for (unsigned i = 0; i < m_config->nbk; i++) {
    if (bk[i]->mrq) memory_pending++;
  }
  banks_1time += memory_pending;
  if (memory_pending > 0) banks_acess_total++;

  unsigned int memory_pending_rw = 0;
  unsigned read_blp_rw = 0;
  unsigned write_blp_rw = 0;
  std::bitset<8> bnkgrp_rw_found;  // assume max we have 8 bank groups

  for (unsigned j = 0; j < m_config->nbk; j++) {
    unsigned grp = get_bankgrp_number(j);
    if (bk[j]->mrq &&
        (((bk[j]->curr_row == bk[j]->mrq->row) && (bk[j]->mrq->rw == READ) &&
          (bk[j]->state == BANK_ACTIVE)))) {
      memory_pending_rw++;
      read_blp_rw++;
      bnkgrp_rw_found.set(grp);
    } else if (bk[j]->mrq &&
               (((bk[j]->curr_row == bk[j]->mrq->row) &&
                 (bk[j]->mrq->rw == WRITE) && (bk[j]->state == BANK_ACTIVE)))) {
      memory_pending_rw++;
      write_blp_rw++;
      bnkgrp_rw_found.set(grp);
    }
  }
  banks_time_rw += memory_pending_rw;
  bkgrp_parallsim_rw += bnkgrp_rw_found.count();
  if (memory_pending_rw > 0) {
    write_to_read_ratio_blp_rw_average +=
        (double)write_blp_rw / (write_blp_rw + read_blp_rw);
    banks_access_rw_total++;
  }

  unsigned int memory_Pending_ready = 0;
  for (unsigned j = 0; j < m_config->nbk; j++) {
    unsigned grp = get_bankgrp_number(j);
    if (bk[j]->mrq &&
        ((!CCDc && !bk[j]->RCDc && !(bkgrp[grp]->CCDLc) &&
          (bk[j]->curr_row == bk[j]->mrq->row) && (bk[j]->mrq->rw == READ) &&
          (WTRc == 0) && (bk[j]->state == BANK_ACTIVE) && !rwq->full()) ||
         (!CCDc && !bk[j]->RCDWRc && !(bkgrp[grp]->CCDLc) &&
          (bk[j]->curr_row == bk[j]->mrq->row) && (bk[j]->mrq->rw == WRITE) &&
          (RTWc == 0) && (bk[j]->state == BANK_ACTIVE) && !rwq->full()))) {
      memory_Pending_ready++;
    }
  }
  banks_time_ready += memory_Pending_ready;
  if (memory_Pending_ready > 0) banks_access_ready_total++;
  ///////////////////////////////////////////////////////////////////////////////////

  bool issued_col_cmd = false;
  bool issued_row_cmd = false;

  if (m_config->dual_bus_interface) {
    // dual bus interface
    // issue one row command and one column command
    for (unsigned i = 0; i < m_config->nbk; i++) {
      unsigned j = (i + prio) % m_config->nbk;
      issued_col_cmd = issue_col_command(j);
      if (issued_col_cmd) break;
    }
    for (unsigned i = 0; i < m_config->nbk; i++) {
      unsigned j = (i + prio) % m_config->nbk;
      issued_row_cmd = issue_row_command(j);
      if (issued_row_cmd) break;
    }
    for (unsigned i = 0; i < m_config->nbk; i++) {
      unsigned j = (i + prio) % m_config->nbk;
      if (!bk[j]->mrq) {
        if (!CCDc && !RRDc && !RTWc && !WTRc && !bk[j]->RCDc && !bk[j]->RASc &&
            !bk[j]->RCc && !bk[j]->RPc && !bk[j]->RCDWRc)
          k--;
        bk[j]->n_idle++;
      }
    }
  } else {
    // single bus interface
    // issue only one row/column command
    for (unsigned i = 0; i < m_config->nbk; i++) {
      unsigned j = (i + prio) % m_config->nbk;
      if (!issued_col_cmd) issued_col_cmd = issue_col_command(j);

      if (!issued_col_cmd && !issued_row_cmd)
        issued_row_cmd = issue_row_command(j);

      if (!bk[j]->mrq) {
        if (!CCDc && !RRDc && !RTWc && !WTRc && !bk[j]->RCDc && !bk[j]->RASc &&
            !bk[j]->RCc && !bk[j]->RPc && !bk[j]->RCDWRc)
          k--;
        bk[j]->n_idle++;
      }
    }
  }

  issued = issued_row_cmd || issued_col_cmd;
  if (!issued) {
    n_nop++;
    n_nop_partial++;
#ifdef DRAM_VIEWCMD
    printf("\tNOP                        ");
#endif
  }
  if (k) {
    n_activity++;
    n_activity_partial++;
  }
  n_cmd++;
  n_cmd_partial++;
  if (issued) {
    issued_total++;
    if (issued_col_cmd && issued_row_cmd) issued_two++;
  }
  if (issued_col_cmd) issued_total_col++;
  if (issued_row_cmd) issued_total_row++;

  // Collect some statistics
  // check the limitation, see where BW is wasted?
  /////////////////////////////////////////////////////////
  unsigned int memory_pending_found = 0;
  for (unsigned i = 0; i < m_config->nbk; i++) {
    if (bk[i]->mrq) memory_pending_found++;
  }
  if (memory_pending_found > 0) banks_acess_total_after++;

  bool memory_pending_rw_found = false;
  for (unsigned j = 0; j < m_config->nbk; j++) {
    if (bk[j]->mrq &&
        (((bk[j]->curr_row == bk[j]->mrq->row) && (bk[j]->mrq->rw == READ) &&
          (bk[j]->state == BANK_ACTIVE)) ||
         ((bk[j]->curr_row == bk[j]->mrq->row) && (bk[j]->mrq->rw == WRITE) &&
          (bk[j]->state == BANK_ACTIVE))))
      memory_pending_rw_found = true;
  }

  if (issued_col_cmd || CCDc)
    util_bw++;
  else if (memory_pending_rw_found) {
    wasted_bw_col++;
    for (unsigned j = 0; j < m_config->nbk; j++) {
      unsigned grp = get_bankgrp_number(j);
      // read
      if (bk[j]->mrq &&
          (((bk[j]->curr_row == bk[j]->mrq->row) && (bk[j]->mrq->rw == READ) &&
            (bk[j]->state == BANK_ACTIVE)))) {
        if (bk[j]->RCDc) RCDc_limit++;
        if (bkgrp[grp]->CCDLc) CCDLc_limit++;
        if (WTRc) WTRc_limit++;
        if (CCDc) CCDc_limit++;
        if (rwq->full()) rwq_limit++;
        if (bkgrp[grp]->CCDLc && !WTRc) CCDLc_limit_alone++;
        if (!bkgrp[grp]->CCDLc && WTRc) WTRc_limit_alone++;
      }
      // write
      else if (bk[j]->mrq &&
               ((bk[j]->curr_row == bk[j]->mrq->row) &&
                (bk[j]->mrq->rw == WRITE) && (bk[j]->state == BANK_ACTIVE))) {
        if (bk[j]->RCDWRc) RCDWRc_limit++;
        if (bkgrp[grp]->CCDLc) CCDLc_limit++;
        if (RTWc) RTWc_limit++;
        if (CCDc) CCDc_limit++;
        if (rwq->full()) rwq_limit++;
        if (bkgrp[grp]->CCDLc && !RTWc) CCDLc_limit_alone++;
        if (!bkgrp[grp]->CCDLc && RTWc) RTWc_limit_alone++;
      }
    }
  } else if (memory_pending_found)
    wasted_bw_row++;
  else if (!memory_pending_found)
    idle_bw++;
  else
    assert(1);

  /////////////////////////////////////////////////////////

  // decrements counters once for each time dram_issueCMD is called
  DEC2ZERO(RRDc);
  DEC2ZERO(CCDc);
  DEC2ZERO(RTWc);
  DEC2ZERO(WTRc);
  for (unsigned j = 0; j < m_config->nbk; j++) {
    DEC2ZERO(bk[j]->RCDc);
    DEC2ZERO(bk[j]->RASc);
    DEC2ZERO(bk[j]->RCc);
    DEC2ZERO(bk[j]->RPc);
    DEC2ZERO(bk[j]->RCDWRc);
    DEC2ZERO(bk[j]->WTPc);
    DEC2ZERO(bk[j]->RTPc);
  }
  for (unsigned j = 0; j < m_config->nbkgrp; j++) {
    DEC2ZERO(bkgrp[j]->CCDLc);
    DEC2ZERO(bkgrp[j]->RTPLc);
  }

#ifdef DRAM_VISUALIZE
  visualize();
#endif
}

bool dram_t::issue_col_command(int j) {
  bool issued = false;
  unsigned grp = get_bankgrp_number(j);
  if (bk[j]->mrq) {  // if currently servicing a memory request
    bk[j]->mrq->data->set_status(
        IN_PARTITION_DRAM, m_context->current_cycle());
    // correct row activated for a READ
    if (!issued && !CCDc && !bk[j]->RCDc && !(bkgrp[grp]->CCDLc) &&
        (bk[j]->curr_row == bk[j]->mrq->row) && (bk[j]->mrq->rw == READ) &&
        (WTRc == 0) && (bk[j]->state == BANK_ACTIVE) && !rwq->full()) {
      if (rw == WRITE) {
        rw = READ;
        rwq->set_min_length(m_config->CL);
      }
      rwq->push(bk[j]->mrq);
      bk[j]->mrq->txbytes += m_config->dram_atom_size;
      CCDc = m_config->tCCD;
      bkgrp[grp]->CCDLc = m_config->tCCDL;
      RTWc = m_config->tRTW;
      bk[j]->RTPc = m_config->BL / m_config->data_command_freq_ratio;
      bkgrp[grp]->RTPLc = m_config->tRTPL;
      issued = true;
      if (bk[j]->mrq->data->get_access_type() == L2_WR_ALLOC_R)
        n_rd_L2_A++;
      else
        n_rd++;

      bwutil += m_config->BL / m_config->data_command_freq_ratio;
      bwutil_partial += m_config->BL / m_config->data_command_freq_ratio;
      bk[j]->n_access++;
      last_cmd_cycle = m_context->current_cycle();
      last_col_cycle = last_cmd_cycle;
      last_read_col_cycle = last_cmd_cycle;

#ifdef DRAM_VERIFY
      PRINT_CYCLE = 1;
      printf("\tRD  Bk:%d Row:%03x Col:%03x \n", j, bk[j]->curr_row,
             bk[j]->mrq->col + bk[j]->mrq->txbytes - m_config->dram_atom_size);
#endif
      // transfer done
      if (!(bk[j]->mrq->txbytes < bk[j]->mrq->nbytes)) {
        bk[j]->mrq = NULL;
      }
    } else
      // correct row activated for a WRITE
      if (!issued && !CCDc && !bk[j]->RCDWRc && !(bkgrp[grp]->CCDLc) &&
          (bk[j]->curr_row == bk[j]->mrq->row) && (bk[j]->mrq->rw == WRITE) &&
          (RTWc == 0) && (bk[j]->state == BANK_ACTIVE) && !rwq->full()) {
        if (rw == READ) {
          rw = WRITE;
          rwq->set_min_length(m_config->WL);
        }
        rwq->push(bk[j]->mrq);

        bk[j]->mrq->txbytes += m_config->dram_atom_size;
        CCDc = m_config->tCCD;
        bkgrp[grp]->CCDLc = m_config->tCCDL;
        WTRc = m_config->tWTR;
        bk[j]->WTPc = m_config->tWTP;
        issued = true;

        if (bk[j]->mrq->data->get_access_type() == L2_WRBK_ACC)
          n_wr_WB++;
        else
          n_wr++;
        bwutil += m_config->BL / m_config->data_command_freq_ratio;
        bwutil_partial += m_config->BL / m_config->data_command_freq_ratio;
        last_cmd_cycle = m_context->current_cycle();
        last_col_cycle = last_cmd_cycle;
        last_write_col_cycle = last_cmd_cycle;
#ifdef DRAM_VERIFY
        PRINT_CYCLE = 1;
        printf(
            "\tWR  Bk:%d Row:%03x Col:%03x \n", j, bk[j]->curr_row,
            bk[j]->mrq->col + bk[j]->mrq->txbytes - m_config->dram_atom_size);
#endif
        // transfer done
        if (!(bk[j]->mrq->txbytes < bk[j]->mrq->nbytes)) {
          bk[j]->mrq = NULL;
        }
      }
  }

  return issued;
}

bool dram_t::issue_row_command(int j) {
  bool issued = false;
  unsigned grp = get_bankgrp_number(j);
  if (bk[j]->mrq) {  // if currently servicing a memory request
    bk[j]->mrq->data->set_status(
        IN_PARTITION_DRAM, m_context->current_cycle());
    //     bank is idle
    // else
    if (!issued && !RRDc && (bk[j]->state == BANK_IDLE) && !bk[j]->RPc &&
        !bk[j]->RCc) {  //
#ifdef DRAM_VERIFY
      PRINT_CYCLE = 1;
      printf("\tACT BK:%d NewRow:%03x From:%03x \n", j, bk[j]->mrq->row,
             bk[j]->curr_row);
#endif
      // activate the row with current memory request
      bk[j]->curr_row = bk[j]->mrq->row;
      bk[j]->state = BANK_ACTIVE;
      RRDc = m_config->tRRD;
      bk[j]->RCDc = m_config->tRCD;
      bk[j]->RCDWRc = m_config->tRCDWR;
      bk[j]->RASc = m_config->tRAS;
      bk[j]->RCc = m_config->tRC;
      prio = (j + 1) % m_config->nbk;
      issued = true;
      n_act_partial++;
      n_act++;
      last_cmd_cycle = m_context->current_cycle();
      last_row_cycle = last_cmd_cycle;
    }

    else
      // different row activated
      if ((!issued) && (bk[j]->curr_row != bk[j]->mrq->row) &&
          (bk[j]->state == BANK_ACTIVE) &&
          (!bk[j]->RASc && !bk[j]->WTPc && !bk[j]->RTPc &&
           !bkgrp[grp]->RTPLc)) {
        // make the bank idle again
        bk[j]->state = BANK_IDLE;
        bk[j]->RPc = m_config->tRP;
        prio = (j + 1) % m_config->nbk;
        issued = true;
        n_pre++;
        n_pre_partial++;
        last_cmd_cycle = m_context->current_cycle();
        last_row_cycle = last_cmd_cycle;
#ifdef DRAM_VERIFY
        PRINT_CYCLE = 1;
        printf("\tPRE BK:%d Row:%03x \n", j, bk[j]->curr_row);
#endif
      }
  }
  return issued;
}

// if mrq is being serviced by dram, gets popped after CL latency fulfilled
class mem_fetch *dram_t::return_queue_pop() { return returnq->pop(); }

class mem_fetch *dram_t::return_queue_top() { return returnq->top(); }

void dram_t::print(FILE *simFile) const {
  unsigned i;
  fprintf(simFile, "DRAM[%d]: %d bks, busW=%d BL=%d CL=%d, ", id, m_config->nbk,
          m_config->busW, m_config->BL, m_config->CL);
  fprintf(simFile, "tRRD=%d tCCD=%d, tRCD=%d tRAS=%d tRP=%d tRC=%d\n",
          m_config->tRRD, m_config->tCCD, m_config->tRCD, m_config->tRAS,
          m_config->tRP, m_config->tRC);
  fprintf(
      simFile,
      "n_cmd=%llu n_nop=%llu n_act=%llu n_pre=%llu n_ref_event=%llu n_req=%llu "
      "n_rd=%llu n_rd_L2_A=%llu n_write=%llu n_wr_bk=%llu bw_util=%.4g\n",
      n_cmd, n_nop, n_act, n_pre, n_ref, n_req, n_rd, n_rd_L2_A, n_wr, n_wr_WB,
      (float)bwutil / n_cmd);
  fprintf(simFile, "n_activity=%llu dram_eff=%.4g\n", n_activity,
          (float)bwutil / n_activity);
  fprintf(simFile,
          "last_cmd_cycle=%llu last_col_cycle=%llu last_read_col_cycle=%llu "
          "last_write_col_cycle=%llu last_row_cycle=%llu\n",
          last_cmd_cycle, last_col_cycle, last_read_col_cycle,
          last_write_col_cycle, last_row_cycle);
  for (i = 0; i < m_config->nbk; i++) {
    fprintf(simFile, "bk%d: %da %di ", i, bk[i]->n_access, bk[i]->n_idle);
  }
  fprintf(simFile, "\n");
  fprintf(simFile,
          "\n------------------------------------------------------------------"
          "------\n");

  printf("\nRow_Buffer_Locality = %.6f", (float)hits_num / access_num);
  printf("\nRow_Buffer_Locality_read = %.6f", (float)hits_read_num / read_num);
  printf("\nRow_Buffer_Locality_write = %.6f",
         (float)hits_write_num / write_num);
  printf("\nBank_Level_Parallism = %.6f",
         (float)banks_1time / banks_acess_total);
  printf("\nBank_Level_Parallism_Col = %.6f",
         (float)banks_time_rw / banks_access_rw_total);
  printf("\nBank_Level_Parallism_Ready = %.6f",
         (float)banks_time_ready / banks_access_ready_total);
  printf("\nwrite_to_read_ratio_blp_rw_average = %.6f",
         write_to_read_ratio_blp_rw_average / banks_access_rw_total);
  printf("\nGrpLevelPara = %.6f \n",
         (float)bkgrp_parallsim_rw / banks_access_rw_total);

  printf("\nBW Util details:\n");
  printf("bwutil = %.6f \n", (float)bwutil / n_cmd);
  printf("total_CMD = %llu \n", n_cmd);
  printf("util_bw = %llu \n", util_bw);
  printf("Wasted_Col = %llu \n", wasted_bw_col);
  printf("Wasted_Row = %llu \n", wasted_bw_row);
  printf("Idle = %llu \n", idle_bw);

  printf("\nBW Util Bottlenecks: \n");
  printf("RCDc_limit = %llu \n", RCDc_limit);
  printf("RCDWRc_limit = %llu \n", RCDWRc_limit);
  printf("WTRc_limit = %llu \n", WTRc_limit);
  printf("RTWc_limit = %llu \n", RTWc_limit);
  printf("CCDLc_limit = %llu \n", CCDLc_limit);
  printf("rwq = %llu \n", rwq_limit);
  printf("CCDLc_limit_alone = %llu \n", CCDLc_limit_alone);
  printf("WTRc_limit_alone = %llu \n", WTRc_limit_alone);
  printf("RTWc_limit_alone = %llu \n", RTWc_limit_alone);

  printf("\nCommands details: \n");
  printf("total_CMD = %llu \n", n_cmd);
  printf("n_nop = %llu \n", n_nop);
  printf("Read = %llu \n", n_rd);
  printf("Write = %llu \n", n_wr);
  printf("L2_Alloc = %llu \n", n_rd_L2_A);
  printf("L2_WB = %llu \n", n_wr_WB);
  printf("n_act = %llu \n", n_act);
  printf("n_pre = %llu \n", n_pre);
  printf("n_ref = %llu \n", n_ref);
  printf("n_req = %llu \n", n_req);
  printf("total_req = %llu \n", n_rd + n_wr + n_rd_L2_A + n_wr_WB);

  printf("\nDual Bus Interface Util: \n");
  printf("issued_total_row = %llu \n", issued_total_row);
  printf("issued_total_col = %llu \n", issued_total_col);
  printf("Row_Bus_Util =  %.6f \n", (float)issued_total_row / n_cmd);
  printf("CoL_Bus_Util = %.6f \n", (float)issued_total_col / n_cmd);
  printf("Either_Row_CoL_Bus_Util = %.6f \n", (float)issued_total / n_cmd);
  printf("Issued_on_Two_Bus_Simul_Util = %.6f \n", (float)issued_two / n_cmd);
  printf("issued_two_Eff = %.6f \n", (float)issued_two / issued_total);
  printf("queue_avg = %.6f \n\n", (float)ave_mrqs / n_cmd);

  fprintf(simFile, "\n");
  fprintf(simFile, "dram_util_bins:");
  for (i = 0; i < 10; i++) fprintf(simFile, " %d", dram_util_bins[i]);
  fprintf(simFile, "\ndram_eff_bins:");
  for (i = 0; i < 10; i++) fprintf(simFile, " %d", dram_eff_bins[i]);
  fprintf(simFile, "\n");
  if (m_config->scheduler_type == DRAM_FRFCFS)
    fprintf(simFile, "mrqq: max=%d avg=%g\n", max_mrqs,
            (float)ave_mrqs / n_cmd);
}

void dram_t::visualize() const {
  printf("RRDc=%d CCDc=%d mrqq.Length=%d rwq.Length=%d\n", RRDc, CCDc,
         mrqq->get_length(), rwq->get_length());
  for (unsigned i = 0; i < m_config->nbk; i++) {
    printf("BK%d: state=%c curr_row=%03x, %2d %2d %2d %2d %p ", i, bk[i]->state,
           bk[i]->curr_row, bk[i]->RCDc, bk[i]->RASc, bk[i]->RPc, bk[i]->RCc,
           bk[i]->mrq);
    if (bk[i]->mrq)
      printf("txf: %d %d", bk[i]->mrq->nbytes, bk[i]->mrq->txbytes);
    printf("\n");
  }
  if (m_frfcfs_scheduler) m_frfcfs_scheduler->print(stdout);
}

void dram_t::print_stat(FILE *simFile) {
  fprintf(simFile,
          "DRAM (%u): n_cmd=%llu n_nop=%llu n_act=%llu n_pre=%llu n_ref=%llu "
          "n_req=%llu n_rd=%llu n_write=%llu bw_util=%.4g ",
          id, n_cmd, n_nop, n_act, n_pre, n_ref, n_req, n_rd, n_wr,
          (float)bwutil / n_cmd);
  fprintf(simFile, "mrqq: %d %.4g mrqsmax=%llu ", max_mrqs,
          (float)ave_mrqs / n_cmd, max_mrqs_temp);
  fprintf(simFile, "\n");
  fprintf(simFile, "dram_util_bins:");
  for (unsigned i = 0; i < 10; i++) fprintf(simFile, " %d", dram_util_bins[i]);
  fprintf(simFile, "\ndram_eff_bins:");
  for (unsigned i = 0; i < 10; i++) fprintf(simFile, " %d", dram_eff_bins[i]);
  fprintf(simFile, "\n");
  max_mrqs_temp = 0;
}

void dram_t::visualizer_print(gzFile visualizer_file) {
  // dram specific statistics
  gzprintf(visualizer_file, "dramncmd: %u %u\n", id, n_cmd_partial);
  gzprintf(visualizer_file, "dramnop: %u %u\n", id, n_nop_partial);
  gzprintf(visualizer_file, "dramnact: %u %u\n", id, n_act_partial);
  gzprintf(visualizer_file, "dramnpre: %u %u\n", id, n_pre_partial);
  gzprintf(visualizer_file, "dramnreq: %u %u\n", id, n_req_partial);
  gzprintf(visualizer_file, "dramavemrqs: %u %u\n", id,
           n_cmd_partial ? (ave_mrqs_partial / n_cmd_partial) : 0);

  // utilization and efficiency
  gzprintf(visualizer_file, "dramutil: %u %u\n", id,
           n_cmd_partial ? 100 * bwutil_partial / n_cmd_partial : 0);
  gzprintf(visualizer_file, "drameff: %u %u\n", id,
           n_activity_partial ? 100 * bwutil_partial / n_activity_partial : 0);

  // reset for next interval
  bwutil_partial = 0;
  n_activity_partial = 0;
  ave_mrqs_partial = 0;
  n_cmd_partial = 0;
  n_nop_partial = 0;
  n_act_partial = 0;
  n_pre_partial = 0;
  n_req_partial = 0;

  // dram access type classification
  auto mem_stats = m_mem_stats->get_stats();
  for (unsigned j = 0; j < m_config->nbk; j++) {
    gzprintf(visualizer_file, "dramglobal_acc_r: %u %u %u\n", id, j,
             mem_stats->mem_access_type_stats[GLOBAL_ACC_R][id][j]);
    gzprintf(visualizer_file, "dramglobal_acc_w: %u %u %u\n", id, j,
             mem_stats->mem_access_type_stats[GLOBAL_ACC_W][id][j]);
    gzprintf(visualizer_file, "dramlocal_acc_r: %u %u %u\n", id, j,
             mem_stats->mem_access_type_stats[LOCAL_ACC_R][id][j]);
    gzprintf(visualizer_file, "dramlocal_acc_w: %u %u %u\n", id, j,
             mem_stats->mem_access_type_stats[LOCAL_ACC_W][id][j]);
    gzprintf(visualizer_file, "dramconst_acc_r: %u %u %u\n", id, j,
             mem_stats->mem_access_type_stats[CONST_ACC_R][id][j]);
    gzprintf(visualizer_file, "dramtexture_acc_r: %u %u %u\n", id, j,
             mem_stats->mem_access_type_stats[TEXTURE_ACC_R][id][j]);
  }
}

void dram_t::set_dram_power_stats(unsigned &cmd, unsigned &activity,
                                  unsigned &nop, unsigned &act, unsigned &pre,
                                  unsigned &rd, unsigned &wr, unsigned &wr_WB,
                                  unsigned &req) const {
  // Point power performance counters to low-level DRAM counters
  cmd = n_cmd;
  activity = n_activity;
  nop = n_nop;
  act = n_act;
  pre = n_pre;
  rd = n_rd;
  wr = n_wr;
  wr_WB = n_wr_WB;
  req = n_req;
}

unsigned dram_t::get_bankgrp_number(unsigned i) {
  if (m_config->dram_bnkgrp_indexing_policy == HIGHER_BITS) {  // higher bits
    return i >> m_config->bk_tag_length;
  } else if (m_config->dram_bnkgrp_indexing_policy ==
             LOWER_BITS) {  // lower bits
    return i & ((m_config->nbkgrp - 1));
  } else {
    assert(1);
  }
  return 0;  // we should never get here
}

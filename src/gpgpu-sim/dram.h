// Copyright (c) 2009-2021, Tor M. Aamodt, Ivan Sham, Ali Bakhoda,
// George L. Yuan, Wilson W.L. Fung, Vijay Kandiah, Nikos Hardavellas,
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

#ifndef DRAM_H
#define DRAM_H

#include <stdio.h>
#include <stdlib.h>
#include <zlib.h>
#include <bitset>
#include <fstream>
#include <iomanip>
#include <set>
#include <sstream>
#include <string>
#include <vector>
#include "delayqueue.h"
#include "memory_pool_interface.h"

#define READ 'R'  // define read and write states
#define WRITE 'W'
#define BANK_IDLE 'I'
#define BANK_ACTIVE 'A'

class memory_stats_manager_t;

class dram_req_t {
 public:
  dram_req_t(class mem_fetch *data, unsigned banks,
             unsigned dram_bnk_indexing_policy, unsigned long long cycle,
             unsigned long long kernel_cycle);

  unsigned int row;             // 目标行地址：请求要访问的 DRAM row（页），ACT 命令打开的对象
  unsigned int col;             // 列地址：row 内的起始 burst 位置，RD/WR 命令的寻址字段
  unsigned int bk;              // bank 编号：请求落在本 channel 内 16 个 bank 中的哪一个
  unsigned int nbytes;          // 待传输总字节数：本次请求需要搬完的数据量（= access size）
  unsigned int txbytes;         // 已传输字节数：每完成一个 burst 累加 atom_size，达到 nbytes 即服务完成
  unsigned int dqbytes;         // 本周期数据总线已占用字节数：统计/限流用，防止一个周期超额占线
  unsigned int age;             // 请求年龄：每周期 +1，调度器用来识别"等太久"的请求（防饿死）
  unsigned int timestamp;       // 进入 DRAM 的时刻（全局周期）：算 queuing latency 的起点
  unsigned char rw;             // 读写标志：1 = 读请求（要返回数据），0 = 写请求
  unsigned long long int addr;  // 原始物理地址：请求的完整地址，用于统计与调试回溯
  unsigned int insertion_time;  // 入队时刻：进入 mrqq/调度队列的周期，算调度等待时间
  class mem_fetch *data;        // 原始请求指针：关联的 mem_fetch（含 SM 来源、TLB 等信息）
};

struct bankgrp_t {
  unsigned int CCDLc;
  unsigned int RTPLc;
};

struct bank_t {
  unsigned int RCDc;
  unsigned int RCDWRc;
  unsigned int RASc;
  unsigned int RPc;
  unsigned int RCc;
  unsigned int WTPc;  // write to precharge
  unsigned int RTPc;  // read to precharge

  unsigned char rw;     // is the bank reading or writing?
  unsigned char state;  // is the bank active or idle?
  unsigned int curr_row;

  dram_req_t *mrq;

  unsigned int n_access;
  unsigned int n_writes;
  unsigned int n_idle;

  unsigned int bkgrpindex;
};

enum bank_index_function {
  LINEAR_BK_INDEX = 0,
  BITWISE_XORING_BK_INDEX,
  IPOLY_BK_INDEX,
  CUSTOM_BK_INDEX
};

enum bank_grp_bits_position { HIGHER_BITS = 0, LOWER_BITS };

class mem_fetch;
class memory_config;

class dram_t {
 public:
  dram_t(unsigned int parition_id, const memory_config *config,
         memory_stats_manager_t *stats, dram_context *context);

  // 判断 DRAM channel 还能不能接收新请求，是调用方决定是否 throttle 的开关
  bool full(bool is_write) const;


  void print(FILE *simFile) const;
  void visualize() const;
  void print_stat(FILE *simFile);
  unsigned que_length() const;
  bool returnq_full() const;
  unsigned int queue_limit() const;
  void visualizer_print(gzFile visualizer_file);

  // 这两行是 dram_t 的响应出口，负责把已经服务完的读请求交还给上层
  class mem_fetch *return_queue_pop();
  class mem_fetch *return_queue_top();

  // ============ 对外接口（仿真循环每周期驱动 / 上游调用） ============
  void push(class mem_fetch *data);   // 请求入口：接收经 L2 miss 后到达本 channel 的访存请求，
                                      // 内部创建 dram_req_t 并入 mrqq（注意 class 关键字是
                                      // mem_fetch 的前置声明写法，本头文件不 include mem_fetch.h）
  void cycle();                       // 时序推进：每仿真周期调用一次，驱动 bank 计数器递减
                                      // （DEC2ZERO）、尝试发射行/列命令、推进 rwq 传输、
                                      // 把完成的读请求放入 returnq、写回请求直接 set_done
  void dram_log(int task);            // 调试日志：受 DYNAMIC_LOG 宏控制的 printf 开关，
                                      // task 区分要打印哪类信息（命令、队列状态等），
                                      // 正常仿真为空操作

  dram_context *m_context;  // Borrowed channel context owned by the pool.
  unsigned int id;                    // 本 DRAM channel 的编号（= memory partition 编号），
                                      // 由 gpgpu_sim 创建时传入（0 ~ gpgpu_n_mem-1），
                                      // push() 里的 assert(id == tlx.chip) 用它校验
                                      // 请求确实被地址解码路由到了正确的 channel

  // Power Model
  void set_dram_power_stats(unsigned &cmd, unsigned &activity, unsigned &nop,
                            unsigned &act, unsigned &pre, unsigned &rd,
                            unsigned &wr, unsigned &wr_WB, unsigned &req) const;

  const memory_config *m_config;

 private:
  bankgrp_t **bkgrp;

  bank_t **bk;
  unsigned int prio;

  unsigned get_bankgrp_number(unsigned i);

  void scheduler_fifo();
  void scheduler_frfcfs();

  bool issue_col_command(int j);
  bool issue_row_command(int j);

  // ============ DRAM 时序约束计数器（每周期 DEC2ZERO 递减，非零即封锁对应命令） ============
  unsigned int RRDc;        // tRRD 计数器：两个 ACT（行激活）命令之间的最小间隔
  unsigned int CCDc;        // tCCD 计数器：两条列命令（RD/WR）之间的最小间隔
  unsigned int RTWc;        // read to write penalty applies across banks
                            // tRTW 计数器：读→写总线转向惩罚（channel 级，跨 bank 共享）
  unsigned int WTRc;        // write to read penalty applies across banks
                            // tWTR 计数器：写→读总线转向惩罚（channel 级，跨 bank 共享）

  unsigned char
      rw;                   // was last request a read or write? (important for RTW, WTR)
                            // 总线方向记录：最近一次列命令是读(1)还是写(0)，
                            // 发射反向命令时用于判断/触发 RTWc、WTRc 计数器装载

  unsigned int pending_writes;  // 等待中的写请求数（进入 WRITE_MODE 排水的判断依据之一）

  // ============ 核心数据队列 ============
  fifo_pipeline<dram_req_t> *rwq;   // 数据总线队列：已发 RD/WR 命令在此等待固定延迟
                                    // （min_len = CL），模拟 burst 在总线上的传输耗时，
                                    // 弹出时累加 txbytes，传完进入 returnq
  fifo_pipeline<dram_req_t> *mrqq;  // MC 入口队列：push() 放入的新请求先在此排队，
                                    // 深度仅 2，随后被调度器（FIFO/FR-FCFS）取走分发到各 bank
  // buffer to hold packets when DRAM processing is over
  // should be filled with dram clock and popped with l2or icnt clock
  fifo_pipeline<mem_fetch> *returnq;  // 返回队列：读请求服务完成后（set_reply）在此
                                      // 等待 memory_partition_unit 取走送回 SM；
                                      // 注释是旧版多时钟设计的遗留说法，本代码库单时钟

  // ============ DRAM 利用率分桶（visualizer 画时间分布用） ============
  unsigned int dram_util_bins[10];   // 总线利用率直方图：按利用率 0~100% 分 10 桶统计周期数
  unsigned int dram_eff_bins[10];    // 有效带宽直方图：排除行命令开销后的有效利用率分桶
  unsigned int last_n_cmd,           // 上一统计窗口的命令数快照
               last_n_activity,      // 上一窗口的活动周期数快照
               last_bwutil;          // 上一窗口的带宽利用率快照（供 visualizer 算增量）

  // ============ 命令计数（print_stat 的核心输出） ============
  unsigned long long n_cmd;          // 发出的命令总数（行+列，含 NOP 之外的）
  unsigned long long n_activity;     // 总线有数据传输的活动周期数
  unsigned long long n_nop;          // 被迫发 NOP（无命令可发）的周期数
  unsigned long long n_act;          // ACT（行激活）命令数
  unsigned long long n_pre;          // PRE（预充电）命令数
  unsigned long long n_ref;          // REF（刷新）命令数
  unsigned long long n_rd;           // RD（读）命令数
  unsigned long long n_rd_L2_A;      // 其中来自 L2 access（非 fill/writeback）的读数
  unsigned long long n_wr;           // WR（写）命令数
  unsigned long long n_wr_WB;        // 其中 writeback 类型的写数
  unsigned long long n_req;          // 进入本 channel 的请求总数（= push 次数）
  unsigned long long max_mrqs_temp;  // 本统计窗口内 mrqq 长度的峰值（临时变量，打印后清零）

  // some statistics to see where BW is wasted?
  // ============ 带宽浪费归因：因某时序约束阻塞而空转的周期数 ============
  unsigned long long wasted_bw_row;    // 行命令（ACT/PRE）占用总线损失的带宽
  unsigned long long wasted_bw_col;    // 读写转向等限制损失的列数据带宽
  unsigned long long util_bw;          // 有效数据传输带宽（实际传数据的周期累计）
  unsigned long long idle_bw;          // 总线完全空闲损失的带宽
  unsigned long long RCDc_limit;       // 因 tRCD（行激活→列命令延迟）阻塞的空转周期
  unsigned long long CCDLc_limit;      // 因 tCCD_L（同 bank group 列命令间隔）阻塞的周期
  unsigned long long CCDLc_limit_alone;// 仅由 tCCD_L 单独造成阻塞的周期（归因去重）
  unsigned long long CCDc_limit;       // 因 tCCD 阻塞的空转周期
  unsigned long long WTRc_limit;       // 因 tWTR（写→读转向）阻塞的空转周期
  unsigned long long WTRc_limit_alone; // 仅由 tWTR 单独造成阻塞的周期
  unsigned long long RCDWRc_limit;     // 因 tRCDWR（读改写延迟）阻塞的空转周期
  unsigned long long RTWc_limit;       // 因 tRTW（读→写转向）阻塞的空转周期
  unsigned long long RTWc_limit_alone; // 仅由 tRTW 单独造成阻塞的周期
  unsigned long long rwq_limit;        // 因 rwq 满无法发列命令的次数

  // row locality, BLP and other statistics
  // ============ 行局部性、bank 级并行度（BLP）与其他统计 ============
  unsigned long long access_num;         // 发射命令总数（行命中率的分母）
  unsigned long long read_num;           // 读命令总数
  unsigned long long write_num;          // 写命令总数
  unsigned long long hits_num;           // 行命中次数（ACT 时目标行已打开）
  unsigned long long hits_read_num;      // 行命中的读命令数
  unsigned long long hits_write_num;     // 行命中的写命令数
  unsigned long long banks_1time;        // 单周期并行访问 bank 数的峰值（BLP 上限）
  unsigned long long banks_acess_total;  // 各周期并行访问 bank 数累加（[typo: ac cess]）
  unsigned long long banks_acess_total_after;  // 激活后口径的并行 bank 数累加（row locality 折损分析）
  unsigned long long banks_time_rw;      // 有读写活动（列命令）的周期数
  unsigned long long banks_access_rw_total;    // 读写周期内的并行 bank 数累加
  unsigned long long banks_time_ready;   // 存在 bank 可被服务（无阻塞）的周期数
  unsigned long long banks_access_ready_total; // 就绪周期内可服务 bank 数累加
  unsigned long long issued_two;         // 同一周期发射 2 条命令的周期数
  unsigned long long issued_total;       // 发射命令总次数
  unsigned long long issued_total_row;   // 行命令（ACT/PRE）发射总数
  unsigned long long issued_total_col;   // 列命令（RD/WR）发射总数
  unsigned long long last_cmd_cycle;     // 上次发射任意命令的全局周期（算 idle 间隔用）
  unsigned long long last_col_cycle;     // 上次发射列命令的周期
  unsigned long long last_read_col_cycle;// 上次发射读命令的周期
  unsigned long long last_write_col_cycle;// 上次发射写命令的周期
  unsigned long long last_row_cycle;     // 上次发射行命令的周期
  double write_to_read_ratio_blp_rw_average;  // 读写混合流下按写读比加权的 BLP 平均值
  unsigned long long bkgrp_parallsim_rw; // 读写操作时 bank group 并行度统计

  unsigned int bwutil;        // 本窗口总线利用率（n_activity/n_cmd，% 整数）
  unsigned int max_mrqs;      // mrqq 历史最大长度
  unsigned int ave_mrqs;      // mrqq 长度累加和（最后除以 n_cmd 得平均值）

  class frfcfs_scheduler *m_frfcfs_scheduler;  // FR-FCFS 调度器（仅当 scheduler_type ==
                                               // DRAM_FRFCFS 时创建；FIFO 模式为 NULL）。
                                               // 持有独立的读写挂起队列，是 dram_t 的唯一协作者

  // ============ visualizer 周期窗口统计（每窗口打印后清零） ============
  unsigned int n_cmd_partial;       // 本窗口命令数
  unsigned int n_activity_partial;  // 本窗口活动周期数
  unsigned int n_nop_partial;       // 本窗口 NOP 数
  unsigned int n_act_partial;       // 本窗口 ACT 数
  unsigned int n_pre_partial;       // 本窗口 PRE 数
  unsigned int n_req_partial;       // 本窗口请求数
  unsigned int ave_mrqs_partial;    // 本窗口 mrqq 长度累加
  unsigned int bwutil_partial;      // 本窗口带宽利用率

  memory_stats_manager_t *m_mem_stats;  // 访存延迟统计管理器（最终输出到统计文件）
  class Stats *mrqq_Dist;   // memory request queue inside DRAM
                            // mrqq 长度的分布直方图（StatCreate 创建，上限 = queue_limit()）

  friend class frfcfs_scheduler;  // 友元声明：允许 frfcfs_scheduler 直接访问 dram_t 的
                                  // 私有成员（bank 数组、bkgrp、rwq 等），
                                  // 否则调度器每次决策都要过一遍 public 接口
};

#endif /*DRAM_H*/

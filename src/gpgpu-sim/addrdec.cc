// Copyright (c) 2009-2011, Wilson W.L. Fung, Tor M. Aamodt, Ali Bakhoda,
// The University of British Columbia
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution. Neither the name of
// The University of British Columbia nor the names of its contributors may be
// used to endorse or promote products derived from this software without
// specific prior written permission.
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

#include "addrdec.h"
#include <math.h>
#include <string.h>
#include "../option_parser.h"
#include "gpu-sim.h"
#include "hashing.h"

static long int powli(long int x, long int y);
static unsigned int LOGB2_32(unsigned int v);
static unsigned next_powerOf2(unsigned n);

static new_addr_type addrdec_packbits(new_addr_type mask, new_addr_type val,
                                      unsigned char high, unsigned char low);
static void addrdec_getmasklimit(new_addr_type mask, unsigned char *high,
                                 unsigned char *low);

linear_to_raw_address_translation::linear_to_raw_address_translation() {
  addrdec_option = NULL;
  ADDR_CHIP_S = 10;
  memset(addrdec_mklow, 0, N_ADDRDEC);
  memset(addrdec_mkhigh, 64, N_ADDRDEC);
  addrdec_mask[0] = 0x0000000000001C00;
  addrdec_mask[1] = 0x0000000000000300;
  addrdec_mask[2] = 0x000000000FFF0000;
  addrdec_mask[3] = 0x000000000000E0FF;
  addrdec_mask[4] = 0x000000000000000F;
  ipoly_non_power2_balanced = 0;
  ipoly_channel_stable_l2slice = 0;
}

void linear_to_raw_address_translation::addrdec_setoption(option_parser_t opp) {
  option_parser_register(opp, "-gpgpu_mem_addr_mapping", OPT_CSTR,
                         &addrdec_option,
                         "mapping memory address to dram model {dramid@<start "
                         "bit>;<memory address map>}",
                         NULL);
  option_parser_register(
      opp, "-gpgpu_mem_addr_test", OPT_BOOL, &run_test,
      "run sweep test to check address mapping for aliased address", "0");
  option_parser_register(opp, "-gpgpu_mem_address_mask", OPT_INT32,
                         &gpgpu_mem_address_mask,
                         "0 = old addressing mask, 1 = new addressing mask, 2 "
                         "= new add. mask + flipped bank sel and chip sel bits",
                         "0");
  option_parser_register(
      opp, "-gpgpu_memory_partition_indexing", OPT_UINT32,
      &memory_partition_indexing,
      "0 = no indexing, 1 = bitwise xoring, 2 = IPoly, 3 = custom indexing",
      "0");
  option_parser_register(
      opp, "-gpgpu_ipoly_non_power2_balanced", OPT_UINT32,
      &ipoly_non_power2_balanced,
      "For non-power-of-two memory partitions: 0 = legacy modulo, 1 = map "
      "IPOLY buckets to channels before subpartitions, 2 = hash into a larger "
      "virtual partition space then range-reduce to the final partition count.",
      "0");
  option_parser_register(
      opp, "-gpgpu_ipoly_channel_stable_l2slice", OPT_UINT32,
      &ipoly_channel_stable_l2slice,
      "For non-power-of-two channel counts, keep the decoded DRAM channel "
      "stable and use IPOLY only to choose the L2 subpartition inside that "
      "channel. This preserves channel/row locality while balancing L2 slices.",
      "0");
}

// partition_address
/*
把一个内存地址里“表示要去哪个内存分区/通道”的那部分信息删掉，只留下“分区内部的位置信息”。

你可以把 GPU 的内存想象成一个大仓库，里面分成了很多个小房间（分区、通道、子分区）。
一个地址就像一张快递单，上面既有“送到哪个房间”的信息，也有“送到房间里哪个货架”的信息。

这个函数 partition_address 干的就是：

把“送到哪个房间”的那几个字擦掉，只留下“房间里哪个货架”的编号。

具体分两种情况：
1. 如果地址安排得很规整（!gap）
房间编号在地址里就是连续的几个二进制位，比如第 8~10 位表示去哪个房间。
那好办，直接拿个“橡皮擦”（掩码）把这几个位擦掉，再把剩下的位拼紧，就得到“房间内地址”了。

就像：
快递单上写着 A栋-3楼-12号，你直接把 A栋-3楼 划掉，只留 12号。

2. 如果地址安排得不规整（gap 为真）
房间数量不是 2 的幂，比如只有 3 个通道。
这时候地址里表示通道的位不是连续的，中间会有“空洞”，不能直接擦。
所以先做一个除法：把地址右移，除以通道数，再左移回去。
这相当于把因为通道编号造成的“空隙”挤掉，让地址变得紧凑。
然后再把子分区的那部分位擦掉。

就像：
快递单上不是简单的 A栋-3楼，而是 1号、3号、5号 这样跳着编号的，你得先重新排一下，把空号去掉，再擦掉楼层信息。

最后返回的是什么？
返回的并不是最终物理地址，而是一个中间地址。
它已经去掉了“去哪个分区”的信息，只剩下分区内部的偏移。
后面还会用这个地址去算具体是哪个分区、哪个 bank、哪一行等等。

一句话总结：
这个函数就是给地址“卸妆”，把表示内存分区/通道的位抹掉，留下分区内部的干净地址。
*/
new_addr_type linear_to_raw_address_translation::partition_address(
    new_addr_type addr) const {
  if (!gap) {
    return addrdec_packbits(~(addrdec_mask[CHIP] | sub_partition_id_mask), addr,
                            64, 0);
  } else {
    // see addrdec_tlx for explanation
    unsigned long long int partition_addr;
    partition_addr = ((addr >> ADDR_CHIP_S) / m_n_channel) << ADDR_CHIP_S;
    partition_addr |= addr & ((1 << ADDR_CHIP_S) - 1);
    // remove the part of address that constributes to the sub partition ID
    partition_addr =
        addrdec_packbits(~sub_partition_id_mask, partition_addr, 64, 0);
    return partition_addr;
  }
}

// Address Decoder Translation
/*
addrdec_tlx() 的作用是将 GPU 的线性内存地址解码成 DRAM 访问地址，
并确定访存请求对应的 Memory Partition 和 L2 Subpartition。

输入：GPU 线性内存地址 new_addr_type addr，例如 0x2000
输出：addrdec_t
*/
void linear_to_raw_address_translation::addrdec_tlx(new_addr_type addr,
                                                    addrdec_t *tlx) const {
  unsigned long long int addr_for_chip, rest_of_addr, rest_of_addr_high_bits;
  if (!gap) {
    tlx->chip = addrdec_packbits(addrdec_mask[CHIP], addr, addrdec_mkhigh[CHIP],
                                 addrdec_mklow[CHIP]);
    tlx->bk = addrdec_packbits(addrdec_mask[BK], addr, addrdec_mkhigh[BK],
                               addrdec_mklow[BK]);
    tlx->row = addrdec_packbits(addrdec_mask[ROW], addr, addrdec_mkhigh[ROW],
                                addrdec_mklow[ROW]);
    tlx->col = addrdec_packbits(addrdec_mask[COL], addr, addrdec_mkhigh[COL],
                                addrdec_mklow[COL]);
    tlx->burst = addrdec_packbits(addrdec_mask[BURST], addr,
                                  addrdec_mkhigh[BURST], addrdec_mklow[BURST]);
    rest_of_addr_high_bits =
        (addr >> (ADDR_CHIP_S + (log2channel + log2sub_partition)));

  } else {
    // Split the given address at ADDR_CHIP_S into (MSBs,LSBs)
    // - extract chip address using modulus of MSBs
    // - recreate the rest of the address by stitching the quotient of MSBs and
    // the LSBs
    addr_for_chip = (addr >> ADDR_CHIP_S) % m_n_channel;
    rest_of_addr = ((addr >> ADDR_CHIP_S) / m_n_channel) << ADDR_CHIP_S;
    rest_of_addr_high_bits = ((addr >> ADDR_CHIP_S) / m_n_channel);
    rest_of_addr |= addr & ((1 << ADDR_CHIP_S) - 1);

    tlx->chip = addr_for_chip;
    tlx->bk = addrdec_packbits(addrdec_mask[BK], rest_of_addr,
                               addrdec_mkhigh[BK], addrdec_mklow[BK]);
    tlx->row = addrdec_packbits(addrdec_mask[ROW], rest_of_addr,
                                addrdec_mkhigh[ROW], addrdec_mklow[ROW]);
    tlx->col = addrdec_packbits(addrdec_mask[COL], rest_of_addr,
                                addrdec_mkhigh[COL], addrdec_mklow[COL]);
    tlx->burst = addrdec_packbits(addrdec_mask[BURST], rest_of_addr,
                                  addrdec_mkhigh[BURST], addrdec_mklow[BURST]);
  }

  switch (memory_partition_indexing) {
    case CONSECUTIVE:
      // Do nothing
      break;
    case BITWISE_PERMUTATION: {
      assert(!gap);
      tlx->chip =
          bitwise_hash_function(rest_of_addr_high_bits, tlx->chip, m_n_channel);
      assert(tlx->chip < m_n_channel);
      break;
    }
    case IPOLY: {
      // assert(!gap);
      unsigned sub_partition_addr_mask = m_n_sub_partition_in_channel - 1;
      const unsigned decoded_channel = tlx->chip;
      const unsigned decoded_sub_partition_in_channel =
          tlx->bk & sub_partition_addr_mask;
      unsigned sub_partition =
          decoded_channel * m_n_sub_partition_in_channel +
          decoded_sub_partition_in_channel;
      // Use a lower shift for IPOLY hash input. The original
      // rest_of_addr_high_bits = addr >> (ADDR_CHIP_S + log2channel +
      // log2sub_partition) = addr >> 20, which strips bits that carry
      // stride variation for GEMM workloads (e.g., stride=4096 varies at
      // bit 12), causing severe bank imbalance (12-28 of 128 banks used).
      // Since IPOLY computes its own partition mapping via XOR, feeding it
      // more address bits (from cache-line boundary upward) gives uniform
      // distribution while preserving IPOLY's conflict-free guarantee for
      // power-of-2 strides.
      const unsigned ipoly_shift = 8;  // 256B interleave granularity
      unsigned long long int ipoly_high_bits =
          gap ? rest_of_addr_high_bits : (addr >> ipoly_shift);

      if (gap && ipoly_channel_stable_l2slice) {
        const unsigned slice_hash_seed =
            ((decoded_channel & 0xf) ^ decoded_sub_partition_in_channel);
        const unsigned slice_hash =
            ipoly_hash_function(addr >> ipoly_shift, slice_hash_seed, 16);
        const unsigned sub_partition_in_channel =
            slice_hash % m_n_sub_partition_in_channel;
        sub_partition = decoded_channel * m_n_sub_partition_in_channel +
                        sub_partition_in_channel;
      } else {
        const unsigned total_sub_partitions =
            m_n_channel * m_n_sub_partition_in_channel;
        const unsigned virtual_sub_partitions =
            (gap && ipoly_non_power2_balanced == 2)
                ? 1024
                : nextPowerOf2_m_n_channel * m_n_sub_partition_in_channel;
        sub_partition = ipoly_hash_function(
            ipoly_high_bits, sub_partition, virtual_sub_partitions);

        if (gap) {  // if it is not 2^n partitions, then take modular
          if (ipoly_non_power2_balanced == 1) {
            const unsigned channel = sub_partition % m_n_channel;
            const unsigned sub_partition_in_channel =
                (sub_partition / m_n_channel) % m_n_sub_partition_in_channel;
            sub_partition = channel * m_n_sub_partition_in_channel +
                            sub_partition_in_channel;
          } else if (ipoly_non_power2_balanced == 2) {
            sub_partition =
                (static_cast<unsigned long long>(sub_partition) *
                 total_sub_partitions) /
                virtual_sub_partitions;
          } else {
            sub_partition =
                sub_partition % (m_n_channel * m_n_sub_partition_in_channel);
          }
        }
      }

      tlx->chip = sub_partition / m_n_sub_partition_in_channel;
      tlx->sub_partition = sub_partition;
      assert(tlx->chip < m_n_channel);
      assert(tlx->sub_partition < m_n_channel * m_n_sub_partition_in_channel);
      return;
      break;
    }
    case RANDOM: {
      // This is an unrealistic hashing using software hashtable
      // we generate a random set for each memory address and save the value in
      new_addr_type chip_address = (addr >> (ADDR_CHIP_S - log2sub_partition));
      tr1_hash_map<new_addr_type, unsigned>::const_iterator got =
          address_random_interleaving.find(chip_address);
      if (got == address_random_interleaving.end()) {
        unsigned new_chip_id =
            rand() % (m_n_channel * m_n_sub_partition_in_channel);
        address_random_interleaving[chip_address] = new_chip_id;
        tlx->chip = new_chip_id / m_n_sub_partition_in_channel;
        tlx->sub_partition = new_chip_id;
      } else {
        unsigned new_chip_id = got->second;
        tlx->chip = new_chip_id / m_n_sub_partition_in_channel;
        tlx->sub_partition = new_chip_id;
      }

      assert(tlx->chip < m_n_channel);
      assert(tlx->sub_partition < m_n_channel * m_n_sub_partition_in_channel);
      return;
      break;
    }
    case CUSTOM:
      /* No custom set function implemented */
      // Do you custom index here
      break;
    default:
      assert("\nUndefined set index function.\n" && 0);
      break;
  }

  // combine the chip address and the lower bits of DRAM bank address to form
  // the subpartition ID
  unsigned sub_partition_addr_mask = m_n_sub_partition_in_channel - 1;
  tlx->sub_partition = tlx->chip * m_n_sub_partition_in_channel +
                       (tlx->bk & sub_partition_addr_mask);
}

void linear_to_raw_address_translation::addrdec_parseoption(
    const char *option) {
  unsigned int dramid_start = 0;
  int dramid_parsed = sscanf(option, "dramid@%d", &dramid_start);
  if (dramid_parsed == 1) {
    ADDR_CHIP_S = dramid_start;
  } else {
    ADDR_CHIP_S = -1;
  }

  const char *cmapping = strchr(option, ';');
  if (cmapping == NULL) {
    cmapping = option;
  } else {
    cmapping += 1;
  }

  addrdec_mask[CHIP] = 0x0;
  addrdec_mask[BK] = 0x0;
  addrdec_mask[ROW] = 0x0;
  addrdec_mask[COL] = 0x0;
  addrdec_mask[BURST] = 0x0;

  int ofs = 63;
  while ((*cmapping) != '\0') {
    switch (*cmapping) {
      case 'D':
      case 'd':
        assert(dramid_parsed != 1);
        addrdec_mask[CHIP] |= (1ULL << ofs);
        ofs--;
        break;
      case 'B':
      case 'b':
        addrdec_mask[BK] |= (1ULL << ofs);
        ofs--;
        break;
      case 'R':
      case 'r':
        addrdec_mask[ROW] |= (1ULL << ofs);
        ofs--;
        break;
      case 'C':
      case 'c':
        addrdec_mask[COL] |= (1ULL << ofs);
        ofs--;
        break;
      case 'S':
      case 's':
        addrdec_mask[BURST] |= (1ULL << ofs);
        addrdec_mask[COL] |= (1ULL << ofs);
        ofs--;
        break;
      // ignore bit
      case '0':
        ofs--;
        break;
      // ignore character
      case '|':
      case ' ':
      case '.':
        break;
      default:
        fprintf(
            stderr,
            "ERROR: Invalid address mapping character '%c' in option '%s'\n",
            *cmapping, option);
    }
    cmapping += 1;
  }

  if (ofs != -1) {
    fprintf(stderr,
            "ERROR: Invalid address mapping length (%d) in option '%s'\n",
            63 - ofs, option);
    assert(ofs == -1);
  }
}

void linear_to_raw_address_translation::init(
    unsigned int n_channel, unsigned int n_sub_partition_in_channel) {
  unsigned i;
  unsigned long long int mask;
  // memory channel / partition 数量
  // 对 n_channel 向下取 log2
  unsigned int nchipbits = ::LOGB2_32(n_channel);
  log2channel = nchipbits;
  log2sub_partition = ::LOGB2_32(n_sub_partition_in_channel);
  m_n_channel = n_channel;
  m_n_sub_partition_in_channel = n_sub_partition_in_channel;
  nextPowerOf2_m_n_channel = ::next_powerOf2(n_channel);
  m_n_sub_partition_total = n_channel * n_sub_partition_in_channel;

  // powli(2, nchipbits)：计算 2 的 nchipbits 次幂
  // :: 调用全局命名空间下的函数，避免和类内函数混淆
  gap = (n_channel - ::powli(2, nchipbits));
  // gap == 0：n_channel 是 2 的幂
  // gap != 0：n_channel 不是 2 的幂，gap 保存它与较小 2 次幂的差值
  if (gap) {
    nchipbits++;
    /*
    意思是：如果 channel 数不是 2 的幂，就把表示 channel 所需 bit 数加 1、
    ，也就是使用向上取整的 bit 数。例如 6 个 channel 需要 3 bit 才能表示编号 0~5
    */
  }
  switch (gpgpu_mem_address_mask) {
    /*
    ADDR_CHIP_S：channel/chip ID 从哪个 bit 位开始插入或切分
    addrdec_mask[CHIP]：哪些地址 bit 表示 chip/channel
    addrdec_mask[BK]：哪些地址 bit 表示 bank
    addrdec_mask[ROW]：哪些地址 bit 表示 row
    addrdec_mask[COL]：哪些地址 bit 表示 column
    addrdec_mask[BURST]：哪些地址 bit 表示 burst
    */
    case 0:
      // old, added 2row bits, use #define ADDR_CHIP_S 10
      ADDR_CHIP_S = 10;
      addrdec_mask[CHIP] = 0x0000000000000000;
      /*
      这不表示没有 chip。
      而是 case 0 不固定 chip 占用哪些 bit，chip mask 是后面根据 channel 数量动态生成的。
      大致流程是：
      先设置 ADDR_CHIP_S = 10
      如果 channel 数是 2 的幂，就把 chip bits 插入到 bit10 开始的位置
      原来的 BK/ROW/COL 中高于 bit10 的部分会相应左移，给 chip 让位
      */
      addrdec_mask[BK] = 0x0000000000000300; //bits 8~9,一共 2 个 bit，对应4 个 bank
      addrdec_mask[ROW] = 0x0000000007FFE000; 
      //bits 13~30,所以 row 使用 18 个 bit
      // 2^18 = 262144 个 row
      addrdec_mask[COL] = 0x0000000000001CFF;
      // 0x00FF -> bits 0~7
      // 0x1C00 -> bits 10~12
      /*
      所以 column 分成两段：
      col 低位：bits 0~7
      col 高位：bits 10~12
      总共 11 个 column bit
      */
      break;
    case 1:
      ADDR_CHIP_S = 13;
      addrdec_mask[CHIP] = 0x0000000000000000;
      addrdec_mask[BK] = 0x0000000000001800;
      addrdec_mask[ROW] = 0x0000000007FFE000;
      addrdec_mask[COL] = 0x00000000000007FF;
      break;
    case 2:
      ADDR_CHIP_S = 11;
      addrdec_mask[CHIP] = 0x0000000000000000;
      addrdec_mask[BK] = 0x0000000000001800;
      addrdec_mask[ROW] = 0x0000000007FFE000;
      addrdec_mask[COL] = 0x00000000000007FF;
      break;
    case 3:
      ADDR_CHIP_S = 11;
      addrdec_mask[CHIP] = 0x0000000000000000;
      addrdec_mask[BK] = 0x0000000000001800;
      addrdec_mask[ROW] = 0x000000000FFFE000;
      addrdec_mask[COL] = 0x00000000000007FF;
      break;

    case 14:
      ADDR_CHIP_S = 14;
      addrdec_mask[CHIP] = 0x0000000000000000;
      addrdec_mask[BK] = 0x0000000000001800;
      addrdec_mask[ROW] = 0x0000000007FFE000;
      addrdec_mask[COL] = 0x00000000000007FF;
      break;
    case 15:
      ADDR_CHIP_S = 15;
      addrdec_mask[CHIP] = 0x0000000000000000;
      addrdec_mask[BK] = 0x0000000000001800;
      addrdec_mask[ROW] = 0x0000000007FFE000;
      addrdec_mask[COL] = 0x00000000000007FF;
      break;
    case 16:
      ADDR_CHIP_S = 16;
      addrdec_mask[CHIP] = 0x0000000000000000;
      addrdec_mask[BK] = 0x0000000000001800;
      addrdec_mask[ROW] = 0x0000000007FFE000;
      addrdec_mask[COL] = 0x00000000000007FF;
      break;
    case 6:
      ADDR_CHIP_S = 6;
      addrdec_mask[CHIP] = 0x0000000000000000;
      addrdec_mask[BK] = 0x0000000000001800;
      addrdec_mask[ROW] = 0x0000000007FFE000;
      addrdec_mask[COL] = 0x00000000000007FF;
      break;
    case 5:
      ADDR_CHIP_S = 5;
      addrdec_mask[CHIP] = 0x0000000000000000;
      addrdec_mask[BK] = 0x0000000000001800;
      addrdec_mask[ROW] = 0x0000000007FFE000;
      addrdec_mask[COL] = 0x00000000000007FF;
      break;
    case 100:
      ADDR_CHIP_S = 1;
      addrdec_mask[CHIP] = 0x0000000000000000;
      addrdec_mask[BK] = 0x0000000000000003;
      addrdec_mask[ROW] = 0x0000000007FFE000;
      addrdec_mask[COL] = 0x0000000000001FFC;
      break;
    case 103:
      ADDR_CHIP_S = 3;
      addrdec_mask[CHIP] = 0x0000000000000000;
      addrdec_mask[BK] = 0x0000000000000003;
      addrdec_mask[ROW] = 0x0000000007FFE000;
      addrdec_mask[COL] = 0x0000000000001FFC;
      break;
    case 106:
      ADDR_CHIP_S = 6;
      addrdec_mask[CHIP] = 0x0000000000000000;
      addrdec_mask[BK] = 0x0000000000001800;
      addrdec_mask[ROW] = 0x0000000007FFE000;
      addrdec_mask[COL] = 0x00000000000007FF;
      break;
    case 160:
      // old, added 2row bits, use #define ADDR_CHIP_S 10
      ADDR_CHIP_S = 6;
      addrdec_mask[CHIP] = 0x0000000000000000;
      addrdec_mask[BK] = 0x0000000000000300;
      addrdec_mask[ROW] = 0x0000000007FFE000;
      addrdec_mask[COL] = 0x0000000000001CFF;
      /*
      不同 case 的区别主要是：
      chip 从哪个 bit 开始
      bank 用哪些 bit
      row 用哪些 bit
      column 用哪些 bit
      是否适合某种 DRAM 组织方式
      */

    default:
      break;
  }

  if (addrdec_option != NULL) addrdec_parseoption(addrdec_option);

  if (ADDR_CHIP_S != -1) {
    if (!gap) {
      // number of chip is power of two:
      // - insert CHIP mask starting at the bit position ADDR_CHIP_S
      mask = ((unsigned long long int)1 << ADDR_CHIP_S) - 1;
      addrdec_mask[BK] =
          ((addrdec_mask[BK] & ~mask) << nchipbits) | (addrdec_mask[BK] & mask);
      addrdec_mask[ROW] = ((addrdec_mask[ROW] & ~mask) << nchipbits) |
                          (addrdec_mask[ROW] & mask);
      addrdec_mask[COL] = ((addrdec_mask[COL] & ~mask) << nchipbits) |
                          (addrdec_mask[COL] & mask);

      for (i = ADDR_CHIP_S; i < (ADDR_CHIP_S + nchipbits); i++) {
        mask = (unsigned long long int)1 << i;
        addrdec_mask[CHIP] |= mask;
      }
    }  // otherwise, no need to change the masks
  } else {
    // make sure n_channel is power of two when explicit dram id mask is used
    assert((n_channel & (n_channel - 1)) == 0);
  }
  // make sure m_n_sub_partition_in_channel is power of two
  assert((m_n_sub_partition_in_channel & (m_n_sub_partition_in_channel - 1)) ==
         0);

  addrdec_getmasklimit(addrdec_mask[CHIP], &addrdec_mkhigh[CHIP],
                       &addrdec_mklow[CHIP]);
  addrdec_getmasklimit(addrdec_mask[BK], &addrdec_mkhigh[BK],
                       &addrdec_mklow[BK]);
  addrdec_getmasklimit(addrdec_mask[ROW], &addrdec_mkhigh[ROW],
                       &addrdec_mklow[ROW]);
  addrdec_getmasklimit(addrdec_mask[COL], &addrdec_mkhigh[COL],
                       &addrdec_mklow[COL]);
  addrdec_getmasklimit(addrdec_mask[BURST], &addrdec_mkhigh[BURST],
                       &addrdec_mklow[BURST]);

  printf("addr_dec_mask[CHIP]  = %016llx \thigh:%d low:%d\n",
         addrdec_mask[CHIP], addrdec_mkhigh[CHIP], addrdec_mklow[CHIP]);
  printf("addr_dec_mask[BK]    = %016llx \thigh:%d low:%d\n", addrdec_mask[BK],
         addrdec_mkhigh[BK], addrdec_mklow[BK]);
  printf("addr_dec_mask[ROW]   = %016llx \thigh:%d low:%d\n", addrdec_mask[ROW],
         addrdec_mkhigh[ROW], addrdec_mklow[ROW]);
  printf("addr_dec_mask[COL]   = %016llx \thigh:%d low:%d\n", addrdec_mask[COL],
         addrdec_mkhigh[COL], addrdec_mklow[COL]);
  printf("addr_dec_mask[BURST] = %016llx \thigh:%d low:%d\n",
         addrdec_mask[BURST], addrdec_mkhigh[BURST], addrdec_mklow[BURST]);

  // create the sub partition ID mask (for removing the sub partition ID from
  // the partition address)
  sub_partition_id_mask = 0;
  if (m_n_sub_partition_in_channel > 1) {
    unsigned n_sub_partition_log2 = LOGB2_32(m_n_sub_partition_in_channel);
    unsigned pos = 0;
    for (unsigned i = addrdec_mklow[BK]; i < addrdec_mkhigh[BK]; i++) {
      if ((addrdec_mask[BK] & ((unsigned long long int)1 << i)) != 0) {
        sub_partition_id_mask |= ((unsigned long long int)1 << i);
        pos++;
        if (pos >= n_sub_partition_log2) break;
      }
    }
  }
  printf("sub_partition_id_mask = %016llx\n", sub_partition_id_mask);

  if (run_test) {
    sweep_test();
  }

  if (memory_partition_indexing == RANDOM) srand(1);
}

#include "../tr1_hash_map.h"

bool operator==(const addrdec_t &x, const addrdec_t &y) {
  return (memcmp(&x, &y, sizeof(addrdec_t)) == 0);
}

bool operator<(const addrdec_t &x, const addrdec_t &y) {
  if (x.chip >= y.chip)
    return false;
  else if (x.bk >= y.bk)
    return false;
  else if (x.row >= y.row)
    return false;
  else if (x.col >= y.col)
    return false;
  else if (x.burst >= y.burst)
    return false;
  else
    return true;
}

class hash_addrdec_t {
 public:
  size_t operator()(const addrdec_t &x) const {
    return (x.chip ^ x.bk ^ x.row ^ x.col ^ x.burst);
  }
};

// a simple sweep test to ensure that two linear addresses are not mapped to the
// same raw address
// 在 GPU 模拟器中，地址映射逻辑必须保证单射性：不同的线性地址应该映射到
// 不同的 raw address。如果发生冲突，模拟出的内存行为就会出错
void linear_to_raw_address_translation::sweep_test() const {
  new_addr_type sweep_range = 16 * 1024 * 1024;

#if tr1_hash_map_ismap == 1
  typedef tr1_hash_map<addrdec_t, new_addr_type> history_map_t;
#else
  typedef tr1_hash_map<addrdec_t, new_addr_type, hash_addrdec_t> history_map_t;
#endif
  history_map_t history_map;

  for (new_addr_type raw_addr = 4; raw_addr < sweep_range; raw_addr += 4) {
    addrdec_t tlx;
    // 对地址做一次译码，把译码结果填到一个结构体里
    addrdec_tlx(raw_addr, &tlx);

    history_map_t::iterator h = history_map.find(tlx);

    if (h != history_map.end()) {
      printf(
          "[AddrDec] ** Error: address decoding mapping aliases two addresses "
          "to same partition with same intra-partition address: %llx %llx\n",
          h->second, raw_addr);
      abort();
    } else {
      // 通道号不能越界
      assert(tlx.chip < m_n_channel);
      // ensure that partition_address() returns the concatenated address
      if ((ADDR_CHIP_S != -1 and raw_addr >= (1ULL << ADDR_CHIP_S)) or
          (ADDR_CHIP_S == -1 and raw_addr >= (1ULL << addrdec_mklow[CHIP]))) {
        assert(raw_addr != partition_address(raw_addr));
      }
      history_map[tlx] = raw_addr;
    }

    // 进度条，每 65536 个打印一次
    if ((raw_addr & 0xffff) == 0) printf("%llu scaned\n", raw_addr);
  }
}

void addrdec_t::print(FILE *fp) const {
  fprintf(fp, "\tchip:%x ", chip);
  fprintf(fp, "\trow:%x ", row);
  fprintf(fp, "\tcol:%x ", col);
  fprintf(fp, "\tbk:%x ", bk);
  fprintf(fp, "\tburst:%x ", burst);
  fprintf(fp, "\tsub_partition:%x ", sub_partition);
}

static long int powli(long int x, long int y)  // compute x to the y
{
  long int r = 1;
  int i;
  for (i = 0; i < y; ++i) {
    r *= x;
  }
  return r;
}

static unsigned int LOGB2_32(unsigned int v) {
  unsigned int shift;
  unsigned int r;

  r = 0;

  shift = ((v & 0xFFFF0000) != 0) << 4;
  v >>= shift;
  r |= shift;
  shift = ((v & 0xFF00) != 0) << 3;
  v >>= shift;
  r |= shift;
  shift = ((v & 0xF0) != 0) << 2;
  v >>= shift;
  r |= shift;
  shift = ((v & 0xC) != 0) << 1;
  v >>= shift;
  r |= shift;
  shift = ((v & 0x2) != 0) << 0;
  v >>= shift;
  r |= shift;

  return r;
}

// compute power of two greater than or equal to n
// https://www.techiedelight.com/round-next-highest-power-2/
unsigned next_powerOf2(unsigned n) {
  // decrement n (to handle the case when n itself
  // is a power of 2)
  n = n - 1;

  // do till only one bit is left
  while (n & (n - 1)) n = n & (n - 1);  // unset rightmost bit

  // n is now a power of two (less than n)

  // return next power of 2
  return n << 1;
}

/*
addrdec_packbits() 的作用是：按照 mask 指定的 bit 位，
从 val 中抽取这些 bit，并把它们压缩成从 bit0 开始的连续整数

val    11001011
mask   10110110
result 10001
*/
static new_addr_type addrdec_packbits(new_addr_type mask, new_addr_type val,
                                      unsigned char high, unsigned char low) {
  unsigned pos = 0;
  new_addr_type result = 0;
  for (unsigned i = low; i < high; i++) {
    if ((mask & ((unsigned long long int)1 << i)) != 0) {
      result |= ((val & ((unsigned long long int)1 << i)) >> i) << pos;
      pos++;
    }
  }
  return result;
}

static void addrdec_getmasklimit(new_addr_type mask, unsigned char *high,
                                 unsigned char *low) {
  *high = 64;
  *low = 0;
  int i;
  int low_found = 0;

  for (i = 0; i < 64; i++) {
    if ((mask & ((unsigned long long int)1 << i)) != 0) {
      if (low_found) {
        *high = i + 1;
      } else {
        *high = i + 1;
        *low = i;
        low_found = 1;
      }
    }
  }
}

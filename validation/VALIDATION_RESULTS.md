# Memory pool validation

All 7 baseline/refactor comparisons and all 4 component modes passed.

Threads: 8; PTX_SIM_MODE_FUNC=0 (detailed simulation).
Tutorial logs reused: True.

| Case | Result | Per-kernel cycles | Cumulative cycles | Final pool accepted | Completed writebacks | Pending |
|---|---|---|---|---|---|---|
| vector_add | PASS | 30133 | 30133 | 500000 | 0 | 0 |
| gemm | PASS | 78989 | 78989 | 419840 | 0 | 0 |
| sm90_stock | PASS | 15247, 11595 | 15247, 26842 | 4096 | 0 | 0 |
| sm120_stock | PASS | 15156, 11454 | 15156, 26610 | 4096 | 0 | 0 |
| sm90_pressure | PASS | 21595, 28312 | 21595, 49907 | 15086 | 1975 | 1 |
| sm90_write_queue | PASS | 21543, 30289 | 21543, 51832 | 14927 | 1928 | 24 |
| sm90_simple | PASS | 15044, 11595 | 15044, 26639 | 4096 | 0 | 0 |
| two_clients_one_channel | PASS |  |  | - | - | - |

Compared exactly: kernel cycles/instructions, L2 totals, and each DRAM channel's
n_cmd/n_nop/n_act/n_pre/n_req/n_rd/n_write/n_wr_bk counters.
Every paired row has an empty differences object in results.json.

The small-L2 pressure and separate-write-queue tests finish with 1 and 24
pool requests outstanding respectively. Legacy kernel termination checks
L2 request trackers, not a complete drain of internal L2 writebacks.
This refactor preserves that termination rule and the baseline cycle/counter values.
Pool accounting remains accepted = responses + writebacks + pending.
These regression cases do not prove that every internal queue is empty at exit.

The component test explicitly drains all 8 requests, ends with pending=0,
checks the correct client for every response/completion, and forces blocked
responses to recover. It passes in FR-FCFS, separate write queue, FIFO, and simple modes.

POOL_COMPONENT_PASSED simple=0 separate=0 scheduler=1 ticks=812 blocked=771
POOL_COMPONENT_PASSED simple=0 separate=1 scheduler=1 ticks=815 blocked=771
POOL_COMPONENT_PASSED simple=0 separate=0 scheduler=0 ticks=812 blocked=771
POOL_COMPONENT_PASSED simple=1 separate=0 scheduler=1 ticks=806 blocked=796

Both full simulator builds passed; build logs are retained.
The isolated baseline has one syntax-only repair: ': : m_id' -> ': m_id'.
The original source directory is not patched for that repair.
One identical repeated baseline GEMM snapshot was merged only after all selected
kernel and per-channel metrics matched; conflicting repeats fail validation.
The simple DRAM case uses the pool latency path and has zero detailed dram_t requests.
See MEMORY_POOL_REFACTOR.md at the project root for design, scope and reproduction.

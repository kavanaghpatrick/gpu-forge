---
name: gpu-distributed
description: >
  This skill should be used when the user asks about distributed GPU computing across multiple Apple Silicon Macs, RDMA over Thunderbolt 5, multi-Mac clusters, GPU cluster networking, MLX distributed operations, collective communication patterns, or scaling AI workloads across machines. Covers RDMA setup and configuration, JACCL backend, Thunderbolt 5 topology, ring and mesh communication patterns, data parallelism, tensor parallelism, pipeline parallelism, expert parallelism, model parallelism, all-reduce operations, ring allreduce, distributed training, distributed inference, Exo framework, dnet, cluster power and cost analysis, macOS Tahoe RDMA enablement, and ibverbs API usage on Apple Silicon. Use when questions mention "distributed GPU", "RDMA", "Thunderbolt 5", "multi-Mac", "GPU cluster", "MLX distributed", "all-reduce", "ring allreduce", "data parallel", "model parallel", "macOS Tahoe", "JACCL", "Exo", "pipeline parallelism", "tensor parallelism", "expert parallelism", "collective communication", or multi-node GPU workloads.
---

# GPU Distributed Knowledge

## Domain Overview

Distributed GPU computing on Apple Silicon enables scaling AI inference and training across multiple Macs connected via high-speed interconnects. The breakthrough came with macOS Tahoe 26.2 (December 2025), which introduced RDMA (Remote Direct Memory Access) over Thunderbolt 5, dramatically reducing inter-node communication latency from ~300 microseconds (TCP) to 5-9 microseconds while providing 50-60 Gbps practical throughput (80 Gbps theoretical).

The core software stack centers on MLX, Apple's machine learning framework, which provides first-class distributed primitives: `all_sum`, `all_gather`, `send`, `recv`, `recv_like`, and distributed group management. The RDMA transport layer is implemented by JACCL (Jack and Angelos' Collective Communication Library), which uses standard InfiniBand verbs (ibverbs) APIs — the same programming model used in HPC data centers with NVIDIA InfiniBand hardware. JACCL requires a fully-connected mesh topology where every Mac has a direct Thunderbolt 5 cable to every other Mac, supporting up to 4 nodes in current configurations.

Multiple parallelism strategies are available for distributing workloads. Pipeline parallelism (layer sharding) assigns different model layers to different nodes, minimizing communication but creating pipeline bubbles. Tensor parallelism shards individual layer weights across nodes, requiring frequent all-reduce operations but enabling larger-than-single-node layers. Expert parallelism routes mixture-of-experts layers to specialized nodes, requiring the least communication bandwidth. Data parallelism replicates the model on each node and averages gradients, suitable for training. Apple internally uses combinations of data parallelism, tensor parallelism, sequence parallelism, and FSDP for foundation model training.

The ecosystem includes frameworks like Exo (peer-to-peer distributed inference with day-0 RDMA support, ring memory-weighted partitioning) and dnet (pipelined-ring execution for distributed LLM inference). Real-world results demonstrate practical viability: a 4x Mac Studio M3 Ultra cluster with 1.5TB unified memory ran Kimi K2 Thinking (1T parameters) and Qwen3 235B at 32 tokens/sec, with the entire cluster drawing under 500W — a fraction of comparable NVIDIA GPU server power consumption.

However, significant constraints remain. The maximum cluster size is 4 Macs in an RDMA mesh. Thunderbolt 5 bandwidth (80 Gbps) is orders of magnitude below NVLink (1800 GB/s), making communication-heavy parallelism strategies less effective. Heterogeneous clusters (mixing MLX and Tinygrad backends) exhibit up to 90% performance degradation. No established distributed MapReduce-over-Metal framework exists for non-ML general-purpose compute workloads on Apple Silicon.

## Key Knowledge Areas

The gpu-distributed skill contains 51 findings across these domains:

- **RDMA over Thunderbolt 5**: macOS Tahoe 26.2 enablement, ibverbs API, recovery mode setup, hardware requirements (M4 Pro/Max, M3 Ultra), security model with IOMMU protection [Findings #303, #309, #412, #413, #414]
- **RDMA Bandwidth & Latency**: 50-60 Gbps practical, 5-9 microsecond latency, comparison with 10GbE/NVLink [Findings #306, #417, #418, #58]
- **JACCL Backend**: MLX RDMA backend via PR #2808, fully-connected mesh requirement, configuration via mlx.distributed_config CLI, named after Jack and Angelos [Findings #329, #415, #416, #59]
- **Network Topology**: Full mesh for JACCL, ring for MLX TCP backend, cable requirements (40Gbps+, under 2 meters), separate Ethernet for coordination [Findings #316, #322, #423, #424]
- **MLX Distributed API**: all_sum, all_gather, send, recv, recv_like operations, distributed group management, mlx.launch orchestration [Findings #324, #426, #333]
- **MLX Distributed Training**: Data-parallel training with nn.average_gradients(), gradient averaging across nodes [Finding #337]
- **Parallelism Strategies**: Pipeline, tensor, expert, and data parallelism; communication requirements comparison; Apple's internal training approach (data + tensor + sequence + FSDP) [Findings #346, #366, #431, #432, #433, #434]
- **Exo Framework**: Peer-to-peer architecture, ring memory-weighted partitioning, pipeline and tensor parallelism support, benchmark results (single M4 Pro 49.3 TPS, 3-device 108.8 TPS batched) [Findings #341, #346, #350, #427, #428, #429, #430]
- **dnet Framework**: FirstBatch's distributed LLM inference with pipelined-ring execution [Findings #369, #435]
- **Reference Clusters**: Jeff Geerling 4x Mac Studio M3 Ultra (1.5TB), Kimi K2 Thinking (1T params), Qwen3 235B at 32 TPS [Findings #354, #419, #425]
- **Cost & Power**: 4x Mac Studio cluster (~$K) vs comparable NVIDIA setup, under 500W total draw, DeepSeek V3.1 at 24-26 TPS [Findings #359, #420]
- **Zero-Copy Transfers**: RDMA provides direct device-to-device data movement without CPU involvement [Findings #377, #421]
- **Distributed Memory Model**: RDMA creates distributed shared memory abstraction across Mac cluster [Finding #67]
- **Non-ML Distributed Compute**: No established MapReduce-over-Metal framework exists [Finding #436]
- **GPU-Centric Communication**: Taxonomy and optimization of reducing CPU involvement in inter-GPU transfers [Finding #549]
- **Production Considerations**: Max 4 Macs, NVLink bandwidth gap, thermal/power monitoring tools (powermetrics, macmon, fluidtop) [Findings #437, #380]

## How to Query

All gpu-distributed knowledge is stored in the GPU Computer knowledge database. Query using the `kb` CLI:

```bash
# List all gpu-distributed findings
${CLAUDE_PLUGIN_ROOT}/scripts/kb skill gpu-distributed

# Search for specific topics
${CLAUDE_PLUGIN_ROOT}/scripts/kb search "RDMA Thunderbolt"
${CLAUDE_PLUGIN_ROOT}/scripts/kb search "JACCL backend"
${CLAUDE_PLUGIN_ROOT}/scripts/kb search "MLX distributed"
${CLAUDE_PLUGIN_ROOT}/scripts/kb search "Exo framework"
${CLAUDE_PLUGIN_ROOT}/scripts/kb search "tensor parallelism"

# Get detailed finding with sources
${CLAUDE_PLUGIN_ROOT}/scripts/kb detail <finding-id>

# Check statistics
${CLAUDE_PLUGIN_ROOT}/scripts/kb stats
```

Replace `${CLAUDE_PLUGIN_ROOT}` with the actual path to the gpu-forge plugin directory.

## Common Patterns & Quick Answers

### Q: How do I set up RDMA over Thunderbolt 5?
**A**: Requires macOS Tahoe 26.2+, Thunderbolt 5 hardware (M4 Pro, M4 Max, or M3 Ultra), and physical access to each Mac. Enable RDMA from macOS Recovery Mode by running `rdma_ctl enable` from Terminal. Cables must be rated 40Gbps+ and kept under 2 meters. Use separate Ethernet connections for coordination traffic. Configure with `mlx.distributed_config` CLI which auto-discovers Thunderbolt connections. [Findings #303, #309, #412, #413, #416, #424]

### Q: What distributed operations does MLX provide?
**A**: MLX exposes six core distributed operations: `all_sum` (collective reduction), `all_gather` (gather tensors from all nodes), `send` (point-to-point send), `recv` (point-to-point receive), `recv_like` (receive matching shape of existing tensor), plus distributed group management. Orchestrate multi-node programs with `mlx.launch`. [Findings #324, #426, #333]

### Q: What is JACCL and how does it relate to MLX?
**A**: JACCL (Jack and Angelos' Collective Communication Library) is MLX's RDMA backend, integrated via PR #2808. It uses standard InfiniBand verbs (ibverbs) APIs for high-performance communication. JACCL requires a fully-connected mesh topology where every Mac has a direct Thunderbolt 5 cable to every other Mac (N*(N-1)/2 cables for N nodes). Apple also ships MLX5 drivers for Mellanox ConnectX adapters as an alternative transport. [Findings #329, #59, #316, #414, #423]

### Q: What are the RDMA bandwidth and latency numbers?
**A**: TB5 RDMA achieves 50-60 Gbps practical throughput (80 Gbps theoretical) with 5-9 microsecond latency — down from ~300 microseconds with TCP networking. For comparison: 10GbE = 10 Gbps, TB5 RDMA = 80 Gbps, NVLink = 1800 GB/s (14,400 Gbps). The NVLink gap means communication-heavy strategies (tensor parallelism) are less efficient on Apple Silicon clusters vs NVIDIA DGX systems. [Findings #306, #417, #418, #437]

### Q: Which parallelism strategy should I use for Apple Silicon clusters?
**A**: Expert parallelism requires the least communication bandwidth, making it ideal for TB5's limited interconnect. Pipeline parallelism (layer sharding) is the next best choice — it minimizes inter-node transfers but creates pipeline bubbles. Tensor parallelism requires frequent all-reduce operations and is most affected by TB5's bandwidth gap vs NVLink. Data parallelism works well for training (replicate model, average gradients). Apple internally combines data, tensor, sequence parallelism, and FSDP for foundation model training. [Findings #346, #366, #431, #434]

### Q: How does Exo work for distributed inference?
**A**: Exo is a peer-to-peer distributed AI inference framework (no master-worker hierarchy) with day-0 RDMA support. Default partitioning uses RingMemoryWeightedPartitioning, assigning layers proportional to each node's available memory. Supports both pipeline and tensor parallelism. Benchmarks: single M4 Pro achieves 49.3 TPS, 3-device cluster gets 39.7 TPS single-request but 108.8 TPS with batched requests. Caveat: heterogeneous MLX + Tinygrad clusters show 90% performance degradation (48 to 5 TPS). [Findings #341, #346, #350, #427, #428, #429, #430]

### Q: What does a real-world Mac cluster look like?
**A**: Jeff Geerling's reference cluster: 4x Mac Studio M3 Ultra (two 512GB + two 256GB = 1.5TB unified memory), fully meshed with Thunderbolt 5 cables. Results: Kimi K2 Thinking (1T parameters) runs successfully, Qwen3 235B at 32 tokens/sec, DeepSeek V3.1 at 24-26 TPS. Total power draw under 500W for trillion-parameter inference. Cost: ~$K for 4x Mac Studios vs comparable NVIDIA server setups. [Findings #354, #419, #420, #425, #359]

### Q: How does distributed training work with MLX?
**A**: MLX supports distributed data-parallel training with `nn.average_gradients()` for gradient averaging across nodes. Pipeline parallel training demonstrated by Matt Beton (mlx-train): 2x M3 Ultra showed near-linear scaling. ICML 2025 introduced KPOP optimizer using Adam in Kronecker-factored eigenspace for communication-efficient distributed training. [Findings #337, #432, #433]

### Q: What are the current limitations of Apple Silicon clusters?
**A**: Key constraints: (1) Maximum 4 Macs in RDMA mesh with current JACCL, (2) TB5 bandwidth (80 Gbps) is 180x less than NVLink (1800 GB/s), making tensor parallelism less effective, (3) Heterogeneous backend clusters (MLX + Tinygrad) suffer 90% degradation, (4) No established distributed MapReduce-over-Metal framework for non-ML compute, (5) RDMA requires physical access (recovery mode) for initial setup, (6) Cables must be under 2 meters. [Findings #437, #430, #436, #413, #424]

### Q: Is there zero-copy support for RDMA transfers?
**A**: Yes. RDMA over TB5 provides true zero-copy data transfers — data moves directly from one Mac's unified memory to another's without CPU involvement or intermediate buffering. This aligns with the GPU-centric communication paradigm of reducing CPU involvement in inter-GPU data transfer. Combined with Apple Silicon's unified memory architecture, this means GPU-accessible memory on one node can be transferred directly to GPU-accessible memory on another. [Findings #377, #421, #549, #67]

### Q: How do I monitor cluster health and performance?
**A**: Mac clusters can be monitored via: `powermetrics` (requires root, detailed power per component), `macmon` (no sudo needed, real-time monitoring), `fluidtop` (GPU utilization), and Activity Monitor. For distributed workloads, monitor per-node GPU utilization and interconnect bandwidth to identify bottlenecks. [Finding #380]

### Q: What about non-ML distributed GPU computing?
**A**: Currently, no established distributed MapReduce-over-Metal framework exists for Apple Silicon. The entire distributed ecosystem is ML-focused (MLX, Exo, dnet). General-purpose distributed GPU compute across Macs would require building custom communication layers on top of RDMA/ibverbs or using MLX's distributed primitives as a transport layer for non-ML workloads. [Finding #436]

## Cross-References

### Related Skills

- **mlx-compute** (Layer 3): MLX is the primary framework for distributed operations on Apple Silicon — mlx-compute covers single-node MLX usage, optimization, and custom Metal kernels that serve as building blocks for distributed workloads
- **unified-memory** (Layer 0): Unified memory architecture is foundational to RDMA zero-copy transfers — understanding SLC behavior, virtual address translation, and coherency informs how distributed memory access patterns perform
- **gpu-perf** (Layer 2): Performance profiling of distributed workloads requires understanding single-node bottlenecks (compute vs memory bound) before scaling across nodes — interconnect overhead analysis builds on gpu-perf profiling techniques
- **gpu-centric-arch** (Layer 4): GPU-centric architecture patterns extend naturally to distributed settings — multi-node GPU-first designs must account for interconnect latency and bandwidth constraints covered in gpu-distributed

### Key Dependencies

This skill depends on:
- **unified-memory** for understanding the memory model that enables zero-copy RDMA transfers
- **mlx-compute** for the single-node MLX foundation that distributed operations extend
- **gpu-perf** for profiling methodology applied to distributed workload analysis

Higher-layer skills that build on gpu-distributed:
- **gpu-centric-arch** uses distributed patterns for multi-node GPU-centric system designs

### Finding ID Reference

Key findings by topic area:
- **RDMA basics**: #58, #67, #303, #306, #309, #313
- **RDMA implementation**: #412, #413, #414, #415, #416, #417, #418
- **RDMA bandwidth/latency**: #306, #417, #418, #419, #420
- **Zero-copy transfers**: #377, #421
- **JACCL backend**: #59, #329, #415, #416
- **Network topology**: #316, #322, #423, #424, #425
- **MLX distributed API**: #324, #333, #337, #426
- **Parallelism strategies**: #346, #366, #431, #432, #433, #434
- **Exo framework**: #341, #346, #350, #427, #428, #429, #430
- **dnet framework**: #369, #435
- **Reference clusters**: #354, #359, #419, #420, #425
- **GPU-centric communication**: #549
- **Non-ML distributed**: #436
- **Production/monitoring**: #380, #437

Use `${CLAUDE_PLUGIN_ROOT}/scripts/kb detail <id>` to retrieve full finding details including source URLs and confidence levels.

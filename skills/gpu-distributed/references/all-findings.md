# gpu-distributed — All Findings (51)

## Finding 435: dnet (firstbatchxyz) is distributed LLM inference using pipelined-ring execution with heterogeneity-aware solver (distilp) that optimizes layer assignment. Auto-detects Thunderbolt. Alpha v0.0.1, Apple Silicon only. mlx_sharding (mzbac) implements pipeline parallelism for MLX LLMs.
**Confidence**: high
**Source**: dnet: Distributed LLM Inference for Apple Silicon
**Evidence**: GitHub repositories for both projects
**Tags**: dnet,mlx-sharding,alternative-frameworks

## Finding 341: Exo 1.0 is a peer-to-peer distributed AI inference framework with day-0 RDMA support, auto-discovery via libp2p, and topology-aware partitioning
**Confidence**: verified
**Source**: high
**Evidence**: No master-worker architecture — devices connect peer-to-peer. Uses MLX as primary inference backend and MLX distributed for inter-device communication. libp2p-based auto-discovery without manual configuration. Topology-aware auto parallel evaluates device resources and network characteristics. Built-in dashboard for cluster management. Custom namespace isolation via EXO_LIBP2P_NAMESPACE. Released under Apache 2.0 license.
**Tags**: 

## Finding 350: Exo pipeline: single M4 Pro 49.3 TPS, 3-device 39.7 TPS single-req but 108.8 TPS multi-req (2.2x scaling). RDMA Qwen3-235B: 31.9 TPS on 4 nodes
**Confidence**: high
**Source**: high
**Evidence**: LLaMA 3.2 3B on M4 Pro: 1 node = 49.3 TPS, 2 nodes = 44.4 TPS single/95.7 TPS multi, 3 nodes = 39.7 TPS single/108.8 TPS multi. Single-request perf degrades with more nodes due to network overhead. Multi-request scales nearly linearly. Network latency, not bandwidth, is the bottleneck — activations for Llama 3.2 3B are <4KB. Qwen3-235B with RDMA: 2 nodes = 26.2 TPS, 4 nodes = 31.9 TPS.
**Tags**: 

## Finding 427: Exo is peer-to-peer distributed inference (no master-worker), uses MLX on Apple Silicon and Tinygrad on CUDA/ROCm. Features: automatic device discovery via libp2p, topology-aware auto-parallel scheduling, OpenAI-compatible REST API on localhost:52415. Version 1.0 has day-0 RDMA support.
**Confidence**: verified
**Source**: exo-explore/exo GitHub repository
**Evidence**: GitHub repository (41.3k stars, Apache 2.0) documents architecture
**Tags**: exo,peer-to-peer,mlx,rdma,openai-api

## Finding 428: Exo default partitioning: RingMemoryWeightedPartitioning assigns layers proportionally to device memory. Exo 1.0 demonstrated heterogeneous cluster inference combining 2 DGX Spark + M3 Ultra: DGX handles prefill 3.8x faster (100 TFLOPS), M3 Ultra handles decode 3.4x faster (819 GB/s bandwidth), achieving 2.8x overall speedup.
**Confidence**: high
**Source**: Combining NVIDIA DGX Spark + Apple Mac Studio with EXO 1.0
**Evidence**: Exo source code for partitioning; blog.exolabs.net DGX Spark benchmark
**Tags**: exo,heterogeneous,dgx-spark,partitioning

## Finding 429: Exo pipeline parallelism on LLaMA 3.2 3B: adding devices decreases single-request perf (49.3→39.7 tok/s on 3 nodes) but multi-request throughput scales nearly linearly (49.3→108.8 tok/s on 3 nodes, ~2.2x). Network latency is bottleneck, not bandwidth.
**Confidence**: high
**Source**: Transparent Benchmarks - 12 days of EXO
**Evidence**: Transparent benchmarks published by EXO Labs
**Tags**: exo,pipeline-parallelism,throughput,benchmark

## Finding 430: Heterogeneous MLX + Tinygrad clusters exhibit 90% performance degradation (48→5 tok/s). Exo's production readiness limited by security, fault tolerance, and tooling gaps. Open-source version is 0.0.15-alpha; full 1.0 not fully public.
**Confidence**: high
**Source**: Deep Dive: Exo - Distributed AI Inference
**Evidence**: Medium deep dive and HN analysis documents performance cliff
**Tags**: exo,heterogeneous,limitations,production

## Finding 346: Exo supports pipeline parallelism (layer sharding) and tensor parallelism (tensor sharding) with up to 1.8x speedup on 2 devices, 3.2x on 4
**Confidence**: verified
**Source**: high
**Evidence**: Pipeline parallelism: splits model layers across devices sequentially. Default strategy is ring memory weighted partitioning — each device runs layers proportional to its memory. Tensor parallelism: shards individual tensors across devices. Handles heterogeneous clusters by accounting for device memory, measuring real-time bandwidth/latency between pairs, and adapting strategy based on topology metrics.
**Tags**: 

## Finding 549: GPU-centric communication (reducing CPU involvement in inter-GPU data transfer) is categorized into GPU-aware (CPU initiates, GPU data), GPU-initiated (GPU triggers transfers), and GPU-driven (GPU manages entire communication flow) paradigms, with each offering progressively lower latency
**Confidence**: high
**Source**: The Landscape of GPU-Centric Communication
**Evidence**: Comprehensive survey of GPU-centric communication mechanisms. Reviews vendor technologies (GPUDirect, NVLink, SHARP), communication libraries (NCCL, RCCL), and research approaches. Categorizes techniques by level of GPU autonomy. Covers intra-node and inter-node scenarios. Identifies that workload decomposition and scheduling, previously handled by hardware, now requires manual programmer management.
**Tags**: gpu-centric,communication,gpu-initiated,gpu-driven,inter-gpu,nccl,rdma

## Finding 421: RDMA over TB5 provides zero-copy transfers — data moves directly between devices without CPU buffering. On Apple Silicon with unified memory, RDMA read/write to a Mac's memory is effectively GPU memory access. However, implementation functions as accelerated collective communications (like NCCL), NOT true shared memory across nodes.
**Confidence**: high
**Source**: HN discussion - 1.5 TB of VRAM on Mac Studio
**Evidence**: HN discussion clarifying zero-copy semantics vs distributed shared memory
**Tags**: zero-copy,unified-memory,nccl,gpu-transfer

## Finding 422: For distributed LLM inference, tensor parallelism shards each layer's weights across all machines. After each layer's computation, all_reduce synchronizes partial results. This requires low-latency communication at every layer. 3.5x speedup for token generation (decode) at batch size 1 over 4 machines demonstrated.
**Confidence**: high
**Source**: HN - macOS 26.2 enables fast AI clusters with RDMA
**Evidence**: HN discussion and Alex Cheema (Exo creator) confirming architecture
**Tags**: tensor-parallelism,allreduce,sharding,inference

## Finding 329: JACCL (Jack and Angelos Collective Communication Library) is MLX's RDMA backend using InfiniBand Verbs over Thunderbolt 5
**Confidence**: verified
**Source**: high
**Evidence**: Named as a pun on NVIDIA's NCCL, tribute to Jack Beasley who led RDMA over Thunderbolt at Apple. Uses ibverbs (InfiniBand verbs) for direct memory access. Requires fully-connected mesh — all node pairs need direct TB5 cables. Config JSON maps rdma device names per rank. Environment vars: MLX_RANK, MLX_JACCL_COORDINATOR, MLX_IBV_DEVICES. Setting MLX_METAL_FAST_SYNCH=1 enables faster CPU-GPU synchronization critical for low-latency comms.
**Tags**: 

## Finding 354: 4x Mac Studio M3 Ultra cluster with 1.5TB unified memory ran Kimi K2 Thinking (1T params) at 15 TPS using RDMA
**Confidence**: high
**Source**: high
**Evidence**: Config: 2x 512GB (,699 each) + 2x 256GB (,099 each) M3 Ultra Mac Studios. Total ~K. Achieved 3.7 TFLOPS FP64 (vs 1.3 TFLOPS single node). Kimi K2 Thinking 1T A32B (32B active params): 4 nodes with Exo RDMA = 28.3 TPS. Llama over 2 nodes: 18.5 TPS standard, 21.6 TPS with Exo RDMA. Under 250 watts per machine. Total power < 500W for 4 nodes. Compact 10-inch rack.
**Tags**: 

## Finding 324: MLX provides all_sum, all_gather, send, recv, recv_like distributed operations with 4 backends: ring, JACCL, NCCL, MPI
**Confidence**: verified
**Source**: high
**Evidence**: all_sum: aggregate arrays by summing across all processes. all_gather: collect arrays from all processes to all. send/recv: point-to-point (ring/JACCL/NCCL only). recv_like: receive matching template shape/dtype. init() returns Group object with rank() and size(). When world_size=1, all ops are no-ops. Multiple backends can coexist. nn.average_gradients() is the recommended pattern for gradient averaging.
**Tags**: 

## Finding 426: MLX exposes six core distributed operations: all_sum (collective reduction), all_gather (array concatenation across ranks), send (point-to-point), recv (explicit receive with shape/dtype), recv_like (shape-inferred receive), init (initialization). Operations become no-ops with single process.
**Confidence**: verified
**Source**: Distributed Communication API - MLX documentation
**Evidence**: MLX API reference documentation
**Tags**: mlx,api,all_sum,all_gather,send,recv

## Finding 337: MLX supports distributed data-parallel training with nn.average_gradients() for gradient sync, and distributed LoRA fine-tuning
**Confidence**: verified
**Source**: high
**Evidence**: nn.average_gradients() walks full gradient structure applying all_sum/size to each tensor. Ring backend works for training over Ethernet/Thunderbolt TCP. JACCL provides ~10x lower latency for gradient sync. MLX has supported distributed training without MPI since Aug 2025. mlx-lm provides built-in LoRA and QLoRA fine-tuning. More complex models need more bandwidth for weight sharing — Thunderbolt preferred over WiFi.
**Tags**: 

## Finding 333: mlx.launch orchestrates distributed programs across nodes; mlx.distributed_config auto-discovers Thunderbolt topology and generates hostfiles
**Confidence**: verified
**Source**: high
**Evidence**: mlx.launch -n 4 script.py runs 4 local processes. mlx.launch --hosts ip1,ip2 script.py for remote nodes via SSH. mlx.launch --backend ring --hostfile file.json script.py for specific backend. mlx.distributed_config --hosts h1,h2 --over thunderbolt --dot visualizes connections. --auto-set auto-configures nodes and saves hostfile. Config discovers TB ring topology by SSHing to each node and extracting connectivity.
**Tags**: 

## Finding 423: JACCL requires fully connected (full mesh) topology. For N nodes: N*(N-1)/2 cables (4 nodes = 6 cables). Apple does not allow cycles. Ring and bandwidth-optimized topologies are listed as future enhancements. Current practical limit is 4 nodes due to port count (Mac Studio M3 Ultra has 6 TB5 ports).
**Confidence**: verified
**Source**: Distributed Communication - MLX documentation
**Evidence**: MLX docs mandate fully connected topology; Jeff Geerling notes practical 4-node limit
**Tags**: topology,full-mesh,scaling,cable-requirements

## Finding 424: TB5 cables must be rated for 40Gbps+ and kept under 2 meters. Separate Ethernet VLAN (10GbE recommended) for management and API communication, with Thunderbolt exclusively for RDMA data plane. IBV device config uses JSON hostfile with SSH hostname, IPs, and rdma device array per rank.
**Confidence**: verified
**Source**: Distributed Communication - MLX documentation
**Evidence**: Stabilise.io and MLX docs describe cabling and config requirements
**Tags**: cabling,configuration,hostfile,management-network

## Finding 425: Jeff Geerling reference cluster: 4x Mac Studio M3 Ultra (two 512GB + two 256GB = 1.5TB total), 32-core CPU, 80-core GPU, 32-core Neural Engine each. Cost ~$38-40K. All published RDMA benchmark results use this config.
**Confidence**: high
**Source**: 1.5 TB of VRAM on Mac Studio - Jeff Geerling
**Evidence**: Jeff Geerling blog with Apple-provided hardware specs
**Tags**: hardware,reference-setup,cost,m3-ultra

## Finding 436: No established distributed MapReduce-over-Metal framework exists for Apple Silicon clusters as of 2026. Apple GPUs lack FP64 support, limiting scientific computing. Parallel reduction on Metal has been optimized for M1 but no framework abstracts this across multiple machines.
**Confidence**: verified
**Source**: Apple Silicon M-Series SoCs for HPC
**Evidence**: Web search found no distributed non-ML GPU frameworks; arXiv:2502.05317 confirms FP64 gap
**Tags**: non-ml,hpc,fp64,gap

## Finding 431: Expert parallelism on 2-4 node M2 Ultra cluster running DBRX (132B, 16 experts): 6.1 tok/s (2 nodes), 7.0 tok/s (4 nodes). Communication overhead 23% for 2 nodes, 33% for 4 nodes over 10GbE. 2-node cluster ($13,198) is 1.15x more cost-efficient per-token than 8x H100 ($289,000).
**Confidence**: verified
**Source**: Multi-Node Expert Parallelism on Apple Silicon
**Evidence**: Peer-reviewed paper (RACS 2025, arXiv:2506.23635)
**Tags**: expert-parallelism,moe,dbrx,cost-efficiency

## Finding 432: Pipeline parallel training demonstrated by Matt Beton (mlx-train): 2x M3 Ultra via Thunderbolt, fine-tuning DeepSeek (671GB) with LoRA at ~30 tok/s with 50% utilization. Required solving 3 MLX incompatibilities: auxiliary loss injection, lower-level value_and_grad, and mx.depends() for dependency enforcement.
**Confidence**: high
**Source**: Pipeline Parallel Training with Apple Silicon
**Evidence**: Blog post and GitHub repo MattBeton/mlx-train
**Tags**: pipeline-parallelism,training,deepseek,lora

## Finding 433: ICML 2025 paper introduced KPOP optimizer using Adam in Kronecker-factored eigenbasis, designed for Apple Silicon's unique constraints (high VRAM, low FLOPS, poor inter-node bandwidth). KPOP iterations take 10-20% longer than Adam but are more efficient per-iteration.
**Confidence**: verified
**Source**: Towards Large-scale Training on Apple Silicon - ICML 2025
**Evidence**: ICML 2025 paper (OpenReview TJjP8d5bms)
**Tags**: kpop,optimizer,training,icml-2025

## Finding 434: Apple uses data parallelism, tensor parallelism, sequence parallelism, and FSDP to train their foundation models. Server model uses Parallel-Track Mixture-of-Experts (PT-MoE) with multiple smaller transformers processing tokens independently, synchronizing at input/output boundaries.
**Confidence**: verified
**Source**: Introducing Apple's On-Device and Server Foundation Models
**Evidence**: Apple Machine Learning Research publication
**Tags**: apple-foundation-models,parallelism,pt-moe

## Finding 437: Current limitations: max 4 Macs in RDMA mesh, NVLink delivers 1800 GB/s vs TB5's 10 GB/s (180x faster), no production monitoring/fault tolerance frameworks for Mac clusters. Mac clusters excel at inference and cost-sensitive deployments but not competitive for training.
**Confidence**: high
**Source**: macOS 26.2 RDMA AI Clusters
**Evidence**: Multiple sources confirm 4-node limit, bandwidth gap, and tooling gaps
**Tags**: limitations,production,nvlink,training

## Finding 303: macOS Tahoe 26.2 (Dec 2025) introduces RDMA over Thunderbolt 5 via rdma_thunderbolt kernel extension and Network.framework APIs
**Confidence**: high
**Source**: high
**Evidence**: RDMA allows direct memory-to-memory transfers bypassing OS/CPU. Enabled via recovery mode command 'rdma_ctl enable'. Exposed through standard InfiniBand APIs (ibverbs). Thunderbolt controller on both ends coordinates direct memory transfers. The feature is exposed to software using standard Infiniband APIs, the same connectivity interface used in supercomputing.
**Tags**: 

## Finding 306: TB5 RDMA achieves 50-60 Gbps practical throughput (80 Gb/s theoretical) with 5-9 microsecond latency, down from 300us with TCP
**Confidence**: high
**Source**: high
**Evidence**: Jeff Geerling measured latency from 300us down to <50us with RDMA. Developer reports show 5-9 microsecond latencies matching datacenter InfiniBand. Practical bandwidth is 50-60 Gbps vs 80 Gb/s theoretical max. Compared to standard Thunderbolt TCP networking, RDMA provides ~99% latency reduction.
**Tags**: 

## Finding 417: TB5 RDMA achieves 5-9 microsecond latency, down from ~300 microseconds with TCP over the same Thunderbolt link — a 30-60x reduction. Multiple independent sources confirm sub-10us latency matching datacenter-class InfiniBand.
**Confidence**: high
**Source**: Apple's RDMA Revolution - Stabilise.io
**Evidence**: Stabilise.io reports 5-9us, Jeff Geerling reports <50us, WebProNews reports sub-10us
**Tags**: rdma,latency,benchmark,thunderbolt-5

## Finding 418: TB5 RDMA bandwidth comparison: 10GbE=10Gbps, TB5 RDMA=80Gbps (50-60 effective), InfiniBand HDR=200Gbps, NVLink 4th gen=900Gbps. TB5 latency (5-9us) matches InfiniBand but at fraction of cost. Good enough for inference, not for training.
**Confidence**: high
**Source**: macOS 26.2 RDMA Thunderbolt 5 AI Clusters
**Evidence**: Multiple blog sources compare interconnect technologies
**Tags**: rdma,bandwidth,infiniband,nvlink,comparison

## Finding 419: Real-world LLM inference on 4x Mac Studio M3 Ultra (1.5TB): Qwen3 235B at 32 tok/s, Kimi K2 Thinking (1T params) at ~30 tok/s, DeepSeek V3.1 (671B) at ~25 tok/s. Adding a 2nd 512GB node increased throughput only 32% (21.1 to 27.8 tok/s), indicating bandwidth saturation at 80Gbps per link.
**Confidence**: high
**Source**: 1.5 TB of VRAM on Mac Studio - Jeff Geerling
**Evidence**: Jeff Geerling benchmark with Apple-provided hardware
**Tags**: benchmark,inference,llm,cluster-performance

## Finding 420: Power efficiency: 4x Mac Studio cluster draws <500W for trillion-parameter inference (600W peak, 66W idle across all four). Comparable 8x NVIDIA H200 setup consumes 5,600W. Cost: ~$38K vs $270K+ for NVIDIA equivalent.
**Confidence**: high
**Source**: AI calculations on Mac cluster - AppleInsider
**Evidence**: AppleInsider and multiple blog sources confirm power and cost figures
**Tags**: power-efficiency,cost-comparison,nvidia

## Finding 309: RDMA requires Thunderbolt 5 — only available on M4 Pro, M4 Max, and M3 Ultra Macs. Base M4 (TB4) cannot use RDMA
**Confidence**: high
**Source**: high
**Evidence**: RDMA-capable devices: M4 Pro Mac Mini, M4 Max Mac Studio, M4 Max MacBook Pro, M3 Ultra Mac Studio. Cannot use the Thunderbolt 5 port next to the Ethernet port on Mac Studio. All five TB5 ports on Mac Studio are RDMA-capable. Mac Mini M4 has TB4 only — clustering without RDMA causes 91% performance degradation.
**Tags**: 

## Finding 412: macOS Tahoe 26.2 RDMA implementation uses standard InfiniBand verbs (ibverbs) API — devices appear as rdma_en2, rdma_en3 etc. The librdma library is loaded dynamically via dlopen for compatibility. Compilation requires SDK >= 26.2 for ibverbs.h headers.
**Confidence**: verified
**Source**: Thunderbolt RDMA communications backend - MLX PR #2808
**Evidence**: MLX PR #2808 documents dynamic loading of librdma and SDK requirements
**Tags**: rdma,ibverbs,macos-26.2,api

## Finding 413: RDMA must be enabled via macOS Recovery Mode by running rdma_ctl enable from Terminal. Cannot be done remotely or with sudo — requires physical access. Deliberate security gate.
**Confidence**: verified
**Source**: Distributed Communication - MLX documentation
**Evidence**: MLX distributed documentation and Jeff Geerling blog confirm procedure
**Tags**: rdma,security,setup,recovery-mode

## Finding 414: Apple ships ibverbs-compatible RDMA APIs and also MLX5 drivers for Mellanox ConnectX NICs natively in macOS, suggesting internal datacenter use. macOS does NOT support RoCE (RDMA over Converged Ethernet), limiting RDMA to Thunderbolt transport only.
**Confidence**: high
**Source**: HN discussion - 1.5 TB of VRAM on Mac Studio
**Evidence**: Hacker News discussion with knowledgeable commenters confirming MLX5 drivers and no RoCE
**Tags**: rdma,ibverbs,mellanox,roce,api

## Finding 415: JACCL (Jack and Angelos' Collective Communication Library) is named in tribute to Jack Beasley who led RDMA over Thunderbolt development at Apple. It supports all_reduce, all_gather, send/recv, and reduce. Point-to-point send/recv has known SIGBUS crash issues with asymmetric timing; workaround uses all_sum broadcasts.
**Confidence**: verified
**Source**: Thunderbolt RDMA communications backend - MLX PR #2808
**Evidence**: MLX PR #2808 and documentation describe JACCL origins and known issues
**Tags**: jaccl,collective-communication,rdma,bugs

## Finding 416: JACCL configuration uses mlx.distributed_config CLI that discovers Thunderbolt connectivity via SSH, validates mesh completeness, and generates config files. Key env vars: MLX_RANK, MLX_JACCL_COORDINATOR (rank 0 IP:port), MLX_IBV_DEVICES (JSON mapping). MLX_METAL_FAST_SYNCH=1 is critical for low-latency communication.
**Confidence**: verified
**Source**: Distributed Communication - MLX documentation
**Evidence**: MLX distributed documentation provides exact commands and env var specs
**Tags**: jaccl,configuration,environment-variables

## Finding 313: RDMA enabling requires physical access (recovery mode), uses IOMMU protection, and should be kept on isolated network
**Confidence**: high
**Source**: high
**Evidence**: Physical access to each Mac required — cannot be enabled remotely. IOMMU for each DMA agent including Thunderbolt controllers restricts memory to explicitly mapped regions. Unauthorized access attempts trigger kernel panic. The rdma_ctl enable command bypasses normal entitlement checks — only enable on trusted machines. Keep TB5 RDMA fabric physically separate from general network.
**Tags**: 

## Finding 377: RDMA over TB5 provides true zero-copy data transfers — data moves directly from one device memory to another without intermediate buffering
**Confidence**: high
**Source**: high
**Evidence**: Zero-copy eliminates extra memory copies between devices. CPU offloading frees cores for compute tasks. Data flows at up to 80 Gb/s with TB5. Only M4 and later chips fully support the enhanced RDMA mode. Software treats Thunderbolt-connected devices as direct memory peers via Network.framework APIs. Developers can write Swift or Objective-C code accessing remote memory directly.
**Tags**: 

## Finding 372: Ray + vLLM can create distributed inference on Mac Mini clusters, but vLLM only has experimental CPU support for Apple Silicon — no GPU acceleration
**Confidence**: high
**Source**: medium
**Evidence**: vLLM was designed for CUDA GPUs, not Apple Silicon. Experimental macOS support is CPU-only, not GPU. Workaround uses Transformers library + PyTorch MPS backend. Focus on 1-7B parameter models with quantization. Terraform-based setup scripts exist (AI-Cluster-Distribution repo). Ray provides abstraction layer coordinating between different hardware types. For Apple Silicon clusters, Exo and MLX distributed are far superior to Ray+vLLM.
**Tags**: 

## Finding 316: JACCL RDMA backend requires fully-connected mesh topology — every Mac must have direct TB5 cable to every other Mac
**Confidence**: high
**Source**: high
**Evidence**: No Thunderbolt 5 switches exist, so no star topology is possible. JACCL requires a cable between all pairs of Macs. Max observed cluster: 4 Macs fully meshed. A 4-node full mesh needs 6 cables (n*(n-1)/2). Apple said all 5 TB5 ports are RDMA-enabled, suggesting 5-node clusters may be possible but untested. Ring topology (MLX ring backend) also works but with higher latency over TCP.
**Tags**: 

## Finding 359: 4x Mac Studio cluster (~K) runs DeepSeek V3.1 at 24-26 TPS; comparable NVIDIA setup starts at K — 94% cost reduction
**Confidence**: high
**Source**: high
**Evidence**: A ,000 four-Mac-Studio cluster now runs 700GB models like DeepSeek V3.1. Comparable NVIDIA hardware (DGX systems) starts at ,000. macOS 26.2 RDMA slashed cost of running frontier-class AI models locally by roughly 94%. Total power under 500W vs kilowatts for NVIDIA DGX. 1.15x more cost-efficient than state-of-the-art NVIDIA H100 GPU AI supercomputers per the expert parallelism paper.
**Tags**: 

## Finding 67: RDMA over Thunderbolt 5 creates a distributed shared memory model across Mac clusters. Key properties: (1) one node can directly read another's memory (bypassing CPU), (2) unified memory on each node means GPU-accessible memory is also RDMA-accessible, (3) sub-10us latency enables fine-grained distributed operations, (4) total memory pool scales linearly (4x 512GB = 2TB). LLMs are sharded across nodes with each Mac storing chunks accessible by any peer. This is NOT cache-coherent across nodes — application-level consistency required.
**Confidence**: high
**Source**: WebProNews - macOS Tahoe 26.2 RDMA for AI Mac Clusters
**Evidence**: Jeff Geerling demo, Apple developer documentation for macOS 26.2, MLX JACCL implementation details.
**Tags**: rdma,distributed-memory,memory-model,cluster,sharding,thunderbolt5

## Finding 369: dnet by FirstBatch is an alternative distributed LLM inference framework for Apple Silicon with pipelined-ring parallelism and disk streaming
**Confidence**: verified
**Source**: medium
**Evidence**: Features: runs models exceeding total cluster memory via compute/I/O overlap. Optimized for unified memory architecture — efficient layer swapping. OpenAI-compatible /v1/chat/completions endpoint. Auto-detects Thunderbolt for high-bandwidth inter-device communication. TUI built in Rust for model loading, topology view, and chat. Fuses pipelined-ring parallelism, disk streaming, and UMA-aware scheduling. Dynamic topology — nodes start without models, API discovers devices and distributes layers via distilp.
**Tags**: 

## Finding 363: Multi-node expert parallelism on M2 Ultra cluster achieved 6.1 tok/s on DBRX 132B with 5.2x MoE speedup, 1.15x more cost-efficient than H100
**Confidence**: verified
**Source**: high
**Evidence**: Research paper (arxiv:2506.23635) tested on Mac Studio cluster with M2 Ultra (76-core GPU, 192GB unified memory) over 10Gb Ethernet. DBRX 132B MoE model: 16 experts per layer, selecting 4 per token. 2-node config: 8 experts per node. Fork-join execution with decentralized self-attention/router halves communications. Expert parallelism chosen over tensor parallel (less communication) and pipeline parallel (better perf at small batch sizes). Apple driver processing creates overhead when weights not prestacked.
**Tags**: 

## Finding 59: MLX integrated JACCL (Thunderbolt RDMA) backend via PR #2808. Uses ibverbs (InfiniBand verbs) over Thunderbolt 5. Features: latency-optimized, full mesh topology, supports all_sum/all_gather/reduce/send/recv. Requires: macOS 26.2+, Thunderbolt 5, M3 Ultra or newer. Setup: enable RDMA in Recovery Mode (rdma_ctl enable). Uses dynamic dlopen() for librdma for cross-version compatibility. Ring and bandwidth-optimized topologies planned.
**Confidence**: verified
**Source**: MLX PR #2808 - Thunderbolt RDMA Communications Backend
**Evidence**: MLX PR #2808 merged, contains detailed benchmarks comparing RDMA vs TCP ring backend on M3 Ultra systems for 4-way all-reduce across 1-1024KB data sizes.
**Tags**: mlx,jaccl,rdma,infiniband,ibverbs,distributed,thunderbolt5

## Finding 366: For Apple Silicon clusters: expert parallelism needs least communication, tensor parallel gives best speedup, pipeline parallel easiest to implement
**Confidence**: verified
**Source**: medium
**Evidence**: Data parallelism: processes larger batches by distributing across nodes, efficient even with slow cross-node comms. Tensor parallelism: parallelizes within an operation (e.g., matrix multiply), best speedup but highest communication. Pipeline parallelism: parallelizes between layers, sequential, easiest partitioning. Expert parallelism (MoE): only routes active experts, requires less communication than tensor parallel. Hybrid strategies combine multiple approaches. Heterogeneous clusters (M4 + M4 Max) need memory-weighted partitioning.
**Tags**: 

## Finding 58: macOS Tahoe 26.2 enables RDMA over Thunderbolt 5 for Mac clusters. Bandwidth: 80 Gb/s bidirectional. Latency: sub-10 microseconds (5-9us measured), down from 300us with TCP. Zero-copy transfers: data moves directly from one device's memory to another's without intermediate buffering or CPU involvement. 4x Mac Studio cluster: 1.5TB pooled unified memory, runs 1T-parameter models (Kimi-K2-Thinking) at 15 tok/s. Qwen3 235B: 31.9 tok/s on 4 nodes.
**Confidence**: high
**Source**: Jeff Geerling - 1.5TB VRAM on Mac Studio via RDMA/TB5
**Evidence**: Jeff Geerling tested 4x Mac Studio. WebProNews reports 3us latency. MLX PR #2808 (JACCL backend) merged for Thunderbolt RDMA distributed communication.
**Tags**: rdma,thunderbolt5,distributed,memory-pooling,zero-copy,latency,cluster

## Finding 322: MLX ring backend uses sequential node ring (TCP); JACCL uses fully-connected mesh (RDMA) with ~10x lower latency
**Confidence**: verified
**Source**: high
**Evidence**: Ring topology: rank 0 <-> rank 1 <-> rank 2 ... rank n-1 -> rank 0. Ring only allows communication with adjacent ranks. Works over Ethernet or Thunderbolt TCP. JACCL mesh: requires direct cables between all pairs. Supports arbitrary send/recv between any pair. JACCL provides ~10x latency improvement over TCP ring. Ring is always available; JACCL requires macOS 26.2+ and TB5.
**Tags**: 

## Finding 380: Mac clusters monitored via powermetrics (root), macmon (no sudo), fluidtop, and ProcessInfo.thermalState API for throttling detection
**Confidence**: verified
**Source**: medium
**Evidence**: powermetrics: CPU usage, timer/interrupt frequencies, C-state stats, frequency distribution, estimated power by SoC subsystem (CPU, GPU, ANE). macmon: streams Apple Silicon stats as newline-delimited JSON, GPU power/frequency/utilization/temperatures. fluidtop: CLI tool with thermal state alerts, supports M1-M5. ProcessInfo.thermalState: nominal to critical states. High temp + frequency drop + elevated utilization = thermal throttling. Mac Studio under 250W per node sustained. 4-node cluster < 500W total.
**Tags**: 


#ifndef TYPES_H
#define TYPES_H

#include <metal_stdlib>
using namespace metal;

struct AttentionParams {
    uint seq_len;           // sequence length N
    uint head_dim;          // head dimension D
    uint num_heads;         // number of Q heads
    uint num_kv_heads;      // number of KV heads (== num_heads for MHA, < for GQA)
    uint block_r;           // tile rows (queries)
    uint block_c;           // tile columns (keys)
    float scale;            // 1/sqrt(D)
    uint variant;           // 0=standard, 1=RoPE, 2=ALiBi, 3=GQA
    // Paged attention specific
    uint page_size;         // tokens per page
    uint num_pages;         // total pages allocated
    uint max_context_len;   // maximum context length
    uint num_partitions;    // partitioned reduce count
    uint _pad0;
    uint _pad1;
    uint _pad2;
    uint _pad3;
};
// Total: 64 bytes, 4-byte aligned

#endif

#ifndef TYPES_H
#define TYPES_H

struct ExpParams {
    uint element_count;  // Number of elements to process
    uint num_passes;     // Passes (exp1), capacity (exp3), num_tiles (exp4), num_dispatches (exp5)
    uint mode;           // Experiment-specific mode selector
};

// Experiment 4: Decoupled lookback flags (packed into upper 2 bits of uint32)
#define TILE_SIZE       256
#define FLAG_NOT_READY  0u
#define FLAG_AGGREGATE  1u
#define FLAG_PREFIX     2u
#define FLAG_SHIFT      30u
#define VALUE_MASK      ((1u << FLAG_SHIFT) - 1u)

#endif

#ifndef TYPES_H
#define TYPES_H

// Decoupled lookback flags (required by exp16_partition)
#define FLAG_NOT_READY  0u
#define FLAG_AGGREGATE  1u
#define FLAG_PREFIX     2u
#define FLAG_SHIFT      30u
#define VALUE_MASK      ((1u << FLAG_SHIFT) - 1u)

// Demo visualization params
struct DemoParams {
    uint element_count;
    uint texture_width;
    uint texture_height;
    uint max_value;   // 0xFFFFFFFF for u32
};

// exp17 structs
struct Exp17Params {
    uint element_count;
    uint num_tiles;
    uint shift;
    uint pass;
};

struct BucketDesc {
    uint offset;
    uint count;
    uint tile_count;
    uint tile_base;
};

// exp16 struct
struct Exp16Params {
    uint element_count;
    uint num_tiles;
    uint num_tgs;
    uint shift;
    uint pass;
};

#endif

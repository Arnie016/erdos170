#include <metal_stdlib>
using namespace metal;

// Vertex IDs start at one. A zero ID denotes an unoccupied coordinate.
kernel void missingDistances(device const uint2 *deletions [[buffer(0)]],
                             device const uint *offsets [[buffer(1)]],
                             device const uint2 *edges [[buffer(2)]],
                             device uchar *missing [[buffer(3)]],
                             constant uint &length [[buffer(4)]],
                             uint2 pos [[thread_position_in_grid]]) {
    uint d = pos.x + 1;
    uint2 removed = deletions[pos.y];
    for (uint i = offsets[d]; i < offsets[d + 1]; ++i) {
        uint2 edge = edges[i];
        if (all(edge != removed.x) && all(edge != removed.y)) {
            missing[pos.y * length + pos.x] = 0;
            return;
        }
    }
    missing[pos.y * length + pos.x] = 1;
}

inline bool survives(uint id, uint2 removed) {
    return id != 0 && id != removed.x && id != removed.y;
}

kernel void bestInsertion(device const uint2 *deletions [[buffer(0)]],
                          device const uint *owners [[buffer(1)]],
                          device const uint *offsets [[buffer(2)]],
                          device const uint *lost [[buffer(3)]],
                          device atomic_uint *best [[buffer(4)]],
                          constant uint &length [[buffer(5)]],
                          uint2 pos [[thread_position_in_grid]]) {
    uint z = pos.x;
    if (owners[z] != 0) return;
    uint p = pos.y;
    uint2 removed = deletions[p];
    uint incumbent = atomic_load_explicit(&best[p], memory_order_relaxed);
    uint failures = 0;
    for (uint i = offsets[p]; i < offsets[p + 1]; ++i) {
        uint d = lost[i];
        bool covered = (z >= d && survives(owners[z - d], removed))
                    || (z + d <= length && survives(owners[z + d], removed));
        if (!covered && ++failures > incumbent / (length + 1)) return;
    }
    // Lower distance deficit wins. Equal deficits select the smallest insertion.
    atomic_fetch_min_explicit(&best[p], failures * (length + 1) + z, memory_order_relaxed);
}

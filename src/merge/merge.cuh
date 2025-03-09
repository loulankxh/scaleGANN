#include <cuda_runtime.h>
#include <vector>
#include <cstdint>


void translateShardGPU(std::vector<std::vector<uint32_t>>& translated_index,
                                   std::vector<std::vector<uint32_t>>& index,
                                   std::vector<uint32_t>& idx_vec,
                                   uint32_t device);
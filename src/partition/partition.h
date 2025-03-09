#include <stdint.h>
#include <vector>


template <typename T>
uint32_t get_partition_num(uint32_t memGPU, size_t npts, uint32_t ndim, uint32_t degree, uint32_t dumplicate_factor);


template <typename T> void get_partitions(uint32_t memGPU, size_t npts, uint32_t ndim, uint32_t max_iters, 
                    uint32_t duplicate_factor, uint32_t partition_num, uint32_t partition_lower_bound,
                    const std::vector<std::vector<T>>& data,
                    std::vector<std::vector<std::vector<T>>>& partitions,
                    std::vector<std::vector<uint32_t>>& idx_map);

template <typename T> std::vector<std::vector<std::vector<T>>> main_partitions(uint32_t memGPU, 
            size_t npts, uint32_t ndim, uint32_t degree, uint32_t max_iters, 
            uint32_t duplicate_factor, uint32_t partition_num,
            const std::vector<std::vector<T>>& data,
            std::vector<std::vector<uint32_t>>& idx_map);
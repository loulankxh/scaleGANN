#include <stdint.h>
#include <vector>

template <typename T>
void get_partition_num(uint32_t memGPU, size_t npts, uint32_t ndim, uint32_t degree, 
                        uint32_t inter_degree, uint32_t threads, uint32_t dumplicate_factor,
                        uint32_t* partition_num, uint32_t* size_limt);


template <typename T> void get_partitions(uint32_t memGPU, size_t npts, uint32_t ndim, uint32_t max_iters, 
                    uint32_t duplicate_factor, uint32_t partition_num, uint32_t size_limit,
                    const std::vector<std::vector<T>>& data,
                    std::vector<std::vector<std::vector<T>>>& partitions,
                    std::vector<std::vector<uint32_t>>& idx_map);

template <typename T> std::vector<std::vector<std::vector<T>>> main_partitions(uint32_t memGPU, 
            size_t npts, uint32_t ndim, uint32_t degree, uint32_t inter_degree, uint32_t threads,
            uint32_t max_iters, uint32_t duplicate_factor, uint32_t partition_num,
            const std::vector<std::vector<T>>& data,
            std::vector<std::vector<uint32_t>>& idx_map);
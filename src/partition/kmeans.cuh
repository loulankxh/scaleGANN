#include <cuda_runtime.h>
#include <vector>
#include <cstdint>

template <typename T>
void kMeansCUDA(uint32_t partition_num, uint32_t ndim, uint32_t max_iters, uint32_t device,
const std::vector<std::vector<T>>& sample, std::vector<std::vector<float>>& centroids);

#include <stdint.h>
#include <vector>

template <typename T>
void kMeansCPU(uint32_t partition_num, uint32_t ndim, uint32_t max_iters, 
const std::vector<std::vector<T>>& sample, std::vector<std::vector<float>>& centroids);
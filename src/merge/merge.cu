#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <iostream>
#include <vector>
#include <cassert>
#include <omp.h>

#include <thread>
#include <chrono>

#include "merge.cuh"
#include "../taskScheduler/gpuManagement.h"

__global__ void translateShardKernel(
    uint32_t* translated_data,
    uint32_t* index_data,
    uint32_t* idx_vec_data,
    uint32_t neighbor_K,
    uint32_t shard_size) {

    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    uint32_t stride = blockDim.x * gridDim.x;
    for (uint32_t i = idx; i < shard_size; i += stride) {
        if (i >= shard_size) return;

        uint32_t global_index = idx_vec_data[i];

        uint32_t* neighbor_list = &index_data[i * neighbor_K];

        uint32_t offset = i * (neighbor_K + 1);
        translated_data[offset] = global_index; 
        for (uint32_t j = 0; j < neighbor_K; j++) {
            uint32_t neighbor_local_index = neighbor_list[j];
            // if (neighbor_local_index >= shard_size) return;
            translated_data[offset+j+1] = idx_vec_data[neighbor_local_index];
        }
    }

    // __syncthreads();
}


void translateShardGPU(std::vector<std::vector<uint32_t>>& translated_index,
                                   std::vector<std::vector<uint32_t>>& index,
                                   std::vector<uint32_t>& idx_vec,
                                   uint32_t gpu_id) {

    uint32_t shard_size = index.size();
    assert(shard_size == idx_vec.size());
    uint32_t neighbor_K = index[0].size();

    // cudaStream_t stream;
    // cudaStreamCreate(&stream); // instead of using default CUDA stream, which will block host CPU

    omp_set_lock(&gpu_locks[gpu_id]);

    cudaSetDevice(gpu_id);
    if (isGPUBusy(gpu_id)) {
        std::cerr << "GPU " << gpu_id << " is still busy. Waiting..." << std::endl;
        cudaEventSynchronize(gpu_end_events[gpu_id]);
    } else{
        std::cerr << "GPU " << gpu_id << " is ready for use" << std::endl;
    }

    cudaEventRecord(gpu_start_events[gpu_id], 0);
    // cudaEventRecord(gpu_start_events[gpu_id], stream);

    auto hostToDevice_Start = std::chrono::high_resolution_clock::now();

    thrust::device_vector<uint32_t> d_translated_data(shard_size * (neighbor_K + 1));
    thrust::device_vector<uint32_t> d_index_data(shard_size * neighbor_K);
    thrust::device_vector<uint32_t> d_idx_vec_data(shard_size);
    for (uint32_t i = 0; i < shard_size; ++i){
        cudaMemcpyAsync(thrust::raw_pointer_cast(d_index_data.data()) + i * neighbor_K, index[i].data(), neighbor_K * sizeof(uint32_t), cudaMemcpyHostToDevice);
    }
    cudaMemcpyAsync(thrust::raw_pointer_cast(d_idx_vec_data.data()), idx_vec.data(), shard_size * sizeof(uint32_t), cudaMemcpyHostToDevice);

    auto hostToDevice_End = std::chrono::high_resolution_clock::now();
    auto hostToDeviceDuration = std::chrono::duration_cast<std::chrono::milliseconds>(hostToDevice_Start - hostToDevice_End);
    printf("Host to Device Transfer time is: %lld ms\n", hostToDeviceDuration.count());
    
    cudaDeviceProp properties;
    cudaGetDeviceProperties(&properties, gpu_id);
    int threads = properties.maxThreadsPerBlock;
    int maxblocks = properties.multiProcessorCount * min(
        properties.maxThreadsPerMultiProcessor / threads,
        properties.maxBlocksPerMultiProcessor);
    int blocks = min(
        (int)((shard_size + threads - 1) / threads),
        maxblocks);

    translateShardKernel<<<blocks, threads>>>(
        thrust::raw_pointer_cast(d_translated_data.data()),
        thrust::raw_pointer_cast(d_index_data.data()),
        thrust::raw_pointer_cast(d_idx_vec_data.data()),
        neighbor_K,
        shard_size
    );

    auto deviceToHost_Start = std::chrono::high_resolution_clock::now();

    uint32_t row_size = neighbor_K + 1;
    translated_index.resize(shard_size);
    for (uint32_t i = 0; i < shard_size; ++i){
        translated_index[i].resize(row_size);
        cudaMemcpyAsync(translated_index[i].data(), thrust::raw_pointer_cast(d_translated_data.data()) + i * row_size, row_size * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    }

    auto deviceToHost_End = std::chrono::high_resolution_clock::now();
    auto deviceToHostDuration = std::chrono::duration_cast<std::chrono::milliseconds>(deviceToHost_Start - deviceToHost_End);
    printf("Device to Host Transfer time is: %lld ms\n", deviceToHostDuration.count());

    cudaEventRecord(gpu_end_events[gpu_id], 0);
    cudaDeviceSynchronize();

    omp_unset_lock(&gpu_locks[gpu_id]);


}
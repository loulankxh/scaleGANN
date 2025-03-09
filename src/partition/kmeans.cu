#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <stdint.h>
#include <cfloat>
#include <vector>
#include <iostream>
#include <cassert>
#include <limits>
#include "kmeans.cuh"


// Lan: why a center may not be the best for every data point ?????
template <typename T>
__global__ void kMeansCUDAKernel(T* data, float* centroids, uint32_t* labels, 
            float* new_centroids, float* residual_, uint32_t* count, 
            uint32_t npts, uint32_t ndim, uint32_t partition_num) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    uint32_t idx = bid * blockDim.x + tid;
    // if data num is larger than thread num
    for (uint32_t i = idx; i < npts; i += blockDim.x * gridDim.x){
        float minDist = FLT_MAX;
        uint32_t bestCentroid = 0;
        for (uint32_t j = 0; j < partition_num; ++j) {
            float dist = 0;
            for (uint32_t d = 0; d < ndim; ++d) {
                float diff = data[i * ndim + d] - centroids[j * ndim + d];
                dist += diff * diff;
            }
            if (dist < minDist) {
                minDist = dist;
                bestCentroid = j;
            }
        }
        labels[i] = bestCentroid;
        residual_[i] = minDist;
    }
    __syncthreads();


    for (uint32_t i = idx; i < npts; i += blockDim.x * gridDim.x){
        uint32_t bestCentroid = labels[i];
        atomicAdd(&count[bestCentroid], 1);
        for (uint32_t d = 0; d < ndim; ++d){
            atomicAdd(&new_centroids[bestCentroid * ndim + d], (float) data[i * ndim + d]);
        }
    }
    __syncthreads();

    if(idx < partition_num){
        for (uint32_t d = 0; d < ndim; ++d){
            centroids[idx * ndim + d] = new_centroids[idx * ndim + d] / count[idx];
            if (centroids[idx * ndim + d] == 0.0) {
                printf("count: %d; idx: %d; dimension: %d; new value: %f\n ", count[idx], idx, d, new_centroids[idx * ndim + d]);
            }
        }
    }
    // __syncthreads();

}

template <typename T>
void kMeansCUDA(uint32_t partition_num, uint32_t ndim, uint32_t max_iters, uint32_t device,
const std::vector<std::vector<T>>& sample, std::vector<std::vector<float>>& centroids){
    printf("Entering GPU for kmeans training...\n");
    uint32_t npts = sample.size();

    thrust::device_vector<T> d_data(npts * ndim);
    thrust::device_vector<float> d_centroids(partition_num * ndim);
    thrust::device_vector<uint32_t> d_labels(npts);
    thrust::device_vector<float> d_residual(npts);
    thrust::device_vector<float> d_new_centroids(partition_num * ndim);
    thrust::device_vector<uint32_t> d_count(partition_num);

    static_assert(std::is_arithmetic<T>::value, "T must be a numeric type");
    for (uint32_t i = 0; i < npts; ++i)
        cudaMemcpy(thrust::raw_pointer_cast(d_data.data()) + i * ndim, sample[i].data(), ndim * sizeof(T), cudaMemcpyHostToDevice);
    for (uint32_t i = 0; i < partition_num; ++i)
        cudaMemcpy(thrust::raw_pointer_cast(d_centroids.data()) + i * ndim, centroids[i].data(), ndim * sizeof(float), cudaMemcpyHostToDevice);
    
    thrust::fill(d_residual.begin(), d_residual.end(), 0.0);
    thrust::fill(d_count.begin(), d_count.end(), 1);
    thrust::fill(d_new_centroids.begin(), d_new_centroids.end(), 0.0f);
    
    // setting CUDA parameters
    cudaSetDevice(device);
    cudaDeviceProp properties;
    cudaGetDeviceProperties(&properties, device);
    int threads = properties.maxThreadsPerBlock;
    int maxblocks = properties.multiProcessorCount * min(
        properties.maxThreadsPerMultiProcessor / threads,
        properties.maxBlocksPerMultiProcessor);
    int blocks = min(
        (int)((npts + threads - 1) / threads),
        maxblocks);

    // using residual for early stop
    std::vector<float> residual_host(npts);
    float residual = FLT_MAX;
    float oldresidual = FLT_MAX;

    for (int iter = 0; iter < max_iters; ++iter) {
        if(iter != 0){
            assert((residual > 0) && "residual must be greater than 0 except the first iteration");
            if((((oldresidual - residual) / residual) < 0.00001) || (residual < std::numeric_limits<float>::epsilon())){
                break;
            }
        }
        printf("Kmeans iteration %d using GPU\n", iter);

        oldresidual = residual;
        residual = 0.0;

        kMeansCUDAKernel<T><<<blocks, threads>>>(thrust::raw_pointer_cast(d_data.data()), thrust::raw_pointer_cast(d_centroids.data()), 
                            thrust::raw_pointer_cast(d_labels.data()), thrust::raw_pointer_cast(d_new_centroids.data()), 
                            thrust::raw_pointer_cast(d_residual.data()), thrust::raw_pointer_cast(d_count.data()), 
                            npts, ndim, partition_num);
        cudaDeviceSynchronize();

        cudaMemcpy(residual_host.data(), thrust::raw_pointer_cast(d_residual.data()), npts * sizeof(float), cudaMemcpyDeviceToHost);
        for(uint32_t i = 0; i < npts; i++){
            residual += residual_host[i];
        }

        thrust::fill(d_residual.begin(), d_residual.end(), 0.0);
        thrust::fill(d_count.begin(), d_count.end(), 1);
        thrust::fill(d_new_centroids.begin(), d_new_centroids.end(), 0);
    }
    
    for (uint32_t i = 0; i < partition_num; ++i){
        cudaMemcpy(centroids[i].data(), thrust::raw_pointer_cast(d_centroids.data()) + i * ndim, ndim * sizeof(float), cudaMemcpyDeviceToHost);
    }
}



template void kMeansCUDA<float>(uint32_t partition_num, uint32_t ndim, uint32_t max_iters, uint32_t device,
const std::vector<std::vector<float>>& sample, std::vector<std::vector<float>>& centroids);
template void kMeansCUDA<uint32_t>(uint32_t partition_num, uint32_t ndim, uint32_t max_iters, uint32_t device,
const std::vector<std::vector<uint32_t>>& sample, std::vector<std::vector<float>>& centroids);
template void kMeansCUDA<uint8_t>(uint32_t partition_num, uint32_t ndim, uint32_t max_iters, uint32_t device,
const std::vector<std::vector<uint8_t>>& sample, std::vector<std::vector<float>>& centroids);
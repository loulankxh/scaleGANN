#include <stddef.h>
#include <stdint.h>
#include <cfloat>
#include <cmath>
#include <cassert>
#include <vector>
#include <random>
#include <iostream>
#include <limits>
#include <ctime>
#include <cstdlib>
#include <queue>
#include <algorithm>
#include <omp.h>
#include <unordered_map>
#include <chrono>

#include "kmeans.h"
#include "kmeans.cuh"
#include "../utils/distance.h"
#include "AtomicWrapper.hpp"

#define SLACK_FACTOR 1.15
#define MAX_SAMPLE 8388608 // 1 << 23
#define SAMPLE_RATE 0.05

template <typename T> // T: type of data in each dimension, like uint_8, uint 32, float, ...
size_t estimate_memory_consumption(size_t npts, uint32_t ndim, 
    uint32_t degree, uint32_t inter_degree, uint32_t threads){
    size_t memData = npts * ndim * sizeof(T);
    // 1. KNN graph (Maximum 3 times); 2. prunded reordering graph & reverse graph; 3. final merged CAGRA graph
    size_t memIndex =  4 * npts * degree * sizeof(uint32_t); // GPU can hold npts < uint32_t
    if (memData > memIndex) memIndex = memData;
    // Lan: Todo: other data like locks
    size_t memOther = 10 * npts * sizeof(uint32_t); // ptrs, 2-hop count
    size_t estimatedConsumption = memData + memIndex + memOther;
    printf("Estimated memory consumption: %zu Bytes\n", estimatedConsumption);
    return estimatedConsumption;
}


template <typename T>
void get_partition_num(uint32_t memGPU, size_t npts, uint32_t ndim, uint32_t degree, 
                    uint32_t inter_degree, uint32_t threads, uint32_t duplicate_factor,
                    uint32_t* partition_num, uint32_t* size_limt){
    size_t memConsumption = estimate_memory_consumption<T>(npts, ndim, degree, inter_degree, threads);
    if (( (size_t) memGPU * 1024 * 1024 * 1024) > SLACK_FACTOR * memConsumption) {
        *partition_num = 1;
    } else {
        *partition_num = (uint32_t)(((double) SLACK_FACTOR * duplicate_factor * memConsumption - 1) / ( (size_t) memGPU * 1024 * 1024 * 1024)) + 1; 
        *size_limt = (uint32_t)(npts * ((double) (memGPU) * 1024 * 1024 * 1024 / ((double) memConsumption * SLACK_FACTOR)));
    }
    printf("Partition number has lower bound %d\n", *partition_num);
    printf("Each partition has size limit: %d\n", *size_limt);
}



template <typename T>
void sampleData(uint32_t mem_GPU, size_t npts, uint32_t ndim, uint32_t partition_num,
    const std::vector<std::vector<T>>& data, std::vector<std::vector<T>>& sample) {
    size_t maxSampleSize = static_cast<size_t>(std::floor(((double)mem_GPU * 1024 * 1024 * 1024 - (double)partition_num * 3 * sizeof(float) - (double)partition_num * sizeof(size_t))
                        / (((double)ndim * sizeof(T) + sizeof(uint32_t)) * SLACK_FACTOR))); // (GPU - 3* centroids - count) / (each node: (data + label) * elastic_factor) 
    if (MAX_SAMPLE < maxSampleSize) maxSampleSize = MAX_SAMPLE;
    if (npts < maxSampleSize) maxSampleSize = npts;
    size_t sampleSize = (size_t) (SAMPLE_RATE * npts);
    sampleSize = std::min(sampleSize, maxSampleSize);
    
    // ramdom sampling
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, npts - 1);
    for (uint32_t i = 0; i < sampleSize; ++i) {
        sample.push_back(data[dis(gen)]);
    }

    // uniform sampling
    // uint32_t step = npts / sampleSize;
    // for (uint32_t i = 0; i < sampleSize; ++i) {
    //     sample.push_back(data[i * step]);
    // }

    printf("Sampled data size: %d\n", sample.size());
}


void print_center(std::vector<std::vector<float>>&  centroids){
    uint32_t n = centroids.size();
    uint32_t k = (n > 0) ? centroids[0].size() : 0;
    for (uint32_t i = 0; i < n; ++i) {
        printf("center genertation of %d: ",  i);
        for (uint32_t j = 0; j < k; ++j) {
            printf("%f ", centroids[i][j]);
        }
        printf("\n");
    }
}


void initialize_centroids_random(std::vector<std::vector<float>>&  centroids){
    std::random_device rd;
    std::mt19937 gen(rd());
    uint32_t n = centroids.size();
    uint32_t k = (n > 0) ? centroids[0].size() : 0;

    float min = -1;
    float max = 1;

    std::uniform_real_distribution<float> dist(min, max);
    for (uint32_t i = 0; i < n; ++i) {
        for (uint32_t j = 0; j < k; ++j) {
            centroids[i][j] = dist(gen);
        }
    }

}


template <typename T> 
void initialize_centroids_kmeansplusplus(const std::vector<std::vector<T>>& data, std::vector<std::vector<float>>&  centroids){
    static_assert(std::is_arithmetic<T>::value, "T must be a numeric type");
        
    uint32_t npts = data.size();
    uint32_t n = centroids.size();
    uint32_t k = (n > 0) ? centroids[0].size() : 0;

    if (npts <= 0){
        throw std::runtime_error("Empty dataset can not be used for genertaing centroids using kmeans plus plus");
    }

    if(k != data[0].size()){
        throw std::runtime_error("Dataset and centroid set must have the same dimension");
    }

    std::vector<float> distances_square(npts, std::numeric_limits<float>::max());

    srand(static_cast<unsigned>(time(0)));
    int firstCenterIndex = rand() % n;
    centroids[0] = std::vector<float> (data[firstCenterIndex].begin(), data[firstCenterIndex].end());

    for (uint32_t i = 1; i < n; ++i) {
        #pragma omp parallel for schedule(static)
        for (size_t j = 0; j < npts; ++j) {
            float dist = l2_distance_square_floatCentroid<T>(data[j], centroids[i-1]);
            distances_square[j] = std::min(distances_square[j], dist);
        }

        double totalDistance_square = 0;
        for (float d : distances_square) {
            totalDistance_square += d;
        }

        double randValue = static_cast<double>(rand()) / RAND_MAX * totalDistance_square;
        double cumulative = 0.0;
        uint32_t nextCenterIndex = 0;

        for (uint32_t j = 0; j < npts; ++j) {
            cumulative += distances_square[j];
            if (cumulative >= randValue) {
                // Lan: to do: add repetition check!!!
                nextCenterIndex = j;
                break;
            }
        }
        
        centroids[i] = std::vector<float> (data[nextCenterIndex].begin(), data[nextCenterIndex].end());
    }

}



template <typename T>
void call_kmeans(uint32_t partition_num, uint32_t ndim, uint32_t max_iters, 
            const std::vector<std::vector<T>>& sample, std::vector<std::vector<float>>& centroids){
    // Use GPU if available, otherwise CPU
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount); // Lan: To do: if GPU is available
    if (deviceCount > 0) {
        std::cout << "Using GPU for K-means\n";
        int deviceFirstAvail = 0; // Lan: To do: get first available, or the most suitable
        kMeansCUDA<T>(partition_num, ndim, max_iters, deviceFirstAvail, sample, centroids);
    } else {
        std::cout << "Using CPU for K-means\n";
        kMeansCPU<T>(partition_num, ndim, max_iters, sample, centroids);
    }
}



template <typename T>
void assign_partitions_limit_prune(uint32_t partition_num, uint32_t ndim, uint32_t duplicate_factor, uint32_t size_limit,
            const std::vector<std::vector<float>>& centroids, const std::vector<std::vector<T>>& data,
            std::vector<std::vector<std::vector<T>>>& partitions,
            std::vector<std::vector<uint32_t>>& idx_map){
    
    assert((duplicate_factor <= partition_num) && "Duplication time must be no greater than the partition number");

    uint32_t thread_count = omp_get_max_threads();

    std::vector<AtomicWrapper<uint32_t>> partition_sizes(partition_num);
    uint32_t estimated_size_bound = static_cast<uint32_t>(1.2 * size_limit);
    for(uint32_t iter = 0; iter < partition_num; iter++){
        idx_map[iter].resize(estimated_size_bound);
        partitions[iter].resize(estimated_size_bound);
    }


    printf("Starting calculating distance & assigning\n");
    #pragma omp parallel for schedule(static) shared(data, centroids)
    for (uint32_t i = 0; i < data.size(); i++) {
        const auto& point = data[i];
        std::vector<std::pair<float, uint32_t>> distances;
        for (uint32_t j = 0; j < partition_num; j++) {
            float distance = 0;
            for (uint32_t d = 0; d < ndim; ++d) {
                float diff = point[d] - centroids[j][d];
                distance += diff * diff;
            }
            distances.emplace_back(distance, j);
        }
        std::sort(distances.begin(), distances.end());
        uint32_t assigned_count = 0;
        for (uint32_t j = 0; j < partition_num; j++) {
            const auto& distance = distances[j];
            const auto& id = distance.second;
            if (assigned_count >= duplicate_factor) {
                break;
            }

            // // Lan: To do, still can not reach the exact equivalence
            uint32_t partition_size_id = partition_sizes[id].load(); // due to concurrency, wrongly fail the condition (partition_size_id >= size_limit) for at most 128 times (thread number)
            if ((partition_size_id >= size_limit) && ((partition_num - j) > (duplicate_factor - assigned_count))) {
                continue;
            }
            uint32_t current_id = partition_sizes[id]++;
            // std::vector<uint32_t> idx_pair = {current_id, i};
            idx_map[id][current_id] = i;
            partitions[id][current_id] = point;
            assigned_count++;
        }
    }

    for(uint32_t iter = 0; iter < partition_num; iter++){
        idx_map[iter].resize(partition_sizes[iter].load());
        partitions[iter].resize(partition_sizes[iter].load());
    }

}



// Lan: try to limit the duplication factor to 2, and reduce the time spent
template <typename T>
void assign_partitions_limit_size(uint32_t partition_num, uint32_t ndim, uint32_t duplicate_factor, uint32_t size_limit,
            const std::vector<std::vector<float>>& centroids, const std::vector<std::vector<T>>& data,
            std::vector<std::vector<std::vector<T>>>& partitions,
            std::vector<std::vector<uint32_t>>& idx_map){
    
    assert((duplicate_factor <= partition_num) && "Duplication time must be no greater than the partition number");

    uint32_t thread_count = omp_get_max_threads();

    std::vector<AtomicWrapper<uint32_t>> partition_sizes(partition_num);
    uint32_t estimated_size_bound = static_cast<uint32_t>(1.2 * size_limit);
    for(uint32_t iter = 0; iter < partition_num; iter++){
        idx_map[iter].resize(estimated_size_bound);
        partitions[iter].resize(estimated_size_bound);
    }


    printf("Starting calculating distance & assigning\n");
    #pragma omp parallel for schedule(static) shared(data, centroids)
    for (uint32_t i = 0; i < data.size(); i++) {
        const auto& point = data[i];
        std::vector<std::pair<float, uint32_t>> distances;
        for (uint32_t j = 0; j < partition_num; j++) {
            float distance = 0;
            for (uint32_t d = 0; d < ndim; ++d) {
                float diff = point[d] - centroids[j][d];
                distance += diff * diff;
            }
            distances.emplace_back(distance, j);
        }
        std::sort(distances.begin(), distances.end());
        uint32_t assigned_count = 0;
        for (uint32_t j = 0; j < partition_num; j++) {
            const auto& distance = distances[j];
            const auto& id = distance.second;
            if (assigned_count >= duplicate_factor) {
                break;
            }

            // // Lan: To do, still can not reach the exact equivalence
            uint32_t partition_size_id = partition_sizes[id].load(); // due to concurrency, wrongly fail the condition (partition_size_id >= size_limit) for at most 128 times (thread number)
            if ((partition_size_id >= size_limit) && ((partition_num - j) > (duplicate_factor - assigned_count))) {
                continue;
            }
            uint32_t current_id = partition_sizes[id]++;
            // std::vector<uint32_t> idx_pair = {current_id, i};
            idx_map[id][current_id] = i;
            partitions[id][current_id] = point;
            assigned_count++;
        }
    }

    for(uint32_t iter = 0; iter < partition_num; iter++){
        idx_map[iter].resize(partition_sizes[iter].load());
        partitions[iter].resize(partition_sizes[iter].load());
    }

}



template <typename T>
void assign_partitions(uint32_t partition_num, uint32_t ndim, uint32_t duplicate_factor,
            const std::vector<std::vector<float>>& centroids, const std::vector<std::vector<T>>& data,
            std::vector<std::vector<std::vector<T>>>& partitions,
            std::vector<std::vector<uint32_t>>& idx_map){
    
    assert((duplicate_factor <= partition_num) && "Duplication time must be no greater than the partition number");

    uint32_t thread_count = omp_get_max_threads();
    omp_lock_t* locks = new omp_lock_t[partition_num];
    for (int i = 0; i < partition_num; i++) {
        omp_init_lock(&locks[i]);
    }
    

    std::vector<std::vector<std::unordered_map<uint32_t, std::vector<T>>>> thread_local_data(thread_count, std::vector<std::unordered_map<uint32_t, std::vector<T>>>(partition_num));
    std::vector<uint32_t> partition_sizes(partition_num, 0);

    printf("Starting calculating distance\n");
    #pragma omp parallel for schedule(static) shared(data, centroids)
    for (uint32_t i = 0; i < data.size(); i++) {
        const auto& point = data[i];
        std::vector<std::pair<float, uint32_t>> distances;
        for (uint32_t j = 0; j < partition_num; j++) {
            float distance = 0;
            for (uint32_t d = 0; d < ndim; ++d) {
                float diff = point[d] - centroids[j][d];
                distance += diff * diff;
            }
            distances.emplace_back(distance, j);
        }
        std::sort(distances.begin(), distances.end());
        uint32_t assigned_count = 0;
        for (uint32_t j = 0; j < partition_num; j++) {
            const auto& distance = distances[j];
            const auto& id = distance.second;
            if (assigned_count >= duplicate_factor) {
                break;
            }

            auto& thread_data = thread_local_data[omp_get_thread_num()];
            thread_data[id][i] = point;
            // printf("pushing back data %d to partition %d\n", i, j);
            assigned_count++;
        }
    }

    printf("Starting assigning centers\n");
    #pragma omp parallel for schedule(static)
    for (uint32_t i = 0; i < partition_num; i++){
        for(uint32_t thread_id = 0; thread_id < thread_count; thread_id++){
            auto& thread_data_i = thread_local_data[thread_id][i];
            for (const auto& [global_idx, point] : thread_data_i){
                // std::vector<uint32_t> idx_pair = {static_cast<uint32_t>(partitions[i].size()), glocal_idx};
                idx_map[i].push_back(global_idx);
                partitions[i].push_back(point);
            }
        }
    }
    printf("Finishing assigning centers\n");
    
    for (int i = 0; i < partition_num; i++) {
        omp_destroy_lock(&locks[i]);
    }
    delete[] locks;
}



template <typename T> 
void get_partitions(uint32_t memGPU, size_t npts, uint32_t ndim, uint32_t max_iters, 
                    uint32_t duplicate_factor, uint32_t partition_num, uint32_t size_limit,
                    const std::vector<std::vector<T>>& data,
                    std::vector<std::vector<std::vector<T>>>& partitions,
                    std::vector<std::vector<uint32_t>>& idx_map){

    auto startTime = std::chrono::high_resolution_clock::now();

    assert((partition_num > 1) && "Partition number must be greater than 1 to generate multiple shards");
    
    std::vector<std::vector<T>> sample;
    sampleData<T>(memGPU, npts, ndim, partition_num, data, sample);

    auto sampleTime = std::chrono::high_resolution_clock::now();
    auto sampleDuration = std::chrono::duration_cast<std::chrono::milliseconds>(sampleTime - startTime);

    std::vector<std::vector<float>> centroids(partition_num, std::vector<float>(ndim));
    initialize_centroids_kmeansplusplus(sample, centroids);
    // print_center(centroids);

    auto centroidTime = std::chrono::high_resolution_clock::now();
    auto centroidDuration = std::chrono::duration_cast<std::chrono::milliseconds>(centroidTime - sampleTime);

    call_kmeans<T>(partition_num, ndim, max_iters, sample, centroids);
    // print_center(centroids);

    auto kmeansTime = std::chrono::high_resolution_clock::now();
    auto kmeansDuration = std::chrono::duration_cast<std::chrono::milliseconds>(kmeansTime - centroidTime);

    printf("Staring assigning partitions\n");
    idx_map.resize(partition_num);
    // uint32_t size_limit = (uint32_t) (1 + duplicate_factor * npts / partition_lower_bound);
    printf("Expected size limit is: %d\n", size_limit);
    assign_partitions_limit_size<T>(partition_num, ndim, duplicate_factor, size_limit, centroids, data, partitions, idx_map);
    // assign_partitions<T>(partition_num, ndim, duplicate_factor, size_limit, centroids, data, partitions, idx_map);
    

    auto assignTime = std::chrono::high_resolution_clock::now();
    auto assignDuration = std::chrono::duration_cast<std::chrono::milliseconds>(assignTime - kmeansTime);

    printf("Partition number: %d\n", partitions.size());
    for(int i = 0 ; i < partitions.size(); i++){
        printf("Partition %d has %d nodes\n", i, partitions[i].size());
    }

    printf("Sample time: %lld ms, centroid time: %lld ms, GPU kmeans time: %lld ms, CPU assign time: %lld ms\n", 
            sampleDuration.count(), centroidDuration.count(), kmeansDuration.count(), assignDuration.count());

}

template <typename T> std::vector<std::vector<std::vector<T>>> 
main_partitions(uint32_t memGPU, size_t npts, uint32_t ndim, uint32_t degree, 
                    uint32_t inter_degree, uint32_t threads,
                    uint32_t max_iters,
                    uint32_t duplicate_factor, uint32_t partition_num, 
                    const std::vector<std::vector<T>>& data, 
                    std::vector<std::vector<uint32_t>>& idx_map){
    
    uint32_t partition_lower_bound = 0;
    uint32_t size_limit = (uint32_t) npts;
    get_partition_num<T>(memGPU, npts, ndim, degree, inter_degree, threads, duplicate_factor, &partition_lower_bound, &size_limit);
    if (partition_lower_bound > partition_num) {
        printf("Needing %d partitions at least, change given partition number %d to %d\n", partition_lower_bound, partition_num, partition_lower_bound);
        partition_num = partition_lower_bound;
    }
    printf("Partition number is %d\n", partition_num);

    if (partition_num > 1){
        printf("Data exceeds GPU memory limit... Prepare for partitioning\n");
        
        std::vector<std::vector<std::vector<T>>> partitions(partition_num);
        get_partitions(memGPU, npts, ndim, max_iters, duplicate_factor, partition_num, size_limit, data, partitions, idx_map);

        return partitions;
    }else{
        // Lan: To do
        printf("Data can fit into one GPU memory\n");
        std::vector<std::vector<std::vector<T>>> partitions(0);
        partitions.push_back(data);

        printf("Partition number: %d\n", partitions.size());
        for(int i = 0 ; i < partitions.size(); i++){
            printf("Partition %d has %d nodes\n", i, partitions[i].size());
        }
        return partitions;
    }

}







template void get_partition_num<float>(uint32_t memGPU, size_t npts, uint32_t ndim, uint32_t degree, uint32_t inter_degree, uint32_t threads, uint32_t dumplicate_factor, uint32_t* partition_num, uint32_t* size_limt);
template void get_partition_num<uint32_t>(uint32_t memGPU, size_t npts, uint32_t ndim, uint32_t degree, uint32_t inter_degree, uint32_t threads, uint32_t dumplicate_factor, uint32_t* partition_num, uint32_t* size_limt);
template void get_partition_num<uint8_t>(uint32_t memGPU, size_t npts, uint32_t ndim, uint32_t degree, uint32_t inter_degree, uint32_t threads, uint32_t dumplicate_factor, uint32_t* partition_num, uint32_t* size_limt);



template void get_partitions<float>(uint32_t memGPU, size_t npts, uint32_t ndim, uint32_t max_iters, 
                    uint32_t duplicate_factor, uint32_t partition_num, uint32_t size_limit,
                    const std::vector<std::vector<float>>& data,
                    std::vector<std::vector<std::vector<float>>>& partitions,
                    std::vector<std::vector<uint32_t>>& idx_map);
template void get_partitions<uint32_t>(uint32_t memGPU, size_t npts, uint32_t ndim, uint32_t max_iters, 
                    uint32_t duplicate_factor, uint32_t partition_num, uint32_t size_limit,
                    const std::vector<std::vector<uint32_t>>& data,
                    std::vector<std::vector<std::vector<uint32_t>>>& partitions,
                    std::vector<std::vector<uint32_t>>& idx_map);
template void get_partitions<uint8_t>(uint32_t memGPU, size_t npts, uint32_t ndim, uint32_t max_iters, 
                    uint32_t duplicate_factor, uint32_t partition_num, uint32_t size_limit,
                    const std::vector<std::vector<uint8_t>>& data,
                    std::vector<std::vector<std::vector<uint8_t>>>& partitions,
                    std::vector<std::vector<uint32_t>>& idx_map);


template std::vector<std::vector<std::vector<float>>> main_partitions<float>(uint32_t memGPU, 
            size_t npts, uint32_t ndim, uint32_t degree, uint32_t inter_degree, uint32_t threads,
            uint32_t max_iters, uint32_t duplicate_factor, uint32_t partition_num,
            const std::vector<std::vector<float>>& data,
            std::vector<std::vector<uint32_t>>& idx_map);
template std::vector<std::vector<std::vector<uint32_t>>> main_partitions<uint32_t>(uint32_t memGPU, 
            size_t npts, uint32_t ndim, uint32_t degree, uint32_t inter_degree, uint32_t threads,
            uint32_t max_iters, uint32_t duplicate_factor, uint32_t partition_num,
            const std::vector<std::vector<uint32_t>>& data,
            std::vector<std::vector<uint32_t>>& idx_map);
template std::vector<std::vector<std::vector<uint8_t>>> main_partitions<uint8_t>(uint32_t memGPU, 
            size_t npts, uint32_t ndim, uint32_t degree, uint32_t inter_degree, uint32_t threads,
            uint32_t max_iters, uint32_t duplicate_factor, uint32_t partition_num,
            const std::vector<std::vector<uint8_t>>& data,
            std::vector<std::vector<uint32_t>>& idx_map);

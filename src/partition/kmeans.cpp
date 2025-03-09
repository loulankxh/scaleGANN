#include <stdint.h>
#include <vector>
#include <cfloat>
#include <iostream>
#include <cfloat>
#include <cassert>
#include <limits>
#include "kmeans.h"


template <typename T>
void kMeansCPU(uint32_t partition_num, uint32_t ndim, uint32_t max_iters, 
const std::vector<std::vector<T>>& sample, std::vector<std::vector<float>>& centroids) {
    printf("Entering CPU for kmeans training...\n");
    std::size_t npts = sample.size();
    std::vector<uint32_t> labels(npts, 0);
    
    float oldresidual = FLT_MAX;
    float residual = FLT_MAX;

    for (uint32_t iter = 0; iter < max_iters; ++iter) {
        if(iter != 0){
            assert((residual > 0) && "residual must be greater than 0 except the first iteration");
            if((((oldresidual - residual) / residual) < 0.00001) || (residual < std::numeric_limits<float>::epsilon())){
                break;
            }
        }
        printf("Kmeans iteration %d using CPU\n", iter);

        // Assign points to nearest centroid
        oldresidual = residual;
        residual = 0.0;
        for (std::size_t i = 0; i < npts; ++i) {
            float minDist = FLT_MAX;
            uint32_t bestCentroid = 0;
            for (uint32_t j = 0; j < partition_num; ++j) {
                float dist = 0;
                for (uint32_t d = 0; d < ndim; ++d) {
                    float diff = sample[i][d] - centroids[j][d];
                    dist += diff * diff;
                }
                if (dist < minDist) {
                    minDist = dist;
                    bestCentroid = j;
                }
            }
            residual += minDist;
            labels[i] = bestCentroid;
        }

        // Update centroids
        std::vector<std::vector<float>> newCentroids(partition_num, std::vector<float>(ndim, 0.0f));
        std::vector<std::size_t> counts(partition_num, 0);
        for (std::size_t i = 0; i < npts; ++i) {
            uint32_t label = labels[i];
            counts[label]++;
            for (uint32_t d = 0; d < ndim; ++d) {
                newCentroids[label][d] += sample[i][d]; // Lan: To do: manager out of flow
            }
        }
        for (uint32_t j = 0; j < partition_num; ++j) {
            for (uint32_t d = 0; d < ndim; ++d) {
                if (counts[j] > 0) {
                    newCentroids[j][d] /= counts[j]; // Lan: To do: manager float type
                }
            }
            centroids[j] = newCentroids[j];
        }
    }
}

template void kMeansCPU<float>(uint32_t partition_num, uint32_t ndim, uint32_t max_iters, 
const std::vector<std::vector<float>>& sample, std::vector<std::vector<float>>& centroids);
template void kMeansCPU<uint32_t>(uint32_t partition_num, uint32_t ndim, uint32_t max_iters, 
const std::vector<std::vector<uint32_t>>& sample, std::vector<std::vector<float>>& centroids);
template void kMeansCPU<uint8_t>(uint32_t partition_num, uint32_t ndim, uint32_t max_iters, 
const std::vector<std::vector<uint8_t>>& sample, std::vector<std::vector<float>>& centroids);
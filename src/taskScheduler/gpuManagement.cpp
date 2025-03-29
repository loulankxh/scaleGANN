#include "gpuManagement.h"


std::vector<omp_lock_t> gpu_locks(GPU_NUM);


std::vector<cudaEvent_t> gpu_start_events(GPU_NUM);
std::vector<cudaEvent_t> gpu_end_events(GPU_NUM);


void initGpuLocks() {
    for (int i = 0; i < GPU_NUM; i++) {
        omp_init_lock(&gpu_locks[i]);
    }
}

void destroyGpuLocks() {
    for (int i = 0; i < GPU_NUM; i++) {
        omp_destroy_lock(&gpu_locks[i]);
    }
}


void initEvents() {
    for (int gpu_id = 0; gpu_id < GPU_NUM; gpu_id++) {
        cudaSetDevice(gpu_id);
        cudaEventCreate(&gpu_start_events[gpu_id]);
        cudaEventCreate(&gpu_end_events[gpu_id]);
    }
}

void destroyEvents() {
    for (int gpu_id = 0; gpu_id < GPU_NUM; gpu_id++) {
        cudaSetDevice(gpu_id);
        cudaEventDestroy(gpu_start_events[gpu_id]);
        cudaEventDestroy(gpu_end_events[gpu_id]);
    }
}

bool isGPUBusy(int gpu_id) {
    cudaSetDevice(gpu_id);
    cudaError_t status_start = cudaEventQuery(gpu_start_events[gpu_id]);
    cudaError_t status_end = cudaEventQuery(gpu_end_events[gpu_id]);
    return ((status_start == cudaErrorNotReady)||(status_end == cudaErrorNotReady));
}
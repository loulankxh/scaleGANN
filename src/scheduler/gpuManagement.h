
#include <cuda_runtime.h>
#include <vector>
#include <omp.h>

#define GPU_NUM 4



extern std::vector<omp_lock_t> gpu_locks;


extern std::vector<cudaEvent_t> gpu_start_events;
extern std::vector<cudaEvent_t> gpu_end_events;


void initGpuLocks();

void destroyGpuLocks(); 

void initEvents(); 

void destroyEvents(); 

bool isGPUBusy(int gpu_id);
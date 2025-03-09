#!/bin/bash

N=10
NUM_GPUS=4
DatasetPath="/home/lanlu/raft/scaleGANN/dataset/sift100M"
DATASET="sift100M"
SHARD_FOLDER="test"
declare -A GPU_PIDS

RAFT_CAGRA="/home/lanlu/miniconda3/envs/rapids_raft/bin/ann/RAFT_CAGRA_ANN_BENCH"

start_time=$(date +%s)

for ((i=0; i<N; i++)); do
    # GPU_ID=$((i % NUM_GPUS))
    GPU_ID=0
    

    TASK_FILE="${DatasetPath}/${SHARD_FOLDER}/${DATASET}_${i}.json"
    OUTPUT_FILE="${DatasetPath}/${SHARD_FOLDER}/${DATASET}_${i}.json.lock"

    if [[ -n "${GPU_PIDS[$GPU_ID]}" ]]; then
        echo "Wait for GPU $GPU_ID finishing task ID ${GPU_PIDS[$GPU_ID]} ..."
        wait ${GPU_PIDS[$GPU_ID]}
    fi

    CUDA_VISIBLE_DEVICES=$GPU_ID $RAFT_CAGRA \
        --build --force \
        --data_prefix=$DatasetPath \
        --benchmark_out_format=json \
        --benchmark_counters_tabular=true \
        --benchmark_out="$OUTPUT_FILE" \
        --raft_log_level=3 \
        "$TASK_FILE" &
    
    GPU_PIDS[$GPU_ID]=$!

    echo "Task $i (GPU $GPU_ID) is started, with PID=${GPU_PIDS[$GPU_ID]}"
done

wait

end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
echo "All tasks are done! Total execution time: ${elapsed_time} s."
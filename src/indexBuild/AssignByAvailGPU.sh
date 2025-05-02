#!/bin/bash

declare -A GPU_PIDS
NUM_GPUS=4
DatasetPath="/home/lanlu/scaleGANN/dataset/laion100M"
DATASET="laion100M"
SHARD_FOLDER="/ScaleGANN/Epsilon1.2"
N=100

RAFT_CAGRA="/home/lanlu/miniconda3/envs/rapids_raft/bin/ann/RAFT_CAGRA_ANN_BENCH"
LOG_FILE=${DatasetPath}/${SHARD_FOLDER}/time.txt

start_time=$(date +%s)

i=0
while (( i < N )); do
    isAvailGPU=-1
    for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
        pid=${GPU_PIDS[$gpu]}
        if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
            isAvailGPU=$gpu
        fi
    done
    if (( isAvailGPU == -1)); then
        wait -n
    fi

    for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
        pid=${GPU_PIDS[$gpu]}
        if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
            TASK_FILE="${DatasetPath}/${SHARD_FOLDER}/partition$i/${DATASET}.json"
            OUTPUT_FILE="${DatasetPath}/${SHARD_FOLDER}/partition$i/${DATASET}.json.lock"

            echo "GPU $gpu is available. Launching task $i ..."

            CUDA_VISIBLE_DEVICES=$gpu $RAFT_CAGRA \
                --build --force \
                --data_prefix=$DatasetPath \
                --benchmark_out_format=json \
                --benchmark_counters_tabular=true \
                --benchmark_out="$OUTPUT_FILE" \
                --raft_log_level=3 \
                "$TASK_FILE" &

            GPU_PIDS[$gpu]=$!
            echo "Task $i running on GPU $gpu with PID ${GPU_PIDS[$gpu]}"
            ((i++))
            break
        fi
    done
done

wait

end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
echo "All tasks are done! Total execution time: ${elapsed_time} s." | tee -a "$LOG_FILE"
DatasetPath="/home/lanlu/raft/scaleGANN/dataset/laion1M"
DATASET="laion1M"

RAFT_CAGRA="/home/lanlu/miniconda3/envs/rapids_raft/bin/ann/RAFT_CAGRA_ANN_BENCH"
TASK_FILE="${DatasetPath}/${DATASET}.json"
OUTPUT_FILE="${DatasetPath}/${DATASET}.json.lock"

$RAFT_CAGRA \
    --build --force \
    --data_prefix=$DatasetPath \
    --benchmark_out_format=json \
    --benchmark_counters_tabular=true \
    --benchmark_out="$OUTPUT_FILE" \
    --raft_log_level=3 \
    "$TASK_FILE" &
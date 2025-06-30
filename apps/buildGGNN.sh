Executable="/home/lanlu/ggnn/build_docker/sift1b_multi"
BaseDir="/home/lanlu/scaleGANN/dataset/sift1B/GGNN"
dataPostfix="bvecs"
SizeUnit=1000000
shardSizeUnit=10
ShardSize=$shardSizeUnit*$SizeUnit
ShardNum=100

LOG_FILE=${BaseDir}/time.txt

start_time=$(date +%s)

for ((i=0; i<ShardNum; i++)); do
    # mkdir -p ${BaseDir}/partition$i
    datapath=${BaseDir}/partition$i/data.${dataPostfix}
    querypath="/home/lanlu/scaleGANN/dataset/sift1B/query.${dataPostfix}"
    graph_dir=${BaseDir}/partition$i/
    # datapath="/home/lanlu/scaleGANN/dataset/sift100M/base.100M.bvecs"
    # querypath="/home/lanlu/scaleGANN/dataset/sift100M/query.bvecs"
    # graph_dir=${BaseDir}/

    task_start_time=$(date +%s)

    $Executable \
        --mode="bs" \
        --base_filename=$datapath \
        --query_filename=$querypath \
        --graph_dir=$graph_dir \
        --base=$shardSizeUnit \
        --shard=$shardSizeUnit 
        # --factor=$SizeUnit 
    
    task_end_time=$(date +%s)
    task_duration=$((task_end_time - task_start_time))

    echo "Task $i finished in ${task_duration} seconds" | tee -a "$LOG_FILE"
done

end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
echo "All tasks are done! Total execution time: ${elapsed_time} s."
# echo "Dataset size: $((ShardNum * shardSizeUnit))M; Shard size: ${shardSizeUnit}M."
echo "Dataset size: 1B; Shard size: ${shardSizeUnit}M."

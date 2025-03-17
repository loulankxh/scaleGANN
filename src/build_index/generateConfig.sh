DataPath="/home/lanlu/scaleGANN/dataset/"
DATASET="SimSearchNet100M"
SHARD_FOLDER="D64_N40"
DIM=256
SHARD_NUM=10

QueryFile="/home/lanlu/scaleGANN/dataset/simSearchNet100M/FB_ssnpp_public_queries.u8bin"
GTFile="/home/lanlu/scaleGANN/dataset/simSearchNet100M/groundtruth.neighbors.ibin"

BUILD_DEG=(32 64)
BUILD_INTERMEDIATE_DEG=(32 64)

for ((i=0; i<SHARD_NUM; i++)); do
    partition_path="$DataPath/$DATASET/$SHARD_FOLDER/partition$i"
    
    # 如果文件夹不存在，则创建
    if [[ ! -d "$partition_path" ]]; then
        mkdir -p "$partition_path/index"
        echo "Created directory: $partition_path"
    fi

    # 2. 生成 JSON 配置文件
    json_file="$partition_path/config.json"
    echo "Writing JSON file: $json_file"

    cat <<EOF > "$json_file"
{
    "dataset": {
        "name": "$DATASET",
        "base_file": "$partition_path/data.u8bin",
        "dims": $DIM,
        "query_file": "$QueryFile",
        "groundtruth_neighbors_file": "$GTFile",
        "distance": "euclidean"
    },
    "search_basic_param": {
        "k": 10,
        "batch_size": 10
    },
    "index": [
EOF

    # 遍历 intermediate_graph_degree
    first_entry=true  # 用于处理 JSON 逗号
    for intermediate_graph_degree in "${BUILD_INTERMEDIATE_DEG[@]}"; do
        for graph_degree in "${BUILD_DEG[@]}"; do
            # 确保 graph_degree <= intermediate_graph_degree
            if [[ "$graph_degree" -le "$intermediate_graph_degree" ]]; then
                index_file="$partition_path/index/raft_cagra.graph_degree${graph_degree}.intermediate_graph_degree${intermediate_graph_degree}.graph_build_algoNN_DESCENT"

                # 处理 JSON 逗号
                if [[ "$first_entry" == false ]]; then
                    echo "," >> "$json_file"
                fi
                first_entry=false

                cat <<EOF >> "$json_file"
        {
            "algo": "raft_cagra",
            "build_param": {
                "graph_degree": $graph_degree,
                "intermediate_graph_degree": $intermediate_graph_degree,
                "graph_build_algo": "NN_DESCENT"
            },
            "name": "raft_cagra.graph_degree${graph_degree}.intermediate_graph_degree${intermediate_graph_degree}.graph_build_algoNN_DESCENT",
            "file": "$index_file",
            "search_params": []
        }
EOF
            fi
        done
    done

    # 结束 JSON
    echo  "
    ]
}" >> "$json_file"

    echo "Generated JSON: $json_file"
done

echo "All partitions processed successfully!"
#!/bin/bash
DataPath="/home/lanlu/scaleGANN/dataset/"
DATASET="sift1B"
SHARD_FOLDER="/NaiveCAGRA"
DIM=128
SHARD_NUM=100
QueryFile="/home/lanlu/scaleGANN/dataset/sift1B/query.public.10K.u8bin"
GTFile="/home/lanlu/scaleGANN/dataset/sift1B/groundtruth.neighbors.ibin"

BUILD_DEG=(64)
BUILD_INTERMEDIATE_DEG=(128)

for ((i=0; i<SHARD_NUM; i++)); do
    partition_path="$DataPath/$DATASET/$SHARD_FOLDER/partition$i"
    
    if [[ ! -d "$partition_path" ]]; then
        echo "Parition$i doesn't exist"
        continue
    fi

    json_file="$partition_path/$DATASET.json"
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


    first_entry=true  
    for intermediate_graph_degree in "${BUILD_INTERMEDIATE_DEG[@]}"; do
        for graph_degree in "${BUILD_DEG[@]}"; do
            if [[ "$graph_degree" -le "$intermediate_graph_degree" ]]; then
                index_file_name="raft_cagra.graph_degree${graph_degree}.intermediate_graph_degree${intermediate_graph_degree}.graph_build_algoNN_DESCENT"
                index_file="$partition_path/index/$index_file_name"

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
            "name": "$index_file_name",
            "file": "$index_file",
            "search_params": []
        }
EOF
            fi
        done
    done

    echo  "
    ]
}" >> "$json_file"

    echo "Generated JSON: $json_file"
done

echo "All partitions processed successfully!"
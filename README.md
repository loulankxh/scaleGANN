ScaleGANN is a highly efficient, scalable, and cost-effective system for ANNS indexing and querying. It is extendable to cloud-native resources or serverless usages. This Readme file gives instruction on how to build and test ScaleGANN.

# Requirement

# Build
## Build Benchmarks
1. Build CAGRA
2. Build DiskANN
3. Build GGNN
## Build ScaleGANN
```
git clone --recurse-submodules https://github.com/loulankxh/scaleGANN.git
mkdir -p build && cd build
cmake .. && make -j
```

# Run 
## Run ScaleGANN
1. Compile the ScaleGANN to get the build folder and files
2. Run index construction (the time of each step will show in the last line of terminal output)
   a. Partition: build/executeDiskPartition
     ```
     E.g., ./executeDiskPartition –data_path “path_to_your_data” –base_folder “path_to_store_data_partitions” –M 16 -R 64 -L 128 -W 2 -E 1.2 -T 80
     ```
   b. Build:
      1. Run [src/build/generateConfig.sh] to generate build configuration
         Line 1-8: change to your local setting
         Line 10-11: change to your build degree, use build_deg=64, build_intermediate_deg=128
      2. Run [src/build/SequentialAssign.sh] for building using one GPU, or [src/build/AssignByAvailGPU.sh] for multi-GPU parallelism
         Change the “RAFT_CAGRA“ path to your path_to_cagra
         Change Line 3-11 to your local settings
   c. Merge: build/executeDiskMerge
   ```
    E.g., ./executeDiskMerge –base_folder “path_where_data_partitions_are_stored” –index_name “name_of_your_shard_index” -R 64 -B 64 -T 80
    Search: build/searchScaleGANN
   ```


# Experiment Results
In the paper, we compared ScaleGANN with three other benchmarks: [DiskANN](https://github.com/microsoft/DiskANN/tree/main), [GGNN](https://github.com/cgtuebingen/ggnn/tree/release_0.5), and Naive CAGRA which we implemented based on [CAGRA](https://github.com/rapidsai/raft/tree/branch-24.10).
## Build experiments
We built indexes using all the above four approaches on various datasets: Sift, Deep, SimSearchNet, Microsoft Turing, Laion. Most of the datasets are from [Bigann Benchamrks](https://big-ann-benchmarks.com/neurips21.html). While [Laion](https://laion.ai) is download at [Laion Download](https://the-eye.eu/public/AI/cah/laion5b/embeddings/laion1B-nolang/).
Build data with details are in folder [experiments]().
## Search experiments
We record full search results including recall, time and query latency in [search results](https://docs.google.com/spreadsheets/d/1_rdrr2zPHzPDIhlvdY1N7BzFtH-y-3tpIPgv03B6Tw8/edit?usp=sharing). 
For more details, also see in folder [experiments]().

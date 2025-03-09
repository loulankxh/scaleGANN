import numpy as np

def u8bin_to_bvecs(input_file, output_file):
    with open(input_file, "rb") as f:
        num_vectors = np.fromfile(f, dtype=np.uint32, count=1)[0]
        dim = np.fromfile(f, dtype=np.uint32, count=1)[0]
        
        vectors = np.fromfile(f, dtype=np.uint8).reshape(num_vectors, dim)
    
    with open(output_file, "wb") as f:
        for vec in vectors:
            np.array([dim], dtype=np.uint32).tofile(f)
            vec.tofile(f)

def ibin_to_ivecs(input_file, output_file):
    with open(input_file, "rb") as f:
        num_vectors = np.fromfile(f, dtype=np.uint32, count=1)[0]
        dim = np.fromfile(f, dtype=np.uint32, count=1)[0]
        
        vectors = np.fromfile(f, dtype=np.int32).reshape(num_vectors, dim)
    
    with open(output_file, "wb") as f:
        for vec in vectors:
            np.array([dim], dtype=np.uint32).tofile(f)
            vec.tofile(f)

def fbin_to_fvecs(input_file, output_file):
    with open(input_file, "rb") as f:
        num_vectors = np.fromfile(f, dtype=np.uint32, count=1)[0]
        dim = np.fromfile(f, dtype=np.uint32, count=1)[0]
        
        vectors = np.fromfile(f, dtype=np.float32).reshape(num_vectors, dim)
    
    with open(output_file, "wb") as f:
        for vec in vectors:
            np.array([dim], dtype=np.uint32).tofile(f)
            vec.tofile(f)

if __name__ == "__main__":
    input_base = "/home/lanlu/raft/scaleGANN/dataset/simSearchNet1M/base.1M.u8bin"
    output_base = "/home/lanlu/ggnn/data/simSearchNet1M/base.bvecs"
    u8bin_to_bvecs(input_base, output_base)

    input_query = "/home/lanlu/raft/scaleGANN/dataset/simSearchNet1M/FB_ssnpp_public_queries.u8bin"
    output_query = "/home/lanlu/ggnn/data/simSearchNet1M/query.bvecs"
    u8bin_to_bvecs(input_query, output_query)

    input_groundtruth = "/home/lanlu/raft/scaleGANN/dataset/simSearchNet1M/groundtruth.neighbors.ibin"
    output_groundtruth = "/home/lanlu/ggnn/data/simSearchNet1M/groundtruth.ivecs"
    ibin_to_ivecs(input_groundtruth, output_groundtruth)


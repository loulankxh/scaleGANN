import numpy as np
import os

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


def fbin_to_fvecs_10GB_chunked(input_file, output_file, chunk_bytes=10 * 1024 * 1024 * 1024):  # 10GB
    # 检查输出文件路径是否存在
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        print(f"Output directory '{output_dir}' does not exist. Creating it...")
        os.makedirs(output_dir, exist_ok=True)

    with open(input_file, "rb") as fin, open(output_file, "wb") as fout:
        # 读取头部
        num_vectors = np.fromfile(fin, dtype=np.uint32, count=1)[0]
        dim = np.fromfile(fin, dtype=np.uint32, count=1)[0]
        chunk_bytes = int(chunk_bytes)
        vec_byte_size = int(dim) * 4  # float32 = 4 bytes
        chunk_size = max(1, chunk_bytes // vec_byte_size)

        print(f"Start converting: {num_vectors} vectors of dim {dim}")
        print(f"Processing {chunk_size} vectors per chunk (~{chunk_size * vec_byte_size / 1024**3:.2f} GB)")

        total_read = 0
        while total_read < num_vectors:
            remaining = num_vectors - total_read
            read_now = min(chunk_size, remaining)

            # 读取数据
            vectors = np.fromfile(fin, dtype=np.float32, count=read_now * dim)
            if vectors.size != read_now * dim:
                raise ValueError("Unexpected EOF or file truncated.")

            vectors = vectors.reshape(read_now, dim)

            # 写入 .fvecs 格式
            for vec in vectors:
                np.array([dim], dtype=np.uint32).tofile(fout)
                vec.tofile(fout)

            total_read += read_now
            print(f"Processed {total_read} / {num_vectors} vectors")

    print("Conversion completed.")

if __name__ == "__main__":
    # input_base = "/home/lanlu/scaleGANN/dataset/laion100M/query.fbin"
    # output_base = "/home/lanlu/scaleGANN/dataset/laion100M/query.fvecs"
    # fbin_to_fvecs_10GB_chunked(input_base, output_base)

    input_base ="/home/lanlu/scaleGANN/dataset/sift1B/base.1B.u8bin"
    output_base ="//home/lanlu/scaleGANN/dataset/sift1B/base.1B.bvecs"
    u8bin_to_bvecs(input_base, output_base)

    input_query = "/home/lanlu/scaleGANN/dataset/sift1B/query.public.10K.u8bin"
    output_query = "/home/lanlu/scaleGANN/dataset/sift1B/query.bvecs"
    u8bin_to_bvecs(input_query, output_query)

    # input_groundtruth = "/home/lanlu/scaleGANN/dataset/simSearchNet1M/groundtruth.neighbors.ibin"
    # output_groundtruth = "/home/lanlu/ggnn/data/simSearchNet1M/groundtruth.ivecs"
    # ibin_to_ivecs(input_groundtruth, output_groundtruth)


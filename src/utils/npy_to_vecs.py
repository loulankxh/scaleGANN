import numpy as np
import struct
import sys

def npy_to_fvecs(npy_file, fvecs_file):
    data = np.load(npy_file)

    if data.dtype != np.float32:
        print("Data type is atually: ", data.dtype)
        print("Converting data type to float32")
        data = data.astype(np.float32)

    num_vectors, dim = data.shape
    print(f"Converting {npy_file} -> {fvecs_file}")
    print(f"Number of vectors: {num_vectors}, Dimension: {dim}")

    with open(fvecs_file, "wb") as f:
        for vec in data:
            f.write(struct.pack("I", dim)) 
            f.write(vec.tobytes())

    print(f"Conversion complete! Saved as {fvecs_file}")

if __name__ == "__main__":
    # if len(sys.argv) < 3:
    #     print("Usage: python npy_to_fvecs.py <input.npy> <output.fvecs>")
    #     sys.exit(1)

    # npy_to_fvecs(sys.argv[1], sys.argv[2])
    npy_file="/home/lanlu/scaleGANN/dataset/laion1M/text_emb_1.npy"
    fvecs_file="/home/lanlu/scaleGANN/dataset/laion1M/query.fvecs"
    npy_to_fvecs(npy_file,fvecs_file)
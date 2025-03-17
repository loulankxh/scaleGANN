import numpy as np
import struct
import sys

def npy_to_fbin(npy_file, fbin_file):
    data = np.load(npy_file)

    if data.dtype != np.float32:
        print("Data type is atually: ", data.dtype)
        print("Converting data type to float32")
        data = data.astype(np.float32)

    num_vectors, dim = data.shape
    print(f"Converting {npy_file} -> {fbin_file}")
    print(f"Number of vectors: {num_vectors}, Dimension: {dim}")

    with open(fbin_file, "wb") as f:
        f.write(struct.pack("II", num_vectors, dim))
        f.write(data.tobytes())

    print(f"Conversion complete! Saved as {fbin_file}")

if __name__ == "__main__":
    # if len(sys.argv) < 3:
    #     print("Usage: python npy_to_fbin.py <input.npy> <output.fbin>")
    #     sys.exit(1)

    # npy_to_fbin(sys.argv[1], sys.argv[2])
    npy_file="/home/lanlu/scaleGANN/dataset/laion1M/img_emb_1.npy"
    fbin_file="/home/lanlu/scaleGANN/dataset/laion1M/base.fbin"
    npy_to_fbin(npy_file,fbin_file)

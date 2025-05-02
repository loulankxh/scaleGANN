import numpy as np
import struct
import random
import os

def load_u8bin(fname):
    with open(fname, "rb") as f:
        data = np.fromfile(f, dtype=np.uint32, count=2)
        npts = data[0]
        dim = data[1]
        print(f"First uint32: {npts}, Second uint32: {dim}")
        data = np.fromfile(f, dtype=np.uint8)
    return data.reshape(npts, dim)


def write_fbin(fname, data):
    with open(fname, "wb") as f:
        np.asarray(data.shape, dtype=np.uint32).tofile(f)  
        data.astype(np.float32).tofile(f)  


def get_subset_fbin(input_file, output_file, num_rows_to_extract):
    try:
        with open(input_file, 'rb') as f:
            rows, cols = np.fromfile(f, dtype=np.int32, count=2)
            print(rows, cols)

            if num_rows_to_extract > rows:
                raise ValueError("Requested number of rows exceeds total rows in the file.")

            subset_data = np.zeros((num_rows_to_extract, cols), dtype=np.float32)


            for i in range(num_rows_to_extract):
                row_data = struct.unpack(f'{cols}f', f.read(cols * 4))
                subset_data[i] = row_data

        with open(output_file, 'wb') as f_out:
            header = np.array([num_rows_to_extract, cols], dtype=np.int32)
            header.tofile(f_out)
            subset_data.astype(np.float32).tofile(f_out)

        print(f"Successfully extracted {num_rows_to_extract} rows to {output_file}")

    except Exception as e:
        print(f"Error: {e}")



def get_subset_u8bin(input_file, output_file, num_rows_to_extract):
    try:
        with open(input_file, 'rb') as f:
            rows, cols = np.fromfile(f, dtype=np.int32, count=2)
            print(rows, cols)

            if num_rows_to_extract > rows:
                raise ValueError("Requested number of rows exceeds total rows in the file.")

            subset_data = np.zeros((num_rows_to_extract, cols), dtype=np.uint8)

            for i in range(num_rows_to_extract):
                row_data = struct.unpack(f'{cols}B', f.read(cols * 1))
                subset_data[i] = row_data
                # print(i)

        with open(output_file, 'wb') as f_out:
            header = np.array([num_rows_to_extract, cols], dtype=np.int32)
            header.tofile(f_out)
            subset_data.astype(np.uint8).tofile(f_out)

        print(f"Successfully extracted {num_rows_to_extract} rows to {output_file}")

    except Exception as e:
        print(f"Error: {e}")


def shuffle_bvecs(input_file, output_file, num_rows, dim):
    
    bytes_per_row = 4 + dim 

    file_size = os.path.getsize(input_file)
    expected_size = num_rows * bytes_per_row
    if file_size != expected_size:
        raise ValueError(f"Ubmatch between input file size ({file_size}) and expected file size ({expected_size})")

    with open(input_file, 'rb') as f:
        data = f.read()

    indices = np.arange(num_rows)
    np.random.shuffle(indices)

    vectors = []
    for i in range(num_rows):
        start = indices[i] * bytes_per_row
        end = start + bytes_per_row
        vectors.append(data[start:end])
        print(i, indices[i])


    with open(output_file, 'wb') as f:
        for vector in vectors:
            f.write(vector)

    print(f"Successfully shuffled {num_rows} rows data, and save to {output_file}")




if __name__ == "__main__":
    input_file = "/home/lanlu/scaleGANN/dataset/msTuring100M/query100K.fbin"
    output_file = "/home/lanlu/scaleGANN/dataset/msTuring100M/query.fbin"
    million = 1000000
    # num_rows_to_extract = 100 * million
    num_rows_to_extract = 10000
    
    get_subset_fbin(input_file, output_file, num_rows_to_extract)

    # input_file = "/home/lanlu/ggnn/data/sift100M/base.bvecs"
    # output_file = "/home/lanlu/ggnn/data/sift100M/baseShuffled.bvecs"
    # num_rows = 100000000
    # dim = 128

    # shuffle_bvecs(input_file, output_file, num_rows, dim)
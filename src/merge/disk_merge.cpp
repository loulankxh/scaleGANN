#include <string>
#include <vector>
#include <cmath>
#include <cassert>
#include <chrono>

#include "../../DiskANN/include/disk_utils.h"
#include "../utils/indexIO.hpp"
#include "../utils/datasetIO.hpp"
#include "disk_merge.h"

void read_idmap(const std::string &fname, std::vector<uint32_t> &ivecs)
{
    uint32_t npts32;
    size_t actual_file_size = get_file_size(fname);
    std::ifstream reader(fname.c_str(), std::ios::binary);
    reader.read((char *)&npts32, sizeof(uint32_t));
    if (actual_file_size != ((size_t)npts32) * sizeof(uint32_t) + sizeof(uint32_t))
    {
        std::stringstream stream;
        stream << "Error reading idmap file. Check if the file is bin file with "
                  "1 dimensional data. Actual: "
               << actual_file_size << ", expected: " << (size_t)npts32 * sizeof(uint32_t) + sizeof(uint32_t) << std::endl;

        throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
    }
    ivecs.resize(npts32);
    reader.read((char *)ivecs.data(), ((size_t)npts32) * sizeof(uint32_t));
    reader.close();
}


#define RAFT_NUMPY_MAGIC_STRING_LENGTH 6
#define RAFT_NUMPY_MAGIC_STRING        "\x93NUMPY"

void read_header(cached_ifstream& is){
    char magic_buf[RAFT_NUMPY_MAGIC_STRING_LENGTH + 2] = {0};
    is.read(magic_buf, RAFT_NUMPY_MAGIC_STRING_LENGTH + 2);
    // printf("magic buffer is: %s\n",magic_buf);

    std::uint8_t header_len_le16[2];
    is.read(reinterpret_cast<char*>(header_len_le16), 2);

    const std::uint32_t header_length = (header_len_le16[0] << 0) | (header_len_le16[1] << 8);
    std::vector<char> header_bytes(header_length);
    is.read(header_bytes.data(), header_length);
    std::string str = std::string(header_bytes.data(), header_length);
}

template <typename T>
T read_once(cached_ifstream& is){
    read_header(is);

    T val;
    is.read(reinterpret_cast<char*>(&val), sizeof(T));
    return val;
}

template <typename T>
struct always_false : std::false_type {};

template <typename T>
void write_header(cached_ofstream& os, bool fortran_order, uint32_t shape_x = 0, uint32_t shape_y = 0, uint32_t shape_z = 0) {
    const std::string header_dict = header_to_string<T>(fortran_order, shape_x, shape_y, shape_z);

    os.write(RAFT_NUMPY_MAGIC_STRING, RAFT_NUMPY_MAGIC_STRING_LENGTH);
    // if (!os) {
    //     throw std::runtime_error("Failed to write magic string to the output stream.");
    // }
    // Use version 1.0
    char ch = 1;
    os.write(&ch, 1);
    ch = 0;
    os.write(&ch, 1);
    // os.put(1);
    // os.put(0);
    // if (!os) {
    //     throw std::runtime_error("Failed to write magic string to the output stream.");
    // }

    const std::uint32_t header_length = 118;
    std::uint8_t header_len_le16[2] = {
        static_cast<std::uint8_t>(header_length & 0xFF),
        static_cast<std::uint8_t>((header_length >> 8) & 0xFF)};
    os.write(reinterpret_cast<const char*>(header_len_le16), 2);
    // if (!os) {
    //     throw std::runtime_error("Failed to write header length to the output stream.");
    // }

    std::size_t preamble_length = RAFT_NUMPY_MAGIC_STRING_LENGTH + 2 + 2 + header_dict.length() + 1;
    // Enforce 64-byte alignment
    std::size_t padding_len = 64 - preamble_length % 64;
    std::string padding(padding_len, ' ');
    // os << header_dict << padding << "\n";
    os.write(header_dict.data(), header_dict.size());
    os.write(padding.data(), padding.size());
    char newline = '\n';
    os.write(&newline, 1);
    // if (!os) {
    //     throw std::runtime_error("Failed to write header data to the output stream.");
    // }
}

template <typename T>
void write_once(cached_ofstream& os, const T& value) {
    write_header<T>(os, false);

    os.write(reinterpret_cast<const char*>(&value), sizeof(T));
    // if (!os) {
    //     throw std::runtime_error("Failed to write the value to the output stream.");
    // }
}



// In the partition step, points are handled sequentially such that within each shard, 
// the points' global id are ascending as in the complete dataset
int DiskANN_merge_sequentialShardRead(const std::string base_folder,
                const uint64_t nshards, uint32_t merge_degree, uint32_t constructed_deg,
                const std::string output_index_file,
                const std::string index_name)
{
    // Read ID maps
    std::vector<std::string> index_file_names(nshards);
    std::vector<std::vector<uint32_t>> idmaps(nshards);
    for (uint64_t shard = 0; shard < nshards; shard++)
    {   
        index_file_names[shard] = base_folder + "/partition" + std::to_string(shard) + "/index/" + index_name;
        read_idmap(base_folder + "/partition" + std::to_string(shard) + "/idmap.ibin", idmaps[shard]);
    }

    // find max node id
    size_t nnodes = 0;
    size_t nelems = 0;
    for (auto &idmap : idmaps)
    {
        for (auto &id : idmap)
        {
            nnodes = std::max(nnodes, (size_t)id);
        }
        nelems += idmap.size();
    }
    nnodes++;
    diskann::cout << "# nodes: " << nnodes << ", merge. degree: " << merge_degree << std::endl;

    // compute inverse map: node -> shards
    std::vector<std::pair<uint32_t, uint32_t>> node_shard;
    node_shard.reserve(nelems);
    for (size_t shard = 0; shard < nshards; shard++)
    {
        diskann::cout << "Creating inverse map -- shard #" << shard << std::endl;
        for (size_t idx = 0; idx < idmaps[shard].size(); idx++)
        {
            size_t node_id = idmaps[shard][idx];
            node_shard.push_back(std::make_pair((uint32_t)node_id, (uint32_t)shard));
        }
    }
    std::sort(node_shard.begin(), node_shard.end(), [](const auto &left, const auto &right) {
        return left.first < right.first || (left.first == right.first && left.second < right.second);
    });
    diskann::cout << "Finished computing node -> shards map" << std::endl;

    // create cached readers
    std::vector<cached_ifstream> index_readers(nshards);
    for (size_t i = 0; i < nshards; i++)
    {   
        index_readers[i].open(index_file_names[i], BUFFER_SIZE_FOR_CACHED_IO);
        char dtype_string[4];
        index_readers[i].read(dtype_string, 4);
        read_once<int>(index_readers[i]); // int version
        read_once<std::uint32_t>(index_readers[i]); // uint32_t rows
        read_once<std::uint32_t>(index_readers[i]); // uint32_t dim
        assert(constructed_deg == read_once<std::uint32_t>(index_readers[i])); // uint32_t deg
        read_once<unsigned short>(index_readers[i]); // unsigned short metric
        read_header(index_readers[i]);
    }

    size_t vamana_metadata_size =
        sizeof(uint64_t) + sizeof(uint32_t) + sizeof(uint32_t) + sizeof(uint64_t); // expected file size + max degree +
                                                                                   // medoid_id + frozen_point info

    // create cached vamana writers
    cached_ofstream merged_vamana_writer(output_index_file, BUFFER_SIZE_FOR_CACHED_IO);
    // write header
    std::string dtype_string = "<f4";
    dtype_string.resize(4);
    merged_vamana_writer.write(dtype_string.c_str(), 4);
    int version = 4;
    write_once<int>(merged_vamana_writer, version);
    uint32_t rows = (uint32_t)nnodes;
    write_once<std::uint32_t>(merged_vamana_writer, rows);
    uint32_t dim = 0;
    write_once<std::uint32_t>(merged_vamana_writer, dim);
    uint32_t deg = merge_degree;
    write_once<std::uint32_t>(merged_vamana_writer, deg);
    unsigned short metric = 0;
    write_once<unsigned short>(merged_vamana_writer, metric);

    diskann::cout << "Starting merge" << std::endl;

    // Gopal. random_shuffle() is deprecated.
    std::random_device rng;
    std::mt19937 urng(rng());

    std::vector<bool> nhood_set(nnodes, 0);
    std::vector<uint32_t> final_nhood;

    // uint32_t nnbrs = 0, shard_nnbrs = 0;
    uint32_t cur_id = 0;
    for (const auto &id_shard : node_shard)
    {
        uint32_t node_id = id_shard.first;
        uint32_t shard_id = id_shard.second;
        if (cur_id < node_id)
        {
            // Gopal. random_shuffle() is deprecated.
            std::shuffle(final_nhood.begin(), final_nhood.end(), urng);
            assert((final_nhood.size() >= merge_degree));
            // write into merged ofstream
            merged_vamana_writer.write((char *)final_nhood.data(), merge_degree * sizeof(uint32_t));
            if (cur_id % 499999 == 1)
            {
                diskann::cout << "." << std::flush;
            }
            cur_id = node_id;
            for (auto &p : final_nhood)
                nhood_set[p] = 0;
            final_nhood.clear();
        }
        // read from shard_id ifstream
        std::vector<uint32_t> shard_nhood(constructed_deg);
        index_readers[shard_id].read((char *)shard_nhood.data(), constructed_deg * sizeof(uint32_t));
        // rename nodes
        for (uint64_t j = 0; j < constructed_deg; j++)
        {
            if (nhood_set[idmaps[shard_id][shard_nhood[j]]] == 0)
            {
                nhood_set[idmaps[shard_id][shard_nhood[j]]] = 1;
                final_nhood.emplace_back(idmaps[shard_id][shard_nhood[j]]);
            }
        }
    }

    // Gopal. random_shuffle() is deprecated.
    std::shuffle(final_nhood.begin(), final_nhood.end(), urng);
    assert((final_nhood.size() >= merge_degree));
    // write into merged ofstream
    merged_vamana_writer.write((char *)final_nhood.data(), merge_degree * sizeof(uint32_t));
    for (auto &p : final_nhood)
        nhood_set[p] = 0;
    final_nhood.clear();

    // write pos metadata
    bool has_dataset = 0;
    write_once<bool>(merged_vamana_writer, has_dataset);

    diskann::cout << "Finished merge" << std::endl;
    return 0;
}


// In the partition step, points are handled parallely such that within each shard, 
// the points' global id are random compared with the complete dataset.
// Thus, we need to utilize "seekg" to locate the current read position
// We use the buffered read, and may need to reload the read buffer.
int scaleGANN_merge(const std::string base_folder,
                const uint64_t nshards, uint32_t max_degree, uint32_t constructed_deg,
                const std::string output_index_file,
                const std::string index_name)
{
    auto startTime = std::chrono::high_resolution_clock::now();

    // Read ID maps
    std::vector<std::string> index_file_names(nshards);
    std::vector<std::vector<uint32_t>> idmaps(nshards);
    for (uint64_t shard = 0; shard < nshards; shard++)
    {   
        index_file_names[shard] = base_folder + "/partition" + std::to_string(shard) + "/index/" + index_name;
        read_idmap(base_folder + "/partition" + std::to_string(shard) + "/idmap.ibin", idmaps[shard]);
    }

    // find max node id
    size_t nnodes = 0;
    size_t nelems = 0;
    for (auto &idmap : idmaps)
    {
        for (auto &id : idmap)
        {
            nnodes = std::max(nnodes, (size_t)id);
        }
        printf("Shard has # element %d\n", idmap.size());
        nelems += idmap.size();
    }
    nnodes++;
    diskann::cout << "# nodes: " << nnodes << ", max. degree: " << max_degree << std::endl;

    // compute inverse map: node -> shards
    std::vector<std::pair<uint32_t, std::pair<uint32_t, uint32_t>>> node_shard;
    node_shard.reserve(nelems);
    for (size_t shard = 0; shard < nshards; shard++)
    {
        diskann::cout << "Creating inverse map -- shard #" << shard << std::endl;
        for (size_t idx = 0; idx < idmaps[shard].size(); idx++)
        {
            size_t node_id = idmaps[shard][idx];
            node_shard.push_back(std::make_pair((uint32_t)node_id, std::make_pair((uint32_t)shard, idx)));
        }
    }
    std::sort(node_shard.begin(), node_shard.end(), [](const auto &left, const auto &right) {
        return left.first < right.first || (left.first == right.first && left.second.first < right.second.first);
    });
    diskann::cout << "Finished computing node -> shards map" << std::endl;

    auto idMapDealTime = std::chrono::high_resolution_clock::now();

    // create cached readers
    std::vector<cached_ifstream> index_readers(nshards);
    uint32_t header_size = 790; // CAGRA index header size
    for (size_t i = 0; i < nshards; i++)
    {   
        index_readers[i].open(index_file_names[i], BUFFER_SIZE_FOR_CACHED_IO);
    }

    // create cached vamana writers
    cached_ofstream merged_vamana_writer(output_index_file, BUFFER_SIZE_FOR_CACHED_IO);
    // write header
    std::string dtype_string = "<f4";
    dtype_string.resize(4);
    merged_vamana_writer.write(dtype_string.c_str(), 4);
    int version = 4;
    write_once<int>(merged_vamana_writer, version);
    uint32_t rows = (uint32_t)nnodes;
    write_once<std::uint32_t>(merged_vamana_writer, rows);
    uint32_t dim = 0;
    write_once<std::uint32_t>(merged_vamana_writer, dim);
    uint32_t deg = max_degree;
    write_once<std::uint32_t>(merged_vamana_writer, deg);
    unsigned short metric = 0;
    write_once<unsigned short>(merged_vamana_writer, metric);

    uint32_t nshards_u32 = (uint32_t)nshards;
    uint32_t one_val = 1;

    diskann::cout << "Starting merge" << std::endl;

    // Gopal. random_shuffle() is deprecated.
    std::random_device rng;
    std::mt19937 urng(rng());

    std::vector<bool> nhood_set(nnodes, 0);
    std::vector<uint32_t> final_nhood;

    // uint32_t nnbrs = 0, shard_nnbrs = 0;
    uint32_t cur_id = 0;
    for (const auto &id_shard : node_shard)
    {
        uint32_t node_id = id_shard.first;
        uint32_t shard_id = id_shard.second.first;
        uint32_t node_local_id = id_shard.second.second;
        if (cur_id < node_id)
        {
            // Gopal. random_shuffle() is deprecated.
            std::shuffle(final_nhood.begin(), final_nhood.end(), urng);
            assert((final_nhood.size() >= max_degree));
            // write into merged ofstream
            merged_vamana_writer.write((char *)final_nhood.data(), max_degree * sizeof(uint32_t));
            if (cur_id % 499999 == 1)
            {
                diskann::cout << "." << std::flush;
            }
            cur_id = node_id;
            for (auto &p : final_nhood)
                nhood_set[p] = 0;
            final_nhood.clear();
        }
        // read from shard_id ifstream
        std::vector<uint32_t> shard_nhood(constructed_deg);
        std::streampos read_pos = header_size + node_local_id * constructed_deg * sizeof(uint32_t);
        index_readers[shard_id].seekg(read_pos);
        assert(((index_readers[shard_id].tellg() + static_cast<std::streampos>(constructed_deg * sizeof(uint32_t))) < index_readers[shard_id].get_file_size()));
        index_readers[shard_id].read((char *)shard_nhood.data(), constructed_deg * sizeof(uint32_t));
        // rename nodes
        for (uint64_t j = 0; j < constructed_deg; j++)
        {
            if (nhood_set[idmaps[shard_id][shard_nhood[j]]] == 0)
            {
                nhood_set[idmaps[shard_id][shard_nhood[j]]] = 1;
                final_nhood.emplace_back(idmaps[shard_id][shard_nhood[j]]);
            }
        }
    }
    // Gopal. random_shuffle() is deprecated.
    std::shuffle(final_nhood.begin(), final_nhood.end(), urng);
    assert((final_nhood.size() >= max_degree));
    // write into merged ofstream
    merged_vamana_writer.write((char *)final_nhood.data(), max_degree * sizeof(uint32_t));
    for (auto &p : final_nhood)
        nhood_set[p] = 0;
    final_nhood.clear();

    // write pos metadata
    bool has_dataset = 0;
    write_once<bool>(merged_vamana_writer, has_dataset);

    auto endTime = std::chrono::high_resolution_clock::now();
    auto idMapDealDuration = std::chrono::duration_cast<std::chrono::milliseconds>(idMapDealTime - startTime);
    auto mergeReadWriteDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - idMapDealTime);
    auto overallDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    printf("idmap dealing duration: %lld milliseconds, merge & read $ write duration: %lld milliseconds\n", idMapDealDuration.count(), mergeReadWriteDuration.count());
    printf("overall duration: %lld milliseconds\n", overallDuration.count());

    diskann::cout << "Finished merge" << std::endl;
    return 0;
}

// In the partition step, points are handled parallely such that within each shard, 
// the points' global id are random compared with the complete dataset.
// Thus, we need to utilize "seekg" to locate the current read position
// We don't use buffer in the read.
int scaleGANN_merge_unCachedRead(const std::string base_folder,
    const uint64_t nshards, uint32_t merge_degree, uint32_t constructed_deg,
    const std::string output_index_file,
    const std::string index_name)
{
auto startTime = std::chrono::high_resolution_clock::now();

// Read ID maps
std::vector<std::string> index_file_names(nshards);
std::vector<std::vector<uint32_t>> idmaps(nshards);
for (uint64_t shard = 0; shard < nshards; shard++)
{   
index_file_names[shard] = base_folder + "/partition" + std::to_string(shard) + "/index/" + index_name;
read_idmap(base_folder + "/partition" + std::to_string(shard) + "/idmap.ibin", idmaps[shard]);
}

// find max node id
size_t nnodes = 0;
size_t nelems = 0;
for (auto &idmap : idmaps)
{
for (auto &id : idmap)
{
nnodes = std::max(nnodes, (size_t)id);
}
printf("Shard has # element %d\n", idmap.size());
nelems += idmap.size();
}
nnodes++;
diskann::cout << "# nodes: " << nnodes << ", merge. degree: " << merge_degree << std::endl;

// compute inverse map: node -> shards
std::vector<std::pair<uint32_t, std::pair<uint32_t, uint32_t>>> node_shard;
node_shard.reserve(nelems);
for (size_t shard = 0; shard < nshards; shard++)
{
diskann::cout << "Creating inverse map -- shard #" << shard << std::endl;
for (size_t idx = 0; idx < idmaps[shard].size(); idx++)
{
size_t node_id = idmaps[shard][idx];
node_shard.push_back(std::make_pair((uint32_t)node_id, std::make_pair((uint32_t)shard, idx)));
}
}
std::sort(node_shard.begin(), node_shard.end(), [](const auto &left, const auto &right) {
return left.first < right.first || (left.first == right.first && left.second.first < right.second.first);
});
diskann::cout << "Finished computing node -> shards map" << std::endl;

auto idMapDealTime = std::chrono::high_resolution_clock::now();

// create cached readers
// std::vector<cached_ifstream> index_readers(nshards);
std::vector<std::unique_ptr<std::ifstream>> index_readers(index_file_names.size());
uint32_t header_size = 790; // CAGRA index header size
for (size_t i = 0; i < nshards; i++)
{   
// index_readers[i].open(index_file_names[i], BUFFER_SIZE_FOR_CACHED_IO);
index_readers[i] = std::make_unique<std::ifstream>(index_file_names[i], std::ios::binary);
if (!index_readers[i]->is_open()) {
std::cerr << "Failed to open file: " << index_file_names[i] << std::endl;
}
}

// create cached vamana writers
cached_ofstream merged_vamana_writer(output_index_file, BUFFER_SIZE_FOR_CACHED_IO);
// write header
std::string dtype_string = "<f4";
dtype_string.resize(4);
merged_vamana_writer.write(dtype_string.c_str(), 4);
int version = 4;
write_once<int>(merged_vamana_writer, version);
uint32_t rows = (uint32_t)nnodes;
write_once<std::uint32_t>(merged_vamana_writer, rows);
uint32_t dim = 0;
write_once<std::uint32_t>(merged_vamana_writer, dim);
uint32_t deg = merge_degree;
write_once<std::uint32_t>(merged_vamana_writer, deg);
unsigned short metric = 0;
write_once<unsigned short>(merged_vamana_writer, metric);

diskann::cout << "Starting merge" << std::endl;

// Gopal. random_shuffle() is deprecated.
std::random_device rng;
std::mt19937 urng(rng());

std::vector<bool> nhood_set(nnodes, 0);
std::vector<uint32_t> final_nhood;

// uint32_t nnbrs = 0, shard_nnbrs = 0;
uint32_t cur_id = 0;
for (const auto &id_shard : node_shard)
{
uint32_t node_id = id_shard.first;
uint32_t shard_id = id_shard.second.first;
uint32_t node_local_id = id_shard.second.second;
if (cur_id < node_id)
{
// Gopal. random_shuffle() is deprecated.
std::shuffle(final_nhood.begin(), final_nhood.end(), urng);
assert((final_nhood.size() >= merge_degree));
// write into merged ofstream
merged_vamana_writer.write((char *)final_nhood.data(), merge_degree * sizeof(uint32_t));
if (cur_id % 499999 == 1)
{
    diskann::cout << "." << std::flush;
}
cur_id = node_id;
for (auto &p : final_nhood)
    nhood_set[p] = 0;
final_nhood.clear();
}
// read from shard_id ifstream
std::vector<uint32_t> shard_nhood(constructed_deg);
std::streampos read_pos = header_size + node_local_id * constructed_deg * sizeof(uint32_t);
index_readers[shard_id]->seekg(read_pos);
// assert(((index_readers[shard_id].tellg() + static_cast<std::streampos>(constructed_deg * sizeof(uint32_t))) < index_readers[shard_id].get_file_size()));
index_readers[shard_id]->read((char *)shard_nhood.data(), constructed_deg * sizeof(uint32_t));
// rename nodes
for (uint64_t j = 0; j < constructed_deg; j++)
{
if (nhood_set[idmaps[shard_id][shard_nhood[j]]] == 0)
{
    nhood_set[idmaps[shard_id][shard_nhood[j]]] = 1;
    final_nhood.emplace_back(idmaps[shard_id][shard_nhood[j]]);
}
}
}
// Gopal. random_shuffle() is deprecated.
std::shuffle(final_nhood.begin(), final_nhood.end(), urng);
assert((final_nhood.size() >= merge_degree));
// write into merged ofstream
merged_vamana_writer.write((char *)final_nhood.data(), merge_degree * sizeof(uint32_t));
for (auto &p : final_nhood)
nhood_set[p] = 0;
final_nhood.clear();

// write pos metadata
bool has_dataset = 0;
write_once<bool>(merged_vamana_writer, has_dataset);

auto endTime = std::chrono::high_resolution_clock::now();
auto idMapDealDuration = std::chrono::duration_cast<std::chrono::milliseconds>(idMapDealTime - startTime);
auto mergeReadWriteDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - idMapDealTime);
auto overallDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
printf("idmap dealing duration: %lld milliseconds, merge & read $ write duration: %lld milliseconds\n", idMapDealDuration.count(), mergeReadWriteDuration.count());
printf("overall duration: %lld milliseconds\n", overallDuration.count());

diskann::cout << "Finished merge" << std::endl;
return 0;
}

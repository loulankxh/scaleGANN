#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstring>
#include <vector>
#include <type_traits>
#include <cmath>
#include <cassert>
#include <filesystem>



#define RAFT_NUMPY_MAGIC_STRING_LENGTH 6

void read_header(std::istream& is){
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
T read_once(std::istream& is){
    read_header(is);

    T val;
    is.read(reinterpret_cast<char*>(&val), sizeof(T));
    return val;
}

void read_neighborLists(std::istream& is, uint32_t num, uint32_t*& val){
    read_header(is);

    is.read(reinterpret_cast<char*>(val), num * sizeof(uint32_t));
    printf("finishing reading the index\n");
}

void read_neighborLists(std::istream& is, std::vector<std::vector<uint32_t>>& index){
    read_header(is);
    uint32_t nrows = index.size();
    uint32_t deg = (nrows > 0) ? index[0].size() : 0;

    for(int i = 0; i < nrows; i++){
        is.read(reinterpret_cast<char*>(index[i].data()), deg * sizeof(uint32_t));
    }

    printf("finishing reading the index\n");
}




#define RAFT_NUMPY_MAGIC_STRING        "\x93NUMPY"

template <typename T>
struct always_false : std::false_type {};

template <typename T>
std::string header_to_string(bool fortran_order, uint32_t shape_x = 0, uint32_t shape_y = 0, uint32_t shape_z = 0){
    std::string dtype;
    if constexpr (std::is_same<T, int>::value) {
        dtype = "<i4";
    } else if constexpr (std::is_same<T, uint32_t>::value) {
        dtype = "<u4";
    } else if constexpr (std::is_same<T, unsigned short>::value) {
        dtype = "<u2";
    } else if constexpr (std::is_same<T, bool>::value) {
        dtype = "|u1";
    } else {
        static_assert(always_false<T>::value, "Unsupported type");
    }

    std::ostringstream oss;
    oss << "{'descr': '" << dtype
        << "', 'fortran_order': " << (fortran_order ? "True" : "False")
        << ", 'shape': " 
        << "(" << ((shape_x > 0) ? std::to_string(shape_x) : "") << ((shape_y > 0) ? "," : "")
        << ((shape_y > 0) ? std::to_string(shape_y): "") << ((shape_z > 0) ? "," : "")
        << ((shape_z > 0) ? std::to_string(shape_z) : "") << ")" << "}";
    return oss.str();
}

template <typename T>
void write_header(std::ostream& os, bool fortran_order, uint32_t shape_x = 0, uint32_t shape_y = 0, uint32_t shape_z = 0) {
    const std::string header_dict = header_to_string<T>(fortran_order, shape_x, shape_y, shape_z);

    os.write(RAFT_NUMPY_MAGIC_STRING, RAFT_NUMPY_MAGIC_STRING_LENGTH);
    if (!os) {
        throw std::runtime_error("Failed to write magic string to the output stream.");
    }
    // Use version 1.0
    os.put(1);
    os.put(0);
    if (!os) {
        throw std::runtime_error("Failed to write magic string to the output stream.");
    }

    const std::uint32_t header_length = 118;
    std::uint8_t header_len_le16[2] = {
        static_cast<std::uint8_t>(header_length & 0xFF),
        static_cast<std::uint8_t>((header_length >> 8) & 0xFF)};
    os.write(reinterpret_cast<const char*>(header_len_le16), 2);
    if (!os) {
        throw std::runtime_error("Failed to write header length to the output stream.");
    }

    std::size_t preamble_length = RAFT_NUMPY_MAGIC_STRING_LENGTH + 2 + 2 + header_dict.length() + 1;
    // Enforce 64-byte alignment
    std::size_t padding_len = 64 - preamble_length % 64;
    std::string padding(padding_len, ' ');
    os << header_dict << padding << "\n";
    if (!os) {
        throw std::runtime_error("Failed to write header data to the output stream.");
    }
}

template <typename T>
void write_once(std::ostream& os, const T& value) {
    write_header<T>(os, false);

    os.write(reinterpret_cast<const char*>(&value), sizeof(T));
    if (!os) {
        throw std::runtime_error("Failed to write the value to the output stream.");
    }
}

void write_neighborLists(std::ostream& os, std::vector<std::vector<uint32_t>>& values) {
    uint32_t rows = values.size();
    uint32_t deg = (rows > 0) ? values[0].size() : 0;
    write_header<uint32_t>(os, false, rows, deg);

    for(int i = 0; i < rows; i++){
        os.write(reinterpret_cast<const char*>(values[i].data()), values[i].size() * sizeof(uint32_t));
        if (!os) {
            throw std::runtime_error("Failed to write the dataset to the output stream.");
        }
    }

    printf("Index written: %zu values\n", (size_t)rows*deg);
}






void readIndexOneShard(const std::string index_file,
    std::vector<std::vector<std::vector<uint32_t>>>& partitions){
    std::ifstream file(index_file, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open file." << std::endl;
        return;
    }
    std::istream& is = file;

    char dtype_string[4];
    is.read(dtype_string, 4);

    int version = read_once<int>(is);
    uint32_t rows = read_once<std::uint32_t>(is);
    uint32_t dim = read_once<std::uint32_t>(is);
    uint32_t deg = read_once<std::uint32_t>(is);
    unsigned short metric =  read_once<unsigned short>(is);

    uint32_t num = rows*deg;
    std::vector<std::vector<uint32_t>> dataset(rows, std::vector<uint32_t>(deg));
    read_neighborLists(is, dataset);
    printf("Read index dimension is: %d * %d\n", dataset.size(), dataset[0].size());
    partitions.push_back(dataset);

    bool has_dataset = read_once<bool>(is);
}


void readIndex(const std::string index_file,
    std::vector<std::vector<uint32_t>>& index){
    std::ifstream file(index_file, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open file." << std::endl;
        return;
    }
    std::istream& is = file;

    char dtype_string[4];
    is.read(dtype_string, 4);

    int version = read_once<int>(is);
    uint32_t rows = read_once<std::uint32_t>(is);
    uint32_t dim = read_once<std::uint32_t>(is);
    uint32_t deg = read_once<std::uint32_t>(is);
    unsigned short metric =  read_once<unsigned short>(is);

    uint32_t num = rows*deg;
    std::vector<std::vector<uint32_t>> dataset(rows, std::vector<uint32_t>(deg));
    read_neighborLists(is, dataset);
    printf("Read index dimension is: %d * %d\n", dataset.size(), dataset[0].size());
    index = dataset;

    bool has_dataset = read_once<bool>(is);
}


void writeIndexMerged(
    const std::string index_file, std::vector<std::vector<uint32_t>>& index){
    std::filesystem::path index_path(index_file);
    std::string dirPath = index_path.parent_path();
    if (!std::filesystem::exists(dirPath)) {
        if (std::filesystem::create_directories(dirPath)) {
            std::cout << "Dir doesn't exists but success to create such dir: " << dirPath << std::endl;
        } else {
            std::cerr << "Dir doesn't exists and fail to create such dir: " << dirPath << std::endl;
        }
    }

    std::ofstream os(index_file, std::ios::out | std::ios::binary);
    if (!os) {
        throw std::runtime_error("Failed to open file for writing.");
    }
    printf("Starting writing index\n");

    // std::string dtype_string = raft::detail::numpy_serializer::get_numpy_dtype<T>().to_string();
    std::string dtype_string = "<f4";
    dtype_string.resize(4);
    os << dtype_string;

    int version = 4;
    write_once<int>(os, version);
    uint32_t rows = index.size();
    write_once<std::uint32_t>(os, rows);
    uint32_t dim = 0;
    write_once<std::uint32_t>(os, dim);
    uint32_t deg = (rows > 0) ? index[0].size() : 0;
    write_once<std::uint32_t>(os, deg);
    unsigned short metric = 0;
    write_once<unsigned short>(os, metric);
    printf("Finishing writing headers\n");

    write_neighborLists(os, index);
    printf("Write index dimension is: %d * %d\n", index.size(), index[0].size());

    printf("Finishing writing neighbors\n");

    bool has_dataset = 0;
    write_once<bool>(os, has_dataset);
}






size_t align8(size_t size) {
    return (size + 7) & ~static_cast<size_t>(7);
}


void getGGNNParas(uint32_t N_shard, uint32_t L, uint32_t S, uint32_t KBuild,
        uint32_t& G, uint32_t&  N_all, uint32_t& ST_all,
        std::vector<uint32_t>& Ns, std::vector<uint32_t>& Ns_offsets, std::vector<uint32_t>& STs_offsets){
    const float growth = powf(N_shard / static_cast<float>(S), 1.f / (L - 1));
    const uint32_t Gf = growth;
    const uint32_t Gc = growth + 1;
    const float S0f = N_shard / (pow(Gf, (L - 1)));
    const float S0c = N_shard / (pow(Gc, (L - 1)));
    const bool is_floor =
        (growth > 0) && ((S0c < KBuild) || (fabs(S0f - S) < fabs(S0c - S)));
    G = (is_floor) ? Gf : Gc;


    Ns.resize(L);
    Ns_offsets.resize(L);
    STs_offsets.resize(L);

    N_all = 0;
    ST_all = 0;
    uint32_t N_current = N_shard;
    for (int l = 0; l < L; l++) {
        Ns[l] = N_current;
        Ns_offsets[l] = N_all;
        STs_offsets[l] = ST_all;
        N_all += N_current;
        if (l) {
        ST_all += N_current;
        N_current /= G;
        }
        else {
        N_current = S;
        for (int i=2;i<L; ++i)
            N_current *= G;
        }
    }
}


template <typename keyT, typename valueT>
void checkGGNNSize(const uint32_t N, const uint32_t K, const uint32_t N_all, const uint32_t ST_all,
        size_t& total_graph_size){
    const size_t graph_size = align8(static_cast<size_t>(N_all) * K * sizeof(keyT));
    const size_t selection_translation_size = align8(ST_all * sizeof(keyT));
    // const size_t nn1_dist_buffer_size = N * sizeof(ValueT);
    const size_t nn1_stats_size = align8(2 * sizeof(valueT));
    size_t calculated_graph_size = graph_size + 2 * selection_translation_size + nn1_stats_size;

    assert((calculated_graph_size == total_graph_size) && "File size should equal calculated index size");
}


void convertMemoryToIndexGGNN(
    const char* h_memory,
    std::vector<std::vector<uint32_t>>& index,
    std::vector<uint32_t>& translation,
    const uint32_t N, 
    const uint32_t K,
    const uint32_t N_all,
    const uint32_t ST_all
) {
    index.resize(N_all);

    for (uint32_t i = 0; i < N_all; ++i) {
        index[i].resize(K);

        size_t offset = static_cast<size_t>(i) * K * sizeof(uint32_t);

        std::memcpy(index[i].data(), h_memory + offset, K * sizeof(uint32_t));
    }

    size_t offset_ST = static_cast<size_t>(N_all) * K * sizeof(uint32_t);
    translation.resize(ST_all);
    std::memcpy(translation.data(), h_memory + offset_ST, ST_all * sizeof(uint32_t));

    // printf("Check ST values last 32\n");
    // size_t offset_N = static_cast<size_t>(N_all - 32) * K * sizeof(uint32_t);
    // size_t offset_ST_32 = static_cast<size_t>(N_all) * K * sizeof(uint32_t) + static_cast<size_t>(ST_all - 32) * sizeof(uint32_t);
    // for(uint32_t i = 0 ; i < 32; i++){
    //     printf("The %d value has neighbors: \n", i);
    //     for(uint32_t count = 0; count < K ; count++){
    //         int nb = 0;
    //         std::memcpy(&nb, h_memory + offset_N, sizeof(uint32_t));
    //         printf("%d ", nb);
    //         offset_N += sizeof(uint32_t);
    //     }
    //     printf("\n");

    //     int value = 0;
    //     std::memcpy(&value, h_memory + offset_ST_32, sizeof(uint32_t));
    //     printf("The %d translation value is %d \n", i, value);
    //     offset_ST_32 += sizeof(uint32_t);
    // }
}


template <typename keyT, typename valueT>
void loadGGNNOneIndex(const std::string& path,
    uint32_t N_shard, uint32_t L, uint32_t S, uint32_t KBuild,
    uint32_t& G, uint32_t&  N_all, uint32_t& ST_all,
    std::vector<uint32_t>& Ns, std::vector<uint32_t>& Ns_offsets, std::vector<uint32_t>& STs_offsets,
    std::vector<std::vector<uint32_t>>& index, std::vector<uint32_t>& translation){

    std::ifstream file;
    file.open(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open ggnn index file: " + path);
    }


    getGGNNParas(N_shard, L, S, KBuild, G, N_all, ST_all, Ns, Ns_offsets, STs_offsets);

    file.seekg(0, std::ios::end);
    std::streampos file_size = file.tellg();
    size_t total_graph_size = (size_t)(static_cast<size_t>(file_size));
    checkGGNNSize<keyT, valueT>(N_shard, KBuild, N_all, ST_all, total_graph_size);
    // printf("file size is %zu\n", total_graph_size);

    char* h_memory = new char[total_graph_size];
    file.seekg(0, std::ios::beg);
    file.read(h_memory, total_graph_size);
    if (!file) {
        std::cerr << "Failed to read data or insufficient data in the file." << std::endl;
    }
    file.close();

    convertMemoryToIndexGGNN(h_memory, index, translation, N_shard, KBuild, N_all, ST_all);
    delete[] h_memory;
}






template int read_once<int>(std::istream& is);
template uint32_t read_once<uint32_t>(std::istream& is);
template unsigned short read_once<unsigned short>(std::istream& is);
template bool read_once<bool>(std::istream& is);


template std::string header_to_string <int>(bool fortran_order, uint32_t shape_x = 0, uint32_t shape_y = 0, uint32_t shape_z = 0);
template std::string header_to_string <uint32_t>(bool fortran_order, uint32_t shape_x = 0, uint32_t shape_y = 0, uint32_t shape_z = 0);
template std::string header_to_string <unsigned short>(bool fortran_order, uint32_t shape_x = 0, uint32_t shape_y = 0, uint32_t shape_z = 0);
template std::string header_to_string <bool>(bool fortran_order, uint32_t shape_x = 0, uint32_t shape_y = 0, uint32_t shape_z = 0);

template void write_header<int>(std::ostream& os, bool fortran_order, uint32_t shape_x = 0, uint32_t shape_y = 0, uint32_t shape_z = 0);
template void write_header<uint32_t>(std::ostream& os, bool fortran_order, uint32_t shape_x = 0, uint32_t shape_y = 0, uint32_t shape_z = 0);
template void write_header<unsigned short>(std::ostream& os, bool fortran_order, uint32_t shape_x = 0, uint32_t shape_y = 0, uint32_t shape_z = 0);
template void write_header<bool>(std::ostream& os, bool fortran_order, uint32_t shape_x = 0, uint32_t shape_y = 0, uint32_t shape_z = 0);

template void write_once<int>(std::ostream& os, const int& value);
template void write_once<uint32_t>(std::ostream& os, const uint32_t& value);
template void write_once<unsigned short>(std::ostream& os, const unsigned short& value);
template void write_once<bool>(std::ostream& os, const bool& value);


template void loadGGNNOneIndex<uint32_t, float>(const std::string& path,
    uint32_t N_shard, uint32_t L, uint32_t S, uint32_t KBuild,
    uint32_t& G, uint32_t&  N_all, uint32_t& ST_all,
    std::vector<uint32_t>& Ns, std::vector<uint32_t>& Ns_offsets, std::vector<uint32_t>& STs_offsets,
    std::vector<std::vector<uint32_t>>& index, std::vector<uint32_t>& translation);

#include <fstream>
#include <sstream>
#include <string>
#include <vector>


void read_header(std::istream& is);

template <typename T>
T read_once(std::istream& is);

void read_neighborLists(std::istream& is, uint32_t num, uint32_t*& val);

void read_neighborLists(std::istream& is, std::vector<std::vector<uint32_t>>& dataset);





template <typename T>
std::string header_to_string(bool fortran_order, uint32_t shape_x = 0, uint32_t shape_y = 0, uint32_t shape_z = 0);

template <typename T>
void write_header(std::ostream& os, bool fortran_order, uint32_t shape_x = 0, uint32_t shape_y = 0, uint32_t shape_z = 0);

template <typename T>
void write_once(std::ostream& os, const T& value);

void write_neighborLists(std::ostream& os, std::vector<std::vector<uint32_t>>& values);




void readIndexOneShard(const std::string index_file,
    std::vector<std::vector<std::vector<uint32_t>>>& partitions);

void readIndex(const std::string index_file,
    std::vector<std::vector<uint32_t>>& index);

void writeIndexMerged(
    const std::string index_file, std::vector<std::vector<uint32_t>>& index);



template <typename keyT, typename valueT>
void loadGGNNOneIndex(const std::string& path,
    uint32_t N_shard, uint32_t L, uint32_t S, uint32_t KBuild,
    uint32_t& G, uint32_t&  N_all, uint32_t& ST_all,
    std::vector<uint32_t>& Ns, std::vector<uint32_t>& Ns_offsets, std::vector<uint32_t>& STs_offsets,
    std::vector<std::vector<uint32_t>>& index, std::vector<uint32_t>& translation);


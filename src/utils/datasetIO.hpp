#include <vector>
#include <string>

uint32_t suffixToType(std::string file);

template <typename T>
std::string typeToSuffix();

template <typename T>
void arrayToVector(T* arr, std::vector<std::vector<T>>& vec);

void readMetadata(const std::string& filename, uint32_t* header);

void readMetadataOneDimension(const std::string& filename, uint32_t* header);

template <typename T>
void readFile(const std::string& filename, std::vector<std::vector<T>>& data);

template <typename T>
void readFileOneDimension(const std::string& filename, std::vector<T>& data);

template <typename T>
void readDatasetPartitions(const std::string& basePath, std::vector<std::vector<std::vector<T>>>& partitions);

void readIdxMaps(const std::string idx_file, std::vector<std::vector<uint32_t>>& idx_map);

template <typename T>
void read_query(const std::string query_file,
    std::vector<std::vector<T>>& query);

void read_groundTruth(const std::string truth_file,
    std::vector<std::vector<uint32_t>>& groundTruth);



template <typename T>
void writeDatasetPartitions(const std::string& basePath, const std::vector<std::vector<std::vector<T>>>& partitions);

void writeIdxMaps(const std::string& basePath, const std::vector<std::vector<uint32_t>>& idx_map);
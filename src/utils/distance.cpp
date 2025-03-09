#include <stdint.h>
#include <vector>
#include <cassert>
#include "distance.h"


template <typename T>
float l2_distance_square(const std::vector<T> n1, std::vector<T> n2){
    uint32_t len = n1.size();
    static_assert(std::is_arithmetic<T>::value, "T must be a numeric type");
    assert(n2.size()==len);

    float dist = 0.0;

    for (uint32_t d = 0; d < len; d++){
        float diff = (float)n1[d] - n2[d];
        dist += diff * diff;
    }

    return dist;
}

template <typename T>
float l2_distance_square_floatCentroid(const std::vector<T> n1, std::vector<float> n2){
    uint32_t len = n1.size();
    static_assert(std::is_arithmetic<T>::value, "T must be a numeric type");
    assert(n2.size()==len);

    float dist = 0;

    for (uint32_t d = 0; d < len; d++){
        float diff = (float) n1[d] - n2[d];
        dist += diff * diff;
    }

    return dist;
}

template float l2_distance_square<float>(const std::vector<float> n1, std::vector<float> n2);
template float l2_distance_square<uint32_t>(const std::vector<uint32_t> n1, std::vector<uint32_t> n2);
template float l2_distance_square<uint8_t>(const std::vector<uint8_t> n1, std::vector<uint8_t> n2);

template float l2_distance_square_floatCentroid<float>(const std::vector<float> n1, std::vector<float> n2);
template float l2_distance_square_floatCentroid<uint32_t>(const std::vector<uint32_t> n1, std::vector<float> n2);
template float l2_distance_square_floatCentroid<uint8_t>(const std::vector<uint8_t> n1, std::vector<float> n2);
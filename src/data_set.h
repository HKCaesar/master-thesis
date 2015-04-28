#ifndef DATA_SET_H
#define DATA_SET_H

#include <vector>
#include <string>
#include <cereal/types/vector.hpp>

struct DataSet {
    template <class Archive>
    void serialize(Archive& ar) {
        ar(CEREAL_NVP(filenames));
    }

    std::vector<std::string> filenames;
};

#endif

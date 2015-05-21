#ifndef BOOTSTRAP_H
#define BOOTSTRAP_H

#include <iostream>
#include <vector>
#include <array>
#include <memory>
#include "ceres/ceres.h"
#include <cereal/types/array.hpp>
#include <cereal/types/memory.hpp>
#include "data_set.h"
#include "image_features.h"
#include "camera_models.h"
#include "types.h"
#include "model.h"
#include "internal.h"

// Takes a Model and perform the bootstrap method on it
class Bootstrap {
public:
    void solve();

    template <class Archive>
    void serialize(Archive& ar) {
        ar(NVP(number_of_samples),
           NVP(size_of_samples),
           NVP(base_model),
           NVP(internals),
           NVP(externals));
    }

    size_t number_of_samples;
    size_t size_of_samples;
    std::shared_ptr<Model> base_model;

    std::vector<internal_t> internals;
    std::vector<std::vector<std::array<double, 6>>> externals;
};

#endif

#ifndef BOOTSTRAP_H
#define BOOTSTRAP_H

#include <iostream>
#include <vector>
#include <array>
#include "ceres/ceres.h"
#include <cereal/types/array.hpp>
#include <cereal/types/memory.hpp>
#include "data_set.h"
#include "image_features.h"
#include "camera_models.h"
#include "types.h"
#include "model.h"
#include "internal.h"

using std::vector;
using std::array;

// Takes a Model and perform the bootstrap method on it
class Bootstrap {
public:
    void solve();

    template <class Archive>
    void serialize(Archive& ar) {
        ar(NVP(number_of_samples),
           NVP(size_of_samples),
           NVP(model));
    }

    int number_of_samples;
    int size_of_samples;

    std::shared_ptr<Model> model;
};

#endif

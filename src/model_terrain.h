#ifndef MODEL_TERRAIN_H
#define MODEL_TERRAIN_H

#include <iostream>
#include <vector>
#include <array>
#include "ceres/ceres.h"
#include <cereal/types/array.hpp>
#include "data_set.h"
#include "image_features.h"
#include "camera_models.h"
#include "types.h"
#include "model.h"
#include "internal.h"

using std::vector;
using std::array;

// Model only the terrain as 3D points
// Cameras and internals are fixed from parent model

struct ModelTerrainReprojectionError : CostFunction<ModelTerrainReprojectionError, 2, 3> {
    const internal_t internal;
    const array<double, 6> external;
    const sensor_t observed;

	ModelTerrainReprojectionError(const internal_t internal, const array<double, 6> external , const sensor_t observed)
		: internal(internal), external(external), observed(observed) {}

	template <typename T>
	bool operator()(const T* const point, T* residuals) const {
        // Subtract observed coordinates
        // TODO: rename proj function
        // T ext[] = {T(external[0]), T(external[1]), T(external[2]), T(external[3]), T(external[4]), T(external[5])};
        bool r = model0_projection<T, double>(internal.data(), external.data(), point, residuals);
        residuals[0] -= T(observed.x);
        residuals[1] -= T(observed.y);
        return r;
    }
};

class ModelTerrain : public Model {
public:
    struct solution {
        vector<array<double, 3>> terrain; // 3 dof ground points

        template <class Archive>
        void serialize(Archive& ar) {
            ar(cereal::make_nvp("terrain", terrain));
        }
    };

    virtual void solve() override;

    template <class Archive>
    void serialize(Archive& ar) {
        ar(cereal::make_nvp("base", cereal::base_class<Model>(this)),
           cereal::make_nvp("cameras", cameras),
           cereal::make_nvp("internal", internal),
           cereal::make_nvp("parent", parent),
           cereal::make_nvp("solutions", solutions));
    }

    vector<array<double, 6>> cameras;
    internal_t internal;
    vector<solution> solutions;
    std::shared_ptr<Model> parent;
};

CEREAL_REGISTER_TYPE(ModelTerrain);

#endif

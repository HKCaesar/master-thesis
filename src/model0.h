#ifndef MODEL_ZERO_H
#define MODEL_ZERO_H

#include <iostream>
#include <vector>
#include <array>
#include "ceres/ceres.h"
#include "data_set.h"
#include "image_features.h"
#include "camera_models.h"
#include "types.h"
#include "model.h"

using std::vector;
using std::array;

// Most basic reprojection error
// 2D ground points (z=0)
// Fixed internals and no distortion

struct Model0ReprojectionError {
    const array<double, 3> internal;
    const sensor_t observed;

	// Factory to hide the construction of the CostFunction object
	static ceres::CostFunction* create(const array<double, 3> internal, const sensor_t observed) {
		// template parameters indicate the numbers of:
        // residuals, params in block 1, 2, 3...
		return (new ceres::AutoDiffCostFunction<Model0ReprojectionError, 2, 6, 2>(
			new Model0ReprojectionError(internal, observed)));
	}

	Model0ReprojectionError(const array<double, 3> internal, const sensor_t observed)
		: internal(internal), observed(observed) {}

	template <typename T>
	bool operator()(const T* const external, const T* const point, T* residuals) const {
        // Subtract observed coordinates
        bool r = model0_projection<T>(internal.data(), external, point, residuals);
        residuals[0] -= T(observed.x);
        residuals[1] -= T(observed.y);
        return r;
    }
};

class Model0 : public Model {
public:

    struct solution {
        vector<array<double, 6>> cameras; // 6 dof cameras
        vector<array<double, 2>> terrain; // 2 dof ground points on flat terrain

        template <class Archive>
        void serialize(Archive& ar) {
            ar(cereal::make_nvp("cameras", cameras),
            cereal::make_nvp("terrain", terrain));
        }
    };

    Model0();
    virtual void solve() override;

    template <class Archive>
    void serialize(Archive& ar) {
        ar(cereal::make_nvp("features", features),
           cereal::make_nvp("internal", internal),
           cereal::make_nvp("pixel_size", pixel_size),
           cereal::make_nvp("solutions", solutions),
           cereal::make_nvp("rows", rows),
           cereal::make_nvp("cols", cols));
    }

    std::shared_ptr<FeaturesGraph> features;

    // 3 dof internals: {f, ppx, ppy}
    array<double, 3> internal;
    double pixel_size;
    double rows, cols;

    // List of solutions, from the initial guess (or parent model) to local optimum
    vector<solution> solutions;
};

CEREAL_REGISTER_TYPE(Model0);

#endif

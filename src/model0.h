#ifndef MODEL_ZERO_H
#define MODEL_ZERO_H

#include <iostream>
#include <vector>
#include <array>
#include "ceres/ceres.h"
#include "data_set.h"
#include "image_features.h"
#include "camera_models.h"

using std::vector;
using std::array;

// Most basic reprojection error
// 2D ground points (z=0)
// Fixed internals and no distortion
struct Model0ReprojectionError {
    const array<double, 3> internal;
	const double observed_x;
	const double observed_y;

	// Factory to hide the construction of the CostFunction object
	static ceres::CostFunction* create(const array<double, 3> internal, const double observed_x, const double observed_y) {
		// template parameters indicate the numbers of:
        // residuals, params in block 1, 2, 3...
		return (new ceres::AutoDiffCostFunction<Model0ReprojectionError, 2, 6, 2>(
			new Model0ReprojectionError(internal, observed_x, observed_y)));
	}

	Model0ReprojectionError(const array<double, 3> internal, double observed_x, double observed_y)
		: internal(internal), observed_x(observed_x), observed_y(observed_y) {}

	template <typename T>
	bool operator()(const T* const external, const T* const point, T* residuals) const {

        // Subtract observed coordinates
        bool r = model0_projection<T>(internal.data(), external, point, residuals);
        residuals[0] -= T(observed_x);
        residuals[1] -= T(observed_y);
        return r;
    }
};

struct Model0 {
    Model0(const ImageFeatures& f, const array<double, 3>& internal, double ps, array<double, 6> left_cam, array<double, 6> right_cam) :
            features(f),
            internal(internal),
            pixel_size(ps) {
        // Initialize from parent model
        // (for now hard coded left-right images)

        // Initialize cameras side by side
        cameras.push_back(left_cam);
        cameras.push_back(right_cam);

        terrain.resize(features.observations.size());

        // For each observation
        for (size_t i = 0; i < features.observations.size(); i++) {
            // Down project to z=0 to initialize terrain
            double dx_left, dy_left;
            double dx_right, dy_right;
            double elevation = 0.0;
            image_to_world(internal.data(), cameras[0].data(), &features.observations[i][0], &elevation, &dx_left, &dy_left);
            image_to_world(internal.data(), cameras[1].data(), &features.observations[i][2], &elevation, &dx_right, &dy_right);

            // Take average of both projections
            terrain[i] = {(dx_left + dx_right)/2.0, (dy_left + dy_right)/2.0};
        }
    }

    template <class Archive>
    void serialize(Archive& ar) {
        ar(cereal::make_nvp("cameras", cameras),
           cereal::make_nvp("terrain", terrain),
           cereal::make_nvp("internal", internal),
           cereal::make_nvp("pixel_size", pixel_size));
    }

    const ImageFeatures& features;

    // 3 dof internals: {f, ppx, ppy}
    const array<double, 3> internal;
    const double pixel_size;

    // Parameters
    vector< array<double, 6> > cameras; // 6 dof cameras
    vector< array<double, 2> > terrain; // 2 dof ground points on flat terrain
};


#endif

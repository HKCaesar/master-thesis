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

struct sensor_t;

struct pixel_t {
    pixel_t(double i, double j) : i(i), j(j) {}
    sensor_t to_sensor(double pixel_size, unsigned long rows, unsigned long cols) const;
    double i, j;
};

struct sensor_t {
    sensor_t(double x, double y) : x(x), y(y) {}
    pixel_t to_pixel(double pixel_size, unsigned long rows, unsigned long cols) const;
    double x, y;
};

inline sensor_t pixel_t::to_sensor(double pixel_size, unsigned long rows, unsigned long cols) const {
    return sensor_t{(j-cols/2)*pixel_size, (rows/2-i)*pixel_size};
}

inline pixel_t sensor_t::to_pixel(double pixel_size, unsigned long rows, unsigned long cols) const {
    return pixel_t{(rows/2-y)/pixel_size, (x+cols/2)/pixel_size};
}

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

struct Solution {
    vector<array<double, 6>> cameras; // 6 dof cameras
    vector<array<double, 2>> terrain; // 2 dof ground points on flat terrain

    template <class Archive>
    void serialize(Archive& ar) {
        ar(cereal::make_nvp("cameras", cameras),
           cereal::make_nvp("terrain", terrain));
    }
};

struct Model0 {
    Model0();
    void manual_setup(std::shared_ptr<ImageFeatures> f, const array<double, 3>& internal, double ps, array<double, 6> left_cam, array<double, 6> right_cam);
    void solve();

    template <class Archive>
    void serialize(Archive& ar) {
        ar(cereal::make_nvp("features", features),
           cereal::make_nvp("internal", internal),
           cereal::make_nvp("pixel_size", pixel_size),
           cereal::make_nvp("solutions", solutions),
           cereal::make_nvp("rows", rows),
           cereal::make_nvp("cols", cols));
    }

    std::shared_ptr<ImageFeatures> features;

    // 3 dof internals: {f, ppx, ppy}
    array<double, 3> internal;
    double pixel_size;
    double rows, cols;

    // List of solutions, from the initial guess (or parent model) to local optimum
    vector<Solution> solutions;
};

#endif

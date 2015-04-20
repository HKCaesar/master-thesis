#ifndef MODEL_ZERO_H
#define MODEL_ZERO_H

#include <iostream>
#include <vector>
#include <array>
#include "ceres/ceres.h"
#include "data_set.h"
#include "image_features.h"

using std::vector;
using std::array;

// This is just a test for setting up cython
void vector_print(double list[3]) {
    std::cout << list[0] << " " << list[1] << " " << list[2] << std::endl;
}

template <typename T>
Eigen::Matrix<T, 3, 3, Eigen::ColMajor> rotation_matrix(const T* external) {
    Eigen::Matrix<T, 3, 3, Eigen::ColMajor> Yaw, Pitch, Roll;
    Yaw << cos(external[5]), -sin(external[5]), T(0),
           sin(external[5]), cos(external[5]), T(0),
           T(0), T(0), T(1);
    Pitch << T(1), T(0), T(0),
             T(0), cos(external[4]), sin(external[4]),
             T(0), -sin(external[4]), cos(external[4]);
    Roll << -cos(external[3]), T(0), -sin(external[3]),
            T(0), T(1), T(0),
            sin(external[3]), T(0), -cos(external[3]);
	return Pitch * Roll * Yaw;
}

template <typename T>
bool model0_projection(
        const double* internal,
        const T* const external,
        const T* const point,
        T* residuals) {

    Eigen::Matrix<T, 3, 3, Eigen::ColMajor> R = rotation_matrix(external);

    // Translate and rotate to camera frame
    Eigen::Matrix<T, 3, 1, Eigen::ColMajor> Q;
    Q << point[0] - external[0], point[1] - external[1], external[2];
    Q = R*Q;

    // Normalized (pin-hole) coordinates
    T x = Q(0, 0) / Q(2, 0);
    T y = Q(1, 0) / Q(2, 0);

    // Apply focal length and principal point
    residuals[0] = internal[0] * x + internal[1];
    residuals[1] = internal[0] * y + internal[2];

    return true;
}

bool model0_projection_double(const double* internal, const double* external, const double* point, double* residuals) {
    return model0_projection<double>(internal, external, point, residuals);
}

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

// Downproject to fixed elevation
// internal size is 3: {f, ppx, ppy}
template <typename T>
void image_to_world(const T* const internal,
                    const T* const external,
                    const T* pix,
                    const T* elevation,
                    T* dx, T* dy) {
    // Rotation matrices
    Eigen::Matrix<T, 4, 4, Eigen::ColMajor> Yaw1, Pitch1, Roll1, T1;
    Eigen::Matrix<T, 4, 3, Eigen::ColMajor> B;
    Eigen::Matrix<T, 3, 4, Eigen::ColMajor> P;
    Yaw1 << cos(external[5]), -sin(external[5]), T(0), T(0),
        sin(external[5]), cos(external[5]), T(0), T(0),
        T(0), T(0), T(1), T(0),
        T(0), T(0), T(0), T(1);

    Pitch1 << T(1), T(0), T(0), T(0),
        T(0), cos(external[4]), sin(external[4]), T(0),
        T(0), -sin(external[4]), cos(external[4]), T(0),
        T(0), T(0), T(0), T(1);

    Roll1 << -cos(external[3]), T(0), -sin(external[3]), T(0),
        T(0), T(1), T(0), T(0),
        sin(external[3]), T(0), -cos(external[3]), T(0),
        T(0), T(0), T(0), T(1);

    Eigen::Matrix<T, 4, 4, Eigen::ColMajor> R1 = Pitch1 * Roll1 * Yaw1;
    T1 << T(1), T(0), T(0), -external[0],
        T(0), T(1), T(0), -external[1],
        T(0), T(0), T(-1), external[2],
        T(0), T(0), T(0), T(1);
    B << T(1), T(0), T(0),
        T(0), T(1), T(0),
        T(0), T(0), elevation[0],
        T(0), T(0), T(1);

    P << internal[0], T(0), internal[1], T(0),
        T(0), internal[0], internal[2], T(0),
        T(0), T(0), T(1), T(0);

    Eigen::Matrix<T, 3, 3, Eigen::ColMajor> A = P * R1* T1*B;
    Eigen::Matrix<T, 3, 1, Eigen::ColMajor> b;
    b << T(pix[0]), T(pix[1]), T(1);
    Eigen::FullPivLU< Eigen::Matrix<T, 3, 3, Eigen::ColMajor> >  lu(A);
    Eigen::Matrix<T, 3, 1, Eigen::ColMajor> sol = lu.solve(b);

    *dx = sol(0, 0) / sol(2, 0);
	*dy = sol(1, 0) / sol(2, 0);
}

void model0_image_to_world_double(const double* const internal,
                                  const double* const external,
                                  const double* pix,
                                  const double* elevation,
                                  double* dx, double* dy) {
    image_to_world<double>(internal, external, pix, elevation, dx, dy);
}

// struct Model0 : public Model {
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

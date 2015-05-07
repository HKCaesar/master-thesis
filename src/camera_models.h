#ifndef CAMERA_MODELS_H
#define CAMERA_MODELS_H

#include "ceres/ceres.h"
#include "internal.h"

// Could use quaternions to implement the above function
// Same inputs, compute Rotatino matrix using quaternions instead
// This uses two template parameters to allow the external pointer to be double* explicitly,
// in case it is not a parameter, and therefore does not need to be differentiated over by Jets
template <typename T, typename ExternalT>
Eigen::Matrix<T, 3, 3, Eigen::ColMajor> rotation_matrix_3(const ExternalT* ext) {
    Eigen::Matrix<T, 3, 3, Eigen::ColMajor> Yaw, Pitch, Roll;

    Yaw   <<  cos(T(ext[5])), -sin(T(ext[5])),            T(0),
              sin(T(ext[5])),  cos(T(ext[5])),            T(0),
                        T(0),            T(0),            T(1);

    Pitch <<            T(1),            T(0),            T(0),
                        T(0),  cos(T(ext[4])),  sin(T(ext[4])),
                        T(0), -sin(T(ext[4])),  cos(T(ext[4]));

    Roll  << -cos(T(ext[3])),            T(0), -sin(T(ext[3])),
                        T(0),            T(1),            T(0),
              sin(T(ext[3])),            T(0), -cos(T(ext[3]));

	return Pitch*Roll*Yaw;
}

template <typename T>
Eigen::Matrix<T, 4, 4, Eigen::ColMajor> rotation_matrix_4(const T* ext) {
    Eigen::Matrix<T, 4, 4, Eigen::ColMajor> Yaw, Pitch, Roll;

    Yaw   <<  cos(ext[5]), -sin(ext[5]),         T(0), T(0),
              sin(ext[5]),  cos(ext[5]),         T(0), T(0),
                     T(0),         T(0),         T(1), T(0),
                     T(0),         T(0),         T(0), T(1);

    Pitch <<         T(1),         T(0),         T(0), T(0),
                     T(0),  cos(ext[4]),  sin(ext[4]), T(0),
                     T(0), -sin(ext[4]),  cos(ext[4]), T(0),
                     T(0),         T(0),         T(0), T(1);

    Roll  << -cos(ext[3]),         T(0), -sin(ext[3]), T(0),
                     T(0),         T(1),         T(0), T(0),
              sin(ext[3]),         T(0), -cos(ext[3]), T(0),
                     T(0),         T(0),         T(0), T(1);

    return Pitch*Roll*Yaw;
}

// This uses two template parameters to allow some blocks to be double* explicitly,
// in case they are not parameters, and therefore do not need to be differentiated over by Jets
// T is the usual ceres magic, while additional types should be specified at the call site to be
// either T or double depending on wheter the function parameter is an optimization parameter or not
// if in doubt, try calling <T, T> or <T, double> until it compiles
template <typename T, typename ExternalT>
bool model0_projection(
        const double* internal,
        const ExternalT* const external,
        const T* const point,
        T* residuals) {

    Eigen::Matrix<T, 3, 3, Eigen::ColMajor> R = rotation_matrix_3<T, ExternalT>(external);

    // Translate and rotate to camera frame
    Eigen::Matrix<T, 3, 1, Eigen::ColMajor> Q;
    Q << point[0] - T(external[0]), point[1] - T(external[1]), T(external[2]);
    Q = R*Q;

    // Normalized (pin-hole) coordinates
    T x = Q(0, 0) / Q(2, 0);
    T y = Q(1, 0) / Q(2, 0);

    // Apply focal length and principal point
    residuals[0] = focal_length(internal) * x + pp_x(internal);
    residuals[1] = focal_length(internal) * y + pp_y(internal);

    return true;
}

// Instanciate the templated version for cython export
inline bool model0_projection_double(const double* internal, const double* external, const double* point, double* residuals) {
    return model0_projection<double>(internal, external, point, residuals);
}

// Downproject to fixed elevation
// internal size is 3: {f, ppx, ppy}
template <typename T>
void image_to_world(const T* const internal,
                    const T* const external,
                    const T* pix,
                    const T* elevation,
                    T* dx, T* dy) {
    // Rotation matrices
    Eigen::Matrix<T, 4, 4, Eigen::ColMajor> T1;
    Eigen::Matrix<T, 4, 3, Eigen::ColMajor> B;
    Eigen::Matrix<T, 3, 4, Eigen::ColMajor> P;

    Eigen::Matrix<T, 4, 4, Eigen::ColMajor> R1 = rotation_matrix_4(external);

    T1 << T(1), T(0), T( 0), -external[0],
          T(0), T(1), T( 0), -external[1],
          T(0), T(0), T(-1),  external[2],
          T(0), T(0), T( 0),         T(1);

    B  << T(1), T(0),         T(0),
          T(0), T(1),         T(0),
          T(0), T(0), elevation[0],
          T(0), T(0),         T(1);

    P  << focal_length(internal),                   T(0), pp_x(internal), T(0),
                            T(0), focal_length(internal), pp_y(internal), T(0),
                            T(0),                   T(0),           T(1), T(0);

    Eigen::Matrix<T, 3, 3, Eigen::ColMajor> A = P * R1 * T1 * B;
    Eigen::Matrix<T, 3, 1, Eigen::ColMajor> b;
    b << T(pix[0]), T(pix[1]), T(1);
    Eigen::FullPivLU< Eigen::Matrix<T, 3, 3, Eigen::ColMajor> >  lu(A);
    Eigen::Matrix<T, 3, 1, Eigen::ColMajor> sol = lu.solve(b);

    *dx = sol(0, 0) / sol(2, 0);
	*dy = sol(1, 0) / sol(2, 0);
}

// Instanciate the templated version for cython export
inline void model0_image_to_world_double(const double* const internal,
                                  const double* const external,
                                  const double* pix,
                                  const double* elevation,
                                  double* dx, double* dy) {
    image_to_world<double>(internal, external, pix, elevation, dx, dy);
}

#endif

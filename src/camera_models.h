#ifndef CAMERA_MODELS_H
#define CAMERA_MODELS_H

#include "ceres/ceres.h"

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
// Could use quaternions to implement the above function
// Same inputs, compute Rotatino matrix using quaternions instead

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

// Instanciate the templated version for cython export
inline void model0_image_to_world_double(const double* const internal,
                                  const double* const external,
                                  const double* pix,
                                  const double* elevation,
                                  double* dx, double* dy) {
    image_to_world<double>(internal, external, pix, elevation, dx, dy);
}

#endif

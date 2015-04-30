#ifndef INTERNAL_H
#define INTERNAL_H

#include <array>

typedef std::array<double, 3> internal_t;

// Templated for use in ceres
template <typename T>
inline T& focal_length(T* data) {return data[0];}
inline double& focal_length(internal_t& data) {return data[0];}

template <typename T>
inline T& pp_x(T* data) {return data[1];}
inline double& pp_x(internal_t& data) {return data[1];}

template <typename T>
inline T& pp_y(T* data) {return data[2];}
inline double& pp_y(internal_t& data) {return data[2];}

// template <typename T>
// T& pixel_size(T* data) {return data[3];}

#endif

#ifndef TYPES_HPP
#define TYPES_HPP

#include <cereal/types/vector.hpp>
#include <cereal/access.hpp>

struct sensor_t;

struct pixel_t {
    pixel_t() : i(0.0), j(0.0) {}
    pixel_t(double i, double j) : i(i), j(j) {}
    sensor_t to_sensor(double pixel_size, unsigned long rows, unsigned long cols) const;
    double i, j;
};

struct sensor_t {
    sensor_t() : x(0.0), y(0.0) {}
    sensor_t(double x, double y) : x(x), y(y) {}
    pixel_t to_pixel(double pixel_size, unsigned long rows, unsigned long cols) const;
    double x, y;
};

inline sensor_t pixel_t::to_sensor(double pixel_size, unsigned long rows, unsigned long cols) const {
    return sensor_t((j-cols/2)*pixel_size, (rows/2-i)*pixel_size);
}

inline pixel_t sensor_t::to_pixel(double pixel_size, unsigned long rows, unsigned long cols) const {
    return pixel_t(rows/2 - y/pixel_size, x/pixel_size + cols/2);
}

namespace cereal {
    template <class Archive>
    void save(Archive& ar, const pixel_t& p) {
        ar(cereal::make_size_tag(static_cast<cereal::size_type>(2)));
        ar(p.i, p.j);
    }

    template <class Archive>
    void load(Archive& ar, pixel_t& p) {
        cereal::size_type size;
        ar(cereal::make_size_tag(size));
        if (static_cast<std::size_t>(size) != 2) {
            throw std::runtime_error("Error loading pixel_t from JSON: incorrect size.");
        }
        ar(p.i, p.j);
    }
}

#endif

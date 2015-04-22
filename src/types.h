#ifndef TYPES_HPP
#define TYPES_HPP

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
    return sensor_t((j-cols/2)*pixel_size, (rows/2-i)*pixel_size);
}

inline pixel_t sensor_t::to_pixel(double pixel_size, unsigned long rows, unsigned long cols) const {
    return pixel_t(rows/2 - y/pixel_size, x/pixel_size + cols/2);
}

#endif

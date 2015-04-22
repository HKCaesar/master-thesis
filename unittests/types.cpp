#include "gtest/gtest.h"
#include "../src/types.h"

#define pixel_abs_error 1e-12
#define sensor_abs_error 1e-12

#define EXPECT_PIXEL_NEAR(a, b) \
    EXPECT_NEAR((a).i, (b).i, pixel_abs_error); \
    EXPECT_NEAR((a).j, (b).j, pixel_abs_error);

#define EXPECT_SENSOR_NEAR(a, b) \
    EXPECT_NEAR((a).x, (b).x, sensor_abs_error); \
    EXPECT_NEAR((a).y, (b).y, sensor_abs_error);

TEST(PixelToSensor, Invertible) {
    double ps = 5e-6;
    double rows = 2000;
    double cols = 4000;

    pixel_t p(0, 0);
    EXPECT_PIXEL_NEAR(p, p.to_sensor(ps, rows, cols).to_pixel(ps, rows, cols));

    p = {31.456, 40.5};
    EXPECT_PIXEL_NEAR(p, p.to_sensor(ps, rows, cols).to_pixel(ps, rows, cols));

    p = {1000, 2000};
    EXPECT_PIXEL_NEAR(p, p.to_sensor(ps, rows, cols).to_pixel(ps, rows, cols));

    p = {2000, 4000};
    EXPECT_PIXEL_NEAR(p, p.to_sensor(ps, rows, cols).to_pixel(ps, rows, cols));

    sensor_t s(0, 0);
    s = {0, 0};
    EXPECT_SENSOR_NEAR(s, s.to_pixel(ps, rows, cols).to_sensor(ps, rows, cols));

    s = {-0.001, -0.001};
    EXPECT_SENSOR_NEAR(s, s.to_pixel(ps, rows, cols).to_sensor(ps, rows, cols));

    s = {2.5, 0.00698};
    EXPECT_SENSOR_NEAR(s, s.to_pixel(ps, rows, cols).to_sensor(ps, rows, cols));

    s = {-45, 0};
    EXPECT_SENSOR_NEAR(s, s.to_pixel(ps, rows, cols).to_sensor(ps, rows, cols));
}

TEST(PixelToSensor, ExpectedValues) {
    double ps = 0.01;
    double rows = 2000;
    double cols = 4000;
    pixel_t p(0, 0);
    sensor_t s(0, 0);

    p = {1000, 2000};
    s = {0, 0};
    EXPECT_SENSOR_NEAR(p.to_sensor(ps, rows, cols), s);
    EXPECT_PIXEL_NEAR(p, s.to_pixel(ps, rows, cols));

    p = {2000, 4000};
    s = {20, -10};
    EXPECT_SENSOR_NEAR(p.to_sensor(ps, rows, cols), s);
    EXPECT_PIXEL_NEAR(p, s.to_pixel(ps, rows, cols));

    p = {0, 0};
    s = {-20, 10};
    EXPECT_SENSOR_NEAR(p.to_sensor(ps, rows, cols), s);
    EXPECT_PIXEL_NEAR(p, s.to_pixel(ps, rows, cols));
}

GTEST_API_ int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

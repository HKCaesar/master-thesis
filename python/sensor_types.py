import numpy as np

def pixel_to_sensor(pixels, pixel_size, rows, cols):
    i, j = pixels[:,0], pixels[:,1]
    return np.column_stack(((j-cols/2) * pixel_size, (rows/2-i) * pixel_size))

def sensor_to_pixel(sensors, pixel_size, rows, cols):
    x, y = sensors[:,0], sensors[:,1]
    return np.column_stack((rows/2 - y/pixel_size, x/pixel_size + cols/2))

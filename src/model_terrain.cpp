#include <memory>
#include "model_terrain.h"

// TODO also use in Model0
// Inverse the features of a given features match at given elevation
// Returns the average of the two ground points
vector<array<double, 3>> inverse_features_average(const obs_pair& edge, const double elevation,
        const double pixel_size,
        const double rows,
        const double cols,
        const internal_t& internal,
        const vector<array<double, 6>>& cameras) {
    
    vector<array<double, 3>> points;
    // For each observation
    for (size_t i = 0; i < edge.obs_a.size(); i++) {
        // Down project to elevation
        sensor_t sens_a = edge.obs_a[i].to_sensor(pixel_size, rows, cols);
        sensor_t sens_b = edge.obs_b[i].to_sensor(pixel_size, rows, cols);

        double dx_a, dy_a;
        double dx_b, dy_b;
        double pix_a[2] = {sens_a.x, sens_a.y};
        double pix_b[2] = {sens_b.x, sens_b.y};
        image_to_world(internal.data(), cameras[edge.cam_a].data(), pix_a, &elevation, &dx_a, &dy_a);
        image_to_world(internal.data(), cameras[edge.cam_b].data(), pix_b, &elevation, &dx_b, &dy_b);

        // Take average of both projections
        points[i] = {(dx_a + dx_b)/2.0, (dy_a + dy_b)/2.0, elevation};
    }
    return points;
}

void ModelTerrain::solve() {
    if (!parent) {
        throw("Solving ModelTerrain but no parent provided.");
    }

    // Initialize cameras from parent
    cameras = parent->final_external();

    const double rows = features->data_set->rows;
    const double cols = features->data_set->cols;

    // Initialize solution[0].terrain by inversing features
    // For now only use two cams (= only edges[0])
    solution sol{inverse_features_average(features->edges[0],
            0.0, // initial elevation
            pixel_size,
            rows,
            cols,
            internal,
            cameras)};
    solutions.push_back(sol);

    // The working solution is the one holding ceres' parameter blocks
    ModelTerrain::solution working_solution(solutions[0]);

    // Setup parameter and residual blocks
    const obs_pair& edge = features->edges[0];
    for (size_t i = 0; i < edge.obs_a.size(); i++) {
        // Residual for left cam
        sensor_t obs_left = edge.obs_a[i].to_sensor(pixel_size, rows, cols);
		ceres::CostFunction* cost_function_left = ModelTerrainReprojectionError::make(internal, cameras[edge.cam_a], obs_left);
		problem.AddResidualBlock(cost_function_left,
			NULL,
			working_solution.terrain[i].data()
			);

        // Residual for right cam
        sensor_t obs_right = edge.obs_b[i].to_sensor(pixel_size, rows, cols);
		ceres::CostFunction* cost_function_right = ModelTerrainReprojectionError::make(internal, cameras[edge.cam_b], obs_right);
		problem.AddResidualBlock(cost_function_right,
			NULL,
			working_solution.terrain[i].data()
			);
    }

    enable_logging(solutions, working_solution);

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << "\n";
}

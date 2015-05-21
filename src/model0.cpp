#include <memory>
#include "model0.h"

ceres::Solver::Summary Model0::solve() {
    ceres::Problem problem;
    // Note: model0 only works with 2 cams, so will only consider the first edge
    obs_pair& edge = features->edges[0];

    // Initialize terrain by down projecting features
    // Right now solutions[0] and solutions[0].cameras is created in base() setup
    // in other models, it is initialized from parent model in Model::solve()
    solutions[0].terrain.resize(edge.obs_a.size());

    double rows = features->data_set->rows;
    double cols = features->data_set->cols;

    // For each observation
    for (size_t i = 0; i < edge.obs_a.size(); i++) {
        // Down project to z=0 to initialize terrain
        sensor_t sens_a = edge.obs_a[i].to_sensor(pixel_size(internal), rows, cols);
        sensor_t sens_b = edge.obs_b[i].to_sensor(pixel_size(internal), rows, cols);

        double dx_a, dy_a;
        double dx_b, dy_b;
        double elevation = 0.0;

        double pix_a[2] = {sens_a.x, sens_a.y};
        double pix_b[2] = {sens_b.x, sens_b.y};
        image_to_world(internal.data(), solutions[0].cameras[0].data(), pix_a, &elevation, &dx_a, &dy_a);
        image_to_world(internal.data(), solutions[0].cameras[1].data(), pix_b, &elevation, &dx_b, &dy_b);

        // Take average of both projections
        solutions[0].terrain[i] = {(dx_a + dx_b)/2.0, (dy_a + dy_b)/2.0};
    }

    // The working solution is the one holding ceres' parameter blocks
    Model0::solution working_solution(solutions[0]);

    // Setup parameter and residual blocks
    for (size_t i = 0; i < edge.obs_a.size(); i++) {
        // Residual for left cam
        sensor_t obs_left = edge.obs_a[i].to_sensor(pixel_size(internal), rows, cols);
		ceres::CostFunction* cost_function_left = Model0ReprojectionError::make(internal, obs_left);
		problem.AddResidualBlock(cost_function_left,
			NULL,
			working_solution.cameras[0].data(),
			working_solution.terrain[i].data()
			);

        // Residual for right cam
        sensor_t obs_right = edge.obs_b[i].to_sensor(pixel_size(internal), rows, cols);
		ceres::CostFunction* cost_function_right = Model0ReprojectionError::make(internal, obs_right);
		problem.AddResidualBlock(cost_function_right,
			NULL,
			working_solution.cameras[1].data(),
			working_solution.terrain[i].data()
			);
    }

    problem.SetParameterBlockConstant(working_solution.cameras[0].data());

    enable_logging(solutions, working_solution);
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    return summary;
}

internal_t Model0::final_internal() const {
    return internal;
}

vector<array<double, 6>> Model0::final_external() const {
    return solutions.back().cameras;
}

vector<array<double, 3>> Model0::final_terrain() const {
    const vector<array<double, 2>>& terrain = solutions.back().terrain;
    vector<array<double, 3>> result(terrain.size());
    for (size_t i = 0; i < result.size(); i++) {
        result[i] = array<double, 3> {terrain[i][0], terrain[i][1], 0.0};
    }
    return result;
}


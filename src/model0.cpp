#include <memory>
#include "model0.h"

void Model0::solve() {
    // Verify features have been computed
    if (!features || features->edges.size() == 0 || features->computed == false) {
        throw std::runtime_error("Attempting to solve model0 but no observations are available");
    }

    // Note: model0 only works with 2 cams, so will only consider the first edge
    obs_pair& edge = features->edges[0];

    // Initialize terrain by down projecting features
    solutions[0].terrain.resize(edge.obs_a.size());

    double rows = features->data_set->rows;
    double cols = features->data_set->cols;

    // For each observation
    for (size_t i = 0; i < edge.obs_a.size(); i++) {
        // Down project to z=0 to initialize terrain
        sensor_t sens_a = edge.obs_a[i].to_sensor(pixel_size, rows, cols);
        sensor_t sens_b = edge.obs_b[i].to_sensor(pixel_size, rows, cols);

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
    ceres::Problem problem;
    for (size_t i = 0; i < edge.obs_a.size(); i++) {
        // Residual for left cam
        sensor_t obs_left = edge.obs_a[i].to_sensor(pixel_size, rows, cols);
		ceres::CostFunction* cost_function_left = Model0ReprojectionError::make(internal, obs_left);
		problem.AddResidualBlock(cost_function_left,
			NULL,
			working_solution.cameras[0].data(),
			working_solution.terrain[i].data()
			);

        // Residual for right cam
        sensor_t obs_right = edge.obs_b[i].to_sensor(pixel_size, rows, cols);
		ceres::CostFunction* cost_function_right = Model0ReprojectionError::make(internal, obs_right);
		problem.AddResidualBlock(cost_function_right,
			NULL,
			working_solution.cameras[1].data(),
			working_solution.terrain[i].data()
			);
    }

    problem.SetParameterBlockConstant(working_solution.cameras[0].data());

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    options.max_linear_solver_iterations = 3;
    options.max_num_iterations = 30;
    options.num_threads = 1;

    // Enable logging of solution at every step
    options.update_state_every_iteration = true;
    std::unique_ptr<LogSolutionCallback<Model0::solution>> solution_logger(
        new LogSolutionCallback<Model0::solution>(solutions, working_solution));
    options.callbacks.push_back(solution_logger.get());

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << "\n";
}

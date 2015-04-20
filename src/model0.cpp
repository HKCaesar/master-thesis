#include "model0.h"

Model0::Model0(const ImageFeatures& f, const array<double, 3>& internal, double ps, array<double, 6> left_cam, array<double, 6> right_cam) :
        features(f),
        internal(internal),
        pixel_size(ps) {
    // Initialize initial solution from parent model
    // (for now hard coded left-right images)

    // Initialize cameras side by side
    Solution init;
    init.cameras.push_back(left_cam);
    init.cameras.push_back(right_cam);

    init.terrain.resize(features.observations.size());

    // For each observation
    for (size_t i = 0; i < features.observations.size(); i++) {
        // Down project to z=0 to initialize terrain
        double dx_left, dy_left;
        double dx_right, dy_right;
        double elevation = 0.0;
        image_to_world(internal.data(), init.cameras[0].data(), &features.observations[i][0], &elevation, &dx_left, &dy_left);
        image_to_world(internal.data(), init.cameras[1].data(), &features.observations[i][2], &elevation, &dx_right, &dy_right);

        // Take average of both projections
        init.terrain[i] = {(dx_left + dx_right)/2.0, (dy_left + dy_right)/2.0};
        std::cout << (dx_left + dx_right)/2.0 << " " << (dy_left + dy_right)/2.0 << std::endl;
    }

    solutions.push_back(init);
}

void Model0::solve() {
    // The working solution is the one holding ceres' parameter blocks
    Solution working_solution(solutions[0]);

    // Setup parameter and residual blocks
    ceres::Problem problem;
    for (size_t i = 0; i < features.observations.size(); i++) {
        // Residual for left cam
		ceres::CostFunction* cost_function_left =
            Model0ReprojectionError::create(internal, pixel_size*features.observations[i][0], pixel_size*features.observations[i][1]);
		problem.AddResidualBlock(cost_function_left,
			NULL,
			working_solution.cameras[0].data(),
			working_solution.terrain[i].data()
			);

        // Residual for right cam
		ceres::CostFunction* cost_function_right =
            Model0ReprojectionError::create(internal, pixel_size*features.observations[i][2], pixel_size*features.observations[i][3]);
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

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << "\n";

    solutions.push_back(working_solution);
}

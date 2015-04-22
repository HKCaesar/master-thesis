#include <memory>
#include "model0.h"

Model0::Model0() : pixel_size(0.0), rows(0.0), cols(0.0) {
}

void Model0::manual_setup(std::shared_ptr<ImageFeatures> f, const array<double, 3>& intern, double ps, array<double, 6> left_cam, array<double, 6> right_cam) {
    features = f;
    internal = intern;
    pixel_size = ps;

    // Initialize initial solution from parent model
    // (for now hard coded left-right images)

    // Initialize cameras side by side
    Solution init;
    init.cameras.push_back(left_cam);
    init.cameras.push_back(right_cam);
    solutions.push_back(init);

}

class LogSolutionCallback : public ceres::IterationCallback {
    Model0& model;
    const Solution& working_solution;
public:
    LogSolutionCallback(Model0& m, const Solution& w) : model(m), working_solution(w) {}
    virtual ~LogSolutionCallback() {}
    virtual ceres::CallbackReturnType operator()(const ceres::IterationSummary&) {
        model.solutions.push_back(working_solution);
        return ceres::SOLVER_CONTINUE;
    }
};

void Model0::solve() {
    // Verify features have been computed
    if (!features || features->observations.size() == 0) {
        throw std::runtime_error("Attempting to solve model0 but no observations are available");
    }

    // Initialize terrain by down projecting features
    solutions[0].terrain.resize(features->observations.size());

    // For each observation
    for (size_t i = 0; i < features->observations.size(); i++) {
        // Down project to z=0 to initialize terrain
        double dx_left, dy_left;
        double dx_right, dy_right;
        double elevation = 0.0;
        double pix_x = pixel_size*features->observations[i][0];
        double pix_y = pixel_size*features->observations[i][1];
        image_to_world(internal.data(), solutions[0].cameras[0].data(), &pix_x, &elevation, &dx_left, &dy_left);
        image_to_world(internal.data(), solutions[0].cameras[1].data(), &pix_y, &elevation, &dx_right, &dy_right);

        // Take average of both projections
        solutions[0].terrain[i] = {(dx_left + dx_right)/2.0, (dy_left + dy_right)/2.0};
    }

    // The working solution is the one holding ceres' parameter blocks
    Solution working_solution(solutions[0]);

    // Setup parameter and residual blocks
    ceres::Problem problem;
    for (size_t i = 0; i < features->observations.size(); i++) {
        // Residual for left cam
        sensor_t obs_left = pixel_t(features->observations[i][0], features->observations[i][1]).to_sensor(pixel_size, rows, cols);
		ceres::CostFunction* cost_function_left = Model0ReprojectionError::create(internal, obs_left);
		problem.AddResidualBlock(cost_function_left,
			NULL,
			working_solution.cameras[0].data(),
			working_solution.terrain[i].data()
			);

        // Residual for right cam
        sensor_t obs_right = pixel_t(features->observations[i][2], features->observations[i][3]).to_sensor(pixel_size, rows, cols);
		ceres::CostFunction* cost_function_right = Model0ReprojectionError::create(internal, obs_right);
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
    std::shared_ptr<LogSolutionCallback> solution_logger(new LogSolutionCallback(*this, working_solution));
    options.callbacks.push_back(solution_logger.get());

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << "\n";
}

#include <iostream>
#include <fstream>
#include <vector>
#include <array>
#include <memory>
#include <string>

using std::tuple;
using std::make_tuple;
using std::vector;
using std::array;
using std::string;
using std::shared_ptr;

#include <cereal/archives/json.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/array.hpp>

#include "camera_models.h"
#include "data_set.h"
#include "image_features.h"

// struct Model0 : public Model {
struct Model0 {
    Model0(const ImageFeatures& f, const array<double, 3>& internal, double ps, array<double, 6> left_cam, array<double, 6> right_cam) :
            features(f),
            internal(internal),
            pixel_size(ps) {
        // Initialize from parent model
        // (for now hard coded left-right images)

        // Initialize cameras side by side
        cameras.push_back(left_cam);
        cameras.push_back(right_cam);

        terrain.resize(features.observations.size());

        // For each observation
        for (size_t i = 0; i < features.observations.size(); i++) {
            // Down project to z=0 to initialize terrain
            double dx_left, dy_left;
            double dx_right, dy_right;
            double elevation = 0.0;
            image_to_world(internal.data(), cameras[0].data(), &features.observations[i][0], &elevation, &dx_left, &dy_left);
            image_to_world(internal.data(), cameras[1].data(), &features.observations[i][2], &elevation, &dx_right, &dy_right);

            // Take average of both projections
            terrain[i] = {(dx_left + dx_right)/2.0, (dy_left + dy_right)/2.0};
        }
    }

    template <class Archive>
    void serialize(Archive& ar) {
        ar(cereal::make_nvp("cameras", cameras),
           cereal::make_nvp("terrain", terrain),
           cereal::make_nvp("internal", internal),
           cereal::make_nvp("pixel_size", pixel_size));
    }

    const ImageFeatures& features;

    // 3 dof internals: {f, ppx, ppy}
    const array<double, 3> internal;
    const double pixel_size;

    // Parameters
    vector< array<double, 6> > cameras; // 6 dof cameras
    vector< array<double, 2> > terrain; // 2 dof ground points on flat terrain
};

// Overload std::array for JSON to use []
namespace cereal {
    template <std::size_t N>
    void save(cereal::JSONOutputArchive& archive, std::array<double, N> const& list) {
        archive(cereal::make_size_tag(static_cast<cereal::size_type>(list.size())));
        for (auto && v : list) {
            archive(v);
        }
    }
}

int main(int argc, char* argv[]) {
    google::InitGoogleLogging(argv[0]);

    if (argc < 3) {
        std::cerr << "Usage: ./model0 data_dir results_dir" << std::endl;
        return -1;
    }

    string data_root = std::string(argv[1]);
    string results_dir = std::string(argv[2]) + "/features_analysis";

    // Test problem for model0
    // This problem definition code could be moved to a 'project file' or equivalent

    // Load images
    // defines filenames, pairwise order
    std::cout << "Loading images..." << std::endl;
    DataSet data_set;
    data_set.filenames.push_back(string("alinta-stockpile/DSC_5522.JPG"));
    data_set.filenames.push_back(string("alinta-stockpile/DSC_5521.JPG"));
    data_set.load(data_root);

    // Match features and obtain observations
    std::cout << "Matching features..." << std::endl;
    ImageFeatures features(data_set, 10);

    // Create model
    std::cout << "Setting up model..." << std::endl;
    const double pixel_size = 0.0085e-3;
    Model0 model(features, {48.3355e-3, 0.0093e-3, -0.0276e-3}, pixel_size, {0, 0, 269, 0, 0, 0}, {0, 0, 269, 0, 0, 0});

    // Setup solver
    std::cout << "Solving..." << std::endl;
    ceres::Problem problem;
    for (size_t i = 0; i < model.features.observations.size(); i++) {
        // Residual for left cam
		ceres::CostFunction* cost_function_left =
            Model0ReprojectionError::create(model.internal, pixel_size*model.features.observations[i][0], pixel_size*model.features.observations[i][1]);
		problem.AddResidualBlock(cost_function_left,
			NULL,
			model.cameras[0].data(),
			model.terrain[i].data()
			);

        // Residual for right cam
		ceres::CostFunction* cost_function_right =
            Model0ReprojectionError::create(model.internal, pixel_size*model.features.observations[i][2], pixel_size*model.features.observations[i][3]);
		problem.AddResidualBlock(cost_function_right,
			NULL,
			model.cameras[1].data(),
			model.terrain[i].data()
			);
    }

    problem.SetParameterBlockConstant(model.cameras[0].data());

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    options.max_linear_solver_iterations = 3;
    options.max_num_iterations = 30;
    options.num_threads = 1;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << "\n";

    std::cout << "==================" << std::endl;
    cereal::JSONOutputArchive sum(std::cout);
    sum(model.cameras[0]);
    sum(model.cameras[1]);

    // Serialize model with cereal
    std::ofstream ofs("model0.json");
    cereal::JSONOutputArchive output(ofs);
    output(
        cereal::make_nvp("data", data_set),
        cereal::make_nvp("features", features),
        cereal::make_nvp("model", model)
    );
}

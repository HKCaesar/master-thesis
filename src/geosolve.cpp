#include <iostream>
#include <fstream>
#include <vector>
#include <array>
#include <memory>
#include <string>
#include <map>
#include <functional>

#include <cereal/archives/json.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/array.hpp>
#include <cereal/types/memory.hpp>

#include "ceres/ceres.h"

#include "project.h"
#include "model0.h"
#include "model_terrain.h"
#include "bootstrap.h"

using std::tuple;
using std::make_tuple;
using std::vector;
using std::array;
using std::string;
using std::shared_ptr;
using namespace std::placeholders;

Project base_model0_project() {
    Project project;

    project.data_set = std::shared_ptr<DataSet>(new DataSet());
    project.data_set->filenames.push_back(string("alinta-stockpile/DSC_5522.JPG"));
    project.data_set->filenames.push_back(string("alinta-stockpile/DSC_5521.JPG"));
    project.data_set->rows = 2832;
    project.data_set->cols = 4256;

    std::shared_ptr<FeaturesGraph> feat(new FeaturesGraph());
    feat->data_set = project.data_set;
    feat->number_of_matches = 10;
    feat->add_edge(0, 1);
    project.features_list.push_back(feat);

    std::shared_ptr<Model0> model(new Model0());

    // Initialize initial solution from parent model
    // (for now hard coded left-right images)
    model->features = feat;
    model->internal = {48.3355e-3, 0.0093e-3, -0.0276e-3, 0.0085e-3};
    Model0::solution init;
    init.cameras.push_back({0, 0, 269, 0, 0, 0});
    init.cameras.push_back({0, 0, 269, 0, 0, 0});
    model->solutions.push_back(init);

    project.models.push_back(model);

    return project;
}

// Add a terrain only model using the previous last model as parent
void add_model_terrain(Project& project) {
    std::shared_ptr<ModelTerrain> model(new ModelTerrain());
    // Set parent to latest model
    model->parent = project.models.back();

    // High density features
    std::shared_ptr<FeaturesGraph> feat(new FeaturesGraph());
    feat->data_set = project.data_set;
    feat->number_of_matches = 500;
    feat->compute_scale = 0.25;
    feat->add_edge(0, 1);
    project.features_list.push_back(feat);

    model->features = feat;
    project.models.push_back(model);
}

// geosolve commands

void base_model0(const string&, const string& project_dir) {
    Project project = base_model0_project();
    project.to_file(project_dir + "/project.json");
}

void base_model0_200(const string&, const string& project_dir) {
    Project project = base_model0_project();
    project.features_list[0]->number_of_matches = 200;
    project.to_file(project_dir + "/project.json");
}

void base_model0_scale(const string&, const string& project_dir, double compute_scale) {
    Project project = base_model0_project();
    project.features_list[0]->compute_scale = compute_scale;
    project.to_file(project_dir + "/project.json");
}

void load_test(const string&, const string& project_dir) {
    Project project = Project::from_file(project_dir + "/project.json");
    project.to_file(project_dir + "/loadtest-output.json");
}

void features(const string& data_dir, const string& project_dir) {
    string project_filename = project_dir + "/project.json";
    Project project = Project::from_file(project_filename);
    for (auto& feat : project.features_list) {
        if (feat->computed) {
            std::cout << "Features already computed, skipping" << std::endl;
        } else {
            std::cout << "Computing features" << std::endl;
            feat->compute(data_dir);
        }
    }
    project.to_file(project_filename);
}

void solve(const string&, const string& project_dir) {
    string project_filename = project_dir + "/project.json";
    Project project = Project::from_file(project_filename);
    for (auto& model : project.models) {
        // If model hasn't been solved yet
        if (model->solved) {
            std::cout << "Model already solved, skipping" << std::endl;
        } else {
            std::cout << "Solving..." << std::endl;
            // Verify features have been computed
            if (!model->features || model->features->edges.size() == 0 || model->features->computed == false) {
                throw std::runtime_error("Attempting to solve model but no observations are available");
            }
            ceres::Solver::Summary summary = model->solve();
            std::cout << summary.FullReport() << "\n";
            model->solved = true;
        }
    }
    project.to_file(project_filename);
}

void model_terrain(const string&, const string& project_dir) {
    string project_filename = project_dir + "/project.json";
    Project project = Project::from_file(project_filename);
    std::cout << "Adding Model Terrain to existing project file" << std::endl;
    add_model_terrain(project);
    project.to_file(project_filename);
}

void bootstrap(const string&, const string& project_dir) {
    string project_filename = project_dir + "/project.json";
    Project project = Project::from_file(project_filename);
    std::cout << "Bootstraping all models" << std::endl;
    for (auto& model : project.models) {
        // TODO how to choose bootstrap parameters here?
        std::shared_ptr<Bootstrap> boot(new Bootstrap());
        boot->base_model = model;
        boot->number_of_samples = 5;
        boot->size_of_samples = 10;
        boot->solve();
        project.bootstraps.push_back(boot);
    }
    project.to_file(project_filename);
}

void help(const string&, const string&, const std::map<string, std::function<void (const string&, const string&)>>& commands) {
    for (auto& it : commands) {
        std::cout << it.first << std::endl;
    }
}

int main(int argc, char* argv[]) {
    google::InitGoogleLogging(argv[0]);

    if (argc < 4) {
        std::cerr << "Usage: ./geosolve <data_dir> <project_dir> command" << std::endl;
        return -1;
    }

    string data_dir(argv[1]);
    string project_dir(argv[2]);
    string command(argv[3]);

    std::map<string, std::function<void (const string&, const string&)>> commands {
        {"base_model0", base_model0},
        {"base_model0_200", base_model0_200},
        {"base_model0_half", std::bind(base_model0_scale, _1, _2, 0.5)},
        {"base_model0_quarter", std::bind(base_model0_scale, _1, _2, 0.25)},
        {"model_terrain", model_terrain},
        {"loadtest", load_test},
        {"features", features},
        {"solve", solve},
        {"bootstrap", bootstrap}
    };

    commands[string("help")] = std::bind(help, _1, _2, commands);

    try {
        commands.at(command)(data_dir, project_dir);
    } catch (std::out_of_range& err) {
        std::cerr << "Invalid command: " << command << std::endl;
        return -1;
    }
}

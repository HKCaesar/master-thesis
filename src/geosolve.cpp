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

#include "data_set.h"
#include "image_features.h"
#include "model0.h"

using std::tuple;
using std::make_tuple;
using std::vector;
using std::array;
using std::string;
using std::shared_ptr;
using namespace std::placeholders;

// Overload std::array for JSON to use []
namespace cereal {
    template <std::size_t N>
    void save(cereal::JSONOutputArchive& archive, std::array<double, N> const& list) {
        archive(cereal::make_size_tag(static_cast<cereal::size_type>(list.size())));
        for (auto && v : list) {
            archive(v);
        }
    }

    template <std::size_t N>
    void load(cereal::JSONInputArchive& archive, std::array<double, N>& list) {
        cereal::size_type size;
        archive(cereal::make_size_tag(size));
        if (static_cast<std::size_t>(size) != N) {
            throw std::runtime_error("Error loading std::array from JSON: incorrect size.");
        }
        for (auto && v : list) {
            archive(v);
        }
    }
}

struct Project {
    std::shared_ptr<DataSet> data_set;
    std::shared_ptr<FeaturesGraph> features;
    std::shared_ptr<Model> model;

    static Project from_file(const std::string& filename) {
        Project p;
        std::ifstream ifs(filename);
        if (!ifs.good()) {
            throw std::runtime_error("Can't open " + filename);
        }
        cereal::JSONInputArchive ar(ifs);
        p.serialize(ar);
        return p;
    }

    void to_file(const std::string& filename) {
        std::ofstream ofs(filename);
        cereal::JSONOutputArchive ar(ofs);
        this->serialize(ar);
    }

    template <class Archive>
    void serialize(Archive& ar) {
        ar(cereal::make_nvp("data_set", data_set),
           cereal::make_nvp("features", features),
           cereal::make_nvp("model", model));
    }
};

Project base_model0_project() {
    Project project;

    project.data_set = std::shared_ptr<DataSet>(new DataSet());
    project.data_set->filenames.push_back(string("alinta-stockpile/DSC_5522.JPG"));
    project.data_set->filenames.push_back(string("alinta-stockpile/DSC_5521.JPG"));
    project.data_set->rows = 2832;
    project.data_set->cols = 4256;

    project.features = std::shared_ptr<FeaturesGraph>(new FeaturesGraph());
    project.features->data_set = project.data_set;
    project.features->number_of_matches = 10;
    project.features->add_edge(0, 1);

    std::shared_ptr<Model0> model(new Model0());

    // Initialize initial solution from parent model
    // (for now hard coded left-right images)
    model->features = project.features;
    model->internal = {48.3355e-3, 0.0093e-3, -0.0276e-3};
    model->pixel_size = 0.0085e-3;
    Model0::solution init;
    init.cameras.push_back({0, 0, 269, 0, 0, 0});
    init.cameras.push_back({0, 0, 269, 0, 0, 0});
    model->solutions.push_back(init);

    project.model = model;

    return project;
}

void base_model0(const string&, const string& project_dir) {
    Project project = base_model0_project();
    project.to_file(project_dir + "/project.json");
}

void base_model0_200(const string&, const string& project_dir) {
    Project project = base_model0_project();
    project.features->number_of_matches = 200;
    project.to_file(project_dir + "/project.json");
}

void base_model0_scale(const string&, const string& project_dir, double compute_scale) {
    Project project = base_model0_project();
    project.features->compute_scale = compute_scale;
    project.to_file(project_dir + "/project.json");
}

void load_test(const string&, const string& project_dir) {
    Project project = Project::from_file(project_dir + "/project.json");
    project.to_file(project_dir + "/loadtest-output.json");
}

void features(const string& data_dir, const string& project_dir) {
    string project_filename = project_dir + "/project.json";
    Project project = Project::from_file(project_filename);
    std::cout << "Computing features" << std::endl;
    project.features->compute(data_dir);
    project.to_file(project_filename);
}

void solve(const string&, const string& project_dir) {
    string project_filename = project_dir + "/project.json";
    std::cout << "Solving..." << std::endl;
    Project project = Project::from_file(project_filename);
    project.model->solve();
    project.to_file(project_filename);
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
        {"loadtest", load_test},
        {"features", features},
        {"solve", solve}
    };

    try {
        commands.at(command)(data_dir, project_dir);
    } catch (std::out_of_range& err) {
        std::cerr << "Invalid command: " << command << std::endl;
        return -1;
    }
}

#include <iostream>
#include <fstream>
#include <vector>
#include <array>
#include <memory>
#include <string>

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
    std::shared_ptr<ImageFeatures> features;
    std::shared_ptr<Model0> model;

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

void base_model0(string filename) {
    Project project;

    project.data_set = std::shared_ptr<DataSet>(new DataSet());
    project.data_set->filenames.push_back(string("alinta-stockpile/DSC_5522.JPG"));
    project.data_set->filenames.push_back(string("alinta-stockpile/DSC_5521.JPG"));

    project.features = std::shared_ptr<ImageFeatures>(new ImageFeatures());
    project.features->data_set = project.data_set;
    project.features->maximum_number_of_matches = 10;

    project.model = std::shared_ptr<Model0>(new Model0());
    const double pixel_size = 0.0085e-3;
    project.model->manual_setup(project.features, {48.3355e-3, 0.0093e-3, -0.0276e-3}, pixel_size, {0, 0, 269, 0, 0, 0}, {0, 0, 269, 0, 0, 0});
    project.model->rows = 2832;
    project.model->cols = 4256;

    project.to_file(filename);
}

int main(int argc, char* argv[]) {
    google::InitGoogleLogging(argv[0]);

    if (argc < 4) {
        std::cerr << "Usage: ./geosolve <data_dir> <command> <project.json>" << std::endl;
        return -1;
    }

    string data_root(argv[1]);
    string command(argv[2]);
    string project_filename(argv[3]);

    if (command == "base") {
        base_model0(project_filename);
    }
    else if (command == "loadtest") {
        Project project = Project::from_file(project_filename);
        project.to_file("loadtest-output.json");
    }
    else if (command == "features") {
        Project project = Project::from_file(project_filename);
        std::cout << "Loading images" << std::endl;
        project.data_set->load(data_root);
        std::cout << "Computing features" << std::endl;
        project.features->compute();
        project.to_file(project_filename);
    }
    else if (command == "solve") {
        std::cout << "Solving..." << std::endl;
        Project project = Project::from_file(project_filename);
        project.model->solve();
        project.to_file(project_filename);
    }
    else {
        std::cerr << "Invalid command: " << command << std::endl;
    }
}

#include <iostream>
#include <memory>
#include <string>

using std::vector;
using std::shared_ptr;

#include <opencv2/opencv.hpp>
#include "opencv2/xfeatures2d.hpp"

struct DataSetPair {
    DataSetPair(string left, string right) {
        image_list.push_back(cv::imread(left));
        image_list.push_back(cv::imread(right));
    }

    vector<cv::Mat> image_list;
};

// Represents image features of a dataset (keypoints and matches)
// As well as pairwise relationship (relative image positions, existence of overlap)
struct ImageFeatures {
    ImageFeatures(const DataSetPair& data_set) {
        // For each image pair compute keypoints and matches
    }

};

// Simple side by side pair initialization

// Abstract model interface
struct Model {
    Model(shared_ptr<Model> p) : parent(p) {
    }

    shared_ptr<Model> parent;
};

// struct Model0 : public Model {
struct Model0 {
    Model0(const DataSetPair& ds, const ImageFeatures& f) : data_set(ds), features(f) {
        // Initialize from parent
        // (for now hard coded left-right images)

    }
};

int main() {
    if (argc < 3) {
        std::cerr << "Usage: ./model0 data_dir results_dir" << std::endl;
        return -1;
    }

    string data = std::string(argv[1]);
    string results_dir = std::string(argv[2]) + "/features_analysis";

    // Load images
    // data set definition file / code
    // defines filenames, pairwise order
    DataSetPair data_set(data + /"alinta-stockpile/DSC_5522.jpg", data + /"alinta-stockpile/DSC_5522.jpg");

    // Match features
    ImageFeatures features(data_set);
    // params: algo

    Model0 model(gps_model, data_set, features, params);
    // Create model 0
    // params: data set, features, optimization params

    // Setup solver model
    // solve
}

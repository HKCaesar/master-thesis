#include <iostream>
#include <fstream>
#include <vector>
#include <array>
#include <tuple>
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

#include <opencv2/opencv.hpp>
#include "opencv2/xfeatures2d.hpp"

#include "camera_models.h"

struct DataSetPair {
    DataSetPair(string l, string r) : left(l), right(r) {
    }

    void load(string data_root) {
        image_left = cv::imread(data_root + "/" + left);
        image_right = cv::imread(data_root + "/" + right);

        if (image_left.data == NULL || image_right.data == NULL) {
            std::cerr << "ERROR: cannot load DataSetPair images\n";
        }
    }

    template <class Archive>
    void serialize(Archive& ar) {
        ar(CEREAL_NVP(left), CEREAL_NVP(right));
    }

    const std::string left;
    const std::string right;

    // TODO load images on demand
    // or manually, ex: if (!isloaded) data_set.load() before computing features
    cv::Mat image_left;
    cv::Mat image_right;
};

// Returns the list of indexes that sort the match array by distance
vector<size_t> argsort(vector<cv::DMatch> matches) {
    // Extend the vector with indexes
    vector< tuple<size_t, cv::DMatch> > indexed_matches;
    size_t n = 0;
    for (n = 0; n < matches.size(); ++n) {
        indexed_matches.push_back(make_tuple(n, matches[n]));
    }

    // Sort by distance
    std::sort(indexed_matches.begin(), indexed_matches.end(),
            [](const tuple<size_t, cv::DMatch>& a, const tuple<size_t, cv::DMatch>& b) {
                return std::get<1>(a).distance < std::get<1>(b).distance;
            });

    // Extract the index
    vector<size_t> indexes(matches.size());
    for (size_t i = 0; i < matches.size(); i++) {
        indexes[i] = std::get<0>(indexed_matches[i]);
    }
    return indexes;
}

template<typename T>
vector<T> reorder(const vector<T>& input, vector<size_t> indexes) {
    vector<T> output(input.size());
    for (size_t i = 0; i < input.size(); i++) {
        output[i] = input[indexes[i]];
    }
    return output;
}

void write_matches_image(string path, cv::Mat image1, cv::Mat image2,
                      vector<cv::KeyPoint> keypoint1, vector<cv::KeyPoint> keypoint2,
                      vector<cv::DMatch> matches,
                      size_t nb_of_features) {
    // Maximum wrap
    if (nb_of_features > matches.size()) {
        nb_of_features = matches.size();
    }

    // Keep nb_of_features elements
    vector<cv::DMatch> good_matches(matches.begin(), matches.begin() + nb_of_features);

    // Draw
    cv::Mat img;
    cv::drawMatches(image1, keypoint1, image2, keypoint2,
            good_matches, img, cv::Scalar::all(-1), cv::Scalar::all(-1),
            vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    cv::imwrite(path, img);
}

// TODO add support for computing any feature on a zoom out input
// see results for features 5% outlier level in report
//
// Represents image features of a dataset (keypoints and matches)
// As well as pairwise relationship (relative image positions, existence of overlap)
struct ImageFeatures {
    ImageFeatures(const DataSetPair& ds, const size_t maximum_number_of_matches) : data_set(ds) {
        cv::Ptr<cv::xfeatures2d::SURF> surf = cv::xfeatures2d::SURF::create();
        cv::Ptr<cv::xfeatures2d::SIFT> sift = cv::xfeatures2d::SIFT::create();
        cv::Ptr<cv::FeatureDetector> detector = sift;
        cv::Ptr<cv::DescriptorExtractor> descriptor = sift;

        // Detect and compute descriptors
        detector->detect(data_set.image_left, keypoint1);
        detector->detect(data_set.image_right, keypoint2);
        descriptor->compute(data_set.image_left, keypoint1, descriptor1);
        descriptor->compute(data_set.image_right, keypoint2, descriptor2);

        if (descriptor1.empty() || descriptor2.empty()) {
            std::cerr << "Empty descriptor!" << std::endl;
        }

        // FLANN needs type of descriptor to be CV_32F
        if (descriptor1.type()!= CV_32F) {
            descriptor1.convertTo(descriptor1, CV_32F);
        }
        if (descriptor2.type()!= CV_32F) {
            descriptor2.convertTo(descriptor2, CV_32F);
        }

        // Match using FLANN
        cv::FlannBasedMatcher matcher;
        matcher.match(descriptor1, descriptor2, matches);

        // Sort matches by distance
        // Note that keypoints don't need to be reordered because the index
        // of a match's keypoint is stored in match.queryIdx and match.trainIdx
        // (i.e. is not given implicitly by the order of the keypoint vector)
        vector<size_t> order = argsort(matches);
        matches = reorder(matches, order);

        write_matches_image("matches.jpg", data_set.image_left, data_set.image_right, keypoint1, keypoint2, matches, maximum_number_of_matches);

        // Store into simple ordered by distance vector of observations
        for (size_t i = 0; i < matches.size() && i < maximum_number_of_matches; i++) {
            // query is kp1, train is kp2 (see declaration of matcher.match
            observations.push_back(array<double, 4>{
                    keypoint1[matches[i].queryIdx].pt.x,
                    keypoint1[matches[i].queryIdx].pt.y,
                    keypoint2[matches[i].trainIdx].pt.x,
                    keypoint2[matches[i].trainIdx].pt.y
                });
        }
    }

    template <class Archive>
    void serialize(Archive& ar) {
        ar(cereal::make_nvp("observations", observations));
    }

    vector<cv::KeyPoint> keypoint1;
    vector<cv::KeyPoint> keypoint2;
    cv::Mat descriptor1;
    cv::Mat descriptor2;
    vector<cv::DMatch> matches;

    vector< array<double, 4> > observations;

    const DataSetPair& data_set;
};

// struct Model0 : public Model {
struct Model0 {
    Model0(const ImageFeatures& f, array<const double, 3> internal, array<double, 6> left_cam, array<double, 6> right_cam) :
        features(f),
        internal(internal) {
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
        ar(cereal::make_nvp("cameras", cameras), cereal::make_nvp("terrain", terrain));
    }

    const ImageFeatures& features;

    // 3 dof internals: {f, ppx, ppy}
    array<const double, 3> internal;

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
    DataSetPair data_set("alinta-stockpile/DSC_5522.JPG", "alinta-stockpile/DSC_5521.JPG");
    data_set.load(data_root);

    // Match features and obtain observations
    std::cout << "Matching features..." << std::endl;
    ImageFeatures features(data_set, 10);

    // Create model
    std::cout << "Setting up model..." << std::endl;
    Model0 model(features, {48.3355e-3, 0.0093e-3, -0.0276e-3}, {0, 0, 269, 0, 0, 0}, {0, 0, 269, 0, 0, 0});

    const double pixel_size = 0.0085e-3;

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

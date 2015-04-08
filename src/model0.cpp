#include <iostream>
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

#include <opencv2/opencv.hpp>
#include "opencv2/xfeatures2d.hpp"

struct DataSetPair {
    DataSetPair(string left, string right) {
        image_left = cv::imread(left);
        image_right = cv::imread(right);
    }

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

// Represents image features of a dataset (keypoints and matches)
// As well as pairwise relationship (relative image positions, existence of overlap)
struct ImageFeatures {
    ImageFeatures(const DataSetPair& ds) : data_set(ds) {
        cv::Ptr<cv::xfeatures2d::SURF> surf = cv::xfeatures2d::SURF::create();
        cv::Ptr<cv::FeatureDetector> detector = surf;
        cv::Ptr<cv::DescriptorExtractor> descriptor = surf;

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

        // Store into simple ordered by distance vector of observations
        for (size_t i = 0; i < matches.size(); i++) {
            // query is kp1, train is kp2 (see declaration of matcher.match
            observations.push_back(array<double, 4>{
                    keypoint1[matches[i].queryIdx].pt.x,
                    keypoint1[matches[i].queryIdx].pt.y,
                    keypoint2[matches[i].trainIdx].pt.x,
                    keypoint2[matches[i].trainIdx].pt.y
                });
        }
    }

    vector<cv::KeyPoint> keypoint1;
    vector<cv::KeyPoint> keypoint2;
    cv::Mat descriptor1;
    cv::Mat descriptor2;
    vector<cv::DMatch> matches;

    vector< array<double, 4> > observations;

    const DataSetPair& data_set;
};

// Simple side by side pair initialization

// Abstract model interface

// struct Model0 : public Model {
struct Model0 {
    Model0(const ImageFeatures& f, array<double, 6> left_cam, array<double, 6> right_cam) : features(f) {
        // Initialize from parent
        // (for now hard coded left-right images)

        // Initialize cameras side by side
        cameras[0] = left_cam;
        cameras[1] = right_cam;

        // Initialize zero height terrain by inverse projection of ground features
        // Down project and take average of two ground points?
    }

    const ImageFeatures& features;

    // Parameters
    vector< array<double, 6> > cameras; // 6 dof cameras
    vector< array<double, 2> > terrain; // 2 dof ground points on flat terrain
};

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: ./model0 data_dir results_dir" << std::endl;
        return -1;
    }

    string data = std::string(argv[1]);
    string results_dir = std::string(argv[2]) + "/features_analysis";

    // Load images
    // data set definition file / code
    // defines filenames, pairwise order
    DataSetPair data_set(data + "/alinta-stockpile/DSC_5522.jpg", data + "/alinta-stockpile/DSC_5522.jpg");

    // Match features
    ImageFeatures features(data_set);
    // params: algo

    Model0 model(features, {0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0});
    // Create model 0
    // params: data set, features, optimization params

    // Setup solver model
    // solve
}

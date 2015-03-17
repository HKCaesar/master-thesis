#include <vector>
#include <string>
#include <tuple>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <fstream>
#include <cmath>
#include <opencv2/opencv.hpp>
#include "opencv2/xfeatures2d.hpp"

using std::string;
using std::vector;
using std::tuple;
using std::make_tuple;

// Make sure directory exists
void mkdirp(string path) {
    system((std::string("mkdir -p ") + path).c_str());
}

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

void write_matches_list(string path, vector<cv::KeyPoint> kp1, vector<cv::KeyPoint> kp2, vector<cv::DMatch> matches) {
    std::ofstream ofs((path + "/matches.txt").c_str());
    ofs << "# Keypoints matches file, x y x y dist" << std::endl;
    for (size_t i = 0; i < matches.size(); i++) {
        // query is kp1, train is kp2 (see declaration of matcher.match())
        ofs << kp1[matches[i].queryIdx].pt.x
            << " " << kp1[matches[i].queryIdx].pt.y
            << " " << kp2[matches[i].trainIdx].pt.x
            << " " << kp2[matches[i].trainIdx].pt.y
            << " " << matches[i].distance
            << std::endl;
    }
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

    // Write to file
    std::ostringstream oss;
    oss << nb_of_features;
    cv::imwrite(path + "/" + oss.str() + ".jpg", img);
}

void features_analysis(string path, cv::Mat image1, cv::Mat image2, cv::Ptr<cv::FeatureDetector> detector, cv::Ptr<cv::DescriptorExtractor> descriptor) {
    mkdirp(path);

    // Detect keypoints
    vector<cv::KeyPoint> keypoint1;
    vector<cv::KeyPoint> keypoint2;
    detector->detect(image1, keypoint1);
    detector->detect(image2, keypoint2);

    // Draw keypoints
    cv::Mat img_keypoints_1;
    cv::Mat img_keypoints_2;
    cv::drawKeypoints(image1, keypoint1, img_keypoints_1, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);
    cv::drawKeypoints(image2, keypoint2, img_keypoints_2, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);

    cv::imwrite(path + "/kp1.jpg", img_keypoints_1);
    cv::imwrite(path + "/kp2.jpg", img_keypoints_2);

    // Compute descriptors
    cv::Mat descriptor1;
    cv::Mat descriptor2;
    descriptor->compute(image1, keypoint1, descriptor1);
    descriptor->compute(image2, keypoint2, descriptor2);

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
    vector<cv::DMatch> matches;
    matcher.match(descriptor1, descriptor2, matches);

    // Sort matches by distance
    // Note that keypoints don't need to be reordered because the index
    // of a match's keypoint is stored in match.queryIdx and match.trainIdx
    // (i.e. is not given implicitly by the order of the keypoint vector)
    vector<size_t> order = argsort(matches);
    matches = reorder(matches, order);

    // Output matches and images
    write_matches_list(path, keypoint1, keypoint2, matches);
    for (size_t nb_exp = 0; nb_exp < 11; nb_exp++) {
        write_matches_image(path, image1, image2, keypoint1, keypoint2, matches, pow(2, nb_exp));
    }
    std::ofstream ofs((path + "/shape.txt").c_str());
    ofs << image1.rows << " " << image1.cols << std::endl;
}

// Run features analysis test for an image pair
void test_image_pair(string path, string path1, string path2) {
    std::cout << "Testing image pair " << path << std::endl;

    // Load images
    cv::Mat image1 = cv::imread(path1);
    cv::Mat image2 = cv::imread(path2);
    if (!image1.data || !image2.data) {
        std::cerr << "Error reading image for " << path << std::endl;
        return;
    }

    mkdirp(path);

    // Perform analysis with different configs
    cv::Ptr<cv::ORB> orb = cv::ORB::create();
    cv::Ptr<cv::MSER> mser = cv::MSER::create();
    cv::Ptr<cv::BRISK> brisk = cv::BRISK::create();
    cv::Ptr<cv::KAZE> kaze = cv::KAZE::create();
    cv::Ptr<cv::xfeatures2d::SIFT> sift = cv::xfeatures2d::SIFT::create();
    cv::Ptr<cv::xfeatures2d::SURF> surf400 = cv::xfeatures2d::SURF::create(400);
    cv::Ptr<cv::xfeatures2d::SURF> surf200 = cv::xfeatures2d::SURF::create(200);

    features_analysis(path + "/ORB", image1, image2, orb, orb);
    features_analysis(path + "/MSER-ORB", image1, image2, mser, orb);
    features_analysis(path + "/BRISK", image1, image2, brisk, brisk);
    features_analysis(path + "/BRISK-ORB", image1, image2, brisk, orb);
    features_analysis(path + "/KAZE", image1, image2, kaze, kaze);
    features_analysis(path + "/KAZE-ORB", image1, image2, kaze, orb);
    features_analysis(path + "/SIFT", image1, image2, sift, sift);
    features_analysis(path + "/SURF-400", image1, image2, surf400, surf400);
    features_analysis(path + "/SURF-200", image1, image2, surf200, surf200);
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: ./features_analysis data_dir results_dir" << std::endl;
        return -1;
    }

    string data = std::string(argv[1]);
    string results_dir = std::string(argv[2]) + "/features_analysis";

    test_image_pair(results_dir + "/lake",
            data + "/features-areas/lake1.jpg",
            data + "/features-areas/lake2.jpg");

    test_image_pair(results_dir + "/desert",
            data + "/features-areas/desert1.jpg",
            data + "/features-areas/desert2.jpg");

    test_image_pair(results_dir + "/cars",
            data + "/features-areas/cars1.jpg",
            data + "/features-areas/cars2.jpg");

    test_image_pair(results_dir + "/water",
            data + "/features-areas/water1.jpg",
            data + "/features-areas/water2.jpg");

    test_image_pair(results_dir + "/industrial",
            data + "/features-areas/industrial1.jpg",
            data + "/features-areas/industrial2.jpg");

    test_image_pair(results_dir + "/entire-image-quarter",
            data + "/alinta-stockpile-quarter/DSC_5521.JPG",
            data + "/alinta-stockpile-quarter/DSC_5522.JPG");
}

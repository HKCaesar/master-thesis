#include <vector>
#include <string>
#include <iostream>
#include <sstream>
#include <opencv2/opencv.hpp>
// #include "opencv2/xfeatures2d.hpp"

using std::string;
using std::vector;

// Make sure directory exists
void mkdirp(string path) {
    system((std::string("mkdir -p ") + path).c_str());
}

void write_matches_image(string path, cv::Mat image1, cv::Mat image2,
                      vector<cv::KeyPoint> keypoint1, vector<cv::KeyPoint> keypoint2,
                      vector<cv::DMatch> matches,
                      double threshold) {
    // Filter based on distance
    vector<cv::DMatch> good_matches;
    for (size_t i = 0; i < matches.size(); i++) {
        if (matches[i].distance <= threshold) {
            good_matches.push_back(matches[i]);
        }
    }
    if (good_matches.size() == 0) {
        return;
    }
    // Draw
    cv::Mat img;
    cv::drawMatches(image1, keypoint1, image2, keypoint2,
            good_matches, img, cv::Scalar::all(-1), cv::Scalar::all(-1),
            vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    // Write to file
    std::ostringstream oss;
    oss << threshold;
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

    // Quick calculation of max and min distances between keypoints
    double min_dist, max_dist;
    for (int i = 0; i < descriptor1.rows; i++) {
        double dist = matches[i].distance;
        if (dist < min_dist) min_dist = dist;
        if (dist > max_dist) max_dist = dist;
    }

    write_matches_image(path, image1, image2, keypoint1, keypoint2, matches, 1e9);
    write_matches_image(path, image1, image2, keypoint1, keypoint2, matches, 500);
    write_matches_image(path, image1, image2, keypoint1, keypoint2, matches, 400);
    write_matches_image(path, image1, image2, keypoint1, keypoint2, matches, 300);
    write_matches_image(path, image1, image2, keypoint1, keypoint2, matches, 200);
    write_matches_image(path, image1, image2, keypoint1, keypoint2, matches, 100);
    write_matches_image(path, image1, image2, keypoint1, keypoint2, matches, 50);
}

// Run features analysis test for an image pair
void test_image_pair(string path, string path1, string path2) {
    // Load images
    cv::Mat image1 = cv::imread(path1);
    cv::Mat image2 = cv::imread(path2);
    if (!image1.data || !image2.data) {
        std::cerr << "Error reading image for " << path << std::endl;
        return;
    }

    mkdirp(path);

    // Perform analysis with different configs
    cv::Ptr<cv::ORB> orb = cv::ORB::create(400);
    cv::Ptr<cv::MSER> fast = cv::MSER::create(400);
    cv::Ptr<cv::BRISK> brisk = cv::BRISK::create();
    // cv::Ptr<cv::SURF> surf = cv::SURF::create(400);

    features_analysis(path + "/ORB400", image1, image2, orb, orb);
    features_analysis(path + "/SURF400", image1, image2, fast, orb);
    features_analysis(path + "/BRISK", image1, image2, brisk, brisk);

}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: ./features_analysis data_dir results_dir" << std::endl;
        return -1;
    }

    string data = std::string(argv[1]);
    string results_dir = std::string(argv[2]) + "/features_analysis";

    test_image_pair(results_dir + "/ab", data + "/alinta-stockpile-quarter/DSC_5590.JPG", data + "/alinta-stockpile-quarter/DSC_5591.JPG");
}

#include <tuple>
#include <stdexcept>
#include "image_features.h"

using std::vector;
using std::array;
using std::string;
using std::tuple;

FeaturesGraph::FeaturesGraph() :
    number_of_matches(0),
    compute_scale(1.0),
    computed(false) {
}

void FeaturesGraph::add_edge(size_t cam_a, size_t cam_b) {
    edges.push_back(obs_pair {cam_a, cam_b, std::vector<pixel_t>(), std::vector<pixel_t>()});
}

// Returns the list of indexes that sort the match array by distance
vector<size_t> argsort(vector<cv::DMatch> matches) {
    // Extend the vector with indexes
    vector< tuple<size_t, cv::DMatch> > indexed_matches;
    size_t n = 0;
    for (n = 0; n < matches.size(); ++n) {
        indexed_matches.push_back(std::make_tuple(n, matches[n]));
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

void FeaturesGraph::compute(const std::string& data_root) {
    if (!data_set) {
        throw std::runtime_error("FeaturesGraph has no associated DataSet.");
    }
    if (number_of_matches <= 0) {
        throw std::runtime_error("FeaturesGraph has invalid maximum number of matches: " + std::to_string(number_of_matches));
    }

    std::vector<cv::Mat> images;
    for (auto& name : data_set->filenames) {
        cv::Mat im = cv::imread(data_root + "/" + name);
        if (im.data == NULL) {
            throw std::runtime_error("Cannot load image " + name);
        }
        if (compute_scale != 1.0) {
            cv::Mat resized;
            cv::resize(im, resized, cv::Size(), compute_scale, compute_scale, cv::INTER_AREA);
            im = resized;
        }
        images.push_back(im);
    }

    cv::Ptr<cv::xfeatures2d::SIFT> sift = cv::xfeatures2d::SIFT::create();
    for (size_t i = 0; i < edges.size(); i++) {
        edges[i].compute(images, sift, sift, compute_scale, number_of_matches);
    }
    computed = true;
}

void obs_pair::compute(const std::vector<cv::Mat>& images,
                       cv::Ptr<cv::FeatureDetector> detector,
                       cv::Ptr<cv::DescriptorExtractor> descriptor,
                       double compute_scale,
                       size_t number_of_matches) {
    std::vector<cv::KeyPoint> keypoint1;
    std::vector<cv::KeyPoint> keypoint2;
    cv::Mat descriptor1;
    cv::Mat descriptor2;
    std::vector<cv::DMatch> matches;

    // Detect and compute descriptors
    detector->detect(images[cam_a], keypoint1);
    detector->detect(images[cam_b], keypoint2);
    descriptor->compute(images[cam_a], keypoint1, descriptor1);
    descriptor->compute(images[cam_b], keypoint2, descriptor2);

    if (descriptor1.empty() || descriptor2.empty()) {
        std::cerr << "Empty descriptor!" << std::endl;
    }

    // Match using FLANN
    // FLANN needs type of descriptor to be CV_32F
    if (descriptor1.type()!= CV_32F) {
        descriptor1.convertTo(descriptor1, CV_32F);
    }
    if (descriptor2.type()!= CV_32F) {
        descriptor2.convertTo(descriptor2, CV_32F);
    }
    cv::FlannBasedMatcher matcher;
    matcher.match(descriptor1, descriptor2, matches);

    // Sort matches by distance
    // Note that keypoints don't need to be reordered because the index
    // of a match's keypoint is stored in match.queryIdx and match.trainIdx
    // (i.e. is not given implicitly by the order of the keypoint vector)
    vector<size_t> order = argsort(matches);
    matches = reorder(matches, order);

    // Store into simple ordered by distance vector of observations
    for (size_t i = 0; i < matches.size() && i < number_of_matches; i++) {
        // query is kp1, train is kp2 (see declaration of matcher.match)
        // saved in pixel coordinates: opencv(y, x) == pixel_t(i, j)
        obs_a.push_back(pixel_t(
                static_cast<double>(keypoint1[matches[i].queryIdx].pt.y) / compute_scale,
                static_cast<double>(keypoint1[matches[i].queryIdx].pt.x) / compute_scale));
        obs_b.push_back(pixel_t(
                static_cast<double>(keypoint2[matches[i].trainIdx].pt.y) / compute_scale,
                static_cast<double>(keypoint2[matches[i].trainIdx].pt.x) / compute_scale));
    }
}


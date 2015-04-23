#include <tuple>
#include <stdexcept>
#include "image_features.h"

using std::vector;
using std::array;
using std::string;
using std::tuple;

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

ImageFeatures::ImageFeatures() :
    maximum_number_of_matches(0) {
}

void ImageFeatures::compute() {
    if (!data_set) {
        throw std::runtime_error("ImageFeatures has no associated DataSet.");
    }
    if (maximum_number_of_matches <= 0) {
        throw std::runtime_error("ImagesFeatures has invalid maximum number of matches: " + std::to_string(maximum_number_of_matches));
    }
    if (data_set->isloaded == false) {
        throw std::runtime_error("ImageFeatures.compute() called but DataSet is not loaded");
    }
    cv::Ptr<cv::xfeatures2d::SURF> surf = cv::xfeatures2d::SURF::create();
    cv::Ptr<cv::xfeatures2d::SIFT> sift = cv::xfeatures2d::SIFT::create();
    cv::Ptr<cv::FeatureDetector> detector = sift;
    cv::Ptr<cv::DescriptorExtractor> descriptor = sift;

    // Detect and compute descriptors
    detector->detect(data_set->images[0], keypoint1);
    detector->detect(data_set->images[1], keypoint2);
    descriptor->compute(data_set->images[0], keypoint1, descriptor1);
    descriptor->compute(data_set->images[1], keypoint2, descriptor2);

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
    for (size_t i = 0; i < matches.size() && i < maximum_number_of_matches; i++) {
        // query is kp1, train is kp2 (see declaration of matcher.match)
        // saved in pixel coordinates: (y, x) == (i, j)
        observations.push_back(array<double, 4>{
                keypoint1[matches[i].queryIdx].pt.y,
                keypoint1[matches[i].queryIdx].pt.x,
                keypoint2[matches[i].trainIdx].pt.y,
                keypoint2[matches[i].trainIdx].pt.x
            });
    }
}

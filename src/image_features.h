#ifndef IMAGE_FEATURES_H
#define IMAGE_FEATURES_H

#include <vector>
#include <array>
#include <memory>
#include <cereal/types/vector.hpp>
#include <cereal/types/array.hpp>
#include <opencv2/opencv.hpp>
#include "opencv2/xfeatures2d.hpp"
#include "data_set.h"

// TODO add support for computing any feature on a zoom out input
// see results for features 5% outlier level in report
//
// Represents image features of a dataset (keypoints and matches)
// As well as pairwise relationship (relative image positions, existence of overlap)
struct ImageFeatures {
    ImageFeatures();

    std::shared_ptr<DataSet> data_set;
    size_t maximum_number_of_matches;
    std::vector<std::array<double, 4>> observations;

    void compute();

    template <class Archive>
    void serialize(Archive& ar) {
        ar(cereal::make_nvp("observations", observations),
           cereal::make_nvp("maximum_number_of_matches", maximum_number_of_matches),
           cereal::make_nvp("data_set", data_set));
    }

private:
    std::vector<cv::KeyPoint> keypoint1;
    std::vector<cv::KeyPoint> keypoint2;
    cv::Mat descriptor1;
    cv::Mat descriptor2;
    std::vector<cv::DMatch> matches;
};

#endif

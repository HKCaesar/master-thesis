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
#include "types.h"

#define NVP(x) CEREAL_NVP(x)

// Data structure for one edge of the features graph
struct obs_pair {
    size_t cam_a, cam_b;
    std::vector<pixel_t> obs_a, obs_b;

    void compute(const std::vector<cv::Mat>&,
                 cv::Ptr<cv::FeatureDetector>,
                 cv::Ptr<cv::DescriptorExtractor>,
                 double compute_scale,
                 size_t number_of_matches);

    template <class Archive>
    void serialize(Archive& ar) {
        ar(NVP(cam_a), NVP(cam_b),
           NVP(obs_a), NVP(obs_b));
    }
};

// TODO customizable algorithm, parameters, etc...
// TODO add support for computing any feature on a zoom out input
// see results for features 5% outlier level in report

// Represents image features of a dataset (keypoints and matches)
// As well as pairwise relationship (relative image positions, existence of overlap)
class FeaturesGraph {
public:
    FeaturesGraph();
    void add_edge(size_t cam_a, size_t cam_b);
    void compute(const std::string& data_dir);

    template <class Archive>
    void serialize(Archive& ar) {
        ar(NVP(data_set),
           NVP(number_of_matches),
           NVP(compute_scale),
           NVP(computed),
           NVP(edges));
    }

    std::shared_ptr<DataSet> data_set;
    size_t number_of_matches; // Number of matches per edge
    double compute_scale; // Down scale factor for computing features
    bool computed; // true iff compute() has been performed
    std::vector<obs_pair> edges; // Edges of the features graph
};


#endif

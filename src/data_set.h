#ifndef DATA_SET_H
#define DATA_SET_H

#include <vector>
#include <string>
#include <cereal/types/vector.hpp>
#include <opencv2/opencv.hpp>

struct DataSet {
    DataSet() : isloaded(false) {}

    void load(std::string data_root) {
        for (auto& name : filenames) {
            cv::Mat im = cv::imread(data_root + "/" + name);
            if (im.data == NULL) {
                std::cerr << "ERROR: cannot load DataSetPair images" << std::endl;
            }
            images.push_back(im);
        }
        isloaded = true;
    }

    template <class Archive>
    void serialize(Archive& ar) {
        ar(CEREAL_NVP(filenames));
    }

    std::vector<std::string> filenames;

    // Non serialized runtime state
    bool isloaded;
    std::vector<cv::Mat> images;
};

#endif

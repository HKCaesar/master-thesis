#include <vector>

#include <iostream>
#include <opencv2/opencv.hpp>

void features_analysis(cv::Mat image1, cv::Mat image2, cv::Ptr<cv::FeatureDetector> detector, cv::Ptr<cv::DescriptorExtractor> descriptor) {
    // Detect keypoints
    std::vector<cv::KeyPoint> keypoint1;
    std::vector<cv::KeyPoint> keypoint2;

    detector->detect(image1, keypoint1);
    detector->detect(image2, keypoint2);

    // Draw keypoints
    cv::Mat img_keypoints_1;
    cv::Mat img_keypoints_2;
    cv::drawKeypoints(image1, keypoint1, img_keypoints_1, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);
    cv::drawKeypoints(image2, keypoint2, img_keypoints_2, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);

    cv::imwrite("KP1.jpg", img_keypoints_1);
    cv::imwrite("KP2.jpg", img_keypoints_2);

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
    std::vector<cv::DMatch> matches;
    matcher.match(descriptor1, descriptor2, matches);
    
    // cv::BFMatcher matcher(cv::NORM_L2);
    // std::vector<cv::DMatch> matches;
    // matcher.match(descriptor1, descriptor2, matches);

    double max_dist = 0;
    double min_dist = 100;

    // Quick calculation of max and min distances between keypoints
    for( int i = 0; i < descriptor1.rows; i++ ) {
        double dist = matches[i].distance;
        if( dist < min_dist ) min_dist = dist;
        if( dist > max_dist ) max_dist = dist;
    }

    std::cout << "Max dist" << max_dist << std::endl;
    std::cout << "Min dist" << min_dist << std::endl;

    //-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist,
    //-- or a small arbitary value ( 0.02 ) in the event that min_dist is very
    //-- small)
    //-- PS.- radiusMatch can also be used here.
    std::vector<cv::DMatch> good_matches;

    for( int i = 0; i < descriptor1.rows; i++ ) {
        if (matches[i].distance <= cv::max(4*min_dist, 0.02)) {
            good_matches.push_back( matches[i]);
        }
    }

    // Draw only "good" matches
    cv::Mat img_matches_good;
    cv::drawMatches( image1, keypoint1, image2, keypoint2,
            good_matches, img_matches_good, cv::Scalar::all(-1), cv::Scalar::all(-1),
            std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

    cv::imwrite("matches_good.jpg", img_matches_good);

    cv::Mat img_matches_all;
    cv::drawMatches( image1, keypoint1, image2, keypoint2,
            matches, img_matches_all, cv::Scalar::all(-1), cv::Scalar::all(-1),
            std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

    cv::imwrite("matches_all.jpg", img_matches_all);

    for( int i = 0; i < (int)good_matches.size(); i++ )
    { printf( "-- Good Match [%d] Keypoint 1: %d  -- Keypoint 2: %d  \n", i, good_matches[i].queryIdx, good_matches[i].trainIdx ); }
}

int main() {
    // Load two images
    cv::Mat image1 = cv::imread("../data/alinta-stockpile-quarter/DSC_5468.JPG");
    cv::Mat image2 = cv::imread("../data/alinta-stockpile-quarter/DSC_5469.JPG");

    if (!image1.data || !image2.data) {
        std::cerr << "Error reading image!" << std::endl;
        return -1;
    }

    cv::Ptr<cv::ORB> orb = cv::ORB::create(400);
    cv::Ptr<cv::ORB> surf = cv::ORB::create(400);
    features_analysis(image1, image2, surf, surf);
}

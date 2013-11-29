#ifndef _AFEATUREEXTRACTOR_H_
#define _AFEATUREEXTRACTOR_H_

#include <opencv2/opencv.hpp>

class AFeatureExtractor
{
public:
	virtual void extractFromMat(const cv::Mat& img, cv::Mat_<float>& feature) = 0;
	virtual int featureDimension(const cv::Mat& img) = 0;
};

#endif
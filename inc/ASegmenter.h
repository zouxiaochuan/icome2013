#ifndef _ASEGMENTER_H_
#define _ASEGMENTER_H_

#include <opencv2/opencv.hpp>

class ASegmenter
{
public:
	virtual void segment(const cv::Mat& img, cv::Mat_<uchar>& mask) = 0;
};

#endif
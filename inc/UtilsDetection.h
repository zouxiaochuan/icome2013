#ifndef _UTILSDETECTION_H_
#define _UTILSDETECTION_H_

#include <opencv2/opencv.hpp>

class UtilsDetection
{
public:
	static cv::Rect FilterBoundingBox(const cv::Mat& img, std::vector<cv::Rect>& boxes, cv::Mat_<float>& confs
		,cv::Mat_<float>& borders);

	static void RankDetection(std::vector<cv::Rect>& boxes, const cv::Mat& mask, cv::Mat_<float>& scores);
};

#endif
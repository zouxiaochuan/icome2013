#ifndef _ADETECTOR_H_
#define _ADETECTOR_H_

#include <opencv2/opencv.hpp>

class ADetector
{
public:
	virtual void detectSingleScale(const cv::Mat& img, std::vector<cv::Rect>& boxes, cv::Mat_<float>& confidence) = 0;
	virtual void detectMultiScale(const cv::Mat& img, std::vector<cv::Rect>& boxes, cv::Mat_<float>& confidence, double scaleFactor
		, double maxSize, double minSize){};

	virtual void save(const std::string& filename) = 0;
	virtual void load(const std::string& filename) = 0;
};

#endif
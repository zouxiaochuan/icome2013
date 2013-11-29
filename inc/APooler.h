#ifndef _APOOLER_H_
#define _APOOLER_H_

#include <opencv2/opencv.hpp>

class APooler
{
public:
	virtual void pool(const cv::Mat& code, cv::Mat& pooled) = 0;
};

#endif
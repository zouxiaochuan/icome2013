#ifndef _APROBAESTIMATOR_H_
#define _APROBAESTIMATOR_H_

#include <opencv2/opencv.hpp>

class AProbaEstimator
{
public:
	virtual void train(const cv::Mat& data) = 0;
	virtual void predict(const cv::Mat& data, cv::Mat_<float>& proba) = 0;

	virtual AProbaEstimator* clone() = 0;
};


#endif
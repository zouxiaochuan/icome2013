#ifndef _ACLASSIFIER_H_
#define _ACLASSIFIER_H_

#include <opencv2/opencv.hpp>
#include <string>

class AClassifier
{
public:
	virtual void train(const cv::Mat& data, const cv::Mat_<int>& label) = 0;
	virtual void predict(const cv::Mat& data, cv::Mat_<int>& label) = 0;
	virtual void predictProba(const cv::Mat& data, cv::Mat_<float>& proba) = 0;
	virtual void getLabels(cv::Mat_<int>& ulabel) = 0;

	virtual void save(const std::string& filename) = 0;
	virtual void load(const std::string& filename) = 0;

	virtual ~AClassifier(){};
};

#endif
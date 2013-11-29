#ifndef _CLASSIFIERNAIVEBAYESSMOOTH_H_
#define _CLASSIFIERNAIVEBAYESSMOOTH_H_

#include "AClassifier.h"
#include <opencv2/opencv.hpp>

class ClassifierNaiveBayesSmooth : public AClassifier
{
public:
	virtual void train(const cv::Mat& data, const cv::Mat_<int>& label) ;
	virtual void predict(const cv::Mat& data, cv::Mat_<int>& label) ;
	virtual void predictProba(const cv::Mat& data, cv::Mat_<float>& proba);
	virtual void save(const std::string& filename);
	virtual void load(const std::string& filename);

public:
	ClassifierNaiveBayesSmooth(int histNum, int nDim, double sigma);

private:

	cv::Mat_<float> _m_matHist;
	double _m_sigma;
};

#endif
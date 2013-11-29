#ifndef _PROBAESTIMATORNAIVEBAYESSMOOTH_H_
#define _PROBAESTIMATORNAIVEBAYESSMOOTH_H_

#include "AProbaEstimator.h"

class ProbaEstimatorNaiveBayesSmooth : public AProbaEstimator
{
public:
	virtual void train(const cv::Mat& data);
	virtual void predict(const cv::Mat& data, cv::Mat_<float>& proba);

	virtual ProbaEstimatorNaiveBayesSmooth* clone();

public:
	ProbaEstimatorNaiveBayesSmooth(int numHist,double sigma) : _m_nHist(numHist),_m_sigma(sigma)
	{};

private:

	int _m_nHist;
	double _m_sigma;

	cv::Mat_<float> _m_matHist;

};

#endif
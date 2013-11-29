#ifndef _CLASSIFIERPROBAESTIMATE_H_
#define _CLASSIFIERPROBAESTIMATE_H_

#include "AProbaEstimator.h"
#include "AClassifier.h"
#include <vector>

class ClassifierProbaEstimate : AClassifier
{

public:
	virtual void train(const cv::Mat& data, const cv::Mat_<int>& label) ;
	virtual void predict(const cv::Mat& data, cv::Mat_<int>& label){};
	virtual void predictProba(const cv::Mat& data, cv::Mat_<float>& proba);
	virtual void getLabels(cv::Mat_<int>& ulabel){};
	virtual void save(const std::string& filename){};
	virtual void load(const std::string& filename){};

public:
	ClassifierProbaEstimate(AProbaEstimator* p);
	~ClassifierProbaEstimate();

private:
	std::vector<AProbaEstimator*> _m_vecEstimator;
	AProbaEstimator* _m_pBaseEstimator;
	cv::Mat_<int> _m_matLabels;
};

#endif
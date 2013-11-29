#ifndef _FEATUREEXTRACTORSPMEX_H_
#define _FEATUREEXTRACTORSPMEX_H_

#include <string>
#include "ALocalExtractor.h"
#include "AFeatureExtractor.h"
#include "APooler.h"
#include "AEncoder.h"

class FeatureExtractorSPMEx : public AFeatureExtractor
{
public:
	FeatureExtractorSPMEx(const std::string& fileCodebook);
	~FeatureExtractorSPMEx();

	virtual void extractFromMat(const cv::Mat& img, cv::Mat_<float>& feature);
	virtual int featureDimension(const cv::Mat& img);

private:

	ALocalExtractor* _m_pLocalExtractor;
	APooler * _m_pPooler;
	AEncoder* _m_pEncoder;
	AFeatureExtractor* _m_pExtractor;
};

#endif
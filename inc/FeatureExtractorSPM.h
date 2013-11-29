#ifndef _FEATUREEXTRACTORSPM_H_
#define _FEATUREEXTRACTORSPM_H_

#include "AFeatureExtractor.h"
#include "APooler.h"
#include "AImageDataset.h"
#include "ALocalExtractor.h"
#include "AEncoder.h"

class FeatureExtractorSPM : public AFeatureExtractor
{
public:
	virtual void extractFromMat(const cv::Mat& img, cv::Mat_<float>& feature);
	virtual int featureDimension(const cv::Mat& img);

public:
	FeatureExtractorSPM(const cv::Mat_<int>& spm, APooler* p);
	FeatureExtractorSPM(cv::Mat_<int>& spm, APooler* pp, ALocalExtractor* pl, AEncoder* pe);

public:
	void extractFromEncodedPath(const std::string& encodedPath, AImageDataset* pdataset, int codeLen, cv::Mat_<float>& feature);

private:
	void _extractFromEncoded(const cv::Mat_<float>& encoded, const cv::Mat_<float>& location, cv::Mat_<float>& feature);
	int  _SPMFactor();


private:
	std::vector<APooler*> _m_vecPooler;
	cv::Mat_<int> _m_matSPM;
	ALocalExtractor* _m_pLocalExtractor;
	AEncoder* _m_pEncoder;

};

#endif
#include "FeatureExtractorSPMEx.h"
#include "UtilsOpencv.hpp"
#include "LocalExtractorDenseSiftMultiScale.h"
#include "EncoderBoFSoft.h"
#include "CodebookVQ.h"
#include "PoolerMax.h"
#include "FeatureExtractorSPM.h"

#include <vector>

using namespace std;

FeatureExtractorSPMEx::FeatureExtractorSPMEx(const std::string& fileCodebook)
{
	vector<int> sizes;
	sizes.push_back(4);
	sizes.push_back(6);
	sizes.push_back(8);
	sizes.push_back(12);

	Mat_<float> mat;
	UtilsOpencv::ReadTxt(fileCodebook,mat);
	CodebookVQ cb;
	cb.setBasis(mat);

	this->_m_pLocalExtractor = new LocalExtractorDenseSiftMultiScale("vlfeat",4,sizes,300);
	this->_m_pEncoder = new EncoderBoFSoft(cb,8); 
	this->_m_pPooler = new PoolerMax();
	Mat_<int> spm;
	spm.create(2,2);
	spm(0,0) = spm(0,1) = 1;
	spm(1,0) = spm(1,1) = 2;

	this->_m_pExtractor = new FeatureExtractorSPM(spm,_m_pPooler,_m_pLocalExtractor,_m_pEncoder);
}


FeatureExtractorSPMEx::~FeatureExtractorSPMEx()
{
	delete _m_pLocalExtractor;
	delete _m_pEncoder;
	delete _m_pPooler;
	delete _m_pExtractor;
}

int FeatureExtractorSPMEx::featureDimension(const cv::Mat& img)
{
	return this->_m_pExtractor->featureDimension(img);
}

void FeatureExtractorSPMEx::extractFromMat(const cv::Mat& img, cv::Mat_<float>& feature)
{
	return this->_m_pExtractor->extractFromMat(img,feature);
}
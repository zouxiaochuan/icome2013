#include "LocalExtractorDenseSiftMultiScale.h"
#include "LocalExtractorDenseSiftVl.h"
#include "UtilsOpencv.hpp"

LocalExtractorDenseSiftMultiScale::LocalExtractorDenseSiftMultiScale(const string& type, int step, const vector<int> sizes, int maxsize)
{
	this->_m_MaxSize = maxsize;
	for(int i=0;i<sizes.size();i++)
	{
		this->_m_extractors.push_back(new  LocalExtractorDenseSiftVl(step,sizes[i],maxsize));
	}
}

LocalExtractorDenseSiftMultiScale::~LocalExtractorDenseSiftMultiScale()
{
	for (int i=0;i<_m_extractors.size();i++)
	{
		delete _m_extractors[i];
	}
}

void LocalExtractorDenseSiftMultiScale::extractFromMat( const Mat& img, Mat_<float>& descriptors, Mat_<float>& location)
{
	vector<Mat_<float> > vecDesc;
	vector<Mat_<float> > vecLoc;

	for( int i=0;i<_m_extractors.size();i++)
	{
		Mat_<float> d,l;
		_m_extractors[i]->extractFromMat(img,d,l);
		vecDesc.push_back(d);
		vecLoc.push_back(l);
	}

	UtilsOpencv::MergeRows(vecDesc,descriptors);
	UtilsOpencv::MergeRows(vecLoc,location);

}
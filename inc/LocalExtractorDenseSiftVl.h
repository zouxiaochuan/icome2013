#ifndef _LOCALEXTRACTORDENSESIFTVL_H_
#define _LOCALEXTRACTORDENSESIFTVL_H_

#include "ALocalExtractor.h"

class LocalExtractorDenseSiftVl : public ALocalExtractor
{
public:
	LocalExtractorDenseSiftVl(int step, int binsize, int maxsize=-1) : mStep(step), mBinSize(binsize), mMaxSize(maxsize)
	{};

	virtual void extractFromMat( const Mat& img, Mat_<float>& descriptors, Mat_<float>& location);

private:
	int mStep;
	int mBinSize;
	int mMaxSize;
};


#endif
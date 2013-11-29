#ifndef _LOCALEXTRACTORDENSESIFTMULTISCALE_H_
#define _LOCALEXTRACTORDENSESIFTMULTISCALE_H_

#include "ALocalExtractor.h"
#include <string>
#include <vector>
using namespace std;

class LocalExtractorDenseSiftMultiScale : public ALocalExtractor
{
public:
	virtual void extractFromMat( const Mat& img, Mat_<float>& descriptors, Mat_<float>& location);

public:
	LocalExtractorDenseSiftMultiScale(const string& type, int step, const vector<int> sizes, int maxsize=-1);
	virtual ~LocalExtractorDenseSiftMultiScale();

private:
	vector<ALocalExtractor*> _m_extractors;
	int _m_MaxSize;
};

#endif
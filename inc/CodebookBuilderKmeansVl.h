#ifndef _CODEBOOKBUILDERKMEANSVL_H
#define _CODEBOOKBUILDERKMEANSVL_H

#include "ACodebookBuilder.h"
#include <opencv2/opencv.hpp>
using namespace cv;

class CodebookBuilderKmeansVl : public ACodebookBuilder
{
public:
	virtual void build(const Mat_<float> data, ACodebook& cb);

public:
	CodebookBuilderKmeansVl(int nCent, int nMaxIter, int nRun) : _m_nCent(nCent),
		_m_MaxIteration(nMaxIter),_m_nRun(nRun)
	{
	}
private:
	int _m_nRun;
	int _m_MaxIteration;
	int _m_nCent;
};

#endif
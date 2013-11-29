#ifndef _CODEBOOKVQ_H_
#define _CODEBOOKVQ_H_

#include "ACodebook.h"
#include <opencv2/opencv.hpp>

using namespace cv;

class CodebookVQ : public ACodebook
{
public:
	virtual void save(const std::string& filename);
	virtual void load(const std::string& filename);

public:
	CodebookVQ();
	CodebookVQ(CodebookVQ& other);

	void setBasis(const Mat_<float>& basis)
	{
		this->m_matBasis = basis;
	}

	Mat_<float> getBasis() const
	{
		return this->m_matBasis;
	}

private:
	Mat_<float> m_matBasis;
};

#endif
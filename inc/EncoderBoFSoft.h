#include "AEncoder.h"
#include "CodebookVQ.h"
#include <opencv2/opencv.hpp>

class EncoderBoFSoft : public AEncoder
{
public:
	virtual void encode(const cv::Mat_<float>& descriptors, cv::Mat_<float>& encoded);
	virtual int codeDimension() const;

public:
	EncoderBoFSoft(const CodebookVQ& codebook, int nz);
	virtual ~EncoderBoFSoft();

private:
	CodebookVQ _m_codebook;
	int _m_nz;
	int _m_nDim;
	int _m_nCode;

	cv::KDTree* _m_pTree;
};
#include "EncoderBoFSoft.h"
#include "CodebookVQ.h"
#include "kdtree.h"
#include <stdexcept>
#include <opencv2/opencv.hpp>

using namespace cv;

EncoderBoFSoft::EncoderBoFSoft(const CodebookVQ& codebook, int nz)
{
	Mat_<float> basis = codebook.getBasis();

	int ndim = basis.cols;
	int ncode = basis.rows;

	this->_m_pTree = new KDTree(codebook.getBasis());

	this->_m_nDim = ndim;
	this->_m_nCode = ncode;
	this->_m_nz = nz;
}

EncoderBoFSoft::~EncoderBoFSoft()
{
	delete this->_m_pTree;
}

void EncoderBoFSoft::encode(const cv::Mat_<float>& descriptors, cv::Mat_<float>& encoded)
{
	int ndata = descriptors.rows;
	int ndim = descriptors.cols;

	if ( ndim != this->_m_nDim)
	{
		throw std::runtime_error("dimension not match when encode");
	}
	
	encoded.create(ndata,this->_m_nCode);
	encoded.setTo(0.0f);
	//encoded.zeros(ndata,this->_m_nCode);

#pragma omp parallel for
	for(int i=0;i<ndata;i++)
	{
		Mat index,dist;
		this->_m_pTree->findNearest(descriptors.row(i),_m_nz,INT_MAX,index,noArray(),dist);

		Scalar mean,std;
		cv::meanStdDev(dist,mean,std);
		cv::divide(std(0),dist,dist);
		
		for(int j=0;j<_m_nz;j++)
		{
			encoded(i,index.at<int>(j)) = dist.at<float>(j);
		}
	}
}

int EncoderBoFSoft::codeDimension() const
{
	return this->_m_nCode;
}
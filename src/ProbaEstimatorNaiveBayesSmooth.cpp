#include "ProbaEstimatorNaiveBayesSmooth.h"
#include <stdexcept>

using namespace cv;

void ProbaEstimatorNaiveBayesSmooth::train(const cv::Mat& data)
{
	if (data.type() != CV_8U)
	{
		throw std::runtime_error("type not support");
	}
	int ndim = data.cols;

	this->_m_matHist.create(ndim,this->_m_nHist);
	this->_m_matHist.setTo(0);

	for(int i=0;i<data.rows;i++)
	{
		for(int j=0;j<data.cols;j++)
		{
			this->_m_matHist(j,data.at<uchar>(i,j)) += 1;
		}
	}

	for(int i=0;i<_m_matHist.rows;i++)
	{
		cv::normalize(_m_matHist.row(i),_m_matHist.row(i),1.0,0.0,NORM_L1);
	}

	for(int i=0;i<_m_matHist.rows;i++)
	{
		cv::GaussianBlur(_m_matHist.row(i),_m_matHist.row(i),Size(),this->_m_sigma,_m_sigma);
	}

	for(int i=0;i<_m_matHist.rows;i++)
	{
		cv::normalize(_m_matHist.row(i),_m_matHist.row(i),1.0,0.0,NORM_L1);
	}
}

void ProbaEstimatorNaiveBayesSmooth::predict(const cv::Mat& data, cv::Mat_<float>& proba)
{
	if (data.type() != CV_8U)
	{
		throw std::runtime_error("type not support");
	}

	proba.create(data.rows,1);
	proba.setTo(0.0f);

	for(int i=0;i<data.rows;i++)
	{
		float p = 1.0;
		for ( int j=0;j<data.cols;j++)
		{
			p *= _m_matHist(j,data.at<uchar>(i,j));
		}

		proba(i) = p;
	}
}

ProbaEstimatorNaiveBayesSmooth* ProbaEstimatorNaiveBayesSmooth::clone()
{
	ProbaEstimatorNaiveBayesSmooth* p = new ProbaEstimatorNaiveBayesSmooth(this->_m_nHist,this->_m_sigma);
	p->_m_matHist = this->_m_matHist.clone();
	return p;
}
#include "ClassifierProbaEstimate.h"
#include "UtilsOpencv.hpp"

using namespace cv;

ClassifierProbaEstimate::ClassifierProbaEstimate( AProbaEstimator* p)
{
	this->_m_pBaseEstimator = p->clone();
}

ClassifierProbaEstimate::~ClassifierProbaEstimate()
{
	for(int i=0;i<this->_m_vecEstimator.size();i++)
	{
		if ( _m_vecEstimator[i])
		{
			delete _m_vecEstimator[i];
		}
	}

	if ( _m_pBaseEstimator)
	{
		delete _m_pBaseEstimator;
	}
}

void ClassifierProbaEstimate::train(const cv::Mat& data, const cv::Mat_<int>& label)
{
	UtilsOpencv::Unique(label,this->_m_matLabels);

	int nlabel = (int) _m_matLabels.total();

	this->_m_vecEstimator.resize(nlabel);
	for(int i=0;i<nlabel;i++)
	{
		_m_vecEstimator[i] = this->_m_pBaseEstimator->clone();
	}

	for(int i=0;i<nlabel;i++)
	{
		Mat_<int> index;
		UtilsOpencv::FindValue(label,_m_matLabels(i),index);
		Mat tdata;
		UtilsOpencv::SelectRows(data,index,tdata);
		_m_vecEstimator[i]->train(tdata);
	}
}

void ClassifierProbaEstimate::predictProba(const cv::Mat& data, cv::Mat_<float>& proba)
{
	Mat_<float> probas((int) this->_m_vecEstimator.size(),data.rows);

	probas.resize(this->_m_vecEstimator.size());

	for(int i=0;i<this->_m_vecEstimator.size();i++)
	{
		Mat_<float> tp;
		_m_vecEstimator[i]->predict(data,tp);
		probas.row(i) = tp.t();
	}

	Mat_<float> sumprob;
	cv::reduce(probas,sumprob,0,CV_REDUCE_SUM);

	cv::divide(probas.row(0),sumprob,proba);
	if ( this->_m_matLabels(0) == 0)
	{
		proba = 1- proba;
	}

	proba = proba.t();
}
#include "ClassifierLiblinear.h"
#include <linear.h>
#include <opencv2/opencv.hpp>
using namespace cv;

void print_null(const char *s) {}


ClassifierLiblinear::ClassifierLiblinear( int type, double c, double bias)
{
	this->_m_param.C = c;
	this->_m_param.solver_type = type;
	this->_m_param.nr_weight = 0;
	this->_m_param.weight = NULL;
	this->_m_param.weight_label = NULL;
	this->_m_param.p = 0.1;
	//this->_m_param.eps = INF;
	switch(_m_param.solver_type)
	{
		case L2R_LR: 
		case L2R_L2LOSS_SVC:
			_m_param.eps = 0.01;
			break;
		case L2R_L2LOSS_SVR:
			_m_param.eps = 0.001;
			break;
		case L2R_L2LOSS_SVC_DUAL: 
		case L2R_L1LOSS_SVC_DUAL: 
		case MCSVM_CS: 
		case L2R_LR_DUAL: 
			_m_param.eps = 0.1;
			break;
		case L1R_L2LOSS_SVC: 
		case L1R_LR:
			_m_param.eps = 0.01;
			break;
		case L2R_L1LOSS_SVR_DUAL:
		case L2R_L2LOSS_SVR_DUAL:
			_m_param.eps = 0.1;
			break;
	}
	this->_m_bias = bias;
	set_print_string_function(print_null);

	this->_m_pModel = NULL;
}

ClassifierLiblinear::~ClassifierLiblinear()
{
	if ( this->_m_pModel)
	{
		free_and_destroy_model(&_m_pModel);
	}
}

void ClassifierLiblinear::train(const cv::Mat& data, const cv::Mat_<int>& label) 
{
	struct problem* prob;
	if ( data.type() == CV_32F)
	{
		prob = this->_readProblem<float>(data,label);
	}
	else if (data.type() == CV_64F)
	{
		prob = this->_readProblem<double>(data,label);
	}
	if (_m_pModel)
	{
		free_and_destroy_model(&_m_pModel);
	}

	this->_m_pModel = ::train(prob,&this->_m_param);

	this->_releaseProblem(prob);
}

void ClassifierLiblinear::predict(const cv::Mat& data, cv::Mat_<int>& label)
{
	if (data.type()==CV_32F)
	{
		this->_predict<float>(data,label);
	}
	else if ( data.type()==CV_64F)
	{
		this->_predict<double>(data,label);
	}
}

void ClassifierLiblinear::predictProba(const cv::Mat& data, cv::Mat_<float>& proba)
{
	if (data.type()==CV_32F)
	{
		this->_predictProba<float>(data,proba);
	}
	else if ( data.type()==CV_64F)
	{
		this->_predictProba<double>(data,proba);
	}
}


void ClassifierLiblinear::save(const std::string& filename)
{
	save_model(filename.c_str(),this->_m_pModel);
}

void ClassifierLiblinear::load(const std::string& filename)
{
	if ( _m_pModel)
	{
		free_and_destroy_model(&_m_pModel);
	}

	this->_m_pModel = load_model(filename.c_str());
	this->_m_bias = _m_pModel->bias;
}
void ClassifierLiblinear::_releaseProblem(struct problem* p)
{
	delete [] p->x[0];
	delete [] p->x;
	delete [] p->y;

	delete p;

}

void ClassifierLiblinear::getLabels(cv::Mat_<int>& ulabel)
{
	ulabel.create(this->_m_pModel->nr_class,1);
	for(int i=0;i<_m_pModel->nr_class;i++)
	{
		ulabel(i) = _m_pModel->label[i];
	}
}
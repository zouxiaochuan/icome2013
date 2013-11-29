#include "AClassifier.h"
#include <linear.h>
#include <opencv2/opencv.hpp>

class ClassifierLiblinear : public AClassifier
{
public:
	virtual void train(const cv::Mat& data, const cv::Mat_<int>& label) ;
	virtual void predict(const cv::Mat& data, cv::Mat_<int>& label) ;
	virtual void predictProba(const cv::Mat& data, cv::Mat_<float>& proba);
	virtual void getLabels(cv::Mat_<int>& ulabel);
	virtual void save(const std::string& filename);
	virtual void load(const std::string& filename);

public:
	ClassifierLiblinear() : _m_pModel(NULL)
	{};
	ClassifierLiblinear( int type, double c, double bias);
	~ClassifierLiblinear();

private:
	template<class T>
	problem* _readProblem(const cv::Mat& data, const cv::Mat_<int>& label);
	void _releaseProblem(struct problem* p);

	template<class T>
	void _predict(const cv::Mat& data, cv::Mat_<int>& label);

	template<class T>
	void _predictProba(const cv::Mat& data, cv::Mat_<float>& proba);

private:
	parameter _m_param;
	double _m_bias;
	struct model* _m_pModel;
};

template<class T>
problem* ClassifierLiblinear::_readProblem(const cv::Mat& data, const cv::Mat_<int>& label)
{

	int nnz = cv::countNonZero(data);

	struct problem* prob = new struct problem();
	prob->bias = this->_m_bias;

	prob->l = data.rows;
	prob->n = data.cols+1;
	prob->x = new struct feature_node*[prob->l];
	prob->y = new double[prob->l];
	prob->x[0] = new struct feature_node[nnz + data.rows*2];

	struct feature_node* xspace = prob->x[0];
	int k=0;
	for(int i=0;i<prob->l;i++)
	{
		prob->x[i] = &xspace[k];
		for( int j=0;j<data.cols;j++)
		{
			T val = data.at<T>(i,j);
			if ( val != 0)
			{
				xspace[k].index = j+1;
				xspace[k].value = val;
				k++;
			}
		}
		xspace[k].index = prob->n;
		xspace[k].value = this->_m_bias;
		k++;
		xspace[k++].index = -1;

		prob->y[i] = label(i);
	}

	return prob;
}

template<class T>
void ClassifierLiblinear::_predict(const cv::Mat& data, cv::Mat_<int>& label)
{
	label.create(data.rows,1);

#pragma omp parallel for
	for(int i=0;i<data.rows;i++)
	{
		cv::Mat r = data.row(i);
		struct feature_node* pfea = new feature_node[countNonZero(r)+2];

		int k=0;
		for(int j=0;j<r.cols;j++)
		{
			T val = r.at<T>(j);
			if ( val != 0)
			{
				pfea[k].index = j+1;
				pfea[k].value = val;
				k++;
			}
		}

		pfea[k].index = r.cols+1;
		pfea[k].value = this->_m_bias;
		k++;
		pfea[k].index = -1;

		double l = ::predict(this->_m_pModel,pfea);
		delete [] pfea;
		label(i) = (int)l;
	}
}

template<class T>
void ClassifierLiblinear::_predictProba(const cv::Mat& data, cv::Mat_<float>& proba)
{
	proba.create(data.rows,1);

#pragma omp parallel for
	for(int i=0;i<data.rows;i++)
	{
		cv::Mat r = data.row(i);
		struct feature_node* pfea = new feature_node[countNonZero(r)+2];

		int k=0;
		for(int j=0;j<r.cols;j++)
		{
			T val = r.at<T>(j);
			if ( val != 0)
			{
				pfea[k].index = j+1;
				pfea[k].value = val;
				k++;
			}
		}

		pfea[k].index = r.cols+1;
		pfea[k].value = this->_m_bias;
		k++;
		pfea[k].index = -1;

		double val;
		::predict_values(this->_m_pModel,pfea,&val);
		delete [] pfea;
		proba(i) = (float) (1.0/(1+exp(-val)));
	}

	if ( this->_m_pModel->label[0] == 0)
	{
		proba = 1- proba;
	}
}
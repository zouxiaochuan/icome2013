#ifndef _PROBAESTIMATORGMMOPENCV_H_
#define _PROBAESTIMATORGMMOPENCV_H_

#include "AProbaEstimator.h"

class GMMOpencv
{
public:
    static const int componentsCount = 10;

    GMMOpencv( cv::Mat& _model );
    double operator()( const cv::Vec3d color ) const;
    double operator()( int ci, const cv::Vec3d color ) const;
    int whichComponent( const cv::Vec3d color ) const;

    void initLearning();
    void addSample( int ci, const cv::Vec3d color );
    void endLearning();

private:
    void calcInverseCovAndDeterm( int ci );
    cv::Mat model;
    double* coefs;
    double* mean;
    double* cov;

    double inverseCovs[componentsCount][3][3];
    double covDeterms[componentsCount];

    double sums[componentsCount][3];
    double prods[componentsCount][3][3];
    int sampleCounts[componentsCount];
    int totalSampleCount;
};

class ProbaEstimatorGMMOpencv : public AProbaEstimator
{
public:
	virtual void train(const cv::Mat& data);
	virtual void predict(const cv::Mat& data, cv::Mat_<float>& proba);

	virtual ProbaEstimatorGMMOpencv* clone();

public:
	ProbaEstimatorGMMOpencv();
	~ProbaEstimatorGMMOpencv();

private:
	GMMOpencv* _m_pGMM;

};

#endif
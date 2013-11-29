#include "UtilsValidate.h"
#include "UtilsOpencv.hpp"
#include "AClassifier.h"

#include <opencv2/opencv.hpp>

using namespace cv;

void UtilsValidate::SplitRandomBalance(const cv::Mat_<int>& label, int ntest, int nrun, cv::Mat_<uchar>& splits)
{
	splits.create(nrun,(int)label.total());
	splits.setTo(1);

	Mat_<int> ulabel;
	UtilsOpencv::Unique(label,ulabel);

	for(int irun=0;irun<nrun;irun++)
	{
		for(int i=0;i<ulabel.total();i++)
		{
			int l = ulabel(i);
			Mat ls = (label == l);
			Mat_<int> lidx;
			UtilsOpencv::Bool2Index(ls,lidx);
			if (lidx.total() < ntest)
			{
				throw runtime_error("not enough test instance");
			}
			Mat_<int> perm;
			UtilsOpencv::RandPerm((int)lidx.total(),perm);
		
			for(int j=0;j<ntest;j++)
			{
				splits(irun, lidx(perm(j))) = 0;
			}
		}
	}
}

void UtilsValidate::ValidateRandomBalance(const cv::Mat_<int>& label, const cv::Mat& data, AClassifier* pClassifier, Mat_<double>& result,
		int ntest, int nrun)
{
	result.create(nrun,1);
	result.setTo(-1.0);

	Mat_<uchar> splits;
	UtilsValidate::SplitRandomBalance(label,ntest,nrun,splits);

	for(int i=0;i<nrun;i++)
	{
		result(i) = GetLoss(label,data,pClassifier,splits.row(i));
	}
}

double UtilsValidate::GetLoss(const cv::Mat_<int>& label, const cv::Mat& data, AClassifier* pClassifier, const Mat_<uchar>& split)
{

	Mat trainData,testData;
	Mat_<int> trainIdx,testIdx;
	Mat_<int> trainLabel,testLabel;
	UtilsOpencv::Bool2Index(split,trainIdx);
	UtilsOpencv::Bool2Index(1-split,testIdx);
	UtilsOpencv::SelectRows(data,trainIdx,trainData);
	UtilsOpencv::SelectRows(data,testIdx,testData);
	UtilsOpencv::SelectRows(label,trainIdx,trainLabel);
	UtilsOpencv::SelectRows(label,testIdx,testLabel);

	pClassifier->train(trainData,trainLabel);
	Mat_<int> preLabel;
	pClassifier->predict(testData,preLabel);

	double nwrong = 0;
	for(int i=0;i<(int)testLabel.total();i++)
	{
		if ( testLabel(i) != preLabel(i))
		{
			nwrong++;
		}
	}

	return nwrong / testLabel.total();
}
#include "ClassifierLiblinear.h"
#include "UtilsOpencv.hpp"
#include "UtilsValidate.h"

int main(int argn, char* argc[])
{
	Mat_<float> data;
	Mat_<int> label;
	UtilsOpencv::ReadSvm(argc[1],data,label);

	ClassifierLiblinear c(2,0.1,1);

	Mat_<double> r;
	UtilsValidate::ValidateRandomBalance(label,data,&c,r,25,10);

	cout << "mean loss:" << cv::mean(r)(0) << endl;

	string bak = argc[1];
	bak.append(".bak");
	UtilsOpencv::WriteSvm(bak,data,label);

}
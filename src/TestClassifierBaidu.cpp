#include "GlobalConfig.h"
#include "ClassifierLiblinear.h"
#include "UtilsValidate.h"
#include "UtilsOpencv.hpp"
#include <stdlib.h>
#include <math.h>
#include <iostream>

using namespace std;

int main(int argn, char* argc[])
{
	GlobalConfig cfg;
	cfg.read(argc[1]);

	string featureFilename = cfg.getFeaturePath();

	cout << "begin load file" << endl;

	Mat_<float> data;
	Mat_<int> label;

	UtilsOpencv::ReadSvm(featureFilename,data,label);

	cout << "load data end" << endl;
	for(int i=-6;i<=6;i++)
	{
		double c = pow(10,(double)i);
		ClassifierLiblinear classifier(2,c,1.0);
		Mat_<double> result;
		UtilsValidate::ValidateRandomBalance(label,data,&classifier,result,1000,10);

		cout << "c=" << c << ", loss=" << mean(result)(0) << endl;
		cout << "result=" << result << endl;
	}
}
#include "GlobalConfig.h"
#include "UtilsOpencv.hpp"
#include "AClassifier.h"

int main(int argn, char* argc[])
{
	GlobalConfig cfg;
	cfg.read(argc[1]);

	string filenameFeature = cfg.getFeaturePath();
	Mat_<float> feature;
	Mat_<int> label;
	UtilsOpencv::ReadSvm(filenameFeature,feature,label);

	AClassifier* pClassifier = cfg.getClassifier();
	pClassifier->train(feature,label);
	pClassifier->save(cfg.getClassifierModelPath());
}
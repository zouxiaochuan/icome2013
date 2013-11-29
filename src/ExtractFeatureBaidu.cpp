#include "FeatureExtractorSPM.h"
#include "GlobalConfig.h"
#include "FeatureExtractorSPM.h"
#include "ImageDatasetBaidu.h"
#include "UtilsOpencv.hpp"

int main(int argn, char* argc[])
{
	GlobalConfig cfg;
	cfg.read(argc[1]);

	FeatureExtractorSPM* pFeatureExtractor = dynamic_cast<FeatureExtractorSPM*>(cfg.getFeatureExtractor());

	Mat_<float> feature;
	pFeatureExtractor->extractFromEncodedPath(cfg.getEncodedPath(),cfg.getDataset(),1024,feature);

	ImageDatasetBaidu* pds = dynamic_cast<ImageDatasetBaidu*>(cfg.getDataset());

	string savepath = cfg.getFeaturePath();

	//FileStorage fs(savepath,FileStorage::WRITE);
	//fs << "feature" << feature;
	//fs << "label" << pds->getLabels();
	//fs.release();

	Mat_<int> label = pds->getLabels();
	UtilsOpencv::WriteSvm(savepath,feature,label);
}
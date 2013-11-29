#include "GlobalConfig.h"
#include "DetectorHumanBaidu.h"
#include "ImageDatasetBaidu.h"

int main(int argn, char* argc[])
{
	GlobalConfig cfg;
	cfg.read(argc[1]);

	DetectorHumanBaidu* pDetector = dynamic_cast<DetectorHumanBaidu* >(cfg.getDetector());
	ImageDatasetBaidu* pDataset = dynamic_cast<ImageDatasetBaidu*>(cfg.getDataset());

	pDetector->learnFromDataset(pDataset);
}
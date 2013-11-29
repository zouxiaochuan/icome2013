#include "CodebookBuilderKmeansVl.h"
#include "ImageDatasetBaidu.h"
#include "CodebookVQ.h"
#include "GlobalConfig.h"

int main(int argn, char* argc[])
{
	GlobalConfig cfg;
	cfg.read(argc[1]);

	string dataRoot = cfg.getRoot().get_child("BaiduRoot").get_value<string>();

	ImageDatasetBaidu dataset(dataRoot);
	
	ALocalExtractor* pExtractor = cfg.getLocalExtractor();
	ACodebookBuilder* pCbBuilder = cfg.getCodebookBuilder();
	ACodebook* pCodebook = cfg.getCodebook();

	pCbBuilder->buildFromDataset(*pCodebook,dataset,*pExtractor,cfg.getCodebookMaxTrainingNum());
	pCodebook->save(cfg.getCodebookPath());
}
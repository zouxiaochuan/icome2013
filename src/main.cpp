#include "UtilsBoost.hpp"
#include "UtilsOpencv.hpp"
#include "FeatureExtractorSPMEx.h"

#include <opencv2/opencv.hpp>
#include <string>

using namespace std;
using namespace boost::filesystem;
using namespace cv;

int main(int argn, char* argv[])
{
	boost::filesystem::path pathImageRoot("");
	boost::filesystem::path pathSaveRoot("");
	boost::filesystem::path pathCb("");

	AFeatureExtractor* p = new FeatureExtractorSPMEx(pathCb.string());

	vector<path> files;
	UtilsBoost::DirRecursive(pathImageRoot,files);

#pragma omp parallel for
	for(int i=0;i<files.size();i++)
	{
		Mat img = imread(files[i].string());
		Mat_<float> feature;
		p->extractFromMat(img,feature);

		path pathFolder = pathSaveRoot;
		pathFolder /= files[i].parent_path().leaf();
		if (!boost::filesystem::exists(pathFolder))
		{
			boost::filesystem::create_directory(pathFolder);
		}
		path pathFile = pathFolder;
		pathFile /= files[i].leaf();
		pathFile = pathFile.replace_extension(".txt");

		UtilsOpencv::WriteTxt(pathFile.string(),feature);
	}

	delete p;
}
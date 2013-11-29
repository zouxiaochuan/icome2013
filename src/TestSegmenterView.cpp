
#include "GlobalConfig.h"
#include "UtilsBoost.hpp"

int main(int argn, char* argv[])
{
	GlobalConfig cfg;
	cfg.read(argv[1]);

	boost::filesystem::path pathImg(argv[2]);
	boost::filesystem::path pathSave(argv[3]);

	vector<boost::filesystem::path> files;
	UtilsBoost::DirRecursive(pathImg,files);

	ASegmenter* pSegmenter = cfg.getSegmenter();

	for(int i=0;i<files.size();i++)
	{
		Mat img = imread(files[i].string());
		boost::filesystem::path filenameSave = pathSave;
		filenameSave /= files[i].leaf();
		
		Mat_<uchar> mask;
		pSegmenter->segment(img,mask);
		Mat imgo;
		cv::resize(img,imgo,Size(mask.cols,mask.rows));
		Mat imgoo;
		imgo.copyTo(imgoo,mask);
		imwrite(filenameSave.string(),imgoo);
	}
}

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

	ADetector* pDetector = cfg.getDetector();

	for(int i=0;i<files.size();i++)
	{
		Mat img = imread(files[i].string());
		boost::filesystem::path filenameSave = pathSave;
		filenameSave /= files[i].leaf();
		
		Mat_<uchar> mask;
		vector<Rect> boxes;
		Mat_<float> conf;
		pDetector->detectMultiScale(img,boxes,conf,1.1,500,50);
		for(int j=0;j<1/*boxes.size()*/;j++)
		{
			cv::rectangle(img,boxes[j],Scalar(0,255,0),5);
		}
		imwrite(filenameSave.string(),img);
	}
}
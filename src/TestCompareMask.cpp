#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>
#include "UtilsBoost.hpp"
#include "UtilsSegmentation.h"
#include <iostream>

using namespace cv;
using namespace std;

int main(int argn, char* argv[])
{
	boost::filesystem::path pathGt(argv[1]);
	boost::filesystem::path pathPr(argv[2]);

	vector<boost::filesystem::path> files;
	UtilsBoost::DirRecursive(pathGt,files);

	float sumr = 0;
	int cnt = 0;

#pragma omp parallel for
	for(int i=0;i<files.size();i++)
	{
		boost::filesystem::path filenamePr = pathPr;
		filenamePr /= files[i].leaf();

		Mat imgGt,imgPr;
		imgGt = imread(files[i].string());
		imgPr = imread(filenamePr.string());

		if ( imgGt.empty() || imgPr.empty())
		{
			continue;
		}

		cv::cvtColor(imgGt,imgGt,CV_BGR2GRAY);
		cv::cvtColor(imgPr,imgPr,CV_BGR2GRAY);

		cv::resize(imgPr,imgPr,Size(imgGt.cols,imgGt.rows));
		cv::threshold(imgPr,imgPr,100,255,THRESH_BINARY);
		cv::threshold(imgGt,imgGt,100,255,THRESH_BINARY);

		imgPr = 255 - imgPr;
		imgGt = 255 - imgGt;
		float r = UtilsSegmentation::CompareTwoMask(imgPr,imgGt);

		int cpr = imgPr.channels();
		int cgt = imgGt.channels();

		if ( r==1.0)
		{
			int xx = 0;
			xx = 1;
		}

#pragma omp critical
		{
			cout << r << endl;
			sumr += r;
			cnt++;
		}
	}

	cout << sumr/cnt << endl;
}
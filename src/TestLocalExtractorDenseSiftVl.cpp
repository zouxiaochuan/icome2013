#include "LocalExtractorDenseSiftVl.h"
#include "UtilsOpencv.hpp"

#include "opencv2/opencv.hpp"
#include <string>
#include <iostream>
#include "boost/filesystem.hpp"

using namespace std;
using namespace cv;
using namespace boost::filesystem;

void test_extractFromMat()
{
	path pathImg("../../data/test.jpg");
	path pathDesc("../../data/test_descs.txt");
	path pathLoc("../../data/test_locs.txt");
	//path pathImg("D:\\work\\icome_challenge\\2013\\code\\data\\test.jpg");
	//path pathDesc("D:\\work\\icome_challenge\\2013\\code\\data\\test_descs.txt");
	//path pathLoc("D:\\work\\icome_challenge\\2013\\code\\data\\test_locs.txt");

	Mat img = imread(pathImg.string());

	Mat_<float> descsGt,locsGt;
	UtilsOpencv::ReadTxt(pathDesc.string(),descsGt);
	UtilsOpencv::ReadTxt(pathLoc.string(),locsGt);
	
	LocalExtractorDenseSiftVl extractor(4,4);
	vector<Mat> rgbs;
	cv::split(img,rgbs);

	Mat_<float> descs,locs;
	extractor.extractFromMat(rgbs[2],descs,locs);

	//check size
	if (locs.size != locsGt.size)
	{
		cout << "FAIL: location size not match" << endl;
		return;
	}
	if ( descs.size != descsGt.size)
	{
		cout << "FAIL: descriptor size not match" << endl;
	}

	//check locations
	for(int i=0;i<locs.rows;i++)
	{
		for(int j=0;j<locs.cols;j++)
		{
			if ( locs(i,j) != locsGt(i,j))
			{
				cout << "FAILE : location not match" << endl;
				return;
			}
		}
	}

	//check descriptor;
	double thresh = 0.1;
	double maxdist = 0;
	descs = 512 * descs;
	descs.setTo(255.0f,descs>255.0f);
	for( int i=0;i<descs.rows;i++)
	{
		Mat diff = descs.row(i) - descsGt.row(i);
		double dist = norm(diff,NORM_L2);

		if (dist > thresh)
		{
			cout << "FAIL: descriptor not match, distance: " << dist << endl;
			return ;
		}

		maxdist = max(maxdist,dist);
	}

	cout << "minimal distance of descriptors: " << maxdist << endl;;
}

int main(int argn, char* argc[])
{

	test_extractFromMat();
	return 0;
}
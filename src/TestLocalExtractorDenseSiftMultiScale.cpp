#include "LocalExtractorDenseSiftMultiScale.h"
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
	//path pathImg("D:\\work\\icome_challenge\\2013\\code\\data\\test.jpg");
	//path pathDesc("D:\\work\\icome_challenge\\2013\\code\\data\\test_descs.txt");
	//path pathLoc("D:\\work\\icome_challenge\\2013\\code\\data\\test_locs.txt");
	path pathTxt("../../data/test_multiscale.txt");

	Mat img = imread(pathImg.string());

	vector<int> sizes;
	sizes.push_back(4);
	sizes.push_back(8);
	sizes.push_back(6);

	LocalExtractorDenseSiftMultiScale extractor("vlfeat",4,sizes);
	vector<Mat> rgbs;
	cv::split(img,rgbs);

	Mat_<float> descs,locs;
	extractor.extractFromMat(rgbs[2],descs,locs);

	cout << "number of descriptors: " << descs.rows << endl;
	cout << "number of location: " << locs.rows << endl;

	UtilsOpencv::WriteTxt(pathTxt.string(),descs);
}

int main(int argn, char* argc[])
{

	test_extractFromMat();
	return 0;
}
#include "EncoderBoFSoft.h"
#include "CodebookVQ.h"
#include <string>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "LocalExtractorDenseSiftMultiScale.h"
#include "UtilsOpencv.hpp"

using namespace cv;
using namespace std;

int main(int argn, char* argc[])
{
	string pathCodebook = argc[1];
	string pathTxt = argc[2];
	string pathImage = argc[3];

	CodebookVQ cb;
	cb.load(pathCodebook);
	EncoderBoFSoft encoder(cb,5);
	Mat img = imread(pathImage);
	
	vector<int> sizes;
	sizes.push_back(4);
	sizes.push_back(8);
	LocalExtractorDenseSiftMultiScale extractor("vlfeat",4,sizes);
	Mat_<float> d,l;
	extractor.extractFromMat(img,d,l);
	cout << "code dimension: " << encoder.codeDimension() << endl;
	Mat_<float> encoded;
	encoder.encode(d,encoded);
	UtilsOpencv::WriteTxt(pathTxt,encoded);
}
#ifndef _ALOCALEXTRACTOR_H_
#define _ALOCALEXTRACTOR_H_

#include "opencv2/opencv.hpp"

using namespace cv;

class ALocalExtractor
{
public:
	virtual void extractFromMat( const Mat& img, Mat_<float>& descriptors, Mat_<float>& location) = 0;
};


#endif
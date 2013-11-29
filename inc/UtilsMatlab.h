#ifndef _UTILSMATLAB_H_
#define _UTILSMATLAB_H_

#include "opencv2/opencv.hpp"

using namespace cv;

class UtilsMatlab
{
public:
	static void Opencv2Matlab(const Mat& in, Mat& out);
};

#endif
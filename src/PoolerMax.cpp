#include "PoolerMax.h"
#include <opencv2/opencv.hpp>

using namespace cv;

void PoolerMax::pool(const Mat& code, Mat& pooled)
{
	pooled.create(1,code.cols,code.type());

	cv::reduce(code,pooled,0,CV_REDUCE_MAX);
}
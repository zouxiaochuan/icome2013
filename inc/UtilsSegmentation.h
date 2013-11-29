#ifndef _UTILSSEGMENTATION_H_
#define _UTILSSEGMENTATION_H_

#include <opencv2/opencv.hpp>
#include "superpixel.h"

class UtilsSegmentation
{
public:
	static void MaxFlowSuperpixel( std::vector<SuperpixelStatistic>& spstat, const cv::Mat_<float>& fgdEnergy,
		const cv::Mat_<float>& bgdEnergy, float gamma, cv::Mat_<int>& label);

	static float CompareTwoMask(cv::Mat mask1, cv::Mat mask2);
};


#endif
#ifndef _UTILSSUPERPIXEL_H_
#define _UTILSSUPERPIXEL_H_

#include <opencv2/opencv.hpp>
#include "superpixel.h"

class UtilsSuperpixel
{
public:
	static int CountLabel(const cv::Mat_<int>& segmentation);
	static void Stat(const cv::Mat_<int>& segmentation, const cv::Mat_<float>& data, 
		const std::vector<SuperpixelStatistic>& stats, cv::Mat_<float>& mdata);
	static int CountEdge(const std::vector<SuperpixelStatistic>& stats);
};

#endif
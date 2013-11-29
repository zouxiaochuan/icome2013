#ifndef _AIMAGEDATASETFORDETECTION_H_
#define _AIMAGEDATASETFORDETECTION_H_

#include "AImageDataset.h"

#include <opencv2/opencv.hpp>

class AImageDatasetForDetection : virtual public AImageDataset
{
public:
	virtual void getBoundingBox(int idx, std::vector<cv::Rect>& boxes) = 0;
};

#endif
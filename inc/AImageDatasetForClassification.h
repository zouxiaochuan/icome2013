#ifndef _AIMAGEDATASETFORCLASSIFICATION_H_
#define _AIMAGEDATASETFORCLASSIFICATION_H_

#include "AImageDataset.h"
#include <opencv2/opencv.hpp>

class AImageDatasetForClassification : virtual public AImageDataset
{
public:
	virtual cv::Mat_<int> getLabels() = 0;
};

#endif
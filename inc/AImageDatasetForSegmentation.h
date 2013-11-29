#ifndef _AIMAGEDATASETFORSEGMENTATION_H_
#define _AIMAGEDATASETFORSEGMENTATION_H_

#include "AImageDataset.h"

class AImageDatasetForSegmentation : virtual public AImageDataset
{
public:
	virtual cv::Mat getMask(int idx) = 0;
};

#endif